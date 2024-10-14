import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from PIL import Image

from typing import Optional, Tuple

from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from .owlv2_weights import load_owlv2_weights, LayerWeights, TextTransformerWeights, VisionTransformerWeights, OWLv2Weights
from .owlv2_config import OWLV2_B16, EncoderParams, ModelParams

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)

class SquarePad:
	def __call__(self, image):
		h, w = image.shape[-2:]
		max_wh = np.max([w, h])
		hp = int(max_wh - w)
		vp = int(max_wh - h)
		padding = (0, 0, hp, vp)
		return TF.pad(image, padding, 0.5, 'constant')
  

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, attn_mask = None):
    bsz, _, _ = x.shape
    xq = F.linear(x, layer_weights.wq, bias=layer_weights.wq_b).view(bsz, -1 ,model_params.n_heads, model_params.head_dim).transpose(1,2).contiguous()
    xk = F.linear(x, layer_weights.wk, bias=layer_weights.wk_b).view(bsz, -1 ,model_params.n_heads, model_params.head_dim).transpose(1,2).contiguous()
    xv = F.linear(x, layer_weights.wv, bias=layer_weights.wv_b).view(bsz, -1 ,model_params.n_heads, model_params.head_dim).transpose(1,2).contiguous()
    output = torch.nn.functional.scaled_dot_product_attention(
        xq,
        xk,
        xv,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=model_params.head_dim**-0.5
    )

    output = output.transpose(1,2)
    output = output.reshape(bsz, -1, model_params.dim)
    output = F.linear(output, layer_weights.wo, layer_weights.wo_b)
    return output

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
    x = F.linear(x, layer_weights.w1, bias=layer_weights.w1_b)
    return F.linear(F.sigmoid(x * 1.702) * x, layer_weights.w2, bias=layer_weights.w2_b)

import torch

def pad_sequences(sequences: list[list[int]]) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padded_seq = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences)

def create_mask(sequences: list[list[int]]) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        mask[i, len(seq):] = True
    return mask

def build_attn_mask(batch_size: int, seqlen: int, start_pos: int) -> torch.Tensor:
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), DEFAULT_MASK_VALUE)
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32)
        mask = mask.unsqueeze(0).expand(batch_size, 1, seqlen, seqlen)
    else:
        mask = torch.zeros((batch_size, 1, seqlen, seqlen))
    
    return mask

def combine_masks(padding_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    padding_mask_4d = padding_mask.unsqueeze(1).unsqueeze(2)
    combined_mask = attn_mask.clone()
    combined_mask.masked_fill_(padding_mask_4d, DEFAULT_MASK_VALUE)
    return combined_mask


def encode_text(tokens, weights: TextTransformerWeights, model_params: ModelParams, attn_mask):
    seq_length = tokens.shape[-1] if tokens is not None else tokens.shape[-2]
    h = weights.tok_embeddings[tokens]
    pos_ids = torch.arange(16).expand((1, -1))
    pos_embs = weights.pos_embeddings[pos_ids[:,:seq_length]]
    h = h + pos_embs


    for i in range(model_params.text_encoder.n_layers):
        residual = h
        h = F.layer_norm(h, (model_params.text_encoder.dim,), weight=weights.layer_weights[i].attention_norm, bias=weights.layer_weights[i].attention_norm_b)
        h_attn  = attention(h, weights.layer_weights[i], model_params.text_encoder, attn_mask=attn_mask)
        h = residual + h_attn
        residual = h
        h = F.layer_norm(h, (model_params.text_encoder.dim,), weight=weights.layer_weights[i].ffn_norm, bias=weights.layer_weights[i].ffn_norm_b)
        h = residual + feed_forward(h, weights.layer_weights[i])
    
    norm_x = F.layer_norm(h, (model_params.text_encoder.dim,), weight=weights.post_norm, bias=weights.post_norm_b)
    pooled_output = norm_x[torch.arange(norm_x.shape[0], device=norm_x.device),tokens.to(torch.int).argmax(dim=-1).to(norm_x.device),    ]
    return pooled_output, h

def encode_vision(pixel_data: torch.Tensor, weights: VisionTransformerWeights, model_params: ModelParams):
    batch_size = pixel_data.shape[0]
    patch_embeds = F.conv2d(pixel_data, weights.patch_embeddings, stride=model_params.patch_size)
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
    class_embeds = weights.cls_embedding.expand(batch_size, 1, -1)
    
    h = torch.cat([class_embeds, patch_embeds], dim=1)
    h = h + weights.pos_embeddings[torch.arange(model_params.num_vision_pos).expand((1, -1))]
    h = F.layer_norm(h, (model_params.vision_encoder.dim,), weight=weights.pre_norm, bias=weights.pre_norm_b)
    for i in range(model_params.vision_encoder.n_layers):
        residual = h
        h = F.layer_norm(h, (model_params.vision_encoder.dim,), weight=weights.layer_weights[i].attention_norm, bias=weights.layer_weights[i].attention_norm_b)
        h_attn  = attention(h, weights.layer_weights[i], model_params.vision_encoder, attn_mask=None)
        h = residual + h_attn
        residual = h
        h = F.layer_norm(h, (model_params.vision_encoder.dim,), weight=weights.layer_weights[i].ffn_norm, bias=weights.layer_weights[i].ffn_norm_b)
        h = residual + feed_forward(h, weights.layer_weights[i])
    pooled_output = h[:,0,:]
    norm_x = F.layer_norm(pooled_output, (model_params.vision_encoder.dim,), weight=weights.post_norm, bias=weights.post_norm_b)

    return norm_x, h
    
if __name__ == '__main__':
    from safetensors import safe_open
    state_dict = {}
    with safe_open("model.safetensors", framework="pt") as f:
        for k in f.keys():
            tens = f.get_tensor(k)
            k = k.replace("owlv2.","").replace(".embeddings","")
            k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
            state_dict[k] = tens
    weights = load_owlv2_weights(state_dict)

    seq = pad_sequences([[1,2,3,4],[1,2],[1,2,3]])
    pad_mask = create_mask([[1,2,3,4],[1,2],[1,2,3]])
    causal_mask = build_attn_mask(3, 4, 0)
    attn_mask = combine_masks(pad_mask, causal_mask)

    inp = torch.rand(1,3601,768) 
    attention(inp, weights.vision_weights.layer_weights[0], OWLV2_B16.vision_encoder)
    inp = torch.rand(1,10,512) 
    attention(inp, weights.text_weights.layer_weights[0], OWLV2_B16.text_encoder)

    embs = encode_text(seq, weights=weights.text_weights, model_params=OWLV2_B16.text_encoder, attn_mask=attn_mask)
    embs = F.linear(embs, weights.text_proj)
    print(embs.shape)
    
    embs = encode_vision(torch.rand(1,3,960,960), weights=weights.vision_weights, model_params=OWLV2_B16)
    print(embs.shape)
    embs = F.linear(embs, weights.vision_proj)
    print(embs.shape)

    """ with torch.no_grad():
        x = torch.ones(1, 12, dtype=torch.int64)
        y = model.text_model.forward(x)
        #y = model.text_model.encoder.layers[0].forward(x)
        print("y",y.shape)
        #print(x-y) """