import torch
import torch.nn.functional as F

from typing import Optional, Tuple

from OWLv2torch.torch_func.owlv2_weights import load_owlv2_weights, LayerWeights, TextTransformerWeights, VisionTransformerWeights, OWLv2Weights, HeadLayerWeights
from OWLv2torch.torch_func.owlv2_config import OWLV2_B16, EncoderParams, ModelParams

from EzLogger import Timer
timer = Timer()

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

def process_sequences(sequences: list[list[int]], start_pos: int = 0, default_mask_value: float = float('-inf')) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)
    
    # Pad sequences and create padding mask
    padded_seqs = torch.zeros((batch_size, max_len), dtype=torch.long)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded_seqs[i, :seq_len] = torch.tensor(seq)
        padding_mask[i, :seq_len] = False
    
    # Build attention mask
    if max_len > 1:
        attn_mask = torch.triu(torch.full((max_len, max_len), default_mask_value), diagonal=1)
        attn_mask = torch.hstack([torch.zeros((max_len, start_pos)), attn_mask]).float()
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, 1, max_len, max_len)
    else:
        attn_mask = torch.zeros((batch_size, 1, max_len, max_len))
    
    # Combine masks
    combined_mask = attn_mask.clone()
    combined_mask.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), default_mask_value)
    
    return padded_seqs, combined_mask

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

def xfrm(h, weights, model_params, attn_mask=None):
    for i in range(model_params.n_layers):
        residual = h
        h = F.layer_norm(h, (model_params.dim,), weight=weights.layer_weights[i].attention_norm, bias=weights.layer_weights[i].attention_norm_b)
        h_attn  = attention(h, weights.layer_weights[i], model_params, attn_mask=attn_mask)
        h = residual + h_attn
        residual = h
        h = F.layer_norm(h, (model_params.dim,), weight=weights.layer_weights[i].ffn_norm, bias=weights.layer_weights[i].ffn_norm_b)
        h = residual + feed_forward(h, weights.layer_weights[i])
    return h 

@timer("encode_text", gpu=True)
def encode_text(tokens, weights: TextTransformerWeights, model_params: ModelParams, attn_mask):
    h = weights.tok_embeddings[tokens]
    pos_ids = torch.arange(16, device=h.device).expand((1, -1))
    pos_embs = weights.pos_embeddings[pos_ids[:,:tokens.shape[-1]]]
    h = h + pos_embs

    h = xfrm(h, weights=weights, model_params=model_params.text_encoder, attn_mask=attn_mask)

    norm_x = F.layer_norm(h, (model_params.text_encoder.dim,), weight=weights.post_norm, bias=weights.post_norm_b)
    pooled_output = norm_x[torch.arange(norm_x.shape[0], device=norm_x.device),tokens.to(torch.int).argmax(dim=-1).to(norm_x.device),    ]
    return pooled_output, h

@timer("encode_vision", gpu=True)
def encode_vision(pixel_data: torch.Tensor, weights: VisionTransformerWeights, model_params: ModelParams):
    batch_size = pixel_data.shape[0]
    patch_embeds = F.conv2d(pixel_data, weights.patch_embeddings, stride=model_params.patch_size)
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
    class_embeds = weights.cls_embedding.expand(batch_size, 1, -1)
    
    h = torch.cat([class_embeds, patch_embeds], dim=1)
    h = h + weights.pos_embeddings[torch.arange(model_params.num_vision_pos).expand((1, -1))]
    h = F.layer_norm(h, (model_params.vision_encoder.dim,), weight=weights.pre_norm, bias=weights.pre_norm_b)

    h = xfrm(h, weights=weights, model_params=model_params.vision_encoder)

    pooled_output = h[:,0,:]
    norm_x = F.layer_norm(pooled_output, (model_params.vision_encoder.dim,), weight=weights.post_norm, bias=weights.post_norm_b)

    return norm_x, h

def class_head(image_embs, query_embs, query_mask, head_weights: HeadLayerWeights, model_params: ModelParams):
    image_class_embeds = F.linear(image_embs, head_weights.w1, bias=head_weights.w1_b)
    if query_embs is None:
        return (torch.zeros((image_class_embeds.shape[0], image_class_embeds.shape[1], model_params.text_encoder.dim), device=image_class_embeds.device), image_class_embeds)

    image_class_embeds = F.normalize(image_class_embeds, dim=-1, eps=1e-6)
    query_embs = F.normalize(query_embs, dim=-1, eps=1e-6)

    pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embs)

    logit_shift = F.linear(image_embs, head_weights.logit_shift, bias=head_weights.logit_shift_b)
    logit_scale = F.elu(F.linear(image_embs, head_weights.logit_scale, bias=head_weights.logit_scale_b)) + 1
    pred_logits = (pred_logits + logit_shift) * logit_scale

    if query_mask is not None:
        query_mask = query_mask.unsqueeze(-2) if query_mask.ndim > 1 else query_mask
        pred_logits = torch.where(query_mask == 0, torch.finfo(pred_logits.dtype).min, pred_logits).float()
    
    return (pred_logits, image_class_embeds)

def compute_box_bias(num_patches: int) -> torch.Tensor:
    # Create grid coordinates
    coords = torch.linspace(1, num_patches, num_patches, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="xy")
    
    # Compute box coordinates and size
    box_coords = torch.stack((xx, yy), dim=-1).view(-1, 2) / num_patches
    box_coords = torch.clip(box_coords, 0.0, 1.0)
    box_size = torch.full((num_patches**2, 2), 1.0 / num_patches)
    
    # Compute biases
    bias = torch.log(torch.cat([box_coords, box_size], dim=-1) + 1e-4)
    bias -= torch.log1p(torch.cat([-box_coords, -box_size], dim=-1) + 1e-4)
    
    return bias    

def text_obj_det(token_ids, attn_mask, pixel_values, w, model_params):
    # Encode vision and text
    with timer("encode", gpu=True):
        _, vision_full = encode_vision(pixel_values, w.vision_weights, model_params)
        text_features,_ = encode_text(token_ids, w.text_weights, model_params, attn_mask)
    
    # Project and normalize features
    text_features = F.linear(text_features, w.text_proj)
    vision_full = F.normalize(vision_full, dim=-1, eps=1e-6)
    text_features = F.normalize(text_features, dim=-1, eps=1e-6)
    
    # Process vision features
    feature_map = F.layer_norm(vision_full, (model_params.vision_encoder.dim,), weight=w.vision_weights.post_norm, bias=w.vision_weights.post_norm_b)
    feature_map = feature_map[:, 1:, :] * torch.broadcast_to(feature_map[:, :1, :], feature_map[:, :-1].shape)
    feature_map = F.layer_norm(feature_map, (model_params.vision_encoder.dim,), weight=w.layer_norm, bias=w.layer_norm_b)
    
    # Reshape feature map
    batch_size = feature_map.shape[0]
    new_size = (batch_size, model_params.num_patches_sqrt, model_params.num_patches_sqrt, feature_map.shape[-1])
    image_feats = feature_map.reshape(new_size).reshape(batch_size, -1, feature_map.shape[-1])
    
    # Reshape text features
    max_text_queries = token_ids.shape[0] // batch_size
    text_features = text_features.reshape(batch_size, max_text_queries, -1)
    token_ids = token_ids.reshape(batch_size, max_text_queries, -1)
    query_mask = token_ids[..., 0] > 0
    
    # Predict classes, objectness, and boxes
    with timer("heads", gpu=True):
        pred_logits, _ = class_head(image_feats, text_features, query_mask, w.class_head, model_params)
        
        objectness = F.gelu(F.linear(image_feats, w.objectness_head.w1, w.objectness_head.w1_b))
        objectness = F.gelu(F.linear(objectness, w.objectness_head.w2, w.objectness_head.w2_b))
        objectness_logits = F.linear(objectness, w.objectness_head.w3, w.objectness_head.w3_b)[..., 0]
        
        pred_boxes = F.gelu(F.linear(image_feats, w.box_head.w1, w.box_head.w1_b))
        pred_boxes = F.gelu(F.linear(pred_boxes, w.box_head.w2, w.box_head.w2_b))
        pred_boxes = F.linear(pred_boxes, w.box_head.w3, w.box_head.w3_b)
        
        box_bias = compute_box_bias(model_params.num_patches_sqrt).to(image_feats.device)
        pred_boxes = F.sigmoid(pred_boxes + box_bias)
    
    return pred_logits, objectness_logits, pred_boxes

if __name__ == '__main__':
    from safetensors import safe_open
    state_dict = {}
    device = "cuda"
    with safe_open("weights/model.safetensors", framework="pt") as f:
        for k in f.keys():
            tens = f.get_tensor(k)
            k = k.replace("owlv2.","").replace(".embeddings","")
            k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
            state_dict[k] = tens.to(device)
    weights = load_owlv2_weights(state_dict)

      
    #text_obj_det(seq, attn_mask, torch.rand(1,3,960,960).to(device), weights, OWLV2_B16)

    