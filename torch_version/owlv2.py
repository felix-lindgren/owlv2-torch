import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from PIL import Image

from typing import Optional, Tuple

from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

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

""" def _create_4d_causal_attention_mask(
    input_shape,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device
    )

    return attention_mask """

class Attention(nn.Module): 
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()
        # project the hidden states to the query, key, and value states
        query_states = self.q_proj(hidden_states) #* self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        query_states = self._shape(query_states, tgt_len, bsz)
        
        # compute the attention weights
        sdpa_attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            #is_causal=(causal_attention_mask is not None),
            scale=self.scale
        )

        sdpa_attn_output = sdpa_attn_output.transpose(1, 2)
        sdpa_attn_output = sdpa_attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(sdpa_attn_output)
        
        return attn_output, None

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout):
        super().__init__()

        self.self_attn = Attention(hidden_size, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            QuickGELUActivation(),
            nn.Linear(mlp_dim, hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)


    def forward(self, x, mask=None):
        residual = x
        
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, mask)
        x = x + residual
        
        residual = x
        
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x

class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim, dropout):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class VisionTower(nn.Module):
    def __init__(self, hidden_size, patch_size, image_size):
        super().__init__()
        
        self.class_embedding = nn.Parameter(torch.randn(hidden_size))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=16,
            stride=16,
            bias=False,
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

        # layer norm
        self.pre_layernorm = nn.LayerNorm(hidden_size)
        self.post_layernorm = nn.LayerNorm(hidden_size)

        # Encoder
        self.encoder = Encoder(hidden_size, num_layers=12, num_heads=12, mlp_dim=3072, dropout=0.0)


    def forward(self, x):
        batch_size = x.shape[0]
        patch_embeds = self.patch_embedding(x)  # shape = [batch_size, num_channels, height, width]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        embeddings = self.pre_layernorm(embeddings)
        
        x = self.encoder(embeddings, None) # no mask for vision transformer
        pooled_output = x[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return pooled_output, x
    
class TextTower(nn.Module):
    def __init__(self, hidden_size, num_positions, vocab_size):
        super().__init__()
        
        # Embeddings
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)
        self.token_embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

        # layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        # Encoder
        self.encoder = Encoder(hidden_size, num_layers=12, num_heads=8, mlp_dim=2048, dropout=0.0)


    def forward(self, input_ids, attention_mask=None):

        # Embeddings
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden = inputs_embeds + position_embeddings

        
        # attn masks
        input_shape = input_ids.size()
        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, hidden.dtype, hidden.device)
        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden.dtype)
        attention_mask = attention_mask + causal_attention_mask

        

        # Encoder
        encoder_outputs = self.encoder(hidden, attention_mask)
        last_hidden_state = self.final_layer_norm(encoder_outputs)

        # take features from the end of tokens embedding (end of token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]

        return pooled_output, encoder_outputs

class BoxPredictionHead(nn.Module):
    def __init__(self, hidden_size, out_dim: int = 4):
        super().__init__()

        width = hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output

class ClassPredictionHead(nn.Module):
    def __init__(self, text_dim, vision_dim):
        super().__init__()

        out_dim = text_dim
        self.query_dim = vision_dim

        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # Get class predictions
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)

            pred_logits = torch.where(query_mask == 0, torch.finfo(pred_logits.dtype).min, pred_logits)
            pred_logits = pred_logits.to(torch.float32)

        return (pred_logits, image_class_embeds)

class OwlV2(nn.Module):
    def __init__(self, project_dim=512, vision_dim=768, text_dim=512, image_size=960, patch_size=16):
        super().__init__()
        
        self.projected_dim = project_dim
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        self.image_size = image_size
        self.patch_size = patch_size

        self.vision_model = VisionTower(hidden_size=self.vision_dim, patch_size=self.patch_size, image_size=image_size)
        self.text_model = TextTower(hidden_size=512, vocab_size=49408, num_positions=16)

        self.visual_projection = nn.Linear(self.vision_dim, self.projected_dim, bias=False)
        self.text_projection = nn.Linear(512, self.projected_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.layer_norm = nn.LayerNorm(self.vision_dim)


        self.class_head = ClassPredictionHead(self.text_dim, self.vision_dim)
        self.box_head = BoxPredictionHead(self.vision_dim)
        self.objectness_head = BoxPredictionHead(self.vision_dim, out_dim=1)

        self.sqrt_num_patches = self.image_size // self.patch_size
        self.box_bias = self.compute_box_bias(self.sqrt_num_patches)

        self.image_transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32,scale=True),
            SquarePad(),
            T.Resize((self.image_size, self.image_size), antialias=True),
            T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        ])

    
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        
        pixel_values = self.image_transform(image)
        pixel_values = pixel_values.unsqueeze(0)

        return pixel_values


    def get_vision_features(self, pixel_values, normalize=True):
        vision_pooled, vision_full = self.vision_model(pixel_values)
        vision_features = self.visual_projection(vision_pooled)
        vision_features = vision_features / (torch.linalg.norm(vision_features, dim=-1, keepdim=True) + 1e-6)
        return vision_features, vision_pooled, vision_full
    
    def get_text_features(self, token_ids, attention_mask, normalize=True):
        y = self.text_model(token_ids, attention_mask)
        y = self.text_projection(y)
        y = y / (torch.linalg.norm(y, dim=-1, keepdim=True) + 1e-6)
        return y

    def forward(self, pixel_values, token_ids, attention_mask):
        vision_features, vision_pooled, vision_full = self.get_vision_features(pixel_values)
        text_features = self.get_text_features(token_ids, attention_mask)

        logit_scale = self.logit_scale.exp().to(device=vision_features.device)
        logits_per_text = torch.matmul(vision_features, text_features.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        return logits_per_image, logits_per_text, vision_features, text_features, vision_full, text_features
    
    def new_forward_object_detection(self, pixel_values, token_ids, attention_mask):

        # Get vision feature map and reshape to PxP

        # Get vision cls token

        # Merge image embedding with cls token

        # Get pooled text features

        # Predict object classes

        # Predict objectness

        # Predict object boxes
        pass


    
    def forward_object_detection(self, pixel_values, token_ids, attention_mask):

        logits_per_image, logits_per_text, vision_features, text_features, vision_full, text_features_raw = self.forward(pixel_values, token_ids, attention_mask)

        feature_map = self.vision_model.post_layernorm(vision_full)
        class_token_out = torch.broadcast_to(feature_map[:, :1, :], feature_map[:, :-1].shape)

        # Merge image embedding with class tokens
        feature_map = feature_map[:, 1:, :] * class_token_out
        feature_map = self.layer_norm(feature_map)
        
        new_size = (
            feature_map.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            feature_map.shape[-1],
        )
        feature_map = feature_map.reshape(new_size)

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        max_text_queries = token_ids.shape[0] // batch_size
        text_features = text_features.reshape(batch_size, max_text_queries, text_features.shape[-1])

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        token_ids = token_ids.reshape(batch_size, max_text_queries, token_ids.shape[-1])
        query_mask = token_ids[..., 0] > 0

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_head(image_feats, text_features, query_mask)

        # Predict objectness
        #objectness_logits = self.objectness_predictor(image_feats)
        objectness_logits = self.objectness_head(image_feats)
        objectness_logits = objectness_logits[..., 0]

        # Predict object boxes
        pred_boxes = self.box_head(image_feats)
        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        box_bias = self.box_bias.to(feature_map.device)
        pred_boxes += box_bias
        pred_boxes = F.sigmoid(pred_boxes)

        return pred_logits, objectness_logits, pred_boxes, class_embeds, (feature_map, text_features)

    @staticmethod
    def normalize_grid_corner_coordinates(num_patches: int) -> torch.Tensor:
        # Create grid coordinates using torch
        x_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        y_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")

        # Stack the coordinates and divide by num_patches
        box_coordinates = torch.stack((xx, yy), dim=-1)
        box_coordinates /= num_patches

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.view(-1, 2)

        return box_coordinates

    def compute_box_bias(self, num_patches: int, feature_map: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        if feature_map is not None:
            raise ValueError("feature_map has been deprecated as an input. Please pass in num_patches instead")
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(num_patches)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    

    
if __name__ == '__main__':
    from safetensors import safe_open
    state_dict = {}
    with safe_open("weights/model.safetensors", framework="pt") as f:
        for k in f.keys():
            tens = f.get_tensor(k)
            k = k.replace("owlv2.","").replace(".embeddings","")
            k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
            state_dict[k] = tens
    model = OwlV2()
    model = model.eval()
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    with torch.no_grad():
        x = torch.randn(1, 3, 960,960)
        token_id, attn_mask = torch.randint(0, 49408, (1, 16)), torch.ones(1, 16)
        x = x.cuda()
        token_id = token_id.cuda()
        attn_mask = attn_mask.cuda()
        pred_logits, _, pred_boxes, _ = model.forward_object_detection(x, token_id, attn_mask)
        print("pred_logits",pred_logits.shape)
        print("pred_boxes",pred_boxes.shape)

    """ with torch.no_grad():
        x = torch.ones(1, 12, dtype=torch.int64)
        y = model.text_model.forward(x)
        #y = model.text_model.encoder.layers[0].forward(x)
        print("y",y.shape)
        #print(x-y) """