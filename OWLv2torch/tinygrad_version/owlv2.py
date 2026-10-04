"""tinygrad port of :class:`OWLv2torch.torch_version.owlv2.OwlV2` (inference only).

Mirrors the torch module layer for layer and loads the same HF safetensors
checkpoint. Image preprocessing is not ported: feed it the ``pixel_values``
produced by the torch model's ``preprocess_image`` (as a numpy array or Tensor).
"""
from typing import Optional

import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.nn.state import load_state_dict, safe_load
from huggingface_hub.file_download import hf_hub_download

from OWLv2torch.utils.hf_hub_utils import find_safetensors_in_cache

MODEL_CONFIGS = {
    "base": dict(
        model_string="google/owlv2-base-patch16-ensemble",
        project_dim=512, vision_dim=768, text_dim=512, image_size=960, patch_size=16,
        mlp_dim=3072, num_layers=12, num_heads=12,
        num_text_layers=12, text_num_heads=8, text_mlp_dim=2048,
    ),
    "large": dict(
        model_string="google/owlv2-large-patch14-ensemble",
        project_dim=768, vision_dim=1024, text_dim=768, image_size=1008, patch_size=14,
        mlp_dim=4096, num_layers=24, num_heads=16,
        num_text_layers=12, text_num_heads=12, text_mlp_dim=3072,
    ),
}

F32_MIN = float(np.finfo(np.float32).min)


def _l2_normalize(x: Tensor) -> Tensor:
    return x / (x.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)


def build_causal_padding_mask(input_ids: Tensor) -> Tensor:
    """[B, 1, L, L] additive causal + padding mask; pad id is 0 (see the torch version)."""
    B, L = input_ids.shape
    causal = Tensor.ones(L, L, dtype=dtypes.bool).triu(1).reshape(1, 1, L, L)
    pad = (input_ids == 0).reshape(B, 1, 1, L)
    return (causal | pad).where(F32_MIN, 0.0).cast(dtypes.float32)


class Attention:
    def __init__(self, hidden_size, num_heads):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, L, D = x.shape
        q, k, v = [
            p(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            for p in (self.q_proj, self.k_proj, self.v_proj)
        ]
        # tinygrad's SDPA scales by 1/sqrt(head_dim), same as the torch module.
        out = q.scaled_dot_product_attention(k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class MLP:
    def __init__(self, hidden_size, mlp_dim):
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).quick_gelu())


class EncoderLayer:
    def __init__(self, hidden_size, num_heads, mlp_dim):
        self.self_attn = Attention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = MLP(hidden_size, mlp_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class Encoder:
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim):
        self.layers = [EncoderLayer(hidden_size, num_heads, mlp_dim) for _ in range(num_layers)]

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class VisionTower:
    def __init__(self, hidden_size, patch_size, image_size, num_layers, num_heads, mlp_dim):
        self.class_embedding = Tensor.zeros(hidden_size)
        self.patch_embedding = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)
        self.num_positions = (image_size // patch_size) ** 2 + 1
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)
        self.pre_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.encoder = Encoder(hidden_size, num_layers, num_heads, mlp_dim)
        # The sequence (P*P + 1 tokens, 3601 for base) is zero-padded up to a
        # multiple of this so the encoder's kernels get well-shaped dims; the
        # padded keys are masked out and the padded rows dropped. 1 disables it.
        self.seq_pad_multiple = 32

    def __call__(self, x: Tensor):
        B = x.shape[0]
        patch_embeds = self.patch_embedding(x).flatten(2).transpose(1, 2)  # [B, P*P, D]
        class_embeds = self.class_embedding.reshape(1, 1, -1).expand(B, 1, -1)
        embeddings = class_embeds.cat(patch_embeds, dim=1)
        # Position ids are 0..N-1, i.e. the whole table in order.
        embeddings = self.pre_layernorm(embeddings + self.position_embedding.weight.unsqueeze(0))

        N = self.num_positions
        pad = -N % self.seq_pad_multiple
        mask = None
        if pad:
            embeddings = embeddings.pad((None, (0, pad), None))
            mask = (Tensor.arange(N + pad) >= N).where(F32_MIN, 0.0).reshape(1, 1, 1, N + pad)
        x = self.encoder(embeddings, mask)
        if pad:
            # contiguous() keeps the slice from being fused back into the last
            # layer, which would hand its kernels the unpadded length again.
            x = x.contiguous()[:, :N]
        pooled_output = self.post_layernorm(x[:, 0, :])
        return pooled_output, x


class TextTower:
    def __init__(self, hidden_size, num_positions, vocab_size, num_layers, num_heads, mlp_dim):
        self.position_embedding = nn.Embedding(num_positions, hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.encoder = Encoder(hidden_size, num_layers, num_heads, mlp_dim)

    def __call__(self, input_ids: Tensor):
        L = input_ids.shape[-1]
        hidden = self.token_embedding(input_ids) + self.position_embedding.weight[:L].unsqueeze(0)
        encoder_outputs = self.encoder(hidden, build_causal_padding_mask(input_ids))
        last_hidden_state = self.final_layer_norm(encoder_outputs)
        # Features from the end-of-text token (highest id under CLIP's tokenizer).
        eot = input_ids.argmax(axis=-1).reshape(-1, 1, 1).expand(-1, 1, last_hidden_state.shape[-1])
        pooled_output = last_hidden_state.gather(1, eot).squeeze(1)
        return pooled_output, encoder_outputs


class BoxPredictionHead:
    def __init__(self, hidden_size, out_dim: int = 4):
        self.dense0 = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        # torch's nn.GELU() default is the exact erf form.
        x = self.dense0(x).gelu(approximate="none")
        x = self.dense1(x).gelu(approximate="none")
        return self.dense2(x)


class ClassPredictionHead:
    def __init__(self, text_dim, vision_dim):
        self.query_dim = vision_dim
        self.dense0 = nn.Linear(vision_dim, text_dim)
        self.logit_shift = nn.Linear(vision_dim, 1)
        self.logit_scale = nn.Linear(vision_dim, 1)

    def __call__(self, image_embeds: Tensor, query_embeds: Optional[Tensor], query_mask: Optional[Tensor]):
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            B, N = image_class_embeds.shape[:2]
            return Tensor.zeros(B, N, self.query_dim), image_class_embeds

        image_class_embeds = _l2_normalize(image_class_embeds)
        query_embeds = _l2_normalize(query_embeds)
        pred_logits = image_class_embeds @ query_embeds.transpose(-1, -2)  # [B, P, Q]

        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds).elu() + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = query_mask.unsqueeze(-2)
            pred_logits = query_mask.where(pred_logits, F32_MIN)
        return pred_logits, image_class_embeds


class OwlV2:
    def __init__(self, model_type="base"):
        cfg = MODEL_CONFIGS["base" if model_type == "base" else "large"]
        self.model_string = cfg["model_string"]
        self.projected_dim = cfg["project_dim"]
        self.vision_dim = cfg["vision_dim"]
        self.text_dim = cfg["text_dim"]
        self.image_size = cfg["image_size"]
        self.patch_size = cfg["patch_size"]

        self.vision_model = VisionTower(
            hidden_size=self.vision_dim, patch_size=self.patch_size, image_size=self.image_size,
            num_layers=cfg["num_layers"], num_heads=cfg["num_heads"], mlp_dim=cfg["mlp_dim"],
        )
        self.text_model = TextTower(
            hidden_size=self.text_dim, num_positions=16, vocab_size=49408,
            num_layers=cfg["num_text_layers"], num_heads=cfg["text_num_heads"], mlp_dim=cfg["text_mlp_dim"],
        )
        self.visual_projection = nn.Linear(self.vision_dim, self.projected_dim, bias=False)
        self.text_projection = nn.Linear(self.text_dim, self.projected_dim, bias=False)
        self.logit_scale = Tensor(2.6592)
        self.layer_norm = nn.LayerNorm(self.vision_dim, eps=1e-5)

        self.class_head = ClassPredictionHead(self.text_dim, self.vision_dim)
        self.box_head = BoxPredictionHead(self.vision_dim)
        self.objectness_head = BoxPredictionHead(self.vision_dim, out_dim=1)

        self.sqrt_num_patches = self.image_size // self.patch_size
        self._load_model(self.model_string)
        # Not a checkpoint weight, so set after loading (strict load would want it).
        self.box_bias = Tensor(self.compute_box_bias(self.sqrt_num_patches)).realize()

    def _load_model(self, model_path):
        cache_path = find_safetensors_in_cache(model_path)
        if len(cache_path) == 0:
            hf_hub_download(repo_id=model_path, filename="model.safetensors")
            cache_path = find_safetensors_in_cache(model_path)
        if len(cache_path) != 1:
            raise RuntimeError(f"Expected one safetensors file for {model_path}, found {cache_path}")
        state_dict = {
            k.replace("owlv2.", "").replace(".embeddings", ""): v
            for k, v in safe_load(str(cache_path[0])).items()
        }
        load_state_dict(self, state_dict, strict=True, verbose=False)

    @staticmethod
    def compute_box_bias(num_patches: int) -> np.ndarray:
        """Same as the torch version, in float32 numpy."""
        coords = np.arange(1, num_patches + 1, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords, indexing="xy")
        box_coordinates = (np.stack((xx, yy), axis=-1) / np.float32(num_patches)).reshape(-1, 2)
        box_coordinates = np.clip(box_coordinates, 0.0, 1.0)
        box_coord_bias = np.log(box_coordinates + 1e-4) - np.log1p(-box_coordinates + 1e-4)
        box_size = np.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = np.log(box_size + 1e-4) - np.log1p(-box_size + 1e-4)
        return np.concatenate([box_coord_bias, box_size_bias], axis=-1).astype(np.float32)

    @staticmethod
    def _as_tensor(x, dtype=None) -> Tensor:
        if isinstance(x, Tensor):
            return x if dtype is None else x.cast(dtype)
        return Tensor(np.ascontiguousarray(x), dtype=dtype)

    def get_vision_features(self, pixel_values, normalize=True):
        vision_pooled, vision_full = self.vision_model(self._as_tensor(pixel_values, dtypes.float32))
        vision_features = self.visual_projection(vision_pooled)
        if normalize:
            vision_features = _l2_normalize(vision_features)
        return vision_features, vision_pooled, vision_full

    def get_text_features(self, token_ids, attention_mask=None, normalize=True):
        # attention_mask is ignored, as in the torch version: padding comes from token_ids == 0.
        y, _ = self.text_model(self._as_tensor(token_ids, dtypes.int32))
        y = self.text_projection(y)
        if normalize:
            y = _l2_normalize(y)
        return y

    def encode_detection_queries(self, token_ids, attention_mask=None):
        token_ids = self._as_tensor(token_ids, dtypes.int32)
        if token_ids.ndim != 2:
            raise ValueError(f"expected token_ids [num_queries, seq_len], got {token_ids.shape}")
        return self.get_text_features(token_ids), token_ids[..., 0] > 0

    def forward(self, pixel_values, token_ids, attention_mask=None):
        vision_features, _, vision_full = self.get_vision_features(pixel_values)
        text_features = self.get_text_features(token_ids)
        logits_per_image = (vision_features @ text_features.T) * self.logit_scale.exp()
        return logits_per_image, logits_per_image.T, vision_features, text_features, vision_full

    __call__ = forward

    def forward_object_detection(self, pixel_values, token_ids, attention_mask=None):
        """``token_ids`` is a shared ``[Q, L]`` query set or batched ``[B, Q, L]``."""
        pixel_values = self._as_tensor(pixel_values, dtypes.float32)
        token_ids = self._as_tensor(token_ids, dtypes.int32)
        B = pixel_values.shape[0]
        if token_ids.ndim == 3:
            if token_ids.shape[0] != B:
                raise ValueError(f"token_ids batch {token_ids.shape[0]} does not match pixel batch {B}")
            Q, L = token_ids.shape[1:]
            text_features = self.get_text_features(token_ids.reshape(B * Q, L)).reshape(B, Q, -1)
        elif token_ids.ndim == 2:
            text_features = self.get_text_features(token_ids)
        else:
            raise ValueError(f"token_ids must be [Q, L] or [B, Q, L], got {token_ids.shape}")
        query_mask = token_ids[..., 0] > 0
        return self.forward_object_detection_from_embeddings(pixel_values, text_features, query_mask)

    def forward_object_detection_from_embeddings(self, pixel_values, query_embeds: Tensor, query_mask: Optional[Tensor] = None):
        pixel_values = self._as_tensor(pixel_values, dtypes.float32)
        B = pixel_values.shape[0]
        if query_embeds.ndim == 2:
            text_features = query_embeds.unsqueeze(0).expand(B, -1, -1)
            if query_mask is not None:
                query_mask = query_mask.unsqueeze(0).expand(B, -1)
        elif query_embeds.ndim == 3:
            text_features = query_embeds
        else:
            raise ValueError(f"query_embeds must be [Q, D] or [B, Q, D], got {query_embeds.shape}")

        _, vision_full = self.vision_model(pixel_values)
        image_feats, feature_map = self._detection_image_features_from_vision(vision_full)

        pred_logits, class_embeds = self.class_head(image_feats, text_features, query_mask)
        objectness_logits = self.objectness_head(image_feats)[..., 0]
        pred_boxes = (self.box_head(image_feats) + self.box_bias).sigmoid()
        return pred_logits, objectness_logits, pred_boxes, class_embeds, (feature_map, text_features)

    def _detection_image_features_from_vision(self, vision_full: Tensor):
        feature_map = self.vision_model.post_layernorm(vision_full)
        feature_map = feature_map[:, 1:, :] * feature_map[:, :1, :]
        feature_map = self.layer_norm(feature_map)
        P = self.sqrt_num_patches
        B, D = feature_map.shape[0], feature_map.shape[-1]
        return feature_map, feature_map.reshape(B, P, P, D)
