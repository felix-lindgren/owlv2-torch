from typing import Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from OWLv2torch.utils.query_batching import normalize_detection_queries

try:
    import open_clip
except ImportError:  # pragma: no cover - exercised only when optional dep is absent.
    open_clip = None


class BoxPredictionHead(nn.Module):
    def __init__(self, hidden_size, out_dim: int = 4):
        super().__init__()

        self.dense0 = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(hidden_size, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output


class ClassPredictionHead(nn.Module):
    def __init__(self, query_dim, image_dim):
        super().__init__()

        self.query_dim = query_dim
        self.image_dim = image_dim
        self.dense0 = nn.Linear(image_dim, query_dim)
        self.logit_shift = nn.Linear(image_dim, 1)
        self.logit_scale = nn.Linear(image_dim, 1)
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
            return pred_logits, image_class_embeds

        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )
        query_embeds = query_embeds / (
            torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )

        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.elu(self.logit_scale(image_embeds)) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)
            pred_logits = torch.where(
                query_mask == 0, torch.finfo(pred_logits.dtype).min, pred_logits
            )
        return pred_logits, image_class_embeds


class OpenClipTower(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-SO400M-14-SigLIP-384",
        pretrained: Optional[str] = "webli",
        cache_dir: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open_clip is required for OWLv2torch.openclip_version. "
                "Install it with `pip install open-clip-torch`."
            )

        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir
        self.model_name = model_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            **model_kwargs,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embed_dim = self._infer_embed_dim()

    def _infer_embed_dim(self) -> int:
        for attr in ("output_dim", "embed_dim"):
            value = getattr(self.model, attr, None)
            if isinstance(value, int):
                return value

        text_projection = getattr(self.model, "text_projection", None)
        if isinstance(text_projection, nn.Linear):
            return text_projection.out_features
        if isinstance(text_projection, torch.Tensor):
            return text_projection.shape[-1]

        visual = getattr(self.model, "visual", None)
        value = getattr(visual, "output_dim", None)
        if isinstance(value, int):
            return value

        raise ValueError(f"Could not infer OpenCLIP embedding dimension for {self.model_name!r}")

    def tokenize(self, texts):
        return self.tokenizer(texts)

    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        return self.preprocess(image).unsqueeze(0)

    def encode_image(self, pixel_values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        try:
            return self.model.encode_image(pixel_values, normalize=normalize)
        except TypeError:
            image_features = self.model.encode_image(pixel_values)
            if normalize:
                image_features = image_features / (
                    torch.linalg.norm(image_features, dim=-1, keepdim=True) + 1e-6
                )
            return image_features

    def encode_text(self, token_ids: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        try:
            return self.model.encode_text(token_ids, normalize=normalize)
        except TypeError:
            text_features = self.model.encode_text(token_ids)
            if normalize:
                text_features = text_features / (
                    torch.linalg.norm(text_features, dim=-1, keepdim=True) + 1e-6
                )
            return text_features

    def _num_prefix_tokens(self, num_tokens: int) -> int:
        visual = getattr(self.model, "visual", None)
        num_prefix_tokens = getattr(visual, "num_prefix_tokens", None)
        if isinstance(num_prefix_tokens, int):
            return num_prefix_tokens

        num_patch_tokens = num_tokens - 1
        if int(num_tokens ** 0.5) ** 2 == num_tokens:
            return 0
        if int(num_patch_tokens ** 0.5) ** 2 == num_patch_tokens:
            return 1
        return 1

    def _tokens_from_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4:
            patch_tokens = value.flatten(2).transpose(1, 2)
            pooled = patch_tokens.mean(dim=1, keepdim=True)
            return torch.cat([pooled, patch_tokens], dim=1)

        if value.ndim == 3:
            num_prefix_tokens = self._num_prefix_tokens(value.shape[1])
            if num_prefix_tokens == 0:
                pooled = value.mean(dim=1, keepdim=True)
                return torch.cat([pooled, value], dim=1)
            if num_prefix_tokens == 1:
                return value

            pooled = value[:, :num_prefix_tokens, :].mean(dim=1, keepdim=True)
            patch_tokens = value[:, num_prefix_tokens:, :]
            return torch.cat([pooled, patch_tokens], dim=1)

        if value.ndim == 2:
            return value.unsqueeze(1)

        raise ValueError(f"Unsupported OpenCLIP dense feature shape: {tuple(value.shape)}")

    def _tokens_from_sequence(self, value) -> torch.Tensor:
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if len(tensors) == 1:
            return self._tokens_from_tensor(tensors[0])
        if len(tensors) >= 2:
            prefix = tensors[0]
            spatial = tensors[-1]
            patch_tokens = self._tokens_from_tensor(spatial)[:, 1:, :]
            if prefix.ndim == 2:
                prefix = prefix.unsqueeze(1)
            if prefix.ndim == 3:
                prefix = prefix.mean(dim=1, keepdim=True)
                return torch.cat([prefix, patch_tokens], dim=1)
        raise ValueError("Could not parse OpenCLIP dense feature sequence")

    def _tokens_from_output(self, output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return self._tokens_from_tensor(output)

        if isinstance(output, (list, tuple)):
            return self._tokens_from_sequence(output)

        if isinstance(output, dict):
            for key in ("image_intermediates", "visual_intermediates", "intermediates"):
                value = output.get(key)
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    return self._tokens_from_output(value[-1])

            for key in ("image_tokens", "tokens", "x", "features", "last_hidden_state"):
                value = output.get(key)
                if value is not None:
                    return self._tokens_from_output(value)

            raise ValueError(
                f"Unsupported OpenCLIP intermediate output keys: {sorted(output.keys())}"
            )

        raise ValueError(f"Unsupported OpenCLIP dense feature output type: {type(output)!r}")

    def _project_visual_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] == self.embed_dim:
            return tokens

        visual = getattr(self.model, "visual", None)
        proj = getattr(visual, "proj", None)
        if isinstance(proj, nn.Linear):
            tokens = proj(tokens)
        elif isinstance(proj, torch.Tensor):
            tokens = tokens @ proj

        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                "OpenCLIP dense visual tokens are not in the shared embedding dimension "
                f"and no usable visual projection was found: got {tokens.shape[-1]}, "
                f"expected {self.embed_dim}."
            )
        return tokens

    def dense_image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        errors = []
        try:
            output = self.model.forward_intermediates(
                image=pixel_values,
                image_indices=1,
                normalize=False,
                normalize_intermediates=True,
                image_output_fmt="NLC",
                image_output_extra_tokens=True,
            )
            return self._project_visual_tokens(self._tokens_from_output(output))
        except Exception as exc:  # pragma: no cover - model-family fallback.
            errors.append(exc)

        visual = getattr(self.model, "visual", None)
        if hasattr(visual, "forward_intermediates"):
            try:
                output = visual.forward_intermediates(
                    pixel_values,
                    indices=1,
                    normalize_intermediates=True,
                    output_fmt="NLC",
                    output_extra_tokens=True,
                )
                return self._project_visual_tokens(self._tokens_from_output(output))
            except Exception as exc:  # pragma: no cover - model-family fallback.
                errors.append(exc)

        for method_name in ("forward_features", "trunk"):
            method = getattr(visual, method_name, None)
            if method is None:
                continue
            try:
                output = method(pixel_values)
                return self._project_visual_tokens(self._tokens_from_output(output))
            except Exception as exc:  # pragma: no cover - model-family fallback.
                errors.append(exc)

        details = "; ".join(str(exc) for exc in errors)
        raise RuntimeError(f"Could not extract dense image tokens from {self.model_name!r}: {details}")


class OwlV2(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-SO400M-14-SigLIP-384",
        pretrained: Optional[str] = "webli",
        cache_dir: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        merge_class_token: bool = True,
    ):
        super().__init__()

        self.openclip = OpenClipTower(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
            model_kwargs=model_kwargs,
        )
        self.model = self.openclip.model
        self.tokenizer = self.openclip.tokenizer
        self.preprocess = self.openclip.preprocess
        self.projected_dim = self.openclip.embed_dim
        self.vision_dim = self.openclip.embed_dim
        self.text_dim = self.openclip.embed_dim
        self.merge_class_token = merge_class_token

        self.layer_norm = nn.LayerNorm(self.vision_dim, eps=1e-5)
        self.class_head = ClassPredictionHead(self.text_dim, self.vision_dim)
        self.box_head = BoxPredictionHead(self.vision_dim)
        self.objectness_head = BoxPredictionHead(self.vision_dim, out_dim=1)

        self.sqrt_num_patches = None
        self.register_buffer("box_bias", torch.empty(0, 4), persistent=False)

    def tokenize(self, texts):
        return self.openclip.tokenize(texts)

    def preprocess_image(self, image):
        return self.openclip.preprocess_image(image)

    def get_vision_features(self, pixel_values, normalize=True):
        vision_features = self.openclip.encode_image(pixel_values, normalize=normalize)
        vision_full = self.openclip.dense_image_tokens(pixel_values)
        vision_pooled = vision_full[:, 0, :]
        return vision_features, vision_pooled, vision_full

    def get_text_features(self, token_ids, attention_mask=None, normalize=True):
        del attention_mask
        return self.openclip.encode_text(token_ids, normalize=normalize)

    def forward(self, pixel_values, token_ids, attention_mask=None):
        vision_features, _, vision_full = self.get_vision_features(pixel_values)
        text_features = self.get_text_features(token_ids, attention_mask)

        logit_scale = getattr(self.model, "logit_scale", None)
        scale = logit_scale.exp() if logit_scale is not None else 1.0
        logits_per_text = torch.matmul(vision_features, text_features.t()) * scale

        logit_bias = getattr(self.model, "logit_bias", None)
        if logit_bias is not None:
            logits_per_text = logits_per_text + logit_bias

        logits_per_image = logits_per_text.t()
        return logits_per_image, logits_per_text, vision_features, text_features, vision_full

    @staticmethod
    def normalize_grid_corner_coordinates(num_patches: int) -> torch.Tensor:
        x_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        y_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")
        box_coordinates = torch.stack((xx, yy), dim=-1)
        box_coordinates /= num_patches
        return box_coordinates.view(-1, 2)

    def compute_box_bias(self, num_patches: int, feature_map: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        if feature_map is not None:
            raise ValueError("feature_map has been deprecated as an input. Please pass in num_patches instead")

        box_coordinates = self.normalize_grid_corner_coordinates(num_patches)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)
        box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)
        return torch.cat([box_coord_bias, box_size_bias], dim=-1)

    def _set_patch_grid_from_tokens(self, num_patch_tokens: int, device: torch.device, dtype: torch.dtype):
        sqrt_num_patches = int(num_patch_tokens ** 0.5)
        if sqrt_num_patches * sqrt_num_patches != num_patch_tokens:
            raise ValueError(
                "OWLv2 detection expects a square patch grid, but OpenCLIP returned "
                f"{num_patch_tokens} patch tokens."
            )

        if self.sqrt_num_patches == sqrt_num_patches and self.box_bias.numel() > 0:
            return

        self.sqrt_num_patches = sqrt_num_patches
        self.box_bias = self.compute_box_bias(sqrt_num_patches).to(device=device, dtype=dtype)

    def forward_object_detection(self, pixel_values, token_ids, attention_mask=None):
        batch_size = pixel_values.shape[0]
        token_ids, attention_mask, max_text_queries = normalize_detection_queries(
            token_ids, attention_mask, batch_size
        )

        vision_full = self.openclip.dense_image_tokens(pixel_values)
        text_features = self.get_text_features(token_ids, attention_mask)

        class_token = vision_full[:, :1, :]
        image_feats = vision_full[:, 1:, :]
        if self.merge_class_token:
            image_feats = image_feats * torch.broadcast_to(class_token, image_feats.shape)
        image_feats = self.layer_norm(image_feats)
        self._set_patch_grid_from_tokens(image_feats.shape[1], image_feats.device, image_feats.dtype)

        feature_map = image_feats.reshape(
            image_feats.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_feats.shape[-1],
        )

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        text_features = text_features.reshape(batch_size, max_text_queries, text_features.shape[-1])
        token_ids = token_ids.reshape(batch_size, max_text_queries, token_ids.shape[-1])
        query_mask = token_ids[..., 0] > 0

        pred_logits, class_embeds = self.class_head(image_feats, text_features, query_mask)
        objectness_logits = self.objectness_head(image_feats)[..., 0]
        pred_boxes = self.box_head(image_feats)
        pred_boxes = torch.sigmoid(pred_boxes + self.box_bias)

        return pred_logits, objectness_logits, pred_boxes, class_embeds, (feature_map, text_features)

    def postprocess_boxes(self, boxes, image_size, compensate_padding=True):
        if isinstance(image_size, (list, tuple)):
            image_height = torch.tensor([i[0] for i in image_size])
            image_width = torch.tensor([i[1] for i in image_size])
        elif isinstance(image_size, torch.Tensor):
            image_height, image_width = image_size.unbind(1)
        else:
            raise ValueError("`target_sizes` must be a list, tuple or torch.Tensor")

        center_x, center_y, width, height = boxes.unbind(-1)
        boxes = torch.stack(
            [
                center_x - 0.5 * width,
                center_y - 0.5 * height,
                center_x + 0.5 * width,
                center_y + 0.5 * height,
            ],
            dim=-1,
        )

        if compensate_padding:
            hl = image_height > image_width
            wl = image_width > image_height
            eps = torch.finfo(boxes.dtype).eps

            if hl.any():
                clip_factor = (
                    (image_width[hl] / image_height[hl])
                    .to(device=boxes.device, dtype=boxes.dtype)
                    .view(-1, 1, 1)
                    .clamp_min(eps)
                )
                boxes[hl, :, 0:4:2] = torch.clamp(
                    boxes[hl, :, 0:4:2], torch.tensor(0.0, device=boxes.device), clip_factor
                )
                boxes[hl, :, 0:4:2] = boxes[hl, :, 0:4:2] / clip_factor

            if wl.any():
                clip_factor = (
                    (image_height[wl] / image_width[wl])
                    .to(device=boxes.device, dtype=boxes.dtype)
                    .view(-1, 1, 1)
                    .clamp_min(eps)
                )
                boxes[wl, :, 1:4:2] = torch.clamp(
                    boxes[wl, :, 1:4:2], torch.tensor(0.0, device=boxes.device), clip_factor
                )
                boxes[wl, :, 1:4:2] = boxes[wl, :, 1:4:2] / clip_factor

        scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
        scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
        return boxes * scale_factor
