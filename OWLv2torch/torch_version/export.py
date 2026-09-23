"""Export the OWLv2 vision tower and detection heads to ONNX.

TensorRT 11 networks are strongly typed: every layer runs in the precision the
ONNX graph declares, and there is no builder flag that lowers fp32 to fp16.
Reduced precision is therefore decided here, at export time. ``fp16=True``
(the default) exports fp16 weights and compute behind fp32 inputs/outputs, so
callers keep feeding and reading fp32 tensors.

The batch axis (and the query axis of the heads) is exported dynamic so that
``build_engine.py`` can serve variable sizes through an optimization profile.
"""

import argparse
import copy
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify

from OWLv2torch.torch_version.owlv2 import OwlV2

OPSET = 18
VISION_OUTPUTS = ["cls_emb", "full_output"]
HEAD_OUTPUTS = ["pred_logits", "objectness_logits", "pred_boxes", "class_embeds", "image_feats"]


class _Fp16Vision(nn.Module):
    """fp32 image -> vision tower in fp16 -> fp32 outputs."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, image):
        dtype = next(self.vision_model.parameters()).dtype
        pooled, full = self.vision_model(image.to(dtype))
        return pooled.float(), full.float()


class DetectionHeads(nn.Module):
    """Dense vision states + shared query embeddings -> detection outputs.

    Mirrors ``OwlV2.forward_object_detection_from_embeddings`` for shared
    ``[num_queries, text_dim]`` queries without a query mask (the mask is
    applied by the caller). ``image_feats`` is exported too so the caller can
    rebuild the feature map without rerunning the layer norms. Holds only the
    head-side modules, so copying it for export leaves both towers alone.
    """

    def __init__(self, model: OwlV2):
        super().__init__()
        self.post_layernorm = model.vision_model.post_layernorm
        self.layer_norm = model.layer_norm
        self.class_head = model.class_head
        self.objectness_head = model.objectness_head
        self.box_head = model.box_head
        self.register_buffer("box_bias", model.box_bias.detach().clone())

    def forward(self, vision_full, query_embeds):
        dtype = self.box_bias.dtype
        # Same as OwlV2._detection_image_features_from_vision.
        feature_map = self.post_layernorm(vision_full.to(dtype))
        image_feats = self.layer_norm(feature_map[:, 1:, :] * feature_map[:, :1, :])
        text = query_embeds.to(dtype).unsqueeze(0).expand(image_feats.shape[0], -1, -1)
        pred_logits, class_embeds = self.class_head(image_feats, text, None)
        objectness_logits = self.objectness_head(image_feats)[..., 0]
        pred_boxes = torch.sigmoid(self.box_head(image_feats) + self.box_bias)
        outs = (pred_logits, objectness_logits, pred_boxes, class_embeds, image_feats)
        return tuple(o.float() for o in outs)


def _export(module, dummies, input_names, output_names, onnx_path, dynamic_axes):
    onnx_path = Path(onnx_path)
    torch.onnx.export(
        module,
        dummies,
        str(onnx_path),
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
    )

    onnx_model = onnx.load(str(onnx_path))
    model_simp, check = simplify(onnx_model)
    if not check:
        print("WARNING: onnx-simplifier validation failed; saving unsimplified model.")
        model_simp = onnx_model
    # The exporter writes weights to a side ``.data`` file; onnx.load pulled them
    # back in, so save a single self-contained file and drop the stale sidecar.
    onnx.save_model(model_simp, str(onnx_path))
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()


def _prepare(module: nn.Module, fp16: bool) -> nn.Module:
    # Export a CPU copy so the caller's (possibly CUDA, fp32) model is untouched.
    module = copy.deepcopy(module).cpu().eval()
    return module.half() if fp16 else module


def export_vision_tower(
    vision_model: nn.Module,
    onnx_path: str,
    image_size: int,
    fp16: bool = True,
):
    wrapped = _Fp16Vision(_prepare(vision_model, fp16))
    with torch.no_grad():
        _export(
            wrapped,
            (torch.rand(1, 3, image_size, image_size),),
            ["image"],
            VISION_OUTPUTS,
            onnx_path,
            {"image": {0: "batch"}, "cls_emb": {0: "batch"}, "full_output": {0: "batch"}},
        )


def export_detection_heads(model: OwlV2, onnx_path: str, fp16: bool = True):
    """Export the class/box/objectness heads, taking ``full_output`` of the vision tower."""
    heads = _prepare(DetectionHeads(model), fp16)
    num_positions = model.vision_model.num_positions
    with torch.no_grad():
        _export(
            heads,
            (torch.randn(1, num_positions, model.vision_dim), torch.randn(16, model.text_dim)),
            ["vision_full", "query_embeds"],
            HEAD_OUTPUTS,
            onnx_path,
            {
                "vision_full": {0: "batch"},
                "query_embeds": {0: "queries"},
                "pred_logits": {0: "batch", 2: "queries"},
                "objectness_logits": {0: "batch"},
                "pred_boxes": {0: "batch"},
                "class_embeds": {0: "batch"},
                "image_feats": {0: "batch"},
            },
        )


def _parse_args():
    p = argparse.ArgumentParser(description="Export OWLv2 vision tower and heads to ONNX.")
    p.add_argument("--model-type", default="base", choices=["base", "large"])
    p.add_argument("--onnx", default=None, help="vision ONNX path (default owlv2_vis_<type>.onnx)")
    p.add_argument("--heads-onnx", default=None, help="heads ONNX path (default owlv2_heads_<type>.onnx)")
    p.add_argument("--no-heads", action="store_true")
    p.add_argument("--no-fp16", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model = OwlV2(model_type=args.model_type).eval()
    fp16 = not args.no_fp16
    onnx_path = args.onnx or f"owlv2_vis_{args.model_type}.onnx"
    export_vision_tower(model.vision_model, onnx_path, model.image_size, fp16=fp16)
    print(f"Exported vision tower to {onnx_path}")
    if not args.no_heads:
        heads_path = args.heads_onnx or f"owlv2_heads_{args.model_type}.onnx"
        export_detection_heads(model, heads_path, fp16=fp16)
        print(f"Exported detection heads to {heads_path}")
