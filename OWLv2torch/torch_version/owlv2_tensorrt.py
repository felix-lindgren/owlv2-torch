"""
TensorRT runtime wrapper for the OWLv2 vision tower and detection heads.

Replaces the old torch2trt-based implementation with a thin wrapper around the
native TensorRT 11 Python API (``execute_async_v3`` + named-tensor I/O). Torch
tensors' ``.data_ptr()`` is passed directly to TRT, so there is no host copy
and no extra dependency beyond ``tensorrt`` itself.

The runner launches on ``torch.cuda.current_stream()`` rather than a private
stream, so downstream PyTorch ops are automatically serialized without an
explicit ``cudaStreamSynchronize`` — CPU sync only happens when the caller
actually pulls data back to host (``.cpu()``, ``.item()``, etc.).

Two engines are used: the vision tower, and a small engine for the
class/box/objectness heads. The heads are kept separate so a fine-tuned head
checkpoint only needs the (seconds-long) heads rebuild, not the vision tower.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from OWLv2torch.torch_version.owlv2 import OwlV2
from EzLogger import Timer

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    trt = None
    TRT_AVAILABLE = False

timer = Timer()

HEAD_OUTPUTS = ["pred_logits", "objectness_logits", "pred_boxes", "class_embeds", "image_feats"]


if TRT_AVAILABLE:
    _TRT_TO_TORCH = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }

    class TRTRunner:
        """Minimal TensorRT runtime for a serialized engine.

        With ``reuse_outputs`` (the default) output buffers are allocated lazily
        and reused while the resolved output shape does not change, so each call
        overwrites the tensors returned by the previous one. Pass
        ``reuse_outputs=False`` when outputs are handed back to callers who may
        keep them around.
        """

        def __init__(self, engine_path, input_names, output_names, device="cuda",
                     reuse_outputs=True):
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)

            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(
                    f"Failed to deserialize TRT engine: {engine_path}. Engines only "
                    f"load on the TensorRT version (now {trt.__version__}) and GPU "
                    "they were built with; delete the engine to rebuild it. Delete "
                    "its ONNX too if it was exported before TensorRT 11, since an "
                    "fp32 ONNX now builds an fp32 engine."
                )

            self.context = self.engine.create_execution_context()
            self.input_names = list(input_names)
            self.output_names = list(output_names)
            self.device = torch.device(device)
            self.reuse_outputs = reuse_outputs
            self._output_buffers: dict[str, torch.Tensor] = {}

        @staticmethod
        def _torch_dtype(trt_dtype):
            try:
                return _TRT_TO_TORCH[trt_dtype]
            except KeyError as e:
                raise TypeError(f"Unsupported TRT dtype: {trt_dtype}") from e

        def max_shape(self, name: str) -> tuple:
            """Largest shape the engine's optimization profile accepts for input ``name``."""
            return tuple(self.engine.get_tensor_profile_shape(name, 0)[2])

        def _bind_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
            t = tensor.to(self.device, non_blocking=True).contiguous()
            expected = self._torch_dtype(self.engine.get_tensor_dtype(name))
            if t.dtype != expected:
                t = t.to(expected)
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            return t

        def _bind_output(self, name: str) -> torch.Tensor:
            shape = tuple(self.context.get_tensor_shape(name))
            buf = self._output_buffers.get(name)
            if buf is None or tuple(buf.shape) != shape or not self.reuse_outputs:
                dtype = self._torch_dtype(self.engine.get_tensor_dtype(name))
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                if self.reuse_outputs:
                    self._output_buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())
            return buf

        def __call__(self, *args, **inputs):
            # Allow positional-by-order as well as keyword-by-name.
            if args:
                if inputs:
                    raise ValueError("Pass inputs either positionally or by name, not both.")
                if len(args) != len(self.input_names):
                    raise ValueError(
                        f"Expected {len(self.input_names)} inputs, got {len(args)}")
                inputs = dict(zip(self.input_names, args))

            # Inputs first (sets shapes so output shapes can resolve).
            kept = [self._bind_input(name, inputs[name]) for name in self.input_names]  # noqa: F841

            outs = [self._bind_output(name) for name in self.output_names]

            stream = torch.cuda.current_stream(self.device)
            ok = self.context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False.")
            # No explicit sync: downstream torch ops on the same stream are
            # ordered w.r.t. this launch; a CPU sync will occur automatically
            # when the caller pulls data back to host.
            return outs

else:  # pragma: no cover - only hit when tensorrt isn't installed
    TRTRunner = None


def _onnx_has_fp16_weights(path: Path) -> bool:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return any(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer)


class OwlV2TRT(OwlV2):
    """OWLv2 variant that runs the vision tower and heads with TensorRT when available.

    The heads engine is used for shared ``[num_queries, text_dim]`` queries
    outside autograd. It bakes in the head weights it was built from: if the
    model's head weights change afterwards (e.g. a fine-tuned checkpoint is
    loaded), detection falls back to the torch heads with a warning until
    :meth:`build_trt_heads` rebuilds the engine from the current weights.
    """

    def __init__(
        self,
        engine_path: str | Path | None = None,
        *,
        onnx_path: str | Path | None = None,
        heads_engine_path: str | Path | None = None,
        heads_onnx_path: str | Path | None = None,
        trt_heads: bool = True,
        output_dir: str | Path = ".",
        model_type: str = "base",
        build_missing: bool = True,
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 8,
        max_queries: int = 256,
        workspace_gb: int = 8,
        fp16: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.engine_path = self._artifact_path(
            engine_path, self.output_dir / f"owlv2_vis_{model_type}.engine"
        )
        self.onnx_path = self._artifact_path(
            onnx_path,
            self.engine_path.with_suffix(".onnx")
            if engine_path is not None
            else self.output_dir / f"owlv2_vis_{model_type}.onnx",
        )
        # Heads artifacts sit next to the vision engine unless given explicitly.
        self.heads_engine_path = self._artifact_path(
            heads_engine_path, self.engine_path.parent / f"owlv2_heads_{model_type}.engine"
        )
        self.heads_onnx_path = self._artifact_path(
            heads_onnx_path, self.heads_engine_path.with_suffix(".onnx")
        )
        self.use_trt_heads = trt_heads
        self.build_missing = build_missing
        self.fp16 = fp16
        self.engine_build_kwargs = {
            "min_batch": min_batch,
            "opt_batch": opt_batch,
            "max_batch": max_batch,
            "max_queries": max_queries,
            "workspace_gb": workspace_gb,
        }
        self.trt = None
        self.trt_heads = None
        super().__init__(model_type=model_type)

    @staticmethod
    def _artifact_path(path: str | Path | None, default: Path) -> Path:
        return Path(path) if path is not None else default

    def _can_build(self) -> bool:
        if not TRT_AVAILABLE:
            print("TensorRT is not installed; ONNX was exported but engine build was skipped.")
            return False
        if not torch.cuda.is_available():
            print("CUDA is not available; ONNX was exported but engine build was skipped.")
            return False
        return True

    def _ensure_engine(self, onnx_path: Path, engine_path: Path, export_fn, what: str) -> bool:
        """Export/build what is missing; True if the engine was built from a fresh export."""
        if engine_path.exists():
            return False

        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        exported = not onnx_path.exists()
        if exported:
            print(f"Exporting OWLv2 {what} ONNX to {onnx_path}")
            export_fn(str(onnx_path))
        elif self.fp16 and not _onnx_has_fp16_weights(onnx_path):
            warnings.warn(
                f"{onnx_path} has fp32 weights; TensorRT 11 will build an fp32 engine "
                "from it. Delete it to re-export in fp16.",
                stacklevel=3,
            )

        if not self._can_build():
            return False

        from OWLv2torch.torch_version.build_engine import build_engine

        print(f"Building OWLv2 {what} TensorRT engine to {engine_path}")
        build_engine(
            onnx_file_path=str(onnx_path),
            engine_file_path=str(engine_path),
            **self.engine_build_kwargs,
        )
        return exported and engine_path.exists()

    def _ensure_artifacts(self):
        from OWLv2torch.torch_version.export import export_detection_heads, export_vision_tower

        self._ensure_engine(
            self.onnx_path,
            self.engine_path,
            lambda path: export_vision_tower(self.vision_model, path, self.image_size, fp16=self.fp16),
            "vision",
        )
        if self.use_trt_heads:
            # Only a fresh export is known to hold the current head weights; an
            # engine built from a pre-existing ONNX gets no fingerprint (trusted).
            if self._ensure_engine(
                self.heads_onnx_path,
                self.heads_engine_path,
                lambda path: export_detection_heads(self, path, fp16=self.fp16),
                "heads",
            ):
                self._write_heads_fingerprint()

    def _load_model(self, model_path):
        super()._load_model(model_path)

        if self.build_missing:
            self._ensure_artifacts()

        trt_ready = TRT_AVAILABLE and torch.cuda.is_available()
        self.trt = None
        if trt_ready and self.engine_path.exists():
            self.trt = TRTRunner(
                self.engine_path,
                input_names=["image"],
                output_names=["cls_emb", "full_output"],
            )
        self._load_trt_heads()

    # ------------------------------------------------------------------ heads

    def _head_parameters(self):
        for module in (
            self.vision_model.post_layernorm,
            self.layer_norm,
            self.class_head,
            self.objectness_head,
            self.box_head,
        ):
            yield from module.parameters()

    def _heads_fingerprint(self) -> list[float]:
        """Per-parameter sum and sum of squares; cheap, and changes with any fine-tune."""
        with torch.no_grad():
            stats = [
                torch.stack([p.double().sum(), p.double().square().sum()])
                for p in self._head_parameters()
            ]
            return torch.stack(stats).flatten().cpu().tolist()

    def _heads_state(self):
        # Changes whenever a head parameter is replaced, moved, or written in place.
        return tuple((id(p), p.data_ptr(), p._version) for p in self._head_parameters())

    def _fingerprint_path(self) -> Path:
        return self.heads_engine_path.with_name(self.heads_engine_path.name + ".json")

    def _write_heads_fingerprint(self):
        self._fingerprint_path().write_text(json.dumps({"fingerprint": self._heads_fingerprint()}))

    def _load_trt_heads(self):
        self.trt_heads = None
        if not (self.use_trt_heads and TRT_AVAILABLE and torch.cuda.is_available()
                and self.heads_engine_path.exists()):
            return
        self.trt_heads = TRTRunner(
            self.heads_engine_path,
            input_names=["vision_full", "query_embeds"],
            output_names=HEAD_OUTPUTS,
            reuse_outputs=False,
        )
        self._trt_heads_max_batch = self.trt_heads.max_shape("vision_full")[0]
        self._trt_heads_max_queries = self.trt_heads.max_shape("query_embeds")[0]
        # An engine without a fingerprint is trusted to match the current weights.
        fp_path = self._fingerprint_path()
        if fp_path.exists():
            self._trt_heads_fingerprint = json.loads(fp_path.read_text())["fingerprint"]
        else:
            self._trt_heads_fingerprint = self._heads_fingerprint()
        self._trt_heads_state = None
        self._trt_heads_stale = False

    def build_trt_heads(self):
        """Re-export and rebuild the heads engine from the current head weights."""
        from OWLv2torch.torch_version.export import export_detection_heads

        for path in (self.heads_engine_path, self.heads_onnx_path):
            path.unlink(missing_ok=True)
        self.heads_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_detection_heads(self, str(self.heads_onnx_path), fp16=self.fp16)
        if not self._can_build():
            return
        from OWLv2torch.torch_version.build_engine import build_engine

        build_engine(
            onnx_file_path=str(self.heads_onnx_path),
            engine_file_path=str(self.heads_engine_path),
            **self.engine_build_kwargs,
        )
        self._write_heads_fingerprint()
        self._load_trt_heads()

    def _trt_heads_match_weights(self) -> bool:
        state = self._heads_state()
        if state != self._trt_heads_state:
            current = self._heads_fingerprint()
            self._trt_heads_stale = not np.allclose(
                current, self._trt_heads_fingerprint, rtol=1e-6, atol=1e-6
            )
            self._trt_heads_state = state
            if self._trt_heads_stale:
                warnings.warn(
                    f"Head weights differ from those in {self.heads_engine_path}; using "
                    "the torch heads. Call build_trt_heads() to rebuild the engine.",
                    stacklevel=3,
                )
        return not self._trt_heads_stale

    def _use_trt_heads_for(self, pixel_values, query_embeds) -> bool:
        return (
            self.trt_heads is not None
            and not torch.is_grad_enabled()
            and query_embeds.ndim == 2
            and query_embeds.shape[-1] == self.text_dim
            and 0 < query_embeds.shape[0] <= self._trt_heads_max_queries
            and pixel_values.shape[0] <= self._trt_heads_max_batch
            and query_embeds.device == pixel_values.device
            and self._trt_heads_match_weights()
        )

    def forward_object_detection_from_embeddings(
        self,
        pixel_values: torch.Tensor,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ):
        if not self._use_trt_heads_for(pixel_values, query_embeds):
            return super().forward_object_detection_from_embeddings(
                pixel_values, query_embeds, query_mask
            )

        batch_size, num_queries = pixel_values.shape[0], query_embeds.shape[0]
        _, vision_full = self._get_vision_outputs(pixel_values)
        pred_logits, objectness_logits, pred_boxes, class_embeds, image_feats = self.trt_heads(
            vision_full=vision_full, query_embeds=query_embeds
        )
        if query_mask is not None:
            if query_mask.shape != (num_queries,):
                raise ValueError(
                    "A shared query_mask must have shape "
                    f"[{num_queries}], got {tuple(query_mask.shape)}"
                )
            keep = query_mask.to(device=pred_logits.device, dtype=torch.bool)
            pred_logits = torch.where(keep, pred_logits, torch.finfo(pred_logits.dtype).min)

        P = self.sqrt_num_patches
        feature_map = image_feats.view(batch_size, P, P, image_feats.shape[-1])
        text_features = query_embeds.to(pred_logits.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return pred_logits, objectness_logits, pred_boxes, class_embeds, (feature_map, text_features)

    # ------------------------------------------------------------------ vision

    @timer("trt_inference")
    def trt_inference(self, pixel_data: torch.Tensor):
        pooled_output, full_vision = self.trt(image=pixel_data)
        # The engine was exported from VisionTower directly, so shapes should
        # already be (B, D) and (B, num_positions, D). Reshapes below are
        # defensive against engines built before shape metadata was preserved.
        if pooled_output.dim() != 2:
            pooled_output = pooled_output.reshape(pixel_data.shape[0], -1)
        if full_vision.dim() != 3:
            full_vision = full_vision.reshape(
                pixel_data.shape[0], self.vision_model.num_positions, self.vision_dim
            )
        return pooled_output, full_vision

    def _get_vision_outputs(self, pixel_values):
        if self.trt is not None:
            return self.trt_inference(pixel_values)
        return self.vision_model(pixel_values)


def _parse_args():
    p = argparse.ArgumentParser(description="Run OWLv2 with an optional TensorRT vision tower.")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--onnx", default=None)
    p.add_argument("--engine", default=None)
    p.add_argument("--heads-engine", default=None)
    p.add_argument("--no-trt-heads", action="store_true")
    p.add_argument("--model-type", default="base", choices=["base", "large"])
    p.add_argument("--image", default="img.jpg")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--max-queries", type=int, default=256)
    p.add_argument("--workspace-gb", type=int, default=8)
    p.add_argument("--no-fp16", action="store_true", help="export fp32 ONNX (fp32 engine)")
    p.add_argument("--no-build", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model = OwlV2TRT(
        engine_path=args.engine,
        onnx_path=args.onnx,
        heads_engine_path=args.heads_engine,
        trt_heads=not args.no_trt_heads,
        output_dir=args.output_dir,
        model_type=args.model_type,
        build_missing=not args.no_build,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        max_queries=args.max_queries,
        workspace_gb=args.workspace_gb,
        fp16=not args.no_fp16,
    )

    if not Path(args.image).exists():
        print(f"No image found at {args.image}; artifact setup is complete.")
        raise SystemExit(0)

    img = Image.open(args.image)
    img_array = np.array(img.resize((model.image_size, model.image_size)))
    inputs = img_array.transpose(2, 0, 1)  # (3, H, W)

    model.eval()
    model.cuda()
    print(inputs.shape, inputs.dtype)

    im_pt = torch.from_numpy(inputs).unsqueeze(0).float().cuda()
    res = model.get_vision_features(im_pt)
    print([t.shape for t in res])

    # Quick micro-benchmark of the vision tower.
    for _ in range(10):
        with timer("infer"):
            model.get_vision_features(im_pt)
    torch.cuda.synchronize()
    timer.print_metrics()
