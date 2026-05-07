"""
TensorRT runtime wrapper for the OWLv2 vision tower.

Replaces the old torch2trt-based implementation with a thin wrapper around the
native TensorRT 10 Python API (``execute_async_v3`` + named-tensor I/O). Torch
tensors' ``.data_ptr()`` is passed directly to TRT, so there is no host copy
and no extra dependency beyond ``tensorrt`` itself.

The runner launches on ``torch.cuda.current_stream()`` rather than a private
stream, so downstream PyTorch ops are automatically serialized without an
explicit ``cudaStreamSynchronize`` — CPU sync only happens when the caller
actually pulls data back to host (``.cpu()``, ``.item()``, etc.).
"""

import argparse
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


if TRT_AVAILABLE:
    _TRT_TO_TORCH = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }

    class TRTRunner:
        """Minimal TensorRT 10 runtime for a serialized engine.

        Buffers for outputs are allocated lazily and reused as long as the
        resolved output shape does not change between calls.
        """

        def __init__(self, engine_path, input_names, output_names, device="cuda"):
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)

            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

            self.context = self.engine.create_execution_context()
            self.input_names = list(input_names)
            self.output_names = list(output_names)
            self.device = torch.device(device)
            self._output_buffers: dict[str, torch.Tensor] = {}

        @staticmethod
        def _torch_dtype(trt_dtype):
            try:
                return _TRT_TO_TORCH[trt_dtype]
            except KeyError as e:
                raise TypeError(f"Unsupported TRT dtype: {trt_dtype}") from e

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
            if buf is None or tuple(buf.shape) != shape:
                dtype = self._torch_dtype(self.engine.get_tensor_dtype(name))
                buf = torch.empty(shape, dtype=dtype, device=self.device)
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


class OwlV2TRT(OwlV2):
    """OWLv2 variant that runs the vision tower with TensorRT when available."""

    def __init__(
        self,
        engine_path: str | Path | None = None,
        *,
        onnx_path: str | Path | None = None,
        output_dir: str | Path = ".",
        model_type: str = "base",
        build_missing: bool = True,
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 8,
        workspace_gb: int = 8,
        fp16: bool = True,
    ):
        self.output_dir = Path(output_dir)
        artifact_stem = f"owlv2_vis_{model_type}"
        self.engine_path = self._artifact_path(
            engine_path, self.output_dir / f"{artifact_stem}.engine"
        )
        self.onnx_path = self._artifact_path(
            onnx_path,
            self.engine_path.with_suffix(".onnx")
            if engine_path is not None
            else self.output_dir / f"{artifact_stem}.onnx",
        )
        self.build_missing = build_missing
        self.engine_build_kwargs = {
            "min_batch": min_batch,
            "opt_batch": opt_batch,
            "max_batch": max_batch,
            "workspace_gb": workspace_gb,
            "fp16": fp16,
        }
        self.trt = None
        super().__init__(model_type=model_type)

    @staticmethod
    def _artifact_path(path: str | Path | None, default: Path) -> Path:
        return Path(path) if path is not None else default

    def _ensure_artifacts(self):
        if self.engine_path.exists():
            return

        self.onnx_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.onnx_path.exists():
            from OWLv2torch.torch_version.export import export_vision_tower

            print(f"Exporting OWLv2 vision ONNX to {self.onnx_path}")
            export_vision_tower(self.vision_model, str(self.onnx_path), self.image_size)

        if not TRT_AVAILABLE:
            print("TensorRT is not installed; ONNX was exported but engine build was skipped.")
            return
        if not torch.cuda.is_available():
            print("CUDA is not available; ONNX was exported but engine build was skipped.")
            return

        from OWLv2torch.torch_version.build_engine import build_engine

        print(f"Building OWLv2 TensorRT engine to {self.engine_path}")
        build_engine(
            onnx_file_path=str(self.onnx_path),
            engine_file_path=str(self.engine_path),
            image_size=self.image_size,
            **self.engine_build_kwargs,
        )

    def _load_model(self, model_path):
        super()._load_model(model_path)

        if self.build_missing:
            self._ensure_artifacts()

        engine_ready = (
            TRT_AVAILABLE
            and torch.cuda.is_available()
            and self.engine_path.exists()
        )
        if engine_ready:
            self.trt = TRTRunner(
                self.engine_path,
                input_names=["image"],
                output_names=["cls_emb", "full_output"],
            )
        else:
            self.trt = None

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

    def get_vision_features(self, pixel_values, normalize=True):
        if self.trt is not None:
            vision_pooled, vision_full = self.trt_inference(pixel_values)
        else:
            vision_pooled, vision_full = self.vision_model(pixel_values)
        vision_features = self.visual_projection(vision_pooled)
        if normalize:
            vision_features = vision_features / (
                torch.linalg.norm(vision_features, dim=-1, keepdim=True) + 1e-6
            )
        return vision_features, vision_pooled, vision_full


def _parse_args():
    p = argparse.ArgumentParser(description="Run OWLv2 with an optional TensorRT vision tower.")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--onnx", default=None)
    p.add_argument("--engine", default=None)
    p.add_argument("--model-type", default="base", choices=["base", "large"])
    p.add_argument("--image", default="img.jpg")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--workspace-gb", type=int, default=8)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--no-build", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model = OwlV2TRT(
        engine_path=args.engine,
        onnx_path=args.onnx,
        output_dir=args.output_dir,
        model_type=args.model_type,
        build_missing=not args.no_build,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
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
