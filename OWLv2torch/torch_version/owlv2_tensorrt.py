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

    def __init__(self, engine_path):
        super().__init__()
        self.engine_path = engine_path
        self.trt = None

    def load_model(self, model_path):
        super().load_model(model_path)

        engine_ready = (
            TRT_AVAILABLE
            and torch.cuda.is_available()
            and Path(self.engine_path).exists()
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


if __name__ == "__main__":
    engine_path = "owlv2_vis.engine"
    img = Image.open("img.jpg")
    img_array = np.array(img.resize((960, 960)))
    inputs = img_array.transpose(2, 0, 1)  # (3, 960, 960)

    model = OwlV2TRT(engine_path)
    model.load_model("weights/model.safetensors")
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
