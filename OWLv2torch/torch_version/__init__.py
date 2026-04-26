from .owlv2 import OwlV2

try:
    from .flame import FlamePipeline, FlameConfig
except ImportError:  # pragma: no cover - optional sklearn dependency
    FlamePipeline = None
    FlameConfig = None

# TensorRT is an optional runtime dependency. Fall back to ``None`` so that
# environments without ``tensorrt`` installed can still import the package.
try:
    from .owlv2_tensorrt import OwlV2TRT
except ImportError:  # pragma: no cover - optional dep
    OwlV2TRT = None
