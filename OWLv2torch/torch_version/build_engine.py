"""Build a TensorRT engine from the ONNX vision tower.

Changes vs. the original script:
  * Workspace pool bumped to 8 GB (ceiling, not reservation; larger lets the
    builder consider more tactics).
  * A real dynamic optimization profile (min/opt/max batch) is configured so
    the engine can serve variable batch sizes at runtime. The ONNX model must
    have been exported with ``dynamic_axes={"image": {0: "batch"}}`` for this
    to have any effect — see ``export.py``.
"""

import argparse
import tensorrt as trt


# Sensible defaults — override via CLI if needed.
DEFAULT_MIN_BATCH = 1
DEFAULT_OPT_BATCH = 1
DEFAULT_MAX_BATCH = 8
DEFAULT_IMAGE_SIZE = 960
DEFAULT_WORKSPACE_GB = 8


def build_engine(
    onnx_file_path: str,
    engine_file_path: str,
    min_batch: int = DEFAULT_MIN_BATCH,
    opt_batch: int = DEFAULT_OPT_BATCH,
    max_batch: int = DEFAULT_MAX_BATCH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    workspace_gb: int = DEFAULT_WORKSPACE_GB,
    fp16: bool = True,
):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # Workspace is a ceiling, not a reservation; larger -> more tactics considered.
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape  # typically (-1 or 1, 3, H, W)
    channels = input_shape[1] if input_shape[1] > 0 else 3

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        (min_batch, channels, image_size, image_size),
        (opt_batch, channels, image_size, image_size),
        (max_batch, channels, image_size, image_size),
    )
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("Failed to create engine")
        return None

    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)

    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(engine_bytes)


def _parse_args():
    p = argparse.ArgumentParser(description="Build an OWLv2 vision TRT engine.")
    p.add_argument("--onnx", default="owlv2_vis.onnx")
    p.add_argument("--engine", default="owlv2_vis.engine")
    p.add_argument("--min-batch", type=int, default=DEFAULT_MIN_BATCH)
    p.add_argument("--opt-batch", type=int, default=DEFAULT_OPT_BATCH)
    p.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--workspace-gb", type=int, default=DEFAULT_WORKSPACE_GB)
    p.add_argument("--no-fp16", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = build_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.engine,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        image_size=args.image_size,
        workspace_gb=args.workspace_gb,
        fp16=not args.no_fp16,
    )
    if engine:
        print(f"TensorRT engine has been created and saved to {args.engine}")
    else:
        print("Failed to create TensorRT engine.")
