"""Build a TensorRT engine from an exported OWLv2 ONNX graph.

Targets TensorRT 11, where networks are always strongly typed: layer
precision comes from the ONNX graph (see ``export.py --no-fp16``), not from a
builder flag. Works for both the vision tower and the detection heads:

  * every input's dim 0 is the batch axis and gets a min/opt/max batch range;
  * ``query_embeds`` (heads only) instead gets a min/opt/max query-count range.

The ONNX must have been exported with those axes dynamic — see ``export.py``.
"""

import argparse
import tensorrt as trt


# Sensible defaults — override via CLI if needed.
DEFAULT_MIN_BATCH = 1
DEFAULT_OPT_BATCH = 1
DEFAULT_MAX_BATCH = 8
DEFAULT_MIN_QUERIES = 1
DEFAULT_OPT_QUERIES = 16
DEFAULT_MAX_QUERIES = 256
DEFAULT_WORKSPACE_GB = 8

QUERY_INPUT = "query_embeds"


def build_engine(
    onnx_file_path: str,
    engine_file_path: str,
    min_batch: int = DEFAULT_MIN_BATCH,
    opt_batch: int = DEFAULT_OPT_BATCH,
    max_batch: int = DEFAULT_MAX_BATCH,
    min_queries: int = DEFAULT_MIN_QUERIES,
    opt_queries: int = DEFAULT_OPT_QUERIES,
    max_queries: int = DEFAULT_MAX_QUERIES,
    workspace_gb: int = DEFAULT_WORKSPACE_GB,
):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # Workspace is a ceiling, not a reservation; larger -> more tactics considered.
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    # parse_from_file resolves external weight files next to the ONNX.
    if not parser.parse_from_file(str(onnx_file_path)):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        rest = tuple(tensor.shape)[1:]
        if tensor.name == QUERY_INPUT:
            lo, opt, hi = min_queries, opt_queries, max_queries
        else:
            lo, opt, hi = min_batch, opt_batch, max_batch
        profile.set_shape(tensor.name, (lo, *rest), (opt, *rest), (hi, *rest))
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
    p = argparse.ArgumentParser(description="Build an OWLv2 TRT engine (vision tower or heads).")
    p.add_argument("--onnx", default="owlv2_vis_base.onnx")
    p.add_argument("--engine", default="owlv2_vis_base.engine")
    p.add_argument("--min-batch", type=int, default=DEFAULT_MIN_BATCH)
    p.add_argument("--opt-batch", type=int, default=DEFAULT_OPT_BATCH)
    p.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    p.add_argument("--min-queries", type=int, default=DEFAULT_MIN_QUERIES)
    p.add_argument("--opt-queries", type=int, default=DEFAULT_OPT_QUERIES)
    p.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    p.add_argument("--workspace-gb", type=int, default=DEFAULT_WORKSPACE_GB)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = build_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.engine,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        min_queries=args.min_queries,
        opt_queries=args.opt_queries,
        max_queries=args.max_queries,
        workspace_gb=args.workspace_gb,
    )
    if engine:
        print(f"TensorRT engine has been created and saved to {args.engine}")
    else:
        print("Failed to create TensorRT engine.")
