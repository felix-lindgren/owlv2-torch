import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set memory pool limit
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Optional: Enable FP16 mode if desired
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Setup fixed input shape
    profile = builder.create_optimization_profile()
    
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    
    # Set fixed shape profile for 960x960
    fixed_shape = (1, input_shape[1], 960, 960)
    
    profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)
    config.add_optimization_profile(profile)
    
    # Build engine
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("Failed to create engine")
        return None

    # Save engine
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    
    # Deserialize the engine_bytes to create a runtime engine
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    
    return engine

# Usage
onnx_file_path = "owlv2_vis.onnx"
engine_file_path = "owlv2_vis.engine"

engine = build_engine(onnx_file_path, engine_file_path)
if engine:
    print(f"TensorRT engine has been created and saved to {engine_file_path}")
else:
    print("Failed to create TensorRT engine.")