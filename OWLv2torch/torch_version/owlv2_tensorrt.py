import numpy as np
import pycuda.driver as cuda
#import pycuda.autoinit
import tensorrt as trt
import time
import torch
from PIL import Image
from owlv2 import OwlV2
from EzLogger import Timer
timer = Timer()

class TensorRTInference:
    def __init__(self, engine_path, cuda_device=0):
        #cuda.init()
        #device = cuda.Device(0)
        #self.cuda_driver_context = device.make_context()
        #self.cuda_driver_context.push()

        self.device = cuda_device
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = list(engine.get_tensor_shape(tensor_name))
            dtype = self.torch_dtype(engine.get_tensor_dtype(tensor_name))
            
            # Allocate host and device buffers
            host_mem = torch.empty(shape, dtype=dtype, pin_memory=True)
            device_mem = torch.empty(shape, dtype=dtype, device=self.device)
            
            # Append the device buffer address to device bindings
            bindings.append(device_mem.data_ptr())
            
            # Append to the appropriate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings
    
    @staticmethod
    def torch_dtype(trt_dtype):
        """Convert TensorRT dtype to PyTorch dtype"""
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        # Add more dtype conversions as needed
        else:
            raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")

    def infer(self, input_data):
        # Transfer input data to device
        self.inputs[0].device.copy_(input_data)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        
        # Run inference
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
        # Transfer predictions back
        #for output in self.outputs:
        #    output.host.copy_(output.device)
        
        # Synchronize CUDA stream
        torch.cuda.current_stream().synchronize()
        
        return [o.device for o in self.outputs]

    """ def infer(self, input_data):
        # Transfer input data to device
        self.cuda_driver_context.push()
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        outputs = []
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(self.outputs[i].host, self.outputs[i].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()
        self.cuda_driver_context.pop()
        return [o.host for o in self.outputs] """
    
    """ def infer(self, input_data):
        # Transfer input data to device
        
        #np.copyto(self.inputs[0].host, input_data.ravel())
        #cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        print(int(self.inputs[0].device), input_data.data_ptr(), int(input_data.numel()*4), self.stream)
        cuda.memcpy_dtod_async(int(self.inputs[0].device), input_data.data_ptr(), int(input_data.numel()*4), self.stream)
        #print(self.inputs[0].device)

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        outputs = []
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(self.outputs[i].host, self.outputs[i].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        return [o.host for o in self.outputs] """


class OwlV2TRT(OwlV2):

    def __init__(self, engine_path):
        super().__init__()
        self.trt =  TensorRTInference(engine_path)

    @timer("trt_infrence")
    def trt_inference(self, pixel_data: torch.Tensor):
        device = pixel_data.device
        output_data = self.trt.infer(pixel_data)
        pooled_output, full_vision = output_data
        pooled_output = pooled_output.reshape((1,-1))
        full_vision = full_vision.reshape((1,self.vision_model.num_positions,self.vision_dim))
        
        return pooled_output, full_vision
    
    def get_vision_features(self, pixel_values):
        vision_pooled, vision_full = self.trt_inference(pixel_values)
        vision_features = self.visual_projection(vision_pooled)
        vision_features = vision_features / (torch.linalg.norm(vision_features, dim=-1, keepdim=True) + 1e-6)
        return vision_features, vision_pooled, vision_full


if __name__ == "__main__":
    
    engine_path = "owlv2_vis.engine"
    trt_inference = TensorRTInference(engine_path)
    img = Image.open("img.jpg")
    img_array = np.array(img.resize((960,960)))
    inputs = img_array.transpose(2, 0, 1)  # (3, 224, 224)
    
    model = OwlV2TRT(engine_path)
    model.eval(),model.cuda()
    print(inputs.shape, inputs.dtype)
    res = model.get_vision_features(torch.from_numpy(inputs).unsqueeze(0).cuda())
    print(res[0].shape, res[1].shape, res[2].shape)

    #timer.print_metrics()
    quit()
    # Run inference
    _,output_data = trt_inference.infer(inputs)
    print(output_data.shape)
    print(output_data.reshape((1,3601,768)).shape)
    for i in range(10):
        with timer("infer"):
            trt_inference.infer(inputs)
