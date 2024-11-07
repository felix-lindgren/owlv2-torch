import numpy as np
import tensorrt as trt
from torch2trt import TRTModule
import torch
from PIL import Image
from pathlib import Path
from safetensors import safe_open
from OWLv2torch.torch_version.owlv2 import OwlV2
from EzLogger import Timer
timer = Timer()

class OwlV2TRT(OwlV2):

    def __init__(self, engine_path):
        super().__init__()
        self.engine_path = engine_path
        

    def load_model(self, model_path):
        super().load_model(model_path)

        if Path(self.engine_path).exists():
            self.trt = TRTModule(self.engine_path, ["image"], output_names=["cls_emb", "full_output"])
        else:
            self.trt = None

    @timer("trt_infrence")
    def trt_inference(self, pixel_data: torch.Tensor):
        device = pixel_data.device
        output_data = self.trt(pixel_data)
        pooled_output, full_vision = output_data
        pooled_output = pooled_output.reshape((1,-1))
        full_vision = full_vision.reshape((1,self.vision_model.num_positions,self.vision_dim))
        
        return pooled_output, full_vision
    
    def get_vision_features(self, pixel_values):
        if not self.trt is None:
            vision_pooled, vision_full = self.trt_inference(pixel_values)
        else:
            vision_pooled, vision_full = self.vision_model(pixel_values)
        vision_features = self.visual_projection(vision_pooled)
        vision_features = vision_features / (torch.linalg.norm(vision_features, dim=-1, keepdim=True) + 1e-6)
        return vision_features, vision_pooled, vision_full


if __name__ == "__main__":
    
    engine_path = "owlv2_vis.engine"
    trt_module = TRTModule("owlv2_vis.engine", ["image"], output_names=["cls_emb", "full_output"])
    img = Image.open("img.jpg")
    img_array = np.array(img.resize((960,960)))
    inputs = img_array.transpose(2, 0, 1)  # (3, 224, 224)
    
    model = OwlV2TRT(engine_path)
    model.load_model("weights/model.safetensors")
    model.eval(),model.cuda()
    print(inputs.shape, inputs.dtype)
    im_pt = torch.from_numpy(inputs).unsqueeze(0).cuda()
    res = model.get_vision_features(im_pt)
    #print(res[0].shape, res[1].shape, res[2].shape)

    
    _,output_data = trt_module(im_pt)
    print(output_data.shape)
    print(output_data.reshape((1,3601,768)).shape)

    #timer.print_metrics()
    quit()
    # Run inference
    _,output_data = trt_inference.infer(inputs)
    print(output_data.shape)
    print(output_data.reshape((1,3601,768)).shape)
    for i in range(10):
        with timer("infer"):
            trt_inference.infer(inputs)
