import torch
import torch.onnx
import onnx

from owlv2 import OwlV2, VisionTower
from safetensors import safe_open
from onnxsim import simplify


state_dict = {}
with safe_open("weights/model.safetensors", framework="pt") as f:
    for k in f.keys():
        tens = f.get_tensor(k)
        k = k.replace("owlv2.","").replace(".embeddings","")
        k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
        state_dict[k] = tens
model = OwlV2()
model.eval()
model.load_state_dict(state_dict)
vision_model = model.vision_model


dummy_input = torch.rand(1,3,960,960)
torch.onnx.export(vision_model,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  "owlv2_vis.onnx",   # where to save the model
                  export_params=True,        # store the trained parameter weights inside the model file
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


onnx_model = onnx.load("owlv2_vis.onnx")
model_simp, check = simplify(onnx_model)
onnx.save_model(model_simp, "owlv2_vis.onnx")