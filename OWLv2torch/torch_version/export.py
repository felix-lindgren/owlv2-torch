"""Export the OWLv2 vision tower to ONNX with a dynamic batch axis.

The dynamic axis is required for ``build_engine.py`` to produce an engine that
accepts variable batch sizes through a TensorRT optimization profile.
"""

import torch
import torch.onnx
import onnx
from onnxsim import simplify
from safetensors import safe_open

from owlv2 import OwlV2


def load_weights(path: str) -> dict:
    state_dict = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            tens = f.get_tensor(k)
            k = k.replace("owlv2.", "").replace(".embeddings", "")
            k = k.replace("mlp.fc1", "mlp.0").replace("mlp.fc2", "mlp.2")
            state_dict[k] = tens
    return state_dict


def main(
    weights_path: str = "weights/model.safetensors",
    onnx_path: str = "owlv2_vis.onnx",
    image_size: int = 960,
):
    model = OwlV2()
    model.eval()
    model.load_state_dict(load_weights(weights_path))
    vision_model = model.vision_model

    dummy_input = torch.rand(1, 3, image_size, image_size)
    torch.onnx.export(
        vision_model,
        dummy_input,
        onnx_path,
        export_params=True,
        input_names=["image"],
        output_names=["cls_emb", "full_output"],
        dynamic_axes={
            "image": {0: "batch"},
            "cls_emb": {0: "batch"},
            "full_output": {0: "batch"},
        },
        opset_version=17,
    )

    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    if not check:
        print("WARNING: onnx-simplifier validation failed; saving unsimplified model.")
        onnx.save_model(onnx_model, onnx_path)
    else:
        onnx.save_model(model_simp, onnx_path)


if __name__ == "__main__":
    main()
