from torch_func.owlv2 import text_obj_det, process_sequences
from torch_func.owlv2_weights import load_owlv2_weights
from torch_func.owlv2_config import OWLV2_B16, get_transform
import torch
import torch.nn as nn
import utils
from PIL import Image
import numpy as np
from transformers import CLIPTokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from hf_version.processing_owlv2 import Owlv2Processor

from EzLogger import Timer

timer = Timer()

def test_pt():
    from safetensors import safe_open
    state_dict = {}
    with safe_open("weights/model.safetensors", framework="pt") as f:
        for k in f.keys():
            tens = f.get_tensor(k)
            k = k.replace("owlv2.","").replace(".embeddings","")
            k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
            state_dict[k] = tens.to("cuda")

    weights = load_owlv2_weights(state_dict)
    image_transform = get_transform(OWLV2_B16)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
    

    image = Image.open('img.jpg')
    image_inputs = image_transform(image).unsqueeze(0)
    text_inputs = tokenizer(["a cat", "a scale", "a plastic bag"], return_tensors="pt", padding=True, truncation=True, )

    with torch.no_grad(), timer("model_run"):
        image_inputs = image_inputs.cuda()
        text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        padded_ids, _attn_maks = process_sequences(text_inputs["input_ids"].tolist())
        padded_ids, _attn_maks = padded_ids.cuda(), _attn_maks.cuda()
        outputs = text_obj_det(padded_ids, _attn_maks, image_inputs, weights, OWLV2_B16) 
    
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    outputs = {"logits": outputs[0], "pred_boxes": outputs[2]}
    outputs = dotdict(outputs)
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    draw_img = utils.load_image('img.jpg')
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected with confidence {round(score.item(), 3)} at location {box}")
        draw_img = utils.draw_bbox(draw_img, box)
    utils.show_image(draw_img)


if __name__ == '__main__':
    test_pt()
    timer.print_metrics()
