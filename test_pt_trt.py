from OWLv2torch.torch_version.owlv2_tensorrt import OwlV2TRT
import torch
import torch.nn as nn
import utils
from PIL import Image
import numpy as np
from transformers import CLIPTokenizer
from OWLv2torch.hf_version.processing_owlv2 import Owlv2Processor
from EzLogger import Timer
from safetensors import safe_open
timer = Timer()

def test_pt():
    model = OwlV2TRT("owlv2_vis.engine")
    model.load_model("weights/model.safetensors")


    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model.cuda()
    #print(model)

    image = Image.open('img.jpg')
    image_inputs = model.preprocess_image(image)
    text_inputs = tokenizer(["a cat", "a scale", "a plastic bag"], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        image_inputs = image_inputs.cuda()
        text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        for i in range(5):
            with timer("inf"):
                outputs = model.forward_object_detection(image_inputs, text_inputs["input_ids"], text_inputs["attention_mask"]) 
    
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