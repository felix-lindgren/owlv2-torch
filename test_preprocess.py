from OWLv2torch import OwlV2, tokenize
import torch
import torch.nn as nn
import utils
from PIL import Image
import numpy as np
from transformers import CLIPTokenizer
from OWLv2torch.hf_version.processing_owlv2 import Owlv2Processor
import matplotlib.pyplot as plt
def test_pt():
    model = OwlV2()
    model.load_model("google/owlv2-base-patch16-ensemble")
    model.eval()
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

    image = Image.open('img.jpg')
    image_inputs = model.preprocess_image(image)
    inputs = processor(images=[image], return_tensors="pt") 
    pixel_values = inputs['pixel_values']
    print(image_inputs.shape, pixel_values.shape)
    diff = torch.sub(image_inputs,pixel_values)
    print(diff.max(), diff.mean())
    print(diff.shape)
    #fig, axs = plt.subplots(1,2)
    #for ax in axs:
    plt.imshow(torch.abs(diff[0]).permute(1,2,0).cpu().numpy())
    plt.show()

    


if __name__ == '__main__':
    test_pt()
