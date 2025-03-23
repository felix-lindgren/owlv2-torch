import transformers
import torch
import numpy as np
import utils
from PIL import Image
from OWLv2torch.hf_version.modeling_owlv2 import Owlv2ForObjectDetection
from OWLv2torch.hf_version.processing_owlv2 import Owlv2Processor
from EzLogger import Timer
timer = Timer()


def test_hf():
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = model.eval()
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

    image = Image.open('img.jpg')
    inputs = processor(images=[image], text=["a cat", "a scale", "a plastic bag"], return_tensors="pt") 
    inputs = {k: v for k, v in inputs.items()}
    outputs = model(**inputs) # Warmup
    for i in range(5):
        with torch.no_grad(), timer("model_run"):
            inputs = {k: v for k, v in inputs.items()}
            outputs = model(**inputs)
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
    test_hf()
    timer.print_metrics()
