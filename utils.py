import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)
    return image

def show_image(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
        image = image.squeeze(0)
        image = image.transpose((1, 2, 0))
    elif isinstance(image, np.ndarray):
        image = image.squeeze(0)
        image = image.transpose((1, 2, 0))
    

    plt.imshow(image)
    plt.show()


def draw_bbox(image, bbox, color=(255, 0, 0)):
    image = image.squeeze(0)
    image = image.transpose(0, 1).transpose(1, 2)
    image = np.array(image)
    image = np.uint8(image * 255)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=color, width=5)
    image = np.array(image)
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)
    return image

if __name__ == '__main__':
    image = load_image('img.jpg')
    image = draw_bbox(image, [100, 100, 300, 300])
    show_image(image)