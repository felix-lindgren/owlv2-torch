from typing import NamedTuple
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import numpy as np
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

params = {
  "text_dim": 512,
  "vision_dim": 768,
  "n_layers": 12,
  "n_vision_heads": 12,
  "n_text_heads": 8,
  "vocab_size": 49408,
  "ffn_dim_multiplier": 4.0,
  "max_seq_len": 16,
  "image_size": 960,
  "patch_size": 16,
}

class EncoderParams(NamedTuple):
  n_heads: int
  dim: int
  head_dim: int
  n_layers: int
class ModelParams(NamedTuple):
  vision_encoder: EncoderParams
  text_encoder: EncoderParams
  max_seq_len: int
  vocab_size: int
  image_size: int
  patch_size: int
  num_patches_sqrt: int
  num_vision_pos: int
  


OWLV2_B16 = ModelParams(
  vision_encoder=EncoderParams(params["n_vision_heads"], params["vision_dim"], params["vision_dim"] // params["n_vision_heads"], params["n_layers"]),
  text_encoder=EncoderParams(params["n_text_heads"], params["text_dim"], params["text_dim"] // params["n_text_heads"], params["n_layers"]),
  max_seq_len=params["max_seq_len"],
  vocab_size=params["vocab_size"],
  image_size=params["image_size"],
  patch_size=params["patch_size"],
  num_patches_sqrt=((params["image_size"] // params["patch_size"])),
  num_vision_pos=((params["image_size"] // params["patch_size"]) ** 2) + 1,
  
)

class SquarePad:
	def __call__(self, image):
		h, w = image.shape[-2:]
		max_wh = np.max([w, h])
		hp = int(max_wh - w)
		vp = int(max_wh - h)
		padding = (0, 0, hp, vp)
		return TF.pad(image, padding, 0.5, 'constant')

def get_transform(model_params):
  return T.Compose([
      T.ToImage(),
      T.ToDtype(torch.float32,scale=True),
      SquarePad(),
      T.Resize((model_params.image_size,model_params.image_size), antialias=True),
      T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
  ])