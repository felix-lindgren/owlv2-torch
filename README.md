# OWLv2-Torch

A standalone PyTorch implementation of [OWLv2](https://arxiv.org/abs/2306.09683) (Open-World Localization v2) for open-vocabulary object detection, with optional TensorRT acceleration.

## Overview

This package provides a clean, dependency-light reimplementation of Google's OWLv2 model. It loads the original weights from HuggingFace Hub and runs inference without requiring the full `transformers` library at runtime.

Key features:

- **Pure PyTorch inference** with weights loaded directly from safetensors
- **TensorRT support** for accelerated inference
- **Visual prototype detection** -- train lightweight prototype embeddings on custom classes while keeping the backbone frozen
- **Base and Large model variants** (`owlv2-base-patch16-ensemble`, `owlv2-large-patch14-ensemble`)

## Installation

Requires Python 3.10+.

```bash
pip install .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Text-conditioned detection

```python
from OWLv2torch import OwlV2, tokenize
from PIL import Image
import torch

model = OwlV2("base")  # or "large"
model.eval()

image = Image.open("photo.jpg")
pixel_values = model.preprocess_image(image)

queries = ["a cat", "a dog", "a person"]
token_ids = tokenize(queries, context_length=16, truncate=True)
attention_mask = token_ids == 0

with torch.no_grad():
    pred_logits, objectness, pred_boxes, class_embeds, _ = model.forward_object_detection(
        pixel_values, token_ids, attention_mask
    )

# Post-process boxes to image coordinates
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
boxes = model.postprocess_boxes(pred_boxes, target_sizes)

probs = torch.max(pred_logits, dim=-1)
scores = torch.sigmoid(probs.values)
labels = probs.indices
```

### Prototype-based detection

Train lightweight visual prototype embeddings for custom object classes while keeping the full model frozen:

```python
from OWLv2torch.torch_version.prototypes import VisualPrototypeBank
from OWLv2torch.torch_version.owlv2 import OwlV2, PrototypeDetector

model = OwlV2("large")
bank = VisualPrototypeBank(num_classes=1, prototypes_per_class=4, dim=model.text_dim)
detector = PrototypeDetector(model, bank).to("cuda")

# Only the prototype embeddings are trainable
optimizer = torch.optim.AdamW(detector.trainable_parameters(), lr=1e-4)

# In your training loop:
outputs = detector(pixel_values)
# Use compute_losses() from OWLv2torch.torch_version.loss
```

## Project Structure

```
OWLv2torch/
  torch_version/      # Main PyTorch implementation
    owlv2.py          #   Model architecture and inference
    prototypes.py      #   Visual prototype bank for few-shot detection
    loss.py            #   Training losses (focal, NWD, Hungarian matching)
    owlv2_tensorrt.py  #   TensorRT wrapper
    build_engine.py    #   TensorRT engine builder
    export.py          #   ONNX/TensorRT export
  torch_func/         # Alternative functional implementation
  hf_version/         # HuggingFace-compatible reference implementation
  utils/
    tokenizer.py       # CLIP BPE tokenizer
    hf_hub_utils.py    # Weight cache lookup
train.py              # Training script with COCO dataset and MLflow logging
utils.py              # Image loading and visualization helpers
```

## Model Variants

| Variant | `model_type` | Image Size | Vision Dim | Text Dim | Params |
|---------|-------------|------------|------------|----------|--------|
| Base    | `"base"`    | 960        | 768        | 512      | ViT-B/16 |
| Large   | `"large"`   | 1008       | 1024       | 768      | ViT-L/14 |

Weights are automatically downloaded from HuggingFace Hub on first use and cached locally.
