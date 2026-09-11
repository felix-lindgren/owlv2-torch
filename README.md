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

Install TensorRT support when you want to build and run the accelerated vision
tower:

```bash
pip install -e ".[trt]"
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

For a fixed query set used across many image batches, encode the text once and
reuse the projected embeddings:

```python
with torch.inference_mode():
    query_embeds, query_mask = model.encode_detection_queries(
        token_ids, attention_mask
    )

    for pixel_batch in image_batches:
        outputs = model.forward_object_detection_from_embeddings(
            pixel_batch, query_embeds, query_mask
        )
```

Two-dimensional token tensors are treated as queries shared by every image.
Use `[batch_size, num_queries, sequence_length]` token tensors when each image
has a different query set.

### TensorRT vision tower

```python
from OWLv2torch import OwlV2TRT

model = OwlV2TRT(output_dir=".")
```

When no paths are supplied, the TensorRT variant writes
`owlv2_vis_base.onnx` and `owlv2_vis_base.engine` in the current directory if
the engine is missing. Use `output_dir="artifacts"` to place both files in a
different folder, or pass `onnx_path` / `engine_path` for exact filenames.

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

### Fashionpedia

The Fashionpedia scripts load
[`detection-datasets/fashionpedia`](https://huggingface.co/datasets/detection-datasets/fashionpedia)
directly through Hugging Face Datasets. Fashionpedia has no public test split,
so evaluation uses `val`. Its Pascal VOC boxes are converted to COCO boxes for
the repository's mAP evaluator.

Run the zero-shot baseline:

```bash
uv run --extra train python tools/test_fashionpedia.py \
  --model-size base
```

Train FLAME refiners from Fashionpedia `train` support images and compare them
with the zero-shot baseline on `val`:

```bash
uv run --extra train python tools/test_fashionpedia_flame.py \
  --model-size base \
  --classes "shirt, blouse" "shoe" \
  --shots 30 \
  --support-images 8
```

Omit `--classes` to run all 46 categories. Use `--limit` for a validation-set
smoke test and `--cache-dir` to control where the 3.48 GB dataset is cached.

#### Text-conditioned fine-tuning

To adapt OWLv2 to fashion while retaining arbitrary text queries, train through
the normal text-conditioned detection path rather than a prototype bank. The
trainer expects the COCO files produced by `tools/convert_fashionpedia_to_coco.py`:

```bash
uv run --extra train python prototype_train/train_text.py \
  --train-annotations /mnt/datasets/fashion/fashionpedia_coco/train/annotations.json \
  --train-images /mnt/datasets/fashion/fashionpedia_coco/train/images \
  --val-annotations /mnt/datasets/fashion/fashionpedia_coco/val/annotations.json \
  --val-images /mnt/datasets/fashion/fashionpedia_coco/val/images \
  --model-type base \
  --batch-size 8 \
  --epochs 20 \
  --run-zero-shot-baseline
```

By default, the text and vision towers remain frozen while the class, box and
objectness heads are trained. Add `--vision-blocks 2` if head-only adaptation
plateaus. `--text-blocks 1` is available for specialist vocabulary, but should
generally be used only after trying the frozen text tower. Comma-separated
Fashionpedia class names are automatically expanded into prompt aliases and a
different prompt variant is sampled for every class on each training step.

LV-MHP-v1's per-person parsing masks can be converted to boxes for the same
trainer. The converter preserves all 18 foreground categories and each person's
instances, and creates a seeded 90/10 split from `train_list.txt` by default:

```bash
uv run python tools/convert_lv_mhp_coco.py \
  --dataset-root /mnt/datasets/fashion/LV-MHP-v1 \
  --output-dir /mnt/datasets/fashion/lv_mhp_coco
```

The output has the same `train/images`, `train/annotations.json`, `val/images`,
and `val/annotations.json` layout shown above. Images are symlinked by default;
use `--image-mode copy` for a self-contained output or `--image-mode none` to
keep using `/mnt/datasets/fashion/LV-MHP-v1/images` directly. Add
`--include-test` to convert the official test list separately, or
`--val-source test` to train on all of `train_list.txt` and validate on that
test list.

LV-MHP labels six of its classes as lateral pairs (`left arm`/`right arm`,
`left shoe`/`right shoe`, `left leg`/`right leg`), which a text query cannot
distinguish. `--merge-class-names` collapses them into one query each:

```bash
uv run --extra train python prototype_train/train_text.py \
  ...dataset arguments... \
  --merge-class-names left_arm+right_arm=arm left_shoe+right_shoe=shoe \
                      left_leg+right_leg=leg
```

Sources are matched against category names after the same cleaning applied to
prompts, and `+` separates them because a comma already separates aliases inside
one category name. Every box stays supervised under the merged label, which is
why this is not the same as `--exclude-class-names left_arm`: dropping one side
of a pair would leave visually identical objects labelled on one side and
unlabelled on the other. Merging changes the class set, so the resulting mAP is
not comparable to a run with a different one -- re-score the old weights with
`--eval-only --init-from` on the new set before reading a difference as
learning.

For short configuration experiments, stop and evaluate by optimizer step rather
than waiting for a complete epoch:

```bash
uv run --extra train python prototype_train/train_text.py \
  ...dataset arguments... \
  --max-steps 200 \
  --eval-every-steps 50 \
  --eval-max-batches 20
```

When `--max-steps` is set, the cosine schedule also advances per optimizer step.
`--eval-every-steps` replaces epoch-based evaluation, while
`--eval-max-batches` makes the reported validation mAP an approximate metric
over the first N validation batches. A final evaluation is always run unless
the last requested step was already evaluated.

The trainer writes compact delta checkpoints to `text_checkpoints/best.pth` and
`text_checkpoints/final.pth`. Load one on top of the same pretrained model:

```python
from OWLv2torch import OwlV2
from prototype_train.train_text import load_text_checkpoint

model = OwlV2("base")
metadata = load_text_checkpoint(model, "text_checkpoints/best.pth")
```

Pass `--save-full-model` if a standalone full state dictionary is preferable.

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
