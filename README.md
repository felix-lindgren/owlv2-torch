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

### Faster and more accurate zero-shot inference

The default preprocessing is HuggingFace's. `set_preprocessing` changes
`preprocess_image` and `postprocess_boxes` (training transforms are
untouched), so the code above runs as is:

```python
model.set_preprocessing(blur=0.5)                 # +0.011 COCO val mAP, same speed
model.set_preprocessing(blur=0.5, pad_drop=True)  # also drop the padding: 1.4x on COCO's aspect mix
model.set_preprocessing(blur=0.5, long_side=768)  # 1.6x faster again, -0.0065 COCO, nothing on DIOR

pixel_values = model.preprocess_image([image_a, image_b])  # cropped to the batch's largest grid
...
boxes = model.postprocess_boxes(pred_boxes, target_sizes)
scores, labels, token = model.select_detections(pred_logits, objectness, top_k=300, multi_label=True)
boxes = boxes.gather(1, token[..., None].expand(-1, -1, 4))
```

- `blur` softens the resized image (sigma in pixels); OWLv2-base does better
  on slightly soft input. It helps natural images with base and is neutral on
  aerial. With large it is +0.002 on COCO and hurts Fashionpedia, so leave it
  off there.
- `long_side` shrinks the image inside the original canvas and drops the
  padding, keeping the pretrained position embeddings (unlike
  `set_image_size`, which resamples them and breaks zero-shot accuracy).
- `multi_label=True` lets a token report several classes: +0.004 COCO and
  +0.03 Fashionpedia mAP, −0.002 on DIOR, so leave it off for aerial images.

Cropped (non-square) inputs run on the torch tower or a padding-free engine
(`PaddingFreeDetectorTRT`, see `tools/arch_speedup/export_pruned.py`); the
fixed-size `OwlV2TRT` engine takes `blur` only. Batch images of similar aspect
ratio together. Measurements are in `docs/arch-speedup-findings.md`.

### TensorRT vision tower

```python
from OWLv2torch import OwlV2TRT

model = OwlV2TRT(output_dir=".")
```

When no paths are supplied, the TensorRT variant writes
`owlv2_vis_base.onnx` / `owlv2_vis_base.engine` (vision tower) and
`owlv2_heads_base.onnx` / `owlv2_heads_base.engine` (class, box and objectness
heads) in the current directory if the engines are missing. Use
`output_dir="artifacts"` to place the files in a different folder, or pass
`onnx_path` / `engine_path` / `heads_engine_path` for exact filenames.

The heads engine serves shared `[num_queries, text_dim]` queries (up to
`max_queries=256`) under `torch.no_grad()` / `inference_mode()`; other calls use
the torch heads. It bakes in the head weights, so after loading fine-tuned heads
call `model.build_trt_heads()` (a few seconds); until then detection falls back
to the torch heads with a warning. Pass `trt_heads=False` to skip it.

TensorRT 11 takes layer precision from the ONNX graph, so fp16 is chosen at
export time (`fp16=True`, the default). Engines and fp32 ONNX files made with
TensorRT 10 must be deleted and rebuilt.

### Few-shot refinement (FLAME)

`Flame` implements [FLAME](https://arxiv.org/abs/2510.17670): zero-shot OWLv2
scores every patch, and a small RBF-SVM per query, trained on a few labelled
proposals, corrects that score. The model itself is not trained: fitting 20
DIOR classes from 100 example images took 21 s on an RTX 3060, most of it the
forward passes.

```python
from OWLv2torch import Flame, OwlV2

model = OwlV2("base").eval().to("cuda")
flame = Flame(model, ["a satellite photo of a chimney", "a satellite photo of a dam"])

# Give it a few images with every instance boxed (labels index the queries)...
flame.add_example("a.jpg", boxes=[[x1, y1, x2, y2], [x1, y1, x2, y2]], labels=[0, 1])
flame.add_example("b.jpg", boxes=[])  # an image containing none of them

# ...or let it pick proposals to ask about and answer yes/no (the paper's active step).
candidates = flame.suggest(unlabelled_images, k=30)  # k per query
flame.add_labels(candidates, answers)                 # one bool per candidate

flame.fit()
boxes, scores, labels = flame.detect("test.jpg")      # threshold=0.01 by default
flame.save("flame.pt")                                # Flame.load("flame.pt", model)
```

Boxes must cover every instance of every query in that image, because anything
left unboxed is used as a negative. A query that ends up with no positives stays
zero-shot. Refined scores are low in absolute terms: F1 peaked at thresholds of
0.005-0.02 on both datasets below.

`tools/test_dior_flame.py` and `tools/test_fashionpedia_flame.py` sample 5 train
images per class, pool them (100 images for DIOR's 20 classes, 230 for
Fashionpedia's 46), and evaluate on the whole DIOR test split (3463 images) or
Fashionpedia val (1158 images):

| | AP@[.5:.95] | AP50 |
|---|---|---|
| DIOR base, zero-shot | 0.069 | 0.122 |
| DIOR base, FLAME with boxes | 0.303 | 0.490 |
| DIOR base, FLAME active (30 yes/no answers per class) | 0.189 | 0.303 |
| DIOR large, zero-shot | 0.100 | 0.167 |
| DIOR large, FLAME with boxes | 0.333 | 0.535 |
| Fashionpedia base, zero-shot | 0.216 | 0.289 |
| Fashionpedia base, FLAME with boxes | 0.294 | 0.409 |

With one class at a time on 3000 DIOR test images (`base`, 3 seeds), a single
example image already doubles zero-shot AP (0.067 to 0.150), 3 images reach
0.215 and 10 reach 0.262. From 3 images on, the only class that ever scored
below zero-shot was bridge (0.007 zero-shot), and by less than 0.01.

Where this departs from the paper, it is because the paper's version measured
worse on DIOR (`base`, 5 images per class, 3 seeds):

- The SVM margin is added to the zero-shot logit (`alpha=8`) instead of
  replacing the score. With the same labels, the SVM score alone reached
  0.064 AP and the sum 0.218.
- `suggest` keeps the `4k` best zero-shot proposals and takes one per k-means
  cluster. The paper's PCA + KDE density band picked about 2 positives out of 30
  and did no better than k-means alone.
- No SMOTE: class-balanced SVM weights already handle the imbalance, and SMOTE
  did not help.

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

Fit FLAME on a few Fashionpedia `train` images per class and compare it with
the zero-shot baseline on `val` (add `--mode active --shots 30` for the yes/no
variant):

```bash
uv run --extra train python tools/test_fashionpedia_flame.py \
  --model-size base \
  --classes "shirt, blouse" "shoe" \
  --support-images 5
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
    flame.py           #   FLAME few-shot refinement (SVM on OWLv2 proposals)
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
