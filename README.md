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

Boxed examples train a shared, regularized box-correction head by default when
`fit()` runs. It uses `ridge=0.01`, `min_iou=0.3` and `strength=0.5`, retains at
most four visual rows per matched object, and refreshes after more boxed
examples are added. Use `Flame(model, queries, box_refinement=False)` for
original boxes. Yes/no-only support keeps the original boxes, and older
checkpoints retain their saved behavior.

FLAME uses the full-kernel Torch C-SVC fitter by default, on the model's device.
Its GPU fitting controls are:

```python
flame = Flame(model, queries, svm_backend="torch", svm_tol=1e-3,
              svm_cache_size_mb=256, svm_working_set_size=512)
```

This solves the ordinary balanced C-SVC dual with an unregularized intercept,
using all labelled examples and the same standardization and `gamma="scale"`
as sklearn. Working sets and a bounded kernel-row cache avoid requiring the
entire N-by-N kernel matrix. CUDA uses a fused Triton SMO loop when Triton is
available, with a slower Torch fallback; CPU fitting is also supported.
`svm_max_iter` limits outer working-set solves (default 300). Each head's
`flame.svms[c]["fit"]` records the global KKT gap and convergence status;
an unfinished fit emits a warning. Use `svm_backend="sklearn"` for the CPU
reference. Saved files preserve their backend; legacy files without a backend
field retain their original sklearn behavior.
See [full-kernel GPU SVM notes](docs/flame-gpu-svm.md) for implementation details
and the DIOR comparison command.

The earlier approximate RBF backends remain available:

```python
flame = Flame(model, queries, svm_backend="nystrom", n_components=512)
# Alternatively: svm_backend="rff" for random Fourier features.
```

Both use the same standardization, RBF gamma and balanced class penalties as the
reference sklearn SVC. They optimize a hinge loss smoothed over a width of 0.01;
`svm_max_iter` controls the Torch Newton iteration budget (default 300).
Nyström includes rare positive examples among its landmarks and folds the
fitted weights into the existing RBF scorer. `svm_seed` controls reproducibility.
Saved FLAME files retain the backend configuration; older files use sklearn.

`tools/benchmark_flame_features.py` compares these backends with sklearn on
identical cached COCO proposals, reporting detection AP, fit time and margin differences. Its
default protocol samples support outside COCO mini500 and evaluates on mini500:

```bash
uv run --extra train python tools/benchmark_flame_features.py \
    --device cuda --components 256 1024 --seeds 0 \
    --output data/flame_features/results.json
```

Experimental [APE feature selection](https://arxiv.org/abs/2304.01195) can reduce
the SVM input dimensions while retaining the original zero-shot logits:

```python
flame = Flame(model, queries, ape_channels=384, ape_source="text",
              ape_similarity_weight=0.9)
```

It requires at least two queries and is disabled by default. `ape_source="text_support"`
also uses positive support features for the inter-class similarity criterion;
`ape_normalize=True` normalizes selected features before SVM standardization.
This uses APE's channel criterion with FLAME's SVM, rather than APE's cache
classifier. On a held-out 500-image DIOR subset (three support draws), it reduced
mean mAP from 32.05 to 31.78 while cutting SVM fit time from 11.5 to 6.7 seconds.
See the [DIOR experiment](docs/flame-ape-findings.md) for the comparison and
benchmark commands.

GDA and Tip-Adapter proposal refiners can also be attached after fitting FLAME:

```python
flame.fit()
flame.fit_refiner("gda", weight=1.0)
# Alternatively: flame.fit_refiner("tip", weight=8.0, tip_beta=5.0)
```

The head uses the stored examples and is included in `detect()` and saved
checkpoints. To use a refiner alone, call `fit_refiner(..., svm_weight=0)`
directly, without first fitting SVMs. GDA uses regularized foreground/background
covariance; Tip-Adapter uses up to 128 positive and 128 negative keys per query
by default (`tip_negative=False` selects a positive-only cache).
On the same DIOR subset, mean mAP was 32.05 for FLAME, 32.14 with GDA and 32.01
with Tip-Adapter. These small differences varied across support draws; both
heads remain optional. See [refiner results and commands](docs/flame-refiners-findings.md).

The SVM can also use features before the class projection, or combine them with
the original embeddings:

```python
flame = Flame(model, queries, feature_source="combined", visual_weight=0.1)
# feature_source="visual" uses only pre-projection visual features.
```

Choose the source before adding support examples. The default is `"class"`.
Combined mode balances the two standardized feature blocks; `visual_weight`
controls their distance contributions rather than their raw dimension counts.
`visual_normalize=False` retains raw visual features. Both representations are
already computed in the same encoder pass, and the original detection heads
are preserved. Settings and the chosen support features are saved together.
APE channel selection currently requires the `"class"` source.

On a fresh matched 500-image DIOR cache (three support draws), the 10% visual
blend improved mean mAP from 32.20 to 32.33, with SVM fit time rising from 11.7
to 25.1 seconds. Visual-only features reached 31.15. The class source remains
the default.

The [feature comparison](docs/flame-visual-features-findings.md) records results
and commands. [Follow-up notes](docs/flame-next-steps.md) track the experiments
on mining negatives after fitting, balancing instances, and correcting boxes.

Support sampling and one extra hard-negative pass can be compared without
encoding images again:

```python
cached = [tuple(t.cpu() for t in p) for p in flame.proposals(support_images)]
support = list(zip(cached, support_boxes, support_labels))
flame.fit_support(support, mining_rounds=1)
# Optional: max_positives_per_object=8, mining_nms=0.7
```

Mining replaces the hard negatives and keeps the initial random anchors, so
the sample budget stays fixed. The caller retains the full proposal cache;
checkpoints save the selected bank and sampling settings. Positive capping is
also available directly through `add_example(..., max_positives_per_object=8)`.
On the same 500-image DIOR subset, Torch-fit baseline/capping/mining/both reached
32.20/32.31/32.29/32.28 mean mAP, with mixed changes across support draws. Sampling
remains optional. See the [sampling comparison](docs/flame-sampling-findings.md)
for GPU timings, convergence checks and commands.

Cached support can train the default box head and classifiers together:

```python
cached = [tuple(t.cpu() for t in p)
          for p in flame.proposal_features(support_images)]
flame.fit_support(list(zip(cached, support_boxes, support_labels)))
boxes, scores, labels = flame.detect("test.jpg")
```

Full five-tensor `proposal_features()` records include the visual features
needed by the box head. Historical four-tensor `proposals()` records still
train classifiers alone. `fit_box_refiner([(visual, proposals, truth), ...])`
can explicitly fit a head from a separate boxed cache or override its settings.

The linear head uses frozen visual features to predict normalized
center shifts and log-size changes before NMS. Ridge regularization favors zero
correction, with explicit bounds on both kinds of offset. Each represented
support object has equal fitting weight, with at most four proposal rows per
object. The base model stays frozen, scores are preserved, and detection still
uses one encoder pass. Checkpoints save the fitted head and compact boxed
support bank for incremental fitting. Yes/no labels alone cannot fit it.

On the same 500 held-out DIOR images and three support draws, the development-
selected setting (`min_iou=0.3, ridge=0.01, strength=0.5`) improved mean mAP from
32.205 to 32.532, AP75 from 31.914 to 33.197, and small-object AP from 13.655 to
14.189. All three draws improved. It adds 12,304 bytes and about 3.1 ms per
cached image on CPU; GPU latency was not measured. See the
[box-head experiment](docs/flame-boxes-findings.md) for details and commands.

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
