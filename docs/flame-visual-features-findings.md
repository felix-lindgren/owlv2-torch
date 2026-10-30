# Pre-projection visual features for FLAME

Experiment date: 2026-10-29. Frozen pretrained OWLv2-base, DIOR, sklearn RBF SVM.
The default feature source remains `"class"`.

**Result:** visual-only features reduced mean held-out mAP by 1.058 points.
The selected 10% visual blend improved it by 0.122 points, with gains in all
three support draws. It also roughly doubled SVM fitting time and increased
cached scoring time by about 42%. Keep this as an optional accuracy/cost
tradeoff, without changing the default.

## Representation and implementation

FLAME normally uses the normalized 512-dimensional embeddings from the class
head. This experiment exposes the 768-dimensional visual features immediately
before `class_head.dense0`, after OWLv2's detection feature normalization.
These features already exist in the same encoder pass. The three sources are:

- `feature_source="class"`: the original 512-dimensional representation.
- `feature_source="visual"`: pre-projection visual features, 768 dimensions.
- `feature_source="combined"`: concatenate both branches, 1,280 dimensions.

Visual rows are L2-normalized by default; `visual_normalize=False` retains their
raw values. Every source keeps the original class logits, objectness and boxes.
There are no backbone updates or additional image encoder passes at deployment.
The SVM can require more fitting/scoring work with the larger representations.

Combined mode applies block weights **after** per-channel standardization,
since standardization would cancel constant scaling applied to input features.
For a visual weight `w`, the distance contribution is proportional to:

```text
(1 - w) * squared_distance(standardized_class) / 512
    + w * squared_distance(standardized_visual) / 768
```

This balances the blocks independently of their different dimension counts.
The existing gamma="scale" rule is retained. Feature choices and selected
support embeddings are saved with checkpoints. Older checkpoints default to
the class source. APE channel selection currently requires that class source.

`proposal_features()` exposes both branches and the detection outputs for
caching; `proposals()` selects the configured representation for ordinary
`add_example()`, `suggest()` and `detect()` calls.

## Protocol and fresh control

- All 20 DIOR classes, queries `a satellite photo of a {name}`, C=1, alpha=8,
  balanced SVC weights, gamma="scale".
- Same five train images per class, pooled into 100 support images for each
  draw, as the earlier APE/GDA/Tip experiments. All instances are available;
  this is not five object instances per class.
- Same 200 development train images, disjoint from all three support draws.
  Settings are selected using support seed 0 on development data.
- Final comparison: the same 500 test images, support seeds 0/1/2. No settings
  are selected from these held-out results.
- Identical proposal rows and binary targets across feature modes, including
  IoU >= 0.7 positives, IoU < 0.3 negatives, and 50 hard plus 50 random negatives
  per support image/query. The benchmark checks target equality across banks.
- Same per-patch class choice, NMS 0.5, top 100 detections, threshold 0.0.
- CPU sklearn fitting, GPU scoring on an RTX 5070 Ti.

Both feature branches and all detection outputs were re-extracted together
into a separate full-float32 cache, `base_visual_float32_v1`. This avoids
combining freshly extracted visual features with historical projected features
or historical boxes. The fresh class control is used for every paired
comparison. On development it reached 31.391 mAP versus 31.063 from the older
cache, so the absolute numbers from the two cache generations are not
interchangeable. No cause for that cache difference is established here.

## Held-out results

All AP values are percentages. Each comparison uses the same freshly extracted
proposals and support labels; the visual blend was fixed on development data.

| Feature source | Seed 0 mAP | Seed 1 mAP | Seed 2 mAP | Mean mAP | AP50 | AP75 |
|---|---:|---:|---:|---:|---:|---:|
| Class, 512 dimensions | 32.616 | 31.406 | 32.591 | **32.204** | 52.766 | 31.915 |
| Visual, 768 dimensions | 31.734 | 30.261 | 31.445 | **31.146** | 50.711 | 31.049 |
| Combined, visual weight 0.1, 1,280 dimensions | 32.630 | 31.631 | 32.719 | **32.327** | 52.709 | 32.130 |

The blend's paired changes were +0.014, +0.225 and +0.128 points. This is a
small, consistent gain on these support draws, not evidence of a large or
universal improvement. AP75 increased by 0.215 points while AP50 decreased by
0.057 points. Box coordinates were unchanged, so these changes come from
scoring and selecting proposals rather than predicting different boxes.

| Feature source | Mean SVM fit, seconds | Cached scoring of 500 images, seconds | Classifier, MiB |
|---|---:|---:|---:|
| Class | 11.728 | 6.560 | 12.174 |
| Visual | 17.410 | 7.971 | 19.880 |
| Combined, visual weight 0.1 | 25.074 | 9.306 | 30.302 |

Fit timings exclude encoder extraction. Scoring includes cache I/O, feature
transfers/normalization, classifier evaluation and postprocessing. These are
not end-to-end detector latency measurements. The blend uses about 2.5 times
the classifier storage, despite retaining one image encoder pass.

All feature modes had classifiers for every query and matching support-label
counts. The fresh-cache zero-shot mAP on this subset was 7.075. The mixture
weight was selected using additional labeled development images; this does
not establish a universal mixture setting for new domains without validation.

## Development selection

Nine feature settings were compared: visual-only and visual weights
0.25/0.5/0.75 with and without row normalization, plus a normalized 0.1 blend
after the smaller visual contribution performed best in the initial sweep.

| Setting | Development mAP |
|---|---:|
| Fresh class control | 31.391 |
| Normalized visual only | 30.015 |
| Raw visual only | 30.018 |
| Normalized combined, visual weight 0.10 | **31.532** |
| Normalized combined, visual weight 0.25 | 31.342 |
| Normalized combined, visual weight 0.50 | 30.900 |
| Normalized combined, visual weight 0.75 | 30.360 |

The fixed held-out blend uses normalized visual features with weight 0.1.
For the visual-only comparison, the 0.003-point raw/normalized difference was
treated as a tie (<0.01 points), retaining the normalized default. Selection is
recorded in `data/flame_visual_dior/selection.json` before held-out evaluation.

## Usage

Choose a source before collecting support examples:

```python
flame = Flame(model, queries, feature_source="visual")
# Or combine both branches with the development-selected visual contribution:
flame = Flame(model, queries, feature_source="combined", visual_weight=0.1)
flame.add_example(image, boxes, labels)
flame.fit()
boxes, scores, labels = flame.detect(query_image)
```

Switching source requires extracting support features again; existing
512-dimensional support banks cannot recover discarded visual dimensions.
Cached full branch outputs can be reused across all source choices.

## Reproduction

```bash
FLAME_VISUAL_ARGS=(
  --device cuda:0 --dataset-label dior
  --annotations /mnt/datasets/sat/dior_coco/train/annotations.json
  --support-image-root /mnt/datasets/sat/dior_coco/train/images
  --query-template 'a satellite photo of a {name}'
  --support-manifest data/flame_ape_dior/support_5.json
  --backends sklearn --cache-dir data/flame_visual_dior
)

.venv/bin/python tools/benchmark_flame_features.py "${FLAME_VISUAL_ARGS[@]}" \
  --eval-image-root /mnt/datasets/sat/dior_coco/train/images \
  --eval-annotations data/flame_ape_dior/dev200.json --seeds 0 \
  --feature-sources class visual combined --visual-weights 0.25 0.5 0.75 \
  --output data/flame_visual_dior/dev_normalized.json

.venv/bin/python tools/benchmark_flame_features.py "${FLAME_VISUAL_ARGS[@]}" \
  --eval-image-root /mnt/datasets/sat/dior_coco/train/images \
  --eval-annotations data/flame_ape_dior/dev200.json --seeds 0 \
  --feature-sources class visual combined --visual-weights 0.25 0.5 0.75 \
  --visual-no-normalize --reference-results data/flame_visual_dior/dev_normalized.json \
  --output data/flame_visual_dior/dev_raw.json

.venv/bin/python tools/benchmark_flame_features.py "${FLAME_VISUAL_ARGS[@]}" \
  --eval-image-root /mnt/datasets/sat/dior_coco/train/images \
  --eval-annotations data/flame_ape_dior/dev200.json --seeds 0 \
  --feature-sources combined --visual-weights 0.1 \
  --reference-results data/flame_visual_dior/dev_normalized.json \
  --output data/flame_visual_dior/dev_low_visual_weight.json

.venv/bin/python tools/benchmark_flame_features.py "${FLAME_VISUAL_ARGS[@]}" \
  --eval-image-root /mnt/datasets/sat/dior_coco/test/images \
  --eval-annotations data/flame_features_dior/test_mini500.json --seeds 0 1 2 \
  --feature-sources class visual combined --visual-weights 0.1 \
  --output data/flame_visual_dior/heldout500.json
```

The output records cache paths, exact support/eval IDs, selected dimensions,
normalization and weighting, AP, fit time, scoring time and classifier size.
`data/` and experiment logs are local artifacts ignored by git.

## Verification

Syntax/import checks passed. Manual save/load smoke checks on actual cached
support proposals reproduced margins exactly for all three sources; their
`detect()` paths produced finite scores. Legacy checkpoint loading retained
the class source. A direct calculation of the weighted block distance agreed
within 1.2e-7. No automated test files were added or changed.
A real-image CUDA smoke check confirmed exactly one encoder call per `detect()`
invocation for each source, with finite detection scores. Checkpoint loading
rejects a representation override that would invalidate stored support features.

The user's requested future ideas are saved in
[FLAME follow-up notes](flame-next-steps.md): post-fit hard-negative mining,
instance balancing, and a small regularized box-correction head. These are not
implemented by this feature comparison.
