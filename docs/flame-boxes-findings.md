# A small regularized box-correction head for FLAME

Experiment date: 2026-10-30. Frozen pretrained OWLv2-base, DIOR, full-kernel
Torch C-SVC. This localization experiment uses the original
classifier, backbone and box predictor are unchanged.

**Result:** mean held-out mAP improved from **32.205 to 32.532 (+0.327
points)**, with gains in all three support draws. AP75 improved by **1.283
points** and small-object AP by **0.535 points**, also positive in every draw.
The selected setting became the default for new boxed-support fits on
2026-10-30 at the user's request. These results cover one 500-image DIOR subset.

## Implementation

`Flame.fit_box_refiner()` fits a shared, class-agnostic linear regressor using
the frozen visual features before the class projection. On OWLv2-base it has
768 inputs and four outputs plus an intercept: 3,076 float32 parameters,
12,304 bytes (about 12 KiB). It adds no image encoder passes.

The targets are ordinary normalized box offsets:

```text
dx = (GT_center_x - proposal_center_x) / proposal_width
dy = (GT_center_y - proposal_center_y) / proposal_height
dw = log(GT_width / proposal_width)
dh = log(GT_height / proposal_height)
```

Support proposals match the ground-truth box with highest IoU, requiring a
minimum IoU. For each represented object, retain its four tightest matching
proposals and give them equal total object weight. Objects without a matching
proposal do not supply regression targets. This uses boxed support; yes/no
labels alone cannot train the regressor.

Visual features are row-normalized, then an intercept feature is appended.
The fit minimizes `sum_i w_i ||x_i W - target_i||^2 + ridge * ||W||^2`, with
weights summing to one and the intercept included in the penalty. A float64 Cholesky
solve gives the closed-form solution, which is stored as float32. The prior is
zero correction; an unfitted head and `strength=0` return the original boxes
exactly. There is no randomly initialized hidden layer or backbone training.

At inference, center offsets are bounded to +/-0.25 of the original width or
height and log-size offsets to +/-0.4. The selected correction strength scales
both bounded offsets. With strength 0.5, this allows at most a 12.5% center
shift and size multiplication between `exp(-0.2)` and `exp(0.2)` (0.819-1.221).
The corrected boxes enter the existing NMS. Classifier scores and per-patch
class choices are unchanged; the boxes retained by NMS can change.

New `Flame` instances enable automatic box refinement. `add_example()` retains
the matched visual rows and original boxes, with at most four rows per object;
`fit()` fits or refreshes the head after new boxed examples. Empty and yes/no-only
support keep the original boxes. Set `box_refinement=False` to disable automatic
collection and fitting, or explicitly call `fit_box_refiner()` for a custom head.

The fitted head, configuration, enable flag and compact support bank are saved
in checkpoints. Loading restores the head and refits the classifiers; boxed
examples saved before fitting are fitted on load. Older checkpoints retain
their original enable behavior and any explicitly saved box head. Loading with
`box_refinement=False` explicitly disables a saved head.

## Protocol and development selection

- Same full-float32 `base_visual_float32_v1` caches as the
  [feature comparison](flame-visual-features-findings.md).
- All 20 DIOR classes, query `a satellite photo of a {name}`, C=1, alpha=8.
- Original class embeddings and uncapped support sampling for the classifier:
  IoU >= 0.7 positives, IoU < 0.3 negatives, 50 hard and 50 random negatives per
  support image/query. No negative mining or feature blend is added.
- Same five train images per class, pooled into 100 fully annotated support
  images per draw. This is an image budget, not five instances per class.
- Development: support seed 0, the same separate 200 train images.
- Held-out: the same 500 test images, support seeds 0/1/2. The box configuration
  is fixed before looking at held-out metrics.
- Per-patch best class, threshold 0, class-wise NMS 0.5, top 100 detections.
- CPU execution with six Torch threads in this session; no CUDA device was
  available. The fitter remains `svm_backend="torch"` throughout.

Twelve box settings compare minimum matching IoU 0.3/0.5, ridge
0.001/0.01/0.1, and correction strength 0.5/1.0. Bounds and the four-proposal cap
are fixed. Select the box setting with highest development mAP. The original
Torch classifier is fitted once per support draw and its proposal scores are
reused for every box setting, including the unchanged-box control.

AP values are percentages; differences are percentage points.

| Development setting | mAP | AP75 | Small-object AP |
|---|---:|---:|---:|
| Original boxes | 31.395 | 34.429 | 10.531 |
| IoU 0.3, ridge 0.01, strength 0.5 (selected) | 31.744 | 35.498 | 10.763 |
| IoU 0.5, ridge 0.01, strength 0.5 | 31.738 | 35.201 | 10.701 |

The selected setting improves development mAP by 0.349 points, with a close
alternative only 0.007 points behind. The support contains 1,106 boxes; 1,067
objects contribute 3,632 regression rows. Development selection is additional
supervision beyond the support image budget.

## Held-out results

| Method | Seed 0 mAP | Seed 1 mAP | Seed 2 mAP | Mean mAP | Mean change |
|---|---:|---:|---:|---:|---:|
| Original boxes | 32.618 | 31.406 | 32.591 | **32.205** | — |
| Selected box head | 32.860 | 31.703 | 33.035 | **32.532** | **+0.327** |

Paired mAP changes were +0.242, +0.297 and +0.443 points. The configuration
was selected on development images before evaluating these three draws.

| Metric | Original boxes | Box head | Change |
|---|---:|---:|---:|
| AP50 | 52.769 | 52.853 | +0.085 |
| AP75 | 31.914 | 33.197 | **+1.283** |
| Small-object AP | 13.655 | 14.189 | **+0.535** |
| Medium-object AP | 29.373 | 29.833 | +0.460 |
| Large-object AP | 43.966 | 44.092 | +0.126 |

AP75 gains were +1.319/+1.325/+1.205 points across seeds 0/1/2; small-object
gains were +0.878/+0.330/+0.396. The larger effect at AP75 is consistent with
improved localization. Class scores are shared exactly between the two arms;
box correction and its effect on NMS account for the measured differences.
The three support draws use the same 500 evaluation images. Broader domain
coverage and full-split results have not been measured here.

All 80 per-class Torch fits across development and held-out runs converged
at the unchanged global KKT tolerance of 1e-3. Every held-out query had a
classifier. Baseline training-label counts matched the saved Torch control;
the largest held-out CPU/GPU baseline mAP difference was 0.0011 points.

## Cost on CPU

Mean component times across the three held-out support draws:

| Component | Original boxes | Box head |
|---|---:|---:|
| Classifier adaptation on 100 support images, seconds | 6.700 | 6.700 (reused) |
| Additional box-head fit, seconds | 0 | **0.369** |
| Classifier scoring of 500 cached images, seconds | 58.252 | 58.252 (reused) |
| Additional box correction on 500 images, seconds | 0 | **1.566** |
| Detection selection and NMS on 500 images, seconds | 2.739 | 2.476 |
| Additional box-head storage, bytes | 0 | **12,304** |

Box fitting includes support matching, normalization and the ridge solve.
The shared classifier's fit alone takes 5.732 seconds; adaptation additionally
includes support sampling. Box correction adds about 3.1 ms per image in this
cached CPU run. These are component timings, excluding model loading, cache
I/O, initial evaluation-tensor transfers and image encoding; they are not
end-to-end detector latency. GPU latency for this head was not measured.

The three support pools contain 1,106/626/807 boxes, of which 1,067/614/765
objects supply regression targets using 3,632/2,201/2,789 proposal rows.

## Usage and reproduction

The box head can reuse the full proposal cache independently of the chosen
SVM feature representation:

```python
flame = Flame(model, queries, svm_backend="torch")
cached = [tuple(t.cpu() for t in p)
          for p in flame.proposal_features(support_images)]
flame.fit_support(list(zip(cached, support_boxes, support_labels)))
boxes, scores, labels = flame.detect(query_image)
flame.save("flame_with_boxes.pt")
```

`proposal_features()` and `proposals()` still expose original boxes for support
matching. `detect()` applies the fitted box correction before selection and
NMS. Cached callers use `flame.refine_boxes(visual_features, proposal_boxes)`.

```bash
.venv/bin/python tools/benchmark_flame_boxes.py --device cpu \
  --reference-results data/flame_visual_dior/dev_normalized.json \
  --baseline-results data/flame_sampling_dior/dev200.json --seeds 0 \
  --output data/flame_boxes_dior/dev200.json \
  --selection-output data/flame_boxes_dior/selection.json

.venv/bin/python tools/benchmark_flame_boxes.py --device cpu \
  --reference-results data/flame_visual_dior/heldout500.json \
  --baseline-results data/flame_sampling_dior/heldout500.json \
  --configs data/flame_boxes_dior/selection.json \
  --output data/flame_boxes_dior/heldout500.json
```

JSON artifacts preserve exact support/evaluation IDs, cache paths, selected
settings, per-class AP, support coverage, convergence diagnostics and timing
components. The runner checks support/evaluation disjointness and refuses
held-out evaluation on the recorded development image paths. Data and logs are
local ignored artifacts.

The comparison runners explicitly disable automatic box refinement so that
their unchanged-box controls reproduce the original protocol. The box-head
runner then fits its explicit candidates. Recorded fit timings predate the
addition of automatic support-row retention; the box inference path is unchanged.

## Manual verification

Syntax/import checks and `git diff --check` passed. Real cached proposals
produced finite, positive-size boxes within the configured correction bounds.
Unfitted and zero-strength heads returned the original boxes exactly, and
empty boxed support was rejected with an informative error.

A full `Flame.save()` / `Flame.load()` smoke check reproduced corrected boxes,
scores and configuration exactly. A checkpoint without the new box-head field
retained its original boxes and scores. A live image produced 100 finite
detections with exactly one encoder call, and model parameter versions stayed
unchanged through fitting and detection. No automated test files were added
or changed for this experiment.

After default promotion, manual checks on real cached proposals confirmed that
automatic fitting reproduces the selected explicit head exactly, retains at
most four rows per object and preserves classifier scores. A real
`add_example()` / `fit()` run trained the head automatically. Save/load retained
the compact bank and correctly fitted pending incremental examples. Explicit
opt-out, historical class-only caches, yes/no-only support, empty ground truth
and old checkpoints retained their intended behavior. Live default detection
again used one encoder call and produced finite outputs.
