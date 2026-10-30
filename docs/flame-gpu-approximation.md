# FLAME with Torch feature approximations

`Flame(..., svm_backend="nystrom" | "rff", n_components=...)` fits on the
OWLv2 model's device without adding dependencies. The current default is the
full-kernel Torch fitter; the measurements below used sklearn as the reference.

The reference and approximate fits share exactly the same labelled proposals,
per-feature mean and sample standard deviation, `gamma="scale"`, `C=1`, balanced
class penalties, and `alpha=8` for blending the classifier margin with OWLv2.
Approximate fits use damped Newton steps in Torch and a Huber-smoothed hinge with width 0.01,
instead of sklearn's unsmoothed hinge. This changes each individual hinge loss
by at most 0.005; it does not guarantee a bound on detection AP.

Nyström selects landmarks with representation from both classes, retaining all
rare positives when they fit within half the feature budget. It constructs
`K(X,L) U diag(lambda**-0.5)` from the landmark kernel eigendecomposition,
with eigenvalues floored at `1e-7`. After fitting, it folds the basis and linear
weights into coefficients over landmarks, so prediction uses the existing RBF
expansion. Fourier features use Gaussian frequencies with variance `2*gamma`,
uniform phases, and the usual `sqrt(2 / n_components)` cosine normalization.
Both classifiers have an unregularized intercept.

The feature maps and exported heads use float32. The Newton gradient and
Hessian calculations use float64, and convergence requires a maximum absolute
gradient below `1e-7` for the normalized objective. Each fitted head exposes
its iteration count, gradient, objective, convergence flag and stop reason.

## COCO protocol

- Pretrained OWLv2-base with default image preprocessing and single-label FLAME
  postprocessing (threshold zero, class-aware NMS 0.5, top 100 per image).
- All 80 classes, prompts `a photo of a {class}`.
- Evaluate on the existing 500-image COCO val2017 mini set, with standard COCO
  box AP over IoU 0.50:0.95 and crowd handling. This mini set was selected for
  class coverage; these numbers are not full val2017 AP.
- Local annotations only include val2017. Select five crowd-free support images
  per class (or all available when fewer) from the other val2017 images, pool them, and annotate all classes
  in each selected image. The entire mini500 is excluded from support selection.
- The support and evaluation images are disjoint. C and alpha remain at their
  existing defaults; the listed feature budgets are compared on the same split.
- Every backend uses the same cached float32 proposal outputs. Labelled example
  embeddings are converted to float16 by the existing FLAME sampler, identically
  for the reference and approximations.
- GPU: RTX 3060 12 GB. sklearn fits on CPU; both approximate classifiers fit
  on CUDA. Fit time includes preprocessing, transfers, map construction and
  optimization, and excludes image encoding and label sampling. Peak fit memory
  is additional Torch-allocated GPU memory above the resident model.
- CPU: AMD Ryzen 5 5600, six cores / twelve threads. Torch 2.11.0+cu130,
  sklearn 1.8.0; eight Torch CPU threads, sequential per-class sklearn fitting.
- Prediction timings include cached-proposal disk I/O, CPU-to-GPU transfers,
  scoring, NMS and detection export, and exclude OWLv2 image encoding.
- Support seed 0 pools 373 images. Every query receives 37,300 negatives;
  positive proposal counts range from 26 to 8,070 (median 293.5).

## Results

AP is reported on a 0–100 scale. Deltas are absolute AP points. This is one
support draw on mini500, without C/alpha tuning or confidence intervals.

| Method | AP | AP50 | Delta vs zero-shot | Delta vs sklearn | Fit (s) | Scoring (s) | Head (MiB) | Extra GPU fit (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 48.5115 | 68.7403 | — | -0.8652 | — | — | — | — |
| sklearn RBF SVC | 49.3768 | 70.3534 | +0.8652 | 0.0000 | 294.98 | 67.63 | 72.48 | — |
| Nyström 256 | 47.0163 | 67.0748 | -1.4952 | -2.3605 | 46.10 | 45.22 | 40.39 | 229.05 |
| Nyström 1024 | 48.2850 | 68.5997 | -0.2265 | -1.0918 | 155.06 | 116.85 | 160.62 | 677.01 |
| Fourier 256 | 41.9126 | 59.6826 | -6.5989 | -7.4641 | 43.37 | 36.47 | 40.47 | 227.93 |
| Fourier 1024 | 47.5430 | 67.1987 | -0.9686 | -1.8338 | 106.47 | 101.53 | 160.94 | 662.50 |

All 320 approximate classifiers converged below the normalized gradient
tolerance. There were no missing classifiers. The 256-feature fits take about
one sixth of the sklearn fit time, but lose more AP than sklearn FLAME adds.
Nyström 1024 is the closest approximation tested: 1.90 times faster to fit,
1.09 AP below sklearn, and 0.23 AP below zero-shot. It also takes 1.73 times as
long to score as sklearn because its expansion has more vectors. Fourier 1024
fits 2.77 times faster than sklearn and loses 1.83 AP.

The Nyström 1024 deficit is uneven across classes: sandwich loses 10.15 AP,
carrot 5.55, dining table 5.38, hot dog 4.68 and skateboard 4.59 relative to
sklearn. Full per-class values are in the JSON. The tested feature budgets do
not preserve the reference FLAME gain on this split. At the time of that
experiment sklearn remained the default; the full-kernel Torch fitter is now
the default for new instances.

Final metrics: [coco_newton_seed0.json](../data/flame_features/coco_newton_seed0.json).
The sklearn control was reused from an earlier pass over the identical cached
proposals and support set; the result marks its provenance. All approximate
numbers in this table use the final Newton solver.

Verification: nine checks passed on CPU/CUDA, covering agreement with sklearn
in the same feature space, full-rank Nyström recovery, rare-positive landmarks,
zero-shot fallback, saved backend settings and older FLAME files.

## Quick DIOR comparison

Reduced follow-up: sklearn and Nyström 1024 only, one support seed, on 200
DIOR test images covering all 20 classes. Five train images per class are
sampled and pooled into 100 support images. Model, C, alpha, loss smoothing,
preprocessing and postprocessing follow the COCO experiment. Prompts are
`a satellite photo of a {name}`, with lowercase names and hyphens normalized to
spaces, matching the existing DIOR evaluator.

| Method | AP | AP50 | Gain vs zero-shot | Delta vs sklearn | Fit (s) | Scoring (s) |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 8.7276 | 14.4173 | — | -27.2515 | — | — |
| sklearn RBF SVC | 35.9792 | 56.9752 | +27.2515 | 0.0000 | 11.70 | 6.56 |
| Nyström 1024 | 35.2905 | 55.8331 | +26.5628 | -0.6887 | 14.85 | 13.18 |

Nyström retains **97.47% of sklearn's improvement over zero-shot** on this
probe. All 20 classifiers converge and none are missing. Its sampled margin
MAE against sklearn is 0.2828, with 0.0766% sign disagreement. Nyström is
slightly slower to fit at this smaller support size, and about twice as slow
to score. The DIOR comparison has 10,000 negatives per class versus 37,300 in
the COCO experiment; both domain and support-set size differ.

The images are selected deterministically with seed 0 and at least five test
images containing each class. This is an exploratory subset, rather than full
DIOR test AP. Train and test have overlapping numeric image IDs for different
files; sampling excludes evaluation image paths, and proposal caches are
namespaced by image root and query vocabulary. Disjoint actual paths and
separate train/test caches were checked. DOTA and Fourier runs were omitted
from this reduced comparison.

Metrics and per-class AP:
[quick_seed0_1024.json](../data/flame_features_dior/quick_seed0_1024.json).

```bash
.venv/bin/python tools/make_coco_mini.py \
  --annotations /mnt/datasets/sat/dior_coco/test/annotations.json \
  --output data/flame_features_dior/test_mini200.json \
  --num-images 200 --min-images-per-class 5 --seed 0

.venv/bin/python tools/benchmark_flame_features.py --dataset-label dior \
  --annotations /mnt/datasets/sat/dior_coco/train/annotations.json \
  --eval-annotations data/flame_features_dior/test_mini200.json \
  --support-image-root /mnt/datasets/sat/dior_coco/train/images \
  --eval-image-root /mnt/datasets/sat/dior_coco/test/images \
  --query-template 'a satellite photo of a {name}' \
  --device cuda:1 --components 1024 --backends nystrom --seeds 0 \
  --cache-dir data/flame_features_dior \
  --output data/flame_features_dior/quick_seed0_1024.json
```

## DIOR support scaling

Follow-up uses the same 200 evaluation images, 20 classes, support seed 0,
1024 features, `C=1`, `alpha=8`, model and hardware as the quick DIOR probe.
Larger support image sets contain all images from the smaller sets. The
5-image-per-class point reuses the previous measurement; the other three
points are fresh fits. Encoding and label sampling are excluded from fit time.
The CPU baseline is the existing sequential full RBF SVC in `Flame.fit()`,
with sklearn's default `cache_size=200`.

| Images/class | Pooled support images | Negatives/class | sklearn fit (s) | Nyström fit (s) | Speedup | sklearn AP | Nyström AP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 100 | 10,000 | 11.70 | 14.85 | 0.79× | 35.979 | 35.290 |
| 10 | 200 | 20,000 | 37.69 | 20.29 | 1.86× | 38.605 | 36.389 |
| 20 | 398 | 39,800 | 129.97 | 30.71 | 4.23× | 39.744 | 37.007 |
| 40 | 792 | 79,200 | 596.98 | 46.92 | 12.72× | 40.612 | 35.987 |

Speedup is sklearn time divided by Nyström time; values below one favor
sklearn. All times fit all 20 classifiers, not one classifier. Every Nyström
classifier converged at every size. The timing crossover is between the
10,000- and 20,000-negative points in this setup. Across a 7.92-fold increase
in negatives, sklearn fit time grew 51.00-fold and Nyström 3.16-fold.

At the largest size, Nyström's speedup costs 4.63 AP versus sklearn, retaining
85.49% of sklearn's improvement over the unchanged 8.73 AP zero-shot control.
At the fixed feature budget, its accuracy stops improving as more support is
added in this draw. Its landmarks are selected afresh when the support pool
changes, using the same seed. These are single measurements at each size,
and quality is assessed on the same small held-out subset.

Additional images contribute positives as well as negatives: the median
positives per query increase from 236.5 to 360.5, 914.0 and 1825.5. The
experiment measures ordinary FLAME fitting as labelled support grows.

Nyström's stored heads stay at 40.16 MiB, while sklearn's grow from 11.66 to
56.92 MiB. Scoring the fixed 200 images stays near 13.2 seconds for Nyström;
sklearn grows from 6.56 to 17.39 seconds. Extra GPU memory during the largest
Nyström fit peaks at 1252.3 MiB above the resident model.

![DIOR fit time and AP loss as the support set grows](../data/flame_features_dior/scaling_summary.png)

The [summary JSON](../data/flame_features_dior/scaling_summary.json) records
support IDs, exact counts, convergence, timings, AP and source result paths.
The [SVG](../data/flame_features_dior/scaling_summary.svg) is available for export.
Support manifests are `data/flame_features_dior/scaling_support_{5,10,20,40}.json`.
The first five images per class reproduce the original seed-0 draw; extra
images use a deterministic class-specific seed `flame-scaling:0:{category_id}`.

To repeat a larger point, use the quick DIOR command above with the matching
`--support-images` count, add `--support-manifest`, and choose a distinct output:

```bash
# Arguments to add/replace in the quick DIOR command for the largest point:
--support-images 40 \
--support-manifest data/flame_features_dior/scaling_support_40.json \
--output data/flame_features_dior/scaling_40.json

MPLCONFIGDIR=/tmp/flame-scaling-matplotlib .venv/bin/python tools/summarize_flame_scaling.py \
  data/flame_features_dior/quick_seed0_1024.json \
  data/flame_features_dior/scaling_{10,20,40}.json \
  --output data/flame_features_dior/scaling_summary
```

## Reproduce

```bash
.venv/bin/python tools/benchmark_flame_features.py \
  --device cuda:1 --components 256 1024 --seeds 0 \
  --output data/flame_features/coco_newton_seed0.json
```

Use the CUDA device index corresponding to the intended GPU on your host.
Cached proposals and full machine-readable metrics live under the ignored
`data/flame_features` directory. The JSON records image IDs, per-class AP,
support counts, missing classifiers, solver diagnostics and sampled margin
differences. Margin comparisons sample the 64 highest-objectness proposals per
evaluation image. `--reference-results` can reuse an earlier sklearn control
after checking that its support/evaluation images, class order, model, C and
alpha match. Reused controls are marked in the output; margin comparisons are
null when their raw reference margins are unavailable.
