# FLAME support sampling and post-fit negative mining

Experiment date: 2026-10-29. Frozen pretrained OWLv2-base, DIOR, projected
512-dimensional features, full-kernel Torch C-SVC on an RTX 5070 Ti.

**Result:** the changes gave small, inconsistent improvements. Mean mAP was
32.204 for the Torch baseline, 32.305 with capped positives, 32.290 after one
mining pass, and 32.279 with the selected combination. Each modification lost
accuracy on one support draw. Keep the sampling changes optional.

New `Flame` instances now default to `svm_backend="torch"`, as requested. An
explicit sklearn backend remains available for reference comparisons. Saved
checkpoints retain their backend; legacy files with no backend field retain
their original sklearn interpretation. `Flame.load(..., svm_backend="torch")`
can explicitly switch an old checkpoint's fitter.

## Implementation

`Flame.fit_support()` rebuilds a training bank from caller-owned cached
`(proposals, ground_truth_boxes_xyxy, labels)` records. Proposals use the usual
four tensors: selected embeddings, original class logits, objectness and boxes.
No encoder pass is needed during sampling or mining. Ground-truth boxes must
cover every instance of every query in those support images.

### Positive caps

Original positives are all proposals whose maximum IoU with their query's GT
boxes is at least 0.7. The capped variant assigns each positive to its best GT
match and keeps at most K proposals per object. It starts with the tightest
proposal, then greedily chooses different feature directions. Objects with
fewer eligible positives retain all of them. The cap reduces the influence of
objects that contribute many near-duplicate patches; it is not exact equal
weighting of all instances.

This is available through `add_example(..., max_positives_per_object=K)` and
`fit_support(..., max_positives_per_object=K)`. The default remains uncapped.

### One mining pass

Initial fitting uses the original sampler: 50 highest zero-shot-score negatives
and 50 random negatives per support image/query, all with IoU < 0.3.
After fitting, every support image's cached proposals are scored with the full
FLAME prediction, including its original logit and objectness. The 50 hard
negatives are replaced by the highest refined-score eligible negatives.

The original 50 random negatives are **fixed anchors**. They are excluded from
hard selection, and the positives remain unchanged within a mining experiment.
The bank is rebuilt, not appended to, and fitted again. Assertions enforce the
same positive/negative counts before and after mining and disjoint hard/random
indices. One pass therefore uses two SVM fits with a fixed training-row budget.

`mining_nms=0.7` additionally suppresses overlapping negative boxes before
selection. If that leaves fewer than the original hard-negative budget, the
sampler fills from the remaining ranked candidates to preserve that budget.
All labels still come exclusively from the support images.

The original uncapped sampler was factored into a shared helper. A manual
comparison on actual cached proposals matched its positive, hard and random
indices exactly against the earlier implementation.

Selected examples and sampling configuration are saved. Reloading refits the
final selected bank; it does not need the original full proposal pool. The
full pool remains caller-owned for another resampling/mining experiment.

## Protocol

- Same fresh float32 proposal cache and support manifests as the
  [visual-feature experiment](flame-visual-features-findings.md).
- All 20 DIOR classes. Five fully annotated train **images** per class, pooled
  into 100 images per support draw. All annotated instances are available.
- Original projected features only, C=1, alpha=8, balanced class penalties,
  gamma="scale", KKT tolerance 1e-3, up to 300 working-set solves per classifier,
  512-variable working sets and a 256 MiB kernel-row cache.
- Per-patch best class, NMS 0.5, top 100 detections and threshold 0.0 are
  unchanged. Objectness and proposal boxes are unchanged.
- Development: 200 train images disjoint from the union of support draws,
  support seed 0 only. These additional labeled images select settings.
- Held-out: the same 500 DIOR test images, support seeds 0/1/2. Settings were
  fixed before this evaluation.
- All fitting uses the full-kernel **Torch GPU fitter**. No sklearn fitting or
  approximate feature solver was used in this experiment. The archived CPU
  controls are used only to check agreement.

Twelve development configurations cover uncapped/2/4/8 positives per object,
each with no mining, one mining pass, or one spatially diverse mining pass.
The best nonbaseline setting was chosen separately for capping, mining and
their combination, including when its development AP was below baseline.

| Development-selected family | Setting | Development mAP |
|---|---|---:|
| Torch baseline | Uncapped, no mining | 31.395 |
| Capped positives | Cap 8, no mining | 31.048 |
| Mining | Uncapped, one pass, no mining NMS | 31.664 |
| Both | Cap 8, one pass, mining NMS 0.7 | 31.676 |

On development, capping reduced positive rows from 5,980 to 3,205; the negative
bank remained 200,000 rows. The uncapped mining pass replaced 44.9% of the
100,000 hard negatives while retaining all 100,000 random negatives.

## Held-out results

AP values are percentages; differences are percentage points.

| Method | Seed 0 mAP | Seed 1 mAP | Seed 2 mAP | Mean mAP | Mean change |
|---|---:|---:|---:|---:|---:|
| Torch baseline | 32.616 | 31.405 | 32.592 | **32.204** | — |
| Cap 8 positives per object | 32.690 | 31.775 | 32.451 | **32.305** | +0.101 |
| One mining pass | 32.908 | 31.563 | 32.400 | **32.290** | +0.086 |
| Cap 8 + diverse mining | 32.854 | 31.551 | 32.430 | **32.279** | +0.074 |

The paired changes are mixed: +0.074/+0.370/-0.140 for capping,
+0.291/+0.158/-0.192 for mining, and +0.238/+0.146/-0.162 for both. The small
mean advantages do not establish a reliable accuracy improvement.

| Method | AP50 | AP75 | Small-object AP |
|---|---:|---:|---:|
| Torch baseline | 52.768 | 31.914 | 13.655 |
| Cap 8 | 52.825 | 32.093 | 13.671 |
| Mining | 52.609 | 32.303 | 13.638 |
| Both | 52.412 | 32.322 | 13.674 |

Mining shifts the aggregate result toward AP75 while reducing AP50, but no
large overall improvement follows from these fixed-budget changes. No new
sampling default was adopted.

## GPU timings and convergence

| Method | Total adaptation, seconds | SVM fit sum, seconds | Cached scoring of 500 images, seconds | Head, MiB |
|---|---:|---:|---:|---:|
| Torch baseline | 2.559 | **1.067** | 5.048 | 12.177 |
| Cap 8 | 2.520 | **0.938** | 4.878 | 11.621 |
| Mining | 5.300 | **2.154** | 5.448 | 16.190 |
| Both | 6.392 | **1.932** | 5.348 | 14.292 |

Adaptation includes support selection, GPU transfers, mining and all SVM fits.
It excludes model loading, cached-file loading, encoder feature extraction and
evaluation. The fit sum includes both fits for mining variants. Scoring uses
preloaded CPU caches and includes feature transfers and postprocessing. These
are not end-to-end detector timings.

One-off GPU fitter warmup/compilation is recorded separately (0.894 seconds in
the held-out run). Fixed training-row budgets do not guarantee equal scoring
cost: the harder negative bank changes the number of support vectors.

All 760 per-class fits across development and held-out stages converged at
the unchanged KKT tolerance. No iteration-limited classifier was evaluated.
Every held-out query had a classifier. The Torch baseline's largest held-out
AP difference from its saved sklearn control was **0.0013 mAP points**, with
identical label counts. Its mean mAP was 32.204, matching the CPU reference at
the reported precision, with a GPU fit time of about 1.1 seconds.

## Usage and reproduction

```python
flame = Flame(model, queries)  # default: full-kernel Torch fitter
cached = [tuple(t.cpu() for t in p) for p in flame.proposals(support_images)]
support = list(zip(cached, support_boxes, support_labels))

flame.fit_support(support, mining_rounds=1)
# Or cap positives only:
flame.fit_support(support, max_positives_per_object=8)
# Or the measured combined variant:
flame.fit_support(support, max_positives_per_object=8,
                  mining_rounds=1, mining_nms=0.7)
```

Calling `fit_support()` replaces the selected training bank with one rebuilt
from the supplied whole support pool. `flame.sampling_stats` reports timings,
label counts, changed hard negatives and solver diagnostics from that call.

```bash
.venv/bin/python tools/benchmark_flame_sampling.py --device cuda:0 \
  --reference-results data/flame_visual_dior/dev_normalized.json \
  --output data/flame_sampling_dior/dev200.json \
  --selection-output data/flame_sampling_dior/selection.json

.venv/bin/python tools/benchmark_flame_sampling.py --device cuda:0 \
  --reference-results data/flame_visual_dior/heldout500.json \
  --configs data/flame_sampling_dior/selection.json \
  --output data/flame_sampling_dior/heldout500.json
```

The JSON outputs preserve exact support/eval IDs, each configuration, per-class
AP, all convergence diagnostics, bank sizes and timing components. Raw data
and logs remain local ignored artifacts.

## Manual verification

Syntax/import checks passed. The uncapped sampler reproduced original selected
indices. A real cached-image smoke check used the default Torch backend,
replaced hard negatives without changing bank sizes, converged, preserved its
configuration on reload and reproduced margins exactly after save/load.
Legacy checkpoints retained their historical backend. No automated test files
were added or changed. The box-correction idea remains queued in
[the follow-up notes](flame-next-steps.md).
