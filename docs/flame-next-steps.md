# FLAME follow-up ideas

Recorded 2026-10-29 at the user's request. These follow up the
[comparison of projected, pre-projection visual, and combined SVM features](flame-visual-features-findings.md).

## 2. Improve support sampling and mine negatives after fitting

Status: implemented and measured; see the [sampling comparison](flame-sampling-findings.md).
Mean held-out changes were +0.101 mAP points for capped positives, +0.086 for
one mining pass and +0.074 for their development-selected combination. Each
variant lost on one support draw, so no new sampling default was adopted.

The ordinary sampler selects hard negatives using original zero-shot scores.
The new optional second pass mines the adapted FLAME classifier's remaining
high-scoring negatives from the same fully annotated support images and refits
once. Cached proposal embeddings avoid more encoder passes or annotation.

The experiment also capped positives per ground-truth object and compared
spatially diverse mined negatives. Large objects can contribute many similar
patches, while class balancing alone does not balance instances. Random easy
negatives were retained unchanged, and mined hard negatives replaced existing
ones at a fixed budget. The cap reduces instance imbalance without imposing
exact equal instance weights.

Mining, positive caps and their combination were isolated while retaining
the original model, support images, feature source, C, alpha and postprocessing.
New experiments use the full-kernel Torch fitter as requested. Annotation
budgets are in images and instances, rather than correlated proposal rows.

## 3. Add a small, regularized box-correction head

Status: implemented and adopted as the default for new boxed-support fits
on 2026-10-30 at the user's request; see the
[box-head experiment](flame-boxes-findings.md) for development selection,
held-out evaluation and usage.
Mean held-out mAP improved by 0.327 points, AP75 by 1.283 and small-object AP
by 0.535, with positive changes in all three support draws. The head has
3,076 parameters. Set `box_refinement=False` for original boxes; yes/no-only
support and older checkpoints retain their previous behavior.

Fit a small residual box-coordinate regressor on cached support features and
ground-truth boxes. Start from zero correction and penalize large changes.
A shared/class-agnostic correction is a useful starting point with few images.
The backbone and original box head remain frozen; the residual can change box
locations/sizes before NMS. This requires boxed support, not yes/no labels alone.

The motivation is a read-only diagnostic of the existing 500-image DIOR cache
(3,961 non-crowd, non-ignored ground-truth boxes):

| Ground-truth size | Box count | Any proposal at IoU >= 0.50 | Any proposal at IoU >= 0.75 |
|---|---:|---:|---:|
| All | 3,961 | 91.24% | 69.15% |
| Small (< 32 squared pixels) | 1,924 | 83.16% | 47.92% |
| Medium | 1,286 | 98.29% | 87.40% |
| Large (>= 96 squared pixels) | 751 | 99.87% | 92.28% |

These are independent best-proposal coverage figures, ignoring class labels
and one-to-one matching; they are not AP. They establish a localization limit
shared by the unchanged-box APE/GDA/Tip comparisons, particularly for small
objects. They do not establish that localization dominates all remaining errors.
The figures use the earlier `base_default_float32_v1` cache recorded by
`data/flame_refiners_dior/heldout500.json`.

The implementation uses matched support proposals to learn normalized box
offsets, with ridge regularization, equal object weights and bounded
corrections. It is evaluated on separate images, including AP75 and small
objects as well as overall AP. The original detector and classifier scores
stay frozen.
