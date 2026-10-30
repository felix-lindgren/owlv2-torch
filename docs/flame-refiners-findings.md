# GDA and Tip-Adapter on FLAME proposals

Experiment date: 2026-10-29. Frozen pretrained OWLv2-base, all 20 DIOR classes.

**Result:** neither auxiliary head gave a convincing accuracy improvement over
FLAME. GDA added 0.094 mAP points on average, with mixed changes across support
draws. Tip-Adapter added -0.039 points, also mixed. Standalone heads fitted much
faster but lost substantial accuracy. Both remain optional.

## What is wired up

`Flame.fit_refiner()` attaches a head to the usual `scores()` / `detect()` path.
It consumes existing labeled proposal embeddings and needs no extra image
encoder passes or backbone updates. It can supplement FLAME's SVM or replace
it. The prediction is:

```text
sigmoid(original_logit + alpha * svm_weight * svm_margin + weight * refiner_margin)
  * sigmoid(objectness)
```

The original proposal boxes, class logits, objectness and postprocessing are
preserved. Auxiliary heads use full-dimensional normalized embeddings and do
not depend on APE. Insufficient labels produce zero correction for that query.
Settings are saved with the examples and refitted on load. Old checkpoints
keep their behavior. `fit()` refreshes an attached head after new labels are
added. Loading a standalone refiner skips SVM fitting.

### GDA / LDA

Based on [A Hard-to-Beat Baseline for Training-free CLIP-based
Adaptation](https://arxiv.org/abs/2402.04087), this is a **binary detection
adaptation per query**. Each query has foreground/background means and a pooled
within-class covariance shared between those two labels. Different queries
have different covariance matrices. Priors are equal, so the many background
rows do not add a class-prior bias.

Covariance is shrunk toward its average variance times the identity matrix.
`gda_shrinkage="auto"` uses `D / (N - 1 + D)`, which gives the empirical Bayes
ridge direction from the paper. A numeric value in `(0, 1]` sets stronger or
weaker shrinkage explicitly. The resulting linear margin is scaled so the two
support class means score +1/-1; this makes fusion weights comparable across
queries. Only a D-by-C weight matrix and C biases are needed at inference.

### Tip-Adapter

Based on [Tip-Adapter](https://arxiv.org/abs/2207.09519) and the authors'
[cache-scoring implementation](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/main.py),
the kernel is `exp(-beta * (1 - cosine_similarity))`. The runner compared both
a positive-only cache and a foreground-minus-background cache. The latter is
an explicit detection adaptation, rather than the paper's original multiclass
image classifier.

Each query keeps at most `tip_cache_size` keys per polarity, selected
deterministically using `seed + query_index`. Cache responses are **averaged**
within each polarity because proposal counts differ strongly between classes.
With an equal number of keys, averaging differs from the paper's sum only by
the fusion scale. A signed cache subtracts the negative mean from the positive
mean. Affinity computation is chunked to bound temporary memory. Both selected
Tip configurations used 128 positive/negative keys, beta 5 and cache seed 0.

## Development and held-out protocol

This reuses the exact cached protocol from the
[APE experiment](flame-ape-findings.md):

- Five fully annotated train **images** per class, pooled into 100 support
  images per support draw. This includes all annotated instances, rather than
  five object instances per class.
- Development: 200 additional train images, disjoint from all three support
  draws. These additional labels select hyperparameters; this is not a strict
  few-shot protocol without labeled validation data.
- Selection uses support seed 0 only. Final evaluation uses the same 500 DIOR
  test images for support seeds 0, 1 and 2. Four configurations were selected
  on development data and fixed before held-out evaluation.
- All heads receive the same support examples as FLAME: IoU >= 0.7 positives,
  IoU < 0.3 negatives, 50 hard and 50 random negatives per image/query.
- FLAME keeps C=1, balanced SVM weights, gamma="scale", alpha=8. Every method
  uses the same objectness multiplication, per-patch class selection, NMS 0.5,
  top 100 detections and score threshold 0.0.
- The runner checks physical support/eval paths and reproduces the old FLAME
  baseline within 1e-6 mAP for every support seed before evaluating refiners.
- CPU sklearn SVM fitting; GDA/cache fitting and scoring on an RTX 5070 Ti.
  No backbone weights were tuned.

The development sweep had 60 configurations:

| Head | Settings | Fusion |
|---|---|---|
| GDA, 24 configurations | shrinkage auto / 0.1 / 0.5 / 0.9; weight 1 / 4 / 16 | SVM weight 0 or 1 |
| Tip, 36 configurations | beta 1 / 5 / 20; positive or signed cache; weight 8 / 32 / 128 | SVM weight 0 or 1 |

Best development settings by family:

| Family | Selected setting | Development mAP |
|---|---|---:|
| FLAME control | SVM only | 31.063 |
| GDA alone | shrinkage 0.9, weight 4 | 24.867 |
| FLAME + GDA | auto shrinkage, weight 1 | 30.893 |
| Tip alone | signed, beta 5, weight 32 | 14.486 |
| FLAME + Tip | signed, beta 5, weight 8 | 31.355 |

The best positive-only Tip blend reached 31.059 on development, essentially the
same as FLAME. Selection did not force a development improvement; the best
nonzero setting in each family was evaluated to show the actual tradeoff.

## Held-out results

All AP values are percentages. Differences are percentage points.

| Method | Seed 0 mAP | Seed 1 mAP | Seed 2 mAP | Mean mAP | Mean AP50 |
|---|---:|---:|---:|---:|---:|
| FLAME | 32.627 | 31.141 | 32.367 | **32.045** | 52.546 |
| GDA alone | 22.161 | 21.785 | 22.934 | 22.293 | 36.399 |
| FLAME + GDA | 32.727 | 31.449 | 32.241 | **32.139** | 52.617 |
| Tip alone | 12.349 | 11.011 | 12.898 | 12.086 | 20.666 |
| FLAME + Tip | 32.572 | 31.239 | 32.207 | **32.006** | 52.494 |

GDA's paired changes were +0.099, +0.308 and -0.126 points. Tip's were -0.056,
+0.098 and -0.160. Those small, inconsistent changes do not justify adopting
either blend as a general accuracy improvement. The standalone classifiers
improve on the 7.077 zero-shot mAP on this subset, but substantially trail FLAME.

Average component timings and storage:

| Component | Fit, seconds | Margins on 500 cached images, seconds | Auxiliary head, MiB |
|---|---:|---:|---:|
| FLAME SVM | 11.603 | 3.381 | — |
| Standalone GDA | 0.307 | 0.437 | 0.039 |
| GDA added to FLAME | 0.298 additional | 0.436 additional | 0.039 additional |
| Tip cache | 0.105 | 1.371 | 9.653 |

These timings exclude image encoder passes and disk loading. Margin timings
include feature transfers from preloaded CPU caches. Scoring/fusion/NMS adds
about 1.6 seconds per 500 images for each method. Blends retain the SVM's fit
and scoring costs as well as their auxiliary head's costs. They are not
end-to-end detector latency measurements.

GDA is a useful compact, fast-fitting alternative when accepting a large AP
reduction is reasonable. This experiment gives no strong reason to add either
head to FLAME for accuracy. The conclusion concerns these background-aware
adaptations on DIOR, not the original image-classification benchmarks.

## Usage

After adding support examples, select one blend:

```python
flame.fit().fit_refiner("gda", weight=1.0, gda_shrinkage="auto")
# Or:
flame.fit().fit_refiner("tip", weight=8.0, tip_beta=5.0,
                        tip_cache_size=128, tip_negative=True)
boxes, scores, labels = flame.detect(image)
```

To skip the SVM entirely, call this directly after adding examples:

```python
flame.fit_refiner("gda", svm_weight=0, weight=4.0, gda_shrinkage=0.9)
```

## Reproduction and checks

```bash
.venv/bin/python tools/benchmark_flame_refiners.py --device cuda:0 \
  --reference-results data/flame_ape_dior/dev_seed0.json \
  --output data/flame_refiners_dior/dev200.json \
  --selection-output data/flame_refiners_dior/selection.json

.venv/bin/python tools/benchmark_flame_refiners.py --device cuda:0 \
  --reference-results data/flame_ape_dior/heldout500.json \
  --configs data/flame_refiners_dior/selection.json \
  --output data/flame_refiners_dior/heldout500.json
```

The output preserves the exact support/eval IDs and every configuration's AP,
component timings, label counts and missing classes. All 20 queries had heads
in the final comparison. Both old and new raw result files are local artifacts
under `data/`, which is ignored by git.

Syntax/import checks passed. Manual smoke checks confirmed GDA's +1/-1 support
mean anchors, agreement between Tip's implementation and its direct kernel
formula (maximum difference 2.98e-8), finite outputs, exact score preservation
when the extra weight is zero, and exact save/load score round trips for a GDA
blend and standalone Tip head. Standalone checkpoint loading skipped SVMs.
