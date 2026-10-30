# APE feature selection for FLAME

Experiment date: 2026-10-29. Frozen pretrained OWLv2-base, DIOR, sklearn RBF SVM.
APE is optional; ordinary FLAME remains the default.

**Result:** the development-selected APE setting did not improve held-out
accuracy. It lost 0.269 mAP points on average and scored lower on all three
support draws. Its smaller classifier fitted about 42% faster. Keep it as an
optional compression experiment, rather than an accuracy default.

## What was implemented

This adapts APE's **prior refinement / channel selection** to FLAME. It does not
implement APE's cache classifier or APE-T. The references are the
[paper](https://arxiv.org/abs/2304.01195) and the authors'
[`cal_criterion`](https://github.com/yangyangyang127/APE/blob/main/utils.py).

For each channel, the criterion rewards variance across normalized category
text embeddings and penalizes average similarity between different classes.
The similarity term uses either text alone (`ape_source="text"`) or text plus
positive support proposals (`"text_support"`). Class means compute the pairwise
similarity sum without explicitly constructing all pairs, allowing unequal
positive counts. The variance term uses text in both cases, as in the reference
implementation. Background remains part of SVM fitting, rather than an extra
category for feature ranking. This criterion requires at least two queries.

The selected channels go through FLAME's existing standardization and RBF SVM.
The original text logits, objectness, proposal boxes, labels, C, alpha and
postprocessing are preserved. `ape_normalize=True` additionally L2-normalizes
the selected vectors before standardization; both settings were compared.
All support embeddings remain full-dimensional so selection can be refitted
after adding examples. Saved checkpoints preserve the APE options and refit
on load; older checkpoints retain ordinary FLAME.

There are no additional backbone passes at inference. Classifier fit and score
time can change with the channel and support-vector counts.

## Protocol

- All 20 DIOR categories, pretrained `owlv2-base`, queries
  `a satellite photo of a {name}`.
- Five **images** per class, pooled into 100 images for each support seed
  0, 1 and 2. Every annotated instance in these images is available; this is
  not five object instances per class. Support selection matches the existing
  cached benchmark, including its seed-0 control.
- Development: 200 cached DIOR **train** images, excluding the union of all
  three support draws. A fixed draw (20261029) takes up to eight cached images
  per class, then fills to 200 from the remaining eligible cached images.
- Held-out evaluation: the existing 500-image DIOR **test** subset,
  `data/flame_features_dior/test_mini500.json`. It was not used to choose APE
  settings. The benchmark checks physical image paths for support/eval overlap.
- Same cached proposals and exactly the same sampled positive/negative rows
  within each support draw for all variants. FLAME uses IoU >= 0.7 positives,
  IoU < 0.3 negatives, 50 hard plus 50 random negatives per image/class,
  balanced SVC weights, `C=1`, `gamma="scale"`, and `alpha=8`.
- Per-patch best class, per-class NMS at 0.5, top 100 detections per image,
  score threshold 0.0, unchanged from the cached benchmark.
- Fit uses CPU sklearn; proposal extraction and scoring use an RTX 5070 Ti.
  Score timings include cache I/O, transfers and postprocessing. They exclude
  backbone inference and should not be quoted as end-to-end detector latency.

## Held-out results

The configuration chosen on development data was 384 channels, text-only
ranking, similarity weight 0.9, without additional L2 normalization. Each row
below uses the same 500 held-out images and paired support examples.

| Support seed | FLAME mAP | APE + FLAME mAP | Difference |
|---|---:|---:|---:|
| 0 | 32.627 | 32.524 | -0.104 |
| 1 | 31.141 | 31.001 | -0.140 |
| 2 | 32.367 | 31.803 | -0.564 |
| Mean | **32.045** | **31.776** | **-0.269** |

| Metric, averaged over support draws | FLAME | APE + FLAME |
|---|---:|---:|
| AP50 | 52.546 | 52.192 |
| AP75 | 31.829 | 31.708 |
| SVM fit, seconds | 11.512 | 6.676 |
| Cached scoring of 500 images, seconds | 5.698 | 5.353 |
| Classifier storage, MiB | 12.142 | 9.149 |

Zero-shot mAP on this subset was 7.077. These are subset results, not the full
DIOR split numbers in the README. Fit timings include channel selection but
exclude backbone extraction. The cached scoring measurement includes I/O and
postprocessing; the roughly 6% reduction does not establish a detector-level
speedup.

There is no held-out accuracy gain to adopt. A plausible explanation is that
APE's multiclass semantic channel criterion discards information useful for
FLAME's foreground/background discrimination, while the existing standardized
RBF SVM already handles the full embedding well. That mechanism was not
isolated by this experiment. The conclusion applies to this feature-selection
integration on DIOR, not to APE's full cache classifier or all target domains.

## Development selection

The development comparison used support seed 0 only. It evaluated 24 reduced
feature settings: 128/256/384 channels, text or text-plus-support ranking,
weight 0.9 without normalization, and weights 0.1/0.5/0.9 with normalization.
Two 512-channel identity controls reproduced the baseline with **zero sampled
margin difference** and identical AP.

| Development setting | mAP@[.5:.95] | Difference from FLAME |
|---|---:|---:|
| Ordinary FLAME, 512 channels | 31.063 | — |
| Selected: 384 channels, text, weight 0.9, no normalization | 31.245 | +0.183 |
| 384 channels, text, weight 0.5, normalization | 31.239 | +0.176 |
| 256 channels, text plus support, weight 0.9, no normalization | 30.895 | -0.168 |

Values are percentages; differences are percentage points. The selected setting
was locked in `data/flame_ape_dior/selection.json` before running the held-out
comparison. The small development gain alone is insufficient evidence of an
accuracy improvement.

## Reproduction

Local manifests and full per-class results are under `data/flame_ape_dior/`:
`support_5.json`, `dev200.json`, `dev_seed0.json`, `dev_l2_seed0.json`,
`selection.json`, and `heldout500.json`. The benchmark output records the exact
support/eval image IDs, cache directories, selected dimensions and all settings.

```bash
APE_DIOR_ARGS=(
  --device cuda:0 --dataset-label dior
  --annotations /mnt/datasets/sat/dior_coco/train/annotations.json
  --support-image-root /mnt/datasets/sat/dior_coco/train/images
  --query-template 'a satellite photo of a {name}'
  --support-manifest data/flame_ape_dior/support_5.json
  --backends sklearn --cache-dir data/flame_features_dior
)

.venv/bin/python tools/benchmark_flame_features.py "${APE_DIOR_ARGS[@]}" \
  --eval-image-root /mnt/datasets/sat/dior_coco/train/images \
  --eval-annotations data/flame_ape_dior/dev200.json --seeds 0 \
  --ape-channels 128 256 384 512 --ape-sources text_support text \
  --output data/flame_ape_dior/dev_seed0.json

.venv/bin/python tools/benchmark_flame_features.py "${APE_DIOR_ARGS[@]}" \
  --eval-image-root /mnt/datasets/sat/dior_coco/train/images \
  --eval-annotations data/flame_ape_dior/dev200.json --seeds 0 \
  --ape-channels 128 256 384 --ape-sources text_support text \
  --ape-similarity-weights 0.1 0.5 0.9 --ape-normalize \
  --reference-results data/flame_ape_dior/dev_seed0.json \
  --output data/flame_ape_dior/dev_l2_seed0.json

.venv/bin/python tools/benchmark_flame_features.py "${APE_DIOR_ARGS[@]}" \
  --eval-image-root /mnt/datasets/sat/dior_coco/test/images \
  --eval-annotations data/flame_features_dior/test_mini500.json --seeds 0 1 2 \
  --ape-channels 384 --ape-sources text --ape-similarity-weights 0.9 \
  --output data/flame_ape_dior/heldout500.json
```

## Verification

Syntax/import checks passed. A manual calculation agreed with the explicit
class-pair criterion. A save/load smoke check on real cached support proposals
reproduced the APE margins exactly, kept scores finite, and confirmed that a
checkpoint without APE fields loads with feature selection disabled.
