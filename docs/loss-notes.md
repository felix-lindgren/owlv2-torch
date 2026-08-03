# Detection loss: current state and DEIM/D-FINE options

Analysis 2026-08-02, against `OWLv2torch/torch_version/text_loss.py` and
`OWLv2torch/torch_version/loss.py`, using Fashionpedia (46 classes, 45,623 train
images, 7.3 anns/image) and the live run's MLflow metrics.

## What the current loss is

DETR-style: Hungarian bipartite one-to-one matching (`scipy.linear_sum_assignment`)
with cost `1·(−sigmoid p) + 5·L1 + 2·(1−GIoU)`, then sigmoid-focal classification on
matched patches, hard-negative-mined focal on background, L1 + GIoU on matched boxes,
and BCE objectness on positives + mined negatives.

**But OWLv2 is not a DETR.** There is no decoder and there are no learned object
queries — each of the 5184 patch tokens (72×72 at 1008px/patch14) is an independent
prediction, and the DETR loss is applied over that dense grid. Two consequences drive
everything below:

1. **Supervision is extremely sparse.** 7.3 positives out of 5184 predictions =
   **0.14%**. DETR-with-300-queries is 2.3%. Sparsity is ~16x worse here, so the
   problem DEIM targets is *more* acute for OWLv2 than for the models DEIM was
   designed on.
2. **There is no NMS anywhere in eval** (verified — `coco_eval` thresholds and
   takes top-k). One-to-one matching is load-bearing; you cannot switch to a
   one-to-many assigner (SimOTA/ATSS/TAL) without adding NMS. This is exactly why
   DEIM's approach — *keep* O2O, increase density via augmentation — is the right
   family for this codebase.

## Finding 1: classification is 0.8% of the loss (fix this first)

`sigmoid_focal_loss(..., reduction="mean")` averages over `[num_matched, num_classes]`,
so it divides by `num_classes`. Verified numerically: `sum/num_boxes` is exactly
**46.0x** larger than `reduction="mean"` for 46 classes.

The `λ_cls=1, λ_l1=5, λ_giou=2` weights are lifted from DETR/D-FINE, which normalize
classification as `sum / num_boxes`. Under `mean`, the effective λ_cls is ~1/46.
Measured weighted contributions from the live run:

| Component | raw | λ | weighted | share |
|---|---:|---:|---:|---:|
| `L_cls` | 0.0036 | 1.0 | 0.0036 | **0.8%** |
| `L_l1` | 0.0075 | 5.0 | 0.0374 | 7.8% |
| `L_giou` | 0.1860 | 2.0 | 0.3719 | **77.7%** |
| `L_obj` | 0.1319 | 0.5 | 0.0660 | 13.8% |

`class_head` is the primary trainable module for adapting to the Fashionpedia
vocabulary, and it receives under 1% of the gradient. **mAP from runs before this is
fixed is confounded.**

Fix: sum over the class dimension, average over predictions — i.e. undo the
`num_classes` division while leaving every other property of the loss alone.

**The prototype trainer has the same pattern**, unfixed: `loss.py:255`, `:279`, `:297`
in `compute_prototype_losses` also use `reduction="mean"`, and there the divisor is
`num_classes × prototypes_per_class`, so the dilution is worse. Left alone
deliberately — it is a separate trainer with separately tuned λ, and changing it
would invalidate existing prototype results.

## Finding 2: hard-negative mining is redundant with focal loss

With `--negative-ratio 20` and 7.3 positives, `_hard_negative_indices` keeps
`min(7.3×20, 512) ≈ 146` negatives — so **~97% of background patches get zero
classification gradient**. Focal loss exists specifically to make mining unnecessary:
its `p^γ` term already down-weights easy negatives. Running both applies the
down-weighting twice and throws away most of the (cheap) signal.

There is also a third, independent reweighting: `L_cls_pos` and `L_cls_neg` are each
meaned separately and then added, pinning the positive:negative ratio at 1:1
regardless of actual counts.

Computing focal over the full `5184×46` grid is ~238k elements — negligible next to a
2.2 s vision forward. There is no performance reason to mine.

## Option A: MAL (DEIM) — recommended, subsumes findings 1 and 2

From `engine/deim/deim_criterion.py` in the DEIM repo, with `gamma: 1.5`:

```python
target_score = (iou_of_matched * onehot).pow(gamma)      # positives target IoU^γ, not 1
pred_score   = sigmoid(logits).detach()
weight       = pred_score.pow(gamma) * (1 - onehot) + onehot
loss = F.binary_cross_entropy_with_logits(logits, target_score, weight=weight,
                                          reduction='none')
loss = loss.sum() / num_boxes    # their loss.mean(1).sum()*C/num_boxes is identical
```

For reference, VFL differs only in `weight = α·p^γ·(1−onehot) + target_score` — VFL
weights positives by `q=IoU`, so a poorly-localized match contributes almost nothing.
MAL gives every positive weight 1 and puts the quality into the *target* (`q^γ`).

Why it fits OWLv2 specifically:

- **One term over all 5184 predictions** replaces the pos/neg split, the mining, and
  the broken normalization in a single change.
- **The class score learns to encode localization quality.** This matters more here
  than in DEIM's setting, because eval ranks by `sigmoid(cls)·sigmoid(objectness)`
  with no NMS — ranking quality *is* the metric.
- **MAL's low-quality-match behaviour is the one you want.** Early in fine-tuning
  OWLv2's matched patches have middling IoU, which is precisely where VFL collapses
  the gradient and MAL does not.

Two caveats:

- **Retune the background scale.** `sum/num_boxes` over a 5184-prediction grid puts
  ~17x more background elements per positive than DETR's 300 queries. Expect to need
  a separate background weight or a lower λ; don't assume DEIM's λ transfers unchanged.
- **MAL overlaps with the objectness head.** Both would then encode box quality.
  Consider lowering `--lambda-objectness` or dropping the objectness loss when
  adopting MAL, rather than training two heads on correlated targets.

## Option B: Dense O2O — best convergence-per-FLOP given the compute budget

The strongest argument is a scheduling one: **a training step costs 2.9 s regardless
of how many objects are in the image.** The vision tower is 78% of the step and is
completely indifferent to target count. Right now you pay 2.9 s for 8×7.3 = 58
positives. A 2×2 mosaic gives ~230 positives for the *same* 2.9 s — a 4x increase in
supervision per unit compute, which is far better ROI than any kernel work
(see `docs/optimization-notes.md`, where the ceiling is ~1.15x).

DEIM's recipe (`configs/base/deim.yml`): `Mosaic` with `mosaic_prob: 0.5` and
`scaling_range: [0.5, 1.5]`, plus `mixup_prob: 0.5` in the collate_fn, enabled for
epochs 4–29 and **disabled for the final 8 epochs** (`no_aug_epoch: 8`). The
turn-it-off tail is not optional — it is what recovers clean-distribution accuracy.

Two adaptations needed here:

- **Small objects.** Mosaic halves linear object size. Fashionpedia's p10 box side is
  27 px and 12.3% of boxes have `sqrt(area) < 32 px`; halving pushes those under the
  14 px patch size, and the fine-grained classes (bead, sequin, rivet, zipper,
  applique) are exactly the small ones. Bias `scaling_range` upward, or use a mosaic
  that crops rather than downscales.
- **Schedule is step-based, not epoch-based.** `--max-steps 2200` × bs 8 = 17,600
  samples against 45,623 train images — under one epoch, so every image is seen at
  most once. Express DEIM's schedule in steps (e.g. mosaic for the first 60% of
  steps, off for the last 40%).

The existing `AugmentedDetectionDataset` already has DEIM's other ops
(`RandomZoomOut`, `RandomIoUCrop`, `SanitizeBoundingBoxes`, `RandomHorizontalFlip`,
photometric) — Mosaic and Mixup are the only missing pieces.

## Option C: focal-style matching cost — small consistency fix

`hungarian_match_text` uses `−sigmoid(logits)[:, target_labels]` as the class cost.
DETR/RT-DETR/DEIM use the focal-style cost so matching is consistent with the loss:

```python
neg = (1-α)·p^γ·(−log(1−p));  pos = α·(1−p)^γ·(−log p);  cost_class = pos − neg
```

Cheap, and worth doing alongside a focal/MAL classification loss.

## What does not transfer from D-FINE

- **FDR (Fine-grained Distribution Refinement)** replaces direct box regression with a
  distribution over the four edges' residuals, refined *iteratively across decoder
  layers*. OWLv2 has no decoder, so the iterative half is inapplicable. You could
  swap `box_head`'s 4 outputs for `4×(2n+1)` bins, but that discards a box head
  pretrained on far more data than you have — and box losses already dominate your
  gradient, so localization is not the obvious bottleneck.
- **GO-LSD** distills the final layer's distribution into earlier layers. It requires
  a multi-layer decoder. Not applicable at all.
- The `loss_fgl: 0.15, loss_ddf: 1.5` entries in DEIM's weight_dict are D-FINE's
  distribution losses; they only exist if you adopt FDR.
- **RT-DETR IoU-aware query selection** and **DN-DETR denoising queries** both need
  object queries / a decoder. Not applicable.

## What has been implemented (2026-08-02)

Findings 1 and 2 and Option A are in. `--class-loss {focal,vfl,mal}` on
`train_text.py`, with `--class-loss-gamma` (default 1.5) and `--class-loss-alpha`
(default: unscaled for MAL, 0.2 for VFL). Box and objectness terms are byte-identical
across the three, so runs are directly comparable. Negative mining still governs the
objectness term under all variants; it is bypassed for classification under mal/vfl.

Measured over 8 real Fashionpedia batches through zero-shot OWLv2-large:

| variant | `L_cls` | pos | neg | cls share | giou share |
|---|---:|---:|---:|---:|---:|
| `focal` (fixed) | 0.7847 | 0.7807 | 0.0040 | 58.2% | 30.5% |
| `mal` | 2.3884 | 2.1860 | 0.2023 | 80.9% | 13.9% |
| `vfl` | 2.0054 | 1.9648 | 0.0405 | 78.0% | 16.0% |

Compare against the 0.8% classification share before the fix. These are step-0
values, so the balance shifts as training converges.

The "retune the background scale" caveat above turned out **not** to bite: with real
OWLv2 logits the MAL background term is 0.20, not the runaway value that synthetic
logits suggest. A real detector's background scores are well enough calibrated that
`sum/num_boxes` over the dense grid behaves, exactly as it does for RetinaNet.

Gradients confirmed distinct — `class_head.dense0.weight` gradient cosine similarity
is 0.92 (focal vs mal), 0.93 (focal vs vfl), 0.99 (mal vs vfl).

### Gotcha found while verifying

`torch.amp.GradScaler`'s default `init_scale=65536` **skips the first 6 optimizer
steps** of a run while it halves down to a workable scale (measured; 2 steps with
`init_scale=1024`). Those steps still advance `global_step` and the LR scheduler, so
a few percent of `--warmup-steps` is spent on no-ops. Negligible in wall clock
(~17 s), but it means short smoke runs show *zero* weight change and look broken.

## Suggested order

1. ~~Fix the `reduction="mean"` normalization~~ — done.
2. ~~Adopt MAL~~ — done, as `--class-loss mal`. Still worth reconsidering
   λ_objectness, since MAL and the objectness head now both encode box quality.
3. Add Mosaic/Mixup Dense O2O with a step-based on/off schedule.
4. Switch the matcher to the focal-style class cost.
5. Skip FDR and GO-LSD.
