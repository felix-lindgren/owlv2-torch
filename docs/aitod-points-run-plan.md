# Run plan — OWLv2 as a point detector for tiny objects (AI-TOD)

Status (2026-10-20, evening): seeds and 2b done, nothing running. The point
loss alone ties the box loss, but **point loss + 2x2 point slots (P3) wins on
every seed**: 0.765 vs 0.748 mean, very tiny recall 0.63 -> 0.77. Slots with
the box loss (P4) lose, to duplicates (diagnosis below). P3's remaining false
positives are mostly background, not duplicates. 2a is not started.

| final point AP | seed 0 | seed 1 | seed 2 | mean | AR@0.5 | very tiny AR@0.5 |
|---|---|---|---|---|---|---|
| P0 box loss | 0.745 | 0.752 | 0.748 | 0.748 | 0.950 | 0.628 |
| P2 point loss + w/h | 0.749 | 0.750 | 0.732 | 0.744 | 0.946 | 0.615 |
| **P3 P2 + `--point-slot-grid 2`** | 0.755 | 0.775 | 0.764 | **0.765** | **0.976** | **0.768** |
| P4 P0 + `--point-slot-grid 2` | 0.740 | 0.740 | 0.727 | 0.736 | 0.968 | 0.743 |

Recall and very tiny recall are three-seed means.

## Where things stand

The xView part of AI-TOD-v2 (`/mnt/datasets/sat/xview_aitod_v2`) is used,
vehicles only. Subsets: `subsets/vehicle_train1000.json` and
`subsets/vehicle_val200.json`, seed-0 random images.

Every arm is scored with OKS point AP on box centres. OKS uses the GT box area
for scale and sigma 0.21, so a miss of about half the box side scores 0.5. The
evaluator is `OWLv2torch/torch_version/point_metrics.py`. On synthetic data it
matches pycocotools' single-keypoint OKS exactly.

The recipe is the same in every arm: base, vb6 + Q/V LoRA r8 on the first 6
blocks, MAL, 2000 steps, flips and rotations, `--no-scale-augment`, 1x zoom.
Launch with `tools/run_aitod_points.sh`.

| arm | step 500 | final | very tiny recall @ OKS 0.5 |
|---|---|---|---|
| zero-shot | — | 0.217 | 0.13 |
| P0 box loss | 0.681 | 0.745 | 0.63 |
| P1 `--point-loss` | 0.617 | 0.735 | 0.62 |
| P2 `--point-loss --point-size-weight 2.5` | 0.678 | **0.749** | 0.63 |

- **Centre-only loss destabilises the box head.** The w/h outputs get no
  gradient while the shared box-head layers train. At step 500 the predicted
  w/h median was 638 px and the top-scored patches threw their centres about
  45 patches away. It recovers by the end. A weak w/h L1
  (`--point-size-weight`) removes the dip, so always set it with
  `--point-loss`.
- **Matching is fine.** Hungarian matching with the OKS cost picks patches as
  local as the GIoU cost does.
- **Very tiny vehicles (<8 model px) sit at about 0.63 recall in every arm.**
  The cause is crowding, not size. P2's misses at OKS 0.5 on val200, split by
  whether another GT centre falls in the same 16 px patch:

  | size (model px) | n | recall | alone in patch | shares a patch | nearest GT > 1 patch |
  |---|---|---|---|---|---|
  | < 8 | 188 | 0.62 | 0.78 | **0.44** | 0.88 |
  | 8–12 | 2735 | 0.89 | 0.95 | **0.79** | 0.97 |
  | 12–16 | 4572 | 0.98 | 0.99 | 0.94 | 0.99 |
  | 16–24 | 2761 | 0.995 | 0.997 | 0.98 | 0.998 |

  64% of all misses share a patch with another GT. Isolated objects are found
  even at half a patch, so the features do see them. The limit is one output
  per token.

  The centre-ownership collision probe (`tools/dior_small_object_probe.py`,
  ceiling 0.90 at 1x) is too pessimistic: trained recall is 0.952, because
  neighbouring patches take some of the colliding objects. Use the breakdown
  above, not the probe, to judge crowding. It is per-GT greedy matching of the
  top 1500 points against centres, bucketed by patch-mates.

## Full train split (2026-10-20/21)

`subsets/vehicle_train_all.json`: all 3704 train-split images, vehicle boxes
only (190,288), no overlap with val200. Same 16 epochs as the 1000-image runs:
7400 steps, cosine over all of them, eval every 500 steps on val200.

| seed 0 | AP | AP@0.5 | AR@0.5 | very tiny AR@0.5 |
|---|---|---|---|---|
| P0 box loss, train1000 | 0.745 | 0.840 | 0.947 | 0.633 |
| P0 box loss, train_all | 0.781 | 0.869 | 0.955 | 0.676 |
| P3 point + slots, train1000 (3-seed mean 0.765) | 0.755 | 0.862 | 0.973 | 0.734 |
| **P3 point + slots, train_all** | **0.808** | 0.904 | 0.981 | 0.878 |

Slots and data stack: P3 − P0 is +0.027 on train_all against +0.010 on
train1000 (seed 0; the 3-seed train1000 gap is +0.017). P3 led P0 at every eval
from step 1500 on. Very tiny AR@0.5 rises from 0.68 to 0.88, so more data helps
the slots resolve crowded patches more than it helps the single-output head.
Medium AR@0.5 drops 0.942 → 0.916, but n = 154.

```sh
ARMS=P3 SEED=0 TRAIN_SUBSET=vehicle_train_all TAG=_trainall STEPS=7400 \
  EXTRA_ARGS="--eval-every-steps 500" tools/run_aitod_points.sh
```

**TP/FP breakdown (eval-only, seed 0 train_all, val200, OKS 0.5).** Greedy
matching in score order. FP kinds: dup = OKS >= 0.5 to an already-matched GT,
loc = best OKS 0.1-0.5, bg = best OKS < 0.1. P3 scores run higher than P0's,
so fixed thresholds mislead; compare at best F1 or equal recall.

| arm | thr | P | R | F1 | TP | FP | dup | loc | bg |
|---|---|---|---|---|---|---|---|---|---|
| P0 best F1 | 0.15 | 0.839 | 0.817 | 0.828 | 8676 | 1669 | 440 | 319 | 910 |
| P3 best F1 | 0.24 | 0.862 | 0.833 | 0.847 | 8852 | 1419 | 221 (16 sibling slots) | 288 | 910 |

FPs needed to reach a recall: 0.80: P3 1064 vs P0 1448; 0.90: 3047 vs 5598;
0.95: 8398 vs 54,116; 0.97: P3 21,371, P0 never gets there.

Slots do not add FPs. At best F1, 0.4% of empty patches have an active slot
(P0: 0.5%), and 99% of single-GT patches use at most one slot. Mean active
slots per token rise with GT in the patch (0.72 / 1.00 / 1.33 / 1.82 for
1 / 2 / 3 / 4+ GT), so the head is conservative: only 13% of two-GT patches
fire two slots at best F1. The slots halve duplicates; background FPs are the
same 910 in both arms and are now 64% of P3's FPs.

**The background FPs are mostly missing labels.** A contact sheet of P3's
910 background FPs at best F1, made by eye from 64 px crops: about 45 of the
top-scored 48 (scores 0.86-0.98) are clear vehicles without a GT point, often
next to labelled ones. In a random 48 from the rest, most above 0.5 are
vehicles and about half below 0.5. So most of the remaining precision gap is
val-label incompleteness, and it affects both arms. Train labels are
presumably just as incomplete, so these vehicles are trained as negatives.

Seed 1: P3 0.8068 (seed 0 0.8081). P0 seeds 1-2 and P3 seed 2 were stopped
for a shutdown; rerun the same launch commands and the script resumes them.

**Low-precision inference, no retraining (P3 seed 0, vision-encoder linears).**

| variant | AP | AP@0.5 | very tiny AR@0.5 |
|---|---|---|---|
| fp16 autocast (reference) | 0.8081 | 0.9044 | 0.878 |
| fp16 accumulation | 0.8080 | 0.9043 | 0.872 |
| FP8 rowwise, MLP only | 0.8060 | 0.9042 | 0.878 |
| FP8 rowwise, all linears | 0.8023 | 0.9013 | 0.888 |
| FP8 tensorwise, all linears | 0.8006 | 0.8997 | 0.872 |
| NVFP4, MLP only | 0.7782 | 0.8863 | 0.825 |
| NVFP4, all linears | 0.7526 | 0.8682 | 0.819 |

FP8 needs no QAT; NVFP4 loses 0.03-0.06 and would need it.

About 65 min on the 5070 Ti. Still open: an equal-compute arm (2000 steps on
train_all) to separate more data from longer training, and seeds 1–2 on
train_all.

## Step 1 — seeds (no code changes)

Done. P2 − P0 is +0.004, −0.002, −0.016: the point loss alone is no better
than the box loss and spreads wider across seeds. It pays off only together
with the slot head (2b).

```sh
SEED=1 tools/run_aitod_points.sh
SEED=2 tools/run_aitod_points.sh
```

Each seed is about 50 min on the 5070 Ti for all three arms. P1 can be dropped
(comment out its `run` line) to save a third. Decision: if P2 >= P0 on every
seed, P2 is the default for step 2. Otherwise use box loss with point eval;
the point eval stands either way.

Do not chain runs with `pgrep -f "prototype_train/train_text.py"` in a wait
loop. The pattern matches the loop's own shell command line, so it waits
forever; that left P2 idle after P1 on 2026-10-19.

## Step 2 — two competing fixes for crowding (both need code)

2a raises pixels per object, which fixes both size and crowding at about 4x
compute. 2b raises outputs per token, which fixes crowding only, at head-only
cost. Since crowding is most of the 1x misses, 2b may recover much of 2a's gain
far more cheaply. Run both with P2's loss and point eval, against the same 1x
P2 baseline.

### 2a — 2x zoom

Goal: put 10 px vehicles at about 24 model px, roughly 1.5 patches, in both
training and eval.

1. **Training crops.** Sample 400 px crops from the 800 px AI-TOD tiles and
   upsample them to 960 (2.4x). Centre the crops on objects so steps are not
   wasted on empty ground. Nothing does this yet. `AugmentedDetectionDataset`
   only has `RandomIoUCrop` (`min_scale` 0.5) behind `--scale-augment`, and
   that flag also turns on `RandomZoomOut`, which shrinks objects. Add a crop
   option without zoom-out.
2. **Tiled point eval.** Point AP exists only in the in-training `coco_eval`
   (`--point-eval`). It has no tiling. `tools/ovd_eval.py` tiles
   (`--tile-size`, `generate_tiles`, `merge_tiled_detections`) but scores
   only box mAP, and its merge uses box IoU. Add point AP there. Merge
   detections that overlapping tiles produce twice with point NMS by distance
   (or OKS) instead of box IoU, and keep `--top-k`/maxDets at 1500.
   Size buckets should then be in source pixels, not model pixels.
3. **Arms:** P0 and P2 at 2x, same steps per image area if affordable.
   Training is GEMM-bound, so 2x zoom costs about 4x per epoch over the same
   image area. Score at 1x and at 2x tiled to separate the effect of the
   training crops from the effect of the test-time zoom.

### 2b — multi-slot point head (no zoom)

Each ViT token emits K = 4 point slots in a 2x2 layout instead of one
prediction. That is a stride-8 output, 120x120 = 14,400 queries at 960 input,
from the unchanged backbone. Points make this cheap: a slot needs only a
score, a centre offset and an embedding for the class head, with no box and no
IoU matching. The w/h auxiliary stays on as in P2.

1. **Slot embeddings.** A linear `768 -> 4 x 768` on each token's final
   embedding (about 2.4M parameters), initialised as the identity tiled four
   times, so at step 0 every slot equals its parent token and pretrained
   behaviour is preserved. The class, objectness and box heads stay shared
   and run on all 14,400 slot embeddings.
2. **Anchors.** Compute `box_bias` for the 120 grid, so each slot's centre
   prior sits at its sub-patch centre. The w/h prior stays at one parent patch
   (`1/60`) or moves to `1/120`; pick one and note it.
3. **Matching and loss.** Hungarian runs over 4x the queries. The CPU cost
   rises with it, and the 1,499-GT train image is the worst case, so measure
   step time. MAL is dense over all slots and normalised by matched count, so
   it needs no reweighting.
4. **Eval.** Unchanged: points are slot centres, and `--eval-top-k 1500` still
   applies.

Upper bound from labels alone: the 1x collision probe on a 120 grid should
match its 2x-zoom row (0.993 overall, 0.97 very_tiny), as the cell size is the
same. As with the probe at 1x, trained recall can beat it.

Risks:
- The slots of one token share the same 16 px of input. A slot head can split
  objects only as finely as that token's embedding encodes them. The
  isolated-object recall above suggests it encodes enough, but it is
  unverified.
- With 14,400 queries, duplicate points on one object become more likely.
  Watch precision (AP vs AR), not just recall.

**Implementation (2026-10-20).** `--point-slot-grid 2` (`OwlV2.enable_point_slots`).
The slots are ordered as a row-major 120x120 grid, so the loss, matching and
`coco_eval` are unchanged. Checkpoints store `point_slot_grid` and recreate the
layer on load. `--point-slot-learning-rate` defaults to the head LR.
- **Centre prior:** the parent's prior shifted by ±1/4 patch (±4 px) to each
  sub-cell centre, not `compute_box_bias(120)`. OWL's prior sits at the patch
  corner, so the plain 120-grid bias would have put three of the four slots up
  to 8 px off what the pretrained head predicts. **Size prior:** one parent
  patch (1/60).
- **Step 0:** scores and w/h equal the parent token's exactly. Each object
  therefore gets four equal-score points ±4 px apart. Zero-shot point AP falls
  from 0.217 to **0.054**, while AR@0.5 rises from 0.579 to 0.626. Training
  has to learn to suppress three of the four. A 60-step smoke run
  (batch 4, 3 val batches) gave 0.434 vs 0.502 without slots.
- **Cost:** Hungarian on random costs at 1,499 GT takes 0.27 s (0.07 s at 3600
  queries); median train image has 22 GT. Step time on the 3060 is
  unchanged at batch 4. Batch 8 OOMs on the 3060 (12 GB) without
  `--grad-checkpointing`; on it, 6 GB at 1.9 s/step.
- If P3 lags P2, the first thing to try is breaking the step-0 symmetry
  (e.g. a per-slot score offset) or a higher `--point-slot-learning-rate`.

**Result (three seeds, table at the top).**
- **P3 beats P0 on every seed** (+0.010, +0.023, +0.016) and P2 on every seed.
- **P4 loses to P0 on every seed** (−0.005, −0.012, −0.020), although its
  recall rises almost as much (AR@0.5 0.968, very tiny 0.74). Its AP@0.5 falls
  (0.826 vs 0.844), so with the box loss the extra slots cost precision. The
  cause is not diagnosed. Duplicates that the IoU quality target does not
  suppress are the first suspect.
- P3's three seeds spread 0.755–0.775, wider than P0's 0.745–0.752.

**Diagnosis (eval-only, all 12 final checkpoints, val200, 3-seed means).**
The script reproduces every logged AP exactly. It is not committed; it lived in
the session scratchpad as `slot_diag.py`. Note that the trainer evaluates at
`--confidence-threshold 0.001`, not `coco_eval`'s 0.05 default; at 0.05, P3's
AP reads 0.718.

Recall at OKS 0.5, alone in its 16 px patch | sharing it with another GT:

| size (model px) | n | P0 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| < 8 | 101 \| 87 | 0.78 \| 0.46 | 0.78 \| 0.42 | **0.88 \| 0.64** | 0.81 \| 0.68 |
| 8–12 | 1682 \| 1053 | 0.95 \| 0.78 | 0.94 \| 0.76 | **0.97 \| 0.92** | 0.96 \| 0.89 |
| 12–16 | 3658 \| 914 | 0.99 \| 0.93 | 0.99 \| 0.93 | 0.99 \| 0.98 | 0.99 \| 0.97 |
| ≥ 24 | 350 \| 16 | 0.95 \| 1.0 | 0.95 \| 1.0 | 0.94 \| 1.0 | 0.93 \| 0.96 |

- **Crowding confirmed.** Slots lift the shares-a-patch column most
  (8–12 px: 0.78 -> 0.92). Isolated sub-8 px objects gain too (0.78 -> 0.88).
  The ≥ 24 px alone bucket dips slightly in both slot arms (0.95 -> 0.93,
  n = 350); watch it.
- **P4 loses to duplicates.** With as many detections as GT, P4 has 1033
  duplicate FPs (OKS ≥ 0.5 to an already-matched GT) vs 438 for P3 and 688 for
  P0, and 366 fewer TPs than P3. Removing all duplicates lifts P4's AP@0.5 from
  0.826 to 0.859, level with P0's oracle 0.863. Why the box loss leaves them is
  still a hypothesis (IoU barely separates slots 8 px apart; OKS does).
- **P3 has little duplicate headroom.** The duplicate oracle gives +0.015
  AP@0.5; at the same operating point there are 1583 background FPs vs 438
  duplicates. Only 22% of P3's duplicates are sibling slots of the same token;
  the rest are neighbouring tokens, as at 1x.
- **Point NMS** (square neighbourhood, before the 1500 cap), AP .50:.95:

  | arm | none | r = 2 | r = 3 | r = 4 | r = 6 | r = 8 |
  |---|---|---|---|---|---|---|
  | P0 | 0.748 | **0.753** | 0.753 | 0.751 | 0.743 | 0.722 |
  | P2 | 0.744 | 0.745 | 0.744 | 0.742 | 0.734 | 0.712 |
  | P3 | 0.765 | **0.767** | 0.767 | 0.765 | 0.756 | 0.728 |
  | P4 | 0.736 | 0.745 | **0.746** | 0.744 | 0.733 | 0.713 |

  2–3 px NMS is a small free gain; ≥ 6 px suppresses real crowded neighbours.
  It narrows but does not close P4's gap. P3 + NMS (0.767) still leads P0 +
  NMS (0.753) by 0.014.
- Launch with `ARMS="P3 P4" tools/run_aitod_points.sh`. On the 5070 Ti the slot
  arms use 13 GB at batch 8 and are ~12% slower per step; the 3060 needs
  `EXTRA_ARGS=--grad-checkpointing`. P3 seed 0 and P4 seed 0 ran on the 3060.

### What to watch (2a and 2b)

Very_tiny and tiny recall, split alone vs shares-a-patch as in the table
above: 2b should mainly lift the shares-a-patch column. The size bucket edges
are in model pixels in the in-training eval; for AI-TOD's 800 px tiles,
multiply source pixels by 1.2.

## Known limits

- `--point-loss` is not implemented for `--batched-loss`/`--compile-step` and
  raises if combined. That path is needed anyway for images with more than 128
  boxes (108 of the 1000 train images).
- The in-training box evaluator (torchmetrics) caps at 100 detections, so box
  mAP on AI-TOD is capped. Use `--point-eval` or `ovd_eval.py
  --eval-max-detections 1500`.
- Results are not comparable to published AI-TOD box AP.
- Point-only data is not supported yet. Mixing in point-labelled datasets for
  an open-vocab remote-sensing model needs two things: the w/h loss masked per
  annotation, and an OKS scale for GT without a box (a per-class size prior or
  a fixed pixel radius).
