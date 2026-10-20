# Run plan — OWLv2 as a point detector for tiny objects (AI-TOD)

Status (2026-10-20, evening): seeds and 2b done, nothing running. The point
loss alone ties the box loss, but **point loss + 2x2 point slots (P3) wins on
every seed**: 0.765 vs 0.748 mean, very tiny recall 0.63 -> 0.77. Slots with
the box loss (P4) lose. Next: an eval-only diagnosis of the AP-vs-recall gap
(duplicates vs ranking, point NMS). 2a is not started.

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
