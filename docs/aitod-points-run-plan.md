# Run plan — OWLv2 as a point detector for tiny objects (AI-TOD)

Status (2026-10-20): concept test done, one seed. Nothing is running. Next up:
seeds, then 2x zoom vs a multi-slot point head, as competing arms.

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

P2 vs P0 is +0.004 on one seed and 200 val images, inside plausible seed noise.

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
