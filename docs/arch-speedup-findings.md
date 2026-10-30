# Findings — architecture speedups for OWLv2-base (AI-TOD P3, COCO)

Date: 2026-10-27. Question: the kernels are at roofline (see
`docs/optimization-notes.md` and the 5070 Ti profiles), so which changes to the
model itself cut work while keeping accuracy, for training and for inference?
And which of them combine?

Reference model for everything below: **P3** (point loss + 2x2 slots), hybrid
(vb6 + first-six Q/V LoRA r8), AI-TOD vehicles `train_all`, 7400 steps, seed 0,
`--fp16-accumulation --attention-bwd-fp16-accumulation`, retrained today:
**0.8036 point AP** on val200 (earlier seeds with the same flags: 0.8030 / 0.8074).
General-domain reference: zero-shot base on COCO mini500, **0.4960 box mAP** in the
probe harness (0.4966 with `tools/ovd_eval.py`'s standard metric, used in the
zero-shot section).
Code, checkpoints and logs are listed at the end. Nothing here is in MLflow.

## Summary

Learned token pruning is the lever. A small scorer, trained with the model,
picks the tokens the later blocks still process, and a per-image threshold
decides how many. Spending part of the saving on input resolution makes the
model both faster and more accurate.

| configuration | point AP | Δ vs ref | tower B=1 | B=8 padded | B=8 packed | training |
|---|---|---|---|---|---|---|
| reference, 960 px | 0.8036 | — | 1.00x | 1.00x | | 1.00x |
| **1120 px, prune after block 3, threshold −3 (18% kept)** | **0.8196** | **+0.016** | **1.75x** | 1.30x | 1.99x | 1.45x |
| 1120 px, prune after block 3, threshold −4 (30% kept) | 0.8232 | +0.020 | 1.53x | 1.00x | 1.65x | 1.45x |
| **960 px, prune after block 4, threshold −3 (21% kept)** | **0.8042** | +0.001 | **2.13x** | 1.71x | 2.45x | ~2x |
| 960 px, prune after block 3, threshold −3 (23% kept) | 0.8029 | −0.001 | 2.45x | 1.71x | 2.82x | ~2x |
| 960 px, prune after block 4, fixed 40% (static shapes) | 0.8033 | 0.000 | 1.78x | 1.91x | | ~2x |

"Packed" is what varlen attention would give a batch of images that keep
different token counts. It is not implemented. "Padded" is what a plain batch
costs today. Training is measured steps/s for 1120 px, and the benchmark
projection for 960 px.

**Zero-shot, on the original weights** (no fine-tuning; see the zero-shot
section), TensorRT, batch 1, COCO mini500's mix of aspect ratios:

| configuration | COCO mAP | Fashionpedia | latency | speedup |
|---|---|---|---|---|
| original, square 960 input | 0.4967 | 0.2091 | 9.17 ms | 1.00x |
| **drop the all-padding tokens** | **0.5061** | 0.2101 | 6.54 ms | **1.40x** |
| + prune after block 4 with a distilled objectness scorer, keep 60% | 0.5025 | 0.2115 | 5.62 ms | 1.63x |

Padding drop is free and slightly better. Pruning on top is near-free on
natural images, but costs ~10% relative on square aerial tiles (DIOR), so it is
domain-dependent zero-shot. Both export to one TensorRT engine that takes any
aspect ratio and matches torch to 0.0003 mAP.

Lower resolution, windows, merging, pooled K/V, block skipping and width
pruning all lose to pruning, and so does training-only patch dropout.

## Where the compute goes

Vision tower, base at 960 px, N = 3601 tokens, 1.09 TFLOP per image:

| part | share |
|---|---|
| attention core (QK^T, AV) | **43.9%** |
| MLP | 37.4% |
| Q/K/V/O projections | 18.7% |
| P3 heads (2x2 slots, 14,400 queries) | +8.8% on top |

Attention grows with N^2, so at 3601 tokens it is the largest piece, and in
training its backward is the largest kernel group even after fp16 accumulation.
**Every token removed saves superlinearly.** On AI-TOD the tokens are mostly
background: only 1.3% of patches hold a vehicle centre (6.8% within one patch,
p99 33%).

## What the tower tolerates without retraining

Probes swap one part of the trained P3 model (or the zero-shot model on COCO),
then score it exactly as the trainer does (the harness reproduces the reference
AP to 0.0002). Speed: 5070 Ti, compiled, fp16 with fp16 GEMM accumulation; train
= B=8 forward+backward of tower + P3 heads with gradients as in the hybrid.

| variant | AI-TOD P3 AP (ref 0.8036) | COCO zero-shot mAP (ref 0.496) | infer B=1 | infer B=8 | train step |
|---|---|---|---|---|---|
| prune 75% of tokens after block 6, learned scorer + 16 registers | 0.7906 (−0.013) | 0.4421 (−0.054) | 1.67x | 1.79x | 1.89x |
| prune 50% after block 6, learned scorer + 16 registers | ~0.80 (−0.005, no registers) | **0.4911 (−0.005)** | 1.39x | 1.46x | 1.51x |
| same, adaptive keep (mean 12% kept at B=1) | 0.7907 | n/a | ~1.9x | (batch max: 25%) | |
| prune after block 4, keep 25%, learned + registers | 0.7784 | n/a | | | |
| input token drop keep 50% (GT oracle) vs mask-only control | 0.8113 vs 0.8141 | n/a | | | 2.44x (training only) |
| windows 15 in blocks 2–8 + 16 global registers | 0.7837 (−0.020) | ~0.15 | 1.25x | 1.19x | 1.19x |
| windows 15 in blocks 3–8 + registers | 0.7893 (−0.014) | | | | |
| windows 15, ViTDet layout (8 blocks) + registers | 0.7771 (−0.026) | 0.126 | 1.27x | 1.23x | 1.23x |
| same without registers | 0.7536 | 0.117 | 1.32x | 1.27x | 1.28x |
| 2x2 merge of the 50% most uniform blocks after block 3 | 0.7624 (−0.041) | 0.4829 (−0.013) | ~1.1x | ~1.2x | 1.11x |
| 2x2 merge 75% after block 2 | 0.6205 | 0.3977 | 1.50x | 1.73x | 1.61x |
| drop the all-padding tokens of non-square images | n/a (square tiles) | **0.5039 (+0.008)**, 28% fewer tokens | 1.49x at 4:3, 2.04x at 16:9 | 1.47x / 2.19x | 2.15x at 16:9 |
| pooled K/V 2x2 (any layout, with or without registers) | 0.38–0.72 | 0.05–0.31 | 1.1–1.2x | | |
| skip block 10 / 11 | 0.772 / 0.760 | 0.494 / 0.494 | 1.09x each | | |
| skip blocks 10+11 | 0.656 | 0.476 | 1.18x | 1.18x | 1.18x |
| MLP 50% / 8 of 12 heads (speed only, needs retraining) | | | 1.15x / 1.14x | 1.15x / 1.21x | 1.13x / 1.22x |
| 800 px input (speed only, needs retraining) | | | 1.58x | 1.66x | 1.59x |

### Findings

1. **OWLv2 attention is global, and it has register tokens.** Pretrained mean
   attention distance is 10–28 patches on the 60x60 grid (uniform would be 31).
   From block 4 on, 5–7 background patches per image grow to >10x the median
   norm (Darcet et al., "ViTs need registers"). Windowing collapses zero-shot
   COCO (0.10–0.28) even with the registers kept global.
2. **Fine-tuning on AI-TOD made blocks 3–8 local** (mean distance 4.6–8 patches,
   70–78% of the mass inside a 15x15 window) **and blocks 9–11 near-uniform**.
   Blocks 0–1 stay global (29% and 36% inside a 15x15 window). So on the
   fine-tuned model, windows in the middle blocks cost little. Keeping the 16
   highest-norm tokens global is worth +0.024 AP, and choosing the windowed
   blocks from the locality profile (2–8 or 3–8) beats the ViTDet layout.
3. **Late token pruning is the big lever for sparse imagery.** A 768→1 probe
   (or 768→128→1) on the block-6 residual stream, distilled from the full
   model's final per-token score on 240 train images, puts 99.98% of the final
   confident tokens (score > 0.3) in its top 25%. Pruning with it costs −0.024;
   keeping the registers halves that to −0.013. With GT, the context loss is
   only ~0.002–0.003, so the rest is scorer ranking, which joint training should
   fix. Dropping pruned tokens beats keeping them as frozen keys/values ("kv"
   mode): stale states fed to later K/V projections hurt (0.66 vs 0.77 after block 3).
4. **The kept tokens barely need the dropped ones.** Removing half the
   tokens at the input (all GT-adjacent tokens kept) leaves the kept tokens'
   detections at 0.811 vs 0.814 for the mask-only control. That is the
   regime of training-time patch dropout.
5. **Merging suits big-object domains, not tiny objects.** On COCO, merging
   the most uniform half of the 2x2 blocks after block 2–3 costs only
   0.013–0.017 without retraining. On AI-TOD it costs 0.03–0.07, because a
   vehicle inside a merged block is averaged away. Proportional attention (log 4
   key bias) is worth +0.04 on COCO, but the bias rules out flash and the fp16
   backward, so 50% merging only reaches ~1.1x.
6. **Padding tokens are free to drop.** OWLv2 pads every image to a square, at
   the bottom and right. Dropping the all-padding patches *raises* COCO mAP
   (0.4960 → 0.5039) while removing 28% of the tokens on mini500 (44% for 16:9
   frames). The kept tokens are a rectangle of the patch grid with their own
   position embeddings, so this is a static crop per aspect ratio.
7. **The fine-tuned model uses every block.** Each single skip costs
   0.025–0.06 (block 0: everything). Zero-shot, blocks 10–11 are nearly free
   (−0.002 each).
8. **Pooled K/V is dead.** It fails in every layout, including the near-uniform
   blocks 10–11 (−0.08). Spread-out attention still reads specific far tokens.
9. **Pruning also holds on general imagery at a higher keep.** On COCO zero-shot,
   a scorer distilled the same way (240 val2017 images outside mini500)
   costs −0.005 mAP when keeping 50% after block 6, and −0.054 at 25% (large
   objects suffer). The keep ratio is a per-domain knob: aerial tiles need 10–25%,
   COCO about 50%.

## Fine-tuning arms (P3 train_all recipe, 7400 steps, seed 0)

Each arm retrains the reference recipe with one change. It is scored by the
trainer's own val200 evaluation, which uses all tokens unless the arm prunes at
test time. The keep and threshold sweeps re-score the final weights with the
probe harness (`prune_sweep.py`), which matches the trainer at the training
keep to 0.0003. Seed noise on this recipe is about ±0.003 (seeds so far: 0.8030,
0.8036, 0.8074). Steps/s are in-training medians on the 5070 Ti. Arms that
overlapped a 3060 run lost about 7% to CPU contention, so the benchmark table
below is the clean speed comparison. E2b and E2c trained on the 3060 (AP only).

| arm | change | point AP | Δ vs ref | train steps/s | run wall clock |
|---|---|---|---|---|---|
| reference | — | 0.8036 | — | 2.55 | 52.4 min |
| **E2** | prune after block 6: keep 25%, 16 registers, scorer trained jointly | 0.7930 at keep 25% | −0.011 | 4.21 (1.65x) | 32.7 min (1.60x) |
| | E2 weights, keep 40% | 0.8060 | +0.002 | | |
| | E2 weights, adaptive −2 (11.4% kept; B=8 max 29%) | **0.8034** | 0.000 | | |
| | E2 weights, all tokens | 0.8064 | +0.003 | | |
| **E2c** | prune after block 4, otherwise as E2 | 0.7863 at keep 25% | −0.017 | (3060) | |
| | E2c weights, keep 40% / 50% | **0.8033** / 0.8042 | 0.000 / +0.001 | | |
| | E2c weights, adaptive −4 (31.4% kept; B=8 max 69%) | 0.8042 | +0.001 | | |
| | E2c weights, adaptive −3 (20.6% kept; B=8 max 49%) | **0.8042** | +0.001 | | |
| | E2c weights, adaptive −2 (12.8% kept; B=8 max 33%) | 0.7994 | −0.004 | | |
| | E2c weights, all tokens | 0.8033 | 0.000 | | |
| **E2b** | prune after block 3, otherwise as E2 | 0.7649 at keep 25% | −0.039 | (3060) | |
| | E2b weights, keep 40% / 50% | 0.7947 / 0.8011 | −0.009 / −0.003 | | |
| | E2b weights, adaptive −4 (35.6% kept; B=8 max 74%) | 0.8043 | +0.001 | | |
| | E2b weights, adaptive −3 (23.0% kept; B=8 max 54%) | **0.8029** | −0.001 | | |
| | E2b weights, adaptive −2 (13.7% kept; B=8 max 35%) | 0.7915 | −0.012 | | |
| | E2b weights, all tokens | 0.8037 | 0.000 | | |
| **E8** | **1120 px** + prune after block 3 (keep 25% of 4900 tokens) | 0.7947 at keep 25% | −0.009 | 3.70 (1.45x) | 37.9 min (1.38x) |
| | E8 weights, keep 40% | 0.8170 | +0.013 | | |
| | E8 weights, adaptive −5 (42.5% kept; B=8 max 81%) | 0.8223 | +0.019 | | |
| | E8 weights, adaptive −4 (29.5% kept; B=8 max 65%) | **0.8232** | **+0.020** | | |
| | E8 weights, adaptive −3 (18.3% kept; B=8 max 45%) | **0.8196** | **+0.016** | | |
| | E8 weights, adaptive −2 (10.7% kept; B=8 max 29%) | 0.8087 | +0.005 | | |
| | E8 weights, all tokens | 0.8209 | +0.017 | | |
| E1 | input patch dropout 50% (GT-adjacent tokens always kept), all tokens at test | 0.7205 | −0.083 | 4.88 (1.91x) | 28.5 min (1.84x) |
| E1c | E1 + density bias on the kept background keys | 0.7144 | −0.089 | 4.01 (1.57x) | 34.1 min (1.54x) |
| **E1b** | E1 for 5500 steps, then 1900 steps on all tokens | 0.7976 | −0.006 | 4.9 then 2.5 | 36.8 min (1.42x; ~1.5x without contention) |
| E6 | 15x15 windows in blocks 2–8 + 16 global registers | 0.7972 | −0.006 | 2.88 (1.13x) | 47.1 min (1.11x) |
| E4 | 800 px input (AI-TOD's native tile size, 50x50 patches) | 0.7697 | −0.034 | 3.81 (1.49x) | 36.3 min (1.44x) |

### What the arms show

1. **A pruning scorer trained with the model holds the reference AP at a
   fraction of the tokens.** The scorer is LayerNorm + 768→192→1 on the input
   of the prune block, at 10x the head learning rate. Training uses BCE against
   the GT centres dilated by one patch, GT tokens are always kept, and so are
   the 16 highest-norm tokens (the registers). Pruned tokens are dropped from
   the remaining blocks and produce no detections. Training with pruning does
   not hurt the full model (all-token AP 0.8064 / 0.8033 / 0.8037 for blocks 6
   / 4 / 3). At test time the keep is free to choose.
2. **The keep must be per image.** Fixed ratios fail on the dense images first.
   The adaptive threshold keeps every token whose scorer logit clears it, plus
   the registers, with a 2% floor. Each image then gets what it needs: at block
   6, threshold −2 keeps 11% on average and 29% for the densest image in a batch
   of 8, with no loss. Fixed-keep numbers undersell early pruning. At block 3 a
   fixed 25% gives 0.765, but the threshold holds the reference at 23%. A
   fixed-keep in-training eval is therefore a poor guide to the final trade-off.
3. **Earlier pruning saves more, up to block 3–4.** The earlier scorers are less
   sure, so they keep more tokens (23% at block 3 and 21% at block 4, against 11%
   at block 6, all at reference AP). They still save more work, because more
   blocks run on the kept tokens: 2.45x and 2.13x at B=1, against 1.84x. Below
   block 3 there is little left to gain. Blocks 0–2 are already ~70% of the
   remaining tower FLOPs, and blocks 0–1 are where attention is still global.
   Block 4 is also the best static-shape option: a fixed 40% keep holds the
   reference (0.8033). Block 3 needs 50% for that (0.8011).
4. **Pruning pays for resolution (E8).** At 1120 px the image has 4900 tokens,
   and the full tower costs 1.6x the reference. With block-3 pruning it is
   cheaper than the 960 px reference and better: +0.017 with all tokens, +0.020
   at threshold −4, +0.016 at threshold −3 (18% kept, 1.75x faster). It also
   trained 1.45x faster than the reference. Native 800 px (E4) loses 0.034, so
   resolution is what these 10–15 px vehicles need. The 960 px default was
   leaving AP on the table.
5. **Training-time patch dropout does not transfer to full-token inference.**
   E1 trains 1.91x faster but loses 0.083. The weights are fine in the regime
   they were trained in. Run sparse at test (GT-adjacent tokens kept), E1
   scores 0.847 against 0.765 with full tokens and the same detections masked.
   The reference model scores 0.811 and 0.814 in those two settings. So E1 has
   specialised to sparse context. The density bias (E1c) reweights each kept
   background key by log(background / kept background), as ToMe's proportional
   attention does. It made things slightly worse, and slower, since the bias
   rules out the fast attention backward. So the mismatch is in the features,
   not in the softmax normalisation. **A full-token tail fixes it (E1b).** AP
   sits at 0.722 when the tail starts at step 5500, reaches 0.794 after 500
   full-token steps, and ends at 0.7976 (−0.006, two to three seed SDs). This
   is FLIP's "unmasked tuning" applied to detection. A 500–1000-step tail would
   likely do and lift the gain from 1.5x toward 1.7x. Inference is unchanged,
   though, and pruning trains as fast while also speeding up inference.
6. **Windows recover most of their loss with training, but the gain is small
   (E6).** 15x15 windows in blocks 2–8, with the 16 highest-norm tokens global,
   cost −0.020 when swapped into the trained model. Trained in, they cost
   −0.006, and the AP has flattened by the end (0.797 over the last three
   evals). That buys 1.2x. The windows only cut the attention core of 7 of the
   12 blocks, about a quarter of the FLOPs, so 1.3x is the ceiling for this
   layout.

### Speed of the fine-tuned variants

5070 Ti, compiled, fp16 with fp16 GEMM accumulation, tower + P3 heads
(`bench_variants.py`; the 960 px reference is 12.1 ms at B=1, 104 ms at B=8 and
303 ms for a B=8 training step). Adaptive keep varies per image. At B=1 the
cost follows the mean keep. A padded batch of 8 costs the batch's largest keep.
A packed batch (varlen attention, not implemented) would cost the mean. Each
column was measured as a fixed keep at that value.

| variant | point AP | B=1 | B=8 padded | B=8 packed | train step (tower, at the training keep) |
|---|---|---|---|---|---|
| reference, 960 px | 0.8036 | 1.00x | 1.00x | | 1.00x |
| E2, block 6, adaptive −2 | 0.8034 | 1.84x | 1.73x | 1.98x | 1.89x |
| E2, block 6, fixed 40% | 0.8060 | 1.50x | 1.58x | | |
| E2c, block 4, adaptive −3 | 0.8042 | 2.13x | 1.71x | 2.45x | 2.40x |
| E2c, block 4, fixed 40% | 0.8033 | 1.78x | 1.91x | | |
| E2b, block 3, adaptive −3 | 0.8029 | 2.45x | 1.71x | 2.82x | 2.80x |
| E2b, block 3, adaptive −4 | 0.8043 | 2.16x | 1.35x | 2.27x | |
| E2b, block 3, fixed 40% | 0.7947 | 1.95x | 2.12x | | 2.17x |
| E8, 1120 px, block 3, adaptive −4 | 0.8232 | 1.53x | 1.00x | 1.65x | trainer: 1.45x steps/s |
| E8, 1120 px, block 3, adaptive −3 | 0.8196 | 1.75x | 1.30x | 1.99x | |
| E8, 1120 px, block 3, adaptive −2 | 0.8087 | 1.99x | 1.65x | 2.26x | |
| E8 weights, 1120 px, all tokens | 0.8209 | 0.63x | 0.65x | | |
| E6, windows in blocks 2–8 + registers | 0.7972 | 1.25x | 1.19x | | 1.19x |
| E4, 800 px | 0.7697 | 1.58x | 1.66x | | 1.59x |

The train-step column is the tower and heads only. A trainer step also spends
~85 ms on data, loss, matching and the optimizer. That turned E2's 1.89x into
1.65x steps/s, and puts blocks 3–4 at about 2x.

## Combinations

Two levers stack when they remove different work. When they remove the same
work, the second one adds little. Cost is FLOPs from the shares above unless a
benchmark is given.

| combination | verdict | evidence |
|---|---|---|
| pruning + higher resolution | **the best mix** | E8 above: +0.016 at 1.75x (B=1), +0.020 at 1.53x, and it trains 1.45x faster. |
| pruning + padding drop | **stacks, if the budget is set on the full grid** | COCO zero-shot, block-6 scorer: pruning alone at 50% 0.4910; padding drop alone 0.5039; both with an 1800-token budget (50% of the full grid, capped at the content) **0.4959**. On a 4:3 image that runs 1.74x at B=1 and 1.87x at B=8, against 1.49x / 1.47x for padding drop alone. Keeping 50% of the *content* instead gave 0.4794. Without padding drop the scorer drops the padding first anyway, so in the late blocks the two remove the same tokens. Padding drop adds its savings before the prune point. |
| pruning + lower resolution | no | 800 px alone costs 0.034 (E4). |
| pruning + windows | no | The blocks that still see every token are 0–2, and blocks 0–1 attend globally. Blocks 3–11 run on ~20% of the tokens, so their attention is already ~5% of full. |
| pruning + training-time patch dropout | not worth it | Amdahl. After block-3 pruning, the tower is ~108 ms of a ~200 ms training step (B=8). The rest is data, loss, matching and the optimizer, about 80–90 ms, and none of it shrinks. Dropout in blocks 0–2 would save ~35 ms (~1.2x) before the full-token tail, and E1b's cost (−0.006) comes with it. |
| pruning + FP8 GEMMs | small | On GeForce cards, FP8 with fp32 accumulation runs at the same tensor-core rate as fp16 with fp16 accumulation, and this pipeline already uses fp16 accumulation. Measured earlier: FP8 over fp16-acc saves 5–10 ms per training step; PTQ costs −0.002 (MLP only). |
| two-stage pruning (block 3, then block 6) | small | After block-3 pruning, blocks 0–2 are ~70% of the remaining tower FLOPs. A second cut at block 6 from 23% to 11% saves ~10% at most, and less in time, since small token counts run less efficiently. |
| windows + patch dropout / windows + resolution | not tested | Both cost −0.006 alone, for less speed than pruning. |

## Zero-shot on the original weights

Neither change below touches the tower or head weights, so open-vocabulary
detection stays intact.

- **Padding drop** has no parameters. OWLv2 resizes the long side to 960 and
  pads the rest at the bottom and right. `OwlV2.padding_free_grid` gives the
  content grid, and the model takes the cropped input directly: position
  embeddings and box priors are sliced to the grid, and boxes stay normalised
  to the padded square, so post-processing is unchanged.
- **Pruning with a distilled objectness scorer.**
  `tools/arch_speedup/distill_scorer.py` trains the 150k-parameter scorer of
  `enable_token_pruning` on the frozen tower. It learns to predict the unpruned
  model's final objectness, sigmoid(objectness logit), from the input of
  block k. It uses 4200 unlabelled images: COCO train, Fashionpedia, LV-MHP,
  DIOR, AI-TOD and AU-AIR. Training takes 3 epochs, ~12 min on the 5070 Ti, with
  half of the images padding-free. No labels and no class names are involved,
  so it prunes the same way for any query. Load it with
  `OwlV2.load_token_scorer`. Held-out share of the confident tokens
  (objectness > 0.1) inside the top 50%: 0.94 / 0.96 / 0.99 for blocks 3 / 4 / 6.

`tools/ovd_eval.py --pad-drop --prune-scorer <file> --prune-keep <fraction>`,
standard COCO metric (maxDets 100), batches of 8 sorted by aspect ratio:

| configuration | COCO mini500 | Fashionpedia val (500) | DIOR test (500, square tiles) |
|---|---|---|---|
| original | 0.4966 | 0.2091 | 0.0781 |
| padding drop | **0.5061** | 0.2101 | (nothing to drop) |
| padding drop + block 4, keep 80% / 70% | | | 0.0756 / 0.0743 |
| padding drop + block 4, keep 60% | 0.5028 | **0.2115** | 0.0697 |
| padding drop + block 4, keep 50% | 0.4996 | 0.2103 | 0.0704 |
| padding drop + block 4, keep 40% | 0.4931 | 0.2060 | 0.0648 |
| padding drop + block 3, keep 50% | 0.4991 | | |
| padding drop + block 6, keep 60% / 50% / 40% / 30% | 0.4922 / 0.4795 / 0.4660 / 0.4495 | | 0.0663 (60%) |
| block 4, keep 50%, no padding drop | 0.4846 | | |

1. **Padding drop is the zero-shot win:** +0.0095 on COCO and +0.001 on
   Fashionpedia, while removing 28% / 33% of the tokens.
2. **Zero-shot pruning wants an early cut.** Block 4 beats block 6 by 0.01–0.02
   at every keep, although the block-6 scorer ranks objectness better. This is
   the reverse of the fine-tuned model. Large objects suffer most at block 6
   (COCO AP_L 0.623 against 0.646), which fits blocks 6–11 of the original model
   still reading object interiors and context globally. The budget counts the
   full grid, so padding is not part of it (see Combinations).
3. **The keep is a per-domain knob, zero-shot too.** On natural images, block 4
   at 60% costs 0.003 against padding drop alone and stays above the original.
   On square aerial tiles it costs 0.008 (−11% relative), and 0.0025 at 80%.
   There zero-shot objectness is weak to begin with (0.078 mAP), so prune
   aerial zero-shot only at ≥80%, or not at all.

### TensorRT

`export.export_padding_free_detector` exports tower and heads as one graph:
image of any multiple of 16 up to 960 x 960 (batch ≤ 8), shared query
embeddings (≤ 1024), and optional pruning. The token budget is
`min(keep x 3600, tokens present)`, a TopK whose K TensorRT 11 resolves from
the input shape at runtime. One engine therefore serves every aspect ratio.
Outputs are logits, objectness, boxes and each token's index in the full grid.
`build_engine.build_detector_engine` builds it, `PaddingFreeDetectorTRT` runs
it, and `tools/arch_speedup/export_pruned.py` does all three and times it.
`ovd_eval.py --trt-detector-engine` scores it. The register tokens are lifted
with `where` rather than `scatter_add`, which TensorRT only has as a plugin.

| engine (fp16, 5070 Ti, 80 queries) | COCO mAP (torch) | 960 x 960 | 720 x 960 (4:3) | 544 x 960 (16:9) | COCO mean, B=1 | B=8 at 960 x 960 |
|---|---|---|---|---|---|---|
| padding-free, no pruning | 0.5061 (0.5061) with padding drop; 0.4967 (0.4966) square | 9.45 ms | 6.80 ms | 4.89 ms | 6.54 ms | 10.65 ms/image |
| + block 4, keep 60% | 0.5025 (0.5028) | 6.36 ms | 5.67 ms | 4.93 ms (fits the budget) | 5.62 ms | 7.25 ms/image |

The per-token maximum differences against torch fp16 are large on some
low-scoring tokens, up to ~5 in a logit, but the kept sets agree to ≥99.8%
and the mAP to 0.0003. Fine-tuned checkpoints export through the same class
once their pruning config (fixed keep) is loaded; point-slot heads and the
adaptive threshold do not export yet.

### Blur, shrinking inside the canvas, multi-label scoring

Three more inference-only changes, all in `OwlV2.set_preprocessing` /
`OwlV2.select_detections` and `ovd_eval.py --blur/--long-side/--multi-label`.
They run on the padding-free engine above unchanged.

- **Blur (σ 0.5 px on the resized image).** OWLv2 wants slightly soft input.
  COCO fed at exactly native scale loses 0.04 mAP; a 0.5 px blur recovers most
  of it, while a 4–8 px shift does not, so it is softness rather than JPEG
  block alignment. Inputs whose long side is already ~960 px probably lose
  accuracy today as well (not tested directly).
- **Shrink inside the canvas (`long_side`).** Resize the long side to 768 or
  704, keep the 60 x 60 position embeddings and box priors, crop the padding,
  and scale boxes by 960 / long side. Resampling the position embeddings
  instead (`set_image_size`) breaks zero-shot: −0.046 at 896, −0.20 at 768.
- **Multi-label scoring.** The top 300 (token, class) pairs ranked by class x
  √objectness, instead of each token's best class x objectness.

| configuration (padding drop on) | COCO val (5000) argmax / multi-label | COCO mini500 argmax / multi-label | Fashionpedia (500) argmax / multi-label | DIOR (500) argmax / multi-label | TRT ms, COCO mean B=1 |
|---|---|---|---|---|---|
| original (HF pipeline, square) | 0.4556 / 0.4596 | 0.4966 | 0.2091 | 0.0781 | 9.17 |
| padding drop | 0.4588 / 0.4629 | 0.5060 / 0.5127 | 0.2105 / 0.2427 | 0.0770 / 0.0748 | 6.54 |
| + blur 0.5 | **0.4695 / 0.4741** | 0.5141 / 0.5197 | 0.2126 / **0.2603** | 0.0766 / 0.0752 | 6.53 |
| + blur 0.5, long side 768 | 0.4630 / 0.4655 | 0.5115 / 0.5156 | 0.2099 / 0.2495 | **0.0782 / 0.0791** | 4.07 |
| + blur 0.5, long side 704 | 0.4597 / 0.4625 | 0.5065 / 0.5109 | | | 3.32 |
| + blur 0.5, block 4 keep 60% | | 0.5113 / 0.5191 | | | 5.62 |

The engine reproduces torch within 0.002. `ovd_eval.py` through the new API
gives 0.5117 / 0.5156 for the 768 row (probe 0.5115 / 0.5156). Multi-label
scoring with class alone (no objectness) is slightly better on mini500 but
worse on full val and Fashionpedia, so √objectness is the default.

- Shrinking is the better speed lever. Against 960 + blur, 768 costs COCO
  0.0065 and Fashionpedia 0.003 / 0.011 (argmax / multi-label), both still
  above the original, and gains on DIOR, where 60% pruning cost 0.008.
  Pruning is only worth it at 960 for near-maximum accuracy.
- Keep the one-class-per-token scoring for aerial (DIOR −0.002 at 960).
- Dead ends. The engine at 960² is 51% GEMMs, 42% fused attention and 7%
  the rest. INT8 GEMMs are 1.9x in isolation, but static INT8 on the MLP's
  second layer collapses COCO to 0.42, and Q/DQ nodes broke TensorRT's
  attention fusion (10.4–10.6 vs 9.45 ms). FP8 GEMMs are no faster than the
  fp16 math TensorRT already uses; FP8 attention is 1.16x. Adaptive pruning
  lands on the fixed-keep curve, prompt ensembles change ±0.001, and NMS
  hurts.

Follow-up GPU measurements on 2026-10-28 are in
[`inference-speed-profile.md`](inference-speed-profile.md): an identity-alias
pooling bypass makes the evaluator 1.54x faster with identical detections;
composed preprocessing plus CUDA graphs gives 1.12x on four fixed-size
PIL-to-host cases. Fixed TensorRT profiles add about 4%, FULL tiling adds
nothing measurable, and actual 2:4 sparse MLP tactics add only about 3%.

**OWLv2-large.** The same sweep on large (1008 px, patch 14) ran on a
padding-free engine (`export_pruned.py --model-type large`; mini500 0.4975
against torch 0.4976). Long sides 812 and 742 are the canvas fractions of
768 and 704. Scores are argmax / multi-label; latency is TRT fp16, B=1, 80
queries, square / 4:3.

| large | COCO val (5000) | COCO mini500 | Fashionpedia (500) | DIOR (500) | ms, 1:1 / 4:3 |
|---|---|---|---|---|---|
| original | 0.4551 / 0.4588 | 0.4975 / 0.5050 | 0.2340 / 0.2631 | 0.0990 / 0.0978 | 48.6 |
| padding drop | 0.4581 / 0.4621 | 0.5005 / 0.5074 | **0.2347 / 0.2640** | 0.0990 / 0.0978 | 48.6 / 32.2 |
| + blur 0.5 | **0.4604 / 0.4644** | **0.5029 / 0.5103** | 0.2289 / 0.2522 | 0.0984 / 0.0968 | 48.6 / 32.2 |
| + blur 0.5, long side 812 | 0.4584 / 0.4619 | 0.5023 / 0.5063 | 0.2249 / 0.2517 | **0.0995** / 0.0931 | 27.9 / 19.0 |
| + blur 0.5, long side 742 | 0.4560 / 0.4595 | 0.4950 / 0.5003 | 0.2264 / 0.2498 | 0.0986 / 0.0909 | 21.3 / 14.5 |

- Padding drop (+0.003 COCO) and multi-label on natural images (+0.004
  COCO, +0.03 Fashionpedia) carry over. Multi-label hurts DIOR more on
  large: −0.001 at full size, −0.006 to −0.008 once shrunk.
- Blur is mostly a base-model effect. On large it is +0.002 on COCO val,
  and it costs Fashionpedia 0.006 / 0.012 and DIOR 0.001.
- Shrinking costs large less on COCO: 812 is 1.7x faster at 4:3 for −0.002.
  Fashionpedia loses 0.004–0.012, part of which is likely the blur, since
  shrinking without blur was not run.
- Zero-shot, large equals base on COCO (0.4551 vs 0.4556) but wins on
  Fashionpedia (0.234 vs 0.209) and DIOR (0.099 vs 0.078).
- For large: padding drop plus multi-label for natural images, no blur, and
  812 as the speed knob.

## Recommendations

1. **Adopt learned token pruning with a per-image threshold.** All three prune
   points hold the reference AP, each trained separately. Pick by goal:
   - Most AP: **1120 px, prune after block 3, threshold −3 or −4.** +0.016 to
     +0.020, 1.5–1.75x faster at B=1, and it trains 1.45x faster than today's
     960 px recipe.
   - Most speed at today's AP: **960 px, prune after block 4 (or 3), threshold
     −3.** 2.1–2.45x at B=1, about 2x faster training.
   - Static shapes (TensorRT without dynamic shapes): **960 px, block 4, fixed
     40%.** 0.8033, 1.78x at B=1, 1.91x at B=8.
2. **Batch adaptively kept tokens with varlen attention.** A padded batch costs
   the densest image's keep, which erases most of the B=8 gain (1.30x padded
   against 1.99x packed for 1120 px at −3). FlashAttention's `cu_seqlens` or
   PyTorch's jagged nested tensors pack each image's kept tokens. Everything
   but attention is per token, so only attention needs the varlen kernel.
3. **Zero-shot (original weights): drop the all-padding tokens and blur by
   0.5 px; shrink to a 768 long side for speed.** Padding drop is free (+0.0095
   COCO) and 1.40x on COCO's aspect mix in TensorRT, 1.9x for 16:9 video. Blur
   adds +0.011 on COCO val. A 768 long side is 1.6x faster again for −0.0065
   COCO, −0.003 to −0.011 Fashionpedia and nothing on aerial, and beats pruning
   as the speed knob. Add
   multi-label scoring for natural images. Ship it as the one padding-free
   TensorRT engine.
4. **Training-time patch dropout only with a full-token tail, and only where
   pruning cannot be used** (dense scenes, a fixed deployment graph): 1.5x
   training for −0.006, no inference gain. Do not use the density correction.
5. **Not worth it here:** lower resolution (−0.034 at 800 px), windows (−0.006
   for 1.2x, and they do not stack with pruning), pooled K/V, 2x2 merging for
   tiny objects, block skipping, MLP/head width pruning, FP8 on top of fp16
   accumulation, and a second prune stage.

Caveats. One seed per arm, but three separately trained prune points agree.
Thresholds were picked on val200 itself, with no separate tuning split. The
curves are smooth and −3 is best for blocks 3, 4 and 1120 px alike, so the
selection bias is small. The scorer is trained on the target classes. For open
vocabulary use it needs a class-agnostic target. On COCO, a scorer distilled
from the zero-shot model holds −0.005 at 50% keep.

### What to explore next

- Port pruning into the repo: the scorer, the `--prune-*` flags, an adaptive
  eval mode in the trainer, and checkpoint loading. Then run two more seeds of
  the two recommended configs (1120 px −3 and 960 px block 4 −3).
- Push resolution further under pruning: 1280 px with block-3 pruning is still
  ~1.3x cheaper than today's reference by FLOPs, and the 960 → 1120 step was
  worth +0.017.
- Varlen packing (recommendation 2) and an export path: TensorRT TopK + Gather
  with a few keep buckets (e.g. 128-token steps, one profile each).
- A class-agnostic scorer (objectness distillation on mixed data) for
  zero-shot use.
- Training throughput now hits the fixed ~85 ms per step (data, loss, matching,
  optimizer). After pruning that is ~40% of a step, the next target for faster
  training.

## Code, checkpoints and logs

- Branch `arch-speedup`. The trainer flags are
  `--prune-after/-keep/-mode/-registers/-loss-weight/-target-dilate`,
  `--train-token-keep/-keep-dilate/-mode/-keep-steps/-density-bias`,
  `--window-layers/-size/-registers`, `--merge-after/-frac/-noprop`,
  `--image-size` and `--drop-vision-layers`, all off by default; default
  outputs are bit-identical to before. The model side adds
  `OwlV2.padding_free_grid`, padding-free inputs, `load_token_scorer`, and
  adaptive/full-grid keeps. The export side adds `PaddingFreeDetector`,
  `export_padding_free_detector`, `build_detector_engine` and
  `PaddingFreeDetectorTRT`.
- Zero-shot: `tools/arch_speedup/distill_scorer.py` (scorers in
  `text_checkpoints/zeroshot_scorers/objectness_scorer_b{3,4,6}.pt`),
  `tools/arch_speedup/export_pruned.py` (engines in `trt_engines/owlv2_base_padfree*.engine`),
  and `tools/ovd_eval.py --pad-drop/--prune-scorer/--prune-keep/--prune-threshold/--trt-detector-engine`.
  Blur, long side and multi-label: `OwlV2.set_preprocessing`,
  `OwlV2.select_detections`, `ovd_eval.py --blur/--long-side/--multi-label`.
- `tools/arch_speedup/run_exp.sh`: the P3 `train_all` recipe from
  `tools/run_aitod_points.sh`, without MLflow; `NAME=<arm> tools/arch_speedup/run_exp.sh <flags>`.
  Arm flags: E2 `--prune-after 6 --prune-keep 0.25 --prune-mode drop --prune-registers 16`;
  E2c/E2b the same with `--prune-after 4` / `3`; E8 adds `--image-size 1120` to
  E2b; E1 `--train-token-keep 0.5`; E1c adds `--train-token-density-bias`; E1b
  adds `--train-token-keep-steps 5500`; E6 `--window-layers 2 3 4 5 6 7 8 --window-size 15 --window-registers 16`;
  E4 `--image-size 800`.
- `tools/arch_speedup/arch_probe.py`: the training-free harness (skip,
  windows, pooled K/V, pruning incl. adaptive, merging, registers, padding
  drop), exact against the reference forward in fp32. `OWL_REPO` points it at
  another checkout of this repo.
  `prune_sweep.py <final.pth> <prune block> [keeps] [thresholds]` re-scores a
  pruning checkpoint (set `IMAGE_SIZE=1120` for E8).
  `bench_variants.py infer|train <B> [filter]` gives the speed tables.
  `probe_run.py`, `prune_scorer.py`, `scorer_controls.py`, `adaptive_prune.py`,
  `coco_scorer.py`, `coco_combo.py`, `pad_drop_eval.py`, `attn_locality.py`,
  `token_norms.py`, `e1_diag.py` and `flops.py` produced the rest.
- Checkpoints: `text_checkpoints/archspeed_<arm>/final.pth` for E2, E2b, E2c
  and E8. They are delta checkpoints plus `token_scorer.*` keys (LayerNorm +
  768→192→1). `prune_sweep.py` shows how to load them: split off the scorer,
  then `load_model`, with `set_image_size(1120)` first for E8. The reference is
  `text_checkpoints/P3_point_slots2_wh2.5_seed0_trainall_archref/final.pth`.
- Logs: `logs/aitod_points/arch_speedup/train_<arm>.log`.
