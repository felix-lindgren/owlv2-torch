# Run plan for the 2026-08-03 working diff

What the uncommitted changes add, and the runs needed to decide each one. Written
against the diff on `master` (class-loss variants, `gpu_augment.BatchAugmentor`,
`set_gradient_checkpointing`, warmup, `--val-transform`). Companion to
`docs/loss-notes.md` (why the loss changed) and `docs/optimization-notes.md`
(where the time goes).

Hardware: **2x RTX 3060 (11.6 GiB usable each)**, both idle. Every run below is
single-GPU, so two arms run concurrently.

## Baselines already on the board

Both from MLflow, both trained with the **pre-fix** classification loss
(`reduction="mean"`, i.e. λ_cls diluted 46x — see `loss-notes.md` finding 1), so
neither is a valid baseline for anything except itself.

| run | config | samples seen | best val mAP | wall clock |
|---|---|---:|---:|---:|
| zero-shot | — | 0 | 0.2404 | — |
| `20260802_164300` | head-only, bs16, 3000 steps, top-k 300 | 48,000 (1.05 ep) | 0.3259 @2000 | 4h45 |
| `20260802_214044` ("vision2") | vision-blocks 2, bs8, 2200 steps, top-k 100 | 17,600 (0.39 ep) | 0.3175 @2200 (still rising) | 2h10 |

Only **vision2** is a usable reference: `--eval-top-k` changed 300 → 100 in this
diff, which moves the metric. Head-only needs re-running at the fixed budget
(run C0 below) before the "does unfreezing help" question means anything.

**Superseded by block A (2026-08-03).** A1 re-ran vision2's exact config with the
*fixed* focal loss and landed at 0.3205 vs 0.3175 — so vision2 remains usable as a
reference despite the broken loss, because the fix turned out to be worth ~nothing
on its own. The current best at this budget is **A2 (MAL) at 0.3268**. See block A
results below; note the ±0.005 noise floor measured there applies to every
single-seed comparison in this document.

## Memory and step-time ceiling (measured today)

`tools/probe_grad_ckpt_memory.py --model-type large --text-blocks 0
--num-queries 46`, large @1008px, AdamW + AMP, peak allocated:

| vision-blocks | batch | ckpt off | ckpt on | step (ckpt on) |
|---:|---:|---:|---:|---:|
| 0 | 8 | 3.83 GiB | 3.83 GiB | 2337 ms |
| 0 | 16 | 5.99 GiB | 5.99 GiB | 4666 ms |
| 2 | 4 | 4.70 GiB | 3.26 GiB | 1467 ms |
| 2 | 8 | 7.70 GiB | 4.82 GiB | 2946 ms |
| 2 | 16 | **OOM** | 7.92 GiB | 5856 ms |
| 6 | 8 | **OOM** | 5.45 GiB | 4083 ms |
| 12 | 8 | **OOM** | 6.40 GiB | 5832 ms |
| 24 (full) | 2 | **OOM** | 3.73 GiB | 2345 ms |
| 24 (full) | 4 | **OOM** | 5.21 GiB | 4611 ms |
| 24 (full) | 8 | **OOM** | 8.30 GiB | 9319 ms |

Three things fall out of this, and they set the shape of the whole plan:

1. **`--batch-size 16` is only reachable with `--grad-checkpointing`**, even at
   vision-blocks 2. Activations cost ~1.44 GiB per trainable block at bs8.
2. **Any unfreeze past 2 blocks requires checkpointing.** Full unfreeze OOMs at
   *batch size 2* without it. `set_gradient_checkpointing` is not a nice-to-have
   for block C — it is the only thing that makes block C exist.
3. Checkpointing is nearly free when little is trainable (+5% at vision-blocks 2,
   and an exact no-op at vision-blocks 0 — 3.83 GiB / 2337 ms either way, which
   confirms `_should_checkpoint`'s frozen-prefix skip works) and expensive when
   everything is (step time goes 2.9 s → 9.3 s at full unfreeze, ~0.29 s per
   extra trainable block). The "~20%" in the `--grad-checkpointing` help text is
   right for shallow unfreeze and much too optimistic for deep.

Sample throughput is flat in batch size (this workload is GEMM-bound), so
**wall clock is set by samples seen, not by batch size**: ~0.35 s/sample at
vision-blocks 2. Budget accordingly.

## Do these before launching anything

**Status 2026-08-03: all but the mosaic schedule are done.** `--compile`,
`--grad-accum-steps`, `drop_last=True` and `--mosaic-prob` are wired and were used
by block A. `--val-batch-size` now exists too (uncommitted at time of writing) but
block A did *not* use it, so block A's wall clock still carries the small-eval-batch
cost. The `--mosaic-prob` default is still 0.5, so the warning below stands. Only
D2's step-based augmentation-off tail remains unimplemented.

Three of the runs below cannot be expressed with today's flags, and two defaults
will silently corrupt the comparisons.

- **`--mosaic-prob` defaults to 0.5 and `--gpu-augment` to True.** A run launched
  with the new defaults *already has mosaic on*. Every arm outside block D must
  pass `--mosaic-prob 0` explicitly, or the loss screen becomes a loss×mosaic
  screen. Consider flipping the default to 0 until block D has run.
- **Gradient accumulation does not exist.** Block B needs a
  `--grad-accum-steps`. Note while adding it that it will *not* be numerically
  identical to a single large batch: `_quality_class_loss` divides by that
  micro-batch's `num_boxes` and `_mean_or_zero` averages per image, so 4x bs4
  weights a 2-object image the same as a 20-object one while 1x bs16 does not.
  That difference is precisely what block B measures, so accumulate the loss
  as-is rather than "fixing" it — but record which convention was used.
- **No step-based augmentation-off tail.** `mosaic_prob` is constant for the whole
  run; DEIM's recipe disables mosaic for the final epochs (`no_aug_epoch: 8`) and
  `loss-notes.md` calls that non-optional. Block D's D2 needs it.
- **Free throughput, no experiment required** (from `optimization-notes.md`):
  wire `model.vision_model.encoder.compile()` behind a `--compile` flag (1.10x
  end-to-end, measured), set `drop_last=True` on the train loader (also avoids a
  recompile on the ragged final batch), and use a larger eval batch size (eval
  peaks at 2.8 GiB and costs 5.5 min × 4 per run). Doing these first cuts ~15%
  off every run in this document.
  **Measured 2026-08-03:** `--compile` + `drop_last` gave 1h59 / 2h00 for A1 / A2
  against vision2's 2h10, i.e. **~8% end-to-end** — consistent with 1.10x on the
  train loop diluted by the four evals, which were left at the small batch size.
  Raising the eval batch size via `--val-batch-size` is the remaining piece of that
  ~15%, and is worth passing on every run from block C onward.

## Standard protocol

Unless stated otherwise, hold everything at vision2's configuration:

```
--model-type large --vision-blocks 2 --batch-size 8 --num-workers 6 \
--head-learning-rate 5e-5 --vision-learning-rate 1e-5 --text-learning-rate 1e-7 \
--text-blocks 0 --warmup-steps 100 --weight-decay 1e-4 --negative-ratio 20 \
--lambda-class 1.0 --lambda-l1 5.0 --lambda-giou 2.0 --lambda-objectness 0.5 \
--eval-top-k 100 --eval-every-steps 550 --max-steps 2200 --seed 0 \
--mosaic-prob 0 --amp --augment
```

**Budget: 17,600 samples seen** (= 2200 steps at bs8), matching vision2. Where a
run changes the batch size, change `--max-steps` to keep samples constant, not
steps. Report `val/map` at the fixed budget, not the best checkpoint — with 4
evals per run the argmax is noise.

**Two seeds minimum for any difference below ~0.01 mAP.** Measured 2026-08-03:
re-running one config at a second seed moved `val/map` by **+0.0058**, and the
per-eval gap between the two seeds reached ±0.0084 while changing sign twice.
A single-seed difference of 0.005 carries no information. This was learned the
expensive way — block A ranked four losses inside its own error bar (finding 9).
Either budget two seeds per arm, or restrict claims to metrics shown to be
seed-stable (`map_large` was; `map_medium` was not).

---

## Block A — classification loss (3 runs, decides everything downstream)

`--class-loss {focal,mal,vfl}` is the highest-leverage change in the diff, and
until it is settled every other comparison is measured on a model whose class
head gets 0.8% of the gradient. Box and objectness terms are byte-identical
across the three arms, so these are directly comparable — and directly
comparable to vision2, which is the same config with the *broken* focal.

| run | change | expected cost |
|---|---|---|
| **A1** | `--class-loss focal` (now fixed: sum over classes) | 2h10 |
| **A2** | `--class-loss mal --class-loss-gamma 1.5` | 2h10 |
| **A3** | `--class-loss vfl --class-loss-gamma 1.5` | 2h10 |

Two GPUs → two waves, ~4.5 h. vision2 (0.3175) is the fourth cell for free.

Watch: `train/L_cls` share of `train/loss_step`. At step 0 the measured shares are
58% (focal), 81% (mal), 78% (vfl) versus 0.8% before the fix. If mAP *drops*
relative to vision2, the λ weights need retuning rather than the loss being wrong
— GIoU went from 78% of the gradient to ~14–30%.

- **A4 (conditional, +2h10):** if MAL or VFL wins, rerun it with
  `--lambda-objectness 0`. `loss-notes.md` flags that MAL and the objectness head
  now both encode box quality, and eval ranks by `sigmoid(cls)·sigmoid(obj)` —
  training two heads on correlated targets is worth one run to rule out.

### Results (2026-08-03, wave 1)

A1 and A2 ran concurrently with `--compile --grad-accum-steps 1`, everything else
at the standard protocol. Reported at the fixed 2200-step budget; both curves rose
monotonically, so the fixed-budget and best-checkpoint numbers coincide here.

| run | class loss | map | map_50 | map_75 | map_small | map_medium | map_large |
|---|---|---:|---:|---:|---:|---:|---:|
| vision2 | focal (broken) | 0.3175 | 0.4310 | 0.3527 | 0.1124 | 0.2749 | 0.3722 |
| **A1** | focal (fixed) | 0.3205 | 0.4380 | 0.3563 | 0.1120 | 0.2798 | 0.3734 |
| **A2** | mal γ1.5 | **0.3268** | **0.4420** | **0.3659** | 0.1149 | 0.2725 | **0.3914** |
| **A3** | vfl γ1.5 | 0.3198 | 0.4343 | 0.3527 | 0.1129 | **0.2816** | 0.3829 |
| **A4** | mal γ1.5, λ_obj 0 | 0.3228 | 0.4354 | 0.3579 | **0.1151** | 0.2686 | 0.3820 |

`val/map` trajectories (550 / 1100 / 1650 / 2200):

| run | 550 | 1100 | 1650 | 2200 |
|---|---:|---:|---:|---:|
| vision2 | 0.2664 | 0.2962 | 0.3125 | 0.3175 |
| A1 focal | 0.2876 | 0.2999 | 0.3091 | 0.3205 |
| A2 mal | 0.2787 | 0.3088 | 0.3145 | 0.3268 |
| A3 vfl | 0.2778 | 0.2970 | 0.3094 | 0.3198 |
| A4 mal λ_obj 0 | 0.2836 | 0.3064 | 0.3158 | 0.3228 |

MLflow: A1 `owlv2_text_20260803_171935`, A2 `owlv2_text_20260803_171939`,
A3 `owlv2_text_20260803_192812`, A4 `owlv2_text_20260803_192815`.

**0. Block A is a null result.** All four losses land within **0.0093** of the
broken-focal baseline — under two noise units (below) for the entire block, after
~8 GPU-hours. Ranking: A2 mal 0.3268 > A4 0.3228 > A1 focal 0.3205 > A3 vfl 0.3198
> vision2 0.3175. **No classification loss in the diff is decisively better than
the one already shipped.** Adopt MAL because it is nominally best and costs
nothing, but do not treat the choice as settled or as a foundation for downstream
blocks — the gap between best and worst is the same size as the run-to-run wobble.

> **Superseded by the seed repeat (finding 9).** A second seed of A2 alone moved
> `val/map` by +0.0058, so this ordering is *inside* seed noise and must not be
> read as a ranking at all. What survives is finding 10: MAL's `map_75` /
> `map_large` advantage reproduces across seeds. Treat "adopt MAL" as justified by
> box quality, not by the 0.3268 headline.

**1. Fixing the focal dilution bug bought nothing.** A1 finished +0.0030 over
vision2 — inside the noise floor (below). The 46x dilution was real and the fix is
correct, but on its own it does not move mAP at this budget. The premise that this
was "the highest-leverage change in the diff" is not supported by A1.

**2. MAL wins, but on box quality, not classification.** A2 is +0.0093 over
vision2 (+2.9% relative) and +0.0063 over fixed focal. Its margins are two to three
times larger on map_75 (+0.0132) and map_large (+0.0192) than on headline map, and
that pattern held at every eval (1100, 1650, 2200) rather than appearing only at
the end. map_small finished a three-way wash. So MAL helps by being IoU-aware, not
by relieving class-head starvation — the opposite of the mechanism this block was
designed to test.

**3. Fixed focal does not hold its gradient share; MAL does.** `train/L_cls` share
of `train/loss_step`:

| steps | A1 focal | A2 mal |
|---|---:|---:|
| 0–20 | 57.7% (predicted 58%) | 83.0% (predicted 81%) |
| 1000–1100 | 25.2% | 74.1% |
| 1550–1650 | 24.7% | 74.3% |

Focal's class term collapses back to GIoU-dominated (58.9% GIoU by step 1650) as
easy negatives are learned and the focal modulation suppresses them, i.e. A1 drifts
back toward vision2's gradient regime within ~1100 steps and then plateaus. This
describes A1's shape but does *not* predict the ranking — the composition was
already stable by 1100 while the mAP ordering kept changing.

**4. Noise floor: ±0.005 on `val/map` between consecutive evals.** The A1/A2 lead
changed hands twice (A1 +0.0089 at 550, A2 +0.0089 at 1100, A2 +0.0020 at 1650,
A2 +0.0063 at 2200). Margins vs vision2 wandered non-monotonically: A2 went
+0.0123 → +0.0126 → +0.0020 → +0.0093. **Every arm is single-seed, and A2's entire
winning margin is about one noise unit.** Block A ranks the four losses; it does
not separate anything closer than ~0.01. Before block C builds on the winner, a
seed repeat of the top arm is the cheapest available insurance.

**5. λ retuning is not needed.** The stated failure mode — mAP dropping because
GIoU fell from 78% to 12–31% of the gradient — did not occur. Both fixed-loss arms
beat vision2 at map_75, so box quality survived the reweighting.

**Correction to the A4 rationale above.** MAL's objectness term is only ~3% of the
loss by step 1100, which suggests `--lambda-objectness 0` is nearly a no-op. It is
not: eval ranks by `sigmoid(cls)·sigmoid(obj)`, so the objectness head still gates
every detection multiplicatively while training on zero gradient. Small gradient
share does not imply small eval effect, and A4 is a sharper test than the
"correlated targets" framing implies — if A4 ≈ A2 the head is redundant under MAL
and `--lambda-objectness` drops out of the recipe entirely; if A4 collapses, the
head is load-bearing at eval time regardless of its loss share.

### Wave 2 findings (A3, A4)

**6. VFL is not a win, and it refutes the "IoU-awareness" story.** A3 finished at
0.3198, +0.0023 over vision2 and *below* fixed focal. Its map_75 is 0.3527 —
identical to vision2's 0.3527 to four decimals, i.e. VFL captured none of the
box-quality gain that finding 2 attributed to IoU-awareness. MAL and VFL are both
dense IoU-aware losses, yet they differ by 0.0070, more than MAL differs from focal
(0.0063). So whatever MAL is doing, "it is IoU-aware" does not explain it. Finding
2's mechanism should be read as descriptive of MAL specifically, not as a general
property that transfers.

**7. The objectness head is redundant under MAL — stop tuning `--lambda-objectness`.**
A4 vs A2 across all four evals: +0.0049, −0.0024, +0.0013, −0.0040. Oscillating
around zero, every delta inside the noise floor, final gap −0.0040. Training the
objectness head on zero gradient while it still gates every detection at eval
costs nothing measurable. This is the most actionable result in the block: it
removes a hyperparameter rather than choosing one. Keep λ_obj at 0.5 (nominally
best, and free), but do not spend runs sweeping it, and do not assume the two heads
are fighting each other — `loss-notes.md`'s correlated-targets concern is not
observable at this budget.

**8. What block A actually bought, in hours.** ~8 GPU-hours across four arms to
learn that the loss choice is worth ≤0.01 mAP, that one hyperparameter can be
ignored, and that two of the diff's three new losses are not improvements. That is
a legitimate negative result and it is worth having — but it argues strongly for
spending the next 8 hours on capacity (block C) rather than on further objective
shaping, and for repeating the top arm at a second seed before anything is built
on the 0.3268 number.

### Seed repeat — A2 at `--seed 1` (2026-08-03)

A2's exact config, seed 0 → 1. MLflow `owlv2_text_20260803_213350`, 1h58.
(This run and C1 also used `--val-batch-size 16`, which is metric-neutral: the val
loader does not set `drop_last`, so every val image is scored regardless of batch
size, and `square_pad=True` pads each image to the fixed 1008px input independently
of its batch.)

| metric | A2 seed 0 | A2 seed 1 | Δ |
|---|---:|---:|---:|
| map | 0.3268 | **0.3326** | **+0.0058** |
| map_50 | 0.4420 | 0.4497 | +0.0077 |
| map_75 | 0.3659 | 0.3715 | +0.0056 |
| map_small | 0.1149 | 0.1095 | −0.0054 |
| map_medium | 0.2725 | 0.2844 | +0.0119 |
| map_large | 0.3914 | 0.3915 | +0.0001 |

Per-eval `val/map` gaps (seed 1 − seed 0): −0.0074, −0.0027, +0.0084, +0.0058.

**9. Seed variance is the same size as the entire block A spread — the block A
ranking is not supportable.** Changing only the seed moved an identical config by
+0.0058, which is 62% of block A's 0.0093 best-to-worst spread, and the per-eval
gap reached ±0.0084 while changing sign twice. MAL's margin over fixed focal
(0.0063) is *smaller* than the seed delta measured on MAL itself. Nothing in this
project should be ranked from single-seed runs at differences below ~0.01; that
applies retroactively to findings 0 and 6 (VFL "loses" by 0.0070, also inside seed
noise) and to every block below as written.

**10. MAL's box-quality advantage does reproduce, and it is the one real result.**
`map_large` came out at 0.3914 and 0.3915 across the two seeds — the most
seed-stable number measured all day — against vision2 0.3722, A1 focal 0.3734,
A3 vfl 0.3829. `map_75` likewise: 0.3659 / 0.3715 versus 0.3527 / 0.3563 / 0.3527.
**Both MAL seeds beat every non-MAL arm on both metrics.** So finding 2 survives
the seed check even though finding 0's headline ranking does not. Adopt MAL for
box quality on medium-to-large objects, not for headline map.

**11. Sub-metrics are noisier than map, with one exception.** `map_medium` moved
0.0119 between seeds — larger than any loss difference in block A — and map_small
moved 0.0054. Per-sub-metric "wins" from single runs (e.g. A4's best-in-block
map_small of 0.1151 against A2's 0.1149) are noise and should be struck. Only
`map_large` was stable enough across seeds to carry an argument.

## Block B — batch size and accumulation (3 runs)

All three arms hit **effective batch 16**, all with `--grad-checkpointing` so that
batching is the only variable (bs16 cannot run without it, per the table above).
1100 optimizer steps × 16 = 17,600 samples in every arm.

| run | flags | step time | est. wall clock |
|---|---|---:|---:|
| **B1** | `--batch-size 4 --grad-accum-steps 4 --max-steps 1100` | 4×1467 ms | ~1h55 |
| **B2** | `--batch-size 8 --grad-accum-steps 2 --max-steps 1100` | 2×2946 ms | ~1h55 |
| **B3** | `--batch-size 16 --grad-accum-steps 1 --max-steps 1100` | 5856 ms | ~1h55 |

Identical wall clock, by construction — this block costs GPU-hours but no
schedule risk, and can fill idle GPU time alongside block C.

What it actually decides: whether the per-micro-batch loss normalisation matters
(see the accumulation caveat above), and whether bs16 is worth its checkpointing
tax elsewhere. It also feeds block D: mosaic tiles are drawn **from within the
batch**, so batch size caps mosaic diversity.

Use A's winner. If block B comes back flat (likely — 46-class detection at
17.6k samples is not batch-noise-limited), the answer is "use bs8 without
checkpointing, it is the cheapest," and B's real value is closing the question.

## Block C — unfreeze depth (4 runs)

The existing head-only run is at a different budget, a different `--eval-top-k`
and the broken loss, so the depth sweep starts from scratch. All arms use
`--grad-checkpointing` (mandatory above vision-blocks 2) and A's winning loss.

| run | flags | step time | wall clock @17.6k samples |
|---|---|---:|---:|
| **C0** | `--vision-blocks 0` (head only, floor) | ~2.4 s | ~1h50 |
| **C1** | `--vision-blocks 6 --vision-learning-rate 5e-6` | 4083 ms | ~2h50 |
| **C2** | `--vision-blocks 12 --vision-learning-rate 2e-6` | 5832 ms | ~4h00 |
| **C3** | `--vision-blocks 24 --vision-learning-rate 1e-6` (full) | 9319 ms | ~6h05 |

vision-blocks 2 is A's winner, already run.

### C1 result (2026-08-04) — ~~no measurable gain from deeper unfreezing~~ CONFOUNDED, see finding 15

C1 ran with MAL γ1.5, `--vision-blocks 6 --vision-learning-rate 5e-6
--grad-checkpointing --val-batch-size 16`. MLflow `owlv2_text_20260803_213605`,
**2h40** against A2's 1h58 (+35% wall clock).

| metric | A2 seed 0 (vb2) | A2 seed 1 (vb2) | **C1 (vb6)** |
|---|---:|---:|---:|
| map | 0.3268 | 0.3326 | **0.3332** |
| map_50 | 0.4420 | 0.4497 | 0.4514 |
| map_75 | 0.3659 | 0.3715 | 0.3704 |
| map_small | 0.1149 | 0.1095 | 0.1154 |
| map_medium | 0.2725 | 0.2844 | 0.2697 |
| map_large | 0.3914 | 0.3915 | **0.3958** |

Trajectory: 0.2596 → 0.3090 → 0.3262 → 0.3332 (slower start under the halved
vision LR, then a steeper climb, as expected for a larger trainable set).

**12. Tripling the trainable vision blocks buys nothing at this budget.** C1's
0.3332 sits +0.0006 above A2 seed 1 (0.3326) and +0.0035 above A2's two-seed mean
— i.e. **inside the seed range of the vb2 config it was meant to beat**, for 35%
more wall clock. The capacity hypothesis that motivated running C1 first (see the
revised ranking below) is not supported. C2/C3 should be considered dead unless
something else changes: the curve from vb2 → vb6 is flat, so vb12 and vb24 are
paying 4x and 9x the step time to explore a direction that produced no gradient of
improvement, and both were already expected to lose to C1.

> **WRONG — retracted 2026-08-04 (finding 15).** This comparison was confounded:
> C1 changed **two** variables against A2, `vision_blocks` 2→6 *and*
> `vision_learning_rate` 1e-5→5e-6. The control run C4 shows the lower vision LR
> costs −0.0050 map on its own, which was masking the depth gain. At a *fixed*
> vision LR, vb6 beats vb2 by **+0.0080 map** — above the noise floor. Depth is
> live, not dead, and the C2/C3 "drop" recommendation that follows from this
> finding must be re-opened.

**13. The one thing worth a follow-up is `map_large`.** C1 reached 0.3958 against
A2's 0.3914 / 0.3915. That +0.0044 would be unremarkable on any other metric, but
map_large is the only metric that came out seed-stable (Δ 0.0001 across the A2
pair). If large-object accuracy is the target, one more seed of C1 would settle
whether this is real. Caveat: a seed-sd estimated from n=2 is itself unreliable,
so this is a hypothesis, not a finding.

> **Confirmed 2026-08-04 (finding 14).** C1 seed 1 landed at `map_large` 0.3947
> against seed 0's 0.3958. Both C1 seeds beat both A2 seeds with no overlap.

### C1 seed repeat + the vision-LR control (2026-08-04)

Two runs, both at the standard 2200-step protocol.

- **C1 seed 1** — C1's exact config, seed 0 → 1. MLflow
  `owlv2_text_20260804_181631`, ~2h40.
- **C4 (new)** — A2's exact config with **only** `--vision-learning-rate` changed,
  1e-5 → 5e-6. `--vision-blocks 2`, no checkpointing, seed 0. MLflow
  `owlv2_text_20260804_210944`, ~2h00. This is the missing cell that makes C1
  interpretable. *On disk it is `text_checkpoints/fashionpedia_large_C2_vb2_vlr5e6_20260804`
  and `logs/train_text_C2_vb2_vlr5e6_20260804.log` — the "C2" there is a launch-time
  misnomer and **not** block C's C2 (vision-blocks 12), which has never been run.*

The 2x2, final values at the 2200-step budget:

| `map` / `map_large` | vision-lr 1e-5 | vision-lr 5e-6 |
|---|---|---|
| **vb2** | A2: 0.3268 / 0.3914, 0.3326 / 0.3915 | **C4: 0.3247 / 0.3893** |
| **vb6** | *never run* | C1: 0.3332 / 0.3958, 0.3322 / 0.3947 |

**14. `map_large` is seed-stable at convergence in both configs, and mid-training
spread is not the floor.** C1's two seeds finished 0.3958 / 0.3947 (Δ 0.0011),
matching the A2 pair's Δ 0.0001. But the same C1 pair differed by **0.0512** at
step 550, 0.0131 at 1100 and 0.0114 at 1650 before collapsing to 0.0011 at 2200 —
so a seed floor read off a partially-annealed eval overstates it by up to 50x.
Related: C1 seed 0 fell 0.4037 → 0.3958 over its last 550 steps while seed 1 *rose*
0.3923 → 0.3947, so that late drop was noise, not the larger model turning over.
**Only compare fully-annealed evals.** On overall `map`, C1's seed spread was
0.0010 against A2's 0.0058 — the ±0.006 floor is a property of a config, not a
constant.

**15. Depth is the driver, not the vision LR — and the confound hid it.** With
vision LR held at 5e-6, vb6 beats vb2 by **+0.0080 map** and **+0.0059 map_large**.
With depth held at vb2, dropping the vision LR 1e-5 → 5e-6 is worth **−0.0050 map**
and **−0.0022 map_large**, i.e. flat to mildly harmful. So C1's headline advantage
over A2 was *understated*: its own halved vision LR was working against the depth
gain it was meant to demonstrate. Finding 12's "vb6 buys nothing" is an artifact of
comparing across two changed variables at once. Caveat: C4 is n=1, so the depth
contrast is one seed against two.

**16. The obvious next run is the untested cell, vb6 @ vision-lr 1e-5.** If the two
effects are additive it should beat every 2200-step arm on both metrics; it is also
the only way to test whether they are additive rather than interacting. ~2h50.

Notes:

- **Drop the vision LR as depth grows.** vision2 used 1e-5 for 2 blocks; applying
  that to all 24 will wreck CLIP's pretrained features on 0.39 of an epoch. The
  values above are a geometric taper; layer-wise LR decay (≈0.9 per block) is the
  principled version and is worth adding if C1/C2 look promising.
- **C3 is a 6-hour run at 308M trainable params on 17.6k samples.** That is
  ~57 samples per trainable million. Expect it to lose to C1; it is worth running
  once to establish that the curve turns over, not as a candidate recipe.
- Optimizer state for full unfreeze is 2.4 GiB (AdamW fp32 moments); that is
  already in the measured 8.30 GiB peak.
- Run C1 and C2 concurrently, then C0 and C3.

## Block E — training budget (new, 2026-08-04)

Motivated by the "revised again" section below: every arm through 2026-08-03 was
still climbing at step 2200 (0.39 of an epoch), so the binding constraint looked
like budget rather than objective or capacity.

| run | flags | status |
|---|---|---|
| **E1** | A2's config at `--max-steps 6600` (3x budget), `--eval-every-steps 550` | **crashed at ~step 4828/6600** |

*On disk E1 is `text_checkpoints/fashionpedia_large_B1_mal6600_20260804` and
`logs/train_text_B1_mal6600_20260804.log`; MLflow `owlv2_text_20260804_181323`. The
"B1" there is a launch-time misnomer and **not** block B's B1 (bs4 × accum 4).*

### E1 partial result — the budget is the biggest lever found so far

E1 died at 22:33 from the MLflow lock bug (see Infrastructure below), 73% through.
Last completed eval was step 4400; the checkpoint at that step is on disk.

| step | 550 | 1100 | 1650 | 2200 | 2750 | 3300 | 3850 | 4400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| map | 0.2823 | 0.3017 | 0.2764 | 0.3191 | 0.3299 | 0.3277 | 0.3390 | **0.3443** |
| map_large | 0.3232 | 0.3523 | 0.3411 | 0.3770 | 0.3906 | 0.3799 | 0.4019 | 0.3921 |

**17. 3x budget beats every 2200-step arm, and it had not flattened when it died.**
E1's 0.3443 at step 4400 is **+0.0117 over the best 2200-step result** (A2 seed 1,
0.3326) — about 2x the noise floor, and the largest margin any change has produced
in this project. It was still climbing. Budget dominates loss choice (≤0.01,
inside noise), unfreeze depth (+0.008) and vision LR (−0.005).

**18. Do not compare a stretched-cosine run to a short one before it anneals.**
`--max-steps` sets `T_max`, so tripling the budget holds the LR near peak for
thousands of steps the short arms never spent there. LR multiplier by step:

| step | A2 (2200) | E1 (6600) |
|---|---:|---:|
| 550 | ×0.891 | ×0.988 |
| 1100 | ×0.537 | ×0.943 |
| 1650 | ×0.160 | ×0.866 |
| 2200 | ×0.000 | ×0.764 |
| 3850 | — | ×0.380 |
| 5500 | — | ×0.069 |

E1 led at 550, trailed at 1100–2200, and only pulled clear after ×0.5. Its dip at
step 1650 (0.2764, below its own step-550 value) was a high-LR wobble, not
degradation — bucketed into 275-step windows `train/L_cls` fell monotonically the
whole run (1.587 → 1.303). **Single-step loss samples are too noisy to read a trend
from; bucket before concluding anything.**

**19. Open question E1 did not get to answer.** Where the curve flattens. That
lives in the last ~1500 steps, at ×0.07 LR and below, which E1 never reached. A
clean rerun is needed — and there is **no resume**: `save_text_checkpoint` stores
weights, class names, epoch and config, but no optimizer/scaler/scheduler state, so
restarting from `last.pth` would drop a warm model into a fresh cosine at full LR
with reset Adam moments. That is not a continuation and would not be comparable.

## Block D — mosaic (3 runs)

**Correction to `loss-notes.md` Option B, measured today.** That section assumed
DEIM's mosaic: 4 images composited and downscaled into one frame, giving 4x
positives per step at half object scale, with a warning about Fashionpedia's
small boxes. The implementation in `gpu_augment.py` does **not** do that. Kornia's
`RandomMosaic` concatenates into a 2S×2S canvas and *crops* an S×S window
(`cropping_mode="slice"`, `start_ratio_range=(0.3, 0.7)`), so:

- Object scale is **preserved**, not halved. Measured on synthetic boxes of
  normalised width 0.2, mosaic output mean width was 0.169 — the shortfall is
  edge clipping, not downscaling. **The small-object concern does not apply to
  this implementation**, and `--mosaic-min-visibility 0.2` handles the clipping.
- Supervision density rises only **~1.45x** (81 boxes out of 56 in an 8-image
  batch), not 4x, because an S×S crop of a 2S×2S canvas sees one image's worth of
  area. Boxes above 1.0x come from objects straddling tile seams appearing twice.
- Each output sample mixes **2–4 source images** (measured), so what mosaic
  actually buys here is context/scale-composition diversity plus seam-crossing
  hard negatives — not the 4x density-per-FLOP argument. Re-read the Option B ROI
  claim with that in mind: at 1.45x it no longer dominates kernel work by an
  order of magnitude.

| run | flags | notes |
|---|---|---|
| **D1** | `--mosaic-prob 0.5 --mosaic-grid 2 2` | the shipped default, vs A's winner at `--mosaic-prob 0` |
| **D2** | D1 + mosaic disabled for the final 40% of steps | needs the schedule flag; DEIM says this is what recovers clean-distribution accuracy |
| **D3** | DEIM-style downscaling mosaic (`output_size=(2S,2S)` + resize to S) | only if D1/D2 win; recovers the true 4x density at the cost of halving object scale, which is the case `loss-notes.md` warned about |

Run D at whichever batch size block B picks — larger batches give the mosaic more
distinct tiles to draw from.

Also worth one cheap check, not a full run: **`--gpu-augment` parity.** It moves
ColorJitter off the CPU workers to kornia's `ColorJiggle` on device. Confirm
dataloader wait time actually drops (it should — jitter is the expensive
per-sample op) and that `--mosaic-prob 0 --gpu-augment` matches
`--no-gpu-augment` on mAP within noise. If it does not, every block-D result is
confounded by the jitter change.

> **Partly resolved 2026-08-04.** `--gpu-augment` defaults to True and **was already
> active in every arm run so far** (A1–A4, both A2 seeds, C1 both seeds, C4, E1).
> With `--mosaic-prob 0` the mosaic op is `None` but `ColorJiggle(0.3, 0.3, 0.2,
> 0.02)` still runs at p=1.0, followed by device-side normalisation, plus
> `horizontal_flip_prob=0.5` in the dataloader. Smoke-tested `BatchAugmentor`
> directly: jitter fires per-image, boxes pass through untouched with mosaic off,
> and mosaic rebuilds boxes while preserving resolution with it on. So **no existing
> arm is confounded relative to another** — they all share the same augmentation —
> and a "rerun with gpu-augs" would be a bit-identical duplicate. What remains
> untested is the `--no-gpu-augment` CPU-path comparison, which only matters if a
> block-D result is ever compared against a pre-`gpu_augment` run.

---

## Block F — the 34-class set (new, 2026-08-05)

Per-class AP from C1, grouped against train-split box counts:

| group | classes | share of boxes | mean AP |
|---|---:|---:|---:|
| main garments + accessories | 27 | 48.9% | 0.472 |
| garment parts | 9 | 45.1% | 0.232 |
| decorations | 10 | 6.0% | 0.049 |

**12 categories dropped**, leaving 34. The decorations go as a group (applique,
bead, bow, flower, fringe, ribbon, rivet, ruffle, sequin, tassel): mean AP 0.049
for 6% of the boxes, and bead/rivet are about one cell of the 14px patch grid, so
this is nearly free. **neckline** goes because it fails the annotation-consistency
test hardest while being common enough to do real damage — 34,257 boxes on 33,159
images generating false negatives across the val set. **epaulette** goes for the
same reason at much lower volume (874 boxes, 507 images, ~0.05% of image area).

`sleeve`, `collar`, `lapel`, `pocket` and `hood` are **kept**, along with `buckle`
and `zipper`. Those seven are the bulk of the 45.1% garment-parts supervision, and
per finding 17 the run is budget-limited — cutting box supervision on an underfit
model is a real risk, not a free cleanup. `sleeve` at 0.571 and `collar` at 0.271
are working.

Measured effect on supervision, both splits:

| split | anns | kept | boxes/image |
|---|---:|---:|---|
| train | 333,391 | 278,334 (83.5%) | 7.3 → 6.1 |
| val | 8,781 | 7,317 (83.3%) | 7.6 → 6.3 |

**No image is emptied by the filter** — all 45,623 train and 1,158 val images
retain at least one kept box — so the trainer needs no image-dropping logic and
epoch size is unchanged.

Implemented as `--exclude-class-names NAME [NAME ...]` in `train_text.py`, matched
on the cleaned full category name with an unmatched name raising rather than
silently keeping the class it was meant to drop. It resolves to `category_ids`
before anything else reads them, so the boxes leave the training targets *and* the
validation ground truth, and the queries leave the prompt set.

> **`val/map` from block F is not comparable to any earlier run.** 34 queries
> instead of 46, and the metric averages over a different, easier set of classes.
> Only compare block F arms to each other.

| run | change | status |
|---|---|---|
| **F1** | 34 classes, `--mosaic-prob 0`, `--max-steps 6600` | **stopped at 4036/6600 (61%)**, GPU 0 |
| **F2** | 34 classes, `--mosaic-prob 0.5 --mosaic-grid 2 2`, `--max-steps 6600` | **stopped at 3894/6600 (59%)**, GPU 1 |
| **F1b** | clean rerun of F1 | **stopped at 4503/6600 (68%)**, GPU 0 — resumable |
| **F2b** | clean rerun of F2 | **stopped at 4456/6600 (68%)**, GPU 1 — resumable |

Everything else at the standard protocol with MAL γ1.5, `--vision-blocks 2`,
`--vision-learning-rate 1e-5`, bs8, seed 0, `--compile --val-batch-size 16`,
`--eval-every-steps 550` (12 evals). MLflow `owlv2_text_20260805_210220` (F1) and
`owlv2_text_20260805_210122` (F2). Checkpoints in
`text_checkpoints/fashionpedia_large_F{1,2}_34cls_{nomosaic,mosaic}_20260805`.

These two arms answer three things at once: the E1 rerun the schedule called for
(F1 is E1's config at the reduced class set, and it carries the lock fix so it
should survive), block D's D1 at a budget where a regularizer can actually pay,
and the class-set change itself. Depth was deliberately held at vb2 rather than
taking finding 16's vb6 — changing the class set and the capacity in the same run
is precisely the two-variables-at-once confound that produced finding 15's
retraction.

### F1/F2 partial results (2026-08-06)

Both stopped by hand at ~60% for a server shutdown, after ~7 GPU-hours. Seven
evals each. Numbers below are from MLflow, not the logs — stdout is block-buffered
through the `nohup` redirect, so the last two evals of each run were still sitting
unflushed in the buffer when the processes were killed. **Read `mlflow.db`, not
the log tail, for anything time-sensitive.**

| step | 550 | 1100 | 1650 | 2200 | 2750 | 3300 | 3850 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **F1** map | 0.3889 | 0.3961 | 0.3660 | 0.4176 | 0.3954 | **0.4326** | 0.4310 |
| **F2** map | 0.3498 | 0.3674 | 0.3555 | 0.3574 | 0.3848 | 0.4103 | 0.4098 |
| F2 − F1 | −0.0391 | −0.0287 | −0.0105 | −0.0602 | −0.0106 | −0.0223 | −0.0212 |

| step | 550 | 1100 | 1650 | 2200 | 2750 | 3300 | 3850 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **F1** map_large | 0.4328 | 0.4381 | 0.4150 | 0.4631 | 0.4473 | **0.4784** | 0.4719 |
| **F2** map_large | 0.3959 | 0.4115 | 0.3965 | 0.4011 | 0.4331 | 0.4507 | 0.4542 |

Checkpoints: F1 `best.pth` = step 3300 (0.4326), `last.pth` = step 3850. F2 both
at step 3850 (0.4098; its 3300 was 0.4103, so `best.pth` is the 3300 weights).

**20. Mosaic trails at every single paired eval, but by ~0.02, not the ~0.06 one
eval suggested.** The sign is negative 7 times out of 7, which is the robust part.
The magnitude is not: the per-eval gap ranges −0.011 to −0.060, and it *narrows*
over training (−0.039 → −0.022 across the run). Reading the −0.0602 at step 2200
as the effect size was wrong — it is the widest of seven draws from a distribution
whose mean is about −0.028. A regularizer starting behind and closing is exactly
the expected shape, so **whether mosaic would have crossed over by step 6600 is
genuinely unresolved**, and D2's mosaic-off tail was never tested at all. Do not
record block D as decided.

**21. Mid-training evals oscillate ±0.03–0.05 in this schedule — five times the
annealed floor.** F1 went up, up, *down*, up, *down*, up, flat; the dips at 1650
and 2750 are 0.03–0.05 below their neighbours. This is finding 14's "mid-training
spread is not the floor" showing up again, and it kills a hypothesis raised while
the runs were live: that the step-1650 dip shared by E1/F1/F2 pointed at a hard
stretch of the seed-0 data order. F1 dips again at 2750 where F2 does not, so the
dips are not aligned between arms and the data-order story does not hold. It also
means **F1 and F2 sharing an epoch-1 sampler order does not make their comparison
tighter than the cross-seed floor** — the oscillation is not common-mode.

**22. The headline jump from 0.333 to 0.43 is mostly re-averaging, not learning.**
mAP is a macro-average over classes, so deleting the worst classes raises it
mechanically. The group means decompose C1 exactly —
`(27×0.472 + 9×0.232 + 10×0.049)/46 = 0.3332` — and re-averaging those *same*
per-class APs over the 36 classes left after dropping only the decorations gives
`(27×0.472 + 9×0.232)/36 = 0.412`, with zero learning benefit assumed. Dropping
neckline and epaulette (both below the parts mean) pushes the arithmetic baseline
higher still. F1 peaked at 0.4326. **So the class cleanup has so far bought ≲0.02
over the pure re-averaging, and possibly nothing.** That is not an argument against
the cleanup — 34k inconsistent neckline boxes are worth removing on their own
terms, and the metric now means something closer to the target — but it must not
be recorded as an accuracy win.

> **The de-confound is cheap and should be run first next session:** evaluate the
> existing **C1 checkpoint** (46-class weights) against the **34-class query set**.
> Same weights, new metric, ~3 minutes. That isolates the arithmetic component
> exactly, and F1's margin over *that* number is the real gain from the cleanup.
>
> **Run 2026-08-06 — finding 24 below. Confirmed: the cleanup bought ~nothing.**

### The de-confound — 46-class weights under the 34-class metric (2026-08-06)

Two existing checkpoints re-scored against the 34-class query set with
`--init-from ... --eval-only`, no training. Same weights, same eval code, same
`--eval-top-k 100`, only the query set and the ground-truth class set change.
~6 minutes each. C1 was the run finding 22 asked for; **A2 seed 0 was added
because C1 is vb6 @ vlr 5e-6 while F1 is vb2 @ vlr 1e-5** — comparing F1 to C1
alone would have swapped the depth confound back in for the metric one.

| weights (all trained on 46 classes) | 46-class map | **34-class map** | 34-class map_large |
|---|---:|---:|---:|
| A2 seed 0 — vb2, vlr 1e-5, 2200 steps (F1's config) | 0.3268 | **0.4167** | 0.4616 |
| C1 — vb6, vlr 5e-6, 2200 steps | 0.3332 | **0.4257** | 0.4704 |
| F1 — *trained on 34 classes*, vb2, 3300/6600 steps | — | 0.4326 | 0.4784 |

**24. Nine tenths of the headline jump is re-averaging, and what is left is not
separable from budget.** Re-scoring alone moves A2 from 0.3268 to 0.4167 —
**+0.0899 of F1's +0.1058 headline jump is pure arithmetic**, measured rather
than estimated. The residual is +0.0159 at F1's peak (step 3300) and +0.0143 at
step 3850. But F1 spent 3300 steps against A2's 2200, and finding 17 put budget
2200 → 4400 at +0.0117 on its own; pro-rating that to F1's shorter extension
leaves roughly **+0.004 to +0.010 attributable to the class-set change, from a
single seed, against a ±0.006 floor.** That is not distinguishable from zero and
is at most one sixth of the headline. Finding 22's "≲0.02 over pure re-averaging,
and possibly nothing" holds, now with the arithmetic term pinned exactly.

The cleanup is still justified — 34k inconsistent neckline boxes are worth
removing and the metric now means something closer to the target — but it is a
**metric change, not an accuracy win**, and the 0.43 number must never be quoted
against a 46-class 0.33 without this decomposition attached. Note also that a
clean attribution was never available from F1 alone: it changed the class set and
the budget together, which is finding 15's two-variables-at-once trap in a new
costume.

**25. The vb6 depth gain reproduces under the new metric.** C1 beats A2 by
**+0.0090 map** and **+0.0088 map_large** on the 34-class set, against the
+0.0080 / +0.0059 that finding 15 measured on the 46-class set at fixed vision
LR. C1 carries the *handicapped* vision LR (5e-6, worth −0.0050 per finding 15)
and still wins by more here. Two independent metrics now agree on depth, which
makes finding 16's untested cell (vb6 @ vlr 1e-5) the best-motivated remaining
config — and it should be run at the 34-class set, where the baseline is now
measured rather than inferred.

### F1b/F2b — the clean rerun pair, stopped at 4400 (2026-08-06, evening)

F1 and F2 relaunched with byte-identical configs (lifted from F1's checkpoint,
not retyped): MAL γ1.5, vb2 @ vlr 1e-5, head 5e-5, bs8, seed 0, 34 classes,
`--compile --val-batch-size 16 --eval-every-steps 550 --max-steps 6600`, and
`python -u` so the log tail is no longer swallowed by the `nohup` buffer.
MLflow `owlv2_text_20260806_194207` (F1b, GPU 0, mosaic off) and
`owlv2_text_20260806_194218` (F2b, GPU 1, mosaic 0.5). Checkpoints in
`text_checkpoints/fashionpedia_large_F{1b,2b}_34cls_{nomosaic,mosaic}_20260806`.

Both stopped by hand after 4h05, at steps 4503 and 4456, ~4 minutes past the
step-4400 eval. **Both `last.pth` carry `training_state` at `global_step` 4400**
(`best_map` 0.4484 / 0.4099), so unlike the F pair these two are resumable and
the 2,200 steps still owed cost ~2h each rather than a restart.

`val/map`:

| step | 550 | 1100 | 1650 | 2200 | 2750 | 3300 | 3850 | 4400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **F1b** no mosaic | 0.3632 | 0.3838 | 0.4056 | 0.3901 | 0.3974 | 0.4214 | 0.4268 | **0.4484** |
| **F2b** mosaic | 0.2847 | 0.3496 | 0.3681 | 0.3782 | 0.3888 | 0.3991 | 0.4038 | 0.4099 |
| F2b − F1b | −0.0785 | −0.0342 | −0.0375 | −0.0119 | −0.0086 | −0.0223 | −0.0230 | −0.0385 |

`val/map_large`:

| step | 550 | 1100 | 1650 | 2200 | 2750 | 3300 | 3850 | 4400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **F1b** | 0.3997 | 0.4283 | 0.4496 | 0.4428 | 0.4508 | 0.4587 | 0.4700 | **0.4894** |
| **F2b** | 0.3405 | 0.3894 | 0.4225 | 0.4253 | 0.4457 | 0.4387 | 0.4506 | 0.4544 |

**26. The curve still has not flattened, and step 4400 produced the largest jump
of the run.** F1b's 0.4484 is the best 34-class number recorded — +0.0158 over
F1's 0.4326 peak — and it arrived as a **+0.0216 step** over 3850, the biggest
single-eval gain anywhere in F1b's trajectory, at ×0.25 LR with 2,200 steps of
annealing left. `map_large` did the same thing (0.4700 → 0.4894, +0.0194). There
is still no evidence of a top. Finding 19's question is now unanswered after
**four** attempts and ~26 GPU-hours (E1 73% crash, F1 61%, F2 59%, F1b/F2b 68%),
but for the first time the answer is two resume-hours away rather than a full
rerun.

**27. Mosaic is behind at 15 of 15 paired evals and the gap is not closing.**
F2b trails F1b at all eight, mean **−0.032** (−0.025 dropping the step-550 point,
where mosaic's slow start is expected). Together with F1/F2's 7-for-7 that is
15 paired evals, two independent runs, one sign. **Finding 20's "narrows over
training" does not survive the repeat:** F1b/F2b's gap *widens* late, −0.0086 at
2750 → −0.0230 at 3850 → −0.0385 at 4400, and the −0.028 mean magnitude that
finding 20 estimated came back at −0.032. A crossover by 6600 now looks unlikely
on the evidence, though the last third is still untested and D2's mosaic-off tail
has never been run. Mosaic as shipped (`--mosaic-prob 0.5`, constant) should not
be adopted.

**28. F1b vs F1 is an accidental data-order repeat, and it re-measures the
mid-training floor at ±0.04.** The two share a config *and* seed 0, but the
resume work materialises the epoch permutation from a per-epoch seeded generator,
so their sample order differs (documented in the resume section — pre-2026-08-06
runs are not bit-reproducible). Per-eval F1b − F1: −0.026, −0.012, **+0.040**,
−0.028, +0.002, −0.011, −0.004. So the pair swings ±0.04 mid-training and
converges as it anneals (0.011 and 0.004 at the last two shared evals), which
confirms finding 21's oscillation on a second pair and shows it is not a property
of one data order. Practical consequence: **F1's 0.4326 peak was a favourable
draw** — F1b at the same step 3300 is 0.4214 — and any effect smaller than ~0.04
read off a single mid-training eval is unsupportable, which is most of what this
document has been trying to measure.

**Do not read F1b's step-2200 value against the re-scored baselines.** F1b at
2200 is 0.3901 against A2's re-scored 0.4167 (finding 24), which looks like
training on 34 classes being actively harmful. It is not readable: per finding 18
`--max-steps` sets `T_max`, so F1b at 2200 is at ×0.76 LR and un-annealed while
A2 finished its cosine there. The comparison only becomes meaningful at 6600.

**23. Three consecutive long runs have now failed to reach their budget** — E1 at
73% (MLflow lock), F1 at 61% and F2 at 59% (manual stop). That is ~15 GPU-hours
spent on finding 19's question, "where does the curve flatten", which is still
unanswered. Both F arms were still climbing at 3300 and flat-to-slightly-down at
3850, at ×0.38 LR with 2,750 steps of annealing left, so neither is near converged.
**Resume support is now the blocking infrastructure item, ahead of any further
experiment.** `save_text_checkpoint` stores weights, class names, epoch and config
but no optimizer/scaler/scheduler state; adding those three is roughly an hour and
would have saved all three of these runs.

> **Done 2026-08-06** — see "Infrastructure — resume" below. Note that neither F
> arm can actually be resumed: their checkpoints predate the feature and carry no
> optimizer state, so the 6600-step question still needs a run started from
> scratch. What resume buys is that the *next* interruption is not another 7
> GPU-hours.

## Infrastructure — resume (2026-08-06)

Closes finding 23. `--resume PATH` continues an interrupted run; `--init-from
PATH` loads weights only and starts a fresh optimizer; `--eval-only` scores a
checkpoint without training.

`last.pth` now also carries a `training_state`: optimizer, scheduler, AMP scaler,
`global_step`, the epoch index and position within it, `best_map`, and the torch /
CUDA / python / prompt RNG states. `best.pth` and `final.pth` deliberately do not
— the AdamW moments are twice the trainable delta, and only `--resume` reads them.
**This means `last.pth` roughly triples**: 121 MB → 363 MB at vision-blocks 2,
322 MB → ~950 MB at vision-blocks 6. The disk is at 92% (38 GB free as of
2026-08-06 evening, up from 15 GB), so that is worth watching before launching
two deep arms concurrently but is no longer close to binding.

Two details that make a resume a genuine continuation rather than a restart:

- **The epoch's sample order is fast-forwarded exactly.** `shuffle=True` draws
  from the DataLoader's own RNG and cannot be positioned mid-epoch, so the epoch
  permutation is now materialised from a per-epoch seeded generator and sliced at
  the resume point. A resumed epoch sees precisely the samples the interrupted one
  had left. The cost: the sample order for a given seed differs from any run
  launched before this change, so pre-2026-08-06 runs are not bit-reproducible.
- **The resumed process rejoins the original MLflow run**, so `val/map` stays one
  series rather than being split across two runs that have to be stitched by hand.
  Train metrics between the last checkpoint and the crash are logged twice at
  those steps, once per attempt.

A resume whose config differs from the run it continues is refused
(`RESUME_CRITICAL_ARGS`), because the optimizer state is keyed by parameter
position, the scheduler by step count, and the learning rates live inside the
restored optimizer state — so a changed `--vision-learning-rate` would be silently
ignored rather than applied. Resuming from `best.pth` is refused for the same
reason it is small: no training state.

Verified end to end on a 24-step run interrupted at step 12 and resumed:
optimizer moments continued (AdamW's internal update counter went 5 → 17 rather
than restarting), scheduler advanced 12 → 24, the resumed epoch had exactly
11 − 1 = 10 batches left, and both negative cases (changed `--vision-blocks`,
`best.pth`) were rejected.

**Not restored:** dataloader-worker RNG. A worker re-seeded at the start of a
truncated epoch reaches a given sample at a different point in its augmentation
stream, so the augmentations after a resume differ from the uninterrupted run.
That is noise-equivalent, not a bias, but it does mean a resumed run is not
bit-identical to an uninterrupted one.

Also worth carrying forward: launch with `python -u`. The block-buffered `nohup`
stdout that hid F1/F2's last two evals is a buffering artifact, and `-u` removes
it — the log tail becomes usable again, though `mlflow.db` remains the
authoritative source.

## Infrastructure — mosaic was broken for every batch it did not fully cover (2026-08-05)

**`--mosaic-prob 0.5` could never have run.** Found by smoke-testing block D's
config before launching F2; the first training step raised
`RuntimeError: The size of tensor a (2) must match the size of tensor b (4)`.

kornia's `RandomMosaic` emits `_params["permutation"]` with **one row per applied
sample**, indexed by rank among the applied samples — not one row per batch entry.
`gpu_augment._apply_mosaic` read those rows as batch-indexed. Consequences:

- `0 < applied < batch_size`: `permutation[:, k]` is shorter than the batch, so
  every gather against it either raises or silently misaligns labels. At
  `--mosaic-prob 0.5` this is the overwhelmingly common case.
- `applied == 0`: kornia additionally skips the per-cell box expansion and returns
  a `B x max_boxes x 4` tensor instead of `B x (cells*max_boxes) x 4`, which breaks
  the size filter's broadcast.
- `applied == batch_size` (i.e. `mosaic_prob=1.0`): rank order equals batch order,
  so the old code was correct. **Every existing test used `mosaic_prob=1.0`**, and
  so did the block D measurements above — which is why this survived, and why
  those measurements stand.

Fixed by scattering the applied rows back onto their batch positions with an
identity fill for the skipped samples (whose own boxes kornia leaves in block 0,
which the existing `applied[:, None] if k else True` mask already handles), plus an
early return for the nothing-applied case. Both the block layout and the
rank-ordering were confirmed empirically against kornia 0.8.3 with solid-colour
source images before the fix was written, rather than assumed.

Regression tests added in `tests/test_gpu_augment.py`: label-to-box attachment at
`mosaic_prob` 0.25/0.5/0.75 over 12 seeds each (asserting a partially-mosaicked
batch actually occurred, and that skipped samples come back byte-identical), and
the nothing-applied passthrough. All four fail on the pre-fix code.

## Infrastructure — the MLflow SQLite lock (2026-08-04)

**E1 lost ~4.5 GPU-hours to this. Fixed, but read before launching parallel arms.**

Two trainers writing `mlflow.db` concurrently (plus read queries against it for
analysis) hit `sqlite3.OperationalError: database is locked`. SQLite's default
rollback journal lets a reader block a writer, and MLflow's internal retry does not
always outlast the contention. The per-step `mlflow.log_metrics` call in
`train_text.py` had no exception handling and runs once per optimizer step, so a
single transient lock propagated out of `main()` and killed the run.

Four earlier non-fatal lock tracebacks appeared in E1's log before the fatal one.
**Treat any lock traceback as the run being at risk**, not as evidence the failure
class is benign — that inference was made during the run and was wrong.

Fixes applied:

1. `log_metrics()` helper in `train_text.py` wraps `mlflow.log_metrics` in
   `try/except MlflowException`, reports to stderr and continues. All metric call
   sites route through it. `mlflow.log_params` at startup is deliberately left
   unguarded — failing loudly there is cheap. Verified both ways: a simulated
   `database is locked` is swallowed, a `ValueError` still propagates.
2. `mlflow.db` set to `PRAGMA journal_mode=WAL` (persistent), so readers and one
   writer coexist.

When querying `mlflow.db` while runs are active, connect read-only
(`file:mlflow.db?mode=ro`) and keep polling minimal. Note `busy_timeout` is
per-connection and does **not** persist in the file; only `journal_mode` does.

~~**Still missing: resume support.**~~ Added 2026-08-06; see "Infrastructure —
resume" above. A crash at hour 5 of a 6-hour run now costs at most one
`--eval-every-steps` interval, since `last.pth` is written at every eval.

## Schedule

Sequential by dependency, two GPUs in parallel within a block:

| phase | runs | GPU-hours | wall clock |
|---|---|---:|---:|
| 0 | ~~`--compile`, `drop_last`, `--grad-accum-steps`, `--val-batch-size`~~ done; mosaic schedule outstanding | — | code only |
| 1 | ~~A1 + A2~~ done (2026-08-03, 4.0 GPU-h, 2h00 wall) | — | — |
| 1b | ~~A3 + A4~~ done (2026-08-03, 4.0 GPU-h, 1h59 wall) | — | — |
| 2 | ~~C0~~ not run + ~~seed repeat of A2~~ done (2026-08-03) | — | — |
| 3 | ~~C1~~ done (2026-08-03); C2 not run | — | — |
| 3b | ~~C1 seed 1 + C4~~ done (2026-08-04, 4.7 GPU-h) | — | — |
| 3c | ~~E1 (3x budget)~~ **crashed at 73%**, 4.5 GPU-h lost | — | — |
| 4 | ~~F1 + F2~~ (34 classes, 6600 steps, mosaic off/on) **stopped at ~60%**, 2026-08-05, ~7 GPU-h | — | — |
| 4b | ~~F1b + F2b~~ (clean rerun of the same pair) **stopped at 4400/6600**, 2026-08-06, 8.2 GPU-h | — | — |
| 4c | **`--resume` F1b + F2b to 6600** ← the cheapest open question in the document | 4.0 | ~2h00 |
| 5 | ~~B1/B2/B3~~ dropped | — | — |
| 6 | C5 (vb6 @ vlr 1e-5) at the 34-class set and F's budget | 3.5 | ~3h30 |
| 7 | ~~D2 (mosaic-off tail) if F2 wins~~ — finding 27 makes a mosaic win unlikely; demote | 2.2+ | ~2h20+ |
| | | **~8 remaining** | **~5h30** |

Phase 0's ~15% throughput win is not in those numbers; applying it first pays for
itself several times over.

If the budget is tighter than that, the ranking by information-per-hour is:
**A (loss) → D1/D2 (mosaic) → C1 (6-block unfreeze) → B (batching)**. Block B is
last because the measured step times say all three arms cost the same and the
plausible outcome is "no difference"; it is a question worth closing, not a
lever worth pulling first.

**Revised after block A wave 1 (2026-08-03).** The loss change is worth about
+0.009 mAP (MAL over vision2), against a ±0.005 per-eval noise floor — real, but
an order of magnitude smaller than this document assumed when it called block A
"the highest-leverage change in the diff" that "decides everything downstream".
It does not decide everything downstream: the box terms are identical across arms
and the spread between the best and worst loss is ~0.01. Downstream blocks should
adopt MAL and stop treating the choice as load-bearing.

That makes **C1 (6-block unfreeze) the most promising remaining lever** — it is
the only untested change that alters model capacity rather than gradient
weighting, and vision2's curve was still rising at the budget cap, which points at
underfitting rather than a loss-shape problem. Suggested revised order:
**C1 → D1/D2 → seed repeats → B**. The strongest argument for that reordering is
in the block A results themselves: all four arms land within 0.01 of each other
while the budget-limited curve has not flattened, so capacity and data diversity
are the binding constraints, not the classification objective.

### Revised again after C1 and the seed repeat (2026-08-04)

**C1 was the right experiment and it came back negative** (finding 12): vb6 landed
inside vb2's seed range. So the capacity half of that hypothesis is dead, and the
objective was already dead (finding 0/9). Two of the three candidate levers have
now been tested and neither moves `val/map` outside noise.

What that leaves is **the budget itself**. Every arm run today — nine of them,
across three losses, two unfreeze depths and two seeds — was still climbing at
step 2200, and all of them land in 0.3175–0.3332. At 17,600 samples the model sees
**0.39 of one epoch**, so the dominant term is almost certainly "not enough
optimizer steps", which no amount of objective shaping or unfrozen depth can fix.

**Recommended next run, in preference to any remaining 2-hour arm:** one long run
at the best-known config (MAL γ1.5, vb2, bs8) at **3x budget — `--max-steps 6600`,
~6 h** — with `--eval-every-steps 550` kept so the curve is dense. That answers
where the curve actually flattens and gives every future comparison a budget worth
running at. Comparing configs at a budget where all of them are underfit by 2.5x is
what produced a day of results inside the noise floor.

Concretely, the remaining blocks should be re-scoped as:

- **C2/C3: drop.** Finding 12 kills the depth sweep.
- **B (batching): drop or defer indefinitely.** It was already last on
  information-per-hour, and its plausible outcome was "no difference" — which is
  now the measured outcome of every other block.
- **D (mosaic): still worth running,** but at the longer budget, since mosaic is a
  regularizer and 0.39 epochs is far too short for regularization to pay.
- **Anything kept must budget two seeds** (protocol note above).

### Revised again after 2026-08-04 (E1, C1 seed 1, C4)

Three runs today: E1 (3x budget, crashed at 73%), C1 seed 1, and C4 (the vision-LR
control). Net effect — **one prior conclusion retracted, and the budget hypothesis
confirmed as the strongest lever.**

Everything measured, best-to-worst by effect size on `val/map`:

| lever | effect | status |
|---|---:|---|
| **Budget 2200 → 4400 steps** | **+0.0117** | finding 17, still climbing when it died |
| Unfreeze depth vb2 → vb6 (fixed LR) | +0.0080 | finding 15, n=1 vs n=2 |
| Vision LR 1e-5 → 5e-6 (fixed depth) | −0.0050 | finding 15, mildly harmful |
| Classification loss (4 arms) | ≤0.0093 | findings 0/9, inside seed noise |
| Seed, same config | ±0.0058 | finding 9 — the floor everything above is read against |

Priority order for the next session:

1. **Rerun E1 clean** (~6h, GPU 0). The lock fix is in, so it should survive.
   This is the only open question with a >0.01 effect, and it answers where the
   curve flattens — which every future comparison needs in order to pick a budget.
2. **C5: vb6 @ vision-lr 1e-5** (~2h50, GPU 1, finding 16). Fills the last cell of
   the 2x2 and tests whether depth and LR are additive. Fits inside E1's window.
3. **A second seed of C4** if the depth claim is going to carry weight — finding 15
   currently rests on one seed against two.
4. **Then D (mosaic)** at whatever budget step 1 establishes, not at 2200.

Re-opened by finding 15: **C2/C3 are no longer clearly dead.** The depth curve is
+0.0080 from vb2 → vb6 at fixed LR, not flat as finding 12 claimed, so vb12 is worth
one run *after* the budget question is settled — at the right budget and at vision-lr
1e-5, not 2e-6. Do not run the original C2/C3 configs as written; their tapered LRs
are exactly the confound that produced the retraction.

Still true from yesterday: block B stays dropped, and nothing below ~0.01 should be
ranked from single-seed runs.

### Revised again after 2026-08-06 (the de-confound + resume)

No training ran today. Two ~6-minute evals and one infrastructure item.

Updated lever table, now that the 34-class metric has a measured baseline:

| lever | effect | status |
|---|---:|---|
| **Class-set 46 → 34, re-averaging only** | **+0.0899** | finding 24, exact — an artifact of the metric, not learning |
| Budget 2200 → 4400 steps | +0.0117 | finding 17, still climbing when it died |
| Unfreeze depth vb2 → vb6 | +0.0080 (46cls) / +0.0090 (34cls) | findings 15 and 25, now reproduced on two metrics |
| Mosaic on vs off | −0.028 mean over 7 paired evals | finding 20, sign robust, magnitude not |
| Class-set 46 → 34, *training* on it | +0.004 to +0.010 | finding 24, n=1, not separable from budget |
| Seed, same config | ±0.0058 | finding 9 — the floor everything above is read against |

**The ordering is unchanged but better supported: budget first, then depth.** What
changed is that depth is now the *best-evidenced* lever (two metrics agree, and C1
won while carrying a handicapped vision LR), while the class-set change has been
demoted from "apparent +0.10" to "not measurable".

Priority for the next session:

1. **A clean 6600-step run at the 34-class set** (~6h). Still finding 19's
   question — where the curve flattens — unanswered after three failed attempts
   and ~15 GPU-hours. Resume now exists, so an interruption costs ≤550 steps
   instead of the whole run. Launch with `python -u`.
2. **C5: vb6 @ vision-lr 1e-5 at the 34-class set** (~3h, other GPU, findings 16
   and 25). The last untested cell of the depth × LR 2x2, and now the
   best-motivated config in the document. Its baseline is A2's re-scored 0.4167.
3. **D2 (mosaic-off tail)** only if the long run leaves time — finding 20 leaves
   mosaic genuinely unresolved, but it is the weakest of the three.

Do not re-run the F arms as a pair to settle the class-set question; finding 24
answers it as well as it can be answered without spending another 10 GPU-hours on
a difference that is at most one noise unit.

### Revised again after 2026-08-06 evening (F1b/F2b)

Priority 1 above was launched and ran 4h05 on both GPUs before being stopped by
hand at step 4400 of 6600 (68%). Unlike every earlier interruption, **this one is
recoverable**: both `last.pth` hold optimizer, scheduler, scaler, RNG and step
state, so `--resume` finishes the pair for ~2h per GPU.

| lever | effect | status |
|---|---:|---|
| **Class-set 46 → 34, re-averaging only** | **+0.0899** | finding 24, exact — an artifact of the metric, not learning |
| **Budget 2200 → 4400 steps** | **+0.0583** (34cls, F1b 0.3901 → 0.4484) | finding 26, biggest jump was the *last* eval; +0.0117 on 46cls per finding 17 |
| Mosaic on vs off | **−0.032** mean over 8 paired evals, 15/15 negative across two pairs | finding 27, sign settled, and the gap widens late |
| Unfreeze depth vb2 → vb6 | +0.0080 (46cls) / +0.0090 (34cls) | findings 15 and 25, two metrics agree |
| Class-set 46 → 34, *training* on it | +0.004 to +0.010 | finding 24, n=1, not separable from budget |
| Data order, same config + seed | ±0.04 mid-training, ~0.005 annealed | finding 28 — the floor everything above is read against |

Two changes to how this table should be read. **Budget's number is inflated by
the schedule, not just by learning** — F1b's 2200 sits mid-cosine at ×0.76 LR
(finding 18), so +0.0583 is "annealed 4400 vs un-annealed 2200" and the honest
statement is only that the curve was still climbing steeply at 68%. And **the
noise row is now ±0.04, not ±0.006, for anything read before the anneal**
(finding 28); the ±0.006 figure applies only to fully-annealed evals.

Priority for the next session:

1. **`--resume` F1b and F2b to 6600** (~2h each, both GPUs, in parallel). Four
   attempts and ~26 GPU-hours have not reached a converged 34-class number; this
   is the first time finishing costs two hours instead of six. It settles finding
   19 (where the curve flattens), gives every future comparison an annealed
   baseline, and closes block D properly by letting the mosaic arm anneal.
   Command: `--resume text_checkpoints/fashionpedia_large_F1b_34cls_nomosaic_20260806`
   with the launch config otherwise unchanged (`RESUME_CRITICAL_ARGS` refuses a
   mismatch, and the LRs live inside the restored optimizer state).
2. **C5: vb6 @ vision-lr 1e-5 at the 34-class set** (~3h, findings 16 and 25),
   after the resumes free the GPUs — and now at the 6600-step budget, against
   F1b's annealed number rather than A2's re-scored 0.4167.
3. **D2 (mosaic-off tail)** — demoted. Finding 27 puts constant mosaic behind at
   15 of 15 paired evals with a widening late gap, so the remaining question is
   narrower than "does mosaic help": it is only whether a mosaic-off tail rescues
   an arm that is otherwise losing. Not worth a GPU before C5.

Also worth carrying forward from this pair: `python -u` worked — the log tail
tracked the run in real time — but every number in the F1b/F2b tables was still
read from `mlflow.db`, which remains the authoritative source.
