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

### C1 result (2026-08-04) — no measurable gain from deeper unfreezing

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

**13. The one thing worth a follow-up is `map_large`.** C1 reached 0.3958 against
A2's 0.3914 / 0.3915. That +0.0044 would be unremarkable on any other metric, but
map_large is the only metric that came out seed-stable (Δ 0.0001 across the A2
pair). If large-object accuracy is the target, one more seed of C1 would settle
whether this is real. Caveat: a seed-sd estimated from n=2 is itself unreliable,
so this is a hypothesis, not a finding.

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

---

## Schedule

Sequential by dependency, two GPUs in parallel within a block:

| phase | runs | GPU-hours | wall clock |
|---|---|---:|---:|
| 0 | ~~`--compile`, `drop_last`, `--grad-accum-steps`, `--val-batch-size`~~ done; mosaic schedule outstanding | — | code only |
| 1 | ~~A1 + A2~~ done (2026-08-03, 4.0 GPU-h, 2h00 wall) | — | — |
| 1b | ~~A3 + A4~~ done (2026-08-03, 4.0 GPU-h, 1h59 wall) | — | — |
| 2 | C0 (+ seed repeat of A2, recommended) | 4.0 | ~2h10 |
| 3 | C1 + C2 | 6.8 | ~4h00 |
| 4 | C3 + B1 | 8.0 | ~6h05 |
| 5 | B2 + B3 | 3.8 | ~1h55 |
| 6 | D1 + D2 | 4.4 | ~2h20 |
| 7 | D3 (conditional) | 2.2 | ~2h20 |
| | | **~36** | **~23 h** |

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
