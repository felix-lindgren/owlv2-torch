# OWLv2-large on Fashionpedia — what we learned

Five days of experiments (2026-08-02 → 2026-08-07), ~42 GPU-hours on 2x RTX 3060,
fine-tuning `google/owlv2-large-patch14-ensemble` for text-conditioned detection
on Fashionpedia. This is the condensed version of `docs/run-plan.md`, which holds
the full trajectories, the retractions and the per-run detail.

Read this first if you are picking the project up. The short version: **the
accuracy levers were mostly duds, the methodology lessons were the real output,
and the one number that matters is that a 3-hour careless comparison and a
3-hour careful one cost the same and only one of them means anything.**

## Where it ended up

| | 46-class metric | 34-class metric |
|---|---:|---:|
| zero-shot, no training | 0.2404 | — |
| best trained (46cls): C1, vb6, 2200 steps | 0.3332 | 0.4257 (re-scored) |
| **best trained (34cls): F1b, vb2, 6600 steps** | — | **0.4488** |

The two columns are different metrics over different class sets and **must never
be quoted against each other** — see "class-set arithmetic" below.

Final recipe: `--model-type large --class-loss mal --class-loss-gamma 1.5
--vision-blocks 2 --vision-learning-rate 1e-5 --head-learning-rate 5e-5
--batch-size 8 --warmup-steps 100 --negative-ratio 20 --eval-top-k 100
--mosaic-prob 0 --amp --augment --compile`, 34 classes, ~4400 steps.

## What worked

**1. Training budget — the biggest real lever, until it saturates.**
Every arm through 2026-08-04 was still climbing at 2200 steps (0.39 epoch), and
extending the run was worth more than any objective or capacity change.
2200 → 4400 was +0.0117 (46cls). But it stops: 4400 → 6600 bought **+0.0004**.
The curve flattens around 4400 steps at this config, and beyond that budget is
not a lever at all.

**2. Unfreeze depth, vb2 → vb6: +0.0080 (46cls), +0.0090 (34cls).**
The only lever that reproduced on two independent metrics, and it won while
carrying a *handicapped* vision LR. Costs ~35% wall clock. This is the
best-evidenced positive result in the project and the vb6 @ vlr 1e-5 cell is
still untested.

**3. MAL classification loss — adopt, but for box quality only.**
`map_large` came out 0.3914 / 0.3915 across two seeds against focal's 0.3734,
and both MAL seeds beat every non-MAL arm on `map_75` and `map_large`. The
headline `map` advantage (+0.0063) is inside seed noise and should not be quoted.

**4. Infrastructure, which paid for itself repeatedly.**
`--compile` + `drop_last` gave ~8% end-to-end. MLflow WAL + a guarded
`log_metrics` stopped a SQLite lock from killing runs (it had already cost 4.5
GPU-hours). `--resume` turned a 68%-complete interruption from a 6-hour rerun
into a 2-hour finish — it recovered this project's only annealed result.

## What didn't work

**1. The classification loss, as a category.** Four arms (focal fixed, MAL, VFL,
MAL without objectness) landed within 0.0093 of the *broken-focal* baseline, for
~8 GPU-hours. That spread is smaller than the seed noise measured on one of the
arms. The block was designed as "the highest-leverage change in the diff"; it was
not.

**2. Fixing a real bug bought nothing.** The focal loss had `reduction="mean"`
diluting the class term 46x — it was 0.8% of the gradient. The fix was correct
and moved mAP +0.0030, i.e. nothing. A genuine bug is not automatically a
performance opportunity.

**3. Mosaic (`--mosaic-prob 0.5`) — RETRACTED 2026-08-08, the arms were buggy.**
Behind at **19 of 19 paired evals** across two independent run pairs, −0.0208 at
the annealed 6600 and negative on all six sub-metrics.

**This comparison is invalid.** kornia's `RandomMosaic` mistranslates boxes on a
partially mosaicked batch, and at `--mosaic-prob 0.5 --batch-size 8` ~99% of
batches are partial. `F2_34cls_mosaic` and `F2b_34cls_mosaic` both ran that
configuration with `--gpu-augment`, so roughly half of every batch carried boxes
at the wrong offsets. The measured cost was **706 of 1,932 retained boxes wrong**
at `p=0.5` versus 0 of 2,286 at `p=1.0`. F1/F1b at `mosaic_prob=0` are unaffected,
so what the 19 evals actually compared was *mosaic-with-corrupted-boxes* against
*no mosaic*. See `docs/lvmhp-findings.md` for the diagnosis and fix.

The original caveat also still stands: at 0.39–1.16 epochs a regularizer has
almost no room to pay. **Mosaic is an open question, not a settled negative.**

**4. The class-set cleanup, as an accuracy win.** Dropping 12 weak classes moved
the headline 0.3332 → 0.43, but **+0.0899 of that +0.1058 was pure re-averaging**,
measured by re-scoring the old weights on the new class set. The residual was not
separable from the budget change bundled into the same run. The cleanup was still
worth doing — 34k inconsistent `neckline` boxes are bad supervision — but it is a
metric change, not learning.

**5. Lowering the vision LR: −0.0050.** Mildly harmful, and it confounded the
first depth result badly enough to force a retraction.

**6. Never run:** batch-size/accumulation (block B, dropped — all three arms were
predicted and measured to cost the same wall clock with "no difference" the
likely outcome), vb12/vb24, the mosaic-off tail (D2), the DEIM-style downscaling
mosaic (D3), and head-only vb0 at a matched budget.

## Per-class breakdown of the final model

From `tools/visualize_text_checkpoint.py` on F1b `final.pth` (34 classes, step
6600). The tool reproduces the training metric to within 0.0004 — map 0.4492 here
against 0.4488 logged, and 0.4285 vs 0.4280 for F2b — so the two passes agree.

Best: `umbrella` 0.877, `dress` 0.796, `hat` 0.774, `pants` 0.772, `glasses` 0.759,
`shoe` 0.692.
Worst: `cape` 0.000, `leg warmer` 0.000, `jumpsuit` 0.072, `pocket` 0.111,
`zipper` 0.134, `lapel` 0.162.

Two things fall out, and the second is a warning:

**Garment parts are still the weak spot, and the decision to keep them was
defensible but is not looking good.** `pocket` (0.111 on 541 val boxes), `zipper`
(0.134 / 194) and `lapel` (0.162 / 135) have enough validation support that these
are real failures, not sampling noise. They were deliberately retained when the
12 decoration classes were dropped, on the grounds that cutting box supervision
from an underfit model is risky. Now that budget is known to saturate at ~4400
steps, that rationale is weaker and a second cleanup pass is worth considering —
subject to the re-scoring protocol, since dropping them raises mAP mechanically.

**Per-class AP on a handful of boxes is meaningless, and this dataset proves it
exactly.** `cape` scored **0.000** and `umbrella` scored **0.877** — and both have
**5 validation boxes**. Same sample size, opposite extremes. Any per-class number
backed by fewer than a few dozen boxes should be treated as a coin flip, and
macro-averaged mAP inherits that variance from every rare class it contains.

## The methodology lessons — the actual deliverable

These cost more to learn than any of the accuracy results, and they are what
should carry to the next dataset.

**1. Measure the noise floor before ranking anything.** Seed variance on an
identical config was **±0.006 annealed** and **±0.04 mid-training**. Block A
ranked four losses inside its own error bar and had to be retracted. A
single-seed difference below ~0.01 carries no information. Budget two seeds, or
restrict claims to metrics shown to be seed-stable (`map_large` was;
`map_medium`, at 0.0119 between seeds, was not).

**2. Only compare fully-annealed evals.** `--max-steps` sets the cosine `T_max`,
so a 6600-step run at step 2200 sits at ×0.76 LR while a 2200-step run has
finished annealing. Comparing them measures the schedule, not the config. The
same pair of runs differed by 0.0512 at step 550 and 0.0011 at 2200.

**3. Change one variable per run.** The first depth result (vb2 → vb6) moved
*both* depth and vision LR, concluded "depth buys nothing", and was retracted a
day later when the control run showed the LR change was masking a real +0.0080
depth gain. This trap recurred in a second costume when a later run changed the
class set and the budget together.

**4. mAP is class-set arithmetic.** It macro-averages over classes, so deleting
weak classes raises it mechanically. **Always re-score existing weights on the
new class set before claiming a win** — it takes ~6 minutes and it converted an
apparent +0.10 into "not measurable".

**5. Read the metrics store, not the log tail.** Block-buffered `nohup` stdout
swallowed the last two evals of two runs. `python -u` fixes the tail, but
`mlflow.db` stayed authoritative throughout. Query it read-only
(`file:mlflow.db?mode=ro`) while runs are live.

**6. Bucket noisy series before reading a trend.** Single-step loss samples
suggested degradation where 275-step buckets showed monotonic improvement.

**7. Negative results are worth recording precisely.** Roughly 30 of the 42
GPU-hours produced "no measurable difference". That is a legitimate output, but
most of it was spent on levers that a noise-floor measurement would have shown
were unresolvable at that budget.
