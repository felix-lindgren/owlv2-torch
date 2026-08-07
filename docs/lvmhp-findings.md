# Findings — OWLv2-base on LV-MHP-v1

Results against `docs/lvmhp-run-plan.md`. Runs live in the `OwlV2-LVMHP` MLflow
experiment; `mlflow.db` is authoritative and the `logs/` files are transcripts.

## Phase 0 — calibration

### L0a — step time and memory (2026-08-07)

`tools/probe_step_cost.py`, RTX 3060 (11.6 GiB usable), base, AMP, 18 classes,
26.2 boxes/image read from the real train annotations, GPU augmentation on,
mosaic off. The measured step reproduces the trainer's inner loop including the
pinned H2D image copy, the text path, Hungarian matching, backward, clipping and
the optimizer step; only the dataloader is excluded.

| vb | bs | compile | step ms | ms/sample | peak reserved | 3600-img epoch | 4500 steps |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 8 | off | 644 | 80 | 2.50 GiB | 4.8 min | 0.81 h |
| 0 | 16 | off | 1264 | 79 | 4.50 GiB | 4.7 min | 1.58 h |
| 0 | 32 | off | 2480 | 78 | 8.34 GiB | 4.6 min | 3.10 h |
| 2 | 8 | off | 833 | 104 | 4.72 GiB | 6.2 min | 1.04 h |
| 2 | 16 | off | 1645 | 103 | 8.64 GiB | 6.2 min | 2.06 h |
| 2 | 32 | off | **OOM** | | | | |
| 6 | 8 | off | 1183 | 148 | 8.76 GiB | 8.9 min | 1.48 h |
| 6 | 16 | off | **OOM** | | | | |
| 0 | 8 | **on** | 585 | 73 | 2.50 GiB | 4.4 min | 0.73 h |
| 0 | 16 | **on** | 1159 | 72 | 4.50 GiB | 4.3 min | 1.45 h |
| 0 | 32 | **on** | 2257 | 70 | 8.34 GiB | 4.2 min | 2.82 h |
| 2 | 8 | **on** | 751 | 94 | 4.04 GiB | 5.6 min | 0.94 h |
| 2 | 16 | **on** | 1482 | 93 | 7.30 GiB | 5.6 min | 1.85 h |
| 2 | 32 | **on** | 2898 | 90 | 11.31 GiB | 5.4 min | 3.62 h |
| 6 | 8 | **on** | 1051 | 131 | 6.91 GiB | 7.9 min | 1.31 h |
| 6 | 16 | **on** | **OOM** | | | | |

**The cost model in the run plan was right.** Predicted ~700–1,000 ms/step at
vb2/bs8; measured 833 eager, 751 compiled. Against large's 2,855 ms that is
**3.4x faster**, matching the predicted 3–4x — the vision tower's 5.3x FLOP
reduction is diluted exactly as expected by the terms that do not shrink.

**`--compile` is worth more on base than on large.** 1.10x at vb0, **1.11x at
vb2**, 1.13x at vb6, versus 1.10x measured on large. Implementation item 4 is
settled: keep `--compile`. The ~10 s one-off cost is 0.3% of a 4,500-step run.

Compile also **reduces peak memory** once blocks are trainable: 4.72 → 4.04 GiB
at vb2 (−14%) and 8.76 → 6.91 GiB at vb6 (−21%), while vb0 is unchanged. The
saving is in activations stored for backward, which is why it appears only where
there is a backward pass through the encoder. It is enough to turn vb2/bs32 from
an OOM into a (very tight) 11.31 GiB fit.

**Batch size is not a live question after all.** The run plan flagged it as
re-opened if bs16/32 fit — they do, and it buys nothing. Time per *sample* is
flat across bs8→bs32: 73 → 70 ms at vb0, 94 → 90 ms at vb2. That is **≤4%** for
4x the memory. The card is already saturated at bs8, which is the same
GEMM-bound conclusion `docs/optimization-notes.md` reached on large. **Keep
`--batch-size 8`**; a larger batch would cost a re-tuned learning rate for a
rounding error of throughput.

Grad checkpointing (compile on, vs. the compiled non-checkpointed rows above):

| vb | bs | step ms | ms/sample | peak reserved | vs. no ckpt |
|---:|---:|---:|---:|---:|---|
| 2 | 8 | 797 | 100 | 2.67 GiB | +6% time, −34% memory |
| 2 | 16 | 1580 | 99 | 4.51 GiB | +7% time, −38% memory |
| 6 | 8 | 1210 | 151 | 3.25 GiB | +15% time, −53% memory |
| 6 | 16 | 2395 | 150 | 5.54 GiB | rescues the OOM |

Cheaper than the ~20% the flag's help text claims, and it does rescue vb6/bs16.
But it buys memory, and memory only buys batch size, which buys ≤4%. At vb6 it
is a net loss: 150 ms/sample checkpointed at bs16 against 131 unchecked at bs8.
**Leave `--grad-checkpointing` off.** Its one remaining use is fitting two runs
on one card (vb2/bs8 drops to 2.67 GiB), which is unnecessary with two GPUs.

Implementation item 3 is closed: `tools/probe_step_cost.py` replaces the missing
`tools/probe_grad_ckpt_memory.py` and covers both axes.

**Settled protocol from L0a:** `--batch-size 8 --compile --no-grad-checkpointing`
at whatever `--vision-blocks` a phase needs. A 4,500-step run is ~56 min of
training at vb2.

### L0b — zero-shot baseline, 18 vs 15 classes (2026-08-07)

`--eval-only` on the 400-image val split, no training, identical weights. The
15-class arm merges the three lateral pairs with the new `--merge-class-names`.

| metric | 18 classes | 15 classes (lateral merged) | Δ |
|---|---:|---:|---:|
| **mAP @[.5:.95]** | **0.2348** | **0.2950** | **+0.0602** |
| mAP @.50 | 0.3734 | 0.4730 | +0.0996 |
| mAP @.75 | 0.2489 | 0.3116 | +0.0627 |
| mAP small | 0.0521 | 0.0723 | +0.0202 |
| mAP medium | 0.1454 | 0.1882 | +0.0428 |
| mAP large | 0.2472 | 0.3084 | +0.0612 |

**Merging the lateral pairs is worth +0.060 mAP before a single gradient step.**
Risk A in the run plan is now quantified, and it is larger than every real effect
Fashionpedia found in five days (the best lever there was +0.008). Two mechanisms,
both arithmetic: the macro-average loses three weak classes and gains three
stronger ones, *and* a detection that picks the wrong side stops being a false
positive plus a false negative.

The rule this forces: **any phase-2 number on the 15-class set must be compared
against 18-class weights re-scored on the 15-class set**, never against an
18-class mAP. `--eval-only --init-from <ckpt> --merge-class-names ...` does
exactly that re-scoring and costs ~30 s.

For scale, Fashionpedia's zero-shot on large over 46 classes was 0.2404. LV-MHP
at 0.2348 on **base** over 18 classes starts from a comparable place with a model
that is 3.4x cheaper per step — a different dataset, so not a like-for-like
comparison, but it does say the headroom above zero-shot is worth chasing here.

Evaluation costs **~30 s** (25 batches at bs16), against a 5.6 min epoch. The run
plan's "dense evals are essentially free" holds: `--eval-every-steps 225` adds
~9% to a 4,500-step run for 20 points of curve.

### Dataset defect, found the hard way during L0c

The first L0c attempt died at step 224 with `OSError: image file is truncated
(19 bytes not processed)` raised inside a dataloader worker. A scan of all 4,000
images found **exactly one** bad file: `train/3492.jpg`, 19 bytes short of a
complete JPEG. The val split is clean.

`prototype_train/train.py` now sets `ImageFile.LOAD_TRUNCATED_IMAGES = True`,
padding the missing scanline instead of raising. Dropping the image was the
alternative and costs ~25 boxes of supervision over a few pixels of bottom edge.
Note the failure mode: this surfaced 4 minutes in because the file is common
enough to be hit in epoch 1, but the same class of defect in a rarer sample
would have taken down a run hours later.

### L0c — noise floor, and the shape of the curve (2026-08-07)

Two runs of the run plan's protocol at the phase-1 budget, seeds 0 and 1,
identical in everything else, one per GPU, 66.8 and 62.3 min. Because they are
also the L1 configuration, this doubles as phase 1's overfitting probe.

| step | seed 0 | seed 1 | \|Δ\| | | step | seed 0 | seed 1 | \|Δ\| |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 225 | 0.3033 | 0.2985 | 0.0048 | | 2475 | 0.3499 | 0.3526 | 0.0027 |
| 450 | 0.3233 | 0.3250 | 0.0016 | | 2700 | 0.3544 | 0.3504 | 0.0039 |
| 675 | 0.3353 | 0.3228 | **0.0125** | | 2925 | 0.3555 | 0.3567 | 0.0011 |
| 900 | 0.3395 | 0.3329 | 0.0066 | | 3150 | 0.3562 | 0.3543 | 0.0019 |
| 1125 | 0.3373 | 0.3363 | 0.0010 | | 3375 | 0.3595 | 0.3559 | 0.0036 |
| 1350 | 0.3461 | 0.3406 | 0.0054 | | 3600 | 0.3590 | 0.3560 | 0.0031 |
| 1575 | 0.3372 | 0.3443 | 0.0071 | | 3825 | 0.3595 | 0.3572 | 0.0023 |
| 1800 | 0.3458 | 0.3420 | 0.0038 | | 4050 | 0.3595 | 0.3583 | 0.0012 |
| 2025 | 0.3487 | 0.3429 | 0.0059 | | 4275 | 0.3607 | 0.3598 | 0.0009 |
| 2250 | 0.3504 | 0.3524 | 0.0020 | | **4500** | **0.3605** | **0.3593** | **0.0011** |

**The annealed noise floor is ±0.001.** That is *better* than Fashionpedia's
±0.006 on 1,158 val images, despite 400 val images and four classes under 110
boxes — the run plan's fear that 400 images would push it to ±0.02 is wrong by a
factor of ~20. The plan does **not** need to shrink to a few large effects;
differences of 0.005 are readable here, and the phase 2–5 comparisons are all
worth running as written.

Two caveats on that number. It is a **single paired difference**, not a standard
deviation — a two-sample range is itself a high-variance estimator, so treat
±0.001 as an optimistic bound and anything under ~0.003 as a tie. And it is
specific to the *annealed* eval: mid-run spread averages 0.0036 and peaks at
**0.0125 at step 675**, more than 10x the annealed floor. That reproduces the
Fashionpedia pattern exactly and re-earns the non-negotiable: **only compare
fully-annealed evals.** Comparing at step 675 would let a coin flip out-vote
every real lever in the plan.

Per-metric floors at the annealed eval: `map_50` 0.0021, `map_75` 0.0012,
`map_medium` 0.0012, `map_large` 0.0017, but **`map_small` 0.0110** — ten times
the headline, on the 6.7% of boxes that are small. `map_small` is not a usable
ranking metric on this dataset.

**The regime does not flip to overfitting — the run plan's prediction 1 is not
confirmed at 10 epochs.** The curve rises, flattens around step ~3375 and stays
flat; it never turns over. Both seeds peak at step 4275 and the annealed 4500 is
0.0002–0.0005 below that, which is a quarter of the noise floor.

Consequences: **L1b is unnecessary** — finding 18's trap (argmax ≠ annealed) does
not bite when the tail is flat, so 4,500 steps annealed is a valid report point
and L2a can be run directly. No early-stopping machinery is needed. Budget is
*saturated* rather than a lever: steps 3375 → 4500 bought 0.0010, at the floor.
Whether the curve eventually turns over past 10 epochs is untested and is now a
cheap question (~35 min for 3,000 more steps) rather than an assumed one.

Training is worth a lot here regardless: **0.2348 zero-shot → 0.3605 annealed,
+0.126**, against Fashionpedia's +0.093 for 3x the GPU time.

## Phase 0 verdict

| | measured | run plan expected |
|---|---|---|
| step time, vb2/bs8 | 751 ms compiled | 700–1,000 ms ✓ |
| speedup vs. large | 3.4x | 3–4x ✓ |
| eval cost | ~30 s | ~40 s ✓ |
| `--compile` on base | 1.11x, −14% memory | untested → **keep** |
| batch size | ≤4% from bs8→bs32 | "live question again" → **dead** |
| noise floor (annealed) | **±0.001** | feared ±0.02 → **plan stands** |
| overfitting by 10 epochs | none, curve flattens | "peak and turn over" → **not seen** |
| lateral merge, zero-shot | **+0.060 mAP** | flagged as a risk → **confirmed, large** |

Settled protocol for phases 1–5: the run plan's block, unchanged, with
`--batch-size 8 --compile --no-grad-checkpointing`, `--eval-every-steps 225`,
`--max-steps 4500`, and reporting at the annealed final eval only. Note
`--negative-ratio` stays at its default 5 — the run plan's protocol block omits
it, deliberately dropping Fashionpedia's 20, so any comparison back to those runs
carries that difference.

Ranking from here is unchanged except that L1/L1b are already answered: **L2
per-class AP → L3b (mosaic tail) → L4 (capacity)**. Two runs fit on the two GPUs
concurrently with zero MLflow lock contention and no measurable slowdown, so each
phase is one ~70 min wall-clock block rather than two.
