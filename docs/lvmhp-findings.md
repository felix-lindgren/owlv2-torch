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

## Phase 2 — laterality

**L2a needed no training run.** Its configuration — the protocol block, 18
classes, 4,500 steps — is exactly L0c's, so L0c seeds 0 and 1 *are* L2a, at
0.3605 and 0.3593 annealed. Phase 2 therefore reduces to two ~40 s `--eval-only`
passes over those weights plus one new training arm.

Per-class AP required an implementation change: `class_metrics` was hardcoded
`False` in `coco_eval`, so `val/map_per_class` had been logging a constant −1 for
every run in the project. `coco_eval` now takes `class_names` and enables class
metrics when it is supplied, and `train_text.py` exposes it as `--per-class-ap`
(printed, and logged as `val/ap/<class>` and `val/ar_100/<class>`). It is off by
default because it costs an extra COCOeval accumulate per class on *every* eval
in a run; the intended use is `--eval-only`.

### L2a per-class AP — the six lateral classes (2026-08-08)

`--eval-only --per-class-ap --init-from text_checkpoints/lvmhp_L0c_seed{0,1}/final.pth`.

| class | AP seed 0 | AP seed 1 | | class | AP seed 0 | AP seed 1 |
|---|---:|---:|---|---|---:|---:|
| upper clothes | 0.7339 | 0.7328 | | scarf | 0.2424 | 0.2493 |
| hat | 0.6959 | 0.6859 | | belt | 0.1975 | 0.2066 |
| hair | 0.6826 | 0.6821 | | **right shoe** | **0.1612** | **0.1533** |
| pants | 0.6817 | 0.6801 | | **left shoe** | **0.1326** | **0.1356** |
| face | 0.5622 | 0.5609 | | torso skin | 0.1182 | 0.1276 |
| dress | 0.5181 | 0.5199 | | **left leg** | **0.1220** | **0.1063** |
| sunglasses | 0.4708 | 0.4618 | | **right arm** | **0.1176** | **0.0984** |
| skirt | 0.4226 | 0.4272 | | **left arm** | **0.1137** | **0.1216** |
| bag | 0.4226 | 0.4216 | | **right leg** | **0.0971** | **0.0995** |

**Risk A is confirmed as strongly as the evidence allows.** The six lateral
classes occupy six of the bottom seven places in both seeds, all between 0.097
and 0.161, against 0.42–0.73 for every non-lateral class except `torso skin`,
`belt` and `scarf`. The run plan's prediction — that "left arm" and "right arm"
differ by one token while being visually near-identical, and that the real
distinction is image-relative geometry a text query cannot express — is what the
numbers show.

### L2a re-scored on the 15-class set (2026-08-08)

Same weights, `--merge-class-names left_arm+right_arm=arm
left_shoe+right_shoe=shoe left_leg+right_leg=leg`. This is the *only* legitimate
baseline for any 15-class number.

| | 18 classes | 15 classes, re-scored | Δ |
|---|---:|---:|---:|
| seed 0 | 0.3605 | **0.4683** | +0.1078 |
| seed 1 | 0.3593 | **0.4689** | +0.1096 |

The class-set arithmetic is **+0.108 on trained weights**, nearly twice the
+0.060 it was worth zero-shot at L0b. The seed spread at 15 classes is 0.0006,
consistent with the ±0.001 annealed floor, so the merged metric is no noisier
than the 18-class one.

**The merge is not just re-averaging — the model already knows where the parts
are and is only guessing the side.** Merging each pair produces an AP far above
either member, not the average of the two:

| pair | left | right | merged | AR@100 left / right / merged |
|---|---:|---:|---:|---|
| shoe | 0.1326 | 0.1612 | **0.5226** | 0.352 / 0.376 / 0.653 |
| arm | 0.1137 | 0.1176 | **0.3890** | 0.329 / 0.353 / 0.572 |
| leg | 0.1220 | 0.0971 | **0.3664** | 0.422 / 0.243 / 0.548 |

(seed 0; seed 1 agrees to ±0.003). Shoe AP more than triples. A pure
re-averaging effect would land the merged class near the mean of its members;
landing at 3–4x means most of the loss was side assignment — a correct box scored
as the wrong side is a false positive *and* a false negative, and merging
converts both into one true positive.

### L2b — native 15-class training (2026-08-08)

Seeds 0 and 1, one per GPU, protocol unchanged from L0c except
`--merge-class-names`, so the class set is the single variable. 61 and 57 min.

| | L2a re-scored | L2b native | Δ |
|---|---:|---:|---:|
| seed 0 | 0.4683 | 0.4706 | **+0.0023** |
| seed 1 | 0.4689 | 0.4704 | **+0.0015** |

**Training on the merged labels is worth +0.002 — a tie by the project's own
bar.** Both seeds move the same direction, but the plan's threshold is ~0.003
and this is below it. Set against the +0.108 the merge is worth for free, L2b
adds nothing that would justify its 60 minutes.

The per-class breakdown says the effect is real but narrow, and explains exactly
why the headline barely moves. Only two classes move beyond the noise, and they
are merged ones:

| class | Δ seed 0 | Δ seed 1 |
|---|---:|---:|
| **arm** | **+0.0146** | **+0.0115** |
| **leg** | **+0.0087** | **+0.0125** |
| shoe | +0.0009 | +0.0011 |
| belt | +0.0156 | −0.0046 |
| all 11 others | −0.0039 … +0.0036 | −0.0056 … +0.0029 |

Native training helps `arm` and `leg` by ~0.012 in both seeds and does nothing
for `shoe`, which was already the strongest of the three at 0.52. `belt` swings
+0.016 / −0.005 and is noise, as an 85-box class should be. Two classes gaining
0.012 in a 15-class macro-average is 2 x 0.012 / 15 = **0.0016** — which is the
headline Δ, arithmetic that closes exactly.

The reading: merging removes an unlearnable side distinction, and the model
converts that freed capacity into better `arm` and `leg` detection. But the
metric is a macro-average over 15 classes, so a real gain on two of them is
diluted to the noise floor. **Adopt the 15-class set; do not treat native
training on it as a separate lever.**

The curve reproduces L0c on the new class set: rise, flatten at ~3,375, flat
tail (3375 → 4500 buys 0.0017 and 0.0042), no turnover at 10 epochs. Mid-run
spread peaks at **0.0140 at step 1,125** against **0.0002 annealed** — an
independent replication of L0c's central warning, now at 70x rather than 10x.

### Mosaic was corrupt at fractional `--mosaic-prob` (found 2026-08-08)

The first Phase 3 attempt was killed at ~step 2,100 of 4,500 and discarded.
kornia's `RandomMosaic` builds a *partial* batch's images indexed by rank among
the applied samples, but transforms the boxes by original batch position. Below
`p=1.0` the boxes therefore come back at the wrong offsets. The wrapper's
existing repair only rescued the **labels** (`gpu_augment.py`'s permutation
scatter); the box coordinates were already wrong by then.

Measured with a probe that fills each source image black except for one marked
rectangle — its own box — and then checks that the pixels inside each returned
box still carry that source's colour:

| `mosaic_prob` | misplaced boxes | note |
|---|---|---|
| 0.5 | **706 / 1,932** | 199 of 200 batches partially mosaicked |
| 1.0 | 0 / 2,286 | never takes the partial path |

At `--batch-size 8 --mosaic-prob 0.5`, ~99% of batches are partial, so
essentially every mosaic batch was affected.

**Why the existing tests passed.** They asserted with *full-image* boxes
(`[0.5, 0.5, 1.0, 1.0]`), which cover the whole tile no matter where they are
translated to — a box in the right tile at the wrong offset is indistinguishable
from a correct one. A first probe of mine repeated the same mistake with
constant-colour images and also reported 0 errors. Only sub-image boxes over
spatially varying content expose it.

**Fix:** drive kornia at `p=1.0` so it only ever sees a full batch, and apply the
per-sample probability in the wrapper afterwards, keeping the original image and
targets for unselected samples. The partial-batch path is now unreachable, and a
hard `RuntimeError` fires if kornia ever returns less than a full batch. Selected
fraction still tracks the requested rate (0.240 / 0.484 / 0.742 measured at
0.25 / 0.5 / 0.75).

`tests/test_gpu_augment.py` gains `marked_box_batch` plus a regression test that
fails on the pre-fix code (9 failures) and passes after (15 pass). Test
appliedness is now read from pass-through *identity* rather than by comparing
images: a mosaic whose cells all come from one source is pixel-identical to that
source, which silently misreports it as skipped.

**Scope — this invalidates two Fashionpedia runs.** Only fractional
`--mosaic-prob` with `--gpu-augment` is affected. Checking `mosaic_prob` across
every logged run:

- Every LV-MHP training result above (L0c, L2b) ran at `--mosaic-prob 0`. **No
  number in this document changes.** The `--eval-only` runs log `mosaic_prob 0.5`
  from the default but never train, so they are untouched.
- **`F2_34cls_mosaic` and `F2b_34cls_mosaic` ran at `mosaic_prob=0.5` with
  `gpu_augment=True`** — both trained on corrupted mosaic boxes for the ~50% of
  each batch that was mosaicked. Their paired partners F1/F1b ran at
  `mosaic_prob=0` and are unaffected.

So Fashionpedia's mosaic-vs-no-mosaic comparison was **mosaic-with-broken-boxes
vs no mosaic**, which is not the experiment it was read as. Its verdict that
mosaic loses is confounded: some of that deficit is corrupted supervision rather
than the regularizer. `docs/fashionpedia-findings.md` needs this caveat, and it
raises the prior that Phase 3 here — the first fair test of mosaic in the
project — could come out differently.

## Phase 2 verdict

| question | answer |
|---|---|
| Are the lateral classes weak? | Yes — 6 of the bottom 7, AP 0.097–0.161 vs 0.42–0.73 for most others |
| Is merging worth it? | **+0.108 on trained weights**, free, no retraining |
| Is it just re-averaging? | **No** — merged AP is 3–4x either member, so most of the loss was side assignment |
| Does training on 15 classes help? | **+0.002, a tie.** Real +0.012 on `arm` and `leg`, diluted by the macro-average |
| Did L2a need a run? | No — L0c *is* L2a |

**New reference point for phases 3–5: 0.4706 (L2b seed 0, 15 classes, annealed
4,500).** Phase 3 onward should run on the 15-class set, since it is strictly
better and costs nothing, and every number must be quoted against a 15-class
baseline. Phase 2 cost ~60 min of GPU time instead of the budgeted ~2 h, because
L2a was already in hand and the per-class question needed only a 40 s eval.

## Phase 3 — regularization (2026-08-08)

Both arms on the 15-class set, seed 0, protocol otherwise identical to L2b, on
the **fixed** mosaic. Baseline is L2b seed 0 = 0.4706. 68 and 62 min.

| run | mosaic | annealed 4,500 | vs. L2b | evals behind baseline |
|---|---|---:|---:|---:|
| L2b | none | **0.4706** | — | — |
| **L3a** | `p=0.5` throughout | 0.4640 | **−0.0065** | **20 of 20** |
| **L3b** | `p=0.5`, off for final 1,800 | 0.4675 | **−0.0030** | 16 of 20 |

**Mosaic loses on LV-MHP even when implemented correctly.** L3a is behind at
every one of its 20 evals and −0.0065 annealed, well past the ~0.003 tie bar.
This is the run plan's most promising direction and the one Fashionpedia's budget
starvation was blamed for; at 10 epochs on 3,600 images, with correct boxes, it
still does not pay.

**The no-aug tail is worth +0.0035 and is the one thing here that works.** L3b
beats L3a at the annealed eval, and the curve shows why: mosaic stops at step
2,700 and L3b closes to **+0.0009 / +0.0006 above baseline at 2,925 / 3,150** —
briefly reaching parity — before settling at −0.0030. So the tail recovers about
half the deficit, and implementation item 2 is validated as doing what DEIM's
`no_aug_epoch` claims. It just is not enough to make mosaic worth turning on.
At −0.0030 L3b sits exactly on the tie bar, so the honest reading is that
**L3b ties with no-mosaic** while L3a clearly loses.

Sub-metrics say the damage is concentrated where this mosaic variant is weakest:

| | map | small | medium | large |
|---|---:|---:|---:|---:|
| L2b (none) | 0.4706 | 0.1884 | 0.3854 | 0.4809 |
| L3a | 0.4640 | **0.1515** | 0.3792 | 0.4723 |
| L3b | 0.4675 | 0.1639 | 0.3871 | 0.4774 |

`map_small` drops **−0.037** under mosaic, over 3x its ±0.011 noise floor, while
medium and large lose only −0.006 and −0.009. This is consistent with the variant
in use: kornia's is a **crop-style** mosaic — original-scale images are
concatenated and an input-sized window is cropped — so unlike the conventional
four-image *downscaled* mosaic it never manufactures small objects. It only
removes context and clips objects at tile seams, which costs the smallest boxes
the most. The plan's D3 ("DEIM-style downscaling mosaic") is therefore still
untested and is a different question from the one L3a answers.

### Phase 3 verdict

| question | answer |
|---|---|
| Does mosaic pay at 10 epochs? | **No.** −0.0065, behind at 20 of 20 evals |
| Does the mosaic-off tail help? | **Yes, +0.0035** over L3a, and visibly at the switch |
| Is mosaic+tail worth running? | No — 0.4675 still ties/loses against 0.4706 for free |
| Was Fashionpedia's verdict right after all? | Directionally yes, but it was reached on corrupted arms and could not have known |

**Reference point is unchanged: 0.4706 (L2b seed 0).** Keep `--mosaic-prob 0`.
Both Phase 3 arms are single-seed; L3a's margin is 6x the annealed floor and
20-of-20 consistent, so it needs no confirmation, while a seed-1 L3b would sharpen
a result that currently sits on the tie bar. Neither changes the recommendation.

### L3c / D3 — the DEIM-style downscaling mosaic (2026-08-08)

The run plan's D3 was never implemented; only kornia's crop-style mosaic existed.
`--mosaic-mode {crop,downscale}` now selects between them, defaulting to `crop`
so nothing existing changes. The downscale path resizes a whole source image into
each grid cell instead of concatenating at original scale and cropping, so
nothing is discarded, every box survives, and a 2x2 grid halves every object's
linear size. It is written in plain tensor ops rather than kornia: there is no
partial-batch path to get wrong, and boxes map by a pure affine rescale in
normalised coordinates, so no clipping or visibility filter is possible.

L3c is L3a with only the mosaic type changed.

| run | mosaic | annealed 4,500 | vs. L2b | vs. L3a |
|---|---|---:|---:|---:|
| L2b | none | 0.4706 | — | — |
| L3a | crop, `p=0.5` | 0.4640 | −0.0065 | — |
| **L3c** | **downscale, `p=0.5`** | **0.4711** | **+0.0005** | **+0.0071** |

**The mosaic *variant* was the whole story.** L3c beats L3a at **19 of 20 evals**
and recovers the entire crop deficit, landing in a dead tie with no mosaic
(+0.0005, far inside the ~0.003 bar, and ahead at only 11 of 20 evals). The
sub-metric that motivated the hypothesis confirms it:

| | map | small | medium | large |
|---|---:|---:|---:|---:|
| L2b (none) | 0.4706 | 0.1884 | 0.3854 | 0.4809 |
| L3a (crop) | 0.4640 | **0.1515** | 0.3792 | 0.4723 |
| L3c (downscale) | 0.4711 | **0.1833** | 0.3864 | 0.4795 |

Crop mosaic cost `map_small` −0.037; downscale costs −0.005, inside its ±0.011
floor. The diagnosis in the Phase 3 verdict was right — the damage was context
removal and seam clipping, not mosaicking as such — but fixing it buys parity,
not a win.

**So mosaic is now genuinely settled on this dataset, in a way Fashionpedia's
retracted result never was: the good variant ties, the bad one loses.** Keep
`--mosaic-prob 0` as the default recipe, since a tie is not worth the extra
complexity. The one live thread is that L3c carries ~4x the boxes per mosaicked
sample (79 vs 20), so it trains on a materially different positive/negative
balance and still keeps pace; pairing it with the `--mosaic-no-aug-steps` tail
that was worth +0.0035 on the crop variant is the obvious next probe if mosaic is
revisited.

## Phase 4 — capacity (partial)

| run | vision blocks | annealed 4,500 | vs. L2b |
|---|---|---:|---:|
| L2b | 2 (protocol) | **0.4706** | — |
| **L4a** | **0 (head only)** | **0.4303** | **−0.0403** |

**The run plan's central Phase 4 hypothesis is wrong.** It argued that at ~254
samples per trainable million, "more capacity is a plausible *liability* here"
and that "the interesting hypothesis is that vb0 wins". vb0 does not win; it
loses by **−0.0403**, 40x the annealed noise floor and the largest effect
measured anywhere in this project. Unfreezing two vision blocks is worth more
than the entire lateral-merge training question, the mosaic question and the
class-set question combined.

L4a is also behind at every one of its 20 evals and its curve flattens early
(0.4251 at step 2,700 to 0.4303 at 4,500), so it is capacity-limited rather than
budget-limited. The head alone cannot adapt OWLv2 to this dataset; the vision
tower has to move.

This makes **L4b (`--vision-blocks 6`) the interesting run**, and it now has a
real prior behind it: if 0 → 2 is worth +0.040, the 2 → 6 slope is the question,
against the plan's fear that 42.5M trainable parameters on 3,600 images overfits.

### L4b — vb6, and the depth ladder (2026-08-08)

Vision LR held at 1e-5 across all three, so depth is the only variable — the
discipline the Fashionpedia retraction forced. 89 min.

| vision blocks | trainable | annealed 4,500 | vs. vb2 | step |
|---|---:|---:|---:|---:|
| 0 | 2.8M | 0.4303 | −0.0403 | 585 ms |
| 2 (protocol) | 16.9M | 0.4706 | — | 751 ms |
| **6** | **45.3M** | **0.4865** | **+0.0159** | 1,051 ms |

**Depth is the dominant lever on this dataset, and it has not saturated at vb6.**
The ladder is +0.0403 for 0 → 2 and **+0.0159 for 2 → 6** — diminishing, but the
second step is still 16x the annealed noise floor and larger than every other
effect in phases 2–3 combined. `map_small` moves 0.1436 → 0.1884 → 0.2024 and
`map_75` 0.4525 → 0.4935 → 0.5437, so this is better localisation, not just
better ranking.

**The run plan's overfitting prediction is refuted twice over.** It argued 42.5M
trainable parameters against 3,600 images (~85 samples per trainable million)
made capacity "a plausible liability". Instead vb6 wins, and its curve shows no
turnover: it is still climbing into the anneal (0.4780 at 2,925 → 0.4865 at
4,500) and its best (0.4868 at step 4,050) is 0.0003 off the annealed value.

One caveat worth recording. vb6 is **behind vb2 for the first ~2,400 steps** and
only pulls ahead after step 2,475, ending ahead at just 13 of 20 evals. A deeper
tower is slower to converge, so any budget-truncated comparison would have
reported the opposite result — a direct instance of the "only compare annealed
evals" rule, and a reason the Fashionpedia depth work was so hard to read.

### L3c seed 1 — confirmation (2026-08-08)

| | seed 0 | seed 1 | \|Δ\| |
|---|---:|---:|---:|
| L3c annealed | 0.4711 | 0.4711 | **0.0000** |

An exact tie between seeds, against a no-mosaic baseline of 0.4706. Mid-run
spread averages 0.0030 and peaks at 0.0180 — again over 5x the annealed floor,
reproducing the pattern from L0c and L2b for a third time.

**The DEIM mosaic verdict is confirmed: it ties.** Two seeds landing on the same
four decimal places, 0.0005 above baseline, is as clean a null as this setup can
produce. The new `--mosaic-mode downscale` implementation is validated
end-to-end, and the mosaic question is closed: the crop variant loses, the
downscale variant ties, neither is worth turning on.

### L4c — vb12, the full tower (2026-08-08)

Base has 12 vision blocks, so this unfreezes all of them. Needs
`--grad-checkpointing`: vb12/bs8 OOMs at 11.6 GiB otherwise (probed before
launch, 1,737 ms/step and 4.23 GiB with it).

| vision blocks | trainable | annealed 4,500 | vs. vb6 |
|---|---:|---:|---:|
| 0 | 2.8M | 0.4303 | −0.0562 |
| 2 | 16.9M | 0.4706 | −0.0159 |
| **6** | **45.3M** | **0.4865** | — |
| 12 (all) | 87.8M | **0.4431** | **−0.0434** |

**Depth peaks at vb6 and reverses hard.** vb12 gives back almost exactly the gain
that vb2 → vb6 bought and lands nearer vb0 than vb6. The ladder is an inverted U
with the optimum at half the tower.

**But this is not the overfitting the run plan predicted, and the distinction
matters.** Overfitting looks like a rise, a peak, then a monotone decline. vb12
does the opposite:

| step | 225 | 900 | 1575 | 2250 | 3150 | 4050 | 4500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vb12 | 0.3635 | 0.4168 | **0.3591** | 0.4289 | 0.4418 | 0.4425 | **0.4431** |

It climbs to 0.4168 by step 900, **collapses to 0.3591 at 1,575** — below its own
step-225 value — then recovers erratically and is **still improving at the final
step**, where its peak sits. A model that is still climbing into the anneal is not
overfitting; it is a model that was destabilised and spent the run recovering.

The likely mechanism is that the earliest blocks hold generic low-level features,
and updating them at the same 1e-5 that suits the last six damages the pretrained
representation. That makes this an **optimisation failure, not a capacity
failure**, and it leaves the plan's overfitting prediction still unconfirmed at
every depth tested. The open question is whether vb12 with a lower vision LR (or
a per-block decay) recovers vb6's number — worth one run, and the first place a
vision-LR change is justified rather than confounding.

**Confound to record:** vb12 ran with `--grad-checkpointing` and vb0/vb2/vb6 did
not, because vb12 does not otherwise fit. Checkpointing recomputes activations
and is mathematically neutral — identical gradients, and `preserve_rng_state` is
on by default — so the comparison should hold, but it is a second variable in a
one-variable comparison. A vb6 + checkpointing run (~1.4 h) would close it.

## Phase 4 verdict

| question | answer |
|---|---|
| Does head-only (vb0) win on 3,600 images? | **No — −0.0403**, the plan's prediction inverted |
| Does vb6 overfit? | **No.** +0.0159 over vb2, no turnover, still climbing at 4,500 |
| Is depth saturated? | **Yes, at vb6.** vb12 is −0.0434 |
| Is vb12's failure overfitting? | **No** — it destabilises at step ~1,575 and is still climbing at 4,500 |
| Does depth need annealing to read? | **Yes** — vb6 trails vb2 until step ~2,475 |

**Best on LV-MHP: 0.4865 (L4b, vb6, 15 classes, annealed 4,500).** The chain from
zero-shot is 0.2348 → 0.2950 (merge) → 0.4706 (vb2) → 0.4865 (vb6), and
**`--vision-blocks 6` is the single most valuable setting found in this project.**

## Phase 5 — transfer to large (2026-08-09)

The winning recipe at `--model-type large`: 15 classes, vb6, vision LR 1e-5,
4,500 steps, no mosaic. Needs `--grad-checkpointing` (large/vb6/bs8 OOMs at
11.6 GiB; 3,819 ms/step and 7.09 GiB with it). **5.49 h**, against base vb6's
1.5 h.

| | base vb6 | large vb6 | Δ |
|---|---:|---:|---:|
| **mAP** | **0.4865** | **0.4903** | **+0.0038** |
| map_50 | 0.7215 | 0.7498 | +0.0283 |
| map_75 | 0.5437 | 0.5370 | −0.0067 |
| map_small | 0.2024 | 0.1875 | −0.0149 |
| map_medium | 0.4061 | 0.3966 | −0.0095 |
| map_large | 0.5001 | 0.5202 | +0.0201 |

**The conclusions transfer, and that is almost all large buys.** +0.0038 is
barely past the ~0.003 tie bar, for **3.6x the step time and 3.7x the wall
clock**. Large leads at 18 of 20 evals, so the direction is consistent rather
than a coin flip, but the magnitude is a rounding error next to `--vision-blocks
6`'s +0.0159 on base — which cost nothing.

The sub-metrics say the two models are good at different things rather than one
dominating: large is **+0.028 at map_50 and +0.020 on large boxes** but **−0.015
on small and −0.007 at map_75**. It finds more objects at loose IoU and localises
worse at tight IoU. Net, those nearly cancel.

**Two caveats, both pushing the same way.** vb6 on large is 6 of **24** blocks —
a quarter of the tower, where vb6 on base was half. Since depth was the dominant
lever on base, large is plausibly under-unfrozen at the setting it was handed,
and a large vb12 (matching the 50% ratio) is the honest version of this test. It
was not run: it OOMs well past checkpointed vb6's 7.09 GiB and would take ~8 h.
Second, large's curve is **flat-to-declining in the tail** (peak 0.4906 at step
3,375, then −0.0003 to the anneal) while base is still climbing (+0.0028 over the
same span), so large is closer to saturated at this budget and extra steps would
not obviously help it.

### Phase 5 verdict

| question | answer |
|---|---|
| Do base conclusions hold on large? | **Yes** — same direction, large ahead at 18 of 20 evals |
| Is base "an iteration tool only"? | **No.** It is within 0.004 of large at 27% of the cost |
| Is large worth it? | **Not on this dataset** — +0.0038 for 3.7x wall clock |
| Was the recipe fair to large? | **Not entirely** — vb6 is 25% of large's tower vs 50% of base's |

## Where the project stands

Best on LV-MHP: **0.4903 (large vb6)**, and **0.4865 (base vb6)** at a quarter of
the cost, which is the number to build on.

The full chain on base: **0.2348** zero-shot (18 cls) → **0.2950** lateral merge
→ **0.4706** trained at vb2 → **0.4865** at vb6. Training is worth +0.19 over
zero-shot on the merged class set, and of the levers actually tested, exactly two
mattered: the **class-set merge** (+0.060 free, and it is a metric change, not
learning) and **vision depth** (+0.056 from vb0 to vb6). Mosaic, native merged
training, batch size and model scale were each worth ≤0.004.

Open, in rough order of expected value:

1. **vb12 at a reduced vision LR**, or per-block LR decay — L4c's collapse is
   instability, not capacity, so the depth optimum may not really be vb6.
2. **large at vb12** — the fair version of phase 5, ~8 h and needs a memory plan.
3. **vb6 + `--grad-checkpointing`** (~1.4 h) to close the L4c confound.
4. The downscale-mosaic + no-aug-tail cell, to complete the 2x2.
5. A final confirmation on the **official test list** (`--val-source test`), which
   the run plan reserved and which nothing has been tuned on. Everything above is
   measured on the seeded 400-image val slice.
