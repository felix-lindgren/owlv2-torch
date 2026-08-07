# Run plan — OWLv2-base on LV-MHP-v1

Next dataset after Fashionpedia. Read `docs/fashionpedia-findings.md` first — this
plan is written against those lessons, and the main risk is inheriting its
*conclusions* instead of its *methods*. Several Fashionpedia results do not
transfer, and the reasons are structural.

Switching to `google/owlv2-base-patch16-ensemble` to buy iteration speed. The
Fashionpedia project spent ~42 GPU-hours to resolve about three questions,
largely because every arm cost 2–6 hours. On base + LV-MHP a full run should be
~1–2 hours, which changes what is worth running.

## Dataset profile

Converted with `tools/convert_lv_mhp_coco.py`:

| split | images | boxes | boxes/image |
|---|---:|---:|---:|
| train | 3,600 | 90,057 | 25.0 |
| val | 400 | 9,504 | 23.8 |

18 categories, median image 634x627, median box 65x87 px. Box scale at the base
model's 960px input: **6.7% small / 47.8% medium / 45.2% large** (COCO
thresholds), and the median box is 1.19% of image area. This is a
medium-and-large-object dataset — Fashionpedia's small-object pain (`map_small`
~0.17 at best) should be much less of a factor.

Per-class train / val box counts:

| class | train | val | | class | train | val |
|---|---:|---:|---|---|---:|---:|
| face | 10,821 | 1,154 | | torso skin | 3,130 | 292 |
| hair | 10,398 | 1,114 | | dress | 2,205 | 214 |
| upper clothes | 8,943 | 976 | | hat | 1,226 | 151 |
| left arm | 8,581 | 939 | | skirt | 1,065 | 99 |
| right arm | 8,579 | 907 | | bag | 975 | 110 |
| pants | 7,303 | 794 | | belt | 973 | 85 |
| left shoe | 7,187 | 739 | | sunglasses | 930 | 121 |
| right shoe | 7,184 | 721 | | scarf | 242 | 29 |
| right leg | 5,168 | 532 | | | | |
| left leg | 5,147 | 527 | | | | |

## Three things that invert the Fashionpedia conclusions

**1. The regime flips from underfit to overfit.** Fashionpedia's 6,600 steps at
bs8 was **1.16 epochs** of 45,623 images; budget was the top lever for three days
and the curve was still climbing at 68%. LV-MHP has 3,600 images — **450 steps per
epoch at bs8** — so the same 6,600 steps is **14.7 epochs**. Expect val mAP to
*peak and turn over* rather than plateau. Everything the run plan says about
budget being the dominant lever is now inapplicable, and early-stopping behaviour
matters for the first time.

**2. Mosaic and augmentation are re-opened, not settled.** Mosaic lost on
Fashionpedia at 19 of 19 paired evals — but `run-plan.md` states outright that
0.39–1.16 epochs is far too short for a regularizer to pay. At ~10–15 epochs on
3,600 images that reasoning no longer holds. **Do not inherit the mosaic verdict.**
Block D effectively restarts here, and this is now one of the more promising
directions rather than a closed question.

**3. Depth may flip sign.** vb6 was the best-evidenced positive on Fashionpedia
(+0.008 / +0.009 on two metrics). But the data-to-capacity ratio collapses here.
A base encoder block is 7.08M parameters (4d² attention + 2·d·mlp at d=768,
mlp=3072), so vb2 is 14.2M and vb6 is 42.5M of trainable vision weights, against
**3,600 images**. That is ~254 and ~85 samples per trainable million,
versus Fashionpedia's ~1,800 at vb2 on large (45,623 images / 25.2M). More
capacity is a plausible *liability* here. Test it; do not assume it.

## Two dataset-specific risks, both worth handling up front

**A. Laterality is a text-conditioning problem, and it is 46% of the supervision.**
Six of the 18 classes are lateral pairs — left/right arm (8,581/8,579), left/right
shoe (7,187/7,184), left/right leg (5,168/5,147) — totalling **41,846 of 90,057
boxes (46.5%)**. OWLv2 conditions detection on a *text* embedding, and "left arm"
vs "right arm" differ by one token while being visually near-identical; the real
distinction is image-relative geometry the text query cannot express. Expect these
six classes to have poor AP and to cap headline mAP.

The obvious response is to merge each pair (→ `arm`, `shoe`, `leg`, giving 15
classes). Two warnings:

- **This is a metric change and will raise mAP mechanically**, exactly like the
  Fashionpedia class cleanup that turned out to be 85% re-averaging. Re-score the
  18-class weights on the 15-class set before claiming any of it as learning.
- **Do not implement it by dropping one side.** `--exclude-class-names` can only
  drop, and dropping `left arm` leaves every left arm unlabelled while a visually
  identical right arm stays positive — that manufactures false negatives. Merging
  needs real support (see implementation items).

**B. The val split is small and several classes are near-unmeasurable.** 400
images, and `scarf` has **29 val boxes across 23 images**; `belt` 85, `skirt` 99,
`bag` 110. Per-class AP for those is noise, and they are 4 of 18 classes in a
macro-average.

Fashionpedia measured how bad this gets: in the final model `cape` scored AP
**0.000** and `umbrella` scored **0.877**, and *both classes have exactly 5
validation boxes*. Identical sample size, opposite extremes. LV-MHP has no class
that thin, but `scarf` at 29 boxes is in the same territory, and macro-averaged
mAP inherits the variance of every rare class in it.

Fashionpedia's noise floor was ±0.006 annealed on 1,158 val images — **assume
LV-MHP's is worse until measured** (L0c), and measure it before ranking anything.
This is the single most expensive lesson from the last project.

Also note the 400-image val is a **seeded 90/10 slice of `train_list.txt`**, not
the official test list. Use it for development; keep the official test list
(`--include-test` / `--val-source test`) for one final confirmation run, so the
reported number is not tuned on.

## Expected cost — estimates, to be measured

Base is **5.3x fewer vision FLOPs per sample** than large, computed the same way
as `docs/optimization-notes.md`: base is 3,601 tokens x d768 x mlp3072 x 12 layers
= 1.09 TF/sample, large is 5,185 x d1024 x mlp4096 x 24 = 5.78 TF/sample (which
reproduces the measured 46.2 TFLOP at bs8).

That does **not** mean 5.3x faster end to end. Scaling large's measured 2,855 ms
step by component: vision forward 2,227 → ~420 ms, backward/optimizer 528 → ~200 ms,
heads 53 → ~25 ms, and **the loss term should rise, not fall** — Hungarian matching
was 34 ms at 6.1 boxes/image and LV-MHP has 25. Rough estimate **~700–1,000 ms/step
at bs8, i.e. 3–4x faster**, giving ~5–8 min/epoch and ~1–2 h for a 10–20 epoch run.
Evals should be ~40 s (400 images vs 1,158, on a faster model), so **dense evals
are essentially free** — use `--eval-every-steps 225` (half-epoch) rather than
Fashionpedia's 550.

Memory is unmeasured for base. It should be far below the 3060's 11.6 GiB at bs8,
and bs16/32 likely fits without `--grad-checkpointing`, which would make batch size
a live question again. Note `tools/probe_grad_ckpt_memory.py`, referenced by
`run-plan.md`, **is not in the tree** — it needs recreating or replacing with a
short timing/memory script.

## Protocol

Start from the Fashionpedia final recipe, adjusted for base and for the denser
supervision:

```
--model-type base --class-loss mal --class-loss-gamma 1.5 \
--vision-blocks 2 --vision-learning-rate 1e-5 --head-learning-rate 5e-5 \
--text-blocks 0 --text-learning-rate 1e-7 --batch-size 8 --val-batch-size 16 \
--num-workers 6 --warmup-steps 100 --weight-decay 1e-4 \
--lambda-class 1.0 --lambda-l1 5.0 --lambda-giou 2.0 --lambda-objectness 0.5 \
--eval-top-k 100 --eval-every-steps 225 --seed 0 \
--mosaic-prob 0 --amp --augment --compile --mlflow
```

Carried over from Fashionpedia because they were cheap or settled: MAL (adopted
for box quality), `--lambda-objectness 0.5` (the head is redundant under MAL but
free, and sweeping it was shown to be a waste), `--compile`, `python -u`,
`--eval-top-k 100` (comfortable against 23.8 boxes/image).

**Reconsider `--negative-ratio 20`.** At 25 positives/image it implies 500 mined
negatives, brushing `--max-negatives-per-image 512` — the negative budget
saturates, which it never did at 6.1 boxes/image. MAL ignores negative mining, so
this only reaches the objectness term, but the interaction is untested at this
density.

Non-negotiables, all inherited: report at a fixed budget not the argmax; only
compare fully-annealed evals; one variable per run; `mlflow.db` is authoritative.

## Suggested runs

### Phase 0 — calibration (~2 h, do not skip)

This is the phase Fashionpedia skipped, and skipping it is why block A produced
eight GPU-hours of unrankable results.

| run | what | cost |
|---|---|---|
| **L0a** | Time and memory-probe base at vb0/vb2/vb6 x bs8/16/32. Replaces the missing probe script. | ~20 min |
| **L0b** | **Zero-shot baseline**, 18 classes and 15-class merged, `--eval-only`. Establishes the arithmetic baseline for any class-set change *before* it can confound a run. | ~5 min |
| **L0c** | **Noise floor**: one config, two seeds, at the phase-1 budget, compared only at the annealed eval. | 2 runs |

L0c decides how to read everything below. If the floor comes back near ±0.02 —
plausible on 400 val images — then most single-run comparisons in this plan are
not worth making and the plan should shrink to the few large effects.

### Phase 1 — find the overfitting point (~2 h)

| run | flags |
|---|---|
| **L1** | Protocol above, `--max-steps 4500` (10 epochs), `--eval-every-steps 225` (20 evals) |

The goal is the *shape*: where val mAP peaks and whether it declines. Because
`--max-steps` sets the cosine `T_max`, a peak found mid-run is not the same as a
run annealed to that budget — so follow with **L1b**, a confirmation run whose
`--max-steps` equals the peak step. This is finding 18's trap applied forward
rather than discovered afterwards.

Watch train-vs-val divergence, which Fashionpedia never had to care about.

### Phase 2 — the laterality question (~2 h + eval)

| run | flags |
|---|---|
| **L2a** | 18 classes as-is, at phase 1's budget (this is L1b if the class set is unchanged) |
| **L2b** | 15 classes, lateral pairs merged |

Score L2b against **L2a's weights re-scored on the 15-class set**, never against
L2a's 18-class number. Also read per-class AP for the six lateral classes in L2a
directly — if they are near zero, that is the cleanest evidence for merging, and
it is available from L2a alone without needing L2b at all.

### Phase 3 — regularization, genuinely re-opened (~4 h)

| run | flags |
|---|---|
| **L3a** | `--mosaic-prob 0.5 --mosaic-grid 2 2` at the winning class set and budget |
| **L3b** | L3a + mosaic disabled for the final ~40% of steps (DEIM's `no_aug_epoch`) |

L3b is the D2 that Fashionpedia never ran, and it needs the step-based schedule
flag that has been the outstanding implementation item since day one. At 10+
epochs this is the most likely place to find a real gain.

### Phase 4 — capacity (~3 h)

| run | flags |
|---|---|
| **L4a** | `--vision-blocks 0` (head only) |
| **L4b** | `--vision-blocks 6 --vision-learning-rate 1e-5` |

vb2 is the protocol default and already run. Hold the vision LR **fixed** across
these — varying depth and LR together is precisely what forced the Fashionpedia
retraction. On 3,600 images the interesting hypothesis is that vb0 wins.

### Phase 5 — transfer check (~4 h, once)

| run | flags |
|---|---|
| **L5** | The winning recipe at `--model-type large` |

Confirms that conclusions drawn on base hold on large before any of this is
treated as a recipe. Worth exactly one run; if it disagrees, base is an iteration
tool only, which is itself worth knowing.

## Implementation items

1. **Class merging** — `--exclude-class-names` cannot express it and dropping one
   side is harmful. Either add `--merge-class-names left_arm,right_arm=arm` to
   `train_text.py` or a converter flag. Needed by phase 2.
2. **Step-based augmentation-off tail** — `mosaic_prob` is constant for the whole
   run. Needed by phase 3 (L3b). Outstanding since the Fashionpedia plan's phase 0.
3. **Base-model memory/step-time probe** — `tools/probe_grad_ckpt_memory.py` is
   referenced by `run-plan.md` but absent from the tree. Needed by L0a.
4. **Confirm `--compile` helps on base.** The 1.10x was measured on large, where
   QuickGELU's memory traffic over `[8, 5185, 4096]` dominated. Base's tensors are
   smaller and the win may differ; it is one A/B in L0a.

## Ranking, if time is short

**L0c (noise floor) → L1 (overfitting point) → L2 per-class AP → L3b (mosaic
tail).** The first two are prerequisites for reading anything else. L2's core
question is answerable from per-class AP without a second training run. L3 is the
highest-upside genuinely-open question, because it is the one Fashionpedia's
budget starvation prevented from ever being tested fairly.
