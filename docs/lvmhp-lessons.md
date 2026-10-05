# What LV-MHP taught — the short version

Full detail in `docs/lvmhp-findings.md`. This is the transferable part, written
for the next dataset. Seven phases, ~27 GPU-hours, OWLv2-base on 3,600 images.

## The result

| stage | mAP |
|---|---:|
| zero-shot, 18 classes | 0.2348 |
| merge 3 lateral class pairs (no training) | 0.2950 |
| train, `--vision-blocks 2` | 0.4706 |
| train, `--vision-blocks 6` | 0.4865 |
| same recipe on `--model-type large` | 0.4903 |
| vb6 + Q/V rank-8 LoRA on the first six blocks | **0.4962** (2 seeds) |

## Of everything tested, two things mattered

**1. Vision depth — an inverted U that turned out to be the optimiser's fault.**
vb0 0.4303 / vb2 0.4706 / **vb6 0.4865** / vb12 0.4431, with vision LR held at
1e-5 throughout. Unfreezing six blocks was the single most valuable setting found
in the project. Both of the run plan's predictions were wrong: head-only did not
win, and vb6 did not overfit.

vb12's collapse is **instability, not overfitting** — it climbs to 0.4168 by step
900, crashes to 0.3591 at 1,575, then recovers and is still improving at the
final step. Updating the earliest blocks at an LR suited to the last six appears
to damage the pretrained representation.

**This was later confirmed and then exploited.** Rank-8 Q/V LoRA at the same
depth turns vb12's 0.4431 into 0.4827 — no collapse — and adding that constrained
update to vb6's first six blocks gives **+0.0099 over vb6 on two seeds**. So the
inverted U was an optimiser artefact, not a property of the representation: the
early blocks do hold usable signal, and the only thing wrong with vb12 was the
size of the step. When a depth ladder turns over, **suspect the update before you
conclude the layers are useless**.

**2. The class set, which is arithmetic and not learning.** Merging three
left/right pairs was **+0.060 zero-shot and +0.108 on trained weights**, with no
gradient step. Training natively on the merged set added +0.002 — a tie.

Everything else was ≤0.004: native merged training, crop mosaic (−0.0065),
DEIM downscale mosaic (+0.0005, two seeds landing on the same four decimals),
batch size (≤4% throughput from bs8→bs32), and model scale (+0.0038 for 3.7x the
wall clock).

## The methodological rules that earned their keep

- **Only compare fully-annealed evals.** The annealed noise floor was ±0.001;
  mid-run spread peaked at 0.0125–0.0180, over 10x wider, in four separate runs.
  vb6 *trails* vb2 until step ~2,475 and ends ahead — a truncated comparison
  reports the opposite conclusion.
- **Re-score old weights on any new class set before claiming a win.** Twice now
  a class-set change produced a bigger number than every real lever combined.
- **One variable per run**, and hold LR fixed when varying depth. This is what
  forced the Fashionpedia retraction.
- **Treat anything under ~0.003 as a tie**, and remember ±0.001 came from a
  single paired difference, so it is an optimistic bound.
- **Phase 0 is not skippable.** Calibrating step time, memory, the noise floor
  and the zero-shot baseline cost ~2 h and made every later number readable.
  Probing memory before launch caught two OOMs that would have killed a 5-hour
  run part-way through.

## Two bugs that invalidated real results

- **`class_metrics=False` was hardcoded**, so `val/map_per_class` logged a
  constant −1 for every run in the project's history. Per-class AP is what
  answers "is this class unlearnable or merely rare". Now `--per-class-ap`.
- **kornia's `RandomMosaic` mistranslates boxes on a partially mosaicked batch**
  (706 of 1,932 boxes wrong at `p=0.5`, 0 at `p=1.0`). This retroactively
  invalidated Fashionpedia's "mosaic loses 19 of 19" verdict — those arms trained
  on corrupted boxes. Fixed by driving kornia at `p=1` and masking outside it.

The shared lesson: **both bugs passed their tests.** The mosaic tests asserted
with full-image boxes, which cover the whole tile wherever they land, so a box in
the right tile at the wrong offset is invisible. Test augmentation with
sub-image boxes over spatially varying content, and *check a new metric is
actually populated* before trusting it.

## LoRA is not the cheap option people assume

Worth knowing before planning the next adaptation budget. Pure Q/V rank-8 LoRA on
all 12 blocks has **15x fewer trainable parameters than vb6 and is still worse on
every axis except disk**: −0.0038 mAP, 1,294 vs 1,051 ms/step, 9.21 vs 6.91 GiB
peak. An adapter on block 0 forces backward through the entire tower, while vb6
stops halfway — so parameter count is simply the wrong proxy for training cost.
What LoRA does buy is checkpoints: 12 MB vs 173 MB, and 36 MB vs 521 MB with
optimiser state.

The corollary is that LoRA earned its place here as an **accuracy** tool, not an
efficiency one: the value was in constraining *where* full-rank updates were
unsafe, not in training fewer parameters.

## What is genuinely untested

Per-block LR decay, LoRA rank and target expansion on the winning hybrid, whether
the hybrid's gain survives at a different LoRA LR (only 1e-4 was run), large at a
matched depth ratio (vb6 is half of base's tower but a quarter of large's), and
the official test list — every number above is on a seeded 400-image val slice.
