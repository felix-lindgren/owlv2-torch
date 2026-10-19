# Run plan — DOTA remote-sensing adapter, held-out on DIOR

Status (2026-10-18): implemented and launched (R1 on the 5070 Ti, R2 on the
3060) via `tools/run_rs_adapter.sh`. Results go in `logs/rs_adapter/*.json`.

## Question

Does adapting OWLv2-base to overhead imagery make few-shot "pointing" (FLAME)
better on *new* remote-sensing classes, or only on classes it was trained on,
and what does it cost on general-domain detection?

Two facts shape the design:

- Forgetting is fast. A 40-step vb6 LV-MHP smoke cost −0.022 on COCO mini500;
  the 40-step DOTA hybrid smoke cost **−0.075** (0.4949 → 0.4201), and the
  full R1 run was at **0.2842 (−0.211) by step 500**. So the adapter
  is treated as swappable (base OWL for general use, RS adapter for overhead
  imagery, FLAME on top of either), not as one set of weights that is good
  everywhere.
- DIOR and DOTA share classes. A DOTA-tuned model looks good on DIOR partly
  because it has seen planes, ships and storage tanks. The test therefore holds
  out **classes**, not only images.

## Data

`HichTala/dota`, exported with
`tools/convert_dior_to_coco.py --dataset HichTala/dota --output-dir /mnt/datasets/sat/dota_coco`.
It is **DOTA v1.5**, not v1.0: the v1.0 15 classes plus `container-crane`, cut by
the uploader into 512×512 patches (HBB, xywh). 54,087 train / 2,000 validation /
6,854 test patches, 759k train boxes, median box side 26 px. `small-vehicle` is
58% of the boxes. v1.5 does not add `airport` (v2.0 does), so the DIOR-only set
below is unchanged. DIOR is never used for training or checkpoint selection;
`best.pth` is chosen on DOTA validation.

The HF download needs ~6 GB of cache and `/` has ~7 GB free, so run the export
with `HF_HUB_CACHE=/mnt/datasets/hf_cache/hub HF_DATASETS_CACHE=/mnt/datasets/hf_cache/datasets`.

## DIOR class split (fixed before any result)

| group | classes |
|---|---|
| shared with DOTA (10) | Airplane, Ship, Storage tank, Vehicle (DOTA small/large vehicle), Harbor, Bridge, Baseball field, Basketball court, Tennis court, Ground track field |
| DIOR-only (10) | Airport, Chimney, Dam, Expressway service area, Expressway toll station, Golf course, Overpass, Stadium, Train station, Wind mill |

Defined in `CLASS_GROUPS` in `tools/test_dior_flame.py`, which prints the
per-class AP table and both group means for zero-shot and FLAME.

## Arms

Both start from pretrained OWLv2-base, text tower frozen (`--text-blocks 0`), and
the LV-MHP LoRA recipe (Q/V, r8, alpha 8, LoRA LR 1e-4, full blocks 1e-5, heads
5e-5, MAL γ 1.5), plus the DIOR teacher's RS augmentation (aerial prompts,
right-angle rotations, scale augment, no mosaic). 4,500 steps × batch 8 ≈ 0.67
DOTA epochs. Seed 0.

| run | topology | why |
|---|---|---|
| **R1** `R1_dota_hybrid_vb6_lora_first6_seed0` | full last 6 + LoRA first 6 | best LV-MHP config (+0.0099 over vb6) |
| **R2** `R2_dota_lora_all12_seed0` | LoRA all 12, no full blocks | the most constrained update; the bet is less forgetting |

Both arms still train the class, box and objectness heads, so neither is
forgetting-free; the adapter checkpoint carries the head deltas, and "base OWL"
means not loading it.

## Evaluation (per arm, chained in the launcher)

1. **FLAME 5-shot on DIOR test** (all 3,463 images), support seeds 0/1/2:
   `tools/test_dior_flame.py --support-images 5 --checkpoint <best.pth>`. The
   pretrained-base baseline is run in the same protocol (the 0.303 reference
   was `--limit 2000`, so it is not the comparison point).
2. **COCO mini500** with `tools/ovd_eval.py --dataset coco --checkpoint`, plus
   the in-training `general/map_delta` trajectory in MLflow (`RS-Adapter`).
3. **WiSE-FT α = 0.5** (`--wise-alpha 0.5`; LoRA is merged first, then every
   weight is interpolated toward pretrained) through both 1 and 2. This is the
   cheap knob for trading domain gain against general ability.

## Decision rules

- The question is settled by **base FLAME vs adapter + FLAME on the DIOR-only
  group mean**. A gain there that holds on all three support seeds is the signal
  that pointing generalizes. A gain only on the shared group is class overlap.
- The FLAME seed spread on the base run is the noise floor; a group-mean
  difference inside it is a tie.
- Expect small absolute numbers on DIOR-only: those are the classes the
  zero-shot model is already weak on.
- Forgetting is read from the final COCO mini500 `map` vs pretrained (0.4962 via
  `ovd_eval.py`). Its delta noise floor is not yet measured.

## UAV leg (openvocab-evals harness)

`tools/run_rs_adapter_uav.sh` runs zero-shot and FLAME 5-shot (support seeds
0/1/2) on Songdo test (tile 1280, support from `songdo:train`) and MAVREC
aerial-val (tile 1024, support from `aerial-train-subset`) for base, R1 and R2,
through `uav-eval` in `../evals`. The harness's OWLv2 detector gained
`flame_support=<dataset:split>`, `flame_tile_size`, `flame_seed`, `flame_shots`,
`checkpoint=` for LoRA checkpoints (merged without PEFT, bit-identical to the
PEFT merge) and `wise_alpha`. Support crops are cut on the eval tile grid so
support objects are at the eval scale. The harness lock now pins OWLv2torch
`e08a6e5`, GDINOtorch `4e83077` and TensorRT 11 (2026-10-18), so engines built
under TensorRT 10 must be rebuilt.

Songdo and MAVREC vehicles overlap DOTA's vehicle classes, so this is a
"somewhat broader" check of adapter + FLAME, not a held-out-class test.

## Not covered yet

- Multiple RS training sources. xView (`/mnt/datasets/sat/xview_coco`) is on disk
  and has no DIOR-only classes either, but the trainer takes one annotation file,
  so mixing needs a merged COCO export with a unified vocabulary.
