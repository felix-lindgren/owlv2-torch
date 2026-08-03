# OWLv2 training performance notes

Measured 2026-08-02 on an **RTX 3060 (12 GB, sm_86)**, torch 2.11.0+cu130, with
`train_text.py --model-type large --batch-size 8 --vision-blocks 2` on Fashionpedia
(46 classes). Benchmarks: `tools/bench_owlv2.py`, `tools/check_compile_numerics.py`.

## Where the time goes

| Stage | Time | Share |
|---|---:|---:|
| Vision tower forward | 2227 ms | 78% |
| Backward + optimizer | 528 ms | 19% |
| Class/box/objectness heads | 53 ms | 2% |
| Text tower | 11 ms | <1% |
| Loss (incl. Hungarian matching) | 34 ms | 1% |
| **Full train step** | **2855 ms** | |

The vision tower does **46.2 TFLOP per forward at 20.7 TFLOPS**. A pure 8192³ fp16
GEMM on this card reaches **27.1 TFLOPS**, so the tower runs at 76% of the practical
roofline. Per encoder layer (d=1024, mlp=4096, 5185 tokens, B=8):
attention 881 GF (46%), MLP 696 GF (36%), qkv+out projections 348 GF (18%).

**Conclusion: this workload is GEMM-bound, not latency- or data-bound.** Wall-clock
gains beyond ~15% require fewer FLOPs (smaller model / lower resolution), not better
kernels.

## Optimizations, measured

Vision-tower forward only, B=8, large:

| Variant | Time | Speedup |
|---|---:|---:|
| eager (baseline) | 2238 ms | 1.00x |
| `encoder.compile()` | 2032 ms | **1.10x** |
| compile + fused QKV | 1985 ms | 1.13x |
| compile + fp16 frozen weights | 1992 ms | 1.12x |

End-to-end train step with `compile()`: **2835 → 2573 ms (1.10x)**.

### Adopt: `model.vision_model.encoder.compile()`

- One line, ~1.10x end to end, ~16 s one-time compile cost.
- Use **in-place** `Module.compile()`, not `encoder = torch.compile(encoder)` —
  reassignment prefixes state_dict keys with `_orig_mod.` and breaks
  `_load_model(strict=True)` plus the delta checkpoint format. In-place compile
  leaves state_dict keys untouched (verified).
- `max-autotune` is useless here: inductor logs
  *"Not enough SMs to use max_autotune_gemm mode"* on a 3060.
- Numerics are safe: compiled-vs-eager-fp16 max logit diff (2.30) is the same
  magnitude as the pre-existing fp16-vs-fp32 diff (2.37), and top-10 detections on
  a real image match fp32 exactly.
- Add `drop_last=True` to the train loader to avoid a recompile on the final
  partial batch.

### Main thing compile buys: QuickGELU

`QuickGELUActivation` (`owlv2.py`) is `input * torch.sigmoid(1.702 * input)` — three
separate elementwise kernels over `[8, 5185, 4096]` fp16, **7.12 ms/layer** versus
**2.04 ms** for a single fused kernel. That is ~120 ms/step (4%) of pure memory
traffic, which inductor fuses away.

Folding 1.702 into the fc1/fc2 weights and using `nn.SiLU` is mathematically exact
(`x·σ(1.702x) == silu(1.702x)/1.702`) and gives 1.05x without compile, but it
rewrites the checkpoint weights — not recommended.

### Not worth it

- **Fused QKV** (+2%): renames parameters, breaking checkpoint compatibility
  without load/save hooks.
- **fp16 frozen weights** (+2%, −0.5 GiB): only valid for the frozen layers.

## Already optimal — do not "fix"

- **`Attention`**: SDPA already selects flash (36.2 ms default / 36.5 ms forced
  flash / 44.3 ms mem-efficient). The strided `_shape` view benchmarks *identically*
  to a contiguous copy, and `sdpa_out.transpose(1,2).reshape(...)` is a **free view**
  — flash returns the matching layout, so there is no hidden copy.
- **Frozen layers** already build no autograd graph (inputs and params both have
  `requires_grad=False`), so wrapping them in `no_grad` gains nothing.
- Autograd correctly skips the input-gradient of the first *trainable* vision block,
  since the preceding frozen block's output does not require grad.

## Minor findings (~1% combined)

- The loss does ~16 CPU↔GPU syncs per step: 8× `cost.cpu()` in
  `hungarian_match_text` and 8× `torch.nonzero` in `_hard_negative_indices`
  (data-dependent output shape). Total loss cost is only 34 ms, so the ceiling is ~1%.
- `tokenize()` on 46 prompts every step costs 1.63 ms and overlaps with async GPU
  work. Pre-tokenizing the fixed prompt pools is cleanup, not a speedup.
- `VisionTower.forward` computes a `pooled_output` the detection path discards, and
  `post_layernorm` runs twice.
- `ClassPredictionHead` uses two `Linear(d→1)` for shift/scale; could be one
  `Linear(d→2)`.

## Evaluation is ~17% of wall clock

1158 val images / bs 8 = 145 batches × ~2.3 s ≈ **5.5 min per eval**. With
`--eval-every-steps 550` over 2200 steps that is 4 evals ≈ 22 min on top of ~107 min
of training. Eval peaks at only 2.8 GiB, so a much larger eval batch size (or
`--eval-max-batches` on intermediate evals) is the cheap fix.

## Loss weighting bug (found 2026-08-02)

`sigmoid_focal_loss(..., reduction="mean")` in `text_loss.py` averages over
`[num_matched, num_classes]`, dividing the classification loss by `num_classes`
(46x for Fashionpedia). The `λ_cls=1, λ_l1=5, λ_giou=2` weights are taken from
DETR/D-FINE, which normalize classification as `sum / num_boxes`.

Measured contribution from the live run's own MLflow metrics:

| Component | raw | λ | weighted | share |
|---|---:|---:|---:|---:|
| `L_cls` | 0.0036 | 1.0 | 0.0036 | **0.8%** |
| `L_l1` | 0.0075 | 5.0 | 0.0374 | 7.8% |
| `L_giou` | 0.1860 | 2.0 | 0.3719 | **77.7%** |
| `L_obj` | 0.1319 | 0.5 | 0.0660 | 13.8% |

So the class head — the primary trainable module — receives under 1% of the
gradient signal. See `docs/loss-notes.md`.
