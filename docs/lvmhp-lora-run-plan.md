# Run plan — vision LoRA on OWLv2-base / LV-MHP-v1

Status: implementation ready; experimental checkpoints have not yet been
produced. The CLI and checkpoint requirements below define the experiment.

This is a continuation of `docs/lvmhp-run-plan.md`, not a replacement for it.
The completed results and their full trajectories remain in
`docs/lvmhp-findings.md`.

Visual prompt tuning is deliberately outside this block. It changes the vision
sequence and answers a different question; mixing it into the first LoRA sweep
would make a win or loss harder to attribute.

## Why this is the next useful experiment

The existing depth ladder isolates a narrow opportunity:

| adaptation | trainable parameters | annealed mAP @ 4,500 | vs. vb6 |
|---|---:|---:|---:|
| heads only (`vb0`) | 2.8M | 0.4303 | -0.0562 |
| last 2 blocks (`vb2`) | 16.9M | 0.4706 | -0.0159 |
| **last 6 blocks (`vb6`)** | **45.3M** | **0.4865** | — |
| all 12 blocks (`vb12`) | 87.8M | 0.4431 | -0.0434 |

Vision adaptation clearly matters, and it had not saturated at six blocks. The
all-block run did not fail like an overfit model: it rose, collapsed at step
1,575, then recovered erratically and was still improving at the final step.
The working explanation is that updating early generic features at the same
`1e-5` learning rate as the last six blocks destabilised the pretrained tower.

LoRA tests whether those early layers can move usefully if each weight update is
constrained to a low-rank subspace. There are three separate questions:

1. **Efficiency:** can Q/V LoRA on the last six blocks match fully tuning those
   blocks with about 15x fewer trainable parameters?
2. **Depth:** can Q/V LoRA across all 12 blocks use early-layer adaptation without
   reproducing the vb12 collapse?
3. **Accuracy:** can full tuning on the winning last six blocks plus LoRA on the
   first six beat vb6? This hybrid is the most direct response to the depth curve:
   retain the high-capacity update where it worked and constrain it where it did
   not.

The primary goal is an accuracy comparison, not merely a small checkpoint. A
pure-LoRA tie is nevertheless useful because it would reduce a vb6 trainable
delta from 45.3M to roughly 3M parameters including the detection heads.

## Fixed LoRA definition

For a frozen linear weight `W`, use

```
y = x W^T + (alpha / rank) x A^T B^T
```

with `A` randomly initialised and `B` zero-initialised. The zero `B` makes the
adapted model exactly equal to pretrained OWLv2 before the first optimiser step.

The first block uses this deliberately narrow configuration:

- vision tower only; the text tower remains frozen;
- `q_proj` and `v_proj` only;
- rank 8, `alpha=8`, so `alpha / rank = 1`;
- no LoRA dropout and no trainable LoRA bias;
- LoRA matrices get their own learning rate;
- the class, box and objectness heads remain fully trainable at `5e-5`;
- `vision_model.post_layernorm` and the detection `layer_norm` remain trainable at
  `1e-5`, matching vb6;
- patch embedding, positional embedding, class token and `visual_projection`
  remain frozen. `visual_projection` is not used by the detection path anyway;
- do not place LoRA and a full-weight update on the same block in this phase.

Q/V is intentionally the first target rather than every linear layer. It is the
smallest conventional LoRA intervention and cleanly answers whether adapting
attention routing is sufficient. K/O and MLP targets are a conditional follow-up,
not variables bundled into the first comparison.

For OWLv2-base (`d=768`), Q/V LoRA adds `4 * d * rank = 24,576` parameters per
block at rank 8:

| configuration | LoRA parameters | total trainable, including heads and norms |
|---|---:|---:|
| LoRA last 6, r8 | 147,456 | 2,912,007 |
| LoRA all 12, r8 | 294,912 | 3,059,463 |
| full last 6 + LoRA first 6, r8 | 147,456 | 45,439,239 |
| existing full last 6 | — | 45,291,783 |

The hybrid barely changes vb6's parameter count. Its purpose is accuracy, while
the two pure-LoRA arms measure parameter efficiency.

## Controlled protocol

Every arm uses the settled 15-class LV-MHP recipe. The comparison point is
**L4b seed 0 = 0.4865**, not L2b/vb2 at 0.4706. Keep all of the following fixed:

```bash
LVMHP_COMMON=(
  --train-annotations /mnt/datasets/fashion/lv_mhp_coco/train/annotations.json
  --train-images /mnt/datasets/fashion/lv_mhp_coco/train/images
  --val-annotations /mnt/datasets/fashion/lv_mhp_coco/val/annotations.json
  --val-images /mnt/datasets/fashion/lv_mhp_coco/val/images
  --model-type base
  --merge-class-names
    left_arm+right_arm=arm
    left_shoe+right_shoe=shoe
    left_leg+right_leg=leg
  --class-loss mal
  --class-loss-gamma 1.5
  --head-learning-rate 5e-5
  --vision-learning-rate 1e-5
  --text-blocks 0
  --text-learning-rate 1e-7
  --batch-size 8
  --val-batch-size 16
  --num-workers 6
  --max-steps 4500
  --warmup-steps 100
  --weight-decay 1e-4
  --negative-ratio 5
  --lambda-class 1.0
  --lambda-l1 5.0
  --lambda-giou 2.0
  --lambda-objectness 0.5
  --eval-top-k 100
  --eval-every-steps 225
  --val-transform fast
  --prompt-profile generic
  --eval-prompt-template "a photo of {name}"
  --mosaic-prob 0
  --augment
  --gpu-augment
  --amp
  --compile
  --mlflow
  --mlflow-experiment OwlV2-LVMHP-LoRA
  --seed 0
)
```

The array is only the shared portion. Each launch must additionally set the GPU,
full-block/LoRA topology, LoRA learning rate, run name and output directory.

Non-negotiable comparison rules inherited from the completed project:

- compare the final, fully annealed step 4,500 evaluation only;
- use the fixed 15-class vocabulary in every arm;
- treat an absolute difference below 0.003 mAP as a tie;
- use headline mAP for selection. `map_75` is the most useful supporting metric
  because vb6's gain was strongly localisation-driven;
- do not rank configurations on `map_small`, whose measured annealed noise floor
  is about 0.011;
- read final values from `mlflow.db`; logs are transcripts, not the authority;
- never initialise a LoRA arm from L4b. Every arm starts from pretrained OWLv2
  so the adaptation methods receive the same initial representation.

## Phase L6 — topology x learning-rate block

Rank, alpha and targets stay fixed. Cross three adaptation topologies with two
LoRA learning rates:

| run | full vision blocks | LoRA placement | LoRA LR | main comparison |
|---|---:|---|---:|---|
| **L6a** | 0 | last 6, Q/V r8 | `3e-5` | low-rank vs. full vb6 |
| **L6b** | 0 | last 6, Q/V r8 | `1e-4` | low-rank vs. full vb6 |
| **L6c** | 0 | all 12, Q/V r8 | `3e-5` | constrained early-layer access |
| **L6d** | 0 | all 12, Q/V r8 | `1e-4` | constrained early-layer access |
| **L6e** | 6 | first 6, Q/V r8 | `3e-5` | add safe early adaptation to vb6 |
| **L6f** | 6 | first 6, Q/V r8 | `1e-4` | add safe early adaptation to vb6 |

The full last-six weights in L6e/f remain at the existing
`--vision-learning-rate 1e-5`; only the early LoRA matrices use the LoRA LR.
This avoids repeating the Fashionpedia error of changing depth and the base
vision learning rate together.

The proposed per-arm flags are:

```bash
# Pure LoRA on the last six blocks
--vision-blocks 0 \
--vision-lora-blocks 6 --vision-lora-placement last \
--vision-lora-targets q_proj v_proj \
--vision-lora-rank 8 --vision-lora-alpha 8 \
--vision-lora-dropout 0 --vision-lora-learning-rate <3e-5|1e-4>

# Pure LoRA on all blocks
--vision-blocks 0 \
--vision-lora-blocks 12 --vision-lora-placement last \
--vision-lora-targets q_proj v_proj \
--vision-lora-rank 8 --vision-lora-alpha 8 \
--vision-lora-dropout 0 --vision-lora-learning-rate <3e-5|1e-4>

# Full last six plus LoRA first six
--vision-blocks 6 \
--vision-lora-blocks 6 --vision-lora-placement first \
--vision-lora-targets q_proj v_proj \
--vision-lora-rank 8 --vision-lora-alpha 8 \
--vision-lora-dropout 0 --vision-lora-learning-rate <3e-5|1e-4>
```

For example, after the memory probe decides whether checkpointing is needed,
L6d would launch as:

```bash
uv run --extra train python -u prototype_train/train_text.py \
  "${LVMHP_COMMON[@]}" \
  --device cuda:0 \
  --vision-blocks 0 \
  --vision-lora-blocks 12 \
  --vision-lora-placement last \
  --vision-lora-targets q_proj v_proj \
  --vision-lora-rank 8 \
  --vision-lora-alpha 8 \
  --vision-lora-dropout 0 \
  --vision-lora-learning-rate 1e-4 \
  --grad-checkpointing \
  --mlflow-run-name L6d_lora_all12_qv_r8_lr1e4_seed0 \
  --output-dir text_checkpoints/lvmhp_L6d_lora_all12_qv_r8_lr1e4_seed0
```

Remove `--grad-checkpointing` if the probe shows that this topology fits safely
and is faster without it.

Suggested execution order on two GPUs:

1. Run L6d and L6f first. They test the two highest-upside hypotheses at the
   primary `1e-4` LoRA LR.
2. Run L6b alongside whichever `3e-5` arm corresponds to the better result from
   step 1.
3. Complete the remaining two cells. Do not stop a run based on an intermediate
   lead: vb6 trailed vb2 until roughly step 2,475 and won only after annealing.

Completing all six cells is preferred. With two GPUs, the expected cost is
roughly 8–12 GPU-hours or 4–6 hours of wall time, pending the memory/step probe.

### What each difference means

- `best(L6a,b)` vs. L4b: whether low-rank Q/V updates can replace full updates at
  the same six-block depth.
- `best(L6c,d)` vs. `best(L6a,b)`: whether early blocks contribute when the
  update is constrained.
- `best(L6e,f)` vs. L4b: the clean accuracy test for adding constrained
  early-layer adaptation to the winning recipe.
- `best(L6c,d)` vs. L4c: whether the vb12 failure was caused by update freedom
  rather than depth itself. This is mechanistic context, not the primary ranking,
  because trainable capacity differs radically.

## Phase L7 — conditional capacity diagnosis

Do not automatically sweep every LoRA knob. Choose the next pair from the L6
result.

### If pure Q/V LoRA ties or beats vb6

At the winning pure-LoRA depth and LR, run rank 4 and rank 16 with
`alpha=rank`. Rank 8 is already available from L6.

| run | change | purpose |
|---|---|---|
| **L7a** | rank 4, `alpha=4` | find whether the same result survives at half the adapter size |
| **L7b** | rank 16, `alpha=16` | find whether rank 8 was capacity-limited |

Hold `alpha/rank=1`; otherwise the rank comparison also changes update scale.
All-12 Q/V has 147,456 / 294,912 / 589,824 LoRA parameters at ranks 4/8/16.

### If pure Q/V LoRA loses clearly

If the best pure arm is more than 0.003 below vb6, rank alone is not the first
suspect. At the better L6 depth and LR, run these two target expansions in
parallel at rank 8:

| run | targets | LoRA parameters if all 12 blocks | diagnosis |
|---|---|---:|---|
| **L7c** | Q, K, V and attention output | 589,824 | attention routing/output was under-parameterised |
| **L7d** | Q/V plus both MLP linears | 1,032,192 | dense channel adaptation is needed for localisation |

If neither recovers to within 0.003 of vb6, stop the pure-LoRA branch. It would
still produce smaller checkpoints, but this project is not storage-bound enough
to trade away a measurable accuracy gain.

### If the hybrid wins

Do not add targets or rank before confirming it. Its claim is an accuracy gain,
and another seed is more valuable than another hyperparameter after selecting on
the validation set.

## Phase L8 — confirmation and final evaluation

The development-set floor is based on only one paired two-seed comparison and is
optimistic. A candidate accuracy gain therefore needs a matched confirmation:

| run | condition |
|---|---|
| **L8a** | selected LoRA or hybrid configuration, seed 1 |
| **L8b** | existing vb6 recipe, seed 1 |

Run both, even though vb6 seed 0 already exists. Claim an accuracy gain only if
the LoRA configuration beats its vb6 counterpart in both seeds and the mean gain
is at least 0.003. A pure-LoRA result within 0.003 in both seeds can instead be
reported as a parameter-efficiency tie.

After selecting and confirming exactly one configuration:

1. run `--eval-only --per-class-ap` on its final checkpoint and the matched vb6
   checkpoint;
2. inspect mAP, `map_75`, medium/large AP and the per-class changes. Use small AP
   descriptively only;
3. evaluate both frozen choices once on the official LV-MHP test list. Do not use
   the official test result to choose rank, targets or LR.

## Memory and runtime calibration

Fewer trainable parameters do **not** imply head-only training memory. Once a
LoRA module appears, its output requires gradients, so every later frozen block
must retain or recompute an activation path:

- last-six LoRA should have a backward span similar to vb6 and may fit without
  checkpointing;
- all-12 LoRA and the hybrid create a gradient path from block 1 and probably
  need `--grad-checkpointing`, just as full vb12 did;
- LoRA removes base-weight gradients and most optimiser state, but the 3,601-token
  activation sequence remains the dominant risk.

Before L6, extend `tools/probe_step_cost.py` to accept the LoRA topology and probe
one Q/V r8 step for last-six, all-12 and hybrid at bs8 with compile on. Record
step time and peak reserved memory with checkpointing off and on. Use the fastest
configuration that fits with at least roughly 0.5 GiB headroom.

If an all-depth arm requires checkpointing, that is acceptable. Checkpointing is
mathematically neutral and the completed project already records the same
confound for L4c. If desired, pair the first checkpointed LoRA wave with a vb6 +
checkpointing rerun, but do not make that rerun a prerequisite for the core block.

## Implementation contract

### Trainer and model

Add these CLI options. `--vision-lora-blocks 0` disables the complete feature and
must be the default; the remaining defaults are inert in that state:

- `--vision-lora-blocks INT` (`0` disables LoRA);
- `--vision-lora-placement {first,last}`;
- `--vision-lora-targets TARGET [TARGET ...]`;
- `--vision-lora-rank INT`;
- `--vision-lora-alpha FLOAT`;
- `--vision-lora-dropout FLOAT`;
- `--vision-lora-learning-rate FLOAT`.

Required behaviour:

- inject adapters only after pretrained weights have loaded;
- validate positive rank/alpha/LR and valid targets when LoRA is enabled;
- reject overlap between full-tuned and LoRA-adapted blocks for this phase;
- allow full last-six plus LoRA first-six for the hybrid;
- put only LoRA parameters in the LoRA optimiser group;
- train the two vision/detection norms whenever either full vision blocks or
  vision LoRA is active;
- keep rank 0 / zero LoRA blocks exactly backward-compatible with every existing
  checkpoint and command;
- count and print LoRA parameters separately from heads, norms and full blocks;
- add every LoRA topology and optimiser setting to `RESUME_CRITICAL_ARGS` and
  MLflow parameters.

Use PEFT for adapter injection, initialization, state conventions and safe
merging. Generate exact fully qualified targets from the resolved layer indices,
for example `vision_model.encoder.layers.6.self_attn.q_proj`, so an adapter can
never attach to a same-named projection in the text tower. Keep `OwlV2` as the
outer model and preserve the CLI and checkpoint schema above.

### Checkpoints and inference

The current delta checkpoint saves parameters solely by `requires_grad`, while
the loader constructs a plain `OwlV2` before loading. Raw LoRA keys would
therefore be unexpected unless the adapter topology is reconstructed first.

Add an adapter-aware checkpoint format/configuration that stores:

- placement, block count or exact layer indices;
- targets, rank, alpha and dropout;
- LoRA matrices plus the existing trainable heads and norms;
- whether full vision blocks are also present.

`--resume`, `--init-from`, `--eval-only`, best/final saving and per-class eval
must all reconstruct adapters before loading their state. Also provide a merge
path that uses PEFT's checked merge to fold each delta into its base weight. A
merged checkpoint should run through the ordinary OWLv2/TensorRT inference graph
with no adapter modules or additional latency.

### Manual smoke checks before GPU-hours

Use lightweight manual checks rather than building a separate verification
suite for the experiment:

1. With zero-initialised `B`, compare fixed-input detector outputs before and
   after LoRA injection; they should agree to floating-point tolerance.
2. Backpropagate one batch and print gradients: LoRA matrices, heads and the two
   norms should receive them; frozen base weights and the text tower should not.
3. Save and reload one step through both `--resume` and `--eval-only`, then check
   that fixed-input outputs reproduce.
4. Merge the adapters into a copy and compare its output with the unmerged LoRA
   model before launching the full block.
5. Confirm that the printed trainable counts match the table above for base/r8.

## Diagnostics to record

In addition to the existing metrics, record:

- total and LoRA-only trainable parameter counts;
- peak reserved memory and steady-state step time for each topology;
- checkpoint size for `final.pth` and `last.pth`;
- final per-block `||delta W||_F / ||W||_F` for each target projection.

The last diagnostic tells whether all-12 LoRA actually uses the early blocks. If
their relative deltas stay near zero, an all-depth tie does not support the claim
that early adaptation helped; it only says the extra adapters were harmless.

## Decision table

Against vb6 seed 0 at 0.4865:

| result | interpretation | next action |
|---|---|---|
| pure LoRA `< 0.4835` | measurable accuracy loss | run target diagnosis only if the curve is stable |
| pure LoRA `0.4835–0.4895` | parameter-efficiency tie | rank screen, then seed confirmation |
| pure LoRA `> 0.4895` | candidate accuracy gain | skip expansion and confirm seeds |
| hybrid `0.4835–0.4895` | no benefit over vb6 | stop hybrid branch |
| hybrid `> 0.4895` | candidate gain from constrained early layers | confirm immediately |
| all-12 LoRA unstable at both LRs | low rank did not solve early-layer instability | stop all-depth branch |

The likely useful outcomes are broader than “new best mAP”: matching vb6 with a
roughly 3.1M-parameter trainable delta would validate LoRA as the default cheap
adaptation method, while a hybrid win would show that the vb6 depth optimum was
an optimisation boundary rather than a representational one.
