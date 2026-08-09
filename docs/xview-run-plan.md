# Executable xView → DIOR run plan

This plan fixes the experimental contract before any fine-tuning result is
inspected. Training is on a source-level split of tiled xView; primary transfer
evaluation is the fixed 20-class DIOR test vocabulary.

## Frozen evaluation contract

`docs/xview-eval-spec.json` is the versioned source of truth. Every DIOR class
belongs to exactly one bucket:

| Bucket | Classes | Interpretation |
|---|---|---|
| direct-label overlap | Airplane, Ship, Storage tank, Vehicle | xView supplies an explicit label |
| context-related | Airport, Harbor | related xView objects/context, but no target label |
| unseen | the other 14, including Bridge | no xView target supervision |

Full DIOR-20 mAP and all three bucket mAP values are reported from the same
detections. Training ontology never changes this vocabulary or its averages.
Exact DIOR target names used as xView training queries are described as
**target-aligned cross-dataset transfer**. The unseen bucket measures
**open-vocabulary retention**.

The fixed evaluation prompt is `a satellite photo of {name}`. Alias pooling is
an eval-time diagnostic, not a metric redefinition: fine-grained queries are
max-pooled into the same target class before COCO scoring.

## 1. Choose source crop scale from measured object sizes

Generate annotation-only 640 and 960 arms. Both are resized to the 960 model
input; 1920 is not a primary arm.

```bash
for tile in 640 960; do
  UV_CACHE_DIR=/tmp/uv-cache uv run python tools/convert_xview_to_coco.py \
    --dataset-root /mnt/datasets/sat/xview \
    --output-dir /mnt/datasets/sat/xview_measure_${tile} \
    --tile-size "$tile" --output-size 960 \
    --category-crops-per-source 0 --empty-crop-policy keep \
    --no-materialize-images
done

UV_CACHE_DIR=/tmp/uv-cache uv run python tools/compare_object_scales.py \
  --xview 640=/mnt/datasets/sat/xview_measure_640/val/annotations.json \
  --xview 960=/mnt/datasets/sat/xview_measure_960/val/annotations.json \
  --output-json artifacts/xview/object-scales.json
```

The script measures `sqrt(box area)` after square-pad/resize for the four direct
classes and selects the arm with the smallest mean absolute log-median mismatch
to DIOR. Record its `baseline_arm` before training. A later scale comparison
uses these same converter flags and holds steps, prompts, sampling and
augmentation fixed. Consider a 640/960 mixture only after both individual arms
have been evaluated.

## 2. Build the final tiled dataset and inspect it

Replace `$TILE` below with the frozen baseline. The source split happens before
tiling. Uniform tiles use a non-overlapping grid and half-open centre ownership;
boxes are clipped once, filtered by visible area, and right/bottom edge crops
are zero-padded. Empty crops are explicit. The category-centred stream samples
a superclass by inverse square-root global frequency, then a box within it.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python tools/convert_xview_to_coco.py \
  --dataset-root /mnt/datasets/sat/xview \
  --output-dir /mnt/datasets/sat/xview_coco_${TILE} \
  --tile-size "$TILE" --output-size 960 \
  --min-box-visibility 0.2 \
  --empty-crop-policy sample --empty-crop-fraction 0.10 \
  --category-crops-per-source 16 \
  --render-samples 12
```

Inspect `train/previews/` and `val/previews/` before a complete conversion is
accepted. The converter chooses sparse, dense, edge-padded and seam-clipped
examples. `conversion_summary.json` records crop counts and retained positive
boxes per stream and mapped superclass; the same data is embedded in COCO
`info.xview_conversion`. Every image also records source image, crop origin,
source tile size, resampling scale, stream and padding.

## 3. Freeze prompts, geometry and sampling

Use the aerial profile. It samples only `{name}`, `a satellite photo of {name}`
and `an aerial photo of {name}`; no person/fashion prompt remains. For the
controlled 640-versus-960 comparison, disable RandomZoomOut/RandomIoUCrop and
mosaic while retaining aerially valid right-angle rotations and both flips.

The following flags are fixed for the baseline and any scale arm:

```text
--prompt-profile aerial
--right-angle-rotations
--horizontal-flip-prob 0.5
--vertical-flip-prob 0.5
--no-scale-augment
--mosaic-prob 0
--category-stream-fraction 0.5
```

`--category-stream-fraction` constructs a seeded epoch with the requested exact
expected ratio, sampling either stream with replacement when needed. Converter
exposure summaries show the actual available crop/positive distribution.

Do not drop Building merely to suppress its classification gradient: visible
unlabelled buildings would become objectness background. If that arm is needed,
retain all boxes and use `--classification-class-weight 'Building=0'`. Geometry
and objectness remain supervised while the Building query contributes no class
loss. The resolved category-to-query map, merges, weights and prompts are stored
under `training_ontology` in every checkpoint.

## 4. Establish matched evaluation baselines

The same command scores stock or fine-tuned weights. Without `--checkpoint` it
uses stock weights; with it, it loads either trainer delta or full checkpoints.
The standard COCO ceiling remains 100 even though inference retains 300
detections. Use `--eval-max-detections 300` only as a separately named,
non-standard dense-source xView metric.

```bash
# Stock base, fixed target queries and all frozen buckets.
UV_CACHE_DIR=/tmp/uv-cache uv run --extra train python tools/ovd_eval.py \
  --dataset dior --model-size base --top-k 300 \
  --eval-max-detections 100 \
  --save-metrics artifacts/xview/dior-stock-base.json

# Matched stock large baseline (same frozen buckets and prompt).
UV_CACHE_DIR=/tmp/uv-cache uv run --extra train python tools/ovd_eval.py \
  --dataset dior --model-size large --top-k 300 \
  --eval-max-detections 100 \
  --save-metrics artifacts/xview/dior-stock-large.json

# Eval-time vocabulary diagnostic before any merged-training arm.
UV_CACHE_DIR=/tmp/uv-cache uv run --extra train python tools/ovd_eval.py \
  --dataset dior --model-size base --alias-pooling --top-k 300 \
  --eval-max-detections 100 \
  --save-metrics artifacts/xview/dior-stock-base-aliases.json

# Identical path for fine-tuned weights; model size is read from the checkpoint.
UV_CACHE_DIR=/tmp/uv-cache uv run --extra train python tools/ovd_eval.py \
  --dataset dior --checkpoint text_checkpoints/X1-seed0/final.pth \
  --top-k 300 --eval-max-detections 100 \
  --save-metrics artifacts/xview/dior-X1-seed0.json

# Separate dense-source xView holdout metric (explicitly non-standard maxDets=300).
UV_CACHE_DIR=/tmp/uv-cache uv run --extra train python tools/ovd_eval.py \
  --dataset coco \
  --ann-file /mnt/datasets/sat/xview_coco_${TILE}/val/annotations.json \
  --image-root /mnt/datasets/sat/xview_coco_${TILE}/val/images \
  --checkpoint text_checkpoints/X1-seed0/final.pth \
  --query-template 'a satellite photo of {name}' \
  --top-k 300 --eval-max-detections 300 \
  --save-metrics artifacts/xview/xview-X1-seed0-maxdets300.json
```

Report the direct-overlap AP first, unseen AP as the retention signal, then
context-related AP and full-20 mAP. Never compare an alias-pooled number with a
single-query number; each trained result gets its matched stock baseline.

## 5. Probe representative density, then train

Probe median, upper-tail and dense-tail crops rather than the mean alone:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra train python tools/probe_step_cost.py \
  --annotations /mnt/datasets/sat/xview_coco_${TILE}/train/annotations.json \
  --density-quantiles 0.5 0.9 0.99 \
  --prompt-profile aerial --model-type base --device cuda:0 \
  --vision-blocks 0 2 6 --batch-sizes 8 16 32 \
  --compile-modes off on --mosaic-prob 0
```

Training keeps xView holdout validation separate from DIOR transfer evaluation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra train python prototype_train/train_text.py \
  --train-annotations /mnt/datasets/sat/xview_coco_${TILE}/train/annotations.json \
  --train-images /mnt/datasets/sat/xview_coco_${TILE}/train/images \
  --val-annotations /mnt/datasets/sat/xview_coco_${TILE}/val/annotations.json \
  --val-images /mnt/datasets/sat/xview_coco_${TILE}/val/images \
  --model-type base --vision-blocks 2 \
  --prompt-profile aerial --right-angle-rotations \
  --horizontal-flip-prob 0.5 --vertical-flip-prob 0.5 \
  --no-scale-augment --mosaic-prob 0 \
  --category-stream-fraction 0.5 \
  --output-dir text_checkpoints/X1-seed0 --seed 0
```

Use X1 seed 0 and seed 1 as the noise-floor pair. There is no separate duplicate
X0d run. Only after X1 is interpretable should an ontology arm be launched; its
complete merge specification is passed once with `--merge-class-names` and then
kept frozen.

## Acceptance checklist

- `object-scales.json` names the frozen 640 or 960 baseline.
- Crop previews cover sparse, dense, edge and seam cases and have been viewed.
- Conversion metadata gives deterministic crop and positive exposure for all
  four direct superclasses.
- `xview-eval-spec.json` partitions DIOR-20 exactly once and is unchanged.
- Stock and trained metrics use the same evaluator, prompt, alias policy and
  `maxDets` policy.
- Checkpoint `training_ontology` contains the resolved merges, prompt set and
  classification weights.
