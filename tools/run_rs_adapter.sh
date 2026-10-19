#!/usr/bin/env bash
# RS adapter experiment (docs/rs-adapter-run-plan.md): train a DOTA-only adapter,
# then FLAME 5-shot on DIOR (shared vs DIOR-only classes) and the COCO mini500
# forgetting probe, for pretrained base, the adapter, and its WiSE-FT midpoint.
#
#   tools/run_rs_adapter.sh hybrid   # vb6 + Q/V LoRA first 6 (also runs the base FLAME baseline)
#   tools/run_rs_adapter.sh lora12   # Q/V LoRA on all 12 blocks, no full blocks
#
# Pick the GPU with CUDA_VISIBLE_DEVICES (0 = 5070 Ti, 1 = 3060 here).
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${1:?usage: $0 hybrid|lora12}
SEED=${SEED:-0}
OUT=logs/rs_adapter
mkdir -p "$OUT"

DOTA=/mnt/datasets/sat/dota_coco
COCO_ANN=/mnt/datasets/coco/annotations/instances_val2017_mini500.json
COCO_IMG=/mnt/datasets/coco/val2017

COMMON=(
  --train-annotations $DOTA/train/annotations.json --train-images $DOTA/train/images
  --val-annotations $DOTA/validation/annotations.json --val-images $DOTA/validation/images
  --general-val-annotations $COCO_ANN --general-val-images $COCO_IMG
  --model-type base --prompt-profile aerial
  --class-loss mal --class-loss-gamma 1.5
  --head-learning-rate 5e-5 --vision-learning-rate 1e-5
  --text-blocks 0 --text-learning-rate 1e-7
  --vision-lora-targets q_proj v_proj --vision-lora-rank 8 --vision-lora-alpha 8
  --vision-lora-dropout 0 --vision-lora-learning-rate 1e-4
  --batch-size 8 --val-batch-size 16 --num-workers 6
  --max-steps 4500 --warmup-steps 100 --eval-every-steps 500
  --augment --gpu-augment --right-angle-rotations --scale-augment --mosaic-prob 0
  --val-transform fast --amp --compile
  --negative-ratio 5 --eval-top-k 100 --weight-decay 1e-4
  --mlflow --mlflow-experiment RS-Adapter --seed "$SEED"
)
case $ARM in
  hybrid) TOPOLOGY=(--vision-blocks 6 --vision-lora-blocks 6 --vision-lora-placement first)
          NAME=R1_dota_hybrid_vb6_lora_first6_seed$SEED ;;
  lora12) TOPOLOGY=(--vision-blocks 0 --vision-lora-blocks 12 --vision-lora-placement first)
          NAME=R2_dota_lora_all12_seed$SEED ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
CKPT=text_checkpoints/$NAME/best.pth

if [ ! -f "$CKPT" ]; then
  uv run --extra train python prototype_train/train_text.py "${COMMON[@]}" "${TOPOLOGY[@]}" \
    --mlflow-run-name "$NAME" --output-dir "text_checkpoints/$NAME" 2>&1 | tee "$OUT/train_$NAME.log"
fi

flame() {  # label, then extra args
  local label=$1; shift
  for s in 0 1 2; do
    [ -f "$OUT/flame_${label}_s$s.json" ] && continue
    uv run --extra train python tools/test_dior_flame.py --support-images 5 --seed "$s" \
      --save-metrics "$OUT/flame_${label}_s$s.json" "$@" 2>&1 | tee "$OUT/flame_${label}_s$s.log"
  done
}
coco() {  # label, then extra args
  local label=$1; shift
  [ -f "$OUT/coco_${label}.json" ] && return
  uv run --extra train python tools/ovd_eval.py --dataset coco --ann-file $COCO_ANN --image-root $COCO_IMG \
    --model-size base --save-metrics "$OUT/coco_${label}.json" "$@" 2>&1 | tee "$OUT/coco_${label}.log"
}

if [ "$ARM" = hybrid ]; then
  flame base --model-size base
  coco base
fi
flame "$NAME" --checkpoint "$CKPT"
coco "$NAME" --checkpoint "$CKPT"
flame "${NAME}_wise0.5" --checkpoint "$CKPT" --wise-alpha 0.5
coco "${NAME}_wise0.5" --checkpoint "$CKPT" --wise-alpha 0.5
