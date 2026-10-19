#!/usr/bin/env bash
# Point-detector concept test on an AI-TOD-v2 (xView part) vehicle subset: train
# the same recipe with the box loss and with --point-loss, and score both (plus
# the zero-shot weights) with OKS point AP on box centres.
#
#   tools/run_aitod_points.sh          # both arms, seed 0
#   SEED=1 tools/run_aitod_points.sh
#
# Subsets: /mnt/datasets/sat/xview_aitod_v2/subsets/vehicle_{train1000,val200}.json
# (seed-0 random images, vehicle annotations only).
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-0}
STEPS=${STEPS:-2000}
OUT=logs/aitod_points
mkdir -p "$OUT"
AITOD=/mnt/datasets/sat/xview_aitod_v2

COMMON=(
  --train-annotations $AITOD/subsets/vehicle_train1000.json --train-images $AITOD/train/images
  --val-annotations $AITOD/subsets/vehicle_val200.json --val-images $AITOD/val/images
  --model-type base --prompt-profile aerial
  --class-loss mal --class-loss-gamma 1.5
  --head-learning-rate 5e-5 --vision-learning-rate 1e-5
  --text-blocks 0 --text-learning-rate 1e-7
  --vision-blocks 6 --vision-lora-blocks 6 --vision-lora-placement first
  --vision-lora-targets q_proj v_proj --vision-lora-rank 8 --vision-lora-alpha 8
  --vision-lora-dropout 0 --vision-lora-learning-rate 1e-4
  --batch-size 8 --val-batch-size 16 --num-workers 6
  --max-steps "$STEPS" --warmup-steps 100 --eval-every-steps 250
  # Zoom-out would shrink 10 px vehicles further; flips and rotations stay.
  --augment --gpu-augment --right-angle-rotations --no-scale-augment --mosaic-prob 0
  --val-transform fast --amp --compile
  --negative-ratio 5 --eval-top-k 1500 --weight-decay 1e-4
  --point-eval
  --mlflow --mlflow-experiment AITOD-Points --seed "$SEED"
)

run() {  # name, then extra args
  local name=$1; shift
  [ -f "text_checkpoints/$name/final.pth" ] && return
  uv run --extra train python prototype_train/train_text.py "${COMMON[@]}" "$@" \
    --mlflow-run-name "$name" --output-dir "text_checkpoints/$name" 2>&1 | tee "$OUT/train_$name.log"
}

run P0_box_loss_seed$SEED --run-zero-shot-baseline
run P1_point_loss_seed$SEED --point-loss
run P2_point_loss_wh2.5_seed$SEED --point-loss --point-size-weight 2.5
