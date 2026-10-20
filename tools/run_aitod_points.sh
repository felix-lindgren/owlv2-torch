#!/usr/bin/env bash
# Point-detector concept test on an AI-TOD-v2 (xView part) vehicle subset: train
# the same recipe with the box loss and with --point-loss, and score both (plus
# the zero-shot weights) with OKS point AP on box centres.
#
#   tools/run_aitod_points.sh          # default arms, seed 0
#   SEED=1 tools/run_aitod_points.sh
#   SEED=1 ARMS="P3 P4" tools/run_aitod_points.sh      # only the point-slot arms
#   EXTRA_ARGS=--grad-checkpointing tools/run_aitod_points.sh   # 12 GB cards with slots
#
# Subsets: /mnt/datasets/sat/xview_aitod_v2/subsets/vehicle_{train1000,val200}.json
# (seed-0 random images, vehicle annotations only).
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-0}
STEPS=${STEPS:-2000}
# P1 (point loss without the w/h L1) lost on seed 0 and is off by default.
ARMS=${ARMS:-P0 P2 P3 P4}
read -ra EXTRA <<< "${EXTRA_ARGS:-}"
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
  [[ " $ARMS " == *" ${name%%_*} "* ]] || return 0
  [ -f "text_checkpoints/$name/final.pth" ] && return
  uv run --extra train python prototype_train/train_text.py "${COMMON[@]}" "${EXTRA[@]}" "$@" \
    --mlflow-run-name "$name" --output-dir "text_checkpoints/$name" 2>&1 | tee "$OUT/train_$name.log"
}

run P0_box_loss_seed$SEED --run-zero-shot-baseline
run P1_point_loss_seed$SEED --point-loss
run P2_point_loss_wh2.5_seed$SEED --point-loss --point-size-weight 2.5
# 2b: 2x2 point slots per token (stride-8 output), with the point and the box loss.
run P3_point_slots2_wh2.5_seed$SEED --point-loss --point-size-weight 2.5 --point-slot-grid 2
run P4_box_slots2_seed$SEED --point-slot-grid 2
