#!/usr/bin/env bash
# P3 train_all recipe from tools/run_aitod_points.sh (no MLflow), with the arch-speedup flags.
#   NAME=E1_keep50 STEPS=7400 tools/arch_speedup/run_exp.sh --train-token-keep 0.5
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME=${NAME:?}
STEPS=${STEPS:-7400}
SEED=${SEED:-0}
AITOD=/mnt/datasets/sat/xview_aitod_v2
PY=/home/fellin/repos/openvocab/owlv2-torch/.venv/bin/python
mkdir -p logs text_checkpoints
resume=()
[ -f "text_checkpoints/$NAME/last.pth" ] && resume=(--resume "text_checkpoints/$NAME")
[ -f "text_checkpoints/$NAME/final.pth" ] && { echo "done: $NAME"; exit 0; }
$PY prototype_train/train_text.py \
  --train-annotations "$AITOD/subsets/vehicle_train_all.json" --train-images "$AITOD/train/images" \
  --val-annotations "$AITOD/subsets/vehicle_val200.json" --val-images "$AITOD/val/images" \
  --model-type base --prompt-profile aerial \
  --class-loss mal --class-loss-gamma 1.5 \
  --head-learning-rate 5e-5 --vision-learning-rate 1e-5 \
  --text-blocks 0 --text-learning-rate 1e-7 \
  --vision-blocks 6 --vision-lora-blocks 6 --vision-lora-placement first \
  --vision-lora-targets q_proj v_proj --vision-lora-rank 8 --vision-lora-alpha 8 \
  --vision-lora-dropout 0 --vision-lora-learning-rate 1e-4 \
  --batch-size 8 --val-batch-size 16 --num-workers 6 \
  --max-steps "$STEPS" --warmup-steps 100 --eval-every-steps 500 \
  --augment --gpu-augment --right-angle-rotations --no-scale-augment --mosaic-prob 0 \
  --val-transform fast --amp --compile \
  --fp16-accumulation --attention-bwd-fp16-accumulation \
  --negative-ratio 5 --eval-top-k 1500 --weight-decay 1e-4 \
  --point-eval --per-class-ap --seed "$SEED" \
  --point-loss --point-size-weight 2.5 --point-slot-grid 2 \
  "$@" "${resume[@]}" --output-dir "text_checkpoints/$NAME" 2>&1 | tee -a "logs/train_$NAME.log"
