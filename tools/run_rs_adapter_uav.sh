#!/usr/bin/env bash
# UAV leg of the RS adapter experiment (docs/rs-adapter-run-plan.md): zero-shot
# and FLAME 5-shot (support seeds 0/1/2, sampled from each dataset's own train
# split) on Songdo test and MAVREC aerial-val, for pretrained base and every
# adapter checkpoint given. Runs in the openvocab-evals harness next to this
# repo (its flame_support detector arg needs OWLv2torch >= e08a6e5).
#
#   tools/run_rs_adapter_uav.sh [checkpoint ...]
#   (default: R1 and R2 best.pth; pick the GPU with CUDA_VISIBLE_DEVICES)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
EVALS=${EVALS:-$REPO/../evals}
OUT=$REPO/logs/rs_adapter
mkdir -p "$OUT"
if [ $# -eq 0 ]; then
  set -- "$REPO"/text_checkpoints/R1_dota_hybrid_vb6_lora_first6_seed0/best.pth \
         "$REPO"/text_checkpoints/R2_dota_lora_all12_seed0/best.pth
fi
# uav-eval runs from $EVALS, so make checkpoint paths absolute (and fail early if missing)
checkpoints=(); for c in "$@"; do checkpoints+=("$(realpath -e "$c")"); done
set -- "${checkpoints[@]}"

# dataset, eval split, support split, tile size (the split's default arm)
TARGETS=("songdo test train 1280" "mavrec aerial-val aerial-train-subset 1024")

run() {  # output label, then detector args
  local out=$1; shift
  [ -f "$OUT/$out.json" ] && return
  (cd "$EVALS" && uv run --no-sync uav-eval \
    --dataset "$dataset" --split "$split" --tile-size "$tile" --batch-size 16 \
    --detector owlv2 --detector-arg model_size=base "$@" \
    --save-metrics "$OUT/$out.json") 2>&1 | tee "$OUT/$out.log"
}

for target in "${TARGETS[@]}"; do
  read -r dataset split support tile <<<"$target"
  for checkpoint in base "$@"; do
    if [ "$checkpoint" = base ]; then label=base; weights=()
    else label=$(basename "$(dirname "$checkpoint")"); weights=(--detector-arg "checkpoint=$checkpoint"); fi
    run "uav_${dataset}_${label}_zs" "${weights[@]}"
    for s in 0 1 2; do
      run "uav_${dataset}_${label}_flame_s$s" "${weights[@]}" \
        --detector-arg "flame_support=$dataset:$support" --detector-arg "flame_tile_size=$tile" \
        --detector-arg "flame_seed=$s"
    done
  done
done
