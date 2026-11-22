#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Precompute LONG-CONTEXT VisionTS++ forecasts (teacher look-back from the paper),
# one job per (dataset, horizon), spread over GPUS. Run in zohar_env BEFORE training
# with SOURCE=visionts_long. Teacher look-back / P come from get_long() in
# vts_pred_reg_multi.sh (ETTm1 2304/96, ETTm2 4032/96, ...).
#
# Usage:
#   GPUS="0 1 2 3" DATASETS="ETTm1 ETTm2" bash scripts/long_term_forecast/precompute_visionts_long.sh
# ═══════════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/../.."
_PRE_GPUS="${GPUS:-${GPU:-0}}"
DATASETS="${DATASETS:-ETTm1 ETTm2}"
source scripts/long_term_forecast/vts_pred_reg_multi.sh   # get_ds / get_long (no jobs launched)
GPUS="$_PRE_GPUS"
PRE_LOG_DIR="${PRE_LOG_DIR:-logs/precompute_visionts_long}"
mkdir -p "$PRE_LOG_DIR"

jobs=()
for ds in $DATASETS; do
  get_long "$ds" >/dev/null || exit 1
  hz="$(get_ds "$ds" | cut -d'|' -f6)"
  for pl in ${HORIZONS:-$hz}; do jobs+=("${ds}:${pl}"); done
done

pre_one() {  # ds pl gpu
  local ds=$1 pl=$2 gpu=$3 T P
  IFS='|' read -r T P <<< "$(get_long "$ds")"
  local logf="$PRE_LOG_DIR/${ds}_T${T}_P${P}_${pl}.log"
  echo "[gpu $gpu] START $ds pl$pl (teacher T=$T P=$P)"
  CUDA_VISIBLE_DEVICES=$gpu python -u utils/precompute_visionts_long_forecast.py \
    --dataset "$ds" --pred_len "$pl" --teacher_seq_len "$T" --periodicity "$P" \
    --ckpt_dir "$CKPT_DIR" > "$logf" 2>&1 \
    && echo "[gpu $gpu] DONE  $ds pl$pl  ($(grep -m1 Coverage "$logf"))" \
    || echo "[gpu $gpu] FAIL  $ds pl$pl -> $logf"
}

gpu_arr=($GPUS); ngpu=${#gpu_arr[@]}
echo "Precompute jobs: ${#jobs[@]} across GPUs: $GPUS ; logs -> $PRE_LOG_DIR/"
for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"
  ( j=$idx
    while [ $j -lt ${#jobs[@]} ]; do
      IFS=':' read -r ds pl <<< "${jobs[$j]}"; pre_one "$ds" "$pl" "$gpu"; j=$((j + ngpu))
    done ) &
done
wait
echo "All precompute jobs finished."
