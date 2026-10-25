#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Standalone VisionTS++ baseline forecaster (NOT your JEPA model).
# Ablation: channel-independent (ci) vs colorized-multivariate (mv) rendering,
# same frozen VisionTS++ weights, to see how much the colorized multivariate
# packing helps. Datasets: ETTh1, ETTh2, exchange_rate.
#
# Each dataset is run at TWO look-backs:
#   - seq_len 96          -> directly comparable to your JEPA runs (short context)
#   - a long context      -> VisionTS++ at its intended strength (ETT 1728, exch 512)
#
# Setup (once): pip install git+https://github.com/Keytoyze/VisionTS.git
#
# Usage:
#   bash scripts/long_term_forecast/visionts_baseline.sh
#   GPUS="0 1 2 3" DATASETS="ETTh1 ETTh2 exchange_rate" bash scripts/long_term_forecast/visionts_baseline.sh
# ═══════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/../.."

GPUS="${GPUS:-0 1 2 3}"
DATASETS="${DATASETS:-ETTh1 ETTh2 exchange_rate}"
MODES="${MODES:-ci mv}"
HORIZONS="${HORIZONS:-96 192 336 720}"
ARCH="${ARCH:-mae_base}"
CKPT_DIR="${CKPT_DIR:-./hf_models/VisionTSpp}"
RESULTS_FILE="${RESULTS_FILE:-result_visionts_baseline.txt}"
LOG_DIR="${LOG_DIR:-logs/visionts_baseline}"
mkdir -p "$LOG_DIR"

# Per-dataset look-backs & periodicity:  "short_seq long_seq periodicity batch_size"
# ETT hourly -> P=24; exchange daily FX -> P=1 (no strong cycle). Long context is
# capped by each dataset's test-split length.
get_cfg() {
  case "$1" in
    ETTh1)         echo "96 1728 24 32" ;;
    ETTh2)         echo "96 1728 24 32" ;;
    exchange_rate) echo "96 512 1 32" ;;
    *) echo "UNKNOWN $1" >&2; return 1 ;;
  esac
}

run_one() {
  # args: dataset seq_len periodicity batch_size mode pred_len gpu
  local name=$1 seq=$2 P=$3 bs=$4 mode=$5 pl=$6 gpu=$7
  local id="VisionTSpp_baseline_${name}_${mode}_${seq}_${pl}"
  local logf="$LOG_DIR/${id}.log"
  local donef="$LOG_DIR/${id}.done"
  if [ -f "$donef" ]; then echo "[gpu $gpu] SKIP: $id"; return 0; fi
  echo "[gpu $gpu] START: $id"
  CUDA_VISIBLE_DEVICES=$gpu python -u utils/eval_visionts_baseline.py \
    --dataset "$name" --mode "$mode" --arch "$ARCH" \
    --seq_len "$seq" --pred_len "$pl" --periodicity "$P" \
    --batch_size "$bs" --ckpt_dir "$CKPT_DIR" \
    --results_file "$RESULTS_FILE" > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$donef"; echo "[gpu $gpu] DONE:  $id"
  else echo "[gpu $gpu] FAIL (rc=$rc): $id -> $logf"; fi
  return $rc
}

worker() {
  local gpu=$1; shift
  for job in "$@"; do
    IFS='|' read -r name seq P bs mode pl <<< "$job"
    run_one "$name" "$seq" "$P" "$bs" "$mode" "$pl" "$gpu"
  done
}

# Build job list: dataset × {short,long} look-back × mode × horizon.
job_list=""
for name in $DATASETS; do
  read -r short long P bs <<< "$(get_cfg "$name")"
  for seq in "$short" "$long"; do
    for mode in $MODES; do
      for pl in $HORIZONS; do
        job_list="$job_list ${name}|${seq}|${P}|${bs}|${mode}|${pl}"
      done
    done
  done
done

gpu_arr=($GPUS); job_arr=($job_list); ngpu=${#gpu_arr[@]}
echo "Dispatching ${#job_arr[@]} jobs across $ngpu gpu(s): $GPUS"
echo "Results -> $RESULTS_FILE ; logs -> $LOG_DIR/"

for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"; sel=""; j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do sel="$sel ${job_arr[$j]}"; j=$((j + ngpu)); done
  worker "$gpu" $sel &
done

wait
echo "All gpus finished. Results in $RESULTS_FILE"
