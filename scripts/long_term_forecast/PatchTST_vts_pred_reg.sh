#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Plain PatchTST + VisionTS++ PREDICTION regularizer (output-space distillation).
#   L = MSE(student, y_true) + LAMBDA * MSE(student, VisionTS++ forecast)
# VisionTS++ is frozen, colorized-multivariate (MV), same 96 look-back as the
# student, computed on-the-fly each batch. This is NOT the JEPA model — it's a
# vanilla PatchTST trained with an extra prediction-matching loss.
#
# Setup (once): pip install git+https://github.com/Keytoyze/VisionTS.git
#
# Env knobs:
#   REG=1|0        1 = with VisionTS++ reg (default), 0 = plain PatchTST baseline
#   LAMBDA=1.0     weight on the regularizer term
#   DATASETS="ETTh1 ETTh2 exchange_rate"
#   GPUS="0 1 2 3"
#   HORIZONS="96 192 336 720"
#
# Usage:
#   bash scripts/long_term_forecast/PatchTST_vts_pred_reg.sh                 # with reg, ETT
#   REG=0 bash scripts/long_term_forecast/PatchTST_vts_pred_reg.sh           # baseline (no reg)
#   LAMBDA=0.5 DATASETS="ETTh1 ETTh2 exchange_rate" bash scripts/long_term_forecast/PatchTST_vts_pred_reg.sh
# ═══════════════════════════════════════════════════════════════════════════

GPUS="${GPUS:-0 1 2 3}"
DATASETS="${DATASETS:-ETTh1 ETTh2}"
HORIZONS="${HORIZONS:-96 192 336 720}"
REG="${REG:-1}"
LAMBDA="${LAMBDA:-1.0}"
CKPT_DIR="${CKPT_DIR:-./hf_models/VisionTSpp}"
LOG_DIR="${LOG_DIR:-logs/patchtst_vts_pred_reg}"
mkdir -p "$LOG_DIR"

# VisionTS++ image-rendering periodicity per dataset (ETT hourly=24, FX=1).
get_P() {
  case "$1" in
    ETTh1|ETTh2|ETTm1|ETTm2) echo 24 ;;
    exchange_rate)           echo 1 ;;
    weather)                 echo 24 ;;
    *) echo 24 ;;
  esac
}

# Per-(dataset,horizon) PatchTST config (matches JEPAVTS_PatchTST_all_datasets.sh).
#   data|root_path|data_path|enc_in|e_layers|d_model|d_ff|n_heads|factor|batch|lr|epochs|patience|dropout|seq_len
get_cfg() {
  local ds=$1 pred=$2
  case "$ds" in
    ETTh1|ETTh2|ETTm1|ETTm2)
      local el nh bt
      case "${ds}_${pred}" in
        ETTh1_96)  el=1; nh=2;  bt=32 ;;
        ETTh1_192) el=1; nh=8;  bt=32 ;;
        ETTh1_336) el=1; nh=8;  bt=32 ;;
        ETTh1_720) el=1; nh=16; bt=32 ;;
        ETTh2_96|ETTh2_192|ETTh2_336|ETTh2_720) el=3; nh=4; bt=32 ;;
        ETTm1_96)  el=1; nh=2;  bt=32  ;;
        ETTm1_192) el=3; nh=2;  bt=128 ;;
        ETTm1_336) el=1; nh=4;  bt=128 ;;
        ETTm1_720) el=3; nh=4;  bt=128 ;;
        ETTm2_96)  el=3; nh=16; bt=32  ;;
        ETTm2_192) el=3; nh=2;  bt=128 ;;
        ETTm2_336) el=1; nh=4;  bt=32  ;;
        ETTm2_720) el=3; nh=4;  bt=128 ;;
        *) echo "UNKNOWN ETT ($ds $pred)" >&2; return 1 ;;
      esac
      echo "${ds}|./dataset/ETT-small/|${ds}.csv|7|${el}|512|2048|${nh}|3|${bt}|0.0001|10|3|0.1|96" ;;
    weather)       echo "custom|./dataset/weather/|weather.csv|21|2|512|2048|4|3|32|0.0001|3|3|0.1|96" ;;
    exchange_rate) echo "custom|./dataset/exchange_rate/|exchange_rate.csv|8|2|512|2048|8|3|32|0.0001|10|3|0.1|96" ;;
    *) echo "UNKNOWN dataset: $ds" >&2; return 1 ;;
  esac
}

run_one() {
  local name=$1 pred_len=$2 gpu=$3
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff n_heads factor batch_size lr epochs patience dropout seq_len <<< "$(get_cfg "$name" "$pred_len")" || return 1
  local P; P=$(get_P "$name")

  local reg_args="" tag="base"
  if [ "$REG" = "1" ]; then
    reg_args="--vts_pred_reg --vts_pred_reg_weight $LAMBDA --vts_pred_periodicity $P --vts_pred_ckpt_dir $CKPT_DIR"
    tag="vtsreg${LAMBDA}"
  fi

  local model_id="${name}_PatchTST_${tag}_${seq_len}_${pred_len}"
  local logf="$LOG_DIR/${model_id}.log"
  local donef="$LOG_DIR/${model_id}.done"
  if [ -f "$donef" ]; then echo "[gpu $gpu] SKIP: $model_id"; return 0; fi

  echo "[gpu $gpu] START: $model_id  (el$e_layers nh$n_heads b$batch_size P$P reg=$REG)"
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model PatchTST \
    --data "$data" \
    --features M \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$enc_in" \
    --c_out "$enc_in" \
    --factor "$factor" \
    --n_heads "$n_heads" \
    --des Exp \
    --itr 1 \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --learning_rate "$lr" \
    --train_epochs "$epochs" \
    --patience "$patience" \
    --batch_size "$batch_size" \
    --dropout "$dropout" \
    $reg_args \
    > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$donef"; echo "[gpu $gpu] DONE:  $model_id"
  else echo "[gpu $gpu] FAIL (rc=$rc): $model_id -> $logf"; fi
  return $rc
}

worker() {
  local gpu=$1; shift
  for job in "$@"; do
    IFS=':' read -r name pred_len <<< "$job"
    run_one "$name" "$pred_len" "$gpu"
  done
}

job_list=""
for name in $DATASETS; do
  for pred_len in $HORIZONS; do
    job_list="$job_list ${name}:${pred_len}"
  done
done

gpu_arr=($GPUS); job_arr=($job_list); ngpu=${#gpu_arr[@]}
echo "PatchTST + VisionTS++ pred-reg (REG=$REG LAMBDA=$LAMBDA)   Datasets: $DATASETS"
echo "Dispatching ${#job_arr[@]} jobs across $ngpu gpu(s): $GPUS ; logs -> $LOG_DIR/"

for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"; sel=""; j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do sel="$sel ${job_arr[$j]}"; j=$((j + ngpu)); done
  worker "$gpu" $sel &
done

wait
echo "All gpus finished. Collect with: python utils/collect_results.py"
