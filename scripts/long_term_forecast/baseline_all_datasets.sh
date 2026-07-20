#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Plain baseline forecasting (NO VibeTS), any supported backbone, all horizons.
# Hyper-params + naming are IDENTICAL to the VibeTS all-datasets scripts and to
# scripts/robustness/forecast_noise.sh (seq_len=96, label_len=0), so:
#   * baseline vs VibeTS is a clean apples-to-apples comparison, and
#   * these checkpoints are reused by forecast_noise.sh's `baseline` variant.
#
#   STUDENT=iTransformer|PatchTST|TimeMixer   (default: iTransformer)
#   DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate"  (default: ETT only)
#   HORIZONS="96 192 336 720"
#   GPUS="0 1 2 3"    spread jobs across gpus (one job per gpu at a time)
#
# PatchTST ETT is tuned PER HORIZON (e_layers/n_heads/batch differ by pred_len),
# matching scripts/long_term_forecast/ETT_script/PatchTST_ETT*.sh.
#
# Usage (this request):
#   STUDENT=iTransformer bash scripts/long_term_forecast/baseline_all_datasets.sh
#   STUDENT=PatchTST    bash scripts/long_term_forecast/baseline_all_datasets.sh
# ═══════════════════════════════════════════════════════════════════════════
set -u

STUDENT="${STUDENT:-iTransformer}"      # iTransformer | PatchTST | TimeMixer
GPUS="${GPUS:-0 1 2 3}"
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2}"
HORIZONS="${HORIZONS:-96 192 336 720}"
LOG_DIR="${LOG_DIR:-logs/baseline_${STUDENT}}"
DS_WINDOW=2   # TimeMixer down-sampling window

mkdir -p "$LOG_DIR"

# Per-(STUDENT, dataset[, pred]) config. Field order:
#   data|root|dpath|enc_in|e_layers|d_model|d_ff|n_heads|factor|batch|ds_layers|lr|epochs|patience|dropout|seq_len
# (pred only matters for PatchTST ETT per-horizon tuning.)
get_cfg() {
  local ds=$1 pred=$2
  case "$STUDENT" in
    TimeMixer)
      case "$ds" in
        ETTh1)         echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|16|32|8|1|128|3|0.01|10|3|0.6|96" ;;
        ETTh2)         echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|16|32|8|1|128|3|0.01|10|3|0.6|96" ;;
        ETTm1)         echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7|2|16|32|8|1|128|3|0.01|10|3|0.1|96" ;;
        ETTm2)         echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7|2|32|64|8|1|128|3|0.01|10|3|0.1|96" ;;
        weather)       echo "custom|./dataset/weather/|weather.csv|21|2|16|32|8|1|128|3|0.01|20|10|0.1|96" ;;
        exchange_rate) echo "custom|./dataset/exchange_rate/|exchange_rate.csv|8|2|16|32|8|1|32|3|0.01|10|3|0.1|96" ;;
        *) echo "UNKNOWN dataset: $ds" >&2; return 1 ;;
      esac ;;
    iTransformer)
      case "$ds" in
        ETTh1)         echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|128|128|8|3|32|0|0.0001|10|3|0.1|96" ;;
        ETTh2)         echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|128|128|8|3|32|0|0.0001|10|3|0.1|96" ;;
        ETTm1)         echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7|2|128|128|8|3|32|0|0.0001|10|3|0.1|96" ;;
        ETTm2)         echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7|2|128|128|8|3|32|0|0.0001|10|3|0.1|96" ;;
        weather)       echo "custom|./dataset/weather/|weather.csv|21|3|512|512|8|3|32|0|0.0001|10|3|0.1|96" ;;
        exchange_rate) echo "custom|./dataset/exchange_rate/|exchange_rate.csv|8|2|128|128|8|3|32|0|0.0001|10|3|0.1|96" ;;
        *) echo "UNKNOWN dataset: $ds" >&2; return 1 ;;
      esac ;;
    PatchTST)
      # d_model/d_ff use run.py defaults (512/2048). Weather nh=4 epochs=3;
      # Exchange nh=8. ETT is tuned PER HORIZON.
      case "$ds" in
        weather)       echo "custom|./dataset/weather/|weather.csv|21|2|512|2048|4|3|32|0|0.0001|3|3|0.1|96" ;;
        exchange_rate) echo "custom|./dataset/exchange_rate/|exchange_rate.csv|8|2|512|2048|8|3|32|0|0.0001|10|3|0.1|96" ;;
        ETTh1|ETTh2|ETTm1|ETTm2)
          local el nh bt
          case "${ds}_${pred}" in
            ETTh1_96)  el=1; nh=2;  bt=32  ;;
            ETTh1_192) el=1; nh=8;  bt=32  ;;
            ETTh1_336) el=1; nh=8;  bt=32  ;;
            ETTh1_720) el=1; nh=16; bt=32  ;;
            ETTh2_96|ETTh2_192|ETTh2_336|ETTh2_720) el=3; nh=4; bt=32 ;;
            ETTm1_96)  el=1; nh=2;  bt=32  ;;
            ETTm1_192) el=3; nh=2;  bt=128 ;;
            ETTm1_336) el=1; nh=4;  bt=128 ;;
            ETTm1_720) el=3; nh=4;  bt=128 ;;
            ETTm2_96)  el=3; nh=16; bt=32  ;;
            ETTm2_192) el=3; nh=2;  bt=128 ;;
            ETTm2_336) el=1; nh=4;  bt=32  ;;
            ETTm2_720) el=3; nh=4;  bt=128 ;;
            *) echo "UNKNOWN PatchTST ETT (ds=$ds pred=$pred); HORIZONS must be in {96,192,336,720}" >&2; return 1 ;;
          esac
          echo "${ds}|./dataset/ETT-small/|${ds}.csv|7|${el}|512|2048|${nh}|3|${bt}|0|0.0001|10|3|0.1|96" ;;
        *) echo "UNKNOWN dataset: $ds" >&2; return 1 ;;
      esac ;;
    *) echo "UNKNOWN STUDENT: $STUDENT" >&2; return 1 ;;
  esac
}

run_one() {
  # args: dataset_name pred_len gpu_id
  local name=$1 pred_len=$2 gpu=$3
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff n_heads factor batch_size ds_layers lr epochs patience dropout seq_len <<< "$(get_cfg "$name" "$pred_len")" || return 1

  # model_id matches the noise script's baseline naming (${name}_${seq}_${pred}).
  local model_id="${name}_${seq_len}_${pred_len}"
  local logf="$LOG_DIR/${model_id}.log"
  local donef="$LOG_DIR/${model_id}.done"

  if [ -f "$donef" ]; then
    echo "[gpu $gpu] SKIP (already done): $STUDENT $model_id"
    return 0
  fi

  local extra=""
  if [ "$STUDENT" = "TimeMixer" ]; then
    extra="--down_sampling_layers $ds_layers --down_sampling_method avg --down_sampling_window $DS_WINDOW"
  fi

  echo "[gpu $gpu] START: $STUDENT $model_id  (el$e_layers nh$n_heads b$batch_size)"
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model $STUDENT \
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
    $extra \
    > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$donef"
    echo "[gpu $gpu] DONE:  $STUDENT $model_id"
  else
    echo "[gpu $gpu] FAIL (rc=$rc): $STUDENT $model_id  ->  see $logf"
  fi
  return $rc
}

worker() {
  local gpu=$1; shift
  for job in "$@"; do
    IFS=':' read -r name pred_len <<< "$job"
    run_one "$name" "$pred_len" "$gpu"
  done
}

# Build job list (name:pred_len).
job_list=""
for name in $DATASETS; do
  for pred_len in $HORIZONS; do
    job_list="$job_list ${name}:${pred_len}"
  done
done

gpu_arr=($GPUS)
job_arr=($job_list)
ngpu=${#gpu_arr[@]}
echo "Baseline backbone: $STUDENT   Datasets: $DATASETS"
echo "Dispatching ${#job_arr[@]} runs across $ngpu gpu(s): $GPUS"
echo "Logs: $LOG_DIR/<model_id>.log"

for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"
  sel=""
  j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do
    sel="$sel ${job_arr[$j]}"
    j=$((j + ngpu))
  done
  nsel=$(printf '%s\n' $sel | grep -c . || true)
  echo "  gpu $gpu  <-  $nsel runs"
  worker "$gpu" $sel &
done

wait
echo "All gpus finished. Per-run mse/mae are in $LOG_DIR/<model_id>.log and result_long_term_forecast.txt"
