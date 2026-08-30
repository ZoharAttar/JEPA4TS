#!/bin/bash
# VibeTS + TimeMixer + masked-horizon teacher on ETT* and weather.
# Single encoder, RP, all paper horizons (96/192/336/720). Prints MSE/MAE at the end.
#
# Prerequisite: per-variable DINO RP embeddings, e.g.
#   for d in ETTh1 ETTh2 ETTm1 ETTm2 weather; do
#     python utils/precompute_embeddings_pervar.py --dataset $d --method RP
#   done
# MAE-base weights download to ./ckpt/ on first --masked_horizon run.
#
#   bash scripts/long_term_forecast/JEPAVTS_TimeMixer_masked_horizon_ett_weather.sh
#   GPUS="0 1 2 3" HORIZON_WEIGHT=1 bash ...
#   PRED_LENS="96 192" bash ...   # subset of horizons
set -u

GPUS="${GPUS:-0}"
LOG_DIR="${LOG_DIR:-logs/jepavts_timemixer_masked_horizon}"
HORIZON_WEIGHT="${HORIZON_WEIGHT:-1}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather}"

MODEL=JEPAVTS
STUDENT=TimeMixer
RENDER=RP
JEPA_WEIGHT=1
JEPA_SCALE=coarse
JEPA_LOSS=mse
DS_WINDOW=2

# data|root_path|data_path|enc_in|e_layers|d_model|d_ff|batch_size|ds_layers|lr|epochs|patience|dropout|seq_len
get_cfg() {
  case "$1" in
    ETTh1)   echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|16|32|128|3|0.01|10|3|0.6|96" ;;
    ETTh2)   echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|16|32|128|3|0.01|10|3|0.6|96" ;;
    ETTm1)   echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7|2|16|32|128|3|0.01|10|3|0.1|96" ;;
    ETTm2)   echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7|2|32|64|128|3|0.01|10|3|0.1|96" ;;
    weather) echo "custom|./dataset/weather/|weather.csv|21|2|16|32|128|3|0.01|20|10|0.1|96" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

run_one() {
  local name=$1 pred_len=$2 gpu=$3
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff batch_size ds_layers lr epochs patience dropout seq_len <<< "$(get_cfg "$name")"

  local model_id="${name}_${RENDER}_${seq_len}_${pred_len}_single_masked_horizon"
  local logf="$LOG_DIR/${model_id}.log"
  local donef="$LOG_DIR/${model_id}.done"

  if [ -f "$donef" ]; then
    echo "[gpu $gpu] SKIP (already done): $model_id"
    return 0
  fi

  echo "[gpu $gpu] START: $model_id"
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model $MODEL \
    --student_model $STUDENT \
    --data "$data" \
    --features M \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --c_out "$enc_in" \
    --des Exp \
    --itr 1 \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --learning_rate "$lr" \
    --train_epochs "$epochs" \
    --patience "$patience" \
    --batch_size "$batch_size" \
    --dropout "$dropout" \
    --down_sampling_layers "$ds_layers" \
    --down_sampling_method avg \
    --down_sampling_window $DS_WINDOW \
    --timemixer_jepa_scale $JEPA_SCALE \
    --per_var_teacher \
    --rendering_methods $RENDER \
    --jepa_weight $JEPA_WEIGHT \
    --jepa_loss_type $JEPA_LOSS \
    --masked_horizon \
    --horizon_weight $HORIZON_WEIGHT \
    > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$donef"
    echo "[gpu $gpu] DONE:  $model_id"
  else
    echo "[gpu $gpu] FAIL (rc=$rc): $model_id  ->  see $logf"
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

fc_metric() {
  local f=$1 mse mae
  [ -f "$f" ] || { echo "NA NA"; return; }
  mse=$(grep -E "^[[:space:]]*MSE:" "$f" | tail -1 | sed -E 's/.*MSE:[[:space:]]*([0-9.eE+-]+).*/\1/')
  mae=$(grep -E "^[[:space:]]*MAE:" "$f" | tail -1 | sed -E 's/.*MAE:[[:space:]]*([0-9.eE+-]+).*/\1/')
  echo "${mse:-NA} ${mae:-NA}"
}

print_summary() {
  echo ""
  echo "=================================================================="
  echo " MASKED-HORIZON  TimeMixer  RP  (MSE / MAE)"
  echo " LOG_DIR=$LOG_DIR  HORIZON_WEIGHT=$HORIZON_WEIGHT"
  echo "=================================================================="
  hdr="Dataset"
  for p in $PRED_LENS; do hdr="$hdr | $p"; done
  hdr="$hdr | Avg"
  echo "$hdr"
  for name in $DATASETS; do
    row="$name"
    sum_mse=0; sum_mae=0; n=0
    for pred in $PRED_LENS; do
      seq_len=$(get_cfg "$name" | awk -F'|' '{print $NF}')
      logf="$LOG_DIR/${name}_${RENDER}_${seq_len}_${pred}_single_masked_horizon.log"
      read -r mse mae <<< "$(fc_metric "$logf")"
      row="$row | ${mse}/${mae}"
      if [ "$mse" != "NA" ] && [ -n "$mse" ]; then
        sum_mse=$(awk "BEGIN{print $sum_mse+$mse}")
        sum_mae=$(awk "BEGIN{print $sum_mae+$mae}")
        n=$((n+1))
      fi
    done
    if [ "$n" -gt 0 ]; then
      avg_mse=$(awk "BEGIN{printf \"%.4f\", $sum_mse/$n}")
      avg_mae=$(awk "BEGIN{printf \"%.4f\", $sum_mae/$n}")
      row="$row | ${avg_mse}/${avg_mae}"
    else
      row="$row | NA"
    fi
    echo "$row"
  done
  echo ""
  echo "Per-horizon (copy-friendly):"
  for name in $DATASETS; do
    seq_len=$(get_cfg "$name" | awk -F'|' '{print $NF}')
    for pred in $PRED_LENS; do
      logf="$LOG_DIR/${name}_${RENDER}_${seq_len}_${pred}_single_masked_horizon.log"
      read -r mse mae <<< "$(fc_metric "$logf")"
      printf "%-10s %4s  MSE: %s  MAE: %s\n" "$name" "$pred" "$mse" "$mae"
    done
  done
  echo ""
  echo "Logs: $LOG_DIR/<dataset>_RP_96_<pred>_single_masked_horizon.log"
}

mkdir -p "$LOG_DIR"

job_list=""
for name in $DATASETS; do
  get_cfg "$name" >/dev/null || exit 1
  for pred_len in $PRED_LENS; do
    job_list="$job_list ${name}:${pred_len}"
  done
done

gpu_arr=($GPUS)
job_arr=($job_list)
ngpu=${#gpu_arr[@]}
echo "Dispatching ${#job_arr[@]} experiments across $ngpu gpu(s): $GPUS"
echo "Datasets: $DATASETS"
echo "Horizons: $PRED_LENS"
echo "Logs: $LOG_DIR"

for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"
  sel=""
  j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do
    sel="$sel ${job_arr[$j]}"
    j=$((j + ngpu))
  done
  [ -z "$sel" ] && continue
  nsel=$(printf '%s\n' $sel | grep -c . || true)
  echo "  gpu $gpu  <-  $nsel experiments"
  worker "$gpu" $sel &
done
wait
echo "All gpus finished."
print_summary
