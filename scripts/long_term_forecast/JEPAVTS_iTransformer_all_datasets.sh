#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# JEPAVTS + iTransformer student, all forecasting datasets, all horizons.
# For each (dataset, horizon): runs single-encoder AND dual-encoder, jepa_weight=1.
#
# PREREQUISITE — precompute per-variable DINO embeddings for every dataset first
# (uses the SAME rendering method as $RENDER below):
#
#   for d in ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic exchange_rate; do
#     python utils/precompute_embeddings_pervar.py --dataset $d --method RP
#   done
#
# The teacher loads embeddings from:
#   {root_path}/dino_embeddings_{data}_{RENDER}_pervar
# which precompute_embeddings_pervar.py produces with matching --dataset/--method.
# ═══════════════════════════════════════════════════════════════════════════

# GPUs to use: each experiment runs on ONE gpu; different experiments run in
# parallel across these gpus (one experiment per gpu at a time). Override via env:
#   GPUS="0 1 2 3" bash scripts/long_term_forecast/JEPAVTS_iTransformer_all_datasets.sh
GPUS="${GPUS:-0 1 2 3}"

# Per-job logs + resume markers (a finished job is skipped on re-run).
LOG_DIR="${LOG_DIR:-logs/jepavts_itransformer}"

MODEL=JEPAVTS
STUDENT=iTransformer
RENDER=RP            # rendering method (must be precomputed); e.g. RP / GAF / LinePlot
JEPA_WEIGHT=1
JEPA_LOSS=mse     # JEPA alignment loss: mse | cosine

# Per-dataset config (portable: no associative arrays, works on bash 3.2+).
# Hyper-params follow the official iTransformer scripts:
#   e_layers / d_model / d_ff / factor / batch_size / lr from the per-dataset
#   baseline scripts; epochs/patience/dropout use iTransformer defaults.
# Field order:
#   data|root_path|data_path|enc_in|e_layers|d_model|d_ff|factor|batch_size|lr|epochs|patience|dropout|seq_len|horizons
get_cfg() {
  case "$1" in
    ETTh1)            echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|128|128|3|32|0.0001|10|3|0.1|96|96 192 336 720" ;;
    ETTh2)            echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|128|128|3|32|0.0001|10|3|0.1|96|96 192 336 720" ;;
    ETTm1)            echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7|2|128|128|3|32|0.0001|10|3|0.1|96|96 192 336 720" ;;
    ETTm2)            echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7|2|128|128|3|32|0.0001|10|3|0.1|96|96 192 336 720" ;;
    weather)          echo "custom|./dataset/weather/|weather.csv|21|3|512|512|3|32|0.0001|10|3|0.1|96|96 192 336 720" ;;
    electricity)      echo "custom|./dataset/electricity/|electricity.csv|321|3|512|512|3|16|0.0005|10|3|0.1|96|96 192 336 720" ;;
    traffic)          echo "custom|./dataset/traffic/|traffic.csv|862|4|512|512|3|16|0.001|10|3|0.1|96|96 192 336 720" ;;
    exchange_rate)    echo "custom|./dataset/exchange_rate/|exchange_rate.csv|8|2|128|128|3|32|0.0001|10|3|0.1|96|96 192 336 720" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

# Order of datasets to run
DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic exchange_rate"

run_one() {
  # args: dataset_name pred_len enc_setting(single|dual) gpu_id
  local name=$1 pred_len=$2 enc=$3 gpu=$4
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff factor batch_size lr epochs patience dropout seq_len horizons <<< "$(get_cfg "$name")"

  local extra=""
  local tag="single"
  if [ "$enc" = "dual" ]; then
    extra="--use_dual_encoder --fusion_type mlp"
    tag="dual"
  fi

  local model_id="${name}_${RENDER}_${seq_len}_${pred_len}_${tag}"
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
    --factor "$factor" \
    --des Exp \
    --itr 1 \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --learning_rate "$lr" \
    --train_epochs "$epochs" \
    --patience "$patience" \
    --batch_size "$batch_size" \
    --dropout "$dropout" \
    --per_var_teacher \
    --rendering_methods $RENDER \
    --jepa_weight $JEPA_WEIGHT \
    --jepa_loss_type $JEPA_LOSS \
    $extra \
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

# Worker: process a queue of jobs sequentially on a single gpu.
worker() {
  local gpu=$1; shift
  for job in "$@"; do
    IFS=':' read -r name pred_len enc <<< "$job"
    run_one "$name" "$pred_len" "$enc" "$gpu"
  done
}

mkdir -p "$LOG_DIR"

# Build the full job list (each entry: name:pred_len:enc).
job_list=""
for name in $DATASETS; do
  cfg="$(get_cfg "$name")" || exit 1
  horizons="$(printf '%s' "$cfg" | awk -F'|' '{print $NF}')"
  for pred_len in $horizons; do
    job_list="$job_list ${name}:${pred_len}:single ${name}:${pred_len}:dual"
  done
done

gpu_arr=($GPUS)
job_arr=($job_list)
ngpu=${#gpu_arr[@]}
echo "Dispatching ${#job_arr[@]} experiments across $ngpu gpu(s): $GPUS"
echo "Logs: $LOG_DIR/<model_id>.log"

# Distribute jobs round-robin so heavy datasets are spread across gpus,
# then launch one background worker per gpu (gpus run in parallel).
for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"
  sel=""
  j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do
    sel="$sel ${job_arr[$j]}"
    j=$((j + ngpu))
  done
  nsel=$(printf '%s\n' $sel | grep -c . || true)
  echo "  gpu $gpu  <-  $nsel experiments"
  worker "$gpu" $sel &
done

wait
echo "All gpus finished. Collect results with: python utils/collect_results.py"
