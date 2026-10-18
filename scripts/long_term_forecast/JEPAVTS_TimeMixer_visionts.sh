#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# JEPAVTS + TimeMixer student, VisionTS++ teacher (instead of DINO).
# Same configs as JEPAVTS_TimeMixer_all_datasets.sh — ONLY the teacher changes
# (RENDER=VisionTSpp). SINGLE-encoder only (one job per dataset+horizon).
#
# PREREQUISITE — precompute the VisionTS++ per-variable teacher embeddings first:
#
#   pip install git+https://github.com/Keytoyze/VisionTS.git   # once
#   python utils/precompute_embeddings_visionts.py --dataset ETTh1
#   # (one run per dataset covers all horizons; see the script's header)
#
# The teacher loads embeddings from:
#   {root_path}/visionts_embeddings_{data}_VisionTSpp_pervar
# (produced by precompute_embeddings_visionts.py). teacher_dim = 768 (mae_base),
# so nothing else in the pipeline changes vs the DINO runs.
# ═══════════════════════════════════════════════════════════════════════════

# Multi-GPU: each experiment runs on ONE gpu; different experiments run in
# parallel across these gpus (one experiment per gpu at a time). Override via env:
#   GPUS="0 1 2 3" bash scripts/long_term_forecast/JEPAVTS_TimeMixer_visionts.sh
GPUS="${GPUS:-0 1 2 3}"

# Datasets to run (override for a quick single-dataset test):
#   DATASETS="ETTh1" bash scripts/long_term_forecast/JEPAVTS_TimeMixer_visionts.sh
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic exchange_rate}"

# Per-job logs + resume markers (a finished job is skipped on re-run).
LOG_DIR="${LOG_DIR:-logs/jepavts_timemixer_visionts}"

MODEL=JEPAVTS
STUDENT=TimeMixer
RENDER=VisionTSpp    # ← teacher = VisionTS++ (was RP for DINO). Must be precomputed.
JEPA_WEIGHT=1
JEPA_SCALE=coarse    # timemixer_jepa_scale: fine | coarse
JEPA_LOSS=mse        # JEPA alignment loss: mse | cosine

# Shared training hyper-params
DS_WINDOW=2

# Per-dataset config (identical to JEPAVTS_TimeMixer_all_datasets.sh).
# Field order:
#   data|root_path|data_path|enc_in|e_layers|d_model|d_ff|batch_size|ds_layers|lr|epochs|patience|dropout|seq_len|horizons
get_cfg() {
  case "$1" in
    ETTh1)            echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|16|32|128|3|0.01|10|3|0.6|96|96 192 336 720" ;;
    ETTh2)            echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|16|32|128|3|0.01|10|3|0.6|96|96 192 336 720" ;;
    ETTm1)            echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7|2|16|32|128|3|0.01|10|3|0.1|96|96 192 336 720" ;;
    ETTm2)            echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7|2|32|64|128|3|0.01|10|3|0.1|96|96 192 336 720" ;;
    weather)          echo "custom|./dataset/weather/|weather.csv|21|2|16|32|128|3|0.01|20|10|0.1|96|96 192 336 720" ;;
    electricity)      echo "custom|./dataset/electricity/|electricity.csv|321|2|16|32|32|3|0.01|20|10|0.1|96|96 192 336 720" ;;
    traffic)          echo "custom|./dataset/traffic/|traffic.csv|862|2|32|64|8|3|0.01|20|3|0.1|96|96 192 336 720" ;;
    exchange_rate)    echo "custom|./dataset/exchange_rate/|exchange_rate.csv|8|2|16|32|32|3|0.01|10|3|0.1|96|96 192 336 720" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

run_one() {
  # args: dataset_name pred_len enc_setting(single|dual) gpu_id
  local name=$1 pred_len=$2 enc=$3 gpu=$4
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff batch_size ds_layers lr epochs patience dropout seq_len horizons <<< "$(get_cfg "$name")"

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
    job_list="$job_list ${name}:${pred_len}:single"
  done
done

gpu_arr=($GPUS)
job_arr=($job_list)
ngpu=${#gpu_arr[@]}
echo "Dispatching ${#job_arr[@]} experiments across $ngpu gpu(s): $GPUS"
echo "Logs: $LOG_DIR/<model_id>.log"

# Distribute jobs round-robin, then launch one background worker per gpu.
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
