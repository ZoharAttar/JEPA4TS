#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# VibeTS (JEPAVTS) + iTransformer student, SINGLE ENCODER, anomaly detection.
# Reconstruction-based: reconstruct each window; per-timestep reconstruction
# error is the anomaly score. During training the per-variable student encoding
# is aligned to the frozen per-variable DINO teacher of the same window.
#
# Hyper-params mirror the official iTransformer anomaly scripts
# (scripts/anomaly_detection/<DATA>/iTransformer.sh): seq_len 100, d_model 128,
# d_ff 128, e_layers 3, batch 128, epochs 10. enc_in / anomaly_ratio are per
# dataset. The baseline (plain iTransformer) uses the SAME config.
#
# PREREQUISITE — precompute per-variable DINO embeddings for the TRAIN split of
# each dataset, using the SAME rendering ($RENDER) and the loader's default
# stride (the precompute script picks the correct default step per dataset):
#
#   for d in PSM MSL SMAP SMD SWAT; do
#     python utils/precompute_embeddings_pervar_anomaly.py --dataset $d --method RP
#   done
#
# The teacher loads from {root_path}/dino_embeddings_{DATA}_{RENDER}_pervar.
# ═══════════════════════════════════════════════════════════════════════════
GPUS="${GPUS:-0 1 2 3}"
LOG_DIR="${LOG_DIR:-logs/jepavts_itransformer_anomaly}"

MODEL=JEPAVTS
STUDENT=iTransformer
RENDER="${RENDER:-RP}"      # rendering method (must be precomputed)
JEPA_WEIGHT="${JEPA_WEIGHT:-1}"
JEPA_LOSS=mse
SEQ=100
D_MODEL=128
D_FF=128
E_LAYERS=3
BATCH=128
EPOCHS="${EPOCHS:-10}"

# Per-dataset config. Field order: root_path|data|enc_in|anomaly_ratio
get_cfg() {
  case "$1" in
    PSM)  echo "./dataset/PSM|PSM|25|1" ;;
    MSL)  echo "./dataset/MSL|MSL|55|1" ;;
    SMAP) echo "./dataset/SMAP|SMAP|25|1" ;;
    SMD)  echo "./dataset/SMD|SMD|38|0.5" ;;
    SWAT) echo "./dataset/SWaT|SWAT|51|1" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

DATASETS="${DATASETS:-PSM MSL SMAP SMD SWAT}"

run_one() {
  local name=$1 gpu=$2
  IFS='|' read -r root_path data enc_in anomaly_ratio <<< "$(get_cfg "$name")" || return 1

  local model_id="${name}_${RENDER}_single"
  local logf="$LOG_DIR/${model_id}.log"
  local donef="$LOG_DIR/${model_id}.done"
  if [ -f "$donef" ]; then
    echo "[gpu $gpu] SKIP (done): $model_id"; return 0
  fi

  echo "[gpu $gpu] START: $model_id"
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path "$root_path" \
    --model_id "$model_id" \
    --model $MODEL \
    --student_model $STUDENT \
    --data "$data" \
    --features M \
    --seq_len $SEQ \
    --pred_len 0 \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --e_layers $E_LAYERS \
    --enc_in "$enc_in" \
    --c_out "$enc_in" \
    --anomaly_ratio "$anomaly_ratio" \
    --batch_size $BATCH \
    --train_epochs $EPOCHS \
    --des Exp --itr 1 \
    --per_var_teacher \
    --rendering_methods $RENDER \
    --jepa_weight $JEPA_WEIGHT \
    --jepa_loss_type $JEPA_LOSS \
    > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$donef"; echo "[gpu $gpu] DONE:  $model_id"
  else echo "[gpu $gpu] FAIL (rc=$rc): $model_id -> $logf"; fi
  return $rc
}

worker() { local gpu=$1; shift; for j in "$@"; do run_one "$j" "$gpu"; done; }

mkdir -p "$LOG_DIR"
gpu_arr=($GPUS); job_arr=($DATASETS); ngpu=${#gpu_arr[@]}
echo "Dispatching ${#job_arr[@]} datasets across $ngpu gpu(s): $GPUS"
for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"; sel=""; j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do sel="$sel ${job_arr[$j]}"; j=$((j + ngpu)); done
  worker "$gpu" $sel &
done
wait
echo "All gpus finished. Results in result_anomaly_detection.txt"
