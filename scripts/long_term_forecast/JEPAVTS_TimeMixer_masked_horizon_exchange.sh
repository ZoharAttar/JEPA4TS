#!/bin/bash
# VibeTS + TimeMixer + masked-horizon teacher, Exchange pilot (seq_len=96, pred_len=96).
# Prerequisite: DINO per-variable RP embeddings for exchange_rate (same as the
# main TimeMixer script). MAE-base weights download to ./ckpt/ on first run.
#
#   python utils/precompute_embeddings_pervar.py --dataset exchange_rate --method RP
#   bash scripts/long_term_forecast/JEPAVTS_TimeMixer_masked_horizon_exchange.sh
#
# Override: GPUS, HORIZON_WEIGHT, PRED_LEN, LOG_DIR
set -u

GPUS="${GPUS:-0}"
LOG_DIR="${LOG_DIR:-logs/jepavts_timemixer_masked_horizon}"
PRED_LEN="${PRED_LEN:-96}"
HORIZON_WEIGHT="${HORIZON_WEIGHT:-1}"

MODEL=JEPAVTS
STUDENT=TimeMixer
RENDER=RP
JEPA_WEIGHT=1
JEPA_SCALE=coarse
JEPA_LOSS=mse
DS_WINDOW=2

mkdir -p "$LOG_DIR"

# exchange_rate row from JEPAVTS_TimeMixer_all_datasets.sh
root_path=./dataset/exchange_rate/
data_path=exchange_rate.csv
data=custom
enc_in=8
e_layers=2
d_model=16
d_ff=32
batch_size=32
ds_layers=3
lr=0.01
epochs=10
patience=3
dropout=0.1
seq_len=96

gpu="${GPUS%% *}"
model_id="exchange_rate_${RENDER}_${seq_len}_${PRED_LEN}_single_masked_horizon"
logf="$LOG_DIR/${model_id}.log"
donef="$LOG_DIR/${model_id}.done"

if [ -f "$donef" ]; then
  echo "SKIP (already done): $model_id"
  exit 0
fi

echo "START: $model_id  (gpu $gpu)"
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
  --pred_len "$PRED_LEN" \
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
rc=$?
if [ $rc -eq 0 ]; then
  touch "$donef"
  echo "DONE:  $model_id"
else
  echo "FAIL (rc=$rc): $model_id  ->  see $logf"
fi
exit $rc
