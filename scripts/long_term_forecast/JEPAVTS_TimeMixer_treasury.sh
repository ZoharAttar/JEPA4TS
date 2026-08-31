#!/bin/bash
# VibeTS + TimeMixer on the FRED Treasury curve (seq_len=96).
# Same hyperparameters as Exchange (enc_in=8). Default is DINO-only (no MAE).
#
#   python utils/download_fred_treasury.py
#   python utils/precompute_embeddings_pervar.py --dataset treasury_yields --method RP
#   VARIANT=baseline bash scripts/long_term_forecast/JEPAVTS_TimeMixer_treasury.sh   # TimeMixer only
#   bash scripts/long_term_forecast/JEPAVTS_TimeMixer_treasury.sh                     # VibeTS (DINO)
#   for L in 192 336 720; do
#     PRED_LEN=$L VARIANT=baseline bash scripts/long_term_forecast/JEPAVTS_TimeMixer_treasury.sh
#     PRED_LEN=$L bash scripts/long_term_forecast/JEPAVTS_TimeMixer_treasury.sh
#   done
#
# MAE (optional, already weaker on Exchange):
#   MASKED=1 bash scripts/long_term_forecast/JEPAVTS_TimeMixer_treasury.sh
#
# Override: GPUS, PRED_LEN, VARIANT, MASKED, HORIZON_WEIGHT, LOG_DIR
set -u

GPUS="${GPUS:-0}"
PRED_LEN="${PRED_LEN:-96}"
VARIANT="${VARIANT:-vibe}"
MASKED="${MASKED:-0}"
HORIZON_WEIGHT="${HORIZON_WEIGHT:-1}"
LOG_DIR="${LOG_DIR:-logs/jepavts_timemixer_treasury}"

MODEL=JEPAVTS
STUDENT=TimeMixer
RENDER=RP
JEPA_WEIGHT=1
JEPA_SCALE=coarse
JEPA_LOSS=mse
DS_WINDOW=2

mkdir -p "$LOG_DIR"

root_path=./dataset/treasury_yields/
data_path=treasury_yields.csv
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

if [ "$VARIANT" = "baseline" ]; then
  tag="TimeMixer"
  MODEL=TimeMixer
elif [ "$MASKED" = "1" ]; then
  tag="single_masked_horizon"
else
  tag="single"
fi

gpu="${GPUS%% *}"
if [ "$VARIANT" = "baseline" ]; then
  model_id="treasury_yields_${seq_len}_${PRED_LEN}_${tag}"
else
  model_id="treasury_yields_${RENDER}_${seq_len}_${PRED_LEN}_${tag}"
fi
logf="$LOG_DIR/${model_id}.log"
donef="$LOG_DIR/${model_id}.done"

if [ -f "$donef" ]; then
  echo "SKIP (already done): $model_id"
  exit 0
fi

extra=()
if [ "$VARIANT" = "baseline" ]; then
  extra+=(--down_sampling_layers "$ds_layers" --down_sampling_method avg --down_sampling_window "$DS_WINDOW")
elif [ "$MASKED" = "1" ]; then
  extra+=(--masked_horizon --horizon_weight "$HORIZON_WEIGHT")
fi

echo "START: $model_id  (gpu $gpu)"
if [ "$VARIANT" = "baseline" ]; then
CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "$model_id" \
  --model TimeMixer \
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
  --freq d \
  "${extra[@]}" \
  > "$logf" 2>&1
else
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
  --freq d \
  "${extra[@]}" \
  > "$logf" 2>&1
fi
rc=$?
if [ $rc -eq 0 ]; then
  touch "$donef"
  echo "DONE:  $model_id"
else
  echo "FAIL (rc=$rc): $model_id  ->  see $logf"
fi
exit $rc
