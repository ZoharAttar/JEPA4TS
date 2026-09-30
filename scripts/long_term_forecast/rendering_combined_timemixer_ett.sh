#!/bin/bash
# Ensemble (combined) forecasting: TimeMixer student, ETTh1 and ETTh2.
#
# Same ensemble as scripts/ablation/rendering_ablation.sh "combined":
#   one shared encoder, one shared predictor,
#   four teachers RP GAF LinePlot Spectrogram, losses summed.
#
# Hyper-params are the ETTh1 / ETTh2 rows of
# scripts/long_term_forecast/JEPAVTS_TimeMixer_all_datasets.sh.
#
# Prerequisite, per dataset, all four folders under ./dataset/ETT-small/:
#   dino_embeddings_ETTh1_RP_pervar
#   dino_embeddings_ETTh1_GAF_pervar
#   dino_embeddings_ETTh1_LinePlot_pervar
#   dino_embeddings_ETTh1_Spectrogram_pervar
#   and the same four names with ETTh2.
#
# Usage:
#   bash scripts/ablation/rendering_combined_timemixer_ett.sh
#   DATASETS="ETTh1" PRED_LENS="96" bash scripts/ablation/rendering_combined_timemixer_ett.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/rendering_combined_timemixer_ett.sh

set -u

DATASETS="${DATASETS:-ETTh1 ETTh2}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
LOG_DIR="${LOG_DIR:-logs/rendering_combined_timemixer_ett}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$LOG_DIR"

# data|root|data_path|enc_in|e_layers|d_model|d_ff|batch|ds_layers|lr|epochs|patience|dropout|seq_len
get_cfg() {
  case "$1" in
    ETTh1) echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|16|32|128|3|0.01|10|3|0.6|96" ;;
    ETTh2) echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|16|32|128|3|0.01|10|3|0.6|96" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

for name in $DATASETS
do
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff batch_size ds_layers lr epochs patience dropout seq_len <<< "$(get_cfg "$name")"
  for pred_len in $PRED_LENS
  do
    model_id="${name}_combined_${seq_len}_${pred_len}"
    logf="$LOG_DIR/${model_id}.log"
    donef="$LOG_DIR/${model_id}.done"
    if [ -f "$donef" ]; then
      echo "SKIP (already done): $model_id"
      continue
    fi
    echo "START: $model_id"
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model JEPAVTS \
      --student_model TimeMixer \
      --data "$data" \
      --features M \
      --seq_len "$seq_len" \
      --label_len 0 \
      --pred_len "$pred_len" \
      --e_layers "$e_layers" \
      --enc_in "$enc_in" \
      --c_out "$enc_in" \
      --d_model "$d_model" \
      --d_ff "$d_ff" \
      --batch_size "$batch_size" \
      --learning_rate "$lr" \
      --train_epochs "$epochs" \
      --patience "$patience" \
      --dropout "$dropout" \
      --down_sampling_layers "$ds_layers" \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --timemixer_jepa_scale coarse \
      --per_var_teacher \
      --rendering_methods RP GAF LinePlot Spectrogram \
      --multi_rendering_alpha_mode same \
      --jepa_weight 1 \
      --jepa_loss_type mse \
      --des Exp \
      --itr 1 \
      > "$logf" 2>&1
    if [ $? -eq 0 ]; then
      touch "$donef"
      echo "DONE:  $model_id"
    else
      echo "FAIL:  $model_id  ->  $logf"
    fi
  done
done

echo ""
echo "Combined TimeMixer  (MSE / MAE)"
for name in $DATASETS
do
  echo "$name"
  for pred_len in $PRED_LENS
  do
    logf="$LOG_DIR/${name}_combined_96_${pred_len}.log"
    if [ ! -f "$logf" ]; then
      echo "  pl${pred_len}  NA"
      continue
    fi
    mse=$(grep -E "MSE:" "$logf" | tail -1 | sed -E 's/.*MSE:[[:space:]]*([0-9.eE+-]+).*/\1/')
    mae=$(grep -E "MAE:" "$logf" | tail -1 | sed -E 's/.*MAE:[[:space:]]*([0-9.eE+-]+).*/\1/')
    echo "  pl${pred_len}  ${mse:-NA} / ${mae:-NA}"
  done
done
