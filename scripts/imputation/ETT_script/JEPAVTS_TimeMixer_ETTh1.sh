#!/bin/bash
# JEPAVTS + TimeMixer, ETTh1 imputation.
#
# Uses the forecasting DINO caches (seq_len 96). No new precompute.
# Horizons are pred_len 96 192 336 720. The teacher hashes the 96-step input,
# so every horizon hits the same cache. A larger pred_len only drops windows
# at the end of each split.
#
# Mask ratios 12.5%, 25%, 37.5%, 50%. After the four runs of one
# (rendering, horizon), the script prints the mean MSE and MAE.
#
# Length 1024 would be a new cache: the hash is the whole input window, and
# 1024 is not the 96-step lookback already stored.
#
# Usage:
#   bash scripts/imputation/JEPAVTS_TimeMixer_ETTh1.sh
#   HORIZONS="96" VARIANTS="RP" bash scripts/imputation/JEPAVTS_TimeMixer_ETTh1.sh

set -u

VARIANTS="${VARIANTS:-RP GAF LinePlot Spectrogram combined}"
COMBINED_METHODS="${COMBINED_METHODS:-RP GAF LinePlot Spectrogram}"
MASK_RATES="${MASK_RATES:-0.125 0.25 0.375 0.5}"
HORIZONS="${HORIZONS:-96 192 336 720}"
SEQ_LEN=96
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

jepa_render_flags() {
  case "$1" in
    combined) echo "--rendering_methods $COMBINED_METHODS --multi_rendering_alpha_mode same" ;;
    *)        echo "--rendering_methods $1" ;;
  esac
}

# metrics.npy is [mae, mse, rmse, mape, mspe]
metrics_path() {
  local model_id=$1 pred_len=$2
  echo "./results/imputation_${model_id}_JEPAVTS_ETTh1_ftM_sl${SEQ_LEN}_ll0_pl${pred_len}_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_TimeMixer/metrics.npy"
}

average_masks() {
  local tag=$1
  shift
  python3 - "$tag" "$@" << 'PY'
import sys
import numpy as np

tag = sys.argv[1]
paths = sys.argv[2:]
mses, maes, missing = [], [], []
for p in paths:
    try:
        m = np.load(p)
    except FileNotFoundError:
        missing.append(p)
        continue
    maes.append(float(m[0]))
    mses.append(float(m[1]))
if missing:
    print(f"AVG {tag}: missing {len(missing)} run(s), not averaged")
    for p in missing:
        print(f"  {p}")
    sys.exit(0)
mse = float(np.mean(mses))
mae = float(np.mean(maes))
line = f"AVG {tag}  mse:{mse}, mae:{mae}  (mean of {len(mses)} mask rates)"
print(line)
with open("result_imputation.txt", "a") as f:
    f.write(line + "\n\n")
PY
}

for variant in $VARIANTS
do
  for pred_len in $HORIZONS
  do
  paths=""
  for mask_rate in $MASK_RATES
  do
    model_id="ETTh1_${variant}_${pred_len}_mask_${mask_rate}"
    python -u run.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id "$model_id" \
      --mask_rate $mask_rate \
      --model JEPAVTS \
      --student_model TimeMixer \
      --data ETTh1 \
      --features M \
      --seq_len $SEQ_LEN \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 16 \
      --d_ff 32 \
      --batch_size 128 \
      --learning_rate 0.01 \
      --train_epochs 10 \
      --patience 3 \
      --dropout 0.6 \
      --down_sampling_layers 3 \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --timemixer_jepa_scale coarse \
      --per_var_teacher \
      $(jepa_render_flags "$variant") \
      --jepa_weight 1 \
      --jepa_loss_type mse \
      --des Exp \
      --itr 1
    paths="$paths $(metrics_path "$model_id" "$pred_len")"
  done
  average_masks "ETTh1_${variant}_${pred_len}" $paths
  done
done
