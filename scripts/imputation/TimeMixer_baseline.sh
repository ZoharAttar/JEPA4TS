#!/bin/bash
# Plain TimeMixer imputation. No teacher, no JEPA loss.
#
# seq_len 96 and horizons pred_len 96 192 336 720, matching
# scripts/imputation/JEPAVTS_TimeMixer_ETTh1.sh and the existing DINO caches.
# Mask ratios 12.5%, 25%, 37.5%, 50%. After the four runs of one
# (dataset, horizon), the script prints the mean MSE and MAE.
# The head fills the 96-step input.
#
# Backbone settings follow scripts/long_term_forecast/JEPAVTS_TimeMixer_all_datasets.sh
# so this baseline matches the JEPA student. ETTh1 uses that file's dropout 0.6
# and patience 3.
#
# Usage:
#   bash scripts/imputation/TimeMixer_baseline.sh
#   DATASETS="ETTh1" HORIZONS="96" MASK_RATES="0.25" bash scripts/imputation/TimeMixer_baseline.sh

set -u

DATASETS="${DATASETS:-ETTh1 weather}"
MASK_RATES="${MASK_RATES:-0.125 0.25 0.375 0.5}"
HORIZONS="${HORIZONS:-96 192 336 720}"
SEQ_LEN=96
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# data|root|data_path|enc_in|e_layers|d_model|d_ff|batch|ds_layers|lr|epochs|patience|dropout
get_cfg() {
  case "$1" in
    ETTh1)   echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|16|32|128|3|0.01|10|3|0.6" ;;
    weather) echo "custom|./dataset/weather/|weather.csv|21|2|16|32|128|3|0.01|20|10|0.1" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

metrics_path() {
  local model_id=$1 data=$2 dm=$3 el=$4 df=$5 pred_len=$6
  echo "./results/imputation_${model_id}_TimeMixer_${data}_ftM_sl${SEQ_LEN}_ll0_pl${pred_len}_dm${dm}_nh8_el${el}_dl1_df${df}_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/metrics.npy"
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

for name in $DATASETS
do
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff batch_size ds_layers lr epochs patience dropout <<< "$(get_cfg "$name")"
  for pred_len in $HORIZONS
  do
  paths=""
  for mask_rate in $MASK_RATES
  do
    model_id="${name}_TimeMixer_${pred_len}_mask_${mask_rate}"
    python -u run.py \
      --task_name imputation \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --mask_rate $mask_rate \
      --model TimeMixer \
      --data "$data" \
      --features M \
      --seq_len $SEQ_LEN \
      --label_len 0 \
      --pred_len "$pred_len" \
      --e_layers "$e_layers" \
      --enc_in "$enc_in" \
      --dec_in "$enc_in" \
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
      --channel_independence 1 \
      --des Exp \
      --itr 1
    paths="$paths $(metrics_path "$model_id" "$data" "$d_model" "$e_layers" "$d_ff" "$pred_len")"
  done
  average_masks "${name}_TimeMixer_${pred_len}" $paths
  done
done
