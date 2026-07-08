#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# FEW-SHOT (in-distribution) forecasting — VibeTS Table-2 protocol:
#   train on PERCENT% of the target's OWN train split, evaluate on its FULL test
#   split, MSE/MAE averaged over horizons {96,192,336,720}.
#
# Produces the two columns you need for the table (others come from the paper):
#   - "VibeTS (Ours)"  -> JEPAVTS + TimeMixer student   (single AND dual encoder)
#   - "TimeMixer"      -> plain TimeMixer baseline
#
# Few-shot is enabled by --percent (added to run.py); the data loader keeps only
# the first PERCENT% of the TRAIN windows (val/test unchanged). percent=100 = full.
#
# Datasets: ETTh1 ETTh2 ETTm1 ETTm2 weather. Per-dataset hyper-params are IDENTICAL
# to scripts/long_term_forecast/JEPAVTS_TimeMixer_all_datasets.sh.
#
# PREREQUISITE (JEPAVTS variants only): few-shot still TRAINS, so per-variable DINO
# embeddings for each target dataset must be precomputed (same RENDER as below):
#   for d in ETTh1 ETTh2 ETTm1 ETTm2 weather; do
#     python utils/precompute_embeddings_pervar.py --dataset $d --method RP
#   done
#   (The plain-TimeMixer baseline variant needs NO embeddings.)
#
# Results:  JEPAVTS -> result_jepa_vts.txt ; TimeMixer -> result_long_term_forecast.txt
# Rows are tagged fs<PERCENT>_..._s<seed>. Average the 4 horizons per dataset.
#
# PARALLELISM: GPUS x JOBS_PER_GPU slots (few-shot trains on little data -> fast,
# so stack several per gpu). Finished jobs are skipped via .done markers.
#
# Usage:
#   GPUS="0 1 2 3 4 5 6 7" JOBS_PER_GPU=2 PERCENT=10 SEEDS="2021" \
#     bash scripts/few_shot/ett_weather_fewshot.sh
# ═══════════════════════════════════════════════════════════════════════════

GPU="${GPU:-0}"
GPUS="${GPUS:-$GPU}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

PERCENT="${PERCENT:-10}"              # few-shot fraction of TRAIN split
SEEDS="${SEEDS:-2021}"               # e.g. "2021 2022 2023" for multi-seed
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather}"
# Which variants to run: jepa_single jepa_dual timemixer
VARIANTS="${VARIANTS:-jepa_single jepa_dual timemixer}"

STUDENT="${STUDENT:-TimeMixer}"      # JEPAVTS student backbone: TimeMixer | iTransformer
RENDER="${RENDER:-RP}"               # JEPAVTS rendering (must be precomputed)
JEPA_WEIGHT="${JEPA_WEIGHT:-1}"
JEPA_SCALE="${JEPA_SCALE:-coarse}"
JEPA_LOSS="${JEPA_LOSS:-mse}"
DS_WINDOW="${DS_WINDOW:-2}"

LOG_DIR="${LOG_DIR:-logs/few_shot}"
mkdir -p "$LOG_DIR"

# Per-dataset config, student-aware:
#   TimeMixer    == JEPAVTS_TimeMixer_all_datasets.sh
#   iTransformer == JEPAVTS_iTransformer_all_datasets.sh (forecasting)
# Unified field order (ds_layers used by TimeMixer, factor used by iTransformer):
#   data|root_path|data_path|enc_in|e_layers|d_model|d_ff|batch|ds_layers|factor|lr|epochs|patience|dropout|seq_len|horizons
get_cfg() {
  case "$STUDENT" in
    TimeMixer)
      case "$1" in
        ETTh1)   echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|16|32|128|3|1|0.01|10|3|0.6|96|96 192 336 720" ;;
        ETTh2)   echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|16|32|128|3|1|0.01|10|3|0.6|96|96 192 336 720" ;;
        ETTm1)   echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7|2|16|32|128|3|1|0.01|10|3|0.1|96|96 192 336 720" ;;
        ETTm2)   echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7|2|32|64|128|3|1|0.01|10|3|0.1|96|96 192 336 720" ;;
        weather) echo "custom|./dataset/weather/|weather.csv|21|2|16|32|128|3|1|0.01|20|10|0.1|96|96 192 336 720" ;;
        *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
      esac ;;
    iTransformer)
      case "$1" in
        ETTh1)   echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|128|128|32|0|3|0.0001|10|3|0.1|96|96 192 336 720" ;;
        ETTh2)   echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|128|128|32|0|3|0.0001|10|3|0.1|96|96 192 336 720" ;;
        ETTm1)   echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7|2|128|128|32|0|3|0.0001|10|3|0.1|96|96 192 336 720" ;;
        ETTm2)   echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7|2|128|128|32|0|3|0.0001|10|3|0.1|96|96 192 336 720" ;;
        weather) echo "custom|./dataset/weather/|weather.csv|21|3|512|512|32|0|3|0.0001|10|3|0.1|96|96 192 336 720" ;;
        *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
      esac ;;
    *) echo "UNKNOWN student: $STUDENT (use TimeMixer|iTransformer)" >&2; return 1 ;;
  esac
}

# Model-family flags per variant. $1=variant $2=ds_layers $3=factor
model_flags() {
  local variant=$1 ds_layers=$2 factor=$3
  # student-specific backbone flags: TimeMixer needs multi-scale downsampling;
  # iTransformer needs its attention factor.
  local sflags=""
  if [ "$STUDENT" = "TimeMixer" ]; then
    sflags="--down_sampling_layers $ds_layers --down_sampling_method avg --down_sampling_window $DS_WINDOW"
  elif [ "$STUDENT" = "iTransformer" ]; then
    sflags="--factor $factor"
  fi
  local jflags="--per_var_teacher --rendering_methods $RENDER --jepa_weight $JEPA_WEIGHT --jepa_loss_type $JEPA_LOSS"
  [ "$STUDENT" = "TimeMixer" ] && jflags="$jflags --timemixer_jepa_scale $JEPA_SCALE"
  case "$variant" in
    jepa_single)
      echo "--model JEPAVTS --student_model $STUDENT $sflags $jflags" ;;
    jepa_dual)
      echo "--model JEPAVTS --student_model $STUDENT $sflags $jflags --use_dual_encoder --fusion_type mlp" ;;
    timemixer|baseline)
      echo "--model $STUDENT $sflags" ;;
    *) echo "UNKNOWN variant: $variant" >&2; return 1 ;;
  esac
}

model_id() {
  local variant=$1 name=$2 pred=$3 seed=$4 seq=$5
  # student tag (non-default only) so TimeMixer ids stay backward-compatible while
  # iTransformer gets its own .done markers / result rows (no collision).
  local stag=""; [ "$STUDENT" != "TimeMixer" ] && stag="${STUDENT}_"
  case "$variant" in
    jepa_single) echo "fs${PERCENT}_${stag}${name}_${RENDER}_${seq}_${pred}_single_s${seed}" ;;
    jepa_dual)   echo "fs${PERCENT}_${stag}${name}_${RENDER}_${seq}_${pred}_dual_s${seed}" ;;
    timemixer|baseline)   echo "fs${PERCENT}_${STUDENT}_${name}_${seq}_${pred}_s${seed}" ;;
  esac
}

# One job: train on PERCENT% then test (run.py is_training=1 trains + tests).
run_one() {
  local name=$1 pred=$2 variant=$3 seed=$4 gpu=$5
  local data root_path data_path enc_in e_layers d_model d_ff batch ds_layers factor lr epochs patience dropout seq horizons
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff batch ds_layers factor lr epochs patience dropout seq horizons <<< "$(get_cfg "$name")" || return 1

  local mid; mid="$(model_id "$variant" "$name" "$pred" "$seed" "$seq")"
  local logf="$LOG_DIR/${mid}.log"
  local donef="$LOG_DIR/${mid}.done"
  if [ "$FORCE" != "1" ] && [ -f "$donef" ]; then
    echo "[gpu $gpu] SKIP (done): $mid"
    return 0
  fi

  echo "[gpu $gpu] START: $mid  (percent=$PERCENT)"
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --seed "$seed" \
    --percent "$PERCENT" \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$mid" \
    --data "$data" \
    --features M \
    --seq_len "$seq" \
    --label_len 0 \
    --pred_len "$pred" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$enc_in" \
    --c_out "$enc_in" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --batch_size "$batch" \
    --learning_rate "$lr" \
    --train_epochs "$epochs" \
    --patience "$patience" \
    --dropout "$dropout" \
    --des Exp --itr 1 \
    $(model_flags "$variant" "$ds_layers" "$factor") \
    > "$logf" 2>&1
  if [ $? -eq 0 ]; then
    touch "$donef"; echo "[gpu $gpu] DONE:  $mid"
  else
    echo "[gpu $gpu] FAIL: $mid -> see $logf"
  fi
}

# Dispatcher: round-robin jobs over (#GPUS x JOBS_PER_GPU) slots.
slot_worker() {
  local gpu=$1; shift
  local job name pred variant seed
  for job in "$@"; do
    IFS='@' read -r name pred variant seed <<< "$job"
    run_one "$name" "$pred" "$variant" "$seed" "$gpu"
  done
}

dispatch() {
  local jobs=("$@"); local njobs=${#jobs[@]}
  [ "$njobs" -eq 0 ] && { echo ">>> no jobs"; return 0; }
  local gpu_arr=($GPUS); local ngpu=${#gpu_arr[@]}
  local nslots=$(( ngpu * JOBS_PER_GPU )); [ "$nslots" -lt 1 ] && nslots=1
  echo ">>> $njobs jobs across $nslots slots ($ngpu gpu(s) x $JOBS_PER_GPU/gpu): $GPUS"
  local s i
  for (( s=0; s<nslots; s++ )); do
    local gpu="${gpu_arr[$(( s % ngpu ))]}"
    local sel=(); i=$s
    while [ $i -lt $njobs ]; do sel+=("${jobs[$i]}"); i=$(( i + nslots )); done
    [ ${#sel[@]} -eq 0 ] && continue
    slot_worker "$gpu" "${sel[@]}" &
  done
  wait
}

echo ">>> few-shot percent=$PERCENT  seeds=$SEEDS  variants=$VARIANTS"
jobs=()
for seed in $SEEDS; do
  for name in $DATASETS; do
    cfg="$(get_cfg "$name")" || exit 1
    horizons="$(printf '%s' "$cfg" | awk -F'|' '{print $NF}')"
    for pred in $horizons; do
      for variant in $VARIANTS; do
        jobs+=("${name}@${pred}@${variant}@${seed}")
      done
    done
  done
done
dispatch "${jobs[@]}"

echo "Done. JEPAVTS rows -> result_jepa_vts.txt ; TimeMixer -> result_long_term_forecast.txt"
echo "Rows tagged fs${PERCENT}_..._s<seed>; average the 4 horizons per dataset."
