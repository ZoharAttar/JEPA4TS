#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Test-time INPUT-NOISE robustness on Exchange-rate forecasting.
#
#   Baseline TimeMixer   vs   JEPAVTS + TimeMixer student
#
# Additive Gaussian noise is injected into the TEST input window (x_enc) only:
#   noise std per variable = (noise% / 100) * that variable's std over the test
#   split. Ground-truth targets stay clean. Deterministic given --seed.
#
# Two phases:
#   1) Train (or reuse) the CLEAN checkpoints once. A checkpoint that already
#      exists on disk (e.g. the JEPAVTS runs you already did) is reused, so only
#      the missing plain-TimeMixer baseline is trained.
#   2) For every noise level, evaluate every horizon/variant on the noisy test
#      input, loading the CLEAN checkpoint via --transfer_checkpoint (no
#      retraining). Each noise level gets its own model_id so results never
#      collide with the clean numbers or with each other.
#
# Usage (defaults shown):
#   bash scripts/robustness/exchange_noise.sh
#   GPU=1 NOISE_LEVELS="0 5 10 20 30" bash scripts/robustness/exchange_noise.sh
#   VARIANTS="baseline jepa_single" bash scripts/robustness/exchange_noise.sh
#
# Env overrides:
#   GPU=0                              gpu id
#   NOISE_LEVELS="0 5 10 20"          percent noise sweep
#   PRED_LENS="96 192 336 720"        horizons
#   VARIANTS="baseline jepa_single jepa_dual"
#   RENDER=RP   SEED=2021
#
# PREREQUISITE for the JEPAVTS variants: per-variable DINO embeddings for
# exchange must already exist (they do if you trained the JEPAVTS runs):
#   python utils/precompute_embeddings_pervar.py --dataset exchange_rate --method RP
# (Only needed at TRAIN time; inference/noisy-eval does not touch the teacher.)
# ═══════════════════════════════════════════════════════════════════════════
set -u

GPU="${GPU:-0}"
NOISE_LEVELS="${NOISE_LEVELS:-0 5 10 20}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
VARIANTS="${VARIANTS:-baseline jepa_single jepa_dual nodino}"
RENDER="${RENDER:-RP}"
SEED="${SEED:-2021}"
LOG_DIR="${LOG_DIR:-logs/exchange_noise}"
CKPT_ROOT="./checkpoints"

# ── Exchange config (matches the exchange row of
#    scripts/long_term_forecast/JEPAVTS_TimeMixer_all_datasets.sh) ──────────
DATA=custom
ROOT=./dataset/exchange_rate/
DPATH=exchange_rate.csv
ENC_IN=8
E_LAYERS=2
D_MODEL=16
D_FF=32
BATCH=32
DS_LAYERS=3
DS_WINDOW=2
LR=0.01
EPOCHS=10
PATIENCE=3
DROPOUT=0.1
SEQ=96
JEPA_SCALE=coarse
JEPA_WEIGHT=1
JEPA_LOSS=mse

mkdir -p "$LOG_DIR"

# ── flag builders ─────────────────────────────────────────────────────────
common_flags() {
  echo "--task_name long_term_forecast \
    --root_path $ROOT --data_path $DPATH --data $DATA \
    --features M --seq_len $SEQ --label_len 0 \
    --e_layers $E_LAYERS --enc_in $ENC_IN --c_out $ENC_IN \
    --des Exp --itr 1 --d_model $D_MODEL --d_ff $D_FF \
    --learning_rate $LR --train_epochs $EPOCHS --patience $PATIENCE \
    --batch_size $BATCH --dropout $DROPOUT --seed $SEED"
}
tm_flags() {   # plain TimeMixer down-sampling
  echo "--down_sampling_layers $DS_LAYERS --down_sampling_method avg --down_sampling_window $DS_WINDOW"
}
jepa_flags() { # JEPAVTS + TimeMixer student
  echo "--model JEPAVTS --student_model TimeMixer \
    --down_sampling_layers $DS_LAYERS --down_sampling_method avg --down_sampling_window $DS_WINDOW \
    --timemixer_jepa_scale $JEPA_SCALE --per_var_teacher --rendering_methods $RENDER \
    --jepa_weight $JEPA_WEIGHT --jepa_loss_type $JEPA_LOSS"
}
nodino_flags() { # NO-DINO ablation: dual student encoders, task loss only.
                 # No teacher / no JEPA / no vision -> needs NO precomputed embeddings.
  echo "--model JEPAVTS --student_model TimeMixer --no_dino --fusion_type mlp \
    --down_sampling_layers $DS_LAYERS --down_sampling_method avg --down_sampling_window $DS_WINDOW \
    --timemixer_jepa_scale $JEPA_SCALE"
}

# Exact clean-checkpoint setting prefix per variant (matches run.py's builder).
# The trailing student suffix (_TimeMixer) is matched by a glob in find_ckpt,
# so this works whether or not the run.py student-suffix fix was in place when
# the checkpoint was trained.
setting_prefix() {
  local variant=$1 pred=$2
  local tail="_${DATA}_ftM_sl${SEQ}_ll0_pl${pred}_dm${D_MODEL}_nh8_el${E_LAYERS}_dl1_df${D_FF}_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0"
  case "$variant" in
    baseline)    echo "long_term_forecast_exchange_rate_${SEQ}_${pred}_TimeMixer${tail}" ;;
    jepa_single) echo "long_term_forecast_exchange_rate_${RENDER}_${SEQ}_${pred}_single_JEPAVTS${tail}" ;;
    jepa_dual)   echo "long_term_forecast_exchange_rate_${RENDER}_${SEQ}_${pred}_dual_JEPAVTS${tail}" ;;
    nodino)      echo "long_term_forecast_exchange_rate_nodino_${SEQ}_${pred}_JEPAVTS${tail}" ;;
    *) echo ""; return 1 ;;
  esac
}

# Find checkpoint.pth for a setting prefix (tolerates an optional _TimeMixer suffix).
find_ckpt() {
  local prefix=$1 d
  for d in "${CKPT_ROOT}/${prefix}"*/; do
    [ -f "${d}checkpoint.pth" ] && { echo "${d}checkpoint.pth"; return 0; }
  done
  return 1
}

variant_extra() {   # model + architecture flags for a variant
  case "$1" in
    baseline)    echo "--model TimeMixer $(tm_flags)" ;;
    jepa_single) echo "$(jepa_flags)" ;;
    jepa_dual)   echo "$(jepa_flags) --use_dual_encoder --fusion_type mlp" ;;
    nodino)      echo "$(nodino_flags)" ;;
  esac
}

clean_model_id() {  # model_id used when TRAINING the clean checkpoint
  local variant=$1 pred=$2
  case "$variant" in
    baseline)    echo "exchange_rate_${SEQ}_${pred}" ;;
    jepa_single) echo "exchange_rate_${RENDER}_${SEQ}_${pred}_single" ;;
    jepa_dual)   echo "exchange_rate_${RENDER}_${SEQ}_${pred}_dual" ;;
    nodino)      echo "exchange_rate_nodino_${SEQ}_${pred}" ;;
  esac
}

# ── Phase 1: train / reuse clean checkpoints ──────────────────────────────
echo "══════════════════════════════════════════════════════════════════"
echo " Phase 1: clean checkpoints (train only what is missing)"
echo "══════════════════════════════════════════════════════════════════"
for v in $VARIANTS; do
  for p in $PRED_LENS; do
    prefix="$(setting_prefix "$v" "$p")"
    if find_ckpt "$prefix" >/dev/null 2>&1; then
      echo "[train] SKIP (ckpt exists): $v pl$p  ($(find_ckpt "$prefix"))"
      continue
    fi
    echo "[train] $v pl$p  (clean)"
    CUDA_VISIBLE_DEVICES=$GPU python -u run.py $(common_flags) \
      --is_training 1 --model_id "$(clean_model_id "$v" "$p")" --pred_len "$p" \
      $(variant_extra "$v") \
      > "$LOG_DIR/train_${v}_pl${p}.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && echo "[train] DONE: $v pl$p" || echo "[train] FAIL (rc=$rc): $v pl$p -> $LOG_DIR/train_${v}_pl${p}.log"
  done
done

# ── Phase 2: noisy evaluation ─────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════"
echo " Phase 2: noisy test-input evaluation  (levels: $NOISE_LEVELS)"
echo "══════════════════════════════════════════════════════════════════"
for n in $NOISE_LEVELS; do
  for v in $VARIANTS; do
    for p in $PRED_LENS; do
      prefix="$(setting_prefix "$v" "$p")"
      ckpt="$(find_ckpt "$prefix")" || { echo "[eval] MISSING ckpt, skip: $v pl$p (noise${n})"; continue; }
      mid="$(clean_model_id "$v" "$p")_noise${n}"
      echo "[eval] $v pl$p noise${n}%  <- $ckpt"
      CUDA_VISIBLE_DEVICES=$GPU python -u run.py $(common_flags) \
        --is_training 0 --model_id "$mid" --pred_len "$p" \
        --transfer_checkpoint "$ckpt" --test_noise "$n" \
        $(variant_extra "$v") \
        > "$LOG_DIR/eval_${v}_pl${p}_noise${n}.log" 2>&1
      rc=$?
      [ $rc -eq 0 ] || echo "[eval] FAIL (rc=$rc): $v pl$p noise${n} -> $LOG_DIR/eval_${v}_pl${p}_noise${n}.log"
    done
  done
done

# ── Summary table (MSE / MAE per variant x horizon x noise level) ─────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " SUMMARY  (MSE / MAE)"
echo "══════════════════════════════════════════════════════════════════"
get_metric() {  # $1=logfile $2=variant -> "MSE MAE"
  local f=$1 v=$2 mse mae
  [ -f "$f" ] || { echo "NA NA"; return; }
  if [ "$v" = "baseline" ]; then
    local line; line=$(grep -E "^mse:" "$f" | tail -1)
    mse=$(echo "$line" | sed -E 's/^mse:([0-9.eE+-]+).*/\1/')
    mae=$(echo "$line" | sed -E 's/.*mae:([0-9.eE+-]+).*/\1/')
  else
    mse=$(grep -E "^[[:space:]]*MSE:" "$f" | tail -1 | sed -E 's/.*MSE:[[:space:]]*([0-9.eE+-]+).*/\1/')
    mae=$(grep -E "^[[:space:]]*MAE:" "$f" | tail -1 | sed -E 's/.*MAE:[[:space:]]*([0-9.eE+-]+).*/\1/')
  fi
  echo "${mse:-NA} ${mae:-NA}"
}

for v in $VARIANTS; do
  echo ""
  echo "### $v"
  header="pred"
  for n in $NOISE_LEVELS; do header="$header | noise${n}% MSE/MAE"; done
  echo "$header"
  for p in $PRED_LENS; do
    row="$p"
    for n in $NOISE_LEVELS; do
      read -r mse mae <<< "$(get_metric "$LOG_DIR/eval_${v}_pl${p}_noise${n}.log" "$v")"
      row="$row | ${mse}/${mae}"
    done
    echo "$row"
  done
done
echo ""
echo "Per-run logs: $LOG_DIR/eval_<variant>_pl<pred>_noise<level>.log"
