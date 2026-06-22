#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Zero-shot CROSS-DATASET transfer on ETT (Table-6 protocol:
# "train on Da, evaluate on Db WITHOUT further training", Da -> Db).
#
# How it works:
#   1) Train a model on each SOURCE dataset (one checkpoint per pred_len).
#   2) For each (source -> target) pair, run test ON THE TARGET test split,
#      loading the SOURCE checkpoint via --transfer_checkpoint (no retraining).
#
# IMPORTANT — architecture comes from the SOURCE:
#   The transferred model is built from the SOURCE dataset's hyper-params
#   (d_model, d_ff, e_layers, ds_layers, dropout, ...), since we load the source
#   checkpoint. Only root_path/data_path/data switch to the TARGET at eval time.
#   All ETT datasets share enc_in=7, so checkpoints load cleanly across them.
#
# Per-dataset hyper-params are IDENTICAL to
#   scripts/long_term_forecast/JEPAVTS_TimeMixer_all_datasets.sh
#
# Results are appended to:
#   - JEPAVTS:  result_jepa_vts.txt
#   - baselines: result_long_term_forecast.txt
# Average the 4 pred_len rows per pair for the Table-6 numbers.
#
# PREREQUISITE (JEPAVTS only): per-variable DINO embeddings for the SOURCE
# datasets must be precomputed (target eval needs NONE — JEPAVTS test runs only
# the student forecast path):
#   for d in ETTh1 ETTh2 ETTm1 ETTm2; do
#     python utils/precompute_embeddings_pervar.py --dataset $d --method RP
#   done
# ═══════════════════════════════════════════════════════════════════════════

GPU="${GPU:-0}"

# --- model selection -------------------------------------------------------
# MODEL=JEPAVTS uses STUDENT/RENDER + JEPA flags; any other MODEL (PatchTST,
# iTransformer, TimesNet, TimeMixer, DLinear, ...) runs as a plain baseline.
MODEL="${MODEL:-JEPAVTS}"
STUDENT="${STUDENT:-TimeMixer}"      # only used when MODEL=JEPAVTS
RENDER="${RENDER:-RP}"               # only used when MODEL=JEPAVTS (must be precomputed)
USE_DUAL="${USE_DUAL:-0}"            # JEPAVTS: 1 -> --use_dual_encoder --fusion_type mlp

# TimeMixer-only knobs (it REQUIRES multi-scale downsampling; encode() iterates
# over a list of scales). ds_layers is taken per-dataset from get_cfg below.
DS_WINDOW="${DS_WINDOW:-2}"
DS_METHOD="${DS_METHOD:-avg}"
JEPA_SCALE="${JEPA_SCALE:-coarse}"   # JEPAVTS+TimeMixer: timemixer_jepa_scale (fine|coarse)
JEPA_WEIGHT="${JEPA_WEIGHT:-1}"
JEPA_LOSS="${JEPA_LOSS:-mse}"

# --- shared experiment config ---------------------------------------------
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-0}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"

# Transfer pairs (source:target) from Table 6.
PAIRS="${PAIRS:-ETTh1:ETTh2 ETTh1:ETTm2 ETTh2:ETTh1 ETTm1:ETTh2 ETTm1:ETTm2 ETTm2:ETTm1}"

LOG_DIR="${LOG_DIR:-logs/zero_shot_transfer}"
mkdir -p "$LOG_DIR"

# Per-dataset config — same values as JEPAVTS_TimeMixer_all_datasets.sh.
# Field order: enc_in|e_layers|d_model|d_ff|batch_size|ds_layers|lr|epochs|patience|dropout
get_cfg() {
  case "$1" in
    ETTh1) echo "7|2|16|32|128|3|0.01|10|3|0.6" ;;
    ETTh2) echo "7|2|16|32|128|3|0.01|10|3|0.6" ;;
    ETTm1) echo "7|2|16|32|128|3|0.01|10|3|0.1" ;;
    ETTm2) echo "7|2|32|64|128|3|0.01|10|3|0.1" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

# data,root_path,data_path resolver (data == dataset name for ETT loaders).
ds_root() { echo "./dataset/ETT-small/"; }
ds_file() { echo "$1.csv"; }

# Source model_id (appears inside the run.py "setting" string, so we can locate
# the checkpoint folder by glob). For JEPAVTS this MATCHES the model_id used by
# JEPAVTS_TimeMixer_all_datasets.sh / JEPAVTS_iTransformer_all_datasets.sh
# (<dataset>_<RENDER>_<seq>_<pred>_<single|dual>), so if you already trained the
# source models with those scripts, training here is SKIPPED and the existing
# checkpoint is reused for transfer. Set REUSE_TRAINED=0 to force the zsT_ tag
# (always train fresh, isolated from your main runs).
REUSE_TRAINED="${REUSE_TRAINED:-1}"
src_tag() {
  local src=$1 pred=$2
  local dtag="single"; [ "$USE_DUAL" = "1" ] && dtag="dual"
  if [ "$MODEL" = "JEPAVTS" ] && [ "$REUSE_TRAINED" = "1" ]; then
    echo "${src}_${RENDER}_${SEQ_LEN}_${pred}_${dtag}"
  elif [ "$MODEL" = "JEPAVTS" ]; then
    echo "zsT_${MODEL}_${STUDENT}_${RENDER}_${dtag}_${src}_${SEQ_LEN}_${pred}"
  else
    echo "zsT_${MODEL}_${src}_${SEQ_LEN}_${pred}"
  fi
}

# Model-family-specific flags. $1 = ds_layers (for TimeMixer downsampling).
model_flags() {
  local ds_layers=$1
  local flags=""
  if [ "$MODEL" = "JEPAVTS" ]; then
    local extra=""
    [ "$USE_DUAL" = "1" ] && extra="--use_dual_encoder --fusion_type mlp"
    flags="--model JEPAVTS --student_model $STUDENT --per_var_teacher --rendering_methods $RENDER --jepa_weight $JEPA_WEIGHT --jepa_loss_type $JEPA_LOSS $extra"
    if [ "$STUDENT" = "TimeMixer" ]; then
      flags="$flags --down_sampling_layers $ds_layers --down_sampling_method $DS_METHOD --down_sampling_window $DS_WINDOW --timemixer_jepa_scale $JEPA_SCALE"
    fi
  else
    flags="--model $MODEL"
    if [ "$MODEL" = "TimeMixer" ]; then
      flags="$flags --down_sampling_layers $ds_layers --down_sampling_method $DS_METHOD --down_sampling_window $DS_WINDOW"
    fi
  fi
  echo "$flags"
}

find_ckpt() {
  local tag=$1
  ls -1 ./checkpoints/*"${tag}"*/checkpoint.pth 2>/dev/null | head -1
}

# ---------------------------------------------------------------------------
# 1) Train every unique SOURCE (dataset, pred_len) once, using SOURCE config.
# ---------------------------------------------------------------------------
sources=$(for p in $PAIRS; do echo "${p%%:*}"; done | sort -u)
echo ">>> Sources to train: $sources"

for src in $sources; do
  IFS='|' read -r enc_in e_layers d_model d_ff batch ds_layers lr epochs patience dropout <<< "$(get_cfg "$src")" || exit 1
  for pred in $PRED_LENS; do
    tag="$(src_tag "$src" "$pred")"
    if [ -n "$(find_ckpt "$tag")" ]; then
      echo "[train] SKIP (checkpoint exists): $tag"
      continue
    fi
    echo "[train] $tag"
    CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$(ds_root "$src")" \
      --data_path "$(ds_file "$src")" \
      --model_id "$tag" \
      --data "$src" \
      --features M \
      --seq_len "$SEQ_LEN" \
      --label_len "$LABEL_LEN" \
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
      $(model_flags "$ds_layers") \
      > "$LOG_DIR/train_${tag}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[train] FAIL: $tag -> see $LOG_DIR/train_${tag}.log"
    fi
  done
done

# ---------------------------------------------------------------------------
# 2) Evaluate each (source -> target) pair on the TARGET test split, loading
#    the SOURCE checkpoint (no retraining). Architecture uses the SOURCE config.
# ---------------------------------------------------------------------------
for pair in $PAIRS; do
  src="${pair%%:*}"
  tgt="${pair##*:}"
  IFS='|' read -r enc_in e_layers d_model d_ff batch ds_layers lr epochs patience dropout <<< "$(get_cfg "$src")" || exit 1
  for pred in $PRED_LENS; do
    tag="$(src_tag "$src" "$pred")"
    ckpt="$(find_ckpt "$tag")"
    if [ -z "$ckpt" ]; then
      echo "[eval] MISSING source checkpoint for $tag — skipping $src->$tgt pl$pred"
      continue
    fi
    eval_id="zsEVAL_${MODEL}_${src}2${tgt}_${SEQ_LEN}_${pred}"
    [ "$MODEL" = "JEPAVTS" ] && eval_id="zsEVAL_${MODEL}_${STUDENT}_${src}2${tgt}_${SEQ_LEN}_${pred}"
    echo "[eval] $src -> $tgt  pl$pred   (ckpt: $ckpt)"
    CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
      --task_name long_term_forecast \
      --is_training 0 \
      --transfer_checkpoint "$ckpt" \
      --root_path "$(ds_root "$tgt")" \
      --data_path "$(ds_file "$tgt")" \
      --model_id "$eval_id" \
      --data "$tgt" \
      --features M \
      --seq_len "$SEQ_LEN" \
      --label_len "$LABEL_LEN" \
      --pred_len "$pred" \
      --e_layers "$e_layers" \
      --enc_in "$enc_in" \
      --dec_in "$enc_in" \
      --c_out "$enc_in" \
      --d_model "$d_model" \
      --d_ff "$d_ff" \
      --batch_size "$batch" \
      --dropout "$dropout" \
      --des Exp --itr 1 \
      $(model_flags "$ds_layers") \
      > "$LOG_DIR/eval_${eval_id}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[eval] FAIL: $eval_id -> see $LOG_DIR/eval_${eval_id}.log"
    fi
  done
done

echo "Done. Per-pair MSE/MAE are in result_*.txt; average the 4 pred_len rows per pair for Table-6 numbers."
