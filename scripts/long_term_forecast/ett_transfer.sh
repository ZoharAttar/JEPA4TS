#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Zero-shot CROSS-DATASET transfer on ETT (reproduces the Table-6 protocol:
# "train on Da, evaluate on Db WITHOUT further training", Da -> Db).
#
# How it works:
#   1) Train a model on each SOURCE dataset (one checkpoint per pred_len).
#   2) For each (source -> target) pair, run test ON THE TARGET test split,
#      loading the SOURCE checkpoint via --transfer_checkpoint (no retraining).
#
# Results are appended to:
#   - JEPAVTS:  result_jepa_vts.txt
#   - baselines: result_long_term_forecast.txt
# and saved under ./results/<setting>/ . Final MSE/MAE per pair are averaged
# over the 4 prediction lengths {96,192,336,720}, matching Table 6.
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
STUDENT="${STUDENT:-iTransformer}"   # only used when MODEL=JEPAVTS
RENDER="${RENDER:-RP}"               # only used when MODEL=JEPAVTS (must be precomputed)
USE_DUAL="${USE_DUAL:-0}"            # JEPAVTS: 1 -> --use_dual_encoder --fusion_type mlp

# --- shared experiment config ---------------------------------------------
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-0}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
ENC_IN=7                              # all ETT datasets have 7 variables
D_MODEL="${D_MODEL:-128}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-2}"
FACTOR="${FACTOR:-3}"
BATCH="${BATCH:-32}"
LR="${LR:-0.0001}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
DROPOUT="${DROPOUT:-0.1}"

# Transfer pairs (source:target) from Table 6.
PAIRS="${PAIRS:-ETTh1:ETTh2 ETTh1:ETTm2 ETTh2:ETTh1 ETTm1:ETTh2 ETTm1:ETTm2 ETTm2:ETTm1}"

LOG_DIR="${LOG_DIR:-logs/zero_shot_transfer}"
mkdir -p "$LOG_DIR"

# data,root_path,data_path resolver (data == dataset name for ETT loaders).
ds_root() { echo "./dataset/ETT-small/"; }
ds_file() { echo "$1.csv"; }

# A short, unique tag identifying the trained source model (appears inside the
# run.py "setting" string, so we can locate the checkpoint folder by glob).
src_tag() {
  local src=$1 pred=$2
  if [ "$MODEL" = "JEPAVTS" ]; then
    local dtag="single"; [ "$USE_DUAL" = "1" ] && dtag="dual"
    echo "zsT_${MODEL}_${STUDENT}_${RENDER}_${dtag}_${src}_${SEQ_LEN}_${pred}"
  else
    echo "zsT_${MODEL}_${src}_${SEQ_LEN}_${pred}"
  fi
}

# Append the model-family-specific flags to the python command.
model_flags() {
  if [ "$MODEL" = "JEPAVTS" ]; then
    local extra=""
    [ "$USE_DUAL" = "1" ] && extra="--use_dual_encoder --fusion_type mlp"
    echo "--model JEPAVTS --student_model $STUDENT --per_var_teacher --rendering_methods $RENDER --jepa_weight 1 --jepa_loss_type mse $extra"
  else
    echo "--model $MODEL"
  fi
}

find_ckpt() {
  # locate checkpoint.pth for a given source model_id (glob on the unique tag)
  local tag=$1
  ls -1 ./checkpoints/*"${tag}"*/checkpoint.pth 2>/dev/null | head -1
}

# ---------------------------------------------------------------------------
# 1) Train every unique SOURCE (dataset, pred_len) once.
# ---------------------------------------------------------------------------
sources=$(for p in $PAIRS; do echo "${p%%:*}"; done | sort -u)
echo ">>> Sources to train: $sources"

for src in $sources; do
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
      --e_layers "$E_LAYERS" \
      --enc_in "$ENC_IN" \
      --dec_in "$ENC_IN" \
      --c_out "$ENC_IN" \
      --factor "$FACTOR" \
      --d_model "$D_MODEL" \
      --d_ff "$D_FF" \
      --batch_size "$BATCH" \
      --learning_rate "$LR" \
      --train_epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --dropout "$DROPOUT" \
      --des Exp --itr 1 \
      $(model_flags) \
      > "$LOG_DIR/train_${tag}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[train] FAIL: $tag -> see $LOG_DIR/train_${tag}.log"
    fi
  done
done

# ---------------------------------------------------------------------------
# 2) Evaluate each (source -> target) pair on the TARGET test split, loading
#    the SOURCE checkpoint (no retraining).
# ---------------------------------------------------------------------------
for pair in $PAIRS; do
  src="${pair%%:*}"
  tgt="${pair##*:}"
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
      --e_layers "$E_LAYERS" \
      --enc_in "$ENC_IN" \
      --dec_in "$ENC_IN" \
      --c_out "$ENC_IN" \
      --factor "$FACTOR" \
      --d_model "$D_MODEL" \
      --d_ff "$D_FF" \
      --batch_size "$BATCH" \
      --dropout "$DROPOUT" \
      --des Exp --itr 1 \
      $(model_flags) \
      > "$LOG_DIR/eval_${eval_id}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[eval] FAIL: $eval_id -> see $LOG_DIR/eval_${eval_id}.log"
    fi
  done
done

echo "Done. Per-pair MSE/MAE are in result_*.txt; average the 4 pred_len rows per pair for Table-6 numbers."
