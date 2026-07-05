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
# Per-dataset hyper-params are IDENTICAL to the matching all-datasets script,
# selected automatically by the effective backbone (STUDENT for JEPAVTS):
#   TimeMixer    -> scripts/long_term_forecast/JEPAVTS_TimeMixer_all_datasets.sh
#   iTransformer -> scripts/long_term_forecast/JEPAVTS_iTransformer_all_datasets.sh
#
# Results are appended to:
#   - JEPAVTS:  result_jepa_vts.txt
#   - baselines: result_long_term_forecast.txt
# Average the 4 pred_len rows per pair for the Table-6 numbers.
#
# PREREQUISITE (JEPAVTS only): per-variable DINO embeddings for the SOURCE
# datasets must be precomputed (standard JEPA / NO-DINO target eval needs NONE —
# test runs only the student forecast path):
#   for d in ETTh1 ETTh2 ETTm1 ETTm2; do
#     python utils/precompute_embeddings_pervar.py --dataset $d --method RP
#   done
#   EXCEPTION — DINO_DIRECT=1: DINO is a live input, so the TARGET datasets also
#   need precomputed <RENDER> embeddings (precompute all four ETT datasets).
#
# ABLATION SWITCHES (env vars):
#   USE_DUAL=1 | NO_DINO=1 | DINO_DIRECT=1   (pick architecture)
#   FUSION=mlp|transformer|weighted|add       (fusion head; default mlp)
# ═══════════════════════════════════════════════════════════════════════════

GPU="${GPU:-0}"

# --- model selection -------------------------------------------------------
# MODEL=JEPAVTS uses STUDENT/RENDER + JEPA flags; any other MODEL (PatchTST,
# iTransformer, TimesNet, TimeMixer, DLinear, ...) runs as a plain baseline.
MODEL="${MODEL:-JEPAVTS}"
STUDENT="${STUDENT:-TimeMixer}"      # only used when MODEL=JEPAVTS
RENDER="${RENDER:-RP}"               # only used when MODEL=JEPAVTS (must be precomputed)
USE_DUAL="${USE_DUAL:-0}"            # JEPAVTS: 1 -> --use_dual_encoder --fusion_type mlp
# NO-DINO ablation: 1 -> two student encoders fused, task loss only (no teacher /
# no JEPA / no vision). Needs NO precomputed embeddings (source OR target). This
# is the capacity-matched control for the dual encoder. Ignores RENDER/USE_DUAL.
NO_DINO="${NO_DINO:-0}"
# DINO-DIRECT ablation: 1 -> single student encoder + frozen DINO injected into
# the fusion/forecast head (no JEPA distillation). DINO is a LIVE input, so it is
# needed at train AND eval -> requires precomputed <RENDER> per-var embeddings for
# BOTH source and TARGET datasets. Ignores USE_DUAL.
DINO_DIRECT="${DINO_DIRECT:-0}"
# Fusion-type ablation: how the two streams are combined in the fusion head.
# Applies to DINO-DIRECT (student ⊕ DINO), NO-DINO and dual encoder (student ⊕
# student). One of: mlp | transformer | weighted | add. Default mlp.
FUSION="${FUSION:-mlp}"

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

# Multi-seed sweep. Each seed trains its OWN source models and gets its OWN
# checkpoints/results (tag suffix _s<seed>), so you can average mean±std across
# seeds. BASE_SEED is the seed the all-datasets scripts used (2021, hard-coded
# before); with REUSE_TRAINED=1 that one seed reuses those existing checkpoints
# (no _s suffix) instead of retraining. Set SEEDS="2021" for a single-seed run.
SEEDS="${SEEDS:-2021 2022 2023 2024}"
BASE_SEED="${BASE_SEED:-2021}"

# Transfer pairs (source:target) from Table 6.
PAIRS="${PAIRS:-ETTh1:ETTh2 ETTh1:ETTm2 ETTh2:ETTh1 ETTm1:ETTh2 ETTm1:ETTm2 ETTm2:ETTm1}"

LOG_DIR="${LOG_DIR:-logs/zero_shot_transfer}"
mkdir -p "$LOG_DIR"

# Effective backbone: the student (when MODEL=JEPAVTS) or the model itself.
BACKBONE="$MODEL"; [ "$MODEL" = "JEPAVTS" ] && BACKBONE="$STUDENT"

# Per-(backbone, dataset) config — IDENTICAL to the matching all-datasets script:
#   TimeMixer    -> JEPAVTS_TimeMixer_all_datasets.sh
#   iTransformer -> JEPAVTS_iTransformer_all_datasets.sh
# Field order: enc_in|e_layers|d_model|d_ff|batch|lr|epochs|patience|dropout|factor|ds_layers
# (factor used by iTransformer; ds_layers used by TimeMixer downsampling.)
get_cfg() {
  local ds=$1
  case "$BACKBONE" in
    TimeMixer)
      case "$ds" in
        ETTh1|ETTh2) echo "7|2|16|32|128|0.01|10|3|0.6|1|3" ;;
        ETTm1)       echo "7|2|16|32|128|0.01|10|3|0.1|1|3" ;;
        ETTm2)       echo "7|2|32|64|128|0.01|10|3|0.1|1|3" ;;
        *) echo "UNKNOWN dataset: $ds" >&2; return 1 ;;
      esac ;;
    iTransformer)
      case "$ds" in
        ETTh1|ETTh2|ETTm1|ETTm2) echo "7|2|128|128|32|0.0001|10|3|0.1|3|0" ;;
        *) echo "UNKNOWN dataset: $ds" >&2; return 1 ;;
      esac ;;
    *) echo "No config for backbone=$BACKBONE dataset=$ds (add it to get_cfg)" >&2; return 1 ;;
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
  local src=$1 pred=$2 seed=$3
  # Fusion suffix: only added for non-default fusion so existing mlp checkpoints /
  # result ids stay backward-compatible. Applies to all fused JEPAVTS arches.
  local ftag=""; [ "$FUSION" != "mlp" ] && ftag="_${FUSION}"
  # NO-DINO / DINO-DIRECT are distinct architectures, so they never reuse the JEPA
  # all-datasets checkpoints — always their own seeded tag, trained fresh.
  if [ "$DINO_DIRECT" = "1" ]; then
    echo "zsT_dinodirect_${STUDENT}${ftag}_${src}_${SEQ_LEN}_${pred}_s${seed}"
    return
  fi
  if [ "$NO_DINO" = "1" ]; then
    echo "zsT_nodino_${STUDENT}${ftag}_${src}_${SEQ_LEN}_${pred}_s${seed}"
    return
  fi
  local dtag="single"; [ "$USE_DUAL" = "1" ] && dtag="dual"
  # Non-mlp fusion only changes anything for the dual encoder (single encoder has
  # no fusion), and it never matches the pretrained mlp checkpoints.
  [ "$dtag" = "single" ] && ftag=""
  # Seed suffix, EXCEPT for BASE_SEED with REUSE_TRAINED (reuse seedless ckpts
  # already trained by the all-datasets scripts, which used the base seed).
  local sfx="_s${seed}"
  if [ "$REUSE_TRAINED" = "1" ] && [ "$seed" = "$BASE_SEED" ]; then sfx=""; fi
  if [ "$MODEL" = "JEPAVTS" ] && [ "$REUSE_TRAINED" = "1" ] && [ -z "$ftag" ]; then
    echo "${src}_${RENDER}_${SEQ_LEN}_${pred}_${dtag}${sfx}"
  elif [ "$MODEL" = "JEPAVTS" ]; then
    echo "zsT_${MODEL}_${STUDENT}_${RENDER}_${dtag}${ftag}_${src}_${SEQ_LEN}_${pred}${sfx}"
  else
    echo "zsT_${MODEL}_${src}_${SEQ_LEN}_${pred}${sfx}"
  fi
}

# Model-family-specific flags. $1 = ds_layers (for TimeMixer downsampling).
model_flags() {
  local ds_layers=$1
  local flags=""
  if [ "$MODEL" = "JEPAVTS" ] && [ "$DINO_DIRECT" = "1" ]; then
    # DINO-DIRECT ablation: single student + DINO injected into the fusion head.
    # Keeps the teacher (per-var, single rendering) — needs embeddings at train
    # AND eval. No JEPA flags.
    flags="--model JEPAVTS --student_model $STUDENT --dino_direct --per_var_teacher --rendering_methods $RENDER --fusion_type $FUSION"
    if [ "$STUDENT" = "TimeMixer" ]; then
      flags="$flags --down_sampling_layers $ds_layers --down_sampling_method $DS_METHOD --down_sampling_window $DS_WINDOW --timemixer_jepa_scale $JEPA_SCALE"
    fi
  elif [ "$MODEL" = "JEPAVTS" ] && [ "$NO_DINO" = "1" ]; then
    # NO-DINO ablation: dual student encoders, task loss only. No teacher flags
    # (--per_var_teacher / --rendering_methods / --jepa_*), no precompute needed.
    flags="--model JEPAVTS --student_model $STUDENT --no_dino --fusion_type $FUSION"
    if [ "$STUDENT" = "TimeMixer" ]; then
      flags="$flags --down_sampling_layers $ds_layers --down_sampling_method $DS_METHOD --down_sampling_window $DS_WINDOW --timemixer_jepa_scale $JEPA_SCALE"
    fi
  elif [ "$MODEL" = "JEPAVTS" ]; then
    local extra=""
    [ "$USE_DUAL" = "1" ] && extra="--use_dual_encoder --fusion_type $FUSION"
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
  # Match tag AND dm/df, because the run.py "setting" string does NOT include the
  # student model name — different students (e.g. iTransformer vs TimeMixer) share
  # the same all-datasets model_id and only differ by d_model/d_ff. Without this,
  # a glob on the tag alone can grab the wrong student's checkpoint.
  local tag=$1 d_model=$2 d_ff=$3
  ls -1 ./checkpoints/*"${tag}"*_dm"${d_model}"_*_df"${d_ff}"_*/checkpoint.pth 2>/dev/null | head -1
}

# ---------------------------------------------------------------------------
# 1) Train every unique SOURCE (dataset, pred_len) once, using SOURCE config.
# ---------------------------------------------------------------------------
sources=$(for p in $PAIRS; do echo "${p%%:*}"; done | sort -u)
echo ">>> Seeds: $SEEDS"
echo ">>> Sources to train: $sources"

for seed in $SEEDS; do
 echo "================ SEED $seed : TRAIN ================"
 for src in $sources; do
  IFS='|' read -r enc_in e_layers d_model d_ff batch lr epochs patience dropout factor ds_layers <<< "$(get_cfg "$src")" || exit 1
  for pred in $PRED_LENS; do
    tag="$(src_tag "$src" "$pred" "$seed")"
    if [ -n "$(find_ckpt "$tag" "$d_model" "$d_ff")" ]; then
      echo "[train] SKIP (checkpoint exists): $tag (dm$d_model df$d_ff)"
      continue
    fi
    echo "[train] $tag (seed $seed)"
    CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --seed "$seed" \
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
      --factor "$factor" \
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
done

# ---------------------------------------------------------------------------
# 2) Evaluate each (source -> target) pair on the TARGET test split, loading
#    the SOURCE checkpoint (no retraining). Architecture uses the SOURCE config.
# ---------------------------------------------------------------------------
for seed in $SEEDS; do
 echo "================ SEED $seed : EVAL ================"
 for pair in $PAIRS; do
  src="${pair%%:*}"
  tgt="${pair##*:}"
  IFS='|' read -r enc_in e_layers d_model d_ff batch lr epochs patience dropout factor ds_layers <<< "$(get_cfg "$src")" || exit 1
  for pred in $PRED_LENS; do
    tag="$(src_tag "$src" "$pred" "$seed")"
    ckpt="$(find_ckpt "$tag" "$d_model" "$d_ff")"
    if [ -z "$ckpt" ]; then
      echo "[eval] MISSING source checkpoint for $tag (dm$d_model df$d_ff) — skipping $src->$tgt pl$pred seed$seed"
      continue
    fi
    dtag="single"; [ "$USE_DUAL" = "1" ] && dtag="dual"
    # Fusion suffix in the result id (non-default only), so fusion-type ablation
    # variants land in distinct result rows.
    eftag=""; [ "$FUSION" != "mlp" ] && eftag="_${FUSION}"
    dual_eftag="$eftag"; [ "$dtag" = "single" ] && dual_eftag=""
    eval_id="zsEVAL_${MODEL}_${src}2${tgt}_${SEQ_LEN}_${pred}_s${seed}"
    [ "$MODEL" = "JEPAVTS" ] && eval_id="zsEVAL_${MODEL}_${STUDENT}_${dtag}${dual_eftag}_${src}2${tgt}_${SEQ_LEN}_${pred}_s${seed}"
    [ "$NO_DINO" = "1" ] && eval_id="zsEVAL_JEPAVTS_nodino_${STUDENT}${eftag}_${src}2${tgt}_${SEQ_LEN}_${pred}_s${seed}"
    [ "$DINO_DIRECT" = "1" ] && eval_id="zsEVAL_JEPAVTS_dinodirect_${STUDENT}${eftag}_${src}2${tgt}_${SEQ_LEN}_${pred}_s${seed}"
    echo "[eval] $src -> $tgt  pl$pred  seed$seed  $dtag  (ckpt: $ckpt)"
    CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
      --task_name long_term_forecast \
      --is_training 0 \
      --seed "$seed" \
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
      --factor "$factor" \
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
done

echo "Done. Per-pair MSE/MAE (one row per seed, tagged _s<seed>) are in result_*.txt;"
echo "average the 4 pred_len rows per pair, then take mean±std across the $(echo $SEEDS | wc -w) seeds."
