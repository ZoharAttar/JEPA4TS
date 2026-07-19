#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Test-time INPUT-NOISE robustness on forecasting, multiple datasets.
#
#   Baseline TimeMixer   vs   JEPAVTS + TimeMixer student   (SINGLE encoder)
#
# Same protocol as scripts/robustness/exchange_noise.sh, generalized to ETT +
# Weather. Additive Gaussian noise is injected into the TEST input window
# (x_enc) only: noise std per variable = (noise% / 100) * that variable's std
# over the test split. Ground-truth targets stay clean. Deterministic given --seed.
#
# Per-dataset TimeMixer hyper-params are IDENTICAL to
# scripts/long_term_forecast/JEPAVTS_TimeMixer_all_datasets.sh, so the CLEAN
# JEPAVTS single-encoder checkpoints you already trained are reused as-is
# (only the missing plain-TimeMixer baselines get trained).
#
# Two phases per dataset:
#   1) Train (or reuse) the CLEAN checkpoints once.
#   2) For every noise level, evaluate every horizon/variant on the noisy test
#      input, loading the CLEAN checkpoint via --transfer_checkpoint (no retrain).
#
# Usage (defaults shown):
#   bash scripts/robustness/forecast_noise.sh
#   GPU=1 DATASETS="ETTh1 weather" bash scripts/robustness/forecast_noise.sh
#   NOISE_LEVELS="0 5 10 20" VARIANTS="baseline jepa_single" bash scripts/robustness/forecast_noise.sh
#
# Env overrides:
#   GPU=0                              gpu id
#   DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather"
#   NOISE_LEVELS="0 5 10 20"          percent noise sweep
#   PRED_LENS="96 192 336 720"        horizons
#   VARIANTS="baseline jepa_single"   (single encoder only, per request)
#   RENDER=RP   SEED=2021
#
# PREREQUISITE for the JEPAVTS variant: per-variable DINO embeddings must exist
# for each dataset (they do if you trained the JEPAVTS single-encoder runs):
#   python utils/precompute_embeddings_pervar.py --dataset ETTh1 --method RP
# (Only needed at TRAIN time; noisy-eval does not touch the teacher.)
# ═══════════════════════════════════════════════════════════════════════════
set -u

GPU="${GPU:-0}"
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather}"
NOISE_LEVELS="${NOISE_LEVELS:-0 5 10 20}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
VARIANTS="${VARIANTS:-baseline jepa_single}"
RENDER="${RENDER:-RP}"
SEED="${SEED:-2021}"
LOG_DIR="${LOG_DIR:-logs/forecast_noise}"
CKPT_ROOT="./checkpoints"

DS_WINDOW=2
JEPA_SCALE=coarse
JEPA_WEIGHT=1
JEPA_LOSS=mse

mkdir -p "$LOG_DIR"

# Per-dataset config (matches JEPAVTS_TimeMixer_all_datasets.sh).
# Field order: data|root_path|data_path|enc_in|e_layers|d_model|d_ff|batch|ds_layers|lr|epochs|patience|dropout|seq_len
get_cfg() {
  case "$1" in
    ETTh1)   echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7|2|16|32|128|3|0.01|10|3|0.6|96" ;;
    ETTh2)   echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7|2|16|32|128|3|0.01|10|3|0.6|96" ;;
    ETTm1)   echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7|2|16|32|128|3|0.01|10|3|0.1|96" ;;
    ETTm2)   echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7|2|32|64|128|3|0.01|10|3|0.1|96" ;;
    weather) echo "custom|./dataset/weather/|weather.csv|21|2|16|32|128|3|0.01|20|10|0.1|96" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

# ── flag builders (use the per-dataset globals set in the loop) ────────────
common_flags() {
  echo "--task_name long_term_forecast \
    --root_path $ROOT --data_path $DPATH --data $DATA \
    --features M --seq_len $SEQ --label_len 0 \
    --e_layers $E_LAYERS --enc_in $ENC_IN --c_out $ENC_IN \
    --des Exp --itr 1 --d_model $D_MODEL --d_ff $D_FF \
    --learning_rate $LR --train_epochs $EPOCHS --patience $PATIENCE \
    --batch_size $BATCH --dropout $DROPOUT --seed $SEED"
}
tm_flags() {
  echo "--down_sampling_layers $DS_LAYERS --down_sampling_method avg --down_sampling_window $DS_WINDOW"
}
jepa_flags() {
  echo "--model JEPAVTS --student_model TimeMixer \
    --down_sampling_layers $DS_LAYERS --down_sampling_method avg --down_sampling_window $DS_WINDOW \
    --timemixer_jepa_scale $JEPA_SCALE --per_var_teacher --rendering_methods $RENDER \
    --jepa_weight $JEPA_WEIGHT --jepa_loss_type $JEPA_LOSS"
}

# Clean-checkpoint setting prefix per variant (matches run.py's builder). The
# trailing _TimeMixer student suffix (for JEPAVTS) is matched by a glob in find_ckpt.
setting_prefix() {
  local variant=$1 pred=$2
  local tail="_${DATA}_ftM_sl${SEQ}_ll0_pl${pred}_dm${D_MODEL}_nh8_el${E_LAYERS}_dl1_df${D_FF}_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0"
  case "$variant" in
    baseline)    echo "long_term_forecast_${NAME}_${SEQ}_${pred}_TimeMixer${tail}" ;;
    jepa_single) echo "long_term_forecast_${NAME}_${RENDER}_${SEQ}_${pred}_single_JEPAVTS${tail}" ;;
    jepa_dual)   echo "long_term_forecast_${NAME}_${RENDER}_${SEQ}_${pred}_dual_JEPAVTS${tail}" ;;
    *) echo ""; return 1 ;;
  esac
}

find_ckpt() {
  local prefix=$1 d
  for d in "${CKPT_ROOT}/${prefix}"*/; do
    [ -f "${d}checkpoint.pth" ] && { echo "${d}checkpoint.pth"; return 0; }
  done
  return 1
}

variant_extra() {
  case "$1" in
    baseline)    echo "--model TimeMixer $(tm_flags)" ;;
    jepa_single) echo "$(jepa_flags)" ;;
    jepa_dual)   echo "$(jepa_flags) --use_dual_encoder --fusion_type mlp" ;;
  esac
}

clean_model_id() {
  local variant=$1 pred=$2
  case "$variant" in
    baseline)    echo "${NAME}_${SEQ}_${pred}" ;;
    jepa_single) echo "${NAME}_${RENDER}_${SEQ}_${pred}_single" ;;
    jepa_dual)   echo "${NAME}_${RENDER}_${SEQ}_${pred}_dual" ;;
  esac
}

get_metric() {  # $1=logfile $2=variant -> "MSE MAE"
  local f=$1 v=$2 mse mae line
  [ -f "$f" ] || { echo "NA NA"; return; }
  if [ "$v" = "baseline" ]; then
    line=$(grep -E "^mse:" "$f" | tail -1)
    mse=$(echo "$line" | sed -E 's/^mse:([0-9.eE+-]+).*/\1/')
    mae=$(echo "$line" | sed -E 's/.*mae:([0-9.eE+-]+).*/\1/')
  else
    mse=$(grep -E "^[[:space:]]*MSE:" "$f" | tail -1 | sed -E 's/.*MSE:[[:space:]]*([0-9.eE+-]+).*/\1/')
    mae=$(grep -E "^[[:space:]]*MAE:" "$f" | tail -1 | sed -E 's/.*MAE:[[:space:]]*([0-9.eE+-]+).*/\1/')
  fi
  echo "${mse:-NA} ${mae:-NA}"
}

# ═══════════════════════════════════════════════════════════════════════════
for ds in $DATASETS; do
  IFS='|' read -r DATA ROOT DPATH ENC_IN E_LAYERS D_MODEL D_FF BATCH DS_LAYERS LR EPOCHS PATIENCE DROPOUT SEQ <<< "$(get_cfg "$ds")" || exit 1
  NAME="$ds"

  echo "══════════════════════════════════════════════════════════════════"
  echo " DATASET: $ds  (data=$DATA enc_in=$ENC_IN dm=$D_MODEL df=$D_FF)"
  echo "══════════════════════════════════════════════════════════════════"

  # ── Phase 1: train / reuse clean checkpoints ──────────────────────────
  echo "  Phase 1: clean checkpoints (train only what is missing)"
  for v in $VARIANTS; do
    for p in $PRED_LENS; do
      prefix="$(setting_prefix "$v" "$p")"
      if find_ckpt "$prefix" >/dev/null 2>&1; then
        echo "  [train] SKIP (ckpt exists): $ds $v pl$p"
        continue
      fi
      echo "  [train] $ds $v pl$p  (clean)"
      CUDA_VISIBLE_DEVICES=$GPU python -u run.py $(common_flags) \
        --is_training 1 --model_id "$(clean_model_id "$v" "$p")" --pred_len "$p" \
        $(variant_extra "$v") \
        > "$LOG_DIR/train_${ds}_${v}_pl${p}.log" 2>&1
      rc=$?
      [ $rc -eq 0 ] && echo "  [train] DONE: $ds $v pl$p" || echo "  [train] FAIL (rc=$rc): $ds $v pl$p -> $LOG_DIR/train_${ds}_${v}_pl${p}.log"
    done
  done

  # ── Phase 2: noisy evaluation ─────────────────────────────────────────
  echo "  Phase 2: noisy test-input evaluation (levels: $NOISE_LEVELS)"
  for n in $NOISE_LEVELS; do
    for v in $VARIANTS; do
      for p in $PRED_LENS; do
        prefix="$(setting_prefix "$v" "$p")"
        ckpt="$(find_ckpt "$prefix")" || { echo "  [eval] MISSING ckpt, skip: $ds $v pl$p (noise${n})"; continue; }
        mid="$(clean_model_id "$v" "$p")_noise${n}"
        echo "  [eval] $ds $v pl$p noise${n}%"
        CUDA_VISIBLE_DEVICES=$GPU python -u run.py $(common_flags) \
          --is_training 0 --model_id "$mid" --pred_len "$p" \
          --transfer_checkpoint "$ckpt" --test_noise "$n" \
          $(variant_extra "$v") \
          > "$LOG_DIR/eval_${ds}_${v}_pl${p}_noise${n}.log" 2>&1
        rc=$?
        [ $rc -eq 0 ] || echo "  [eval] FAIL (rc=$rc): $ds $v pl$p noise${n} -> $LOG_DIR/eval_${ds}_${v}_pl${p}_noise${n}.log"
      done
    done
  done
done

# ── Summary tables (per dataset x variant x horizon x noise level) ────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " SUMMARY  (MSE / MAE)"
echo "══════════════════════════════════════════════════════════════════"
for ds in $DATASETS; do
  IFS='|' read -r DATA ROOT DPATH ENC_IN E_LAYERS D_MODEL D_FF BATCH DS_LAYERS LR EPOCHS PATIENCE DROPOUT SEQ <<< "$(get_cfg "$ds")"
  NAME="$ds"
  echo ""
  echo "########## $ds ##########"
  for v in $VARIANTS; do
    echo ""
    echo "### $ds / $v"
    header="pred"
    for n in $NOISE_LEVELS; do header="$header | noise${n}% MSE/MAE"; done
    echo "$header"
    for p in $PRED_LENS; do
      row="$p"
      for n in $NOISE_LEVELS; do
        read -r mse mae <<< "$(get_metric "$LOG_DIR/eval_${ds}_${v}_pl${p}_noise${n}.log" "$v")"
        row="$row | ${mse}/${mae}"
      done
      echo "$row"
    done
  done
done
echo ""
echo "Per-run logs: $LOG_DIR/eval_<dataset>_<variant>_pl<pred>_noise<level>.log"
