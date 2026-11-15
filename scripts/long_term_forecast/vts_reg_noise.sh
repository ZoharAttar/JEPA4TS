#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Test-time INPUT-NOISE robustness for the VisionTS++ prediction regularizer.
#
# Three variants, same datasets/horizons/noise levels:
#   patchtst_base    plain PatchTST                              (lambda = 0)
#   patchtst_vtsreg  PatchTST + VisionTS++ MV prediction reg     (lambda = REG_LAMBDA)
#   visionts         VisionTS++ alone, zero-shot MV forecaster   (no training)
#
# Additive Gaussian noise is injected into the TEST input window (x_enc) only:
#   noise std per variable = (noise% / 100) * that variable's std over the test
#   split. Ground-truth targets stay clean. Deterministic given --seed.
# (Same definition as scripts/robustness/forecast_noise.sh / exchange_noise.sh.)
#
# Two phases:
#   1) Train (or reuse) the CLEAN PatchTST checkpoints. Configs come from
#      scripts/long_term_forecast/vts_pred_reg_multi.sh (sourced), so checkpoints
#      already trained by that script (or PatchTST_vts_pred_reg.sh) are reused.
#   2) For every noise level, evaluate every variant/horizon on the noisy test
#      input. PatchTST variants load their CLEAN checkpoint via
#      --transfer_checkpoint (no retraining; the VisionTS++ teacher is a training-only
#      loss, so it is not needed at test time). VisionTS++ is evaluated directly.
#
# Usage:
#   bash scripts/robustness/vts_reg_noise.sh
#   GPU=1 DATASETS="ETTh1 ETTh2 exchange_rate" NOISE_LEVELS="0 5 10 20 30" \
#     bash scripts/robustness/vts_reg_noise.sh
#
# Env overrides:
#   GPU=0                          gpu id
#   DATASETS="ETTh1 ETTh2 exchange_rate"
#   NOISE_LEVELS="0 5 10 20"       percent noise sweep
#   HORIZONS=""                    empty = each dataset's standard horizons
#   VARIANTS="patchtst_base patchtst_vtsreg visionts"
#   REG_LAMBDA=1.0   SEED=2021
#   NOISE_LOG_DIR=logs/vts_reg_noise
#
# Outputs: per-run logs in $NOISE_LOG_DIR, a printed summary, and
#          $NOISE_LOG_DIR/summary.tsv (long format, pastes straight into Excel).
# ═══════════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/../.."

# Load get_ds / model_args / run_one (config only; sourcing does not launch jobs).
source scripts/long_term_forecast/vts_pred_reg_multi.sh

GPU="${GPU:-0}"
NOISE_LEVELS="${NOISE_LEVELS:-0 5 10 20}"
VARIANTS="${VARIANTS:-patchtst_base patchtst_vtsreg visionts}"
REG_LAMBDA="${REG_LAMBDA:-1.0}"
SEED="${SEED:-2021}"
NOISE_LOG_DIR="${NOISE_LOG_DIR:-logs/vts_reg_noise}"
VTS_RESULTS="${VTS_RESULTS:-result_visionts_noise.txt}"
mkdir -p "$NOISE_LOG_DIR"

# Datasets the standalone VisionTS++ evaluator supports (utils/eval_visionts_baseline.py).
VTS_EVAL_DATASETS="ETTh1 ETTh2 exchange_rate"

variant_lambda() { case "$1" in patchtst_base) echo 0 ;; patchtst_vtsreg) echo "$REG_LAMBDA" ;; esac; }

# model_id used at TRAIN time (matches run_one in vts_pred_reg_multi.sh).
train_model_id() {
  local v=$1 ds=$2 seq=$3 pred=$4
  case "$v" in
    patchtst_base)   echo "${ds}_PatchTST_base_${seq}_${pred}" ;;
    patchtst_vtsreg) echo "${ds}_PatchTST_vtsreg${REG_LAMBDA}_${seq}_${pred}" ;;
  esac
}

argval() { echo " $1 " | sed -n "s/.* $2 \([^ ]*\) .*/\1/p"; }

# Locate the clean checkpoint: exact run.py setting first, then a glob fallback.
find_ckpt() {
  local mid=$1 data=$2 seq=$3 pred=$4 margs=$5
  local dm nh el dff
  dm=$(argval "$margs" --d_model); nh=$(argval "$margs" --n_heads)
  el=$(argval "$margs" --e_layers); dff=$(argval "$margs" --d_ff)
  local exact="./checkpoints/long_term_forecast_${mid}_PatchTST_${data}_ftM_sl${seq}_ll0_pl${pred}_dm${dm}_nh${nh}_el${el}_dl1_df${dff}_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
  [ -f "$exact" ] && { echo "$exact"; return 0; }
  local d
  for d in ./checkpoints/long_term_forecast_${mid}_PatchTST_*/; do
    [ -f "${d}checkpoint.pth" ] && { echo "${d}checkpoint.pth"; return 0; }
  done
  return 1
}

get_metric() {  # $1=logfile -> "MSE MAE" from the last "mse:..., mae:..." line
  local f=$1 line
  [ -f "$f" ] || { echo "NA NA"; return; }
  line=$(grep -E "^mse:" "$f" | tail -1)
  [ -n "$line" ] || { echo "NA NA"; return; }
  echo "$(echo "$line" | sed -E 's/^mse:([0-9.eE+-]+).*/\1/') $(echo "$line" | sed -E 's/.*mae:([0-9.eE+-]+).*/\1/')"
}

horizons_of() { local cfg; cfg="$(get_ds "$1")" || return 1; echo "${HORIZONS:-$(printf '%s' "$cfg" | cut -d'|' -f6)}"; }

echo "Variants: $VARIANTS | Datasets: $DATASETS | Noise: $NOISE_LEVELS | lambda=$REG_LAMBDA | GPU $GPU"

for ds in $DATASETS; do
  IFS='|' read -r data root_path data_path enc_in seq _hz P <<< "$(get_ds "$ds")" || exit 1
  echo "══════════════════════════════════════════════════════════════════"
  echo " DATASET: $ds"
  echo "══════════════════════════════════════════════════════════════════"

  # ── Phase 1: clean PatchTST checkpoints (train only what is missing) ──
  for v in $VARIANTS; do
    [ "$v" = "visionts" ] && continue
    for p in $(horizons_of "$ds"); do
      margs="$(model_args PatchTST "$ds" "$p")" || exit 1
      mid="$(train_model_id "$v" "$ds" "$seq" "$p")"
      if find_ckpt "$mid" "$data" "$seq" "$p" "$margs" >/dev/null; then
        echo "  [train] SKIP (ckpt exists): $mid"
      else
        echo "  [train] $mid (clean)"
        run_one PatchTST "$ds" "$p" "$(variant_lambda "$v")" "$GPU"
      fi
    done
  done

  # ── Phase 2: noisy test-input evaluation ──
  for n in $NOISE_LEVELS; do
    for v in $VARIANTS; do
      for p in $(horizons_of "$ds"); do
        logf="$NOISE_LOG_DIR/eval_${ds}_${v}_pl${p}_noise${n}.log"
        if [ "$v" = "visionts" ]; then
          if [[ " $VTS_EVAL_DATASETS " != *" $ds "* ]]; then
            echo "  [eval] SKIP visionts on $ds (evaluator supports: $VTS_EVAL_DATASETS)"; continue
          fi
          echo "  [eval] $ds visionts pl$p noise${n}%"
          CUDA_VISIBLE_DEVICES=$GPU python -u utils/eval_visionts_baseline.py \
            --dataset "$ds" --mode mv --seq_len "$seq" --pred_len "$p" --periodicity "$P" \
            --test_noise "$n" --seed "$SEED" --ckpt_dir "$CKPT_DIR" \
            --results_file "$VTS_RESULTS" > "$logf" 2>&1 \
            || echo "  [eval] FAIL: $ds visionts pl$p noise${n} -> $logf"
        else
          margs="$(model_args PatchTST "$ds" "$p")" || exit 1
          mid="$(train_model_id "$v" "$ds" "$seq" "$p")"
          ckpt="$(find_ckpt "$mid" "$data" "$seq" "$p" "$margs")" \
            || { echo "  [eval] MISSING ckpt, skip: $mid (noise${n})"; continue; }
          echo "  [eval] $ds $v pl$p noise${n}%"
          CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
            --task_name long_term_forecast --is_training 0 \
            --root_path "$root_path" --data_path "$data_path" --data "$data" \
            --model_id "${mid}_noise${n}" --features M \
            --seq_len "$seq" --label_len 0 --pred_len "$p" \
            --enc_in "$enc_in" --dec_in "$enc_in" --c_out "$enc_in" \
            --des Exp --itr 1 $margs \
            --transfer_checkpoint "$ckpt" --test_noise "$n" --seed "$SEED" \
            > "$logf" 2>&1 \
            || echo "  [eval] FAIL: $ds $v pl$p noise${n} -> $logf"
        fi
      done
    done
  done
done

# ── Summary ───────────────────────────────────────────────────────────────
tsv="$NOISE_LOG_DIR/summary.tsv"
printf "dataset\tvariant\thorizon\tnoise_pct\tmse\tmae\n" > "$tsv"
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " SUMMARY (MSE / MAE)"
echo "══════════════════════════════════════════════════════════════════"
for ds in $DATASETS; do
  echo ""
  echo "########## $ds ##########"
  header="variant | pred"
  for n in $NOISE_LEVELS; do header="$header | noise${n}%"; done
  echo "$header"
  for v in $VARIANTS; do
    for p in $(horizons_of "$ds"); do
      row="$v | $p"
      for n in $NOISE_LEVELS; do
        read -r mse mae <<< "$(get_metric "$NOISE_LOG_DIR/eval_${ds}_${v}_pl${p}_noise${n}.log")"
        row="$row | ${mse}/${mae}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ds" "$v" "$p" "$n" "$mse" "$mae" >> "$tsv"
      done
      echo "$row"
    done
  done
done
echo ""
echo "Excel-ready table: $tsv"
echo "Per-run logs:      $NOISE_LOG_DIR/eval_<dataset>_<variant>_pl<pred>_noise<level>.log"
