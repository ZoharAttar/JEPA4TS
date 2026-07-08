#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Table 5 — student-variant / fusion-strategy ablation under ZERO-SHOT
# cross-domain forecasting. TimeMixer student. 6 ETT source->target pairs;
# each cell is the AVERAGE over pred_len {96,192,336,720} (average the 4 rows).
#
# Columns (each a separate run over scripts/zero_shot_transfer/ett_transfer.sh):
#   1) Single Encoder / Baseline (Shared)  -> USE_DUAL=0
#   2) Dual Encoder  / MLP Fusion          -> USE_DUAL=1 FUSION=mlp
#   3) Dual Encoder  / Learned Interp.     -> USE_DUAL=1 FUSION=weighted
#   4) Dual Encoder  / Transformer Fusion  -> USE_DUAL=1 FUSION=transformer
#
# The fusion type is encoded in BOTH the checkpoint tag and the eval id (non-mlp
# only), so the variants get independent checkpoints + result rows. REUSE_TRAINED=1:
# the single + dual-mlp variants reuse the all-datasets checkpoints; weighted /
# transformer fusion train fresh (their fusion head differs -> distinct arch).
#
# Result rows in result_jepa_vts.txt:
#   single       :  zsEVAL_JEPAVTS_TimeMixer_single_<src>2<tgt>_96_<pred>_s<seed>
#   dual mlp     :  zsEVAL_JEPAVTS_TimeMixer_dual_<src>2<tgt>_96_<pred>_s<seed>
#   dual weighted:  zsEVAL_JEPAVTS_TimeMixer_dual_weighted_<src>2<tgt>_96_<pred>_s<seed>
#   dual transf. :  zsEVAL_JEPAVTS_TimeMixer_dual_transformer_<src>2<tgt>_96_<pred>_s<seed>
#
# PREREQUISITE: per-variable DINO embeddings for the SOURCE datasets (RP):
#   for d in ETTh1 ETTh2 ETTm1 ETTm2; do
#     python utils/precompute_embeddings_pervar.py --dataset $d --method RP
#   done
#
# Usage:
#   GPUS="0 1 2 3" JOBS_PER_GPU=2 bash scripts/zero_shot_transfer/fusion_ablation.sh
#   VARIANTS="dual_weighted dual_transformer" bash scripts/zero_shot_transfer/fusion_ablation.sh
# ═══════════════════════════════════════════════════════════════════════════
here=$(cd "$(dirname "$0")" && pwd)

# Which columns to run: single dual_mlp dual_weighted dual_transformer dual_add
VARIANTS="${VARIANTS:-single dual_mlp dual_weighted dual_transformer dual_add}"

export MODEL=JEPAVTS
export STUDENT="${STUDENT:-TimeMixer}"
export RENDER="${RENDER:-RP}"
export NO_DINO=0
export DINO_DIRECT=0
export JEPA_WEIGHT=1
export SEEDS="${SEEDS:-2021}"          # table averages horizons, not seeds
export PRED_LENS="${PRED_LENS:-96 192 336 720}"
export PAIRS="${PAIRS:-ETTh1:ETTh2 ETTh1:ETTm2 ETTh2:ETTh1 ETTm1:ETTh2 ETTm1:ETTm2 ETTm2:ETTm1}"
export REUSE_TRAINED="${REUSE_TRAINED:-1}"
export GPUS="${GPUS:-0}"
export JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

for v in $VARIANTS; do
  case "$v" in
    single)           dual=0; fusion=mlp ;;
    dual_mlp)         dual=1; fusion=mlp ;;
    dual_weighted)    dual=1; fusion=weighted ;;
    dual_transformer) dual=1; fusion=transformer ;;
    dual_add)         dual=1; fusion=add ;;
    *) echo "UNKNOWN variant: $v (use single|dual_mlp|dual_weighted|dual_transformer|dual_add)" >&2; exit 1 ;;
  esac
  echo "══════════════════════════════════════════════════════════════════"
  echo "  VARIANT = $v   (USE_DUAL=$dual  FUSION=$fusion)"
  echo "══════════════════════════════════════════════════════════════════"
  USE_DUAL="$dual" FUSION="$fusion" bash "$here/ett_transfer.sh"
done

echo "All fusion variants finished. Average the 4 pred_len rows per pair, per variant,"
echo "from result_jepa_vts.txt to fill Table 5."
