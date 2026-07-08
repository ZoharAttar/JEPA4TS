#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Table 6 — lambda (jepa_weight) sensitivity under ZERO-SHOT cross-domain
# forecasting. Standard JEPAVTS, single-encoder, TimeMixer student.
#
# Sweeps lambda in {1, 10, 20} over the 6 ETT source->target pairs; each table
# cell is the AVERAGE over pred_len {96,192,336,720} (average those 4 rows after).
#
# This is a thin wrapper over scripts/zero_shot_transfer/ett_transfer.sh. That
# script now encodes jepa_weight into BOTH the checkpoint tag and the eval id, so
# the three lambdas get INDEPENDENT checkpoints + result rows (previously they
# collided into one). Result rows in result_jepa_vts.txt:
#   lambda=1  :  zsEVAL_JEPAVTS_TimeMixer_single_<src>2<tgt>_96_<pred>_s<seed>
#   lambda!=1 :  zsEVAL_JEPAVTS_TimeMixer_single_jw<lambda>_<src>2<tgt>_96_<pred>_s<seed>
#
# PREREQUISITE: per-variable DINO embeddings for the SOURCE datasets (RP):
#   for d in ETTh1 ETTh2 ETTm1 ETTm2; do
#     python utils/precompute_embeddings_pervar.py --dataset $d --method RP
#   done
#
# Usage:
#   GPUS="0 1 2 3" JOBS_PER_GPU=2 bash scripts/zero_shot_transfer/lambda_ablation.sh
#   LAMBDAS="1 5 10 20" SEEDS="2021 2022" bash scripts/zero_shot_transfer/lambda_ablation.sh
# ═══════════════════════════════════════════════════════════════════════════
here=$(cd "$(dirname "$0")" && pwd)

LAMBDAS="${LAMBDAS:-1 10 20}"

export MODEL=JEPAVTS
export STUDENT="${STUDENT:-TimeMixer}"
export RENDER="${RENDER:-RP}"
export USE_DUAL=0                       # single encoder (Table 6 protocol)
export NO_DINO=0
export DINO_DIRECT=0
export FUSION=mlp
export SEEDS="${SEEDS:-2021}"          # table averages horizons, not seeds
export PRED_LENS="${PRED_LENS:-96 192 336 720}"
export PAIRS="${PAIRS:-ETTh1:ETTh2 ETTh1:ETTm2 ETTh2:ETTh1 ETTm1:ETTh2 ETTm1:ETTm2 ETTm2:ETTm1}"
export GPUS="${GPUS:-0}"
export JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

for lam in $LAMBDAS; do
  echo "══════════════════════════════════════════════════════════════════"
  echo "  LAMBDA (jepa_weight) = $lam"
  echo "══════════════════════════════════════════════════════════════════"
  # lambda=1 reuses the all-datasets checkpoints (trained with jepa_weight=1);
  # every other lambda MUST train fresh (its own jw-tagged checkpoint).
  if [ "$lam" = "1" ]; then
    export REUSE_TRAINED=1
  else
    export REUSE_TRAINED=0
  fi
  JEPA_WEIGHT="$lam" bash "$here/ett_transfer.sh"
done

echo "All lambda runs finished. Aggregate with:"
echo "  bash scripts/zero_shot_transfer/lambda_ablation.sh   # (re-run is a no-op; evals are marked .done)"
echo "Then average the 4 pred_len rows per pair, per lambda, from result_jepa_vts.txt."
