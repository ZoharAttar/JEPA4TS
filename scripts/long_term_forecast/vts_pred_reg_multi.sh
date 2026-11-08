#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# VisionTS++ PREDICTION regularizer across multiple students and datasets.
#   L = MSE(student, y_true) + LAMBDA * MSE(student, VisionTS++ forecast)
# VisionTS++ frozen, MV (colorized multivariate) by default, same 96 look-back.
# The regularizer lives in Exp_Long_Term_Forecast, so it works for ANY student.
#
# Students : PatchTST (per-horizon tuned) | DLinear | TimeMixer (per-dataset tuned)
# LAMBDA=0 is the plain baseline for that student (no VisionTS++).
#
# Setup (once): pip install git+https://github.com/Keytoyze/VisionTS.git
#
# Usage:
#   bash scripts/long_term_forecast/vts_pred_reg_multi.sh
#   STUDENTS="DLinear TimeMixer" DATASETS="ETTh1 ETTh2 exchange_rate" \
#     LAMBDAS="0 1.0" GPUS="0 1 2 3" bash scripts/long_term_forecast/vts_pred_reg_multi.sh
# ═══════════════════════════════════════════════════════════════════════════

GPUS="${GPUS:-0 1 2 3}"
STUDENTS="${STUDENTS:-PatchTST DLinear TimeMixer}"
DATASETS="${DATASETS:-ETTh1 ETTh2 exchange_rate}"
HORIZONS="${HORIZONS:-96 192 336 720}"
LAMBDAS="${LAMBDAS:-0 1.0}"          # 0 = baseline; 1.0 = MV reg (best from the sweep)
VTS_RENDER="${VTS_RENDER:-mv}"       # mv | ci
# Student channel-handling variants:
TIMEMIXER_CI="${TIMEMIXER_CI:-1}"        # TimeMixer: 1 = channel-independent (default), 0 = channel-mixing
DLINEAR_INDIVIDUAL="${DLINEAR_INDIVIDUAL:-0}"  # DLinear: 0 = shared weights (default), 1 = per-channel weights
CKPT_DIR="${CKPT_DIR:-./hf_models/VisionTSpp}"
LOG_DIR="${LOG_DIR:-logs/vts_pred_reg_multi}"
mkdir -p "$LOG_DIR"

# VisionTS++ rendering periodicity. NB: at seq_len=96 the MV teacher is only
# meaningful when the daily period fits in 96 -> ETTh*/exchange are fine; ETTm*
# (P=96) and weather (P=144) are degenerate at 96 (<=1 column) and the teacher
# target will be weak there.
get_P() { case "$1" in
  ETTh1|ETTh2) echo 24 ;;
  ETTm1|ETTm2) echo 96 ;;
  weather)     echo 144 ;;
  exchange_rate) echo 1 ;;
  *) echo 24 ;; esac; }

# data|root_path|data_path|enc_in
get_data() { case "$1" in
  ETTh1) echo "ETTh1|./dataset/ETT-small/|ETTh1.csv|7" ;;
  ETTh2) echo "ETTh2|./dataset/ETT-small/|ETTh2.csv|7" ;;
  ETTm1) echo "ETTm1|./dataset/ETT-small/|ETTm1.csv|7" ;;
  ETTm2) echo "ETTm2|./dataset/ETT-small/|ETTm2.csv|7" ;;
  weather) echo "custom|./dataset/weather/|weather.csv|21" ;;
  exchange_rate) echo "custom|./dataset/exchange_rate/|exchange_rate.csv|8" ;;
  *) echo "UNKNOWN $1" >&2; return 1 ;; esac; }

# Student-specific run.py flags (everything except data/seq/pred/reg args).
model_args() {
  local student=$1 ds=$2 pred=$3
  case "$student" in
    PatchTST)
      local el nh bt ep=10
      case "${ds}_${pred}" in
        ETTh1_96)  el=1; nh=2;  bt=32 ;;   ETTh1_192) el=1; nh=8;  bt=32 ;;
        ETTh1_336) el=1; nh=8;  bt=32 ;;   ETTh1_720) el=1; nh=16; bt=32 ;;
        ETTh2_*)   el=3; nh=4;  bt=32 ;;
        ETTm1_96)  el=1; nh=2;  bt=32  ;;  ETTm1_192) el=3; nh=2;  bt=128 ;;
        ETTm1_336) el=1; nh=4;  bt=128 ;;  ETTm1_720) el=3; nh=4;  bt=128 ;;
        ETTm2_96)  el=3; nh=16; bt=32  ;;  ETTm2_192) el=3; nh=2;  bt=128 ;;
        ETTm2_336) el=1; nh=4;  bt=32  ;;  ETTm2_720) el=3; nh=4;  bt=128 ;;
        weather_*) el=2; nh=4;  bt=32; ep=3 ;;
        exchange_rate_*) el=2; nh=8; bt=32 ;;
        *) echo "UNKNOWN PatchTST ${ds}_${pred}" >&2; return 1 ;;
      esac
      echo "--model PatchTST --e_layers $el --n_heads $nh --d_model 512 --d_ff 2048 --factor 3 --batch_size $bt --learning_rate 0.0001 --train_epochs $ep --patience 3 --dropout 0.1" ;;
    DLinear)
      local ind=""; [ "$DLINEAR_INDIVIDUAL" = "1" ] && ind="--individual"
      echo "--model DLinear --e_layers 2 --d_model 512 --batch_size 32 --learning_rate 0.005 --train_epochs 10 --patience 3 --dropout 0.1 $ind" ;;
    TimeMixer)
      local el dm dff bt ds_l lr ep pat drop
      case "$ds" in
        ETTh1|ETTh2)   el=2; dm=16; dff=32; bt=128; ds_l=3; lr=0.01; ep=10; pat=3;  drop=0.6 ;;
        ETTm1)         el=2; dm=16; dff=32; bt=128; ds_l=3; lr=0.01; ep=10; pat=3;  drop=0.1 ;;
        ETTm2)         el=2; dm=32; dff=64; bt=128; ds_l=3; lr=0.01; ep=10; pat=3;  drop=0.1 ;;
        weather)       el=2; dm=16; dff=32; bt=128; ds_l=3; lr=0.01; ep=20; pat=10; drop=0.1 ;;
        exchange_rate) el=2; dm=16; dff=32; bt=32;  ds_l=3; lr=0.01; ep=10; pat=3;  drop=0.1 ;;
        *) echo "UNKNOWN TimeMixer $ds" >&2; return 1 ;;
      esac
      echo "--model TimeMixer --e_layers $el --d_model $dm --d_ff $dff --batch_size $bt --down_sampling_layers $ds_l --down_sampling_method avg --down_sampling_window 2 --channel_independence $TIMEMIXER_CI --learning_rate $lr --train_epochs $ep --patience $pat --dropout $drop" ;;
    *) echo "UNKNOWN student $student" >&2; return 1 ;;
  esac
}

run_one() {
  local student=$1 ds=$2 pred=$3 lam=$4 gpu=$5
  IFS='|' read -r data root_path data_path enc_in <<< "$(get_data "$ds")" || return 1
  local P; P=$(get_P "$ds")
  local margs; margs="$(model_args "$student" "$ds" "$pred")" || return 1

  local reg_args="" tag="base"
  if [ "$lam" != "0" ]; then
    reg_args="--vts_pred_reg --vts_pred_reg_weight $lam --vts_pred_periodicity $P --vts_pred_render $VTS_RENDER --vts_pred_ckpt_dir $CKPT_DIR"
    if [ "$VTS_RENDER" = "ci" ]; then tag="vtsregci${lam}"; else tag="vtsreg${lam}"; fi
  fi

  # Variant suffix so channel-mixing / per-channel runs get distinct model_ids.
  local sfx=""
  [ "$student" = "TimeMixer" ] && [ "$TIMEMIXER_CI" = "0" ] && sfx="cm"
  [ "$student" = "DLinear" ] && [ "$DLINEAR_INDIVIDUAL" = "1" ] && sfx="ind"
  local sname="$student"; [ -n "$sfx" ] && sname="${student}${sfx}"
  local model_id="${ds}_${sname}_${tag}_96_${pred}"
  local logf="$LOG_DIR/${model_id}.log"
  local donef="$LOG_DIR/${model_id}.done"
  if [ -f "$donef" ]; then echo "[gpu $gpu] SKIP: $model_id"; return 0; fi

  echo "[gpu $gpu] START: $model_id  (P$P lam=$lam)"
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path "$root_path" --data_path "$data_path" --data "$data" \
    --model_id "$model_id" --features M \
    --seq_len 96 --label_len 0 --pred_len "$pred" \
    --enc_in "$enc_in" --dec_in "$enc_in" --c_out "$enc_in" \
    --des Exp --itr 1 \
    $margs $reg_args \
    > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$donef"; echo "[gpu $gpu] DONE:  $model_id"
  else echo "[gpu $gpu] FAIL (rc=$rc): $model_id -> $logf"; fi
  return $rc
}

worker() {
  local gpu=$1; shift
  for job in "$@"; do
    IFS=':' read -r student ds pred lam <<< "$job"
    run_one "$student" "$ds" "$pred" "$lam" "$gpu"
  done
}

job_list=""
for student in $STUDENTS; do
  for ds in $DATASETS; do
    for pred in $HORIZONS; do
      for lam in $LAMBDAS; do
        job_list="$job_list ${student}:${ds}:${pred}:${lam}"
      done
    done
  done
done

gpu_arr=($GPUS); job_arr=($job_list); ngpu=${#gpu_arr[@]}
echo "Students: $STUDENTS | Datasets: $DATASETS | LAMBDAS: $LAMBDAS | render: $VTS_RENDER"
echo "Dispatching ${#job_arr[@]} jobs across $ngpu gpu(s): $GPUS ; logs -> $LOG_DIR/"

for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"; sel=""; j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do sel="$sel ${job_arr[$j]}"; j=$((j + ngpu)); done
  worker "$gpu" $sel &
done

wait
echo "All gpus finished. Collect with: python utils/collect_results.py"
