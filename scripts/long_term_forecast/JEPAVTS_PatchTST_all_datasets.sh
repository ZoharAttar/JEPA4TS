#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# JEPAVTS (VibeTS) + PatchTST student, full (in-distribution) forecasting.
# Default: SINGLE encoder, all ETT datasets, all horizons, jepa_weight=1.
#
#   ENC=single|dual|both   which encoder setting(s) to run   (default: single)
#   DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate"  (default: ETT only)
#   GPUS="0 1 2 3"         gpus to spread jobs across (one job per gpu at a time)
#   RENDER=RP             rendering method (must be precomputed)
#
# PatchTST ETT is tuned PER HORIZON (e_layers/n_heads/batch differ by pred_len),
# matching scripts/long_term_forecast/ETT_script/PatchTST_ETT*.sh. d_model/d_ff use
# run.py defaults (512/2048), same as the Exchange/Weather PatchTST baselines.
#
# PREREQUISITE — precompute per-variable DINO embeddings for every dataset first
# (same rendering method as $RENDER below):
#
#   for d in ETTh1 ETTh2 ETTm1 ETTm2; do
#     python utils/precompute_embeddings_pervar.py --dataset $d --method RP
#   done
#
# The teacher loads embeddings from:
#   {root_path}/dino_embeddings_{data}_{RENDER}_pervar
#
# Usage (this request — VibeTS + PatchTST single encoder, all ETT):
#   bash scripts/long_term_forecast/JEPAVTS_PatchTST_all_datasets.sh
# ═══════════════════════════════════════════════════════════════════════════

GPUS="${GPUS:-0 1 2 3}"
LOG_DIR="${LOG_DIR:-logs/jepavts_patchtst}"
ENC="${ENC:-single}"          # single | dual | both
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2}"
HORIZONS="${HORIZONS:-96 192 336 720}"

MODEL=JEPAVTS
STUDENT=PatchTST
RENDER="${RENDER:-RP}"       # rendering method (must be precomputed): RP / GAF / LinePlot / Spectrogram
JEPA_WEIGHT="${JEPA_WEIGHT:-1}"
JEPA_LOSS="${JEPA_LOSS:-mse}"  # JEPA alignment loss: mse | cosine

# Per-(dataset, pred_len) config. PatchTST ETT tuning is per horizon.
# Field order:
#   data|root_path|data_path|enc_in|e_layers|d_model|d_ff|n_heads|factor|batch|lr|epochs|patience|dropout|seq_len
get_cfg() {
  local ds=$1 pred=$2
  case "$ds" in
    ETTh1|ETTh2|ETTm1|ETTm2)
      local el nh bt
      case "${ds}_${pred}" in
        ETTh1_96)  el=1; nh=2;  bt=32  ;;
        ETTh1_192) el=1; nh=8;  bt=32  ;;
        ETTh1_336) el=1; nh=8;  bt=32  ;;
        ETTh1_720) el=1; nh=16; bt=32  ;;
        ETTh2_96|ETTh2_192|ETTh2_336|ETTh2_720) el=3; nh=4; bt=32 ;;
        ETTm1_96)  el=1; nh=2;  bt=32  ;;
        ETTm1_192) el=3; nh=2;  bt=128 ;;
        ETTm1_336) el=1; nh=4;  bt=128 ;;
        ETTm1_720) el=3; nh=4;  bt=128 ;;
        ETTm2_96)  el=3; nh=16; bt=32  ;;
        ETTm2_192) el=3; nh=2;  bt=128 ;;
        ETTm2_336) el=1; nh=4;  bt=32  ;;
        ETTm2_720) el=3; nh=4;  bt=128 ;;
        *) echo "UNKNOWN PatchTST ETT (ds=$ds pred=$pred); HORIZONS must be in {96,192,336,720}" >&2; return 1 ;;
      esac
      echo "${ds}|./dataset/ETT-small/|${ds}.csv|7|${el}|512|2048|${nh}|3|${bt}|0.0001|10|3|0.1|96" ;;
    weather)       echo "custom|./dataset/weather/|weather.csv|21|2|512|2048|4|3|32|0.0001|3|3|0.1|96" ;;
    exchange_rate) echo "custom|./dataset/exchange_rate/|exchange_rate.csv|8|2|512|2048|8|3|32|0.0001|10|3|0.1|96" ;;
    *) echo "UNKNOWN dataset: $ds" >&2; return 1 ;;
  esac
}

run_one() {
  # args: dataset_name pred_len enc_setting(single|dual) gpu_id
  local name=$1 pred_len=$2 enc=$3 gpu=$4
  IFS='|' read -r data root_path data_path enc_in e_layers d_model d_ff n_heads factor batch_size lr epochs patience dropout seq_len <<< "$(get_cfg "$name" "$pred_len")" || return 1

  local extra=""
  local tag="single"
  if [ "$enc" = "dual" ]; then
    extra="--use_dual_encoder --fusion_type mlp"
    tag="dual"
  fi

  local model_id="${name}_${RENDER}_${seq_len}_${pred_len}_${tag}"
  local logf="$LOG_DIR/${model_id}.log"
  local donef="$LOG_DIR/${model_id}.done"

  if [ -f "$donef" ]; then
    echo "[gpu $gpu] SKIP (already done): $model_id"
    return 0
  fi

  echo "[gpu $gpu] START: $model_id  (el$e_layers nh$n_heads b$batch_size)"
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model $MODEL \
    --student_model $STUDENT \
    --data "$data" \
    --features M \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$enc_in" \
    --c_out "$enc_in" \
    --factor "$factor" \
    --n_heads "$n_heads" \
    --des Exp \
    --itr 1 \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --learning_rate "$lr" \
    --train_epochs "$epochs" \
    --patience "$patience" \
    --batch_size "$batch_size" \
    --dropout "$dropout" \
    --per_var_teacher \
    --rendering_methods $RENDER \
    --jepa_weight $JEPA_WEIGHT \
    --jepa_loss_type $JEPA_LOSS \
    $extra \
    > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$donef"
    echo "[gpu $gpu] DONE:  $model_id"
  else
    echo "[gpu $gpu] FAIL (rc=$rc): $model_id  ->  see $logf"
  fi
  return $rc
}

worker() {
  local gpu=$1; shift
  for job in "$@"; do
    IFS=':' read -r name pred_len enc <<< "$job"
    run_one "$name" "$pred_len" "$enc" "$gpu"
  done
}

mkdir -p "$LOG_DIR"

# Which encoder settings to run.
case "$ENC" in
  single) ENC_SETTINGS="single" ;;
  dual)   ENC_SETTINGS="dual" ;;
  both)   ENC_SETTINGS="single dual" ;;
  *) echo "Invalid ENC='$ENC' (use single|dual|both)" >&2; exit 1 ;;
esac

# Build the full job list (each entry: name:pred_len:enc).
job_list=""
for name in $DATASETS; do
  for pred_len in $HORIZONS; do
    for enc in $ENC_SETTINGS; do
      job_list="$job_list ${name}:${pred_len}:${enc}"
    done
  done
done

gpu_arr=($GPUS)
job_arr=($job_list)
ngpu=${#gpu_arr[@]}
echo "Backbone: $STUDENT   ENC: $ENC   Datasets: $DATASETS"
echo "Dispatching ${#job_arr[@]} experiments across $ngpu gpu(s): $GPUS"
echo "Logs: $LOG_DIR/<model_id>.log"

for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"
  sel=""
  j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do
    sel="$sel ${job_arr[$j]}"
    j=$((j + ngpu))
  done
  nsel=$(printf '%s\n' $sel | grep -c . || true)
  echo "  gpu $gpu  <-  $nsel experiments"
  worker "$gpu" $sel &
done

wait
echo "All gpus finished. Collect results with: python utils/collect_results.py"
