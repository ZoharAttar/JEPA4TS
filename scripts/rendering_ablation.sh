#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# TABLE 6 — Rendering / plotting-strategy ablation (iTransformer backbone).
#
# Variants (columns of the "Plotting Method" table):
#   baseline   No Plotting  -> standalone iTransformer, NO visual distillation
#   RP         Recurrence Plot           -> JEPAVTS + iTransformer, teacher=RP
#   GAF        Gramian Angular Field     -> JEPAVTS + iTransformer, teacher=GAF
#   LinePlot   Line Plot                 -> JEPAVTS + iTransformer, teacher=LinePlot
#   Spectrogram Spectrogram              -> JEPAVTS + iTransformer, teacher=Spectrogram
#   combined   Combined/Ensemble         -> JEPAVTS + iTransformer, multi-rendering
#                                            (RP+GAF+LinePlot+Spectrogram, shared enc+predictor)
#
# Tasks:
#   forecast  -> Exchange-rate, horizons {96,192,336,720}   (MSE / MAE)
#   classify  -> SelfRegulationSCP1                         (Accuracy)
#
# Usage (defaults shown):
#   bash scripts/ablation/rendering_ablation.sh                       # 8-gpu parallel
#   GPUS="0 1 2 3" bash scripts/ablation/rendering_ablation.sh
#   TASK=classify GPUS="0 1" bash scripts/ablation/rendering_ablation.sh
#   VARIANTS="baseline RP" TASK=forecast bash scripts/ablation/rendering_ablation.sh
#
# Env overrides:
#   GPUS="0 1 2 3 4 5 6 7"            gpus to spread jobs across (round-robin)
#   JOBS_PER_GPU=1                    concurrent jobs per gpu (raise for tiny forecast jobs)
#   TASK="both"                       forecast | classify | both
#   VARIANTS="baseline RP GAF LinePlot combined"
#   COMBINED_METHODS="RP GAF LinePlot"
#   ENC="single"                      single | dual  (JEPAVTS variants only)
#   PRED_LENS="96 192 336 720"        forecast horizons
#
# ── PREREQUISITE: per-variable DINO embeddings must exist for every rendering
#    you run (baseline needs none). The teacher loads from
#      {root_path}/dino_embeddings_{configs.data}_{METHOD}_pervar
#    so the folder names must be:
#      Forecast (data=custom):  ./dataset/exchange_rate/dino_embeddings_custom_<METHOD>_pervar
#      Classify (data=UEA):     ./dataset/SelfRegulationSCP1/dino_embeddings_UEA_<METHOD>_pervar
#    (This is the SAME naming your working RP runs already use.) Precompute with:
#      python utils/precompute_embeddings_pervar.py     --dataset exchange_rate    --method GAF
#      python utils/precompute_embeddings_pervar_cls.py --dataset SelfRegulationSCP1 --method GAF
#    for METHOD in {RP, GAF, LinePlot, Spectrogram}. If the produced folder is named with the
#    dataset (e.g. ..._exchange_rate_GAF_pervar) instead of custom/UEA, rename or
#    symlink it to match, exactly as you did for RP.
# ═══════════════════════════════════════════════════════════════════════════
set -u

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"     # gpus to spread jobs across (round-robin)
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"   # concurrent jobs per gpu (bump for tiny forecast jobs)
TASK="${TASK:-both}"
VARIANTS="${VARIANTS:-baseline RP GAF LinePlot Spectrogram combined}"
COMBINED_METHODS="${COMBINED_METHODS:-RP GAF LinePlot Spectrogram}"
ENC="${ENC:-single}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
LOG_DIR="${LOG_DIR:-logs/rendering_ablation}"
mkdir -p "$LOG_DIR"

STUDENT=iTransformer
JEPA_WEIGHT=1
JEPA_LOSS=mse

enc_flags=""; enc_tag="single"
[ "$ENC" = "dual" ] && { enc_flags="--use_dual_encoder --fusion_type mlp"; enc_tag="dual"; }

# Rendering args for a JEPAVTS variant.
jepa_render_flags() {
  case "$1" in
    combined) echo "--rendering_methods $COMBINED_METHODS --multi_rendering_alpha_mode same" ;;
    *)        echo "--rendering_methods $1" ;;
  esac
}

# ── Forecast config (Exchange, iTransformer — matches
#    scripts/long_term_forecast/JEPAVTS_iTransformer_all_datasets.sh) ────────
F_ROOT=./dataset/exchange_rate/ ; F_DPATH=exchange_rate.csv ; F_DATA=custom
F_ENC_IN=8 ; F_ELAYERS=2 ; F_DMODEL=128 ; F_DFF=128 ; F_FACTOR=3
F_BATCH=32 ; F_LR=0.0001 ; F_EPOCHS=10 ; F_PAT=3 ; F_DROP=0.1 ; F_SEQ=96

run_forecast() {
  local gpu=$1 variant=$2 pred=$3 mid
  if [ "$variant" = "baseline" ]; then mid="exchange_rate_baseline_${F_SEQ}_${pred}"
  else mid="exchange_rate_${variant}_${enc_tag}_${F_SEQ}_${pred}"; fi
  local logf="$LOG_DIR/fc_${variant}_${enc_tag}_pl${pred}.log"
  local donef="$LOG_DIR/fc_${variant}_${enc_tag}_pl${pred}.done"
  [ -f "$donef" ] && { echo "[gpu $gpu][fc] SKIP $mid"; return 0; }

  local common="--task_name long_term_forecast --is_training 1 \
    --root_path $F_ROOT --data_path $F_DPATH --data $F_DATA \
    --features M --seq_len $F_SEQ --label_len 0 --pred_len $pred \
    --e_layers $F_ELAYERS --enc_in $F_ENC_IN --c_out $F_ENC_IN --factor $F_FACTOR \
    --des Exp --itr 1 --d_model $F_DMODEL --d_ff $F_DFF \
    --learning_rate $F_LR --train_epochs $F_EPOCHS --patience $F_PAT \
    --batch_size $F_BATCH --dropout $F_DROP --model_id $mid"

  echo "[gpu $gpu][fc] START $mid"
  if [ "$variant" = "baseline" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -u run.py $common --model iTransformer > "$logf" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu python -u run.py $common \
      --model JEPAVTS --student_model $STUDENT --per_var_teacher \
      $(jepa_render_flags "$variant") --jepa_weight $JEPA_WEIGHT --jepa_loss_type $JEPA_LOSS \
      $enc_flags > "$logf" 2>&1
  fi
  [ $? -eq 0 ] && { touch "$donef"; echo "[gpu $gpu][fc] DONE $mid"; } || echo "[gpu $gpu][fc] FAIL $mid -> $logf"
}

# ── Classify config (SelfRegulationSCP1, iTransformer — matches
#    scripts/classification/iTransformer.sh / JEPAVTS_iTransformer_all_datasets.sh) ─
C_ROOT=./dataset/SelfRegulationSCP1/ ; DATASET_CLS=SelfRegulationSCP1
C_ELAYERS=3 ; C_DMODEL=128 ; C_DFF=256 ; C_TOPK=3
C_BATCH=16 ; C_LR=0.001 ; C_EPOCHS=100 ; C_PAT=10

run_classify() {
  local gpu=$1 variant=$2 des
  if [ "$variant" = "baseline" ]; then des="baseline"; else des="${variant}_${enc_tag}"; fi
  local logf="$LOG_DIR/cls_${variant}_${enc_tag}.log"
  local donef="$LOG_DIR/cls_${variant}_${enc_tag}.done"
  [ -f "$donef" ] && { echo "[gpu $gpu][cls] SKIP $DATASET_CLS $des"; return 0; }

  # model_id MUST be the bare dataset name (UEA loader locates the .ts files);
  # variants are kept distinct via --des (part of run.py's setting string).
  local common="--task_name classification --is_training 1 \
    --root_path $C_ROOT --model_id $DATASET_CLS --data UEA \
    --e_layers $C_ELAYERS --d_model $C_DMODEL --d_ff $C_DFF --top_k $C_TOPK \
    --batch_size $C_BATCH --learning_rate $C_LR --train_epochs $C_EPOCHS --patience $C_PAT \
    --des $des --itr 1"

  echo "[gpu $gpu][cls] START $DATASET_CLS $des"
  if [ "$variant" = "baseline" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -u run.py $common --model iTransformer > "$logf" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu python -u run.py $common \
      --model JEPAVTS --student_model $STUDENT --per_var_teacher \
      $(jepa_render_flags "$variant") --jepa_weight $JEPA_WEIGHT --jepa_loss_type $JEPA_LOSS \
      --jepa_hidden_dim 256 $enc_flags > "$logf" 2>&1
  fi
  [ $? -eq 0 ] && { touch "$donef"; echo "[gpu $gpu][cls] DONE $des"; } || echo "[gpu $gpu][cls] FAIL $des -> $logf"
}

# Execute a single job token "fc:<variant>:<pred>" or "cls:<variant>:-" on a gpu.
run_token() {
  local gpu=$1 token=$2 kind variant arg
  IFS=':' read -r kind variant arg <<< "$token"
  case "$kind" in
    fc)  run_forecast "$gpu" "$variant" "$arg" ;;
    cls) run_classify "$gpu" "$variant" ;;
  esac
}

# Worker: run its assigned tokens on one gpu, up to JOBS_PER_GPU concurrently.
worker() {
  local gpu=$1; shift
  local running=0
  for token in "$@"; do
    if [ "$JOBS_PER_GPU" -le 1 ]; then
      run_token "$gpu" "$token"
    else
      run_token "$gpu" "$token" &
      running=$((running + 1))
      if [ "$running" -ge "$JOBS_PER_GPU" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
    fi
  done
  wait
}

# ── Build the job list, then dispatch across gpus in parallel ─────────────
job_list=""
if [ "$TASK" = "forecast" ] || [ "$TASK" = "both" ]; then
  for v in $VARIANTS; do for p in $PRED_LENS; do job_list="$job_list fc:${v}:${p}"; done; done
fi
if [ "$TASK" = "classify" ] || [ "$TASK" = "both" ]; then
  for v in $VARIANTS; do job_list="$job_list cls:${v}:-"; done
fi

gpu_arr=($GPUS)
job_arr=($job_list)
ngpu=${#gpu_arr[@]}
echo "══════════ Dispatching ${#job_arr[@]} jobs across $ngpu gpu(s): $GPUS (${JOBS_PER_GPU}/gpu), enc=$enc_tag ══════════"

# Round-robin assign jobs to gpus, then launch one background worker per gpu.
for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"
  sel=""
  j=$idx
  while [ $j -lt ${#job_arr[@]} ]; do
    sel="$sel ${job_arr[$j]}"
    j=$((j + ngpu))
  done
  [ -z "$sel" ] && continue
  nsel=$(printf '%s\n' $sel | grep -c . || true)
  echo "  gpu $gpu  <-  $nsel jobs"
  worker "$gpu" $sel &
done
wait
echo "All gpus finished."

# ── Summary ─────────────────────────────────────────────────────────────────
fc_metric() {  # $1=logfile $2=variant -> "MSE MAE"
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

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " SUMMARY  (Table 6)"
echo "══════════════════════════════════════════════════════════════════"

if [ "$TASK" = "forecast" ] || [ "$TASK" = "both" ]; then
  echo ""
  echo "### Exchange forecasting  (MSE / MAE)"
  hdr="Plotting Method"
  for p in $PRED_LENS; do hdr="$hdr | pl$p"; done
  hdr="$hdr | Avg"
  echo "$hdr"
  for v in $VARIANTS; do
    row="$v"
    sum_mse=0; sum_mae=0; n=0
    for p in $PRED_LENS; do
      read -r mse mae <<< "$(fc_metric "$LOG_DIR/fc_${v}_${enc_tag}_pl${p}.log" "$v")"
      row="$row | ${mse}/${mae}"
      if [ "$mse" != "NA" ]; then
        sum_mse=$(awk "BEGIN{print $sum_mse+$mse}")
        sum_mae=$(awk "BEGIN{print $sum_mae+$mae}")
        n=$((n+1))
      fi
    done
    if [ $n -gt 0 ]; then
      avg_mse=$(awk "BEGIN{printf \"%.4f\", $sum_mse/$n}")
      avg_mae=$(awk "BEGIN{printf \"%.4f\", $sum_mae/$n}")
      row="$row | ${avg_mse}/${avg_mae}"
    else
      row="$row | NA"
    fi
    echo "$row"
  done
fi

if [ "$TASK" = "classify" ] || [ "$TASK" = "both" ]; then
  echo ""
  echo "### SelfRegulationSCP1 classification  (Accuracy %)"
  echo "Plotting Method | Accuracy"
  for v in $VARIANTS; do
    f="$LOG_DIR/cls_${v}_${enc_tag}.log"
    acc="NA"
    if [ -f "$f" ]; then
      acc=$(grep -E "accuracy:" "$f" | tail -1 | sed -E 's/.*accuracy:[[:space:]]*([0-9.eE+-]+).*/\1/')
      [ -n "$acc" ] && acc=$(awk "BEGIN{printf \"%.2f\", $acc*100}")
    fi
    echo "$v | ${acc:-NA}"
  done
fi

echo ""
echo "Per-run logs: $LOG_DIR/{fc,cls}_<variant>_<enc>*.log"
