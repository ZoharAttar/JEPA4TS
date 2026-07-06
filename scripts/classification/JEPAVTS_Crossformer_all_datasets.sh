#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# JEPAVTS + Crossformer student, UEA classification datasets.
# For each dataset: runs single-encoder AND dual-encoder, jepa_weight=1.
#
# Crossformer natively keeps a PER-VARIABLE axis (DSW embedding + two-stage
# attention), so — like iTransformer — its per-variable encoding aligns
# naturally with the per-variable DINO teacher (no channel-folding needed,
# unlike TimesNet-CI). JEPAVTS calls the student's classification_encode()
# (a pure behavior-preserving split of classification(), logic unchanged),
# which returns [B, N, out_seg_num, d_model]; JEPAVTS permutes it to the 4D
# convention [B, N, d_model, out_seg_num] to reuse the 4D per-var predictor /
# fusion, and decodes with the student's NATIVE classification head.
#
# PREREQUISITE — precompute per-variable DINO embeddings for each dataset first
# (uses the SAME rendering method as $RENDER below). These are STUDENT-AGNOSTIC,
# so if you already precomputed them for the TimesNet / iTransformer classification
# runs, they are reused as-is here — no need to recompute:
#
#   for d in UWaveGestureLibrary EthanolConcentration SelfRegulationSCP2 \
#            Handwriting SelfRegulationSCP1 Heartbeat PEMS-SF FaceDetection; do
#     python utils/precompute_embeddings_pervar_cls.py --dataset $d --method Spectrogram
#   done
#
# ⚠️  CACHE PATH: at training time the teacher (models/JEPAVTS.py :: VisionTSTeacher)
#     loads from:
#         {root_path}/dino_embeddings_UEA_{RENDER}_pervar
#     because configs.data == "UEA". Make sure the precomputed cache folder for
#     each dataset is named with "UEA" (not the dataset name), e.g.
#         ./dataset/Handwriting/dino_embeddings_UEA_Spectrogram_pervar
#     otherwise training fails with FileNotFoundError on the first batch.
#
# NOTE: variable-length datasets (JapaneseVowels, SpokenArabicDigits) are omitted
#       here — their padded train-time windows don't hash-match the padding-trimmed
#       precompute, so they need the teacher-side padding fix first. (Same as the
#       TimesNet / iTransformer classification scripts.)
# ═══════════════════════════════════════════════════════════════════════════

# GPUs to use: each experiment runs on ONE gpu; different experiments run in
# parallel across these gpus (one experiment per gpu at a time). Override via env:
#   GPUS="0 1 2 3" bash scripts/classification/JEPAVTS_Crossformer_all_datasets.sh
GPUS="${GPUS:-0 1 2 3}"

# Per-job logs + resume markers (a finished job is skipped on re-run).
LOG_DIR="${LOG_DIR:-logs/jepavts_crossformer_cls}"

MODEL=JEPAVTS
STUDENT=Crossformer
RENDER=Spectrogram   # rendering method (must be precomputed); GAF / RP / LinePlot / Spectrogram
JEPA_WEIGHT=1
JEPA_LOSS=mse        # JEPA alignment loss: mse | cosine
JEPA_HIDDEN=256

# Per-dataset config (portable: no associative arrays, works on bash 3.2+).
# Hyper-params are IDENTICAL to the official Crossformer classification script
# (scripts/classification/Crossformer.sh): e_layers=3, d_model=128, d_ff=256,
# top_k=3 for EVERY dataset; batch_size=16, lr=0.001, train_epochs=100,
# patience=10 are constant. factor/n_heads use run.py defaults (factor=1,
# n_heads=8 — d_model=128 is divisible by 8), exactly as in the official script.
# top_k is passed for parity but Crossformer ignores it. NOTE: --enc_in is
# intentionally NOT passed — exp_jepa_vts_classification overwrites it with the
# real variable count from the data before building the model.
# Field order:
#   root_path|e_layers|d_model|d_ff|top_k|epochs
get_cfg() {
  case "$1" in
    UWaveGestureLibrary)  echo "./dataset/UWaveGestureLibrary/|3|128|256|3|100" ;;
    EthanolConcentration) echo "./dataset/EthanolConcentration/|3|128|256|3|100" ;;
    SelfRegulationSCP2)   echo "./dataset/SelfRegulationSCP2/|3|128|256|3|100" ;;
    Handwriting)          echo "./dataset/Handwriting/|3|128|256|3|100" ;;
    SelfRegulationSCP1)   echo "./dataset/SelfRegulationSCP1/|3|128|256|3|100" ;;
    Heartbeat)            echo "./dataset/Heartbeat/|3|128|256|3|100" ;;
    PEMS-SF)              echo "./dataset/PEMS-SF/|3|128|256|3|100" ;;
    FaceDetection)        echo "./dataset/FaceDetection/|3|128|256|3|100" ;;
    # Variable-length (in the official script but omitted from the default run
    # below — the teacher's padded-window hashing doesn't match the precompute
    # yet; add them to DATASETS once the teacher-side padding fix is in):
    JapaneseVowels)       echo "./dataset/JapaneseVowels/|3|128|256|3|100" ;;
    SpokenArabicDigits)   echo "./dataset/SpokenArabicDigits/|3|128|256|3|100" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

# Order of datasets to run (smallest precompute/compute first).
DATASETS="${DATASETS:-UWaveGestureLibrary EthanolConcentration SelfRegulationSCP2 Handwriting SelfRegulationSCP1 Heartbeat PEMS-SF FaceDetection}"

run_one() {
  # args: dataset_name enc_setting(single|dual) gpu_id
  local name=$1 enc=$2 gpu=$3
  IFS='|' read -r root_path e_layers d_model d_ff top_k epochs <<< "$(get_cfg "$name")"

  local extra=""
  local tag="single"
  if [ "$enc" = "dual" ]; then
    extra="--use_dual_encoder --fusion_type mlp"
    tag="dual"
  fi

  # job tag: used only for per-job logs/markers (distinguishes single vs dual).
  local job="${name}_${RENDER}_${tag}"
  local logf="$LOG_DIR/${job}.log"
  local donef="$LOG_DIR/${job}.done"

  if [ -f "$donef" ]; then
    echo "[gpu $gpu] SKIP (already done): $job"
    return 0
  fi

  echo "[gpu $gpu] START: $job"
  # IMPORTANT: --model_id must be the bare dataset name — the UEA data loader
  # uses it to locate <name>_TRAIN.ts / <name>_TEST.ts. single vs dual is kept
  # distinct via --des (which is part of run.py's setting string).
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path "$root_path" \
    --model_id "$name" \
    --model $MODEL \
    --student_model $STUDENT \
    --data UEA \
    --e_layers "$e_layers" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --top_k "$top_k" \
    --batch_size 16 \
    --learning_rate 0.001 \
    --train_epochs "$epochs" \
    --patience 10 \
    --des "${tag}_${RENDER}" \
    --itr 1 \
    --per_var_teacher \
    --rendering_methods $RENDER \
    --jepa_weight $JEPA_WEIGHT \
    --jepa_loss_type $JEPA_LOSS \
    --jepa_hidden_dim $JEPA_HIDDEN \
    $extra \
    > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$donef"
    echo "[gpu $gpu] DONE:  $job"
  else
    echo "[gpu $gpu] FAIL (rc=$rc): $job  ->  see $logf"
  fi
  return $rc
}

# Worker: process a queue of jobs sequentially on a single gpu.
worker() {
  local gpu=$1; shift
  for job in "$@"; do
    IFS=':' read -r name enc <<< "$job"
    run_one "$name" "$enc" "$gpu"
  done
}

mkdir -p "$LOG_DIR"

# Build the full job list (each entry: name:enc).
job_list=""
for name in $DATASETS; do
  get_cfg "$name" >/dev/null || exit 1
  job_list="$job_list ${name}:single ${name}:dual"
done

gpu_arr=($GPUS)
job_arr=($job_list)
ngpu=${#gpu_arr[@]}
echo "Dispatching ${#job_arr[@]} experiments across $ngpu gpu(s): $GPUS"
echo "Logs: $LOG_DIR/<job>.log"

# Distribute jobs round-robin across gpus, then launch one worker per gpu.
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
echo "All gpus finished."
