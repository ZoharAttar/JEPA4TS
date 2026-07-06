export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

# ═══════════════════════════════════════════════════════════════════════════
# TimeMixer baseline, UEA classification (per-dataset config).
#
# There is NO official TimeMixer UEA classification config (the repo targets
# forecasting), so this table is our own. Two entries (JapaneseVowels,
# SelfRegulationSCP1) are tuned/confirmed-better; the rest are sensible starting
# points chosen by sequence length (short seq -> fewer down_sampling layers).
#
# TimeMixer specifics for classification:
#   * --down_sampling_method MUST be avg/max/conv (None makes classification()
#     iterate over the batch dim and crash).
#   * --channel_independence 0 is REQUIRED: classification() embeds the full
#     [B,T,N] series (no B*N fold), so the embedding must be built for enc_in
#     channels. With CI=1 it is built for 1 channel and crashes on N-channel input.
#   * classification head is Linear(d_model * seq_len, num_class); UEA seq_len can
#     be long, so keep d_model modest on long-sequence datasets.
#
# enc_in / seq_len / num_class are auto-set from the data in exp_classification.py.
# ═══════════════════════════════════════════════════════════════════════════

# Constant across datasets.
DS_METHOD="${DS_METHOD:-avg}"
CHANNEL_INDEPENDENCE="${CHANNEL_INDEPENDENCE:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.001}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"

# Per-dataset config (portable: no associative arrays, bash 3.2+).
# Field order:
#   root_path|e_layers|d_model|d_ff|ds_layers|ds_window
#   (^) JapaneseVowels & SelfRegulationSCP1 are tuned; others are starting points.
get_cfg() {
  case "$1" in
    JapaneseVowels)       echo "./dataset/JapaneseVowels/|3|64|128|1|2" ;;   # tuned: short seq, less downsampling + more capacity
    SelfRegulationSCP1)   echo "./dataset/SelfRegulationSCP1/|4|32|64|3|2" ;; # tuned: deeper, mid capacity
    EthanolConcentration) echo "./dataset/EthanolConcentration/|3|32|64|3|2" ;; # very long seq (~1751)
    FaceDetection)        echo "./dataset/FaceDetection/|2|64|128|2|2" ;;     # short seq (~62), many vars
    Handwriting)          echo "./dataset/Handwriting/|3|32|64|3|2" ;;
    Heartbeat)            echo "./dataset/Heartbeat/|3|32|64|3|2" ;;
    PEMS-SF)              echo "./dataset/PEMS-SF/|3|32|64|3|2" ;;            # 963 vars; keep d_model modest
    SelfRegulationSCP2)   echo "./dataset/SelfRegulationSCP2/|3|32|64|3|2" ;;
    SpokenArabicDigits)   echo "./dataset/SpokenArabicDigits/|2|32|64|2|2" ;; # short seq (~93)
    UWaveGestureLibrary)  echo "./dataset/UWaveGestureLibrary/|3|32|64|3|2" ;;
    *) echo "UNKNOWN dataset: $1" >&2; return 1 ;;
  esac
}

DATASETS="${DATASETS:-EthanolConcentration FaceDetection Handwriting Heartbeat JapaneseVowels PEMS-SF SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits UWaveGestureLibrary}"

for name in $DATASETS; do
  IFS='|' read -r root_path e_layers d_model d_ff ds_layers ds_window <<< "$(get_cfg "$name")" || exit 1
  echo "==================== TimeMixer classification: $name (dm$d_model df$d_ff el$e_layers ds$ds_layers) ===================="
  python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path "$root_path" \
    --model_id "$name" \
    --model $model_name \
    --data UEA \
    --e_layers "$e_layers" \
    --batch_size "$BATCH_SIZE" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --down_sampling_layers "$ds_layers" \
    --down_sampling_window "$ds_window" \
    --down_sampling_method "$DS_METHOD" \
    --channel_independence "$CHANNEL_INDEPENDENCE" \
    --des 'Exp' \
    --itr 1 \
    --learning_rate "$LR" \
    --train_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --enc_in 3
done
