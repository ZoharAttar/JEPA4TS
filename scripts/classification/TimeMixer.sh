export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

# TimeMixer is a MULTI-SCALE model, so classification REQUIRES the down-sampling
# flags. In particular --down_sampling_method must be avg/max/conv: with the
# default (None) __multi_scale_process_inputs returns a raw tensor instead of a
# list and classification() crashes iterating over the batch dim.
DS_LAYERS="${DS_LAYERS:-3}"
DS_WINDOW="${DS_WINDOW:-2}"
DS_METHOD="${DS_METHOD:-avg}"

# NOTE: the classification head is Linear(d_model * seq_len, num_class), and UEA
# seq_len can be very long (e.g. EthanolConcentration ~1751), so keep d_model
# modest (16-32) — unlike iTransformer which can use d_model=2048.
D_MODEL="${D_MODEL:-16}"
D_FF="${D_FF:-32}"
E_LAYERS="${E_LAYERS:-3}"

# MUST be 0 for classification: TimeMixer.classification() embeds the full
# multivariate series [B,T,N] directly (it does NOT fold variables into the
# batch like the forecast path). With channel_independence=1 the embedding is
# built for 1 channel and crashes on N-channel input.
CHANNEL_INDEPENDENCE="${CHANNEL_INDEPENDENCE:-0}"

# enc_in / seq_len / num_class are auto-set from the data in exp_classification.py,
# so --enc_in below is only a placeholder (ignored/overridden).
DATASETS="${DATASETS:-EthanolConcentration FaceDetection Handwriting Heartbeat JapaneseVowels PEMS-SF SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits UWaveGestureLibrary}"

for name in $DATASETS; do
  echo "==================== TimeMixer classification: $name ===================="
  python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/$name/ \
    --model_id $name \
    --model $model_name \
    --data UEA \
    --e_layers $E_LAYERS \
    --batch_size 16 \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --down_sampling_layers $DS_LAYERS \
    --down_sampling_window $DS_WINDOW \
    --down_sampling_method $DS_METHOD \
    --channel_independence $CHANNEL_INDEPENDENCE \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --patience 10 \
    --enc_in 3
done
