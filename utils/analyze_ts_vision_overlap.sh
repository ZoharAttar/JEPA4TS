export CUDA_VISIBLE_DEVICES=0

# Same arch as your training + --use_dual_encoder
DUAL_COMMON="--task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model JEPAVTS \
  --student_model TimeMixer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --batch_size 16 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --per_var_teacher \
  --rendering_methods RP \
  --use_dual_encoder \
  --fusion_type mlp \
  --granularity all \
  --max_samples 1024 --kernel_cap 1024 \
  --use_gpu --gpu 0"

# One analysis run per checkpoint (one per jepa_weight value)
python -u analyze_ts_vision_overlap.py $DUAL_COMMON \
  --checkpoint ./checkpoints/long_term_forecast_ETTh2_RP_dual_jw1_JEPAVTS_ETTh2_*/checkpoint.pth \
  --out ./analysis/etth2_RP_dual_jw1.json

python -u analyze_ts_vision_overlap.py $DUAL_COMMON \
  --checkpoint ./checkpoints/long_term_forecast_ETTh2_RP_dual_jw1.5_JEPAVTS_ETTh2_*/checkpoint.pth \
  --out ./analysis/etth2_RP_dual_jw1.5.json

python -u analyze_ts_vision_overlap.py $DUAL_COMMON \
  --checkpoint ./checkpoints/long_term_forecast_ETTh2_RP_dual_jw10_JEPAVTS_ETTh2_*/checkpoint.pth \
  --out ./analysis/etth2_RP_dual_jw10.json

python -u analyze_ts_vision_overlap.py $DUAL_COMMON \
  --checkpoint ./checkpoints/long_term_forecast_ETTh2_RP_dual_jw20_JEPAVTS_ETTh2_*/checkpoint.pth \
  --out ./analysis/etth2_RP_dual_jw20.json
