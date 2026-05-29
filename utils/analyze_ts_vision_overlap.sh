# from /Users/zattar/Documents/University/Master/JEPA4TS
export CUDA_VISIBLE_DEVICES=0

# --- PatchTST | dual encoder | per-var teacher | all granularities ---
python -u analyze_ts_vision_overlap.py \
  --task_name long_term_forecast \
  --model JEPAVTS --student_model PatchTST \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 1 --d_layers 1 --factor 3 --n_heads 2 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --d_ff 256 \
  --rendering_methods RP \
  --per_var_teacher \
  --use_dual_encoder \
  --fusion_type mlp \
  --granularity all \
  --max_samples 1024 --kernel_cap 1024 \
  --batch_size 64 \
  --use_gpu --gpu 0 \
  --checkpoint ./checkpoints/<YOUR_DUAL_SETTING>/checkpoint.pth \
  --out ./analysis/patchtst_dual_etth1_96.json


# --- TimeMixer (channel_independence=1) | dual encoder | per-var teacher ---
python -u analyze_ts_vision_overlap.py \
  --task_name long_term_forecast \
  --model JEPAVTS --student_model TimeMixer \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 1 --d_layers 1 --factor 3 --n_heads 2 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --d_ff 256 \
  --channel_independence 1 \
  --down_sampling_layers 0 --down_sampling_window 1 --down_sampling_method avg \
  --decomp_method moving_avg --moving_avg 25 --use_norm 1 \
  --rendering_methods RP \
  --per_var_teacher \
  --use_dual_encoder \
  --fusion_type mlp \
  --granularity all \
  --max_samples 1024 --kernel_cap 1024 \
  --batch_size 64 \
  --use_gpu --gpu 0 \
  --checkpoint ./checkpoints/<YOUR_DUAL_SETTING>/checkpoint.pth \
  --out ./analysis/timemixer_dual_etth1_96.json
