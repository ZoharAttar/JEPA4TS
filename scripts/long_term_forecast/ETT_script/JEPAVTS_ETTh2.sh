export CUDA_VISIBLE_DEVICES=0

# ═══════════════════════════════════════════════════════════
# Common config (matches TimeMixer_ETTh1.sh / ETTh2)
# ═══════════════════════════════════════════════════════════
COMMON="--task_name long_term_forecast \
  --is_training 1 \
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
  --des Exp \
  --itr 1 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --train_epochs 10 \
  --patience 10 \
  --batch_size 128 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --per_var_teacher"

# ═══════════════════════════════════════════════════════════
# TABLE 1: JEPAVTS + TimeMixer + RP + horizon96
# ═══════════════════════════════════════════════════════════

# --- Single encoder, varying jepa_weight ---
python -u run.py $COMMON --model_id ETTh2_RP_single_jw1    --rendering_methods RP --jepa_weight 1
python -u run.py $COMMON --model_id ETTh2_RP_single_jw1.5  --rendering_methods RP --jepa_weight 1.5
python -u run.py $COMMON --model_id ETTh2_RP_single_jw10   --rendering_methods RP --jepa_weight 10
python -u run.py $COMMON --model_id ETTh2_RP_single_jw20   --rendering_methods RP --jepa_weight 20

# --- Dual encoder MLP, varying jepa_weight ---
python -u run.py $COMMON --model_id ETTh2_RP_dual_mlp_jw1    --rendering_methods RP --jepa_weight 1   --use_dual_encoder --fusion_type mlp
python -u run.py $COMMON --model_id ETTh2_RP_dual_mlp_jw1.5  --rendering_methods RP --jepa_weight 1.5 --use_dual_encoder --fusion_type mlp
python -u run.py $COMMON --model_id ETTh2_RP_dual_mlp_jw10   --rendering_methods RP --jepa_weight 10  --use_dual_encoder --fusion_type mlp
python -u run.py $COMMON --model_id ETTh2_RP_dual_mlp_jw20   --rendering_methods RP --jepa_weight 20  --use_dual_encoder --fusion_type mlp

# --- Dual encoder, varying fusion type (jepa_weight=1) ---
python -u run.py $COMMON --model_id ETTh2_RP_dual_weighted_jw1    --rendering_methods RP --jepa_weight 1 --use_dual_encoder --fusion_type weighted
python -u run.py $COMMON --model_id ETTh2_RP_dual_add_jw1         --rendering_methods RP --jepa_weight 1 --use_dual_encoder --fusion_type add
python -u run.py $COMMON --model_id ETTh2_RP_dual_transformer_jw1 --rendering_methods RP --jepa_weight 1 --use_dual_encoder --fusion_type transformer

# --- Learned loss, single encoder ---
python -u run.py $COMMON --model_id ETTh2_RP_single_learned --rendering_methods RP --learned_loss_weights

# --- Learned loss, dual encoder, varying fusion type ---
python -u run.py $COMMON --model_id ETTh2_RP_dual_mlp_learned         --rendering_methods RP --learned_loss_weights --use_dual_encoder --fusion_type mlp
python -u run.py $COMMON --model_id ETTh2_RP_dual_weighted_learned     --rendering_methods RP --learned_loss_weights --use_dual_encoder --fusion_type weighted
python -u run.py $COMMON --model_id ETTh2_RP_dual_add_learned          --rendering_methods RP --learned_loss_weights --use_dual_encoder --fusion_type add
python -u run.py $COMMON --model_id ETTh2_RP_dual_transformer_learned  --rendering_methods RP --learned_loss_weights --use_dual_encoder --fusion_type transformer

# ═══════════════════════════════════════════════════════════
# TABLE 2: JEPAVTS + TimeMixer + GAF + horizon96
# ═══════════════════════════════════════════════════════════

# --- Single encoder ---
python -u run.py $COMMON --model_id ETTh2_GAF_single_jw1   --rendering_methods GAF --jepa_weight 1
python -u run.py $COMMON --model_id ETTh2_GAF_single_jw1.5 --rendering_methods GAF --jepa_weight 1.5

# --- Dual encoder MLP ---
python -u run.py $COMMON --model_id ETTh2_GAF_dual_mlp_jw1   --rendering_methods GAF --jepa_weight 1   --use_dual_encoder --fusion_type mlp
python -u run.py $COMMON --model_id ETTh2_GAF_dual_mlp_jw1.5 --rendering_methods GAF --jepa_weight 1.5 --use_dual_encoder --fusion_type mlp

# ═══════════════════════════════════════════════════════════
# TABLE 3: JEPAVTS + TimeMixer + GAF+RP + horizon96 (jepa_weight=1)
# ═══════════════════════════════════════════════════════════

# --- 1-all share (shared encoder, shared predictor) ---
# same alpha
python -u run.py $COMMON --model_id ETTh2_GAFRP_allshare_same    --rendering_methods RP GAF --jepa_weight 1 --multi_rendering_alpha_mode same
# divide alpha
python -u run.py $COMMON --model_id ETTh2_GAFRP_allshare_divided --rendering_methods RP GAF --jepa_weight 1 --multi_rendering_alpha_mode divided

# --- 2-shared encoder, separate predictor ---
python -u run.py $COMMON --model_id ETTh2_GAFRP_shared_enc_sep_pred --rendering_methods RP GAF --jepa_weight 1 --multi_predictor

# --- 3-all separate (separate encoder, separate predictor) ---
python -u run.py $COMMON --model_id ETTh2_GAFRP_all_separate --rendering_methods RP GAF --jepa_weight 1 --multi_encoder --multi_predictor

# --- 4-shared predictor (separate encoder, shared predictor) ---
python -u run.py $COMMON --model_id ETTh2_GAFRP_sep_enc_shared_pred --rendering_methods RP GAF --jepa_weight 1 --multi_encoder

# --- Same 5 configs but with dual encoder MLP ---
# 1-all share, same alpha + dual
python -u run.py $COMMON --model_id ETTh2_GAFRP_allshare_same_dual    --rendering_methods RP GAF --jepa_weight 1 --multi_rendering_alpha_mode same    --use_dual_encoder --fusion_type mlp
# 1-all share, divide alpha + dual
python -u run.py $COMMON --model_id ETTh2_GAFRP_allshare_divided_dual --rendering_methods RP GAF --jepa_weight 1 --multi_rendering_alpha_mode divided --use_dual_encoder --fusion_type mlp
# 2-shared encoder, separate predictor + dual
python -u run.py $COMMON --model_id ETTh2_GAFRP_shared_enc_sep_pred_dual --rendering_methods RP GAF --jepa_weight 1 --multi_predictor --use_dual_encoder --fusion_type mlp
# 3-all separate + dual
python -u run.py $COMMON --model_id ETTh2_GAFRP_all_separate_dual --rendering_methods RP GAF --jepa_weight 1 --multi_encoder --multi_predictor --use_dual_encoder --fusion_type mlp
# 4-shared predictor + dual
python -u run.py $COMMON --model_id ETTh2_GAFRP_sep_enc_shared_pred_dual --rendering_methods RP GAF --jepa_weight 1 --multi_encoder --use_dual_encoder --fusion_type mlp
