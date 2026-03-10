import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from utils.plotting import visionTS_plot, transform_RP_batch
from transformers import ViTModel
import os
import hashlib
import numpy as np

class VisionTSTeacher(nn.Module):
    """
    Loads pre-computed DINO embeddings from cache.
    Works with ANY batch size since each sample is cached individually by hash.
    
    Args:
        dataset_name: e.g. "ETTh2"
        hidden_size: DINO embedding dimension (768 for dinov2-base)
        rendering_method: e.g. "RP", "GAF". When set, cache_dir becomes
                          dino_embeddings_{dataset_name}_{rendering_method}.
                          When None, uses legacy path dino_embeddings_{dataset_name}.
    """
    
    def __init__(self, dataset_name="ETTh1", hidden_size=768, rendering_method=None):
        super().__init__()
        self.rendering_method = rendering_method
        if rendering_method:
            self.cache_dir = f"./dataset/ETT-small/dino_embeddings_{dataset_name}_{rendering_method}"
        else:
            self.cache_dir = f"./dataset/ETT-small/dino_embeddings_{dataset_name}"
        self.hidden_size = hidden_size
        
        if not os.path.exists(self.cache_dir):
            raise ValueError(f"❌ Cache not found: {self.cache_dir}\n   Run precompute_embeddings.py first!")
        
        n_cached = len([f for f in os.listdir(self.cache_dir) if f.endswith('.npy')])
        tag = f" [{rendering_method}]" if rendering_method else ""
        print(f"✅ VisionTSTeacher{tag}: Loading from cache")
        print(f"✅ Cache dir: {self.cache_dir}")
        print(f"✅ Cached embeddings: {n_cached}")
        print(f"✅ Teacher hidden_size: {self.hidden_size}")
    
    def _get_hash(self, ts_array):
        return hashlib.md5(ts_array.astype(np.float32).tobytes()).hexdigest()[:16]
    
    def forward(self, x_enc):
        """
        x_enc: [batch, seq_len, nvars] - ANY batch size works!
        Returns: [batch, hidden_size]
        """
        batch_size = x_enc.shape[0]
        device = x_enc.device
        x_np = x_enc.detach().cpu().numpy().astype(np.float32)
        
        embeddings = []
        for i in range(batch_size):
            ts_hash = self._get_hash(x_np[i])
            cache_path = os.path.join(self.cache_dir, f"{ts_hash}.npy")
            
            if os.path.exists(cache_path):
                emb = np.load(cache_path)
            else:
                raise FileNotFoundError(f"Embedding not found: {cache_path}\nRun precompute_embeddings.py!")
            
            embeddings.append(torch.tensor(emb, dtype=torch.float32))
        
        return torch.stack(embeddings, dim=0).to(device)

# class VisionTSTeacher(nn.Module):
#     """
#     Wrapper around VisionTS to use as frozen teacher encoder
#     VisionTS handles time series to visual conversion internally
#     """
    
#     def __init__(self, vit_model='facebook/dinov2-base'):
#         super().__init__()
#         #load ViT
#         self.vis_fm = ViTModel.from_pretrained("facebook/dinov2-base")
#         # self.vis_fm = timm.create_model(vit_model, pretrained=True)
#         self.vis_fm.eval()
#         for param in self.vis_fm.parameters():
#             param.requires_grad = False

#         # Get hidden dimension from ViT model
#         # self.hidden_size = self.vis_fm.num_features
#         self.hidden_size = self.vis_fm.config.hidden_size


#         print(f"✅ ViT Teacher loaded: {vit_model}")
#         print(f"✅ Teacher hidden_size: {self.hidden_size}")
        

#     def forward(self, x_enc):
#         x_image = transform_RP_batch(x_enc)
#         # teacher_encoding = self.vis_fm.forward_features(x_image)[:, 0, :]  # CLS token
#         outputs = self.vis_fm(pixel_values=x_image)
#         teacher_encoding = outputs.last_hidden_state[:, 0, :]  # CLS token
#         return teacher_encoding


class JEPAPredictor(nn.Module):
    """
    JEPA predictor: maps student encoding to teacher encoding space
    Supports both 4D (PatchTST) and 1D (DLinear, iTransformer, etc.) inputs
    """
    
    def __init__(self, student_dim, teacher_dim, hidden_dim=512, input_shape=None):
        """
        Args:
            student_dim: d_model dimension (used for 1D input)
            teacher_dim: teacher embedding dimension (e.g., 768)
            hidden_dim: hidden layer dimension for MLP
            input_shape: tuple (n_vars, d_model, patch_num) for 4D input
                        - If provided: expects 4D input [batch, n_vars, d_model, patch_num]
                        - If None: expects 1D input [batch, student_dim]
        
        Examples:
            # For PatchTST (4D):
            predictor = JEPAPredictor(512, 768, 512, input_shape=(7, 512, 12))
            
            # For DLinear/iTransformer (1D):
            predictor = JEPAPredictor(512, 768, 512, input_shape=None)
        """
        super().__init__()
        
        self.input_shape = input_shape
        
        if input_shape is not None:
            # For 4D input (PatchTST, CNN-based models)
            # Input: [batch, n_vars, d_model, patch_num]
            n_vars, d_model, patch_num = input_shape
            flatten_dim = n_vars * d_model * patch_num
            
            self.predictor = nn.Sequential(
                nn.Flatten(start_dim=1),  # Flatten all spatial dims
                nn.Linear(flatten_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, teacher_dim)
            )
        else:
            # For 1D input (DLinear, iTransformer, Transformer, etc.)
            # Input: [batch, student_dim]
            self.predictor = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, teacher_dim)
            )
    
    def forward(self, student_encoding):
        """
        Forward pass
        
        Args:
            student_encoding: 
                - 4D: [batch_size, n_vars, d_model, patch_num] (if input_shape was provided)
                - 1D: [batch_size, student_dim] (if input_shape was None)
        
        Returns:
            [batch_size, teacher_dim] - predicted teacher encoding
        """
        return self.predictor(student_encoding)

class EncodingFusion(nn.Module):
    """
    Fuses two encoder outputs using MLP, Transformer, or simple operations
    """
    def __init__(self, d_model, fusion_type='mlp', hidden_factor=2, n_heads=8):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'mlp':
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model * hidden_factor),
                nn.LayerNorm(d_model * hidden_factor),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * hidden_factor, d_model)
            )
        elif fusion_type == 'transformer':
            # Project concatenated input to d_model
            self.input_proj = nn.Linear(d_model * 2, d_model)
            
            # Transformer encoder with 2 layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * hidden_factor,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
        elif fusion_type == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion_type == 'add':
            pass  # Simple addition
    
    def forward(self, enc1, enc2):
        """
        Supports both 3D and 4D inputs:
        - 3D (TimesNet): [bs, seq_len, d_model]
        - 4D (PatchTST): [bs, nvars, d_model, patch_num]
        """
        if self.fusion_type == 'add':
            return enc1 + enc2
        
        elif self.fusion_type == 'weighted':
            alpha = torch.sigmoid(self.alpha)
            return alpha * enc1 + (1 - alpha) * enc2
        
        elif self.fusion_type == 'mlp':
            # Handle both 3D and 4D inputs
            if enc1.dim() == 3:
                concat = torch.cat([enc1, enc2], dim=-1)
                bs, seq_len, d_model_2 = concat.shape
                concat_flat = concat.reshape(-1, d_model_2)
                fused_flat = self.fusion(concat_flat)
                fused = fused_flat.reshape(bs, seq_len, -1)
                return fused
            else:
                concat = torch.cat([enc1, enc2], dim=2)
                bs, nvars, d_model_2, patch_num = concat.shape
                concat_flat = concat.permute(0, 1, 3, 2).reshape(-1, d_model_2)
                fused_flat = self.fusion(concat_flat)
                fused = fused_flat.reshape(bs, nvars, patch_num, -1).permute(0, 1, 3, 2)
                return fused
        
        elif self.fusion_type == 'transformer':
            if enc1.dim() == 3:
                # 3D: [bs, seq_len, d_model]
                concat = torch.cat([enc1, enc2], dim=-1)  # [bs, seq_len, d_model*2]
                projected = self.input_proj(concat)        # [bs, seq_len, d_model]
                fused = self.transformer(projected)        # [bs, seq_len, d_model]
                return fused
            else:
                # 4D: [bs, nvars, d_model, patch_num]
                bs, nvars, d_model, patch_num = enc1.shape
                # Reshape to [bs * nvars, patch_num, d_model] for transformer
                enc1_3d = enc1.permute(0, 1, 3, 2).reshape(bs * nvars, patch_num, d_model)
                enc2_3d = enc2.permute(0, 1, 3, 2).reshape(bs * nvars, patch_num, d_model)
                concat = torch.cat([enc1_3d, enc2_3d], dim=-1)  # [bs*nvars, patch_num, d_model*2]
                projected = self.input_proj(concat)              # [bs*nvars, patch_num, d_model]
                fused = self.transformer(projected)              # [bs*nvars, patch_num, d_model]
                # Reshape back to 4D
                fused = fused.reshape(bs, nvars, patch_num, d_model).permute(0, 1, 3, 2)
                return fused


class Model(nn.Module):
    """
    JEPAVTS: Joint-Embedding Predictive Architecture for Vision-Time Series
    Using VisionTS as the frozen teacher encoder
    
    Architecture:
    - Teacher: VisionTS (frozen) - processes raw TS as visual
    - Student: Time series model (trainable) - processes raw TS
    - Predictor: Aligns student to teacher representations (trainable)
    """
    
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name

        # Determine which architecture to use
        self.use_dual_encoder = getattr(configs, 'use_dual_encoder', False)
        
        print("\n" + "="*50)
        print("Initializing JEPAVTS Model")
        if self.use_dual_encoder:
            print("Architecture: DUAL ENCODER 🔀")
        else:
            print("Architecture: SINGLE ENCODER →")
        print("="*50)
        
        print("\n" + "="*50)
        print("Initializing JEPAVTS Model")
        print("="*50)
        
        # Get student model name
        student_model_name = getattr(configs, 'student_model', 'PatchTST')
        
        # Teacher(s): VisionTS (frozen) - supports multi-rendering
        print(f"\n📊 Loading teacher vision encoder...")
        self.data = getattr(configs, 'data')
        self.rendering_methods = getattr(configs, 'rendering_methods', None)
        
        if self.rendering_methods and len(self.rendering_methods) > 0:
            # Multi-rendering mode: one teacher per rendering method
            self.multi_rendering = True
            self.num_renderings = len(self.rendering_methods)
            self.teachers = nn.ModuleList([
                VisionTSTeacher(dataset_name=self.data, rendering_method=method)
                for method in self.rendering_methods
            ])
            self.teacher_dim = self.teachers[0].hidden_size
            
            self.multi_rendering_alpha_mode = getattr(configs, 'multi_rendering_alpha_mode', 'same')
            self.per_method_alphas = getattr(configs, 'per_method_alphas', None)
            
            print(f"✅ Multi-rendering mode: {self.rendering_methods}")
            print(f"✅ Alpha mode: {self.multi_rendering_alpha_mode}")
            if self.multi_rendering_alpha_mode == 'per_method' and self.per_method_alphas:
                print(f"✅ Per-method alphas: {self.per_method_alphas}")
        else:
            # Single rendering mode (backward compatible)
            self.multi_rendering = False
            self.num_renderings = 1
            self.teacher = VisionTSTeacher(dataset_name=self.data)
            self.teacher_dim = self.teacher.hidden_size
        
        print(f"✅ Teacher dimension: {self.teacher_dim}")
        
        # Student: Time Series Encoder (trainable)
        self.use_multi_encoder = (
            getattr(configs, 'multi_encoder', False) and self.multi_rendering
        )
        
        if self.use_multi_encoder:
            # One student encoder per rendering method
            print(f"\n🎓 Building {self.num_renderings} student models (one per rendering)...")
            self.students_multi = nn.ModuleList([
                self._build_student_model(student_model_name, configs)
                for _ in self.rendering_methods
            ])
            self.student = self.students_multi[0]  # reference for shape detection & decode
            for method in self.rendering_methods:
                print(f"✅ Student [{method}]: {student_model_name}")
        else:
            print(f"\n🎓 Building student model: {student_model_name}")
            self.student = self._build_student_model(student_model_name, configs)
            
            if self.use_dual_encoder:
                self.student1 = self._build_student_model(student_model_name, configs)
                self.student2 = self._build_student_model(student_model_name, configs)

                self.encoder_fusion = EncodingFusion(d_model=configs.d_model,
                                                fusion_type=getattr(configs, 'fusion_type', 'mlp'))
        self.student_dim = configs.d_model
        print(f"✅ Student dimension: {self.student_dim}")

        # Calculate encoding shape dynamically based on student model type
        student_model_name = getattr(configs, 'student_model', 'PatchTST')
        self.student_model_name = student_model_name

        if student_model_name == 'PatchTST' and hasattr(self.student, 'patch_embedding'):
            # PatchTST: encoding shape is [B, nvars, d_model, patch_num]
            patch_len = self.student.patch_embedding.patch_len
            stride = self.student.patch_embedding.stride
            patch_num = int((configs.seq_len - patch_len) / stride + 2)
            self.encoding_type = '4D'  # [B, nvars, d_model, patch_num]
            predictor_input_shape = (configs.enc_in, configs.d_model, patch_num)
        elif student_model_name == 'TimesNet':
            # TimesNet: encoding shape is [B, T, d_model]
            patch_num = None
            self.encoding_type = '3D'  # [B, T, d_model]
            predictor_input_shape = (configs.seq_len, configs.d_model, 1)  # Treat as [B, T, d_model, 1]
        else:
            # For non-patch models (DLinear, etc.), use 1D input
            patch_num = None
            self.encoding_type = '1D'
            predictor_input_shape = None

        print(f"✅ Encoding type: {self.encoding_type}")

        # JEPA Predictor(s) (trainable)
        # multi_predictor and multi_encoder are independent:
        #   --multi_predictor              → 1 student, k predictors (option 2)
        #   --multi_encoder --multi_predictor → k students, k predictors (option 3)
        #   --multi_encoder                → k students, 1 shared predictor (option 4)
        self.use_multi_predictor = (
            getattr(configs, 'multi_predictor', False) and self.multi_rendering
        )
        
        if self.use_multi_predictor:
            # One predictor per rendering method (separate z'_x per rendering)
            print(f"\n🔗 Building {self.num_renderings} JEPA predictors (one per rendering)...")
            self.predictors = nn.ModuleList([
                JEPAPredictor(
                    student_dim=configs.d_model,
                    teacher_dim=self.teacher_dim,
                    hidden_dim=getattr(configs, 'jepa_hidden_dim', 512),
                    input_shape=predictor_input_shape
                )
                for _ in self.rendering_methods
            ])
            for method in self.rendering_methods:
                print(f"✅ JEPA predictor [{method}]: {configs.d_model} -> {self.teacher_dim}")
        else:
            # Single shared predictor (original behaviour)
            print(f"\n🔗 Building JEPA predictor...")
            self.predictor = JEPAPredictor(
                    student_dim=configs.d_model,
                    teacher_dim=self.teacher_dim,
                    hidden_dim=getattr(configs, 'jepa_hidden_dim', 512),
                    input_shape=predictor_input_shape
                )
            print(f"✅ JEPA predictor: {configs.d_model} -> {self.teacher_dim}")
        
        # Store patch_num for classification head
        self.patch_num = patch_num if patch_num is not None else 1
        
        # Prediction head for forecasting task
        if self.task_name != 'classification':
            self.forecast_head = nn.Linear(self.student_dim, configs.pred_len * configs.c_out)
            print(f"✅ Forecast head: {self.student_dim} -> {configs.pred_len * configs.c_out}")
        
        # Classification head
        if self.task_name == 'classification':
            self.num_classes = configs.num_class
            # Calculate flatten dimension for classification head based on encoding type
            if self.encoding_type == '4D':
                # PatchTST: [B, nvars, d_model, patch_num]
                flatten_dim = configs.enc_in * configs.d_model * self.patch_num
            elif self.encoding_type == '3D':
                # TimesNet: [B, T, d_model]
                flatten_dim = configs.seq_len * configs.d_model
            else:
                flatten_dim = configs.d_model
            
            self.classification_head = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(flatten_dim, configs.d_model),
                nn.LayerNorm(configs.d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(configs.d_model, self.num_classes)
            )
            print(f"✅ Classification head: {flatten_dim} -> {self.num_classes} classes")
        
        # Learnable loss weights (uncertainty-based)
        self.use_learned_loss_weights = getattr(configs, 'learned_loss_weights', False)
        if self.use_learned_loss_weights:
            # Initialize log-variance parameters (start with equal weighting ~0.5 each)
            self.w_pred = nn.Parameter(torch.tensor([0.0]))  # log(σ²) for prediction loss
            self.w_jepa = nn.Parameter(torch.tensor([0.0]))  # log(σ²) for JEPA loss
            print(f"✅ Using learned loss weights (uncertainty-based)")

        print("\n" + "="*50)
        print("JEPAVTS Model Initialized Successfully!")
        print("="*50 + "\n")
        
    # def _build_student_model(self, model_name, configs):
    #     """Build the student time series model"""
    #     if model_name == 'PatchTST':
    #         from models.PatchTST import Model as PatchTSTModel
    #         return PatchTSTModel(configs)
    #     elif model_name == 'TimesNet':
    #         from models.TimesNet import Model as TimesNetModel
    #         return TimesNetModel(configs)
    #     elif model_name == 'DLinear':
    #         from models.DLinear import Model as DLinearModel
    #         return DLinearModel(configs)
    #     elif model_name == 'iTransformer':
    #         from models.iTransformer import Model as iTransformerModel
    #         return iTransformerModel(configs)
    #     elif model_name == 'Transformer':
    #         from models.Transformer import Model as TransformerModel
    #         return TransformerModel(configs)
    #     else:
    #         raise NotImplementedError(f"Student model {model_name} not implemented")
    
    def _build_student_model(self, model_name, configs):
        """Build student model using exp_basic's model registry"""
        from exp.exp_basic import Exp_Basic
        
        if model_name not in Exp_Basic.MODEL_DICT:
            raise NotImplementedError(f"Student model {model_name} not found")
        
        return Exp_Basic.MODEL_DICT[model_name].Model(configs)

    def compute_weighted_loss(self, pred_loss, jepa_loss):
      """
      Uncertainty-based multi-task loss weighting.
      L = l_pred * exp(-w_pred) + w_pred + l_jepa * exp(-w_jepa) + w_jepa
      
      This automatically learns the optimal balance between losses.
      """
      if self.use_learned_loss_weights:
          weighted_pred = pred_loss * torch.exp(-self.w_pred) + self.w_pred
          weighted_jepa = jepa_loss * torch.exp(-self.w_jepa) + self.w_jepa
          total_loss = weighted_pred + weighted_jepa
          
          # Return individual components for logging
          return total_loss, {
              'pred_weight': torch.exp(-self.w_pred).item(),
              'jepa_weight': torch.exp(-self.w_jepa).item(),
              'w_pred': self.w_pred.item(),
              'w_jepa': self.w_jepa.item()
          }
      else:
          return None, None
    
    def teacher_forward(self, x_enc):
        """
        Teacher forward pass (always with no_grad).
        x_enc: [batch_size, seq_len, n_vars] - raw time series
        
        Returns:
            - Multi-rendering mode: list of [batch_size, teacher_dim] (one per rendering)
            - Single mode: [batch_size, teacher_dim]
        """
        with torch.no_grad():
            if self.multi_rendering:
                return [teacher(x_enc) for teacher in self.teachers]
            else:
                return self.teacher(x_enc)
    
    def _predict_teacher(self, student_encoding):
        """
        Apply JEPA predictor(s) to student encoding.
        
        Returns:
            - Multi-predictor mode: list of [batch, teacher_dim] (one z' per rendering)
            - Single predictor mode: [batch, teacher_dim]
        """
        if self.use_multi_predictor:
            return [pred(student_encoding) for pred in self.predictors]
        else:
            return self.predictor(student_encoding)
    
    def student_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Student forward pass
        Returns predictions and optionally encodings
        """
        # Get student encoding
        student_encoding, means, stdev = self.student.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Predict teacher encoding(s) via JEPA
        predicted_teacher_encoding = self._predict_teacher(student_encoding)
        forecast_output = self.student.forecast_decode(student_encoding, means, stdev)
        
        if return_all:
            return forecast_output, student_encoding, predicted_teacher_encoding
        
        return forecast_output

    def student_forward_dual_encoder(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Student forward pass
        Returns predictions and optionally encodings
        """
        # Get student encoding
        student_encoding1, means1, stdev1 = self.student1.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        student_encoding2, means2, stdev2 = self.student2.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        predicted_teacher_encoding = self._predict_teacher(student_encoding2)

        combined_encoding = self.encoder_fusion(student_encoding1, student_encoding2)
        forecast_output = self.student1.forecast_decode(combined_encoding, means1, stdev1)

        if return_all:
            return forecast_output, predicted_teacher_encoding
        
        return forecast_output
    
    def student_forward_multi_encoder(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Multi-encoder forward: k student encoders, fused for decode.
        
        Two modes controlled by use_multi_predictor:
          - multi_predictor=True  (opt 3): predictor_i(student_encoding_i) → z'_i
          - multi_predictor=False (opt 4): predictor(fused_encoding) → z' (shared)
        """
        # Each student encodes the same input independently
        results = [
            s.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
            for s in self.students_multi
        ]
        student_encodings = [r[0] for r in results]
        
        # Fuse student encodings (mean) for forecasting decode
        fused_encoding = torch.stack(student_encodings, dim=0).mean(dim=0)
        
        # JEPA prediction(s)
        if self.use_multi_predictor:
            # Option 3: each predictor_i paired with student_encoding_i
            predicted_teacher = [
                pred(enc)
                for pred, enc in zip(self.predictors, student_encodings)
            ]
        else:
            # Option 4: shared predictor on fused encoding
            predicted_teacher = self.predictor(fused_encoding)
        
        # Decode using first student's decoder and normalisation stats
        means_0, stdev_0 = results[0][1], results[0][2]
        forecast_output = self.students_multi[0].forecast_decode(
            fused_encoding, means_0, stdev_0)
        
        if return_all:
            return forecast_output, student_encodings, predicted_teacher
        
        return forecast_output
    
    def classification_forward(self, x_enc, padding_mask=None, return_all=False):
        """
        Classification forward pass
        x_enc: [batch, seq_len, n_vars]
        padding_mask: [batch, seq_len] (optional)
        Returns: class logits [batch, num_classes]
        """
        # Get student encoding 
        student_encoding = self.student.encode(x_enc)
        # Classification output
        class_logits = self.classification_head(student_encoding)
        
        if return_all and self.training:
            # Get JEPA prediction(s) for teacher alignment
            predicted_teacher_encoding = self._predict_teacher(student_encoding)
            return class_logits, student_encoding, predicted_teacher_encoding
        
        return class_logits
    
    def classification_forward_dual_encoder(self, x_enc, padding_mask=None, return_all=False):
        """
        Classification forward pass with dual encoder
        """

        student_encoding1 = self.student1.encode(x_enc)
        student_encoding2 = self.student2.encode(x_enc)
        
        # JEPA prediction(s) from encoder2
        predicted_teacher_encoding = self._predict_teacher(student_encoding2)
        
        # Fuse encodings
        combined_encoding = self.encoder_fusion(student_encoding1, student_encoding2)
        
        # Classification output
        class_logits = self.classification_head(combined_encoding)
        
        if return_all and self.training:
            return class_logits, predicted_teacher_encoding
        
        return class_logits
    
    def classification_forward_multi_encoder(self, x_enc, padding_mask=None, return_all=False):
        """
        Multi-encoder classification: k students, fused for classification.
        Shared or multi predictor controlled by use_multi_predictor.
        """
        student_encodings = [s.encode(x_enc) for s in self.students_multi]
        
        # Fuse student encodings (mean) for classification
        fused_encoding = torch.stack(student_encodings, dim=0).mean(dim=0)
        
        # JEPA prediction(s)
        if self.use_multi_predictor:
            predicted_teacher = [
                pred(enc)
                for pred, enc in zip(self.predictors, student_encodings)
            ]
        else:
            predicted_teacher = self.predictor(fused_encoding)
        
        class_logits = self.classification_head(fused_encoding)
        
        if return_all and self.training:
            return class_logits, student_encodings, predicted_teacher
        
        return class_logits
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Main forward - behavior depends on task_name, training mode and architecture
        
        For classification:
            x_enc: [batch, seq_len, n_vars] - time series input
            x_mark_enc: padding_mask or None
            x_dec, x_mark_dec: ignored (can be None)
        
        For forecasting:
            x_enc: [batch, seq_len, n_vars]
            x_mark_enc: time features
            x_dec: decoder input
            x_mark_dec: decoder time features
        """
        # Classification task
        if self.task_name == 'classification':
            padding_mask = x_mark_enc  # In classification, x_mark_enc is used as padding_mask
            if self.use_multi_encoder:
                return self.classification_forward_multi_encoder(
                    x_enc, padding_mask, return_all=self.training)
            elif self.use_dual_encoder:
                return self.classification_forward_dual_encoder(
                    x_enc, padding_mask, return_all=self.training)
            else:
                return self.classification_forward(
                    x_enc, padding_mask, return_all=self.training)
        
        # Forecasting task
        if self.use_multi_encoder:
            return self.student_forward_multi_encoder(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
        elif self.use_dual_encoder:
            return self.student_forward_dual_encoder(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
        else:
            return self.student_forward(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
