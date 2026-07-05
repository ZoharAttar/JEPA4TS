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
        per_var: if True, loads per-variable embeddings [N, 768] per sample
                 from cache with _pervar suffix. If False, loads averaged [768].
    """
    
    def __init__(self, dataset_name="ETTh1", hidden_size=768, rendering_method=None, per_var=False, root_path=None):
        super().__init__()
        self.rendering_method = rendering_method
        self.per_var = per_var
        
        suffix = "_pervar" if per_var else ""
        root = root_path.rstrip('/') if root_path else "./dataset/ETT-small/"
        if rendering_method:
            self.cache_dir = f"{root}/dino_embeddings_{dataset_name}_{rendering_method}{suffix}"
        else:
            self.cache_dir = f"{root}/dino_embeddings_{dataset_name}{suffix}"
        self.hidden_size = hidden_size
        
        if not os.path.exists(self.cache_dir):
            raise ValueError(f"❌ Cache not found: {self.cache_dir}\n   Run precompute_embeddings.py first!")
        
        n_cached = len([f for f in os.listdir(self.cache_dir) if f.endswith('.npy')])
        tag = f" [{rendering_method}]" if rendering_method else ""
        pv_tag = " [per_var]" if per_var else ""
        print(f"✅ VisionTSTeacher{tag}{pv_tag}: Loading from cache")
        print(f"✅ Cache dir: {self.cache_dir}")
        print(f"✅ Cached embeddings: {n_cached}")
        print(f"✅ Teacher hidden_size: {self.hidden_size}")
    
    def _get_hash(self, ts_array):
        return hashlib.md5(ts_array.astype(np.float32).tobytes()).hexdigest()[:16]
    
    def forward(self, x_enc):
        """
        x_enc: [batch, seq_len, nvars]
        Returns:
            per_var=False: [batch, hidden_size]
            per_var=True:  [batch, nvars, hidden_size]
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
    JEPA predictor: maps student encoding to teacher encoding space.
    Supports both 4D (PatchTST) and 1D (DLinear, iTransformer, etc.) inputs.
    
    When per_var=True and input_shape is provided (4D), the predictor operates
    per-variable: each variable's [d_model, patch_num] is independently mapped
    to [teacher_dim] using a shared MLP, producing [B, n_vars, teacher_dim].
    """
    
    def __init__(self, student_dim, teacher_dim, hidden_dim=512, input_shape=None, per_var=False):
        """
        Args:
            student_dim: d_model dimension (used for 1D input)
            teacher_dim: teacher embedding dimension (e.g., 768)
            hidden_dim: hidden layer dimension for MLP
            input_shape: tuple (n_vars, d_model, patch_num) for 4D input
            per_var: if True, predict per-variable instead of flattening all vars
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.per_var = per_var
        
        if input_shape is not None:
            n_vars, d_model, patch_num = input_shape
            if per_var:
                # Per-variable: shared MLP applied to each variable independently
                flatten_dim = d_model * patch_num
            else:
                # Original: flatten all variables together
                flatten_dim = n_vars * d_model * patch_num
            
            layers = []
            if not per_var:
                layers.append(nn.Flatten(start_dim=1))
            layers.extend([
                nn.Linear(flatten_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, teacher_dim)
            ])
            self.predictor = nn.Sequential(*layers)
        else:
            self.predictor = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, teacher_dim)
            )
    
    def forward(self, student_encoding):
        """
        Args:
            student_encoding: 
                - 4D: [B, n_vars, d_model, patch_num]
                - 1D: [B, student_dim]
        Returns:
            per_var=True  + 4D input: [B, n_vars, teacher_dim]
            per_var=False + 4D input: [B, teacher_dim]
            1D input:                 [B, teacher_dim]
        """
        if self.per_var and self.input_shape is not None:
            B, N, D, P = student_encoding.shape
            x = student_encoding.reshape(B * N, D * P)
            x = self.predictor(x)       # [B*N, teacher_dim]
            return x.reshape(B, N, -1)  # [B, N, teacher_dim]
        return self.predictor(student_encoding)

class EncodingFusion(nn.Module):
    """
    Fuses N encoder outputs using MLP, Transformer, or simple operations.
    Supports any number of inputs (2 for dual encoder, K+1 for multi+dual, etc.)
    """
    def __init__(self, d_model, num_inputs=2, fusion_type='mlp', hidden_factor=2, n_heads=8):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_inputs = num_inputs
        
        if fusion_type == 'mlp':
            self.fusion = nn.Sequential(
                nn.Linear(d_model * num_inputs, d_model * hidden_factor),
                nn.LayerNorm(d_model * hidden_factor),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * hidden_factor, d_model)
            )
        elif fusion_type == 'transformer':
            self.input_proj = nn.Linear(d_model * num_inputs, d_model)
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
            self.alphas = nn.Parameter(torch.ones(num_inputs) / num_inputs)
        elif fusion_type == 'add':
            pass
    
    def forward(self, *encodings):
        """
        Args: N encodings, each 3D [bs, seq_len, d_model] or 4D [bs, nvars, d_model, patch_num]
              Can also pass a single list of encodings.
        """
        if len(encodings) == 1 and isinstance(encodings[0], (list, tuple)):
            encodings = encodings[0]
        
        if self.fusion_type == 'add':
            result = encodings[0]
            for enc in encodings[1:]:
                result = result + enc
            return result
        
        elif self.fusion_type == 'weighted':
            weights = torch.softmax(self.alphas, dim=0)
            result = weights[0] * encodings[0]
            for i, enc in enumerate(encodings[1:], 1):
                result = result + weights[i] * enc
            return result
        
        elif self.fusion_type == 'mlp':
            if encodings[0].dim() == 3:
                concat = torch.cat(list(encodings), dim=-1)
                bs, seq_len, cat_dim = concat.shape
                concat_flat = concat.reshape(-1, cat_dim)
                fused_flat = self.fusion(concat_flat)
                fused = fused_flat.reshape(bs, seq_len, -1)
                return fused
            else:
                concat = torch.cat(list(encodings), dim=2)
                bs, nvars, cat_dim, patch_num = concat.shape
                concat_flat = concat.permute(0, 1, 3, 2).reshape(-1, cat_dim)
                fused_flat = self.fusion(concat_flat)
                fused = fused_flat.reshape(bs, nvars, patch_num, -1).permute(0, 1, 3, 2)
                return fused
        
        elif self.fusion_type == 'transformer':
            if encodings[0].dim() == 3:
                concat = torch.cat(list(encodings), dim=-1)
                projected = self.input_proj(concat)
                fused = self.transformer(projected)
                return fused
            else:
                bs, nvars, d_model, patch_num = encodings[0].shape
                encs_3d = [e.permute(0, 1, 3, 2).reshape(bs * nvars, patch_num, d_model)
                           for e in encodings]
                concat = torch.cat(encs_3d, dim=-1)
                projected = self.input_proj(concat)
                fused = self.transformer(projected)
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
        # NO-DINO ablation: two student encoders fused, task loss only. No teacher,
        # no JEPA predictor, no vision — a capacity-matched control for the dual
        # encoder. Needs no precomputed embeddings (teacher never built/called).
        self.no_dino = getattr(configs, 'no_dino', False)
        # DINO-DIRECT ablation: single student encoder, but the frozen DINO
        # embedding is injected directly into the fusion/forecast head instead of
        # being a JEPA distillation target. DINO is a live input (train AND test).
        self.dino_direct = getattr(configs, 'dino_direct', False)
        
        print("\n" + "="*50)
        print("Initializing JEPAVTS Model")
        if self.no_dino:
            print("Architecture: NO DINO (dual encoders, no teacher/JEPA) 🚫")
        elif self.dino_direct:
            print("Architecture: DINO DIRECT (single encoder + DINO into fusion head) 🎯")
        elif self.use_dual_encoder:
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
        self.per_var_teacher = getattr(configs, 'per_var_teacher', False)
        if self.no_dino:
            # NO-DINO: never build/query the teacher. Force teacher-related state
            # off so downstream code (encoding-type detection, forward) is safe.
            self.per_var_teacher = False
            self.rendering_methods = None

        # Channel-independent TimesNet for classification: TimesNet mixes channels
        # in its embedding (Conv1d: enc_in -> d_model), so it has no native
        # per-variable representation. To align each variable with its own DINO
        # embedding (per_var teacher), we fold the variables into the batch and
        # encode each as a univariate series (built with enc_in=1), producing a
        # per-variable encoding [B, N, d_model, T] — exactly like channel-
        # independent PatchTST/TimeMixer. Only enabled for classification with a
        # per-variable teacher; plain TimesNet stays channel-mixed.
        self.timesnet_ci = (
            student_model_name == 'TimesNet'
            and self.task_name == 'classification'
            and self.per_var_teacher
        )
        if self.timesnet_ci:
            print("✅ Channel-independent TimesNet: per-variable encoding [B, N, d_model, T]")

        if self.per_var_teacher:
            print("✅ Per-variable teacher mode: each variable gets its own DINO embedding")
        
        if self.no_dino:
            # NO-DINO ablation: no teacher at all.
            self.multi_rendering = False
            self.num_renderings = 1
            self.teacher = None
            self.teacher_dim = 768  # placeholder, unused (no predictor)
            print("📊 Skipping teacher (NO-DINO ablation)")
        elif self.rendering_methods and len(self.rendering_methods) > 0:
            # Multi-rendering mode: one teacher per rendering method
            self.multi_rendering = True
            self.num_renderings = len(self.rendering_methods)
            self.teachers = nn.ModuleList([
                VisionTSTeacher(dataset_name=self.data, rendering_method=method,
                                per_var=self.per_var_teacher,root_path=getattr(configs,'root_path',None))
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
            self.teacher = VisionTSTeacher(dataset_name=self.data,
                                           per_var=self.per_var_teacher,root_path=getattr(configs,'root_path',None))
            self.teacher_dim = self.teacher.hidden_size
        
        print(f"✅ Teacher dimension: {self.teacher_dim}")
        
        # Student: Time Series Encoder (trainable)
        self.use_multi_encoder = (
            getattr(configs, 'multi_encoder', False) and self.multi_rendering
        )
        fusion_type = getattr(configs, 'fusion_type', 'mlp')
        
        if self.use_multi_encoder and self.use_dual_encoder:
            # K JEPA encoders (one per rendering) + 1 forecast encoder
            print(f"\n🎓 Building {self.num_renderings} JEPA student models + 1 forecast encoder...")
            self.students_multi = nn.ModuleList([
                self._build_student_model(student_model_name, configs)
                for _ in self.rendering_methods
            ])
            self.student_forecast = self._build_student_model(student_model_name, configs)
            self.student = self.students_multi[0]  # reference for shape detection
            for method in self.rendering_methods:
                print(f"✅ JEPA Student [{method}]: {student_model_name}")
            print(f"✅ Forecast Student: {student_model_name}")
            num_fusion_inputs = self.num_renderings + 1
            self.encoder_fusion = EncodingFusion(
                d_model=configs.d_model, num_inputs=num_fusion_inputs,
                fusion_type=fusion_type)
            print(f"✅ Fusion: {fusion_type} ({num_fusion_inputs} inputs)")
        elif self.use_multi_encoder:
            # K encoders (one per rendering), no separate forecast encoder
            print(f"\n🎓 Building {self.num_renderings} student models (one per rendering)...")
            self.students_multi = nn.ModuleList([
                self._build_student_model(student_model_name, configs)
                for _ in self.rendering_methods
            ])
            self.student = self.students_multi[0]  # reference for shape detection & decode
            for method in self.rendering_methods:
                print(f"✅ Student [{method}]: {student_model_name}")
        elif self.use_dual_encoder or self.no_dino:
            # 1 forecast encoder + 1 JEPA encoder, fused. (NO-DINO uses the same
            # two encoders + fusion for a capacity-matched control; the JEPA
            # branch is simply not trained against a teacher.)
            print(f"\n🎓 Building dual encoder (forecast + JEPA)...")
            self.student = self._build_student_model(student_model_name, configs)
            self.student1 = self._build_student_model(student_model_name, configs)
            self.student2 = self._build_student_model(student_model_name, configs)
            self.encoder_fusion = EncodingFusion(
                d_model=configs.d_model, num_inputs=2, fusion_type=fusion_type)
            print(f"✅ Fusion: {fusion_type} (2 inputs)")
        else:
            # Single encoder for everything
            print(f"\n🎓 Building student model: {student_model_name}")
            self.student = self._build_student_model(student_model_name, configs)
        self.student_dim = configs.d_model
        print(f"✅ Student dimension: {self.student_dim}")

        # DINO-DIRECT: project the frozen DINO embedding to d_model and fuse it
        # with the single student's encoding (DINO as a live second stream).
        if self.dino_direct:
            self.dino_projector = nn.Sequential(
                nn.Linear(self.teacher_dim, configs.d_model),
                nn.LayerNorm(configs.d_model),
                nn.GELU(),
            )
            self.dino_fusion_module = EncodingFusion(
                d_model=configs.d_model, num_inputs=2, fusion_type=fusion_type)
            print(f"✅ DINO-DIRECT projector: {self.teacher_dim} -> {configs.d_model}")
            print(f"✅ DINO-DIRECT fusion: {fusion_type} (student ⊕ DINO)")

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
        elif student_model_name == 'iTransformer':
            # iTransformer.encode returns [B, nvars(+covariate tokens), d_model].
            # For JEPA we keep the nvars variable tokens and treat each as a
            # patch_num=1 token → predictor input [B, nvars, d_model, 1].
            patch_num = 1
            self.encoding_type = 'itransformer'
            predictor_input_shape = (configs.enc_in, configs.d_model, patch_num)
        elif student_model_name == 'TimeMixer':
            patch_num = None
            self.timemixer_jepa_scale = getattr(configs, 'timemixer_jepa_scale', 'fine')
            ds_layers = getattr(configs, 'down_sampling_layers', 0)
            ds_window = getattr(configs, 'down_sampling_window', 1)
            if self.timemixer_jepa_scale == 'coarse' and ds_layers > 0:
                self.timemixer_jepa_idx = -1
                jepa_T = configs.seq_len // (ds_window ** ds_layers)
            else:
                self.timemixer_jepa_idx = 0
                jepa_T = configs.seq_len
            self.timemixer_jepa_T = jepa_T
            if getattr(configs, 'channel_independence', 1):
                # channel_independence=True: enc [B*N, T, d_model] → [B, N, d_model, T]
                self.encoding_type = '4D_timemixer'
                predictor_input_shape = (configs.enc_in, configs.d_model, jepa_T)
            else:
                # channel_independence=False: enc [B, seq_len, d_model] (all vars mixed)
                self.encoding_type = '3D'
                predictor_input_shape = (jepa_T, configs.d_model, 1)
            print(f"✅ TimeMixer JEPA scale: {self.timemixer_jepa_scale} "
                  f"(idx={self.timemixer_jepa_idx}, T={jepa_T}, "
                  f"flatten={configs.d_model * jepa_T})")
        elif student_model_name == 'TimesNet':
            if self.timesnet_ci:
                # Channel-independent: per-variable encoding [B, N, d_model, T].
                # Reuse the 4D machinery (predictor / classification head / fusion)
                # by treating the time axis as patch_num.
                patch_num = configs.seq_len
                self.encoding_type = '4D'  # [B, nvars, d_model, T]
                predictor_input_shape = (configs.enc_in, configs.d_model, configs.seq_len)
            else:
                # Channel-mixed (original): [B, T, d_model]
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
        
        if self.no_dino or self.dino_direct:
            # NO-DINO / DINO-DIRECT ablations: no JEPA predictor at all.
            self.predictor = None
            print("\n🔗 Skipping JEPA predictor (ablation: no JEPA distillation)")
        elif self.use_multi_predictor:
            # One predictor per rendering method (separate z'_x per rendering)
            print(f"\n🔗 Building {self.num_renderings} JEPA predictors (one per rendering)...")
            self.predictors = nn.ModuleList([
                JEPAPredictor(
                    student_dim=configs.d_model,
                    teacher_dim=self.teacher_dim,
                    hidden_dim=getattr(configs, 'jepa_hidden_dim', 512),
                    input_shape=predictor_input_shape,
                    per_var=self.per_var_teacher
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
                    input_shape=predictor_input_shape,
                    per_var=self.per_var_teacher
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
            elif self.encoding_type == 'itransformer':
                # iTransformer: [B, nvars, d_model, 1] (one token per variable)
                flatten_dim = configs.enc_in * configs.d_model
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
        
        # Channel-independent TimesNet: build a UNIVARIATE backbone (enc_in=1).
        # The variables are folded into the batch at encode time (see _encode_cls),
        # so each variable is embedded/encoded independently.
        if getattr(self, 'timesnet_ci', False) and model_name == 'TimesNet':
            import copy
            ci_configs = copy.copy(configs)
            ci_configs.enc_in = 1
            ci_configs.c_out = 1
            return Exp_Basic.MODEL_DICT[model_name].Model(ci_configs)
        
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
        
        Returns (shape depends on per_var_teacher):
            - per_var=False: [batch_size, teacher_dim]  (or list thereof)
            - per_var=True:  [batch_size, n_vars, teacher_dim]  (or list thereof)
        """
        with torch.no_grad():
            if self.multi_rendering:
                return [teacher(x_enc) for teacher in self.teachers]
            else:
                return self.teacher(x_enc)
    
    def _to_jepa_encoding(self, student_encoding):
        """Convert a raw student encoding into the shape expected by the JEPA
        predictor.

        For iTransformer, `encode()` returns [B, nvars(+covariate tokens), d_model]
        (logic untouched in the model). Here we keep only the nvars variable tokens
        and add a trailing dim → [B, nvars, d_model, 1], matching the predictor's
        input_shape=(enc_in, d_model, 1). For every other backbone this is a no-op.

        Note: decoding always uses the full (unmodified) encoding; only the JEPA
        alignment branch uses this view.
        """
        if self.encoding_type == 'itransformer':
            N = self.configs.enc_in
            return student_encoding[:, :N, :].unsqueeze(-1)
        return student_encoding

    def _predict_teacher(self, student_encoding):
        """
        Apply JEPA predictor(s) to student encoding.
        
        Returns (shape depends on per_var_teacher):
            - per_var=False: [batch, teacher_dim]  (or list thereof)
            - per_var=True:  [batch, n_vars, teacher_dim]  (or list thereof)
        """
        if self.use_multi_predictor:
            return [pred(student_encoding) for pred in self.predictors]
        else:
            return self.predictor(student_encoding)
    
    def _inject_fused_into_enc_list(self, enc_list, combined_encoding, B):
        """Inject fused JEPA encoding into enc_list at the configured scale index."""
        jepa_idx = self.timemixer_jepa_idx
        N = self.configs.enc_in
        T, D = enc_list[jepa_idx].shape[1], enc_list[jepa_idx].shape[2]
        fused = combined_encoding.permute(0, 1, 3, 2).reshape(B * N, T, D)
        if jepa_idx == 0:
            return [fused] + enc_list[1:]
        return enc_list[:jepa_idx] + [fused]

    def _timemixer_encode(self, student, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Encode with TimeMixer and reshape selected-scale encoding to 4D for JEPA.
        Scale is controlled by timemixer_jepa_scale ('fine' → [0], 'coarse' → [-1]).
        Returns: (jepa_encoding [B, N, d_model, T], enc_out_list, x_list, B_size)
        """
        enc_out_list, x_list, B = student.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        enc_jepa = enc_out_list[self.timemixer_jepa_idx]  # [B*N, T, d_model]
        N = self.configs.enc_in
        T, D = enc_jepa.shape[1], enc_jepa.shape[2]
        jepa_encoding = enc_jepa.reshape(B, N, T, D).permute(0, 1, 3, 2)  # [B, N, d_model, T]
        return jepa_encoding, enc_out_list, x_list, B

    def student_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Student forward pass
        Returns predictions and optionally encodings
        """
        if self.encoding_type == '4D_timemixer':
            jepa_encoding, enc_out_list, x_list, B = self._timemixer_encode(
                self.student, x_enc, x_mark_enc, x_dec, x_mark_dec)
            predicted_teacher_encoding = self._predict_teacher(jepa_encoding)
            forecast_output = self.student.forecast_decode(B, enc_out_list, x_list)
            if return_all:
                return forecast_output, jepa_encoding, predicted_teacher_encoding
            return forecast_output

        student_encoding, means, stdev = self.student.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        predicted_teacher_encoding = self._predict_teacher(self._to_jepa_encoding(student_encoding))
        forecast_output = self.student.forecast_decode(student_encoding, means, stdev)
        
        if return_all:
            return forecast_output, student_encoding, predicted_teacher_encoding
        
        return forecast_output

    def student_forward_dual_encoder(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Student forward pass with dual encoders.
        Returns predictions and optionally encodings.
        """
        if self.encoding_type == '4D_timemixer':
            jepa_enc1, enc_list1, x_list1, B1 = self._timemixer_encode(
                self.student1, x_enc, x_mark_enc, x_dec, x_mark_dec)
            jepa_enc2, enc_list2, x_list2, B2 = self._timemixer_encode(
                self.student2, x_enc, x_mark_enc, x_dec, x_mark_dec)
            predicted_teacher_encoding = self._predict_teacher(jepa_enc2)
            combined_encoding = self.encoder_fusion(jepa_enc1, jepa_enc2)
            fused_enc_list = self._inject_fused_into_enc_list(enc_list1, combined_encoding, B1)
            forecast_output = self.student1.forecast_decode(B1, fused_enc_list, x_list1)
            if return_all:
                return forecast_output, predicted_teacher_encoding
            return forecast_output

        student_encoding1, means1, stdev1 = self.student1.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        student_encoding2, means2, stdev2 = self.student2.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        predicted_teacher_encoding = self._predict_teacher(self._to_jepa_encoding(student_encoding2))

        combined_encoding = self.encoder_fusion(student_encoding1, student_encoding2)
        forecast_output = self.student1.forecast_decode(combined_encoding, means1, stdev1)

        if return_all:
            return forecast_output, predicted_teacher_encoding
        
        return forecast_output
    
    def _dino_embedding(self, x_enc):
        """Global (per-sample) frozen DINO vector [B, teacher_dim] for DINO-DIRECT.

        Handles multi-rendering (averages across renderings) and per-variable
        teachers (mean-pools over the variable axis) so the result is a single
        [B, teacher_dim] vector regardless of teacher configuration.
        """
        emb = self.teacher_forward(x_enc)          # tensor or list (multi-rendering)
        if isinstance(emb, list):
            emb = torch.stack(emb, dim=0).mean(dim=0)
        if emb.dim() == 3:                          # per-var [B, N, teacher_dim]
            emb = emb.mean(dim=1)                   # -> [B, teacher_dim]
        return emb

    def student_forward_dino_direct(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        DINO-DIRECT forecast (single student): inject the frozen DINO embedding
        directly into the fusion/forecast head (no JEPA distillation). DINO is a
        live input, so it is computed here at BOTH train and inference. Returns
        (forecast, None) when return_all so the loop knows there is no JEPA loss.
        """
        dino_proj = self.dino_projector(self._dino_embedding(x_enc))  # [B, d_model]

        if self.encoding_type == '4D_timemixer':
            jepa_enc, enc_list, x_list, B = self._timemixer_encode(
                self.student, x_enc, x_mark_enc, x_dec, x_mark_dec)
            N, D, T = jepa_enc.shape[1], jepa_enc.shape[2], jepa_enc.shape[3]
            dino_enc = dino_proj.view(B, 1, D, 1).expand(B, N, D, T)
            combined_encoding = self.dino_fusion_module(jepa_enc, dino_enc)
            fused_enc_list = self._inject_fused_into_enc_list(enc_list, combined_encoding, B)
            forecast_output = self.student.forecast_decode(B, fused_enc_list, x_list)
            if return_all:
                return forecast_output, None
            return forecast_output

        student_encoding, means, stdev = self.student.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        Np, D = student_encoding.shape[1], student_encoding.shape[2]
        dino_enc = dino_proj.unsqueeze(1).expand(-1, Np, D)   # [B, N', d_model]
        combined_encoding = self.dino_fusion_module(student_encoding, dino_enc)
        forecast_output = self.student.forecast_decode(combined_encoding, means, stdev)
        if return_all:
            return forecast_output, None
        return forecast_output

    def student_forward_no_dino(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        NO-DINO forecast: two student encoders fused, then decode. Task loss only
        (no teacher, no predictor, no JEPA). Same fuse+decode as the dual encoder,
        just without the teacher-alignment branch. Returns (forecast, None) when
        return_all so the training loop can detect 'no JEPA loss'.
        """
        if self.encoding_type == '4D_timemixer':
            jepa_enc1, enc_list1, x_list1, B1 = self._timemixer_encode(
                self.student1, x_enc, x_mark_enc, x_dec, x_mark_dec)
            jepa_enc2, enc_list2, x_list2, B2 = self._timemixer_encode(
                self.student2, x_enc, x_mark_enc, x_dec, x_mark_dec)
            combined_encoding = self.encoder_fusion(jepa_enc1, jepa_enc2)
            fused_enc_list = self._inject_fused_into_enc_list(enc_list1, combined_encoding, B1)
            forecast_output = self.student1.forecast_decode(B1, fused_enc_list, x_list1)
            if return_all:
                return forecast_output, None
            return forecast_output

        student_encoding1, means1, stdev1 = self.student1.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        student_encoding2, means2, stdev2 = self.student2.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)

        combined_encoding = self.encoder_fusion(student_encoding1, student_encoding2)
        forecast_output = self.student1.forecast_decode(combined_encoding, means1, stdev1)

        if return_all:
            return forecast_output, None
        return forecast_output

    def student_forward_multi_encoder(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Multi-encoder forward: k student encoders, fused for decode.
        
        Two modes controlled by use_multi_predictor:
          - multi_predictor=True  (opt 3): predictor_i(student_encoding_i) → z'_i
          - multi_predictor=False (opt 4): predictor(fused_encoding) → z' (shared)
        """
        if self.encoding_type == '4D_timemixer':
            tm_results = [
                self._timemixer_encode(s, x_enc, x_mark_enc, x_dec, x_mark_dec)
                for s in self.students_multi
            ]
            jepa_encodings = [r[0] for r in tm_results]
            fused_jepa = torch.stack(jepa_encodings, dim=0).mean(dim=0)

            if self.use_multi_predictor:
                predicted_teacher = [
                    pred(enc) for pred, enc in zip(self.predictors, jepa_encodings)
                ]
            else:
                predicted_teacher = self.predictor(fused_jepa)

            enc_list_0, x_list_0, B0 = tm_results[0][1], tm_results[0][2], tm_results[0][3]
            fused_enc_list = self._inject_fused_into_enc_list(enc_list_0, fused_jepa, B0)
            forecast_output = self.students_multi[0].forecast_decode(B0, fused_enc_list, x_list_0)

            if return_all:
                return forecast_output, jepa_encodings, predicted_teacher
            return forecast_output

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
                pred(self._to_jepa_encoding(enc))
                for pred, enc in zip(self.predictors, student_encodings)
            ]
        else:
            # Option 4: shared predictor on fused encoding
            predicted_teacher = self.predictor(self._to_jepa_encoding(fused_encoding))
        
        # Decode using first student's decoder and normalisation stats
        means_0, stdev_0 = results[0][1], results[0][2]
        forecast_output = self.students_multi[0].forecast_decode(
            fused_encoding, means_0, stdev_0)
        
        if return_all:
            return forecast_output, student_encodings, predicted_teacher
        
        return forecast_output
    
    def student_forward_multi_dual(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Multi-encoder + dual encoder: K JEPA encoders (one per rendering) + 1 forecast encoder.
        All K+1 encodings are fused for decode. JEPA encoders align with their respective teachers.
        """
        if self.encoding_type == '4D_timemixer':
            # Forecast encoder
            fc_jepa, fc_enc_list, fc_x_list, fc_B = self._timemixer_encode(
                self.student_forecast, x_enc, x_mark_enc, x_dec, x_mark_dec)
            # K JEPA encoders
            tm_results = [
                self._timemixer_encode(s, x_enc, x_mark_enc, x_dec, x_mark_dec)
                for s in self.students_multi
            ]
            jepa_encodings = [r[0] for r in tm_results]

            if self.use_multi_predictor:
                predicted_teacher = [
                    pred(enc) for pred, enc in zip(self.predictors, jepa_encodings)
                ]
            else:
                fused_jepa = torch.stack(jepa_encodings, dim=0).mean(dim=0)
                predicted_teacher = self.predictor(fused_jepa)

            all_encs = [fc_jepa] + jepa_encodings
            fused = self.encoder_fusion(all_encs)
            fused_enc_list = self._inject_fused_into_enc_list(fc_enc_list, fused, fc_B)
            forecast_output = self.student_forecast.forecast_decode(fc_B, fused_enc_list, fc_x_list)

            if return_all:
                return forecast_output, jepa_encodings, predicted_teacher
            return forecast_output

        # Non-TimeMixer path
        fc_enc, fc_means, fc_stdev = self.student_forecast.encode(
            x_enc, x_mark_enc, x_dec, x_mark_dec)
        results = [
            s.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
            for s in self.students_multi
        ]
        jepa_encodings = [r[0] for r in results]

        if self.use_multi_predictor:
            predicted_teacher = [
                pred(self._to_jepa_encoding(enc))
                for pred, enc in zip(self.predictors, jepa_encodings)
            ]
        else:
            fused_jepa = torch.stack(jepa_encodings, dim=0).mean(dim=0)
            predicted_teacher = self.predictor(self._to_jepa_encoding(fused_jepa))

        all_encs = [fc_enc] + jepa_encodings
        fused = self.encoder_fusion(all_encs)
        forecast_output = self.student_forecast.forecast_decode(fused, fc_means, fc_stdev)

        if return_all:
            return forecast_output, jepa_encodings, predicted_teacher
        return forecast_output

    def _encode_cls(self, student, x_enc):
        """Encode a batch for classification.

        For channel-independent TimesNet, fold the variables into the batch so
        each variable is embedded/encoded as its own univariate series, then
        return a per-variable 4D encoding [B, N, d_model, T]. For every other
        backbone this just calls the student's native encode (e.g. TimesNet
        channel-mixed [B, T, d_model]).
        """
        if self.timesnet_ci:
            B, T, N = x_enc.shape
            x = x_enc.permute(0, 2, 1).reshape(B * N, T, 1)  # [B*N, T, 1]
            enc = student.encode(x)                          # [B*N, T, d_model]
            D = enc.shape[-1]
            enc = enc.reshape(B, N, T, D).permute(0, 1, 3, 2)  # [B, N, d_model, T]
            return enc
        if self.student_model_name == 'iTransformer':
            # iTransformer is inverted: classification_encode returns one token
            # per variable [B, N, d_model]. Add a trailing patch dim so it reuses
            # the 4D per-var predictor / fusion / classification head machinery
            # (patch_num=1). Each variable token then aligns with its own DINO
            # per-variable teacher embedding.
            enc = student.classification_encode(x_enc, None)  # [B, N, d_model]
            return enc.unsqueeze(-1)                          # [B, N, d_model, 1]
        return student.encode(x_enc)

    def classification_forward(self, x_enc, padding_mask=None, return_all=False):
        """
        Classification forward pass
        x_enc: [batch, seq_len, n_vars]
        padding_mask: [batch, seq_len] (optional)
        Returns: class logits [batch, num_classes]
        """
        # Get student encoding 
        student_encoding = self._encode_cls(self.student, x_enc)
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

        student_encoding1 = self._encode_cls(self.student1, x_enc)
        student_encoding2 = self._encode_cls(self.student2, x_enc)
        
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
        student_encodings = [self._encode_cls(s, x_enc) for s in self.students_multi]
        
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
    
    def classification_forward_multi_dual(self, x_enc, padding_mask=None, return_all=False):
        """
        Multi-encoder + dual encoder classification:
        K JEPA encoders + 1 forecast encoder, all fused for classification.
        """
        fc_encoding = self._encode_cls(self.student_forecast, x_enc)
        jepa_encodings = [self._encode_cls(s, x_enc) for s in self.students_multi]

        if self.use_multi_predictor:
            predicted_teacher = [
                pred(enc) for pred, enc in zip(self.predictors, jepa_encodings)
            ]
        else:
            fused_jepa = torch.stack(jepa_encodings, dim=0).mean(dim=0)
            predicted_teacher = self.predictor(fused_jepa)

        all_encs = [fc_encoding] + jepa_encodings
        fused = self.encoder_fusion(all_encs)
        class_logits = self.classification_head(fused)

        if return_all and self.training:
            return class_logits, jepa_encodings, predicted_teacher
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
            padding_mask = x_mark_enc
            if self.use_multi_encoder and self.use_dual_encoder:
                return self.classification_forward_multi_dual(
                    x_enc, padding_mask, return_all=self.training)
            elif self.use_multi_encoder:
                return self.classification_forward_multi_encoder(
                    x_enc, padding_mask, return_all=self.training)
            elif self.use_dual_encoder:
                return self.classification_forward_dual_encoder(
                    x_enc, padding_mask, return_all=self.training)
            else:
                return self.classification_forward(
                    x_enc, padding_mask, return_all=self.training)
        
        # Forecasting task
        if self.no_dino:
            return self.student_forward_no_dino(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
        if self.dino_direct:
            return self.student_forward_dino_direct(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
        if self.use_multi_encoder and self.use_dual_encoder:
            return self.student_forward_multi_dual(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
        elif self.use_multi_encoder:
            return self.student_forward_multi_encoder(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
        elif self.use_dual_encoder:
            return self.student_forward_dual_encoder(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
        else:
            return self.student_forward(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=self.training)
