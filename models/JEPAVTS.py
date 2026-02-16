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
    """
    
    def __init__(self, dataset_name="ETTh1", hidden_size=768):
        super().__init__()
        self.cache_dir = f"./dataset/ETT-small/dino_embeddings_{dataset_name}"
        self.hidden_size = hidden_size
        
        if not os.path.exists(self.cache_dir):
            raise ValueError(f"❌ Cache not found: {self.cache_dir}\n   Run precompute_embeddings.py first!")
        
        n_cached = len([f for f in os.listdir(self.cache_dir) if f.endswith('.npy')])
        print(f"✅ VisionTSTeacher: Loading from cache")
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
    Fuses two encoder outputs using MLP
    """
    def __init__(self, d_model, fusion_type='mlp', hidden_factor=2):
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
                # 3D input: [bs, seq_len, d_model] (TimesNet)
                concat = torch.cat([enc1, enc2], dim=-1)  # [bs, seq_len, d_model*2]
                bs, seq_len, d_model_2 = concat.shape
                concat_flat = concat.reshape(-1, d_model_2)
                fused_flat = self.fusion(concat_flat)
                fused = fused_flat.reshape(bs, seq_len, -1)
                return fused
            else:
                # 4D input: [bs, nvars, d_model, patch_num] (PatchTST)
                concat = torch.cat([enc1, enc2], dim=2)  # [bs, nvars, d_model*2, patch_num]
                bs, nvars, d_model_2, patch_num = concat.shape
                concat_flat = concat.permute(0, 1, 3, 2).reshape(-1, d_model_2)
                fused_flat = self.fusion(concat_flat)
                fused = fused_flat.reshape(bs, nvars, patch_num, -1).permute(0, 1, 3, 2)
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
        self.dino_direct = getattr(configs, 'dino_direct', False)
        self.dino_fusion = getattr(configs, 'dino_fusion', False)
        self.dual_encoder_only = getattr(configs, 'dual_encoder_only', False)
        
        print("\n" + "="*50)
        print("Initializing JEPAVTS Model")
        if self.dual_encoder_only:
            print("Architecture: DUAL ENCODER ONLY (No DINO, No Teacher, No JEPA) 🔄")
        elif self.dino_fusion:
            print("Architecture: DINO FUSION (Student ⊕ DINO → Fusion) 🔗")
        elif self.dino_direct:
            print("Architecture: DINO DIRECT (Student + DINO concat) 🎯")
        elif self.use_dual_encoder:
            print("Architecture: DUAL ENCODER 🔀")
        else:
            print("Architecture: SINGLE ENCODER →")
        print("="*50)
        
        # Get student model name
        student_model_name = getattr(configs, 'student_model', 'PatchTST')
        
        # Teacher: VisionTS (frozen) - skip for dual_encoder_only mode
        if not self.dual_encoder_only:
            print(f"\n📊 Loading teacher vision encoder...")
            self.data = getattr(configs, 'data')
            self.teacher = VisionTSTeacher(dataset_name=self.data)
            self.teacher_dim = self.teacher.hidden_size
            print(f"✅ Teacher dimension: {self.teacher_dim}")
        else:
            print(f"\n📊 Skipping teacher (dual_encoder_only mode)")
            self.teacher = None
            self.teacher_dim = 768  # Default, not used
        
        # Student: Time Series Encoder (trainable)
        print(f"\n🎓 Building student model: {student_model_name}")
        self.student = self._build_student_model(student_model_name, configs)
        
        if self.use_dual_encoder or self.dual_encoder_only:
            self.student1 = self._build_student_model(student_model_name, configs)
            self.student2 = self._build_student_model(student_model_name, configs)

            self.encoder_fusion = EncodingFusion(d_model=configs.d_model,
                                            fusion_type=getattr(configs, 'fusion_type', 'mlp')  # mlp, weighted, or add
                                        )
            print(f"✅ Dual Encoder Fusion: {getattr(configs, 'fusion_type', 'mlp')}")
        
        # DINO Direct/Fusion modes: project DINO to student dim
        if self.dino_direct or self.dino_fusion:
            self.dino_projector = nn.Sequential(
                nn.Linear(768, configs.d_model),  # 768 = DINO dim
                nn.LayerNorm(configs.d_model),
                nn.GELU(),
            )
            print(f"✅ DINO Projector: 768 -> {configs.d_model}")
        
        # DINO Fusion: also needs the fusion module
        if self.dino_fusion:
            self.dino_encoder_fusion = EncodingFusion(
                d_model=configs.d_model,
                fusion_type=getattr(configs, 'fusion_type', 'mlp')
            )
            print(f"✅ DINO-Student Fusion: {getattr(configs, 'fusion_type', 'mlp')}")
        
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

        # JEPA Predictor (trainable) - skip for dual_encoder_only mode
        if not self.dual_encoder_only:
            print(f"\n🔗 Building JEPA predictor...")
            self.predictor = JEPAPredictor(
                    student_dim=configs.d_model,
                    teacher_dim=self.teacher_dim,
                    hidden_dim=getattr(configs, 'jepa_hidden_dim', 512),
                    input_shape=predictor_input_shape
                )
            print(f"✅ JEPA predictor: {self.student_dim} -> {self.teacher_dim}")
        else:
            print(f"\n🔗 Skipping JEPA predictor (dual_encoder_only mode)")
            self.predictor = None
        
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
            
            if self.dino_direct:
                # DINO Direct mode: concat student encoding + DINO embedding
                # DINO embedding is 768-dim
                concat_dim = flatten_dim + self.teacher_dim
                self.classification_head = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(concat_dim, configs.d_model * 2),
                    nn.LayerNorm(configs.d_model * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(configs.d_model * 2, self.num_classes)
                )
                print(f"✅ Classification head (DINO Direct): {flatten_dim} + {self.teacher_dim} -> {self.num_classes} classes")
            else:
                self.classification_head = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(flatten_dim, configs.d_model),
                    nn.LayerNorm(configs.d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(configs.d_model, self.num_classes)
                )
                print(f"✅ Classification head: {flatten_dim} -> {self.num_classes} classes")
        
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
    
    def teacher_forward(self, x_enc):
        """
        Teacher forward pass (always with no_grad)
        x_enc: [batch_size, seq_len, n_vars] - raw time series
        Returns: teacher_encoding [batch_size, teacher_dim]
        """
        with torch.no_grad():
            teacher_encoding = self.teacher(x_enc)
        return teacher_encoding
    
    def student_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Student forward pass
        Returns predictions and optionally encodings
        """
        # Get student encoding
        student_encoding, means, stdev = self.student.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Predict teacher encoding via JEPA
        predicted_teacher_encoding = self.predictor(student_encoding)
        forecast_output = self.student.forecast_decode(student_encoding, means, stdev)
        
        if return_all:
            return forecast_output, student_encoding, predicted_teacher_encoding
        
        return forecast_output

    def student_forward_dino_direct(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        DINO Direct for forecasting: concat student encoding with DINO embedding
        Then project to forecast. NO JEPA loss.
        """
        # Get student encoding
        student_encoding, means, stdev = self.student.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Get DINO embedding
        with torch.no_grad():
            dino_embedding = self.teacher(x_enc)  # [B, 768]
        
        # For forecasting with dino_direct, we add DINO info to each position
        # Project DINO to student dimension and add to encoding
        dino_projected = self.dino_projector(dino_embedding)  # [B, d_model]
        
        # Expand DINO to match student encoding shape and add
        if student_encoding.dim() == 4:
            # PatchTST: [B, nvars, d_model, patch_num]
            bs, nvars, d_model, patch_num = student_encoding.shape
            dino_expanded = dino_projected.unsqueeze(1).unsqueeze(-1).expand(-1, nvars, -1, patch_num)
        else:
            # TimesNet: [B, T, d_model]
            bs, seq_len, d_model = student_encoding.shape
            dino_expanded = dino_projected.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Simple addition (concat in feature space would change dimensions)
        combined_encoding = student_encoding + dino_expanded
        
        # Decode forecast
        forecast_output = self.student.forecast_decode(combined_encoding, means, stdev)
        
        if return_all:
            return forecast_output, None, None  # No JEPA loss
        
        return forecast_output

    def student_forward_dino_fusion(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        DINO Fusion for forecasting: fuse student encoding with projected DINO
        NO JEPA loss
        """
        # Get student encoding
        student_encoding, means, stdev = self.student.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Get DINO embedding
        with torch.no_grad():
            dino_embedding = self.teacher(x_enc)  # [B, 768]
        
        # Project DINO to student dimension
        dino_projected = self.dino_projector(dino_embedding)  # [B, d_model]
        
        # Expand DINO to match student encoding shape
        if student_encoding.dim() == 4:
            # PatchTST: [B, nvars, d_model, patch_num]
            bs, nvars, d_model, patch_num = student_encoding.shape
            dino_expanded = dino_projected.unsqueeze(1).unsqueeze(-1).expand(-1, nvars, -1, patch_num)
        else:
            # TimesNet: [B, T, d_model]
            bs, seq_len, d_model = student_encoding.shape
            dino_expanded = dino_projected.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Fuse student + DINO
        fused_encoding = self.dino_encoder_fusion(student_encoding, dino_expanded)
        
        # Decode forecast
        forecast_output = self.student.forecast_decode(fused_encoding, means, stdev)
        
        if return_all:
            return forecast_output, None, None  # No JEPA loss
        
        return forecast_output

    def student_forward_dual_encoder(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Student forward pass
        Returns predictions and optionally encodings
        """
        # Get student encoding
        student_encoding1, means1, stdev1 = self.student1.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        student_encoding2, means2, stdev2 = self.student2.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        predicted_teacher_encoding = self.predictor(student_encoding2)

        combined_encoding = self.encoder_fusion(student_encoding1, student_encoding2)
        forecast_output = self.student1.forecast_decode(combined_encoding, means1, stdev1)

        if return_all:
            return forecast_output, predicted_teacher_encoding
        
        return forecast_output

    def student_forward_dual_encoder_only(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False):
        """
        Dual encoder fusion ONLY - no DINO, no teacher, no JEPA loss
        Two PatchTST encoders fused together, task loss only
        """
        # Get encodings from both students
        student_encoding1, means1, stdev1 = self.student1.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        student_encoding2, means2, stdev2 = self.student2.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Fuse encodings
        combined_encoding = self.encoder_fusion(student_encoding1, student_encoding2)
        
        # Decode forecast
        forecast_output = self.student1.forecast_decode(combined_encoding, means1, stdev1)

        if return_all:
            # Return None for predicted_teacher_encoding to indicate no JEPA loss
            return forecast_output, None, None
        
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
            # Get JEPA prediction for teacher alignment
            predicted_teacher_encoding = self.predictor(student_encoding)
            return class_logits, student_encoding, predicted_teacher_encoding
        
        return class_logits

    def classification_forward_dino_direct(self, x_enc, padding_mask=None, return_all=False):
        """
        DINO Direct mode: concatenate student encoding with DINO embedding
        NO JEPA loss - DINO embedding used directly
        """
        # Get student encoding
        student_encoding = self.student.encode(x_enc)  # [B, T, d_model]
        
        # Get DINO embedding directly from teacher (no JEPA predictor)
        with torch.no_grad():
            dino_embedding = self.teacher(x_enc)  # [B, 768]
        
        # Flatten student encoding
        student_flat = student_encoding.reshape(student_encoding.shape[0], -1)  # [B, T*d_model]
        
        # Concatenate student + DINO
        concat_features = torch.cat([student_flat, dino_embedding], dim=-1)  # [B, T*d_model + 768]
        
        # Classification (head expects already flattened input for dino_direct)
        # But our head has Flatten as first layer, so we need to add dummy dim
        concat_features = concat_features.unsqueeze(1)  # [B, 1, concat_dim]
        class_logits = self.classification_head(concat_features)
        
        if return_all and self.training:
            # No JEPA loss in dino_direct mode, return None for compatibility
            return class_logits, None, None
        
        return class_logits

    def classification_forward_dino_fusion(self, x_enc, padding_mask=None, return_all=False):
        """
        DINO Fusion mode: fuse student encoding with projected DINO embedding
        Uses EncodingFusion to combine student and DINO (DINO as virtual second encoder)
        NO JEPA loss
        """
        # Get student encoding
        student_encoding = self.student.encode(x_enc)  # [B, T, d_model]
        
        # Get DINO embedding
        with torch.no_grad():
            dino_embedding = self.teacher(x_enc)  # [B, 768]
        
        # Project DINO to student dimension
        dino_projected = self.dino_projector(dino_embedding)  # [B, d_model]
        
        # Expand DINO to match student encoding shape
        # student_encoding: [B, T, d_model]
        # dino_projected: [B, d_model] -> [B, T, d_model] (broadcast across time)
        batch_size, seq_len, d_model = student_encoding.shape
        dino_expanded = dino_projected.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, d_model]
        
        # Fuse student encoding with DINO (using EncodingFusion)
        fused_encoding = self.dino_encoder_fusion(student_encoding, dino_expanded)  # [B, T, d_model]
        
        # Classification output
        class_logits = self.classification_head(fused_encoding)
        
        if return_all and self.training:
            # No JEPA loss in dino_fusion mode
            return class_logits, None, None
        
        return class_logits
    
    def classification_forward_dual_encoder(self, x_enc, padding_mask=None, return_all=False):
        """
        Classification forward pass with dual encoder
        """

        student_encoding1 = self.student1.encode(x_enc)
        student_encoding2 = self.student2.encode(x_enc)
        
        # JEPA prediction from encoder2
        predicted_teacher_encoding = self.predictor(student_encoding2)
        
        # Fuse encodings
        combined_encoding = self.encoder_fusion(student_encoding1, student_encoding2)
        
        # Classification output
        class_logits = self.classification_head(combined_encoding)
        
        if return_all and self.training:
            return class_logits, predicted_teacher_encoding
        
        return class_logits

    def classification_forward_dual_encoder_only(self, x_enc, padding_mask=None, return_all=False):
        """
        Dual encoder fusion ONLY for classification - no DINO, no teacher, no JEPA loss
        Two PatchTST encoders fused together, task loss only
        """
        # Get encodings from both students
        student_encoding1 = self.student1.encode(x_enc)
        student_encoding2 = self.student2.encode(x_enc)
        
        # Fuse encodings
        combined_encoding = self.encoder_fusion(student_encoding1, student_encoding2)
        
        # Classification output
        class_logits = self.classification_head(combined_encoding)
        
        if return_all and self.training:
            # Return None to indicate no JEPA loss
            return class_logits, None, None
        
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
            
            # Dual Encoder Only mode (no DINO, no teacher, no JEPA loss)
            if self.dual_encoder_only:
                if self.training:
                    return self.classification_forward_dual_encoder_only(x_enc, padding_mask, return_all=True)
                else:
                    return self.classification_forward_dual_encoder_only(x_enc, padding_mask, return_all=False)
            # DINO Fusion mode (Student ⊕ DINO → Fusion, no JEPA loss)
            elif self.dino_fusion:
                if self.training:
                    return self.classification_forward_dino_fusion(x_enc, padding_mask, return_all=True)
                else:
                    return self.classification_forward_dino_fusion(x_enc, padding_mask, return_all=False)
            # DINO Direct mode (concat, no JEPA loss)
            elif self.dino_direct:
                if self.training:
                    return self.classification_forward_dino_direct(x_enc, padding_mask, return_all=True)
                else:
                    return self.classification_forward_dino_direct(x_enc, padding_mask, return_all=False)
            # Dual Encoder mode (with JEPA loss)
            elif self.use_dual_encoder:
                if self.training:
                    return self.classification_forward_dual_encoder(x_enc, padding_mask, return_all=True)
                else:
                    return self.classification_forward_dual_encoder(x_enc, padding_mask, return_all=False)
            # Single Encoder mode (with JEPA loss)
            else:
                if self.training:
                    return self.classification_forward(x_enc, padding_mask, return_all=True)
                else:
                    return self.classification_forward(x_enc, padding_mask, return_all=False)
        
        # Forecasting task
        # Dual Encoder Only mode (no DINO, no teacher, no JEPA loss)
        if self.dual_encoder_only:
            if self.training:
                return self.student_forward_dual_encoder_only(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=True)
            else:
                return self.student_forward_dual_encoder_only(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False)
        # DINO Direct mode (student + DINO add, no JEPA loss)
        elif self.dino_direct:
            if self.training:
                return self.student_forward_dino_direct(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=True)
            else:
                return self.student_forward_dino_direct(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False)
        # DINO Fusion mode (student ⊕ DINO fusion, no JEPA loss)
        elif self.dino_fusion:
            if self.training:
                return self.student_forward_dino_fusion(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=True)
            else:
                return self.student_forward_dino_fusion(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False)
        # Dual Encoder mode (with JEPA loss)
        elif self.use_dual_encoder:
            if self.training:
                return self.student_forward_dual_encoder(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=True)
            else:
                return self.student_forward_dual_encoder(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False)
        # Single Encoder mode (with JEPA loss)
        else:
            if self.training:
                return self.student_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=True)
            else:
                return self.student_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False)
