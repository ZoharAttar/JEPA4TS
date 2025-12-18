import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from utils.plotting import visionTS_plot
from transformers import ViTModel


class VisionTSTeacher(nn.Module):
    """
    Wrapper around VisionTS to use as frozen teacher encoder
    VisionTS handles time series to visual conversion internally
    """
    
    def __init__(self, vit_model='vit_base_patch16_224'):
        super().__init__()
        #load ViT
        self.vis_fm = ViTModel.from_pretrained("facebook/vit-mae-base")
        # self.vis_fm = timm.create_model(vit_model, pretrained=True)
        self.vis_fm.eval()
        for param in self.vis_fm.parameters():
            param.requires_grad = False

        # Get hidden dimension from ViT model
        # self.hidden_size = self.vis_fm.num_features
        self.hidden_size = self.vis_fm.config.hidden_size


        print(f"✅ ViT Teacher loaded: {vit_model}")
        print(f"✅ Teacher hidden_size: {self.hidden_size}")
        

    def forward(self, x_enc):
        x_image = visionTS_plot(x_enc)
        # teacher_encoding = self.vis_fm.forward_features(x_image)[:, 0, :]  # CLS token
        outputs = self.vis_fm(pixel_values=x_image)
        teacher_encoding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return teacher_encoding


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
        enc1, enc2: [bs, nvars, d_model, patch_num]
        Returns: [bs, nvars, d_model, patch_num]
        """
        if self.fusion_type == 'add':
            return enc1 + enc2
        
        elif self.fusion_type == 'weighted':
            alpha = torch.sigmoid(self.alpha)
            return alpha * enc1 + (1 - alpha) * enc2
        
        elif self.fusion_type == 'mlp':
            # Concatenate
            concat = torch.cat([enc1, enc2], dim=2)  # [bs, nvars, d_model*2, patch_num]
            
            # Reshape for MLP
            bs, nvars, d_model_2, patch_num = concat.shape
            concat_flat = concat.permute(0, 1, 3, 2).reshape(-1, d_model_2)
            
            # Fuse
            fused_flat = self.fusion(concat_flat)
            
            # Reshape back
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
        
        # Teacher: VisionTS (frozen)
        print(f"\n📊 Loading teacher vision encoder...")
        self.teacher = VisionTSTeacher(
            vit_model=getattr(configs, 'vit_model', 'vit_base_patch16_224')
        )
        self.teacher_dim = self.teacher.hidden_size
        print(f"✅ Teacher dimension: {self.teacher_dim}")
        
        # Student: Time Series Encoder (trainable)
        print(f"\n🎓 Building student model: {student_model_name}")
        self.student = self._build_student_model(student_model_name, configs)
        
        if self.use_dual_encoder:
            self.student1 = self._build_student_model(student_model_name, configs)
            self.student2 = self._build_student_model(student_model_name, configs)

            self.encoder_fusion = EncodingFusion(d_model=configs.d_model,
                                            fusion_type=getattr(configs, 'fusion_type', 'mlp')  # mlp, weighted, or add
                                        )
        self.student_dim = configs.d_model
        print(f"✅ Student dimension: {self.student_dim}")

        # Calculate patch_num dynamically based on student model type
        student_model_name = getattr(configs, 'student_model', 'PatchTST')

        if student_model_name == 'PatchTST' and hasattr(self.student, 'patch_embedding'):
            # Get values from PatchTST
            patch_len = self.student.patch_embedding.patch_len
            stride = self.student.patch_embedding.stride
            patch_num = int((configs.seq_len - patch_len) / stride + 2)
        else:
            # For non-patch models (DLinear, etc.), use 1D input
            patch_num = None

        # JEPA Predictor (trainable)
        print(f"\n🔗 Building JEPA predictor...")
        self.predictor = JEPAPredictor(
                student_dim=configs.d_model,
                teacher_dim=self.teacher_dim,
                hidden_dim=getattr(configs, 'jepa_hidden_dim', 512),
                input_shape=(configs.enc_in, configs.d_model, patch_num)  # 4D
            )
        print(f"✅ JEPA predictor: {self.student_dim} -> {self.teacher_dim}")
        
        # Prediction head for forecasting task
        self.forecast_head = nn.Linear(self.student_dim, configs.pred_len * configs.c_out)
        print(f"✅ Forecast head: {self.student_dim} -> {configs.pred_len * configs.c_out}")
        
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
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Main forward - behavior depends on training mode and architecture
        """
        # Choose architecture based on config
        if self.use_dual_encoder:
            # Use dual encoder architecture
            if self.training:
                return self.student_forward_dual_encoder(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=True)
            else:
                return self.student_forward_dual_encoder(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False)
        else:
            # Use single encoder architecture
            if self.training:
                return self.student_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=True)
            else:
                return self.student_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, return_all=False)
