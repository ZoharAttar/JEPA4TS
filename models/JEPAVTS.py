import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTSTeacher(nn.Module):
    """
    Wrapper around VisionTS to use as frozen teacher encoder
    VisionTS handles time series to visual conversion internally
    """
    
    def __init__(self, seq_len, n_vars, frozen=True):
        super().__init__()
        
        try:
            from visionts import VisionTS
            print("Loading VisionTS model...")
            
            # Initialize VisionTS with checkpoint directory
            self.visionts = VisionTS(arch='mae_base', ckpt_dir='./ckpt/')
            
            print("✅ VisionTS loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Error loading VisionTS: {e}")
            import traceback
            traceback.print_exc()
            self.visionts = None
        
        # VisionTS outputs 768-dim embeddings (MAE base)
        self.hidden_size = 768
        self.seq_len = seq_len
        self.n_vars = n_vars
        
        # Hyperparameters for VisionTS (from their demo)
        # These control how VisionTS processes time series as images
        self.align_const = 0.4  # Alignment constant for image generation
        self.norm_const = 0.4   # Normalization constant
        self.periodicity = 1    # Periodicity (1 for non-seasonal data)
        
        # Create fallback projection
        self.fallback_proj = nn.Linear(n_vars, self.hidden_size)
        
        if frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
    
    def forward(self, x_enc):
        """
        x_enc: [batch_size, seq_len, n_vars] - raw time series
        Returns: [batch_size, hidden_size] - teacher encoding
        """
        with torch.no_grad():
            batch_size = x_enc.shape[0]
            
            if self.visionts is not None:
                try:
                    # ✅ Step 1: Update config (REQUIRED before each forward)
                    context_len = x_enc.shape[1]
                    pred_len = context_len  # Same length for encoding
                    
                    self.visionts.update_config(
                        context_len,
                        pred_len,
                        align_const=self.align_const,
                        norm_const=self.norm_const,
                        periodicity=self.periodicity
                    )
                    
                    # ✅ Step 2: Call VisionTS forward
                    # Returns predictions [batch, pred_len, n_vars]
                    y_pred = self.visionts.forward(x_enc)
                    
                    # ✅ Step 3: Convert predictions to embedding
                    # Pool across time and variables to get fixed-size embedding
                    embedding = y_pred.reshape(batch_size, -1)  # Flatten [batch, pred_len*n_vars]
                    
                    # Project to target hidden size
                    if not hasattr(self, 'visionts_proj'):
                        input_dim = embedding.shape[-1]
                        self.visionts_proj = nn.Linear(input_dim, self.hidden_size).to(x_enc.device)
                        self.visionts_proj.requires_grad = False
                    
                    embedding = self.visionts_proj(embedding)  # [batch, hidden_size]
                    
                    # print(f"✅ VisionTS working! Output: {embedding.shape}")
                    return embedding
                    
                except Exception as e:
                    print(f"⚠️ VisionTS failed: {e}, using fallback")
                    # Print traceback only once for debugging
                    if not hasattr(self, '_error_logged'):
                        import traceback
                        traceback.print_exc()
                        self._error_logged = True
            
            # Fallback: Simple pooling + projection
            pooled = x_enc.mean(dim=1)  # [batch, n_vars]
            embedding = self.fallback_proj(pooled)  # [batch, hidden_size]
            return embedding


class TimeSeriesEncoder(nn.Module):
    """
    Wrapper for any TS model to extract intermediate encodings
    """
    
    def __init__(self, base_model, d_model):
        super().__init__()
        self.base_model = base_model
        self.d_model = d_model
        self.encoding = None
        self.n_vars = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to extract encodings from base model"""
        
        # For PatchTST - extract after encoder
        if hasattr(self.base_model, 'encoder'):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    enc_out = output[0]
                else:
                    enc_out = output
                # enc_out shape: [batch * nvars, seq, d_model] for PatchTST
                # Global average pooling over sequence dimension
                self.encoding = enc_out.mean(dim=1)  # [batch * nvars, d_model]
            
            self.base_model.encoder.register_forward_hook(hook_fn)
            print("✅ Hook registered for encoder")
        
        # For models with nested encoder
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'encoder'):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    enc_out = output[0]
                else:
                    enc_out = output
                self.encoding = enc_out.mean(dim=1)
            
            self.base_model.model.encoder.register_forward_hook(hook_fn)
            print("✅ Hook registered for nested encoder")
        else:
            print("⚠️ No encoder found, will use alternative method")
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward and extract encoding
        Returns: encoding [batch_size, d_model * n_vars] or pooled [batch_size, d_model]
        """
        # Store n_vars for reshaping
        batch_size = x_enc.shape[0]
        self.n_vars = x_enc.shape[2]  # [batch, seq, n_vars]
        
        # Run forward pass (triggers hooks)
        _ = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.encoding is None:
            # Fallback if hook didn't work
            print("⚠️ Hook didn't capture encoding, using fallback")
            self.encoding = x_enc.mean(dim=1)  # [batch, n_vars]
            if self.encoding.shape[-1] != self.d_model:
                if not hasattr(self, 'fallback_proj'):
                    self.fallback_proj = nn.Linear(self.encoding.shape[-1], self.d_model).to(x_enc.device)
                self.encoding = self.fallback_proj(self.encoding)
            return self.encoding
        
        # For PatchTST: encoding is [batch * nvars, d_model]
        # Reshape to [batch, nvars, d_model] then pool or flatten
        if self.encoding.shape[0] == batch_size * self.n_vars:
            # Reshape: [batch * nvars, d_model] -> [batch, nvars, d_model]
            encoding_reshaped = self.encoding.view(batch_size, self.n_vars, self.d_model)
            # Pool across variables: [batch, nvars, d_model] -> [batch, d_model]
            final_encoding = encoding_reshaped.mean(dim=1)
        else:
            # Already in correct shape [batch, d_model]
            final_encoding = self.encoding
        
        return final_encoding


class JEPAPredictor(nn.Module):
    """
    JEPA predictor: maps student encoding to teacher encoding space
    """
    
    def __init__(self, student_dim, teacher_dim, hidden_dim=512, num_layers=2):
        super().__init__()
        
        layers = []
        current_dim = student_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        # Final layer to teacher dimension
        layers.append(nn.Linear(current_dim, teacher_dim))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, student_encoding):
        """
        student_encoding: [batch_size, student_dim]
        Returns: [batch_size, teacher_dim]
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
        print(f"\n📊 Loading VisionTS as frozen teacher...")
        self.teacher = VisionTSTeacher(
            seq_len=configs.seq_len,
            n_vars=configs.enc_in,
            frozen=True
        )
        self.teacher.eval()
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
        # self.student = TimeSeriesEncoder(self.student_base, configs.d_model)
        self.student_dim = configs.d_model
        print(f"✅ Student dimension: {self.student_dim}")

        
        
        # JEPA Predictor (trainable)
        print(f"\n🔗 Building JEPA predictor...")
        self.predictor = JEPAPredictor(
            student_dim=self.student_dim,
            teacher_dim=self.teacher_dim,
            hidden_dim=getattr(configs, 'jepa_hidden_dim', 512),
            num_layers=getattr(configs, 'jepa_num_layers', 2)
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