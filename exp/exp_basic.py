import os
import torch
import numpy as np
from models import (
    Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer,
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, 
    FiLM, iTransformer, Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, 
    TemporalFusionTransformer, SCINet, PAttn, TimeXer, WPMixer, MultiPatchFormer, 
    KANAD, MSGNet, TimeFilter
)


class Exp_Basic(object):
    MODEL_DICT = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            'SCINet': SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
            'MSGNet': MSGNet,
            'TimeFilter': TimeFilter
        }

    def __init__(self, args):
        self.args = args
        self.model_dict = Exp_Basic.MODEL_DICT
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _make_noise_ctx(self, test_data):
        """Build a test-input-noise context, or None when --test_noise <= 0.

        Robustness eval: additive Gaussian noise is added to the input window
        (x_enc) only. For each variable the noise std is
            (test_noise / 100) * std(variable over the test split).
        Deterministic given --seed so a given noise level is reproducible.
        Returns (noise_pct, std_vec[1,1,N], generator) or None.
        """
        noise_pct = float(getattr(self.args, 'test_noise', 0.0) or 0.0)
        if noise_pct <= 0:
            return None
        arr = np.asarray(test_data.data_x, dtype=np.float32)  # [T, N] (normalized)
        std = np.std(arr, axis=0)  # per-variable std over the test split
        std_vec = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 1, -1)
        seed = int(getattr(self.args, 'seed', 2021))
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        print(f"🌫️  Test-input noise ENABLED: std = {noise_pct:g}% of per-variable std "
              f"(input window only, seed={seed})")
        return noise_pct, std_vec, gen

    def _apply_noise(self, batch_x, noise_ctx):
        """Add the pre-built Gaussian noise to a test input batch (no-op if None)."""
        if noise_ctx is None:
            return batch_x
        noise_pct, std_vec, gen = noise_ctx
        noise = torch.randn(batch_x.shape, generator=gen,
                            device=batch_x.device, dtype=batch_x.dtype)
        return batch_x + (noise_pct / 100.0) * std_vec * noise

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
