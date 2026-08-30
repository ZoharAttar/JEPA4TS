"""
Masked-horizon teacher: VisionTS-style image (left = lookback x, right = zeros)
fed to a frozen MAE. Target is the mean-pooled decoder latent of the masked
(right / future) patches.

Not the same rendering as the DINO teacher (RP/GAF/...). Image layout follows
Keytoyze/VisionTS (visionts/model.py).
"""

import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mae_vit import mae_vit_base_patch16

MAE_CKPT_NAME = 'mae_visualize_vit_base.pth'
MAE_CKPT_URL = 'https://dl.fbaipublicfiles.com/mae/visualize/' + MAE_CKPT_NAME

# MAE-base decoder width (not the 768 encoder dim).
HORIZON_DIM = 512


def _ensure_mae_checkpoint(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, MAE_CKPT_NAME)
    if os.path.isfile(path):
        return path
    print(f"Downloading MAE checkpoint to {path}")
    print(f"  from {MAE_CKPT_URL}")
    try:
        urllib.request.urlretrieve(MAE_CKPT_URL, path)
    except Exception as e:
        if os.path.isfile(path):
            os.remove(path)
        raise RuntimeError(
            f"Failed to download MAE weights. Save {MAE_CKPT_NAME} under {ckpt_dir} "
            f"manually from {MAE_CKPT_URL}"
        ) from e
    return path


class MaskedHorizonTeacher(nn.Module):
    """
    Frozen MAE teacher. Forward:
        x_enc [B, seq_len, nvars]  (lookback only)
    Returns:
        per_var=True:  [B, nvars, 512]
        per_var=False: [B, 512]
    """

    def __init__(self, seq_len, pred_len, periodicity=1, align_const=0.4,
                 norm_const=0.4, ckpt_dir='./ckpt/', per_var=True,
                 chunk_size=32, interpolation='bilinear'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.align_const = align_const
        self.norm_const = norm_const
        self.per_var = per_var
        self.chunk_size = chunk_size
        self.hidden_size = HORIZON_DIM

        self.mae = mae_vit_base_patch16()
        ckpt_path = _ensure_mae_checkpoint(ckpt_dir)
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        missing, unexpected = self.mae.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  MAE load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        for p in self.mae.parameters():
            p.requires_grad = False
        self.mae.eval()

        img_size = self.mae.patch_embed.img_size
        self.image_size = img_size[0] if isinstance(img_size, (tuple, list)) else int(img_size)
        ps = self.mae.patch_embed.patch_size
        self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        self.num_patch = self.image_size // self.patch_size
        self._configure_mask(seq_len, pred_len)
        self.interp_mode = interpolation

        print("MaskedHorizonTeacher: frozen MAE-base")
        print(f"  seq_len={seq_len} pred_len={pred_len} periodicity={periodicity}")
        print(f"  num_patch_input={self.num_patch_input} num_patch_output={self.num_patch_output} "
              f"mask_ratio={self.mask_ratio:.3f}")
        print(f"  horizon_dim={self.hidden_size} per_var={per_var} chunk_size={chunk_size}")

    def _configure_mask(self, context_len, pred_len):
        periodicity = self.periodicity
        pad_left = 0
        pad_right = 0
        if context_len % periodicity != 0:
            pad_left = periodicity - context_len % periodicity
        if pred_len % periodicity != 0:
            pad_right = periodicity - pred_len % periodicity
        self.pad_left = pad_left
        self.pad_right = pad_right

        input_ratio = (pad_left + context_len) / (
            pad_left + context_len + pad_right + pred_len)
        num_patch_input = int(input_ratio * self.num_patch * self.align_const)
        if num_patch_input == 0:
            num_patch_input = 1
        self.num_patch_input = num_patch_input
        self.num_patch_output = self.num_patch - num_patch_input
        self.input_width = num_patch_input * self.patch_size
        self.output_width = self.num_patch_output * self.patch_size

        mask = torch.ones((self.num_patch, self.num_patch))
        mask[:, :num_patch_input] = 0
        self.register_buffer('struct_mask', mask.float().reshape(1, -1), persistent=False)
        self.mask_ratio = float(mask.mean().item())

    def train(self, mode=True):
        super().train(False)
        self.mae.eval()
        return self

    def _render(self, x_enc):
        """
        x_enc: [B, S, N] lookback.
        Returns image [B*N, 3, 224, 224] and noise [B*N, L].
        """
        B, S, N = x_enc.shape
        means = x_enc.mean(1, keepdim=True)
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = stdev / self.norm_const
        x = x / stdev
        x = x.permute(0, 2, 1)  # [B, N, S]
        x = F.pad(x, (self.pad_left, 0), mode='replicate')
        f = self.periodicity
        # [B, N, periods * f] -> [B*N, 1, f, periods]
        x_2d = x.reshape(B * N, 1, x.shape[-1] // f, f).permute(0, 1, 3, 2)
        x_resize = F.interpolate(
            x_2d, size=(self.image_size, self.input_width),
            mode=self.interp_mode, align_corners=False)
        zeros = x_resize.new_zeros(
            x_resize.shape[0], 1, self.image_size, self.output_width)
        canvas = torch.cat([x_resize, zeros], dim=-1)
        image = canvas.repeat(1, 3, 1, 1)
        noise = self.struct_mask.expand(image.shape[0], -1)
        return image, noise

    def forward(self, x_enc):
        """
        x_enc: [B, seq_len, nvars]
        """
        B, _, N = x_enc.shape
        image, noise = self._render(x_enc)
        pooled = []
        chunk = self.chunk_size if self.chunk_size and self.chunk_size > 0 else image.shape[0]
        with torch.no_grad():
            for start in range(0, image.shape[0], chunk):
                end = min(start + chunk, image.shape[0])
                pooled.append(self.mae.forward_masked_latents(
                    image[start:end], self.mask_ratio, noise[start:end]))
        latents = torch.cat(pooled, dim=0)  # [B*N, 512]
        latents = latents.reshape(B, N, -1)
        if self.per_var:
            return latents
        return latents.mean(dim=1)
