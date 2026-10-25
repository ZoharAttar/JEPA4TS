"""
VTSDecoderProbe: can a time-series encoder drive VisionTS++'s pixel decoder?

Pipeline (student replaces VisionTS++'s ViT encoder):

    TS --[PatchTST encoder]--> features --[MLP]--> visible-patch latent
       --[FROZEN VisionTS++ decoder]--> pixels --[VisionTS++ un-render]--> forecast

Only the PatchTST encoder, the MLP, and a learnable CLS token are trained; the
VisionTS++ decoder (and its deterministic un-render + denorm) are frozen. The
loss is the plain forecast MSE (handled by Exp_Long_Term_Forecast), i.e. the
same objective as the other forecasting runs. If the frozen decoder can turn the
student's mapped features into good forecasts, the student encoder already
carried the "visual information" VisionTS++ needs.

Rendering modes (two experiments):
    vts_render_mode = 'ci' -> per-variable grayscale image (color=False, nvars=1
                             per image; student features mapped per variable)
    vts_render_mode = 'mv' -> colorized multivariate image (color=True, all vars
                             in one image; whole-sample features mapped)

How the injection works: VisionTS++'s MAE forward is
    latent, mask, ids = forward_encoder(img);  pred = forward_decoder(latent, ids)
and its internal reconstruction loss is disabled (returns None, pred, mask). We
temporarily replace `forward_encoder` with a closure returning OUR latent (from
the student+MLP) and the fixed `ids_restore`, then call VisionTS++'s own forward
so all the rendering / un-render / denorm / quantile logic is reused unchanged.

Requires: pip install git+https://github.com/Keytoyze/VisionTS.git
"""

import os
import torch
import torch.nn as nn
import einops

from models import PatchTST


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.render_mode = getattr(configs, 'vts_render_mode', 'ci')
        assert self.render_mode in ('ci', 'mv')

        # ---- Student encoder: PatchTST (we use only .encode) ----
        self.student = PatchTST.Model(configs)
        patch_len = self.student.patch_embedding.patch_len
        stride = self.student.patch_embedding.stride
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)

        # ---- Frozen VisionTS++ decoder ----
        from visionts import VisionTSpp
        arch = getattr(configs, 'vts_arch', 'mae_base')
        ckpt_dir = getattr(configs, 'vts_ckpt_dir', './hf_models/VisionTSpp')
        ckpt_path = os.path.join(ckpt_dir, f"visiontspp_{arch.split('_')[1]}.ckpt")
        self.vts = VisionTSpp(
            arch=arch, finetune_type='none', ckpt_path=ckpt_path, load_ckpt=True,
            quantile=True, clip_input=0, complete_no_clip=False,
            color=(self.render_mode == 'mv'),
        )
        P = getattr(configs, 'vts_periodicity', 24)
        self.vts.update_config(context_len=configs.seq_len, pred_len=configs.pred_len,
                               periodicity=P, norm_const=0.4, align_const=0.4)
        for p in self.vts.parameters():
            p.requires_grad = False

        self.embed_dim = self.vts.vision_model.cls_token.shape[-1]   # 768 (base)

        # Fixed visible-token count + restore order from VisionTS++'s mask buffer.
        noise = self.vts.mask                                        # [1, L], 0=visible 1=masked
        L = noise.shape[1]
        self.n_visible = int(L * (1 - self.vts.mask_ratio))
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)              # [1, L]
        self.register_buffer('ids_restore', ids_restore.long())

        # Learnable CLS token fed into the decoder alongside the mapped tokens.
        self.cls = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.cls, std=0.02)

        # ---- MLP: student features -> [*, n_visible, embed_dim] ----
        # CI: per-variable image  -> in = d_model * patch_num
        # MV: one packed image    -> in = n_vars * d_model * patch_num
        per_var = self.d_model * self.patch_num
        in_dim = per_var if self.render_mode == 'ci' else self.enc_in * per_var
        hidden = getattr(configs, 'vts_mlp_hidden', 2048)
        out_dim = self.n_visible * self.embed_dim
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def _decode_to_forecast(self, x_for_vts, latent):
        """Inject `latent` as the encoder output and run VisionTS++'s forward,
        reusing its rendering / un-render / denorm. Returns [B, pred_len, nvars]."""
        B = latent.shape[0]
        ids = self.ids_restore.expand(B, -1)

        def enc_fn(imgs, mask_ratio, noise=None):
            return latent, None, ids

        orig = self.vts.vision_model.forward_encoder
        self.vts.vision_model.forward_encoder = enc_fn
        try:
            y = self.vts.forward(x_for_vts)          # quantile -> [median, qlist]
        finally:
            self.vts.vision_model.forward_encoder = orig
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, N = x_enc.shape
        feats, _, _ = self.student.encode(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B,N,d,pn]

        if self.render_mode == 'ci':
            f = einops.rearrange(feats, 'b n d p -> (b n) d p')      # [B*N, d, pn]
            tokens = self.mlp(f).view(B * N, self.n_visible, self.embed_dim)
            latent = torch.cat([self.cls.expand(B * N, -1, -1), tokens], dim=1)
            x_for_vts = einops.rearrange(x_enc, 'b l n -> (b n) l 1')  # nvars=1 per image
            y = self._decode_to_forecast(x_for_vts, latent)           # [B*N, pred_len, 1]
            y = einops.rearrange(y, '(b n) p 1 -> b p n', b=B)
        else:  # mv
            tokens = self.mlp(feats).view(B, self.n_visible, self.embed_dim)
            latent = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)
            y = self._decode_to_forecast(x_enc, latent)               # [B, pred_len, N]

        return y[:, -self.pred_len:, :]
