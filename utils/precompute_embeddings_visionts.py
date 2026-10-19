"""
Precompute per-variable VisionTS++ teacher embeddings for JEPA4TS.

This is the VisionTS++ analogue of utils/precompute_embeddings_pervar.py.
Instead of RP/GAF/LinePlot/Spectrogram + DINOv2, it renders each series the
*VisionTS++ way* and runs the VisionTS++ MAE encoder as the teacher:

    1. Per-series instance norm   (mean subtract, std / norm_const, norm_const=0.4)
    2. Reshape the 1-D window into a 2-D periodicity grid:
           n (p f) -> n f p        (f = periodicity, e.g. 24 for hourly ETTh1)
    3. Resize the grid to the MAE image size (224x224) and clip
    4. Replicate to 3 channels (channel-independent, one image per variable)
    5. Run the VisionTS++ MAE *encoder* at mask_ratio=0 (full render, no
       forecast mask) and take the CLS token -> [768] per variable

Each sample is cached individually by the md5 hash of its [seq_len, N] float32
array -- the SAME hashing scheme as the DINO teacher -- so the result is a
drop-in replacement for the per-variable DINO cache: [N, 768] per sample.

Output cache dir:
    {root}/visionts_embeddings_{dataset}_VisionTSpp_pervar/{hash}.npy   # [N, 768]

Setup (Colab / GPU box):
    pip install git+https://github.com/Keytoyze/VisionTS.git   # provides `visionts.VisionTSpp`
    # weights auto-download from HuggingFace (Lefei/VisionTSpp) on first run,
    # or pass --ckpt_path to a pre-downloaded visiontspp_base.ckpt

Usage:
    python utils/precompute_embeddings_visionts.py --dataset ETTh1
    python utils/precompute_embeddings_visionts.py --dataset ETTh1 --arch mae_base --periodicity 24
    python utils/precompute_embeddings_visionts.py --dataset ETTh1 --zip   # zip when done (Colab)
"""

import os
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import einops
from tqdm import tqdm
import shutil

# Ensure project root is importable when run as `python utils/...`
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================
# DATASET CONFIGS  (mirrors precompute_embeddings_pervar.py)
# `periodicity` is the VisionTS++ seasonality used to build the 2-D grid.
# ============================================
DATASET_CONFIGS = {
    'ETTh1': {'data': 'ETTh1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh1.csv',
              'cache_base': './dataset/ETT-small/visionts_embeddings_ETTh1',
              'freq': 'h', 'features': 'M', 'target': 'OT', 'periodicity': 24},
    'ETTh2': {'data': 'ETTh2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh2.csv',
              'cache_base': './dataset/ETT-small/visionts_embeddings_ETTh2',
              'freq': 'h', 'features': 'M', 'target': 'OT', 'periodicity': 24},
    'ETTm1': {'data': 'ETTm1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm1.csv',
              'cache_base': './dataset/ETT-small/visionts_embeddings_ETTm1',
              'freq': 't', 'features': 'M', 'target': 'OT', 'periodicity': 96},
    'ETTm2': {'data': 'ETTm2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm2.csv',
              'cache_base': './dataset/ETT-small/visionts_embeddings_ETTm2',
              'freq': 't', 'features': 'M', 'target': 'OT', 'periodicity': 96},
    'weather': {'data': 'custom', 'root_path': './dataset/weather/', 'data_path': 'weather.csv',
                'cache_base': './dataset/weather/visionts_embeddings_weather',
                'freq': 'h', 'features': 'M', 'target': 'OT', 'periodicity': 24},
    'electricity': {'data': 'custom', 'root_path': './dataset/electricity/', 'data_path': 'electricity.csv',
                    'cache_base': './dataset/electricity/visionts_embeddings_electricity',
                    'freq': 'h', 'features': 'M', 'target': 'OT', 'periodicity': 24},
    'traffic': {'data': 'custom', 'root_path': './dataset/traffic/', 'data_path': 'traffic.csv',
                'cache_base': './dataset/traffic/visionts_embeddings_traffic',
                'freq': 'h', 'features': 'M', 'target': 'OT', 'periodicity': 24},
    'exchange_rate': {'data': 'custom', 'root_path': './dataset/exchange_rate/', 'data_path': 'exchange_rate.csv',
                      'cache_base': './dataset/exchange_rate/visionts_embeddings_exchange_rate',
                      'freq': 'd', 'features': 'M', 'target': 'OT', 'periodicity': 7},
    'national_illness': {'data': 'custom', 'root_path': './dataset/', 'data_path': 'national_illness.csv',
                         'cache_base': './dataset/visionts_embeddings_national_illness',
                         'freq': 'w', 'features': 'M', 'target': 'OT', 'periodicity': 52},
}

# MAE backbone -> embedding dim produced by the CLS token
ARCH_DIM = {'mae_base': 768, 'mae_large': 1024, 'mae_huge': 1280}

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")


# ============================================
# VisionTS++ per-variable rendering
# ============================================
def render_visionts_pervar(x, periodicity, image_size=224, norm_const=0.4,
                           clip_val=5.0, device=None):
    """Render a single sample into one VisionTS++ image per variable.

    x: [1, seq_len, N]  ->  returns [N, 3, image_size, image_size]

    Mirrors VisionTSpp's pipeline (normalize -> periodicity segmentation ->
    resize -> 3 channels -> clip), but in channel-independent "full render"
    mode: the whole look-back fills the image, no forecast mask band.
    """
    if device is None:
        device = x.device
    _, seq_len, nvars = x.shape

    # 1. Per-series instance normalization (VisionTS++: std / norm_const)
    means = x.mean(1, keepdim=True).detach()              # [1, 1, N]
    x_enc = x - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev = stdev / norm_const
    x_enc = x_enc / stdev
    x_enc = einops.rearrange(x_enc, 'b s n -> (b n) s')   # [N, seq_len]

    # 2. Pad on the left so the length divides the periodicity, then segment.
    #    'replicate' matches VisionTS's default padding_mode.
    pad_left = (periodicity - seq_len % periodicity) % periodicity
    if pad_left > 0:
        x_enc = F.pad(x_enc.unsqueeze(1), (pad_left, 0), mode='replicate').squeeze(1)
    x_2d = einops.rearrange(x_enc, 'n (p f) -> n 1 f p', f=periodicity)  # [N, 1, f, p]

    # 3. Resize the periodicity grid to the full MAE image (no mask band).
    x_img = F.interpolate(x_2d, size=(image_size, image_size),
                          mode='bilinear', align_corners=False)          # [N, 1, H, W]

    # 4. Channel-independent: replicate the single grey image to 3 channels.
    x_img = x_img.expand(-1, 3, -1, -1).contiguous()                     # [N, 3, H, W]

    # 5. Clip (VisionTS++ default, clip_input=0 path -> clip to [-5, 5]).
    x_img = torch.clip(x_img, -clip_val, clip_val)
    return x_img.to(device)


def load_visionts_encoder(arch, ckpt_path, ckpt_dir, device):
    """Load VisionTS++ weights and return the frozen MAE vision_model (encoder)."""
    from visionts import VisionTSpp
    kwargs = dict(arch=arch, finetune_type='none', quantile=True,
                  clip_input=0, complete_no_clip=False, color=True)
    if ckpt_path:
        model = VisionTSpp(ckpt_path=ckpt_path, load_ckpt=True, **kwargs)
    else:
        # Weights auto-download from HuggingFace (Lefei/VisionTSpp) into ckpt_dir
        default_ckpt = os.path.join(ckpt_dir, f"visiontspp_{arch.split('_')[1]}.ckpt")
        model = VisionTSpp(ckpt_path=default_ckpt, load_ckpt=True, **kwargs)
    vision_model = model.vision_model.to(device).eval()
    for p in vision_model.parameters():
        p.requires_grad = False
    return vision_model


@torch.no_grad()
def encode_images(vision_model, images, pool='cls', chunk=16):
    """images: [N, 3, H, W] -> teacher embeddings [N, D] at mask_ratio=0.

    pool:
      'cls'  -> the CLS token               latent[:, 0, :]
      'mean' -> mean over the patch tokens  latent[:, 1:, :].mean(1)

    Passes deterministic ascending `noise` so random_masking keeps all patches
    in natural order (mask_ratio=0 keeps everything; this makes the embeddings
    fully reproducible).
    """
    n_patches = vision_model.patch_embed.num_patches
    outs = []
    for start in range(0, images.shape[0], chunk):
        x = images[start:start + chunk]
        noise = torch.arange(n_patches, device=x.device, dtype=x.dtype)
        noise = noise.unsqueeze(0).expand(x.shape[0], -1)              # [chunk, P]
        latent, _, _ = vision_model.forward_encoder(x, mask_ratio=0.0, noise=noise)
        if pool == 'mean':
            outs.append(latent[:, 1:, :].mean(dim=1))                  # patch-mean [chunk, D]
        else:
            outs.append(latent[:, 0, :])                               # CLS token  [chunk, D]
    return torch.cat(outs, dim=0)


def get_ts_hash(ts_array):
    """md5 of the [seq_len, N] float32 sample -- identical to the DINO teacher."""
    return hashlib.md5(ts_array.astype(np.float32).tobytes()).hexdigest()[:16]


def main():
    from data_provider.data_factory import data_provider

    parser = argparse.ArgumentParser(description="Precompute per-variable VisionTS++ teacher embeddings")
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--arch', type=str, default='mae_base', choices=list(ARCH_DIM.keys()))
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to visiontspp_*.ckpt. If omitted, auto-downloaded from HuggingFace.')
    parser.add_argument('--ckpt_dir', type=str, default='./hf_models/VisionTSpp',
                        help='Dir to download/find VisionTS++ weights when --ckpt_path is not given.')
    parser.add_argument('--pool', type=str, default='cls', choices=['cls', 'mean'],
                        help="Teacher vector: 'cls' token (default) or 'mean' of patch tokens. "
                             "Each writes a SEPARATE cache (method tag VisionTSpp vs "
                             "VisionTSpp_meanpool), so both can coexist.")
    parser.add_argument('--periodicity', type=int, default=0, help='0 = dataset default from config')
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--clip_val', type=float, default=5.0)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Keep at 1 for hash-based caching (one sample at a time).')
    parser.add_argument('--enc_chunk', type=int, default=16,
                        help='Max variable-images fed to the encoder at once (GPU memory).')
    parser.add_argument('--zip', action='store_true', help='Zip cache dir when done (Colab).')
    cli_args = parser.parse_args()

    cfg = DATASET_CONFIGS[cli_args.dataset]
    periodicity = cli_args.periodicity if cli_args.periodicity > 0 else cfg['periodicity']

    # Method tag -> cache dir. 'cls' keeps the original name (VisionTSpp) so
    # existing caches/runs are unaffected; 'mean' uses VisionTSpp_meanpool.
    method_tag = 'VisionTSpp' if cli_args.pool == 'cls' else 'VisionTSpp_meanpool'
    cache_dir = f"{cfg['cache_base']}_{method_tag}_pervar"
    os.makedirs(cache_dir, exist_ok=True)

    data_args = argparse.Namespace(
        data=cfg['data'], root_path=cfg['root_path'], data_path=cfg['data_path'],
        features=cfg['features'], target=cfg['target'],
        seq_len=cli_args.seq_len, label_len=cli_args.label_len, pred_len=cli_args.pred_len,
        embed='timeF', freq=cfg['freq'], batch_size=cli_args.batch_size, num_workers=0,
        seasonal_patterns='Monthly', augmentation_ratio=0, task_name='long_term_forecast',
    )

    print(f"Dataset:     {cli_args.dataset}")
    print(f"Arch:        {cli_args.arch}  (teacher dim = {ARCH_DIM[cli_args.arch]})")
    print(f"Pool:        {cli_args.pool}  (method tag = {method_tag})")
    print(f"Periodicity: {periodicity}")
    print(f"Image size:  {cli_args.image_size}")
    print(f"Device:      {DEVICE}")
    print(f"Cache dir:   {cache_dir}")

    print("\nLoading VisionTS++ encoder...")
    vision_model = load_visionts_encoder(cli_args.arch, cli_args.ckpt_path,
                                         cli_args.ckpt_dir, DEVICE)

    existing = set(f[:-4] for f in os.listdir(cache_dir) if f.endswith('.npy'))
    print(f"Existing cached: {len(existing)}")

    for flag in ['train', 'val', 'test']:
        print(f"\n{'='*50}\nProcessing {flag.upper()} split...\n{'='*50}")
        _, data_loader = data_provider(data_args, flag)
        generated, skipped = 0, 0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(data_loader):
            batch_x = batch_x.float().to(DEVICE)   # [B, seq_len, N]
            for b in range(batch_x.shape[0]):
                sample = batch_x[b]                # [seq_len, N]
                sample_np = sample.cpu().numpy().astype(np.float32)
                ts_hash = get_ts_hash(sample_np)

                cache_path = os.path.join(cache_dir, f"{ts_hash}.npy")
                if ts_hash in existing or os.path.exists(cache_path):
                    existing.add(ts_hash)
                    skipped += 1
                    continue

                images = render_visionts_pervar(
                    sample.unsqueeze(0), periodicity=periodicity,
                    image_size=cli_args.image_size, norm_const=cli_args.norm_const,
                    clip_val=cli_args.clip_val, device=DEVICE)        # [N, 3, H, W]
                embedding = encode_images(vision_model, images, pool=cli_args.pool,
                                          chunk=cli_args.enc_chunk)                  # [N, D]

                np.save(cache_path, embedding.cpu().numpy())
                existing.add(ts_hash)
                generated += 1

        print(f"  Generated: {generated}, Skipped: {skipped}")

    total = len([f for f in os.listdir(cache_dir) if f.endswith('.npy')])
    print(f"\nDone! Total embeddings: {total}  ->  {cache_dir}")

    if cli_args.zip:
        zip_path = shutil.make_archive(cache_dir, 'zip', cache_dir)
        print(f"Zipped to: {zip_path}")
        try:
            from google.colab import files
            files.download(zip_path)
            print("Download triggered!")
        except ImportError:
            print("Not in Colab, zip saved locally.")


if __name__ == '__main__':
    main()
