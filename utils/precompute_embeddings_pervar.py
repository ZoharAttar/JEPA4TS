"""
Precompute per-variable DINO embeddings for JEPA4TS.

Each sample gets one DINO embedding per variable: saved as [N, 768] instead of [768].

Usage:
    python utils/precompute_embeddings_pervar.py --dataset ETTh1 --method RP
    python utils/precompute_embeddings_pervar.py --dataset ETTm2 --method GAF
    python utils/precompute_embeddings_pervar.py --dataset weather --method LinePlot

    python utils/precompute_embeddings_pervar.py --dataset exchange_rate --method Spectrogram

    # Run all combos:
    for d in ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic exchange_rate national_illness; do
      for m in RP GAF LinePlot Spectrogram; do
        python utils/precompute_embeddings_pervar.py --dataset $d --method $m
      done
    done

Note: datasets other than ETT*/weather/national_illness require the corresponding
CSV to be present under ./dataset/<name>/ (e.g. electricity, traffic, exchange_rate).
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
from tqdm import tqdm
from transformers import AutoModel
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work when running this script
# directly (python utils/precompute_embeddings_pervar.py). When Python runs a
# script, sys.path[0] is the script's directory (utils/), so sibling packages
# like `data_provider` are not found unless the project root is added.
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATASET_CONFIGS = {
    'ETTh1': {
        'data': 'ETTh1',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh1.csv',
        'cache_base': './dataset/ETT-small/dino_embeddings_ETTh1',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
    },
    'ETTh2': {
        'data': 'ETTh2',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh2.csv',
        'cache_base': './dataset/ETT-small/dino_embeddings_ETTh2',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
    },
    'ETTm1': {
        'data': 'ETTm1',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTm1.csv',
        'cache_base': './dataset/ETT-small/dino_embeddings_ETTm1',
        'freq': 't',
        'features': 'M',
        'target': 'OT',
    },
    'ETTm2': {
        'data': 'ETTm2',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTm2.csv',
        'cache_base': './dataset/ETT-small/dino_embeddings_ETTm2',
        'freq': 't',
        'features': 'M',
        'target': 'OT',
    },
    'weather': {
        'data': 'custom',
        'root_path': './dataset/weather/',
        'data_path': 'weather.csv',
        'cache_base': './dataset/weather/dino_embeddings_weather',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
    },
    'electricity': {
        'data': 'custom',
        'root_path': './dataset/electricity/',
        'data_path': 'electricity.csv',
        'cache_base': './dataset/electricity/dino_embeddings_electricity',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
    },
    'traffic': {
        'data': 'custom',
        'root_path': './dataset/traffic/',
        'data_path': 'traffic.csv',
        'cache_base': './dataset/traffic/dino_embeddings_traffic',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
    },
    'exchange_rate': {
        'data': 'custom',
        'root_path': './dataset/exchange_rate/',
        'data_path': 'exchange_rate.csv',
        'cache_base': './dataset/exchange_rate/dino_embeddings_exchange_rate',
        'freq': 'd',
        'features': 'M',
        'target': 'OT',
    },
    'treasury_yields': {
        'data': 'custom',
        'root_path': './dataset/treasury_yields/',
        'data_path': 'treasury_yields.csv',
        # Train uses --data custom → {root}/dino_embeddings_custom_{method}_pervar
        'cache_base': './dataset/treasury_yields/dino_embeddings_custom',
        'freq': 'd',
        'features': 'M',
        'target': 'OT',
    },
    'national_illness': {
        'data': 'custom',
        'root_path': './dataset/',
        'data_path': 'national_illness.csv',
        'cache_base': './dataset/dino_embeddings_national_illness',
        'freq': 'w',
        'features': 'M',
        'target': 'OT',
    },
}

IMAGE_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# TRANSFORM FUNCTIONS (per-variable, no averaging)
# ============================================

def transform_GAF_pervar(x, image_size=518, device=None):
    """Returns [B, N, 3, H, W] — one 3-channel GAF image per variable."""
    from pyts.image import GramianAngularField
    if device is None:
        device = x.device
    batch_size, seq_len, nvars = x.shape
    gaf_image_size = min(seq_len, image_size)

    x_flat = x.detach().cpu().permute(0, 2, 1).reshape(-1, seq_len).numpy()
    gaf = GramianAngularField(image_size=gaf_image_size, method='summation')
    gaf_images = gaf.fit_transform(x_flat)
    gaf_images = torch.tensor(gaf_images, dtype=x.dtype).unsqueeze(1)

    if gaf_image_size != image_size:
        gaf_images = F.interpolate(gaf_images, size=(image_size, image_size),
                                   mode='bilinear', align_corners=False)

    # [B*N, 1, H, W] → [B, N, 1, H, W] → expand to 3 channels
    gaf_images = gaf_images.reshape(batch_size, nvars, 1, image_size, image_size)
    gaf_images = gaf_images.expand(-1, -1, 3, -1, -1).contiguous()
    return gaf_images.to(device)


def transform_RP_pervar(x, image_size=518, device=None):
    """Returns [B, N, 3, H, W] — one 3-channel RP image per variable."""
    import einops
    from pyts.image import RecurrencePlot
    if device is None:
        device = x.device
    batch_size, seq_len, nvars = x.shape

    x_flat = x.detach().cpu().permute(0, 2, 1).reshape(-1, seq_len).numpy()
    rp = RecurrencePlot()
    rp_images = rp.fit_transform(x_flat)
    rp_images = torch.tensor(rp_images, dtype=torch.float32)

    # [B*N, H, W] → [B, N, H, W]
    rp_images = einops.rearrange(rp_images, '(b n) h w -> b n h w', b=batch_size)
    # Resize each image
    rp_images = F.interpolate(
        rp_images.reshape(batch_size * nvars, 1, rp_images.shape[-2], rp_images.shape[-1]),
        size=(image_size, image_size), mode='bilinear', align_corners=False
    )
    # [B*N, 1, H, W] → [B, N, 1, H, W] → [B, N, 3, H, W]
    rp_images = rp_images.reshape(batch_size, nvars, 1, image_size, image_size)
    rp_images = rp_images.expand(-1, -1, 3, -1, -1).contiguous()
    return rp_images.to(device)


def transform_lineplot_pervar(x, image_size=518, device=None):
    """Returns [B, N, 3, H, W] — one 3-channel line plot per variable."""
    import torchvision.transforms as T
    from io import BytesIO
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if device is None:
        device = x.device
    batch_size, seq_len, nvars = x.shape
    x_cpu = x.detach().cpu().numpy()
    dpi = 100
    fig_inches = image_size / dpi
    to_tensor = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    all_images = []
    for i in range(batch_size):
        var_images = []
        for v in range(nvars):
            fig, ax = plt.subplots(1, 1, figsize=(fig_inches, fig_inches), dpi=dpi)
            ax.plot(x_cpu[i, :, v], linewidth=0.8, color='blue')
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            var_images.append(to_tensor(img))
        all_images.append(torch.stack(var_images))  # [N, 3, H, W]

    return torch.stack(all_images).to(device)  # [B, N, 3, H, W]


def _apply_matplotlib_cmap(img, cmap_name):
    """img: [B, 1, H, W] in [0,1]  ->  [B, 3, H, W] in [0,1] via matplotlib cmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    arr = img.squeeze(1).detach().cpu().numpy()
    rgba = cmap(arr)
    rgb = rgba[..., :3]
    rgb = torch.from_numpy(rgb).to(img.device, dtype=img.dtype).permute(0, 3, 1, 2)
    return rgb.contiguous()


def transform_spectrogram_pervar(
    x, image_size=518, device=None,
    n_fft=None, hop_length=None, win_length=None,
    log_scale=True, colormap=None,
):
    """Per-variable STFT spectrogram. Returns [B, N, 3, H, W] in [0, 1].

    STFT params default adaptively to seq_len; colormap=None -> grayscale
    replicated to 3 channels. Identical to the classification renderer so the
    same series produces the same embedding across tasks.
    """
    if device is None:
        device = x.device
    B, L, N = x.shape

    if n_fft is None:
        target = max(8, L // 8)
        n_fft = max(16, min(256, 2 ** int(math.log2(target))))
    if win_length is None:
        win_length = min(n_fft, L) if L > 0 else n_fft
    if hop_length is None:
        hop_length = max(1, n_fft // 4)

    x_flat = x.permute(0, 2, 1).reshape(B * N, L).to(device).float()
    if L < n_fft:
        x_flat = F.pad(x_flat, (0, n_fft - L))

    window = torch.hann_window(win_length, device=device)
    stft = torch.stft(
        x_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = stft.abs()
    if log_scale:
        mag = torch.log1p(mag)

    mn = mag.amin(dim=(-2, -1), keepdim=True)
    mx = mag.amax(dim=(-2, -1), keepdim=True)
    img = (mag - mn) / (mx - mn + 1e-8)

    img = img.unsqueeze(1)
    img = F.interpolate(img, size=(image_size, image_size),
                        mode='bilinear', align_corners=False)

    if colormap is None:
        img = img.expand(-1, 3, -1, -1).contiguous()
    else:
        img = _apply_matplotlib_cmap(img, colormap)

    return img.reshape(B, N, 3, image_size, image_size)


TRANSFORM_MAP = {
    'GAF': transform_GAF_pervar,
    'RP': transform_RP_pervar,
    'LinePlot': transform_lineplot_pervar,
    'Spectrogram': transform_spectrogram_pervar,
}

# ============================================
# UTILS
# ============================================

def get_ts_hash(ts_array):
    return hashlib.md5(ts_array.astype(np.float32).tobytes()).hexdigest()[:16]


def main():
    from data_provider.data_factory import data_provider

    parser = argparse.ArgumentParser(description="Precompute per-variable DINO embeddings")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--method', type=str, required=True,
                        choices=list(TRANSFORM_MAP.keys()),
                        help='Rendering method')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Keep at 1 for caching by hash; higher only if memory allows')
    parser.add_argument('--dino_batch_size', type=int, default=7,
                        help='Max variables to feed DINO at once (controls GPU memory)')
    # Spectrogram-only knobs (ignored for other methods). Keep defaults to match
    # the training-time teacher cache (dino_embeddings_<data>_Spectrogram_pervar).
    parser.add_argument('--n_fft', type=int, default=0, help='Spectrogram: 0 = adaptive')
    parser.add_argument('--hop_length', type=int, default=0, help='Spectrogram: 0 = n_fft // 4')
    parser.add_argument('--win_length', type=int, default=0, help='Spectrogram: 0 = n_fft')
    parser.add_argument('--no_log', action='store_true', help='Spectrogram: disable log1p')
    parser.add_argument('--colormap', type=str, default=None,
                        help='Spectrogram: matplotlib cmap (e.g. viridis); default = grayscale')
    parser.add_argument('--zip', action='store_true', help='Zip cache dir when done')
    cli_args = parser.parse_args()

    cfg = DATASET_CONFIGS[cli_args.dataset]
    transform_fn = TRANSFORM_MAP[cli_args.method]

    extra_kwargs = {}
    if cli_args.method == 'Spectrogram':
        extra_kwargs = {
            'n_fft': cli_args.n_fft if cli_args.n_fft > 0 else None,
            'hop_length': cli_args.hop_length if cli_args.hop_length > 0 else None,
            'win_length': cli_args.win_length if cli_args.win_length > 0 else None,
            'log_scale': not cli_args.no_log,
            'colormap': cli_args.colormap,
        }

    cache_dir = f"{cfg['cache_base']}_{cli_args.method}_pervar"
    os.makedirs(cache_dir, exist_ok=True)

    data_args = argparse.Namespace(
        data=cfg['data'],
        root_path=cfg['root_path'],
        data_path=cfg['data_path'],
        features=cfg['features'],
        target=cfg['target'],
        seq_len=cli_args.seq_len,
        label_len=cli_args.label_len,
        pred_len=cli_args.pred_len,
        embed='timeF',
        freq=cfg['freq'],
        batch_size=cli_args.batch_size,
        num_workers=0,
        seasonal_patterns='Monthly',
        augmentation_ratio=0,
        task_name='long_term_forecast',
    )

    print(f"Dataset:    {cli_args.dataset}")
    print(f"Method:     {cli_args.method}")
    print(f"Device:     {DEVICE}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Seq len:    {cli_args.seq_len}")
    print(f"Batch size: {cli_args.batch_size}")

    print("\nLoading DINOv2-base...")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    dino.eval()

    existing = set(f.replace('.npy', '') for f in os.listdir(cache_dir) if f.endswith('.npy'))
    print(f"Existing cached: {len(existing)}")

    for flag in ['train', 'val', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {flag.upper()} split...")
        print(f"{'='*50}")

        data_set, data_loader = data_provider(data_args, flag)
        generated = 0
        skipped = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(data_loader):
                batch_x = batch_x.float().to(DEVICE)
                B = batch_x.shape[0]

                for b in range(B):
                    sample = batch_x[b]  # [seq_len, N]
                    sample_np = sample.cpu().numpy().astype(np.float32)
                    ts_hash = get_ts_hash(sample_np)

                    if ts_hash in existing:
                        skipped += 1
                        continue

                    cache_path = os.path.join(cache_dir, f"{ts_hash}.npy")
                    if os.path.exists(cache_path):
                        existing.add(ts_hash)
                        skipped += 1
                        continue

                    # Transform single sample: [1, seq_len, N] → [1, N, 3, H, W]
                    x_images = transform_fn(
                        sample.unsqueeze(0), image_size=IMAGE_SIZE, device=DEVICE,
                        **extra_kwargs
                    )

                    # Run DINO per variable (chunked to control memory)
                    N = x_images.shape[1]
                    var_images = x_images[0]  # [N, 3, H, W]
                    embeddings = []

                    for start in range(0, N, cli_args.dino_batch_size):
                        end = min(start + cli_args.dino_batch_size, N)
                        chunk = var_images[start:end]  # [chunk, 3, H, W]
                        out = dino(pixel_values=chunk)
                        embeddings.append(out.last_hidden_state[:, 0, :])  # [chunk, 768]

                    embedding = torch.cat(embeddings, dim=0)  # [N, 768]
                    np.save(cache_path, embedding.cpu().numpy())
                    existing.add(ts_hash)
                    generated += 1

        print(f"  Generated: {generated}, Skipped: {skipped}")

    total_cached = len([f for f in os.listdir(cache_dir) if f.endswith('.npy')])
    print(f"\nDone! Total embeddings: {total_cached}")

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
