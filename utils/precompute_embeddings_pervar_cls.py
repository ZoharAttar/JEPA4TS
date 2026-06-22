"""
Precompute per-variable DINO embeddings for JEPA4TS classification datasets.

Supports any rendering method registered in METHOD_REGISTRY:
    - GAF
    - RP
    - LinePlot
    - Spectrogram

Per-variable: each sample's i-th variable gets its own DINO embedding.
Output: one [N, 768] .npy per sample, keyed by md5 of the (padding-trimmed)
float32 time series. A manifest.json maps hash -> {split, label, true_len}.

Targets UEA classification datasets (Handwriting, JapaneseVowels, etc.)
loaded through data_provider with task_name='classification'.

Usage:
    python utils/precompute_embeddings_pervar_cls.py --dataset Handwriting --method Spectrogram
    python utils/precompute_embeddings_pervar_cls.py --dataset Handwriting --method RP
    python utils/precompute_embeddings_pervar_cls.py --dataset Handwriting --method GAF

    # Sweep all methods on one dataset:
    for m in GAF RP LinePlot Spectrogram; do
      python utils/precompute_embeddings_pervar_cls.py --dataset Handwriting --method $m
    done
"""

import os
import math
import json
import argparse
import hashlib
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

# Reuse the existing per-variable transforms (single source of truth).
from precompute_embeddings_pervar import (
    transform_GAF_pervar,
    transform_RP_pervar,
    transform_lineplot_pervar,
)


DATASET_CONFIGS = {
    'Handwriting': {
        'data': 'UEA',
        'root_path': './dataset/Handwriting/',
        'cache_base': './dataset/Handwriting/dino_embeddings_Handwriting',
    },
    'JapaneseVowels': {
        'data': 'UEA',
        'root_path': './dataset/JapaneseVowels/',
        'cache_base': './dataset/JapaneseVowels/dino_embeddings_JapaneseVowels',
    },
    'SpokenArabicDigits': {
        'data': 'UEA',
        'root_path': './dataset/SpokenArabicDigits/',
        'cache_base': './dataset/SpokenArabicDigits/dino_embeddings_SpokenArabicDigits',
    },
    'UWaveGestureLibrary': {
        'data': 'UEA',
        'root_path': './dataset/UWaveGestureLibrary/',
        'cache_base': './dataset/UWaveGestureLibrary/dino_embeddings_UWaveGestureLibrary',
    },
}

IMAGE_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# Spectrogram (new — per-variable)
# ============================================

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
    """
    Per-variable STFT spectrogram rendering.

    Args:
        x: [B, seq_len, N] time series
        image_size: output H = W
        n_fft / hop_length / win_length: STFT params; defaults adapt to seq_len
        log_scale: apply log1p to magnitude
        colormap: matplotlib cmap name (e.g. 'viridis') or None for grayscale-replicated

    Returns:
        [B, N, 3, image_size, image_size] float tensor in [0, 1]
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


# ============================================
# Method registry
# ============================================

METHOD_REGISTRY = {
    'GAF':         {'fn': transform_GAF_pervar,     'extra_keys': []},
    'RP':          {'fn': transform_RP_pervar,      'extra_keys': []},
    'LinePlot':    {'fn': transform_lineplot_pervar,'extra_keys': []},
    'Spectrogram': {'fn': transform_spectrogram_pervar,
                    'extra_keys': ['n_fft', 'hop_length', 'win_length',
                                   'log_scale', 'colormap']},
}


def _method_cache_suffix(method, kwargs):
    """Method-aware suffix so different STFT params don't collide."""
    if method != 'Spectrogram':
        return ''
    n_fft = kwargs.get('n_fft') or 'auto'
    hop = kwargs.get('hop_length') or 'auto'
    win = kwargs.get('win_length') or 'auto'
    log_flag = 1 if kwargs.get('log_scale', True) else 0
    cmap = kwargs.get('colormap') or 'none'
    return f"_n{n_fft}_h{hop}_w{win}_log{log_flag}_cmap{cmap}"


# ============================================
# Utils
# ============================================

def get_ts_hash(ts_array):
    return hashlib.md5(ts_array.astype(np.float32).tobytes()).hexdigest()[:16]


def _label_to_int(label_tensor):
    arr = np.asarray(label_tensor.detach().cpu().numpy()).ravel()
    return int(arr[0]) if arr.size else -1


# ============================================
# Main
# ============================================

def main():
    from data_provider.data_factory import data_provider

    parser = argparse.ArgumentParser(
        description="Precompute per-variable DINO embeddings (classification, any method).")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='UEA dataset name')
    parser.add_argument('--method', type=str, required=True,
                        choices=list(METHOD_REGISTRY.keys()),
                        help='Rendering method')
    parser.add_argument('--seq_len', type=int, default=0,
                        help='Max sequence length for padding (0 = auto from dataset)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='DataLoader batch size; each sample still hashed/embedded individually')
    parser.add_argument('--dino_batch_size', type=int, default=7,
                        help='Max variables fed to DINO at once')

    # Spectrogram-only knobs (ignored for other methods)
    parser.add_argument('--n_fft', type=int, default=0, help='Spectrogram: 0 = adaptive')
    parser.add_argument('--hop_length', type=int, default=0, help='Spectrogram: 0 = n_fft // 4')
    parser.add_argument('--win_length', type=int, default=0, help='Spectrogram: 0 = n_fft')
    parser.add_argument('--no_log', action='store_true', help='Spectrogram: disable log1p')
    parser.add_argument('--colormap', type=str, default=None,
                        help='Spectrogram: matplotlib cmap (e.g. viridis); default = grayscale')

    parser.add_argument('--mirror_ett_path', action='store_true',
                        help='Also mirror cache under ./dataset/ETT-small/... '
                             'to match the hardcoded path in JEPAVTS.VisionTSTeacher')
    parser.add_argument('--zip', action='store_true', help='Zip cache dir when done')
    cli = parser.parse_args()

    cfg = DATASET_CONFIGS[cli.dataset]
    method_entry = METHOD_REGISTRY[cli.method]
    transform_fn = method_entry['fn']

    extra_kwargs = {}
    if cli.method == 'Spectrogram':
        extra_kwargs = {
            'n_fft': cli.n_fft if cli.n_fft > 0 else None,
            'hop_length': cli.hop_length if cli.hop_length > 0 else None,
            'win_length': cli.win_length if cli.win_length > 0 else None,
            'log_scale': not cli.no_log,
            'colormap': cli.colormap,
        }

    cache_suffix = _method_cache_suffix(cli.method, extra_kwargs)
    cache_dir = f"{cfg['cache_base']}_{cli.method}_pervar{cache_suffix}"
    os.makedirs(cache_dir, exist_ok=True)

    if cli.mirror_ett_path:
        mirror_dir = (f"./dataset/ETT-small/dino_embeddings_"
                      f"{cli.dataset}_{cli.method}_pervar")
        os.makedirs(mirror_dir, exist_ok=True)
    else:
        mirror_dir = None

    data_args = argparse.Namespace(
        task_name='classification',
        data=cfg['data'],
        root_path=cfg['root_path'],
        model_id=cli.dataset,
        seq_len=cli.seq_len if cli.seq_len > 0 else 1024,
        batch_size=cli.batch_size,
        num_workers=0,
        embed='timeF',
        freq='h',
        augmentation_ratio=0,
    )

    print(f"Dataset:     {cli.dataset}")
    print(f"Method:      {cli.method} (per-var)")
    print(f"Device:      {DEVICE}")
    print(f"Cache dir:   {cache_dir}")
    if mirror_dir:
        print(f"Mirror dir:  {mirror_dir}")

    if cli.seq_len == 0:
        probe_set, _ = data_provider(data_args, 'TRAIN')
        max_len = int(getattr(probe_set, 'max_seq_len', 0))
        if max_len <= 0:
            raise RuntimeError("Could not infer max_seq_len from dataset; pass --seq_len.")
        data_args.seq_len = max_len
        print(f"Auto seq_len: {data_args.seq_len}")
    print(f"Seq len:     {data_args.seq_len}")
    print(f"Batch size:  {cli.batch_size}")

    print("\nLoading DINOv2-base ...")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    dino.eval()

    existing = set(f.replace('.npy', '') for f in os.listdir(cache_dir) if f.endswith('.npy'))
    print(f"Existing cached: {len(existing)}")

    manifest_path = os.path.join(cache_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    else:
        manifest = {}

    for flag in ['TRAIN', 'TEST']:
        print(f"\n{'='*50}")
        print(f"Processing {flag} split ...")
        print(f"{'='*50}")

        data_set, data_loader = data_provider(data_args, flag)
        generated = 0
        skipped = 0

        with torch.no_grad():
            for batch_x, labels, padding_masks in tqdm(data_loader):
                batch_x = batch_x.float()
                padding_masks = padding_masks.bool()
                B = batch_x.shape[0]

                for b in range(B):
                    true_len = int(padding_masks[b].sum().item())
                    if true_len <= 0:
                        continue
                    sample = batch_x[b, :true_len]
                    sample_np = sample.cpu().numpy().astype(np.float32)
                    ts_hash = get_ts_hash(sample_np)

                    cache_path = os.path.join(cache_dir, f"{ts_hash}.npy")

                    if ts_hash in existing or os.path.exists(cache_path):
                        existing.add(ts_hash)
                        skipped += 1
                    else:
                        x_images = transform_fn(
                            sample.unsqueeze(0).to(DEVICE),
                            image_size=IMAGE_SIZE, device=DEVICE,
                            **extra_kwargs,
                        )

                        N = x_images.shape[1]
                        var_images = x_images[0]
                        chunks = []
                        for start in range(0, N, cli.dino_batch_size):
                            end = min(start + cli.dino_batch_size, N)
                            out = dino(pixel_values=var_images[start:end])
                            chunks.append(out.last_hidden_state[:, 0, :])
                        embedding = torch.cat(chunks, dim=0).cpu().numpy()

                        np.save(cache_path, embedding)
                        if mirror_dir:
                            np.save(os.path.join(mirror_dir, f"{ts_hash}.npy"), embedding)
                        existing.add(ts_hash)
                        generated += 1

                    manifest[ts_hash] = {
                        'split': flag,
                        'label': _label_to_int(labels[b]),
                        'true_len': true_len,
                        'n_vars': int(sample.shape[-1]),
                    }

        with open(manifest_path, 'w') as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

        print(f"  Generated: {generated}, Skipped: {skipped}")

    total_cached = len([f for f in os.listdir(cache_dir) if f.endswith('.npy')])
    print(f"\nDone! Total embeddings: {total_cached}")
    print(f"Manifest: {manifest_path}  ({len(manifest)} entries)")

    if cli.zip:
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
