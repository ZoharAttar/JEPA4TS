"""
Precompute per-variable DINO embeddings for ANOMALY-DETECTION datasets.

Anomaly detection here is reconstruction-based and unsupervised: the model reads a
sliding window of length ``win_size`` (== seq_len) and reconstructs it; large
reconstruction error => anomaly. A "sample" is therefore ONE window ``[win_size, N]``.
VibeTS aligns each window's per-variable student encoding to the DINO embedding of
that window's per-variable rendering.

This script renders each TRAIN window per variable, embeds it with frozen DINOv2
(CLS token, 768-d), and caches one ``[N, 768]`` array per window, keyed by the MD5
hash of the window (exactly the key ``DINOTeacher`` uses at train time), into:

    {root_path}/dino_embeddings_{DATA}_{METHOD}_pervar

so that at training time ``models/JEPAVTS.py :: DINOTeacher`` (which builds the
same path from ``configs.data``) loads them with no recompute.

IMPORTANT
---------
* Only the TRAIN split needs embeddings: the teacher is queried ONLY during
  training (test-time anomaly scoring uses the reconstruction error alone).
* ``--step`` MUST match the stride the training data loader uses, otherwise
  training will request windows that were never precomputed and crash with
  FileNotFoundError. The default anomaly loaders use ``step=1`` for train
  (SWaT uses ``step=100``). If you change ``--step`` here you must change the
  loader/data_factory stride used at train time to match.

Usage
-----
    # default: train split, win_size=100, step=1  (matches the stock loaders)
    python utils/precompute_embeddings_pervar_anomaly.py --dataset PSM --method RP

    # bigger stride (fewer files) -- ONLY if you also train with step=100:
    python utils/precompute_embeddings_pervar_anomaly.py --dataset SMD --method RP --step 100

    # all datasets:
    for d in PSM MSL SMAP SMD SWAT; do
      python utils/precompute_embeddings_pervar_anomaly.py --dataset $d --method RP
    done
"""

import os
import argparse
import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so imports work when running this script
# directly (python utils/precompute_embeddings_pervar_anomaly.py).
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Reuse the EXACT rendering + hashing used by the forecasting/classification
# precompute (and, transitively, by the training-time teacher) so the same window
# produces the same embedding key everywhere.
from utils.precompute_embeddings_pervar import (
    TRANSFORM_MAP, IMAGE_SIZE, DEVICE, get_ts_hash,
)
from data_provider.data_loader import (
    PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader,
)

# Anomaly datasets. ``data`` MUST equal the run.py --data value so the teacher's
# cache path ({root}/dino_embeddings_{data}_{method}_pervar) matches. ``default_step``
# is the stride the stock training loader uses (see data_loader.py).
ANOMALY_CONFIGS = {
    'PSM':  {'loader': PSMSegLoader,  'data': 'PSM',  'root_path': './dataset/PSM/',  'default_step': 1},
    'MSL':  {'loader': MSLSegLoader,  'data': 'MSL',  'root_path': './dataset/MSL/',  'default_step': 1},
    'SMAP': {'loader': SMAPSegLoader, 'data': 'SMAP', 'root_path': './dataset/SMAP/', 'default_step': 1},
    'SMD':  {'loader': SMDSegLoader,  'data': 'SMD',  'root_path': './dataset/SMD/',  'default_step': 100},
    'SWAT': {'loader': SWATSegLoader, 'data': 'SWAT', 'root_path': './dataset/SWaT/', 'default_step': 1},
}


def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-variable DINO embeddings for anomaly detection")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(ANOMALY_CONFIGS.keys()), help='Anomaly dataset name')
    parser.add_argument('--method', type=str, required=True,
                        choices=list(TRANSFORM_MAP.keys()), help='Rendering method')
    parser.add_argument('--win_size', type=int, default=100,
                        help='Window length (== run.py --seq_len at train time)')
    parser.add_argument('--step', type=int, default=None,
                        help='Sliding-window stride. Default = the stock loader stride '
                             '(1 for PSM/MSL/SMAP/SMD, 100 for SWAT). MUST match the '
                             'stride used during training.')
    parser.add_argument('--flags', type=str, nargs='+', default=['train'],
                        help="Which splits to precompute (default: train only, which is "
                             "all the teacher needs). Options: train val test.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Windows loaded per iteration (throughput only; caching is per window)')
    parser.add_argument('--dino_batch_size', type=int, default=8,
                        help='Max variables fed to DINO at once (controls GPU memory)')
    parser.add_argument('--num_workers', type=int, default=4)
    # Spectrogram-only knobs (ignored for other methods). Keep defaults to match
    # the training-time teacher cache.
    parser.add_argument('--n_fft', type=int, default=0, help='Spectrogram: 0 = adaptive')
    parser.add_argument('--hop_length', type=int, default=0, help='Spectrogram: 0 = n_fft // 4')
    parser.add_argument('--win_length', type=int, default=0, help='Spectrogram: 0 = n_fft')
    parser.add_argument('--no_log', action='store_true', help='Spectrogram: disable log1p')
    parser.add_argument('--colormap', type=str, default=None,
                        help='Spectrogram: matplotlib cmap (e.g. viridis); default = grayscale')
    cli_args = parser.parse_args()

    cfg = ANOMALY_CONFIGS[cli_args.dataset]
    transform_fn = TRANSFORM_MAP[cli_args.method]
    step = cli_args.step if cli_args.step is not None else cfg['default_step']

    extra_kwargs = {}
    if cli_args.method == 'Spectrogram':
        extra_kwargs = {
            'n_fft': cli_args.n_fft if cli_args.n_fft > 0 else None,
            'hop_length': cli_args.hop_length if cli_args.hop_length > 0 else None,
            'win_length': cli_args.win_length if cli_args.win_length > 0 else None,
            'log_scale': not cli_args.no_log,
            'colormap': cli_args.colormap,
        }

    root = cfg['root_path'].rstrip('/')
    cache_dir = f"{root}/dino_embeddings_{cfg['data']}_{cli_args.method}_pervar"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Dataset:    {cli_args.dataset}  (data={cfg['data']})")
    print(f"Method:     {cli_args.method}")
    print(f"Device:     {DEVICE}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Win size:   {cli_args.win_size}")
    print(f"Step:       {step}  (stock train stride = {cfg['default_step']})")
    print(f"Splits:     {cli_args.flags}")
    if step != cfg['default_step']:
        print("⚠️  --step differs from the stock loader stride. Training MUST use the "
              "SAME stride or it will request un-cached windows (FileNotFoundError).")

    print("\nLoading DINOv2-base...")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    dino.eval()

    existing = set(f.replace('.npy', '') for f in os.listdir(cache_dir) if f.endswith('.npy'))
    print(f"Existing cached: {len(existing)}")

    for flag in cli_args.flags:
        print(f"\n{'='*50}\nProcessing {flag.upper()} split...\n{'='*50}")
        data_set = cfg['loader'](
            args=None, root_path=cfg['root_path'], win_size=cli_args.win_size,
            step=step, flag=flag)
        data_loader = DataLoader(
            data_set, batch_size=cli_args.batch_size, shuffle=False,
            num_workers=cli_args.num_workers, drop_last=False)
        print(f"{flag}: {len(data_set)} windows")

        generated, skipped = 0, 0
        with torch.no_grad():
            for batch_x, _ in tqdm(data_loader):
                batch_x = batch_x.float().to(DEVICE)  # [B, win_size, N]
                for b in range(batch_x.shape[0]):
                    sample = batch_x[b]                       # [win_size, N]
                    ts_hash = get_ts_hash(sample.cpu().numpy().astype(np.float32))
                    if ts_hash in existing:
                        skipped += 1
                        continue
                    cache_path = os.path.join(cache_dir, f"{ts_hash}.npy")
                    if os.path.exists(cache_path):
                        existing.add(ts_hash)
                        skipped += 1
                        continue

                    # [1, win_size, N] -> [1, N, 3, H, W]
                    x_images = transform_fn(
                        sample.unsqueeze(0), image_size=IMAGE_SIZE, device=DEVICE,
                        **extra_kwargs)
                    var_images = x_images[0]                  # [N, 3, H, W]
                    N = var_images.shape[0]

                    embeddings = []
                    for start in range(0, N, cli_args.dino_batch_size):
                        chunk = var_images[start:start + cli_args.dino_batch_size]
                        out = dino(pixel_values=chunk)
                        embeddings.append(out.last_hidden_state[:, 0, :])  # [chunk, 768]
                    embedding = torch.cat(embeddings, dim=0)              # [N, 768]

                    np.save(cache_path, embedding.cpu().numpy())
                    existing.add(ts_hash)
                    generated += 1

        print(f"  Generated: {generated}, Skipped: {skipped}")

    total = len([f for f in os.listdir(cache_dir) if f.endswith('.npy')])
    print(f"\nDone! Total embeddings in cache: {total}")


if __name__ == "__main__":
    main()
