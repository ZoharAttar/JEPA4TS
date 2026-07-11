"""
Rendering-method analysis for ANOMALY DETECTION.

Goal: decide which image rendering (RP / GAF / LinePlot / Spectrogram) gives the
most *discriminative* frozen-DINO teacher for a given anomaly dataset — mirroring
the forecasting/classification rendering analysis in vision_model_test.ipynb.

Idea: a good rendering maps different time-series windows to well-SEPARATED DINO
embeddings (low similarity / high distance between samples). A rendering that
collapses everything to nearly the same image gives an uninformative teacher.

For each (dataset, method) we:
  1. sample train windows (unsupervised; a window is [win_size, N]),
  2. render every variable of every window and embed it with frozen DINOv2 (CLS, 768-d),
  3. measure how spread-out those per-variable embeddings are via
        - mean pairwise COSINE SIMILARITY  (lower  = more distinct = better)
        - mean pairwise EUCLIDEAN DISTANCE  (higher = more distinct = better)
     Both are computed on a random subsample of embeddings (--max_embeddings) so
     the O(M^2) pairwise matrix stays tractable.

The method with the LOWEST cosine similarity (equivalently the highest Euclidean
distance) is the recommended rendering for that dataset.

Usage
-----
    # one dataset, all four renderings
    python utils/analyze_rendering_anomaly.py --datasets PSM \
        --methods RP GAF LinePlot Spectrogram

    # all anomaly datasets, save a CSV
    python utils/analyze_rendering_anomaly.py \
        --datasets PSM MSL SMAP SMD SWAT \
        --methods RP GAF LinePlot Spectrogram \
        --n_windows 150 --csv rendering_anomaly_analysis.csv

Notes
-----
* This is analysis-only: it does NOT write to the teacher cache and does NOT need
  the precompute to have run. Use --n_windows to keep it fast (it renders every
  variable of every sampled window, so cost scales with n_windows * N).
* win_size defaults to 100 to match the anomaly training seq_len.
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.precompute_embeddings_pervar import TRANSFORM_MAP, IMAGE_SIZE, DEVICE
from utils.precompute_embeddings_pervar_anomaly import ANOMALY_CONFIGS


@torch.no_grad()
def collect_embeddings(dataset, method, win_size, step, n_windows,
                       dino, dino_batch_size, extra_kwargs, seed):
    """Render + embed a subsample of train windows; return [M, 768] (M = windows*N)."""
    cfg = ANOMALY_CONFIGS[dataset]
    transform_fn = TRANSFORM_MAP[method]
    data_set = cfg['loader'](args=None, root_path=cfg['root_path'],
                             win_size=win_size, step=step, flag='train')
    n_total = len(data_set)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n_total, size=min(n_windows, n_total), replace=False)

    embeddings = []
    for i in idxs:
        sample = torch.as_tensor(np.asarray(data_set[i][0]), dtype=torch.float32)  # [win, N]
        x_images = transform_fn(sample.unsqueeze(0).to(DEVICE),
                                image_size=IMAGE_SIZE, device=DEVICE, **extra_kwargs)
        var_images = x_images[0]  # [N, 3, H, W]
        N = var_images.shape[0]
        for start in range(0, N, dino_batch_size):
            chunk = var_images[start:start + dino_batch_size]
            out = dino(pixel_values=chunk)
            embeddings.append(out.last_hidden_state[:, 0, :].cpu())  # [chunk, 768]
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 768)


def pairwise_stats(emb, max_embeddings, seed):
    """Mean/std of off-diagonal pairwise cosine similarity and Euclidean distance."""
    M = emb.shape[0]
    if M > max_embeddings:
        rng = np.random.default_rng(seed)
        sel = rng.choice(M, size=max_embeddings, replace=False)
        emb = emb[sel]
        M = max_embeddings
    emb = emb.float()

    # cosine similarity
    normed = F.normalize(emb, dim=-1)
    cos = normed @ normed.T
    off = ~torch.eye(M, dtype=torch.bool)
    cos_off = cos[off]

    # euclidean distance
    dist = torch.cdist(emb.unsqueeze(0), emb.unsqueeze(0)).squeeze(0)
    dist_off = dist[off]

    return {
        'n_embeddings': M,
        'cos_mean': cos_off.mean().item(),
        'cos_std': cos_off.std().item(),
        'euc_mean': dist_off.mean().item(),
        'euc_std': dist_off.std().item(),
    }


def main():
    p = argparse.ArgumentParser(description="Rendering analysis for anomaly detection")
    p.add_argument('--datasets', type=str, nargs='+', default=['PSM'],
                   choices=list(ANOMALY_CONFIGS.keys()))
    p.add_argument('--methods', type=str, nargs='+',
                   default=['RP', 'GAF', 'LinePlot', 'Spectrogram'],
                   choices=list(TRANSFORM_MAP.keys()))
    p.add_argument('--win_size', type=int, default=100,
                   help='Window length (match anomaly training seq_len)')
    p.add_argument('--step', type=int, default=None,
                   help='Sliding stride (default = stock loader stride per dataset). '
                        'Only affects which windows are available to sample.')
    p.add_argument('--n_windows', type=int, default=150,
                   help='Train windows sampled per (dataset, method)')
    p.add_argument('--max_embeddings', type=int, default=1000,
                   help='Cap on embeddings used for the O(M^2) pairwise stats')
    p.add_argument('--dino_batch_size', type=int, default=8)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--csv', type=str, default=None, help='Optional path to write results CSV')
    # Spectrogram-only knobs (match training-time teacher defaults)
    p.add_argument('--n_fft', type=int, default=0)
    p.add_argument('--hop_length', type=int, default=0)
    p.add_argument('--win_length', type=int, default=0)
    p.add_argument('--no_log', action='store_true')
    p.add_argument('--colormap', type=str, default=None)
    args = p.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading DINOv2-base...")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    dino.eval()

    rows = []
    for dataset in args.datasets:
        cfg = ANOMALY_CONFIGS[dataset]
        step = args.step if args.step is not None else cfg['default_step']
        print(f"\n{'='*70}\nDataset: {dataset}  (win={args.win_size}, step={step})\n{'='*70}")
        for method in args.methods:
            extra_kwargs = {}
            if method == 'Spectrogram':
                extra_kwargs = {
                    'n_fft': args.n_fft if args.n_fft > 0 else None,
                    'hop_length': args.hop_length if args.hop_length > 0 else None,
                    'win_length': args.win_length if args.win_length > 0 else None,
                    'log_scale': not args.no_log,
                    'colormap': args.colormap,
                }
            emb = collect_embeddings(dataset, method, args.win_size, step,
                                     args.n_windows, dino, args.dino_batch_size,
                                     extra_kwargs, args.seed)
            if emb.shape[0] < 2:
                print(f"  {method:12s}: not enough embeddings ({emb.shape[0]})")
                continue
            stats = pairwise_stats(emb, args.max_embeddings, args.seed)
            stats.update({'dataset': dataset, 'method': method})
            rows.append(stats)
            print(f"  {method:12s}: cos_sim={stats['cos_mean']:.4f}±{stats['cos_std']:.4f}  "
                  f"euclid={stats['euc_mean']:.4f}±{stats['euc_std']:.4f}  "
                  f"(M={stats['n_embeddings']})")

        # recommendation per dataset: lowest cosine similarity
        d_rows = [r for r in rows if r['dataset'] == dataset]
        if d_rows:
            best_cos = min(d_rows, key=lambda r: r['cos_mean'])
            best_euc = max(d_rows, key=lambda r: r['euc_mean'])
            print(f"  --> lowest cos_sim : {best_cos['method']} "
                  f"({best_cos['cos_mean']:.4f})")
            print(f"  --> highest euclid : {best_euc['method']} "
                  f"({best_euc['euc_mean']:.4f})")
            tag = "AGREE" if best_cos['method'] == best_euc['method'] else "DISAGREE"
            print(f"  --> recommended rendering: {best_cos['method']}  [{tag}]")

    if args.csv and rows:
        keys = ['dataset', 'method', 'n_embeddings',
                'cos_mean', 'cos_std', 'euc_mean', 'euc_std']
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in keys})
        print(f"\nSaved CSV -> {args.csv}")


if __name__ == "__main__":
    main()
