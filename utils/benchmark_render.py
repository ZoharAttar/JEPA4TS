"""
Micro-benchmark for the rendering + DINO teacher cost (precompute).

Times ONE window and lets you multiply by the number of windows, so you never
have to run the full precompute to estimate its cost. Reports:
  * per-window rendering time (RP / GAF / LinePlot / Spectrogram)
  * per-window rendering + DINOv2 embedding time (the real precompute unit)
  * peak GPU memory used by DINO
  * an estimated TOTAL precompute time for --n_windows windows

Rendering/DINO cost depends on the window SHAPE (seq_len, n_vars), not the actual
values, so we use a random window of the right shape -- this is faithful for
timing/memory while keeping the script trivial and dependency-light.

Usage
-----
    # PSM anomaly window: 100 steps, 25 variables, spectrogram, 132382 windows
    python utils/benchmark_render.py --seq_len 100 --n_vars 25 \
        --method Spectrogram --n_windows 132382

    # Exchange forecasting window: 96 steps, 8 variables, RP
    python utils/benchmark_render.py --seq_len 96 --n_vars 8 --method RP \
        --n_windows 5120

    # time all four renderings at once
    python utils/benchmark_render.py --seq_len 100 --n_vars 25 \
        --method RP GAF LinePlot Spectrogram --n_windows 132382
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.precompute_embeddings_pervar import TRANSFORM_MAP, IMAGE_SIZE, DEVICE


def _sync():
    if DEVICE == 'cuda':
        torch.cuda.synchronize()


def fmt_time(s):
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s/60:.1f}min"
    return f"{s/3600:.2f}h"


def bench_method(method, x, dino, dino_batch_size, warmup, iters, extra_kwargs):
    transform_fn = TRANSFORM_MAP[method]

    # ---- warmup (CUDA init, cudnn autotune, model on device) ----
    for _ in range(warmup):
        imgs = transform_fn(x, image_size=IMAGE_SIZE, device=DEVICE, **extra_kwargs)
        var_images = imgs[0]
        with torch.no_grad():
            for s in range(0, var_images.shape[0], dino_batch_size):
                dino(pixel_values=var_images[s:s + dino_batch_size])
    _sync()

    # ---- time rendering only ----
    t0 = time.perf_counter()
    for _ in range(iters):
        imgs = transform_fn(x, image_size=IMAGE_SIZE, device=DEVICE, **extra_kwargs)
    _sync()
    render_t = (time.perf_counter() - t0) / iters

    # ---- time rendering + DINO (the real precompute unit) ----
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        imgs = transform_fn(x, image_size=IMAGE_SIZE, device=DEVICE, **extra_kwargs)
        var_images = imgs[0]
        with torch.no_grad():
            embs = []
            for s in range(0, var_images.shape[0], dino_batch_size):
                out = dino(pixel_values=var_images[s:s + dino_batch_size])
                embs.append(out.last_hidden_state[:, 0, :])
            _ = torch.cat(embs, dim=0)
    _sync()
    full_t = (time.perf_counter() - t0) / iters
    peak_mem = (torch.cuda.max_memory_allocated() / 1024**3) if DEVICE == 'cuda' else 0.0

    return render_t, full_t, peak_mem


def main():
    p = argparse.ArgumentParser(description="Rendering + DINO precompute micro-benchmark")
    p.add_argument('--seq_len', type=int, required=True, help='window length (e.g. 100 anomaly, 96 forecast)')
    p.add_argument('--n_vars', type=int, required=True, help='number of variables N (per-variable rendering)')
    p.add_argument('--method', type=str, nargs='+', default=['RP'],
                   choices=list(TRANSFORM_MAP.keys()))
    p.add_argument('--n_windows', type=int, default=1,
                   help='total windows to extrapolate the estimate to (see the per-dataset count)')
    p.add_argument('--dino_batch_size', type=int, default=8, help='variables fed to DINO at once')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10, help='timed windows to average over')
    # spectrogram knobs (match training defaults)
    p.add_argument('--n_fft', type=int, default=0)
    p.add_argument('--hop_length', type=int, default=0)
    p.add_argument('--win_length', type=int, default=0)
    p.add_argument('--no_log', action='store_true')
    p.add_argument('--colormap', type=str, default=None)
    args = p.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Window: seq_len={args.seq_len}, n_vars={args.n_vars}  "
          f"(=> {args.n_vars} images per window, image size {IMAGE_SIZE})")
    print(f"Windows to extrapolate: {args.n_windows}  "
          f"(=> {args.n_windows * args.n_vars} total renders)\n")

    print("Loading DINOv2-base...")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    dino.eval()

    # random window of the correct shape: [1, seq_len, n_vars]
    x = torch.randn(1, args.seq_len, args.n_vars, device=DEVICE)

    print(f"\n{'Method':12s} {'render/win':>12s} {'render+DINO/win':>16s} "
          f"{'peak GPU':>10s} {'est. total':>12s}")
    print("-" * 66)
    for method in args.method:
        extra_kwargs = {}
        if method == 'Spectrogram':
            extra_kwargs = {
                'n_fft': args.n_fft if args.n_fft > 0 else None,
                'hop_length': args.hop_length if args.hop_length > 0 else None,
                'win_length': args.win_length if args.win_length > 0 else None,
                'log_scale': not args.no_log,
                'colormap': args.colormap,
            }
        render_t, full_t, peak_mem = bench_method(
            method, x, dino, args.dino_batch_size, args.warmup, args.iters, extra_kwargs)
        est_total = full_t * args.n_windows
        print(f"{method:12s} {render_t*1000:10.2f}ms {full_t*1000:14.2f}ms "
              f"{peak_mem:8.2f}GB {fmt_time(est_total):>12s}")

    print("\nNote: 'est. total' = (render+DINO per window) x --n_windows, single process.")
    print("Divide by the number of parallel GPUs/processes you will actually use.")


if __name__ == "__main__":
    main()
