"""
Precompute TimesFM forecasts as a regularization target (non-vision teacher).

This is the TimesFM counterpart to the VisionTS++ prediction regularizer. It lets
you ablate "is the *vision* the thing that helps, or just a strong foundation-model
forecast?" — TimesFM is a decoder-only TS foundation model with no vision at all.

For each input window [seq_len, N] (hashed, same md5 scheme as the other teachers),
TimesFM forecasts each variable (univariate, channel-independent) for `pred_len`
steps and we cache [pred_len, N]. Training then adds:

    L += lambda * MSE(student_forecast, timesfm_forecast)

We feed the SCALED series (data_provider's StandardScaler); TimesFM instance-norms
internally and denorms, so the cached forecast is in the same scaled space as the
student's output -> the MSE is well posed.

Cache (horizon-specific, unlike the embedding teachers):
    {root}/timesfm_forecast_{stem}_{pred_len}/{hash}.npy        # [pred_len, N]
where stem = data_path without ".csv" (ETTh1, exchange_rate, ...).

Setup:
    pip install timesfm[torch]        # TimesFM 3.0; weights auto-download from HF

Usage:
    python utils/precompute_timesfm_forecast.py --dataset ETTh1 --pred_len 96
    # one run per (dataset, horizon):
    for pl in 96 192 336 720; do
      python utils/precompute_timesfm_forecast.py --dataset ETTh1 --pred_len $pl
    done
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATASET_CONFIGS = {
    'ETTh1': {'data': 'ETTh1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh1.csv', 'freq': 'h'},
    'ETTh2': {'data': 'ETTh2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh2.csv', 'freq': 'h'},
    'ETTm1': {'data': 'ETTm1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm1.csv', 'freq': 't'},
    'ETTm2': {'data': 'ETTm2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm2.csv', 'freq': 't'},
    'weather': {'data': 'custom', 'root_path': './dataset/weather/', 'data_path': 'weather.csv', 'freq': 'h'},
    'exchange_rate': {'data': 'custom', 'root_path': './dataset/exchange_rate/', 'data_path': 'exchange_rate.csv', 'freq': 'd'},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_timesfm(device, batch_size):
    """Load TimesFM 3.0 (PyTorch). Import name differs by version, so try both."""
    try:
        from timesfm3 import TimesFM3Evaluator, ModelConfig
    except ImportError:
        from timesfm import TimesFM3Evaluator, ModelConfig  # fallback for other layouts
    cfg = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=batch_size,
        device=device,
    )
    return TimesFM3Evaluator(cfg)


def forecast_batch(forecaster, contexts, horizon):
    """contexts: list of 1D float32 arrays -> [len(contexts), horizon]."""
    outs = list(forecaster.predict_batch(contexts, horizon=horizon,
                                         return_quantiles=False))
    return np.stack([np.asarray(o.forecast)[:horizon] for o in outs], axis=0)


def get_ts_hash(ts_array):
    return hashlib.md5(ts_array.astype(np.float32).tobytes()).hexdigest()[:16]


def main():
    from data_provider.data_factory import data_provider

    p = argparse.ArgumentParser(description="Precompute TimesFM forecasts (reg target)")
    p.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=0)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--batch_size', type=int, default=32, help='data_provider batch size')
    p.add_argument('--tfm_batch', type=int, default=32, help='TimesFM per_core_batch_size')
    args = p.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    stem = os.path.splitext(cfg['data_path'])[0]
    cache_dir = os.path.join(cfg['root_path'], f"timesfm_forecast_{stem}_{args.pred_len}")
    os.makedirs(cache_dir, exist_ok=True)

    data_args = argparse.Namespace(
        data=cfg['data'], root_path=cfg['root_path'], data_path=cfg['data_path'],
        features='M', target='OT', seq_len=args.seq_len, label_len=args.label_len,
        pred_len=args.pred_len, embed='timeF', freq=cfg['freq'],
        batch_size=args.batch_size, num_workers=0, seasonal_patterns='Monthly',
        augmentation_ratio=0, task_name='long_term_forecast',
    )

    print(f"Dataset: {args.dataset}  pred_len: {args.pred_len}  device: {DEVICE}")
    print(f"Cache:   {cache_dir}")
    print("Loading TimesFM 3.0 ...")
    forecaster = load_timesfm(DEVICE, args.tfm_batch)

    existing = set(f[:-4] for f in os.listdir(cache_dir) if f.endswith('.npy'))
    for flag in ['train', 'val', 'test']:
        print(f"\n{'='*50}\n{flag.upper()}\n{'='*50}")
        _, loader = data_provider(data_args, flag)
        generated, skipped = 0, 0
        for batch_x, _, _, _ in tqdm(loader):
            x = batch_x.float().numpy().astype(np.float32)   # [B, L, N]
            B, L, N = x.shape
            contexts, metas = [], []
            for b in range(B):
                h = get_ts_hash(x[b])
                if h in existing:
                    skipped += 1
                    continue
                for n in range(N):
                    contexts.append(x[b, :, n])              # [L]
                metas.append((h, N))
            if not contexts:
                continue
            fc = forecast_batch(forecaster, contexts, args.pred_len)  # [sum_N, pred_len]
            idx = 0
            for (h, n_vars) in metas:
                f = fc[idx:idx + n_vars].T.astype(np.float32)        # [pred_len, N]
                np.save(os.path.join(cache_dir, f"{h}.npy"), f)
                existing.add(h)
                idx += n_vars
                generated += 1
        print(f"  Generated: {generated}, Skipped: {skipped}")

    total = len([f for f in os.listdir(cache_dir) if f.endswith('.npy')])
    print(f"\nDone! Total cached forecasts: {total}  ->  {cache_dir}")


if __name__ == '__main__':
    main()
