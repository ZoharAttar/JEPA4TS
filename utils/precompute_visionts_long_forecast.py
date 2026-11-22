"""
Precompute LONG-CONTEXT VisionTS++ forecasts as the prediction-regularizer target.

The student keeps its 96-step look-back; the frozen VisionTS++ teacher sees a much
longer history ending at the SAME time step (the paper's zero-shot setting, e.g.
ETTm1: 2304 steps with P=96) and forecasts the same next pred_len steps:

    time ──────────────────────────────┬──────────►
    teacher: [════ teacher_seq_len ════]│ → teacher forecast
    student:                 [═ 96 ═════]│ → student forecast

Each teacher forecast [pred_len, N] is cached under the md5 hash of the STUDENT's
96-step window (same hash as the other cached teachers), so training looks it up
from the student's own batch. Only the TRAIN split is needed (the regularizer is a
training-only loss). The earliest student windows have less than teacher_seq_len of
history before them, so they get no cache entry; training masks the reg term there.

Scaling: data_provider's StandardScaler is fit on the full train split regardless of
seq_len, so the student and teacher windows see identical scaled values.

Cache:  {root}/visionts_long_forecast_{stem}_T{teacher_seq_len}_P{P}_{pred_len}/{hash}.npy

Usage (zohar_env, GPU):
    python utils/precompute_visionts_long_forecast.py --dataset ETTm1 --pred_len 96 \
        --teacher_seq_len 2304 --periodicity 96
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.eval_visionts_baseline import load_visionts_forecaster  # noqa: E402

DATASET_CONFIGS = {
    'ETTh1': {'data': 'ETTh1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh1.csv', 'freq': 'h'},
    'ETTh2': {'data': 'ETTh2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh2.csv', 'freq': 'h'},
    'ETTm1': {'data': 'ETTm1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm1.csv', 'freq': 't'},
    'ETTm2': {'data': 'ETTm2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm2.csv', 'freq': 't'},
    'weather': {'data': 'custom', 'root_path': './dataset/weather/', 'data_path': 'weather.csv', 'freq': 'h'},
    'electricity': {'data': 'custom', 'root_path': './dataset/electricity/', 'data_path': 'electricity.csv', 'freq': 'h'},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_ts_hash(ts_array):
    return hashlib.md5(ts_array.astype(np.float32).tobytes()).hexdigest()[:16]


@torch.no_grad()
def forecast_mv_grouped(model, x, pred_len, group=21):
    """VisionTS++ MV forecast; splits wide datasets into channel groups of <= `group`
    variables (same rule as the on-the-fly regularizer). x: [B, L, N] -> [B, pred_len, N]."""
    def fwd(inp):
        out = model.forward(inp)
        return out[0] if isinstance(out, (list, tuple)) else out
    B, L, N = x.shape
    if N <= group:
        return fwd(x)[:, -pred_len:, :]
    n_chunks = -(-N // group)
    pad = n_chunks * group - N
    if pad:
        x = torch.cat([x, x[..., -1:].expand(B, L, pad)], dim=-1)
    xc = x.reshape(B, L, n_chunks, group).permute(0, 2, 1, 3).reshape(B * n_chunks, L, group)
    y = fwd(xc)[:, -pred_len:, :]
    y = y.reshape(B, n_chunks, pred_len, group).permute(0, 2, 1, 3).reshape(B, pred_len, n_chunks * group)
    return y[..., :N]


def main():
    from data_provider.data_factory import data_provider

    p = argparse.ArgumentParser(description="Precompute long-context VisionTS++ forecasts (reg target)")
    p.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument('--pred_len', type=int, required=True)
    p.add_argument('--teacher_seq_len', type=int, required=True, help='teacher look-back, e.g. 2304')
    p.add_argument('--periodicity', type=int, required=True, help='VisionTS++ P, e.g. 96 for ETTm')
    p.add_argument('--student_seq_len', type=int, default=96)
    p.add_argument('--arch', type=str, default='mae_base')
    p.add_argument('--ckpt_dir', type=str, default='./hf_models/VisionTSpp')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--norm_const', type=float, default=0.4)
    p.add_argument('--align_const', type=float, default=0.4)
    args = p.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    stem = os.path.splitext(cfg['data_path'])[0]
    T, S, P = args.teacher_seq_len, args.student_seq_len, args.periodicity
    cache_dir = os.path.join(cfg['root_path'], f"visionts_long_forecast_{stem}_T{T}_P{P}_{args.pred_len}")
    os.makedirs(cache_dir, exist_ok=True)

    data_args = argparse.Namespace(
        data=cfg['data'], root_path=cfg['root_path'], data_path=cfg['data_path'],
        features='M', target='OT', seq_len=T, label_len=0, pred_len=args.pred_len,
        embed='timeF', freq=cfg['freq'], batch_size=args.batch_size, num_workers=0,
        seasonal_patterns='Monthly', augmentation_ratio=0, task_name='long_term_forecast',
    )
    train_set, _ = data_provider(data_args, 'train')
    # Our own loader: no shuffle, keep the last partial batch (the default train loader drops it).
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    print(f"Dataset: {args.dataset}  pred_len: {args.pred_len}  teacher T={T}  P={P}  student S={S}  device: {DEVICE}")
    print(f"Teacher windows in train split: {len(train_set)}")
    print(f"Cache: {cache_dir}")

    model = load_visionts_forecaster(args.arch, None, args.ckpt_dir, True, DEVICE)  # color=True (MV)
    model.update_config(context_len=T, pred_len=args.pred_len, periodicity=P,
                        norm_const=args.norm_const, align_const=args.align_const)

    existing = set(f[:-4] for f in os.listdir(cache_dir) if f.endswith('.npy'))
    generated = skipped = 0
    for batch_x, _, _, _ in tqdm(loader):
        x = batch_x.float()                                    # [B, T, N]
        hashes = [get_ts_hash(x[b, -S:, :].numpy()) for b in range(x.shape[0])]
        keep = [i for i, h in enumerate(hashes) if h not in existing]
        skipped += len(hashes) - len(keep)
        if not keep:
            continue
        y = forecast_mv_grouped(model, x[keep].to(DEVICE), args.pred_len).cpu().numpy()
        for j, i in enumerate(keep):
            np.save(os.path.join(cache_dir, f"{hashes[i]}.npy"), y[j].astype(np.float32))
            existing.add(hashes[i])
            generated += 1

    n_student = len(train_set) + T - S   # student windows in the same split (seq_len S)
    print(f"\nGenerated: {generated}, skipped: {skipped}")
    print(f"Coverage: {len(existing)} of ~{n_student} student train windows "
          f"(~{100 * len(existing) / max(n_student, 1):.1f}%); the rest are masked in training.")


if __name__ == '__main__':
    main()
