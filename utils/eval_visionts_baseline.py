"""
Standalone VisionTS++ baseline forecaster (NOT the teacher, NOT your JEPA model).

Runs the pretrained VisionTS++ model as a zero-shot forecaster on a dataset and
reports MSE/MAE in the SAME scaled space your JEPA runs use (data_provider applies
StandardScaler; VisionTS++ does its own internal instance-norm + denorm, so feeding
scaled data yields scaled predictions -> directly comparable to your JEPA numbers).

Two rendering modes (the ablation):
  --mode ci   : channel-INDEPENDENT. Each variable is rendered as its own full
                224x224 grayscale image and forecast separately (original-VisionTS
                style). Input folded to [B*N, L, 1].
  --mode mv   : colorized MULTIVARIATE. All variables packed into ONE RGB image,
                each variable on a color channel (VisionTS++'s native mode,
                color=True). Input kept as [B, L, N].

Both modes use the SAME frozen VisionTS++ weights and the SAME look-back, so the
only thing that changes is the rendering -> isolates "does the colorized
multivariate packing help?".

Setup (same as the teacher precompute):
    pip install git+https://github.com/Keytoyze/VisionTS.git

Usage:
    python utils/eval_visionts_baseline.py --dataset ETTh1 --mode mv \
        --seq_len 1728 --pred_len 96 --periodicity 24
    python utils/eval_visionts_baseline.py --dataset ETTh1 --mode ci \
        --seq_len 96 --pred_len 96 --periodicity 24
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import einops
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Per-dataset config: data_provider wiring + default periodicity for the image grid.
DATASET_CONFIGS = {
    'ETTh1': {'data': 'ETTh1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh1.csv',
              'freq': 'h', 'periodicity': 24},
    'ETTh2': {'data': 'ETTh2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh2.csv',
              'freq': 'h', 'periodicity': 24},
    'exchange_rate': {'data': 'custom', 'root_path': './dataset/exchange_rate/',
                      'data_path': 'exchange_rate.csv', 'freq': 'd', 'periodicity': 1},
}

ARCH_DIM = {'mae_base': 768, 'mae_large': 1024, 'mae_huge': 1280}
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")


def load_visionts_forecaster(arch, ckpt_path, ckpt_dir, color, device):
    """Load the full (encoder+decoder) VisionTS++ forecaster."""
    from visionts import VisionTSpp
    default_ckpt = ckpt_path or os.path.join(ckpt_dir, f"visiontspp_{arch.split('_')[1]}.ckpt")
    model = VisionTSpp(arch=arch, finetune_type='none', ckpt_path=default_ckpt, load_ckpt=True,
                       quantile=True, clip_input=0, complete_no_clip=False, color=color)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def forecast_mv(model, batch_x, pred_len):
    """Colorized multivariate: feed all variables at once -> [B, pred_len, N]."""
    y = model.forward(batch_x)                 # quantile=True -> [median, quantile_list]
    if isinstance(y, (list, tuple)):
        y = y[0]
    return y[:, -pred_len:, :]


@torch.no_grad()
def forecast_ci(model, batch_x, pred_len, var_chunk=64):
    """Channel-independent: fold variables into the batch, forecast each alone.

    batch_x: [B, L, N] -> [B, pred_len, N].
    """
    B, L, N = batch_x.shape
    x = einops.rearrange(batch_x, 'b l n -> (b n) l 1')      # [B*N, L, 1]
    outs = []
    for s in range(0, x.shape[0], var_chunk):
        xi = x[s:s + var_chunk]
        yi = model.forward(xi)
        if isinstance(yi, (list, tuple)):
            yi = yi[0]
        outs.append(yi[:, -pred_len:, :])                    # [chunk, pred_len, 1]
    y = torch.cat(outs, dim=0)                               # [B*N, pred_len, 1]
    return einops.rearrange(y, '(b n) p 1 -> b p n', b=B)    # [B, pred_len, N]


def main():
    from data_provider.data_factory import data_provider

    p = argparse.ArgumentParser(description="VisionTS++ standalone baseline forecaster")
    p.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument('--mode', type=str, required=True, choices=['ci', 'mv'],
                   help="ci = channel-independent (per-var grayscale), mv = colorized multivariate")
    p.add_argument('--arch', type=str, default='mae_base', choices=list(ARCH_DIM.keys()))
    p.add_argument('--ckpt_path', type=str, default=None)
    p.add_argument('--ckpt_dir', type=str, default='./hf_models/VisionTSpp')
    p.add_argument('--seq_len', type=int, default=96, help='look-back / context length')
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--periodicity', type=int, default=0, help='0 = dataset default')
    p.add_argument('--norm_const', type=float, default=0.4)
    p.add_argument('--align_const', type=float, default=0.4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--var_chunk', type=int, default=64, help='CI mode: max var-images per encoder call')
    p.add_argument('--results_file', type=str, default='result_visionts_baseline.txt')
    args = p.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    periodicity = args.periodicity if args.periodicity > 0 else cfg['periodicity']
    color = (args.mode == 'mv')   # colorized for multivariate; grayscale for CI

    data_args = argparse.Namespace(
        data=cfg['data'], root_path=cfg['root_path'], data_path=cfg['data_path'],
        features='M', target='OT', seq_len=args.seq_len, label_len=0, pred_len=args.pred_len,
        embed='timeF', freq=cfg['freq'], batch_size=args.batch_size, num_workers=0,
        seasonal_patterns='Monthly', augmentation_ratio=0, task_name='long_term_forecast',
    )

    tag = f"VisionTSpp_baseline_{args.dataset}_{args.mode}_{args.seq_len}_{args.pred_len}"
    print(f"{'='*60}\n{tag}\n{'='*60}")
    print(f"mode={args.mode}  color={color}  periodicity={periodicity}  device={DEVICE}")

    model = load_visionts_forecaster(args.arch, args.ckpt_path, args.ckpt_dir, color, DEVICE)
    model.update_config(context_len=args.seq_len, pred_len=args.pred_len,
                        periodicity=periodicity, norm_const=args.norm_const,
                        align_const=args.align_const)

    _, loader = data_provider(data_args, 'test')

    se_sum, ae_sum, count = 0.0, 0.0, 0   # running sums for MSE / MAE in scaled space
    for batch_x, batch_y, _, _ in tqdm(loader, desc=tag):
        batch_x = batch_x.float().to(DEVICE)
        true = batch_y[:, -args.pred_len:, :].float().to(DEVICE)
        if args.mode == 'mv':
            pred = forecast_mv(model, batch_x, args.pred_len)
        else:
            pred = forecast_ci(model, batch_x, args.pred_len, var_chunk=args.var_chunk)
        diff = (pred - true)
        se_sum += torch.sum(diff ** 2).item()
        ae_sum += torch.sum(torch.abs(diff)).item()
        count += diff.numel()

    mse = se_sum / count
    mae = ae_sum / count
    rmse = mse ** 0.5
    print(f"\n{tag}\nmse:{mse}, mae:{mae}, rmse:{rmse}")

    with open(args.results_file, 'a') as f:
        f.write(f"{tag}\nmse:{mse}, mae:{mae}, rmse:{rmse}\n\n")
    print(f"Appended to {args.results_file}")


if __name__ == '__main__':
    main()
