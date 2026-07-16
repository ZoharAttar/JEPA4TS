"""
Inference (forward-only) latency micro-benchmark for the TimeMixer backbone.

This is the test-time runtime that both TimeMixer AND VibeTS incur: at inference
VibeTS single-encoder forecasting runs ONLY student.forecast_decode (no predictor,
no teacher, no rendering), so its latency equals the plain TimeMixer's.

Reports ms/batch, ms/sample, and peak GPU memory. Use --batch_size 1 for
per-sample latency (matches a "Estimated Runtime" column), or a larger batch for
throughput.

Usage
-----
    # Exchange config, per-sample latency
    python utils/benchmark_infer.py --enc_in 8 --d_model 16 --d_ff 32 \
        --pred_len 96 --batch_size 1

    # ETTm2 config
    python utils/benchmark_infer.py --enc_in 7 --d_model 32 --d_ff 64 --pred_len 96
"""
import argparse
import time
import torch
from models import TimeMixer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enc_in', type=int, default=8)
    p.add_argument('--d_model', type=int, default=16)
    p.add_argument('--d_ff', type=int, default=32)
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--down_sampling_layers', type=int, default=3)
    p.add_argument('--down_sampling_window', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=200)
    args = p.parse_args()

    cfg = argparse.Namespace(
        task_name='long_term_forecast', seq_len=args.seq_len, label_len=0,
        pred_len=args.pred_len, enc_in=args.enc_in, dec_in=args.enc_in, c_out=args.enc_in,
        d_model=args.d_model, d_ff=args.d_ff, e_layers=args.e_layers, d_layers=1,
        n_heads=8, factor=1, dropout=0.1, embed='timeF', freq='h', activation='gelu',
        moving_avg=25, decomp_method='moving_avg', top_k=5, use_norm=1,
        channel_independence=1, down_sampling_layers=args.down_sampling_layers,
        down_sampling_window=args.down_sampling_window, down_sampling_method='avg')

    model = TimeMixer.Model(cfg).float().to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())

    B, L, N = args.batch_size, args.seq_len, args.enc_in
    x = torch.randn(B, L, N, device=DEVICE)
    x_mark = torch.zeros(B, L, 4, device=DEVICE)
    # TimeMixer forecast ignores dec inputs internally; pass zeros of right shape.
    x_dec = torch.zeros(B, args.pred_len, N, device=DEVICE)
    x_mark_dec = torch.zeros(B, args.pred_len, 4, device=DEVICE)

    def sync():
        if DEVICE == 'cuda':
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x, x_mark, x_dec, x_mark_dec)
        sync()
        if DEVICE == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(x, x_mark, x_dec, x_mark_dec)
        sync()
        dt = (time.perf_counter() - t0) / args.iters

    peak = (torch.cuda.max_memory_allocated() / 1024**2) if DEVICE == 'cuda' else 0.0
    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ""))
    print(f"Config: enc_in={N}, d_model={args.d_model}, pred_len={args.pred_len}, "
          f"params={n_params:,}")
    print(f"Batch size: {B}")
    print(f"Latency: {dt*1000:.3f} ms/batch  |  {dt*1000/B:.4f} ms/sample")
    print(f"Peak GPU: {peak:.1f} MB")


if __name__ == "__main__":
    main()
