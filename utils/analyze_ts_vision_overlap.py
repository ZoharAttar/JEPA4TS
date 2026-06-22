"""
analyze_ts_vision_overlap.py
Diagnoses representation overlap between TS encoder(s) and the frozen vision
teacher to decide whether a decorrelation loss is needed.

Granularities:
    sample             -> 1 vector per sample (pool over n_vars + time/patch)
    per_var            -> 1 vector per (sample, variable)  (pool over time/patch only)
                          --> requires --per_var_teacher AND encoding_type in {4D, 4D_timemixer}
    per_var_centered   -> per_var, but subtract per-variable mean (removes variable-identity confound)
    per_var_split      -> compute metrics per variable separately, report mean ± std across variables
    all                -> run all of the above in one pass

Pairs (auto-selected from architecture):
    single encoder:  A0(student vs teacher), D(pred_teacher vs teacher)
    dual encoder:    A (student1 vs pred_teacher), B(student1 vs student2), D
    multi encoder:   pairwise(student_i vs student_j), each(student_i vs teacher), D
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from models.JEPAVTS import Model as JEPAVTS


# ============================== ARGS ==============================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_name', type=str, default='long_term_forecast')
    p.add_argument('--model', type=str, default='JEPAVTS')
    p.add_argument('--model_id', type=str, default='analysis')
    p.add_argument('--student_model', type=str, required=True,
                   choices=['PatchTST', 'TimeMixer', 'TimesNet', 'DLinear', 'iTransformer'])
    # data
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--root_path', type=str, required=True)
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--freq', type=str, default='h')
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=64)
    # shapes
    p.add_argument('--seq_len', type=int, required=True)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, required=True)
    p.add_argument('--enc_in', type=int, default=7)
    p.add_argument('--dec_in', type=int, default=7)
    p.add_argument('--c_out', type=int, default=7)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--factor', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--activation', type=str, default='gelu')
    p.add_argument('--channel_independence', type=int, default=1)
    p.add_argument('--down_sampling_layers', type=int, default=0)
    p.add_argument('--down_sampling_window', type=int, default=1)
    p.add_argument('--down_sampling_method', type=str, default='avg')
    p.add_argument('--use_norm', type=int, default=1)
    p.add_argument('--decomp_method', type=str, default='moving_avg')
    p.add_argument('--moving_avg', type=int, default=25)
    # JEPA model knobs (must match the checkpoint)
    p.add_argument('--rendering_methods', type=str, nargs='*', default=None)
    p.add_argument('--per_var_teacher', action='store_true')
    p.add_argument('--use_dual_encoder', action='store_true')
    p.add_argument('--multi_encoder', action='store_true')
    p.add_argument('--multi_predictor', action='store_true')
    p.add_argument('--jepa_hidden_dim', type=int, default=512)
    p.add_argument('--learned_loss_weights', action='store_true')
    p.add_argument('--multi_rendering_alpha_mode', type=str, default='same')
    p.add_argument('--per_method_alphas', type=float, nargs='*', default=None)
    p.add_argument('--fusion_type', type=str, default='mlp')
    # analysis
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--flag', type=str, default='val', choices=['train', 'val', 'test'])
    p.add_argument('--max_samples', type=int, default=1024)
    p.add_argument('--kernel_cap', type=int, default=1024)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--granularity', type=str, default='sample',
                   choices=['sample', 'per_var', 'per_var_centered', 'per_var_split', 'all'])
    # device
    p.add_argument('--use_gpu', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--use_multi_gpu', action='store_true')
    p.add_argument('--num_class', type=int, default=0)
    return p.parse_args()


# ============================== POOLING ==============================
def reduce_time_keep_var(enc, encoding_type):
    """Pool only the time/patch axis, KEEP the variable axis. Returns [B, N, D]."""
    if encoding_type == '4D':              # PatchTST: [B, N, D, P]
        return enc.mean(dim=3)
    if encoding_type == '4D_timemixer':    # TimeMixer CI: [B, N, D, T]
        return enc.mean(dim=3)
    raise ValueError(
        f"per-var granularity requires encoding_type in {{4D, 4D_timemixer}}, "
        f"got {encoding_type}.  Run with --granularity sample instead.")


def reduce_to_sample(enc, encoding_type):
    """Reduce to [B, D] (pool over n_vars + time)."""
    if encoding_type == '4D':              # [B, N, D, P]
        return enc.mean(dim=(1, 3))
    if encoding_type == '4D_timemixer':    # [B, N, D, T]
        return enc.mean(dim=(1, 3))
    if encoding_type == '3D':              # [B, T, D]
        return enc.mean(dim=1)
    return enc                              # [B, D]


def teacher_keep_var(t):
    """Returns [B, N, 768] (requires per_var=True)."""
    if isinstance(t, list):
        t = t[0]
    if t.dim() != 3:
        raise ValueError("per-var analysis requires --per_var_teacher so teacher is [B, N, D].")
    return t


def teacher_to_sample(t):
    """Returns [B, 768]."""
    if isinstance(t, list):
        t = t[0]
    if t.dim() == 3:
        return t.mean(dim=1)
    return t


# ============================== METRICS ==============================
def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    num = (Xc.T @ Yc).pow(2).sum()
    den = (Xc.T @ Xc).norm() * (Yc.T @ Yc).norm() + 1e-12
    return (num / den).item()


def _rbf(X):
    sq = torch.cdist(X, X).pow(2)
    with torch.no_grad():
        sig = sq[sq > 0].median().sqrt().clamp(min=1e-6)
    return torch.exp(-sq / (2 * sig ** 2))


def rbf_cka(X, Y):
    n = X.shape[0]
    K, L = _rbf(X), _rbf(Y)
    H = torch.eye(n, device=X.device) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    num = (Kc * Lc).sum()
    den = (Kc * Kc).sum().sqrt() * (Lc * Lc).sum().sqrt() + 1e-12
    return (num / den).item()


def hsic(X, Y):
    n = X.shape[0]
    K, L = _rbf(X), _rbf(Y)
    H = torch.eye(n, device=X.device) - 1.0 / n
    return ((K @ H @ L @ H).diagonal().sum() / (n - 1) ** 2).item()


def cross_corr_frob(X, Y):
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
    C = (Xn.T @ Yn) / X.shape[0]
    return C.pow(2).mean().sqrt().item()


def linear_probe_r2(X, Y, ridge=1e-2):
    n, d = X.shape
    X1 = torch.cat([X, torch.ones(n, 1, device=X.device)], dim=1)
    A = X1.T @ X1 + ridge * torch.eye(d + 1, device=X.device)
    W = torch.linalg.solve(A, X1.T @ Y)
    Yh = X1 @ W
    sse = (Y - Yh).pow(2).sum(0)
    sst = (Y - Y.mean(0)).pow(2).sum(0).clamp(min=1e-12)
    return (1 - sse / sst).mean().item()


def shuffled_baseline(fn, X, Y, n_shuffles=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = [fn(X, Y[torch.randperm(Y.shape[0], generator=g)]) for _ in range(n_shuffles)]
    return float(np.mean(vals)), float(np.std(vals))


METRICS = [
    ('linear_cka',      linear_cka,      False),
    ('rbf_cka',         rbf_cka,         True),
    ('hsic',            hsic,            True),
    ('cross_corr_frob', cross_corr_frob, False),
    ('probe_r2_X→Y',    linear_probe_r2, False),
    ('probe_r2_Y→X',    lambda X, Y: linear_probe_r2(Y, X), False),
]


# ============================== COLLECTION ==============================
@torch.no_grad()
def collect(model, loader, device, max_samples, want_per_var):
    """
    Returns a dict of tensors. If want_per_var=True: shapes are [N_total, N_vars, D].
    Otherwise [N_total, D].
    Buckets are keyed by role: 'student' / 'student1' / 'student2' / 'student_i' /
                               'student_forecast' / 'pred_teacher' / 'teacher'.
    """
    model.eval()
    buckets = {}
    n_seen = 0

    arch_dual  = getattr(model, 'use_dual_encoder', False)
    arch_multi = getattr(model, 'use_multi_encoder', False)

    pool_s = (lambda e: reduce_time_keep_var(e, model.encoding_type)) if want_per_var else \
             (lambda e: reduce_to_sample(e, model.encoding_type))
    pool_t = teacher_keep_var if want_per_var else teacher_to_sample

    def _push(key, val):
        buckets.setdefault(key, []).append(val.float().cpu())

    for batch in loader:
        x = batch[0].float().to(device)
        if len(batch) >= 4:
            y, xm, ym = batch[1].float().to(device), batch[2].float().to(device), batch[3].float().to(device)
            dec_inp = torch.zeros_like(y[:, -model.configs.pred_len:, :]).float()
            dec_inp = torch.cat([y[:, :model.configs.label_len, :], dec_inp], dim=1).to(device)
        else:
            xm, ym, dec_inp = None, None, None

        if arch_multi and arch_dual:
            fc_enc, _, _ = model.student_forecast.encode(x, xm, dec_inp, ym)
            jepa_encs = [s.encode(x, xm, dec_inp, ym)[0] for s in model.students_multi]
            _push('student_forecast', pool_s(fc_enc))
            for i, e in enumerate(jepa_encs):
                _push(f'student_jepa_{i}', pool_s(e))
            pred_t = model._predict_teacher(
                torch.stack(jepa_encs, 0).mean(0) if not model.use_multi_predictor else jepa_encs)
            _push('pred_teacher', pool_t(pred_t))

        elif arch_multi:
            jepa_encs = [s.encode(x, xm, dec_inp, ym)[0] for s in model.students_multi]
            for i, e in enumerate(jepa_encs):
                _push(f'student_{i}', pool_s(e))
            pred_t = model._predict_teacher(
                torch.stack(jepa_encs, 0).mean(0) if not model.use_multi_predictor else jepa_encs)
            _push('pred_teacher', pool_t(pred_t))

        elif arch_dual:
            if model.encoding_type == '4D_timemixer':
                e1, _, _, _ = model._timemixer_encode(model.student1, x, xm, dec_inp, ym)
                e2, _, _, _ = model._timemixer_encode(model.student2, x, xm, dec_inp, ym)
            else:
                e1, _, _ = model.student1.encode(x, xm, dec_inp, ym)
                e2, _, _ = model.student2.encode(x, xm, dec_inp, ym)
            _push('student1', pool_s(e1))
            _push('student2', pool_s(e2))
            _push('pred_teacher', pool_t(model._predict_teacher(e2)))

        else:
            if model.encoding_type == '4D_timemixer':
                e, _, _, _ = model._timemixer_encode(model.student, x, xm, dec_inp, ym)
            else:
                e, _, _ = model.student.encode(x, xm, dec_inp, ym)
            _push('student', pool_s(e))
            _push('pred_teacher', pool_t(model._predict_teacher(e)))

        _push('teacher', pool_t(model.teacher_forward(x)))

        n_seen += x.shape[0]
        if n_seen >= max_samples:
            break

    return {k: torch.cat(v, 0)[:max_samples] for k, v in buckets.items()}


# ============================== REPORTING ==============================
def _print_table(title, rows):
    width = 90
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"{'Metric':<18} {'Observed':>10} {'Shuffled µ ± σ':>22} {'Δ/σ':>8} {'Verdict':>16}")
    print("-" * width)
    for m, obs, mu, sd, z, verdict in rows:
        print(f"{m:<18} {obs:>10.4f} {mu:>10.4f} ± {sd:<7.4f} {z:>7.1f} {verdict:>16}")


def _verdict(z):
    if z > 4:    return 'SHARED ⚠'
    if z > 2:    return 'some overlap'
    return 'independent ✓'


def report_pair(title, X, Y, kernel_cap, seed=0):
    rows, out = [], {}
    for mname, fn, is_kernel in METRICS:
        Xc = X[:kernel_cap] if is_kernel else X
        Yc = Y[:kernel_cap] if is_kernel else Y
        obs = fn(Xc, Yc)
        mu, sd = shuffled_baseline(fn, Xc, Yc, n_shuffles=5, seed=seed)
        z = (obs - mu) / (sd + 1e-12)
        rows.append((mname, obs, mu, sd, z, _verdict(z)))
        out[mname] = {'observed': obs, 'shuffled_mean': mu, 'shuffled_std': sd, 'z': z}
    _print_table(title + f"   X: {tuple(X.shape)}   Y: {tuple(Y.shape)}", rows)
    return out


def report_pair_per_var_split(title, X3, Y3, kernel_cap, seed=0):
    """X3, Y3: [B, N, D]. Returns per-variable rows + per-metric mean/std across vars."""
    N = X3.shape[1]
    per_var = []
    for n in range(N):
        Xn, Yn = X3[:, n, :], Y3[:, n, :]
        row = {}
        for mname, fn, is_kernel in METRICS:
            Xc = Xn[:kernel_cap] if is_kernel else Xn
            Yc = Yn[:kernel_cap] if is_kernel else Yn
            obs = fn(Xc, Yc)
            mu, sd = shuffled_baseline(fn, Xc, Yc, n_shuffles=3, seed=seed + n)
            row[mname] = {'observed': obs, 'shuffled_mean': mu, 'shuffled_std': sd,
                          'z': (obs - mu) / (sd + 1e-12)}
        per_var.append(row)

    # Aggregate
    agg_rows = []
    agg = {}
    for mname, _, _ in METRICS:
        obs_arr = np.array([v[mname]['observed'] for v in per_var])
        mu_arr  = np.array([v[mname]['shuffled_mean'] for v in per_var])
        sd_arr  = np.array([v[mname]['shuffled_std'] for v in per_var])
        z_arr   = np.array([v[mname]['z'] for v in per_var])
        agg_rows.append((mname, obs_arr.mean(), mu_arr.mean(), sd_arr.mean(),
                         z_arr.mean(), _verdict(z_arr.mean())))
        agg[mname] = {
            'observed_mean': float(obs_arr.mean()), 'observed_std': float(obs_arr.std()),
            'z_mean': float(z_arr.mean()), 'z_std': float(z_arr.std()),
            'per_variable_observed': obs_arr.tolist(),
            'per_variable_z':        z_arr.tolist(),
        }
    _print_table(title + f"   per-var mean ± aggregated over N={N}", agg_rows)
    # Quick per-variable CKA summary
    print("\n  per-variable linear_cka:")
    for n in range(N):
        print(f"    var {n:>2}:  observed={per_var[n]['linear_cka']['observed']:.4f}   "
              f"Δ/σ={per_var[n]['linear_cka']['z']:+.1f}")
    return agg


# ============================== PAIR SELECTION ==============================
def select_pairs(buckets):
    """Returns list of (label, X_key, Y_key) tuples based on which buckets exist."""
    has = lambda *keys: all(k in buckets for k in keys)
    pairs = []
    if has('student', 'teacher'):
        pairs += [
            ('A0:  student vs teacher  (TS vs vision)', 'student', 'teacher'),
            ('D :  pred_teacher vs teacher  (JEPA sanity)', 'pred_teacher', 'teacher'),
        ]
    if has('student1', 'student2'):
        pairs += [
            ('A :  student1 vs pred_teacher  (forecast vs vision-aligned)',
             'student1', 'pred_teacher'),
            ('B :  student1 vs student2  (fusion-input redundancy)',
             'student1', 'student2'),
            ('D :  pred_teacher vs teacher  (JEPA sanity)',
             'pred_teacher', 'teacher'),
        ]
    multi_keys = sorted([k for k in buckets if k.startswith('student_') and k != 'student_forecast'])
    if multi_keys:
        for i, ki in enumerate(multi_keys):
            for kj in multi_keys[i + 1:]:
                pairs.append((f'pair:  {ki} vs {kj}', ki, kj))
        for ki in multi_keys:
            pairs.append((f'{ki} vs teacher', ki, 'teacher'))
        if 'student_forecast' in buckets:
            pairs.append(('student_forecast vs teacher', 'student_forecast', 'teacher'))
        pairs.append(('D :  pred_teacher vs teacher  (JEPA sanity)', 'pred_teacher', 'teacher'))
    return pairs


# ============================== DRIVER ==============================
def run_granularity(granularity, buckets, args):
    print("\n\n" + "#" * 90)
    print(f"#  GRANULARITY = {granularity}")
    print("#" * 90)
    results = {}
    pairs = select_pairs(buckets)
    for label, ka, kb in pairs:
        X3 = buckets[ka]   # [B, N, D] if per_var else [B, D]
        Y3 = buckets[kb]
        if granularity == 'sample':
            X = X3 if X3.dim() == 2 else X3.mean(1)
            Y = Y3 if Y3.dim() == 2 else Y3.mean(1)
            results[label] = report_pair(label, X, Y, args.kernel_cap, args.seed)
        elif granularity == 'per_var':
            X = X3.reshape(-1, X3.shape[-1])
            Y = Y3.reshape(-1, Y3.shape[-1])
            results[label] = report_pair(label, X, Y, args.kernel_cap, args.seed)
        elif granularity == 'per_var_centered':
            Xc = (X3 - X3.mean(0, keepdim=True)).reshape(-1, X3.shape[-1])
            Yc = (Y3 - Y3.mean(0, keepdim=True)).reshape(-1, Y3.shape[-1])
            results[label] = report_pair(label, Xc, Yc, args.kernel_cap, args.seed)
        elif granularity == 'per_var_split':
            results[label] = report_pair_per_var_split(label, X3, Y3, args.kernel_cap, args.seed)
    return results


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    needs_per_var = args.granularity != 'sample'
    if needs_per_var and not args.per_var_teacher:
        raise SystemExit("Granularity ≠ sample requires --per_var_teacher (so teacher is per-channel).")

    print(f"\n=== Building JEPAVTS  student={args.student_model}  "
          f"dual={args.use_dual_encoder}  multi={args.multi_encoder}  "
          f"per_var={args.per_var_teacher} ===")
    model = JEPAVTS(args).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        miss, unexp = model.load_state_dict(state, strict=False)
        print(f"Loaded {args.checkpoint}   (missing={len(miss)}, unexpected={len(unexp)})")
    else:
        print("⚠ No checkpoint — analysing UNTRAINED model (use only as architectural baseline).")

    _, loader = data_provider(args, flag=args.flag)
    print(f"\nCollecting encodings from {args.flag} loader (cap {args.max_samples} samples)...")
    buckets = collect(model, loader, device, args.max_samples, want_per_var=needs_per_var)
    for k, v in buckets.items():
        print(f"  {k:<22} shape={tuple(v.shape)}")

    granularities = (['sample', 'per_var', 'per_var_centered', 'per_var_split']
                     if args.granularity == 'all' else [args.granularity])
    if 'sample' in granularities and needs_per_var:
        # When per_var collection was done, also build the sample-pooled view
        pass  # handled inside run_granularity via .mean(1)

    all_results = {}
    for g in granularities:
        all_results[g] = run_granularity(g, buckets, args)

    print("\n" + "=" * 90)
    print("LEGEND")
    print("  Δ/σ  = (observed − shuffled mean) / shuffled std.")
    print("  SHARED ⚠      > 4σ above null   → add a decorrelation loss between this pair.")
    print("  some overlap  2σ–4σ             → borderline; ablate both.")
    print("  independent ✓ within ±2σ        → no decor needed for this pair.")
    print("=" * 90)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        meta = {
            'student': args.student_model, 'data': args.data,
            'checkpoint': args.checkpoint, 'flag': args.flag,
            'granularity': args.granularity,
            'arch': {
                'dual_encoder':   args.use_dual_encoder,
                'multi_encoder':  args.multi_encoder,
                'multi_predictor':args.multi_predictor,
                'per_var_teacher':args.per_var_teacher,
                'rendering_methods': args.rendering_methods,
            },
            'sample_counts': {k: int(v.shape[0]) for k, v in buckets.items()},
            'shapes':        {k: list(v.shape)   for k, v in buckets.items()},
            'results': all_results,
        }
        with open(args.out, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
