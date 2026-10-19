"""
Leak-free horizon reconstruction with the frozen ShapeNet-pretrained Point-MAE.

Fix vs experiment.py: whole future timesteps are hidden. The encoder only receives
points from the context timesteps t < T_CTX (so FPS centers come from context only),
and the decoder is queried at future (time, var) centers whose VALUE coordinate is a
non-leaking placeholder (last observed value of that variable). The value axis is
min-max scaled with CONTEXT statistics only. Metrics are computed on the true future
points and compared with persistence / context-mean baselines. No training.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import test, VARS, NVAR, T, CKPT, OUT
from pointmae_model import load_pointmae

T_CTX = 88                 # context timesteps; horizon = T - T_CTX = 58 (~40%)
N_BINS = 5                 # future time bins per variable -> N_BINS*NVAR query centers
PLACEHOLDER = 'last'       # value coord of query centers: 'last' observed value or 'mean'


def make_inputs(win, placeholder=PLACEHOLDER):
    ctx = win[:T_CTX]
    # time axis: full window range is known a priori; value axis: CONTEXT stats only
    lo_y, hi_y = ctx.min(), ctx.max()
    sy = max(hi_y - lo_y, 1e-6)
    tx = lambda t: 2 * np.asarray(t, np.float32) / (T - 1) - 1
    ty = lambda y: 2 * (np.asarray(y, np.float32) - lo_y) / sy - 1
    tz = lambda v: 2 * np.asarray(v, np.float32) / (NVAR - 1) - 1

    tt = np.repeat(np.arange(T_CTX), NVAR); vv = np.tile(np.arange(NVAR), T_CTX)
    ctx_cloud = np.stack([tx(tt), ty(ctx.reshape(-1)), tz(vv)], 1)

    fut = win[T_CTX:]
    ft = np.repeat(np.arange(T_CTX, T), NVAR); fv = np.tile(np.arange(NVAR), T - T_CTX)
    fut_cloud = np.stack([tx(ft), ty(fut.reshape(-1)), tz(fv)], 1)   # ground truth only

    bins = np.linspace(T_CTX, T - 1, 2 * N_BINS + 1)[1::2]            # bin centres in time
    qt = np.repeat(bins, NVAR); qv = np.tile(np.arange(NVAR), N_BINS)
    base = ctx[-1] if placeholder == 'last' else ctx.mean(0)
    qy = base[qv]
    q = np.stack([tx(qt), ty(qy), tz(qv)], 1)
    return ctx_cloud, fut_cloud, q.astype(np.float32), (lo_y, sy)


def decode_values(rebuilt, fut, inv):
    """Turn reconstructed points into a per-variable forecast at each future timestep
    (kernel average over points assigned to that variable), in standardized units."""
    lo_y, sy = inv
    t = (rebuilt[:, 0] + 1) / 2 * (T - 1)
    y = (rebuilt[:, 1] + 1) / 2 * sy + lo_y
    v = np.clip(np.round((rebuilt[:, 2] + 1) / 2 * (NVAR - 1)).astype(int), 0, NVAR - 1)
    pred = np.full_like(fut, np.nan)
    for k in range(NVAR):
        s = v == k
        if not s.any():
            continue
        for i, tq in enumerate(range(T_CTX, T)):
            w = np.exp(-0.5 * ((t[s] - tq) / 6.0) ** 2)
            pred[i, k] = (w * y[s]).sum() / max(w.sum(), 1e-9)
    return pred


def chamfer(a, b):
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def main():
    model = load_pointmae(CKPT, mask_ratio=0.6)
    offsets = list(range(0, 2880 - T, 120))
    res = []
    for off in offsets:
        win = test[off:off + T]
        ctx_cloud, fut_cloud, q, inv = make_inputs(win)
        with torch.no_grad():
            reb, vis = model.forecast(torch.from_numpy(ctx_cloud)[None], torch.from_numpy(q)[None])
        reb = reb.numpy()
        fut = win[T_CTX:]
        pred = decode_values(reb, fut, inv)
        m_model = np.nanmean((pred - fut) ** 2)
        m_last = np.mean((win[T_CTX - 1] - fut) ** 2)
        m_mean = np.mean((win[:T_CTX].mean(0) - fut) ** 2)
        rng = np.random.RandomState(off)  # sanity: same-shape random cloud in the box
        rand = rng.uniform(-1, 1, reb.shape).astype(np.float32)
        res.append((off, m_model, m_last, m_mean, chamfer(reb, fut_cloud), chamfer(rand, fut_cloud)))
        if off in (0, 1200, 2400):
            plot(win, pred, reb, inv, off)
    r = np.array(res)
    print(f'{"offset":>7} {"MSE model":>10} {"MSE last":>10} {"MSE ctxmean":>12} {"Chf model":>10} {"Chf random":>11}')
    for row in r:
        print(f'{int(row[0]):>7} ' + ' '.join(f'{x:>10.4f}' for x in row[1:]))
    print(f'\nmean over {len(r)} windows: MSE model={r[:,1].mean():.4f}  persistence={r[:,2].mean():.4f}  '
          f'ctx-mean={r[:,3].mean():.4f}  | Chamfer model={r[:,4].mean():.4f} random={r[:,5].mean():.4f}')
    print(f'model beats persistence in {(r[:,1] < r[:,2]).mean():.0%} of windows')


def plot(win, pred, reb, inv, off):
    lo_y, sy = inv
    t = (reb[:, 0] + 1) / 2 * (T - 1); y = (reb[:, 1] + 1) / 2 * sy + lo_y
    v = np.clip(np.round((reb[:, 2] + 1) / 2 * (NVAR - 1)).astype(int), 0, NVAR - 1)
    fig, axs = plt.subplots(1, NVAR, figsize=(20, 3))
    for k, a in enumerate(axs):
        a.plot(np.arange(T), win[:, k], c='0.3', lw=1, label='GT')
        a.axvline(T_CTX - 0.5, c='k', ls=':')
        a.scatter(t[v == k], y[v == k], s=6, c='tab:red', alpha=.6, label='recon pts')
        a.plot(np.arange(T_CTX, T), pred[:, k], c='tab:blue', lw=1.5, label='kernel fcst')
        a.set_title(VARS[k], fontsize=9)
        if k == 0: a.legend(fontsize=6)
    fig.suptitle(f'Leak-free Point-MAE horizon reconstruction (frozen, no training) — offset {off}')
    fig.savefig(f'{OUT}/leakfree_off{off}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
