"""
Out-of-the-box test: can the ShapeNet-pretrained Point-MAE reconstruct a MASKED
HORIZON of an ETTh1 window, when the window is encoded as a 3D point cloud
    point = (time_index, value, variable_index)  ->  (x, y, z)

NO training. Pure inference with frozen pretrained weights.
We compare two maskings: 'horizon' (mask the latest-in-time patches, the forecast
region) vs 'rand' (the masking Point-MAE was actually trained with) as a ceiling.
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pointmae_model import load_pointmae

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pointmae_pretrain.pth')
ETTH1 = '/dataset/ETT-small/ETTh1.csv'
VARS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
NVAR = len(VARS)
T = 146                      # timesteps per window  (T*7 = 1022 points ~ 1024)
MASK_RATIO = 0.6             # Point-MAE was pretrained at 0.6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

# ----------------- data: standard ETTh1 split, standardize on TRAIN -----------------
df = pd.read_csv(ETTH1)
data = df[VARS].values.astype(np.float32)        # (17420, 7)
n_train = 12 * 30 * 24                            # 8640
mu = data[:n_train].mean(0)
sd = data[:n_train].std(0)
data_std = (data - mu) / sd                       # per-variable standardized
test_start = 12 * 30 * 24 + 4 * 30 * 24           # 11520  (standard test region)
test_end = 12 * 30 * 24 + 8 * 30 * 24             # 14400
test = data_std[test_start:test_end]              # (2880, 7)


def build_cloud(win):
    """win: (T,7) standardized. Returns cloud (N,3) in [-1,1]^3 per-axis, plus inverse info."""
    tt = np.repeat(np.arange(T), NVAR)                       # time idx
    vv = np.tile(np.arange(NVAR), T)                         # var idx
    yy = win.reshape(-1)                                     # standardized value
    raw = np.stack([tt, yy, vv], 1).astype(np.float32)       # (N,3)
    # per-axis min-max -> [-1,1]
    lo = raw.min(0); hi = raw.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    cloud = 2 * (raw - lo) / span - 1
    return cloud, (lo, span)


def axes_back(points, axinfo):
    """inverse of per-axis min-max; returns (time_idx, value_std, var_idx float)."""
    lo, span = axinfo
    raw = (points + 1) / 2 * span + lo
    return raw[:, 0], raw[:, 1], raw[:, 2]


def chamfer(a, b):
    """mean nearest-neighbour distance both directions (numpy, small N)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def run_window(model, win, mask_mode, axinfo, cloud):
    model.MAE_encoder.mask_mode = mask_mode
    pts = torch.from_numpy(cloud).unsqueeze(0)
    out = model.reconstruct(pts)
    reb = out['rebuild_masked'].numpy()
    gt = out['gt_masked'].numpy()
    vis = out['vis'].numpy()
    ch = chamfer(reb, gt)
    return reb, gt, vis, ch


def plot_window(win, axinfo, packs, tag):
    # packs: dict mode -> (reb, gt, vis, ch)
    fig = plt.figure(figsize=(16, 9))
    # ---- 3D clouds (one per mode) ----
    for j, mode in enumerate(packs):
        reb, gt, vis, ch = packs[mode]
        ax = fig.add_subplot(2, len(packs), j + 1, projection='3d')
        ax.scatter(vis[:, 0], vis[:, 2], vis[:, 1], s=4, c='0.6', label='visible ctx')
        ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=6, c='tab:green', label='GT masked')
        ax.scatter(reb[:, 0], reb[:, 2], reb[:, 1], s=6, c='tab:red', label='reconstructed')
        ax.set_title(f'{mode} mask  |  Chamfer={ch:.3f}')
        ax.set_xlabel('time'); ax.set_ylabel('var'); ax.set_zlabel('value')
        if j == 0: ax.legend(loc='upper left', fontsize=7)
    # ---- per-variable value-vs-time for the HORIZON mode ----
    reb, gt, vis, ch = packs['horizon']
    rt, rv, rvar = axes_back(reb, axinfo)
    rvar_i = np.clip(np.round(rvar).astype(int), 0, NVAR - 1)
    gax = fig.add_subplot(2, 1, 2)
    gax.axis('off')
    sub = [fig.add_axes([0.04 + 0.137 * k, 0.06, 0.12, 0.33]) for k in range(NVAR)]
    # horizon boundary (in time index): patches with largest time were masked;
    # approximate boundary = min reconstructed time
    for k in range(NVAR):
        a = sub[k]
        a.plot(np.arange(T), win[:, k], c='0.3', lw=1.0, label='GT')
        sel = rvar_i == k
        a.scatter(rt[sel], rv[sel], s=8, c='tab:red', label='recon', zorder=3)
        a.set_title(VARS[k], fontsize=9)
        a.tick_params(labelsize=6)
        if k == 0: a.legend(fontsize=6)
    fig.suptitle(f'Point-MAE (ShapeNet-pretrained, NO training) on ETTh1 window — {tag}\n'
                 f"horizon Chamfer={packs['horizon'][3]:.3f}  vs  rand Chamfer={packs['rand'][3]:.3f}",
                 fontsize=12)
    p = f'{OUT}/{tag}.png'
    fig.savefig(p, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return p


def main():
    model = load_pointmae(CKPT, mask_mode='horizon', mask_ratio=MASK_RATIO)
    np.random.seed(0)
    offsets = [0, 900, 1800]      # three windows across the test set
    rows = []
    for oi, off in enumerate(offsets):
        win = test[off:off + T]               # (T,7)
        cloud, axinfo = build_cloud(win)
        packs = {}
        for mode in ['horizon', 'rand']:
            np.random.seed(0)                 # fix rand mask for reproducibility
            packs[mode] = run_window(model, win, mode, axinfo, cloud)
        p = plot_window(win, axinfo, packs, tag=f'window{oi}_off{off}')
        rows.append((off, packs['horizon'][3], packs['rand'][3], p))
        print(f'window {oi} off={off}: horizon Chamfer={packs["horizon"][3]:.4f}  '
              f'rand Chamfer={packs["rand"][3]:.4f}  -> {p}')
    print('\nSummary (lower Chamfer = better reconstruction):')
    print(f'{"offset":>8} {"horizon":>10} {"rand":>10}')
    for off, h, r, _ in rows:
        print(f'{off:>8} {h:>10.4f} {r:>10.4f}')


if __name__ == '__main__':
    main()
