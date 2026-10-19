"""
Linear / MLP probe for horizon forecasting on top of the FROZEN Point-MAE encoder.

Question answered: once we remove the ambiguous point->value decoding, do the frozen
Point-MAE features actually CONTAIN forecastable information about the ETTh1 horizon?

Design (leak-free):
  * Encoder (Group + MAE_encoder) is FROZEN; only a small head is trained.
  * The encoder sees the CONTEXT timesteps only (t < T_CTX), scaled context-only.
  * Pooled context feature (mean+max over tokens) -> head -> horizon values (H*NVAR).
  * Target & metrics in per-variable standardized units (train-split stats), so the
    numbers are directly comparable to experiment_leakfree.py (model 1.117, ctx-mean 0.632).
  * Baselines: persistence (last ctx value) and context-mean.

Runs on a CPU laptop: features for all windows are extracted once and cached to .npz,
then the head trains in seconds. Re-runs load the cache.

    python3 probe_forecast.py            # linear + MLP probe, prints a table
    python3 probe_forecast.py --fresh    # ignore cache, re-extract features
"""
import os, sys, argparse, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from experiment import data_std, n_train, VARS, NVAR, T, CKPT, OUT
from experiment_leakfree import make_inputs, T_CTX
from pointmae_model import load_pointmae

H = T - T_CTX                       # horizon length (58)
RESIDUAL = True                     # predict change from last ctx value (standard, helps)
STRIDE_TRAIN, STRIDE_EVAL = 4, 12   # window strides; raise to go faster, lower for more data
EPOCHS, LR, WD = 200, 1e-3, 1e-4
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe_feats.npz')
SEED = 0
# use a GPU if present: NVIDIA/CUDA first, then Apple/MPS, else CPU
DEVICE = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available() else 'cpu')

# standard ETTh1 splits (rows), on the already-standardized data_std
SPLITS = {'train': (0, n_train),                       # 0 .. 8640
          'val':   (n_train, n_train + 2880),          # 8640 .. 11520
          'test':  (n_train + 2880, n_train + 5760)}   # 11520 .. 14400


@torch.no_grad()
def encode_context(model, ctx_cloud):
    """Frozen feature: pool the encoder tokens over the context cloud (no masking)."""
    pts = torch.from_numpy(ctx_cloud)[None].float().to(DEVICE)
    nb, center = model.group_divider(pts)              # Group(num_group=64, size=32)
    enc = model.MAE_encoder
    x = enc.norm(enc.blocks(enc.encoder(nb), enc.pos_embed(center)))   # (1, G, 384)
    return torch.cat([x.mean(1), x.max(1).values], -1)[0].cpu().numpy()  # (768,)


def build_split(model, split):
    lo, hi = SPLITS[split]
    arr = data_std[lo:hi]
    stride = STRIDE_TRAIN if split == 'train' else STRIDE_EVAL
    feats, targs, lasts, ctxmeans = [], [], [], []
    for off in range(0, len(arr) - T + 1, stride):
        win = arr[off:off + T]
        ctx_cloud, _, _, _ = make_inputs(win)
        feats.append(encode_context(model, ctx_cloud))
        fut = win[T_CTX:]                               # (H, NVAR) standardized target
        targs.append(fut.reshape(-1))
        lasts.append(win[T_CTX - 1])                    # persistence baseline
        ctxmeans.append(win[:T_CTX].mean(0))            # context-mean baseline
    return (np.asarray(feats, np.float32), np.asarray(targs, np.float32),
            np.asarray(lasts, np.float32), np.asarray(ctxmeans, np.float32))


def get_features(fresh=False):
    if (not fresh) and os.path.exists(CACHE):
        d = np.load(CACHE)
        print(f'[cache] loaded features from {CACHE}')
        return {s: (d[f'{s}_X'], d[f'{s}_Y'], d[f'{s}_last'], d[f'{s}_cm']) for s in SPLITS}
    print(f'[device] using {DEVICE}')
    model = load_pointmae(CKPT, mask_ratio=0.6, device=DEVICE)
    out = {}
    for s in SPLITS:
        print(f'[extract] {s} ...', flush=True)
        out[s] = build_split(model, s)
        print(f'          {s}: {out[s][0].shape[0]} windows')
    np.savez(CACHE, **{f'{s}_X': out[s][0] for s in SPLITS},
             **{f'{s}_Y': out[s][1] for s in SPLITS},
             **{f'{s}_last': out[s][2] for s in SPLITS},
             **{f'{s}_cm': out[s][3] for s in SPLITS})
    print(f'[cache] saved features to {CACHE}')
    return out


class Head(nn.Module):
    def __init__(self, din, dout, hidden=0, dropout=0.1):
        """hidden: 0 = linear; int = one hidden layer; list/tuple = deep MLP."""
        super().__init__()
        dims = [] if not hidden else ([hidden] if isinstance(hidden, int) else list(hidden))
        layers, d = [], din
        for h in dims:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_target(Y, last):
    """Y: (N, H*NVAR) absolute std targets. Returns training target (residual or absolute)."""
    if not RESIDUAL:
        return Y.copy()
    base = np.repeat(last, H, axis=0).reshape(last.shape[0], H * NVAR, order='F')  # tile last over H
    # last is (N,NVAR); we need (N, H*NVAR) matching Y's reshape order (H,NVAR)->flatten
    base = np.tile(last[:, None, :], (1, H, 1)).reshape(last.shape[0], -1)
    return Y - base


def to_pred(raw, last):
    if not RESIDUAL:
        return raw
    base = np.tile(last[:, None, :], (1, H, 1)).reshape(last.shape[0], -1)
    return raw + base


def mse_abs(pred, Y):
    return float(np.mean((pred - Y) ** 2))


def train_head(data, hidden, tag, epochs=EPOCHS):
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr, Ytr, ltr, _ = data['train']
    Xva, Yva, lva, _ = data['val']
    din, dout = Xtr.shape[1], Ytr.shape[1]
    # standardize features (helps optimization)
    fmu, fsd = Xtr.mean(0), Xtr.std(0) + 1e-6
    nz = lambda X: (X - fmu) / fsd
    Ttr = make_target(Ytr, ltr)
    tmu, tsd = Ttr.mean(0), Ttr.std(0) + 1e-6          # standardize regression target
    model = Head(din, dout, hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()
    Xt = torch.tensor(nz(Xtr)).to(DEVICE); Yt = torch.tensor((Ttr - tmu) / tsd).to(DEVICE)
    Xv = torch.tensor(nz(Xva)).to(DEVICE)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xt), Yt); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                raw = model(Xv).cpu().numpy() * tsd + tmu
            predv = to_pred(raw, lva)
            m = mse_abs(predv, Yva)
            if m < best:
                best, best_state = m, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    # evaluate on test
    Xte, Yte, lte, _ = data['test']
    with torch.no_grad():
        raw = model(torch.tensor(nz(Xte)).to(DEVICE)).cpu().numpy() * tsd + tmu
    predte = to_pred(raw, lte)
    return {'tag': tag, 'val_mse': best, 'test_mse': mse_abs(predte, Yte),
            'pred': predte, 'Y': Yte, 'last': lte}


def baselines(data, split='test'):
    _, Y, last, cm = data[split]
    pers = np.tile(last[:, None, :], (1, H, 1)).reshape(last.shape[0], -1)
    ctxm = np.tile(cm[:, None, :], (1, H, 1)).reshape(cm.shape[0], -1)
    return mse_abs(pers, Y), mse_abs(ctxm, Y)


def plot_examples(res, idxs=(0, 10, 20)):
    pred, Y = res['pred'], res['Y']
    for ex in idxs:
        if ex >= len(pred): continue
        p = pred[ex].reshape(H, NVAR); y = Y[ex].reshape(H, NVAR)
        fig, axs = plt.subplots(1, NVAR, figsize=(20, 3))
        for k, a in enumerate(axs):
            a.plot(range(H), y[:, k], c='0.3', lw=1.2, label='GT horizon')
            a.plot(range(H), p[:, k], c='tab:red', lw=1.2, label='probe')
            a.axhline(res['last'][ex, k], c='tab:blue', ls=':', lw=1, label='persistence')
            a.set_title(VARS[k], fontsize=9)
            if k == 0: a.legend(fontsize=6)
        fig.suptitle(f"{res['tag']} probe — test window {ex}  (test MSE {res['test_mse']:.3f})")
        fig.savefig(f"{OUT}/probe_{res['tag']}_ex{ex}.png", dpi=100, bbox_inches='tight')
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', action='store_true', help='ignore feature cache')
    args = ap.parse_args()
    data = get_features(fresh=args.fresh)
    # increasing head capacity: if a bigger/deeper head still can't beat
    # context-mean, the limit is the frozen features, not the read-out.
    configs = [
        ('linear',   0,            EPOCHS),
        ('mlp256',   256,          EPOCHS),
        ('mlp1024',  1024,         400),
        ('mlp2x512', [512, 512],   400),
    ]
    runs = [train_head(data, hidden=h, tag=t, epochs=e) for (t, h, e) in configs]
    pers, ctxm = baselines(data, 'test')
    for r in runs:
        plot_examples(r)

    print('\n==== horizon forecast, standardized MSE on ETTh1 TEST '
          f'(H={H}, {data["test"][0].shape[0]} windows) ====')
    print(f'{"method":>16} {"test MSE":>10} {"val MSE":>10}')
    for r in runs:
        print(f'{r["tag"]:>16} {r["test_mse"]:>10.4f} {r["val_mse"]:>10.4f}')
    print(f'{"persistence":>16} {pers:>10.4f} {"-":>10}')
    print(f'{"context-mean":>16} {ctxm:>10.4f} {"-":>10}')
    print(f'\n(for reference, frozen point-decoder leak-free run: ~1.117;  '
          f'RESIDUAL={RESIDUAL})')
    best = min(runs, key=lambda r: r['test_mse'])
    verdict = 'BEATS' if best['test_mse'] < ctxm else 'does NOT beat'
    print(f'best probe ({best["tag"]}, test {best["test_mse"]:.4f}) {verdict} '
          f'the context-mean baseline ({ctxm:.4f}).')
    print(f'example plots -> {OUT}/probe_*_ex*.png')


if __name__ == '__main__':
    main()
