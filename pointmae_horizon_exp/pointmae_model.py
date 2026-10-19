"""
Self-contained, pure-PyTorch Point-MAE (pretrain model) for CPU/MPS inference.
Architecture copied verbatim from Pang-Yatian/Point-MAE (models/Point_MAE.py),
with these changes so it runs with NO CUDA extensions:
  * Group: FPS + kNN reimplemented in pure torch (no pointnet2_ops / knn_cuda).
  * Chamfer loss removed (inference / visualization only).
  * logger / registry / timm-registry imports stripped.
  * Masking extended: mask_mode in {'rand','block','horizon'}; 'horizon' masks the
    patches whose center has the LARGEST time coordinate (axis 0) = the future.
The state_dict keys are unchanged, so the official pretrain.pth loads cleanly.
"""
import numpy as np
import torch
import torch.nn as nn
try:
    from timm.models.layers import DropPath, trunc_normal_
except ModuleNotFoundError:
    # minimal stand-ins: DropPath is identity in eval; trunc_normal_ only inits
    # weights that the loaded checkpoint overwrites anyway.
    def trunc_normal_(t, std=0.02, **kw):
        with torch.no_grad():
            return t.normal_(0, std)

    class DropPath(nn.Module):
        def __init__(self, p=0.):
            super().__init__()
            self.p = p

        def forward(self, x):
            return x  # eval-time no-op


# ----------------------------- pure-torch FPS + kNN -----------------------------
def index_points(points, idx):
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    rep = [1] + list(idx.shape[1:])
    bi = torch.arange(B, device=points.device).view(view_shape).repeat(rep)
    return points[bi, idx, :]


def fps(xyz, npoint):
    """Farthest point sampling. xyz: (B,N,3) -> centers (B,npoint,3)."""
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)  # deterministic start
    bi = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[bi, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return index_points(xyz, centroids)


def knn_idx(xyz, center, k):
    """xyz:(B,N,3) center:(B,G,3) -> idx:(B,G,k) nearest neighbours in xyz."""
    dist = torch.cdist(center, xyz)  # B G N
    return dist.topk(k, dim=-1, largest=False)[1]


# ----------------------------- model -----------------------------
class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1))

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        B, N, C = xyz.shape
        center = fps(xyz, self.num_group)
        idx = knn_idx(xyz, center, self.group_size)  # B G M
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        idx = (idx + idx_base).view(-1)
        neighborhood = xyz.view(B * N, -1)[idx, :].view(B, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate)
            for i in range(depth)])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    def forward(self, x, pos, return_token_num):
        for block in self.blocks:
            x = block(x + pos)
        return self.head(self.norm(x[:, -return_token_num:]))


class MaskTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        tc = cfg['transformer_config']
        self.mask_ratio = tc['mask_ratio']
        self.trans_dim = tc['trans_dim']
        self.depth = tc['depth']
        self.drop_path_rate = tc['drop_path_rate']
        self.num_heads = tc['num_heads']
        self.encoder_dims = tc['encoder_dims']
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.mask_type = tc['mask_type']
        self.mask_mode = tc.get('mask_mode', self.mask_type)  # 'rand'|'block'|'horizon'
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(embed_dim=self.trans_dim, depth=self.depth,
                                         drop_path_rate=dpr, num_heads=self.num_heads)
        self.norm = nn.LayerNorm(self.trans_dim)

    def _mask_rand(self, center):
        B, G, _ = center.shape
        n_mask = int(self.mask_ratio * G)
        m = np.zeros([B, G])
        for i in range(B):
            row = np.hstack([np.zeros(G - n_mask), np.ones(n_mask)])
            np.random.shuffle(row)
            m[i] = row
        return torch.from_numpy(m).bool().to(center.device)

    def _mask_horizon(self, center):
        """Mask the n_mask patches with the LARGEST time coordinate (axis 0)."""
        B, G, _ = center.shape
        n_mask = int(self.mask_ratio * G)
        t = center[..., 0]  # B G  (time axis)
        order = torch.argsort(t, dim=-1, descending=True)  # latest first
        mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
        mask.scatter_(1, order[:, :n_mask], True)
        return mask

    def forward(self, neighborhood, center, mask_override=None):
        if mask_override is not None:
            bool_masked_pos = mask_override
        elif self.mask_mode == 'horizon':
            bool_masked_pos = self._mask_horizon(center)
        else:
            bool_masked_pos = self._mask_rand(center)
        group_input_tokens = self.encoder(neighborhood)
        B, G, C = group_input_tokens.size()
        x_vis = group_input_tokens[~bool_masked_pos].reshape(B, -1, C)
        masked_center = center[~bool_masked_pos].reshape(B, -1, 3)
        pos = self.pos_embed(masked_center)
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        return x_vis, bool_masked_pos


class Point_MAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        tc = cfg['transformer_config']
        self.trans_dim = tc['trans_dim']
        self.MAE_encoder = MaskTransformer(cfg)
        self.group_size = cfg['group_size']
        self.num_group = cfg['num_group']
        self.drop_path_rate = tc['drop_path_rate']
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))
        self.decoder_depth = tc['decoder_depth']
        self.decoder_num_heads = tc['decoder_num_heads']
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(embed_dim=self.trans_dim, depth=self.decoder_depth,
                                              drop_path_rate=dpr, num_heads=self.decoder_num_heads)
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.increase_dim = nn.Sequential(nn.Conv1d(self.trans_dim, 3 * self.group_size, 1))
        trunc_normal_(self.mask_token, std=.02)

    @torch.no_grad()
    def reconstruct(self, pts, mask_override=None):
        """pts: (B,N,3). Returns dict with everything needed to visualize."""
        neighborhood, center = self.group_divider(pts)
        x_vis, mask = self.MAE_encoder(neighborhood, center, mask_override=mask_override)
        B, _, C = x_vis.shape
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        _, Nmask, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, Nmask, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, Nmask)
        B, M, C = x_rec.shape
        rebuild = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        gt_masked = neighborhood[mask].reshape(B * M, -1, 3)
        # absolute coords (add back centre)
        rebuild_abs = rebuild + center[mask].unsqueeze(1)          # (B*M, group, 3)
        gt_masked_abs = gt_masked + center[mask].unsqueeze(1)
        vis_abs = (neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                   + center[~mask].unsqueeze(1))
        return {
            'mask': mask, 'center': center,
            'rebuild_masked': rebuild_abs.reshape(-1, 3),
            'gt_masked': gt_masked_abs.reshape(-1, 3),
            'vis': vis_abs.reshape(-1, 3),
        }

    @torch.no_grad()
    def forecast(self, ctx_pts, query_centers, n_vis=None):
        """Leak-free horizon reconstruction.
        ctx_pts: (B,Nc,3) points of the CONTEXT timesteps only (future points are never
        seen, so neither tokens nor centers can carry future information).
        query_centers: (B,Q,3) decoder queries at future (time, var); their value
        coordinate must be a non-leaking placeholder chosen by the caller.
        Returns rebuilt absolute points (B*Q*group_size,3) and the visible cloud."""
        n_vis = n_vis or int(round((1 - self.MAE_encoder.mask_ratio) * self.num_group))
        neighborhood, center = Group(n_vis, self.group_size)(ctx_pts)
        tokens = self.MAE_encoder.encoder(neighborhood)
        enc = self.MAE_encoder
        x_vis = enc.norm(enc.blocks(tokens, enc.pos_embed(center)))
        B, Q, _ = query_centers.shape
        x_full = torch.cat([x_vis, self.mask_token.expand(B, Q, -1)], dim=1)
        pos_full = torch.cat([self.decoder_pos_embed(center),
                              self.decoder_pos_embed(query_centers)], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, Q)
        rebuild = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, Q, -1, 3)
        return ((rebuild + query_centers.unsqueeze(2)).reshape(-1, 3),
                (neighborhood + center.unsqueeze(2)).reshape(-1, 3))


DEFAULT_CFG = {
    'group_size': 32, 'num_group': 64, 'loss': 'cdl2',
    'transformer_config': {
        'mask_ratio': 0.6, 'mask_type': 'rand', 'mask_mode': 'rand',
        'trans_dim': 384, 'encoder_dims': 384, 'depth': 12, 'drop_path_rate': 0.1,
        'num_heads': 6, 'decoder_depth': 4, 'decoder_num_heads': 6,
    },
}


def load_pointmae(ckpt_path, mask_mode='rand', mask_ratio=0.6, device='cpu'):
    cfg = {**DEFAULT_CFG}
    cfg['transformer_config'] = {**DEFAULT_CFG['transformer_config'],
                                 'mask_mode': mask_mode, 'mask_ratio': mask_ratio}
    model = Point_MAE(cfg)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['base_model']
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if 'loss_func' not in m]
    unexpected = [u for u in unexpected if 'loss_func' not in u]
    print(f'[load] missing={missing}  unexpected={unexpected}')
    model.eval().to(device)
    return model
