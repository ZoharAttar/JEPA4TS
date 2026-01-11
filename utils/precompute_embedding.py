import os
import sys
import numpy as np
import torch
import hashlib
from tqdm import tqdm
from transformers import ViTModel
import torch.nn.functional as F
import einops
from pyts.image import RecurrencePlot

sys.path.append('/Users/zattar/Documents/University/Master/JEPA4TS')

from data_provider.data_factory import data_provider
import argparse

# ============================================
# CONFIGURATION
# ============================================

CACHE_DIR = "./dataset/ETT-small/dino_embeddings"
IMAGE_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================
# SAME transform_RP_batch
# ============================================

def transform_RP_batch(x, image_size=518, device=None):
    if device is None:
        device = x.device
    batch_size, seq_len, nvars = x.shape
    rp = RecurrencePlot()
    x_cpu = x.detach().cpu()
    x_flat = x_cpu.permute(0, 2, 1).reshape(-1, seq_len).numpy()
    rp_images = rp.fit_transform(x_flat)
    rp_images = torch.tensor(rp_images, dtype=torch.float32)
    rp_images = einops.rearrange(rp_images, '(b n) h w -> b n h w', b=batch_size)
    rp_images = F.interpolate(rp_images, size=(image_size, image_size), mode='bilinear', align_corners=False)
    rp_images = rp_images.mean(dim=1, keepdim=True)
    rp_images = rp_images.repeat(1, 3, 1, 1)
    return rp_images.to(device)

def get_ts_hash(ts_array):
    return hashlib.md5(ts_array.astype(np.float32).tobytes()).hexdigest()[:16]

# ============================================
# MAIN
# ============================================

def precompute_from_dataloader():
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    
    # MUST MATCH YOUR TRAINING CONFIG
    args.data = 'ETTh1'
    args.root_path = './dataset/ETT-small/'
    args.data_path = 'ETTh1.csv'
    args.features = 'M'
    args.target = 'OT'
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    args.embed = 'timeF'
    args.freq = 'h'
    args.batch_size = 1  # ← BATCH SIZE = 1 for exact same processing!
    args.num_workers = 0
    args.seasonal_patterns = 'Monthly'
    args.augmentation_ratio = 0
    args.task_name = 'long_term_forecast'
    
    print(f"Device: {DEVICE}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Batch size: 1 (exact same as single-sample processing)")
    
    # Load DINO
    print("\nLoading DINO...")
    dino = ViTModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    dino.eval()
    
    existing = set(f.replace('.npy', '') for f in os.listdir(CACHE_DIR) if f.endswith('.npy'))
    print(f"Existing cached: {len(existing)}")
    
    for flag in ['train', 'val', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {flag.upper()} split...")
        print(f"{'='*50}")
        
        data_set, data_loader = data_provider(args, flag)
        
        generated = 0
        skipped = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(data_loader):
                # batch_x is [1, seq_len, nvars] since batch_size=1
                batch_x = batch_x.float().to(DEVICE)
                sample_np = batch_x[0].cpu().numpy().astype(np.float32)
                ts_hash = get_ts_hash(sample_np)
                
                if ts_hash in existing:
                    skipped += 1
                    continue
                
                cache_path = os.path.join(CACHE_DIR, f"{ts_hash}.npy")
                
                if not os.path.exists(cache_path):
                    # Process single sample [1, seq_len, nvars]
                    x_image = transform_RP_batch(batch_x, image_size=IMAGE_SIZE, device=DEVICE)
                    outputs = dino(pixel_values=x_image)
                    embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
                    np.save(cache_path, embedding.cpu().numpy().squeeze(0))
                    generated += 1
                
                existing.add(ts_hash)
        
        print(f"  Generated: {generated}, Skipped: {skipped}")
    
    total_cached = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.npy')])
    print(f"\n✅ Done! Total embeddings: {total_cached}")

if __name__ == "__main__":
    precompute_from_dataloader()
