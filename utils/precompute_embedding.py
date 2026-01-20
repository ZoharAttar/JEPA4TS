import os
import sys
import numpy as np
import torch
import hashlib
import argparse
import shutil
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F
import einops
from pyts.image import RecurrencePlot

from data_provider.data_factory import data_provider

# ============================================
# DATASET CONFIGS - WITH CORRECT PARAMS
# ============================================

DATASET_CONFIGS = {
    'ETTh1': {
        'data': 'ETTh1',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh1.csv',
        'cache_dir': './dataset/ETT-small/dino_embeddings_ETTh1',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
        'enc_in': 7,
    },
    'ETTh2': {
        'data': 'ETTh2',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh2.csv',
        'cache_dir': './dataset/ETT-small/dino_embeddings_ETTh2',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
        'enc_in': 7,
    },
    'ETTm1': {
        'data': 'ETTm1',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTm1.csv',
        'cache_dir': './dataset/ETT-small/dino_embeddings_ETTm1',
        'freq': 't',
        'features': 'M',
        'target': 'OT',
        'enc_in': 7,
    },
    'ETTm2': {
        'data': 'ETTm2',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTm2.csv',
        'cache_dir': './dataset/ETT-small/dino_embeddings_ETTm2',
        'freq': 't',
        'features': 'M',
        'target': 'OT',
        'enc_in': 7,
    },
    'weather': {
        'data': 'custom',
        'root_path': './dataset/weather/',
        'data_path': 'weather.csv',
        'cache_dir': './dataset/weather/dino_embeddings',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
        'enc_in': 21,
    },
}

IMAGE_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# transform_RP_batch
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
# ZIP AND DOWNLOAD
# ============================================

def zip_and_download(cache_dir, dataset_name):
    """Zip the cache directory and trigger download in Colab"""
    zip_path = f"/content/dino_embeddings_{dataset_name}.zip"
    
    print(f"\n📦 Creating zip file: {zip_path}")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', cache_dir)
    
    try:
        from google.colab import files
        print(f"📥 Downloading {zip_path}...")
        files.download(zip_path)
        print("✅ Download started!")
    except ImportError:
        print(f"⚠️ Not in Colab. Zip saved at: {zip_path}")

# ============================================
# MAIN
# ============================================

def precompute_for_dataset(dataset_name, seq_len=96, label_len=48, pred_len=96, download=True):
    if dataset_name not in DATASET_CONFIGS:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASET_CONFIGS.keys())}")
        return
    
    config = DATASET_CONFIGS[dataset_name]
    cache_dir = config['cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Pre-computing embeddings for: {dataset_name}")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Cache dir: {cache_dir}")
    print(f"Freq: {config['freq']}, Features: {config['enc_in']}")
    
    # Use dataset-specific config
    args = argparse.Namespace(
        data=config['data'],
        root_path=config['root_path'],
        data_path=config['data_path'],
        features=config['features'],
        target=config['target'],
        freq=config['freq'],
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        embed='timeF',
        batch_size=1,
        num_workers=0,
        seasonal_patterns='Monthly',
        augmentation_ratio=0,
        task_name='long_term_forecast',
    )
    
    print("\nLoading DINO...")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    dino.eval()
    
    existing = set(f.replace('.npy', '') for f in os.listdir(cache_dir) if f.endswith('.npy'))
    print(f"Existing cached: {len(existing)}")
    
    for flag in ['train', 'val', 'test']:
        print(f"\nProcessing {flag.upper()}...")
        
        try:
            data_set, data_loader = data_provider(args, flag)
        except Exception as e:
            print(f"  Skipping {flag}: {e}")
            continue
        
        generated = 0
        skipped = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(data_loader, desc=flag):
                batch_x = batch_x.float().to(DEVICE)
                sample_np = batch_x[0].cpu().numpy().astype(np.float32)
                ts_hash = get_ts_hash(sample_np)
                
                if ts_hash in existing:
                    skipped += 1
                    continue
                
                cache_path = os.path.join(cache_dir, f"{ts_hash}.npy")
                
                if not os.path.exists(cache_path):
                    x_image = transform_RP_batch(batch_x, image_size=IMAGE_SIZE, device=DEVICE)
                    outputs = dino(pixel_values=x_image)
                    embedding = outputs.last_hidden_state[:, 0, :]
                    np.save(cache_path, embedding.cpu().numpy().squeeze(0))
                    generated += 1
                
                existing.add(ts_hash)
        
        print(f"  Generated: {generated}, Skipped: {skipped}")
    
    total = len([f for f in os.listdir(cache_dir) if f.endswith('.npy')])
    print(f"\n✅ Done! Total embeddings for {dataset_name}: {total}")
    
    if download:
        zip_and_download(cache_dir, dataset_name)

# ============================================
# CLI
# ============================================

if __name__ == "__main__":
    for ds in ['ETTh2', 'ETTm1', 'ETTm2', 'weather']:
        precompute_for_dataset(ds, seq_len=96, label_len=48, pred_len=96, download=True)
