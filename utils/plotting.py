import torch
import torch.nn.functional as F
import einops

def visionTS_plot(x, norm_const=0.4, periodicity=1, image_size=224
                  , fp64=False):
    """
    Convert time series to image for Vision Transformer input
    
    Args:
        x: Time series [batch, seq_len, nvars]
        norm_const: Normalization constant (default: 0.4)
        periodicity: Periodicity for segmentation (default: 1)
        image_size: Output image size (default: 224)
        fp64: Use float64 for variance (default: False)
    
    Returns:
        image: image for ViT
    """
    batch_size, seq_len, nvars = x.shape
    
    # 1. Normalization
    means = x.mean(1, keepdim=True).detach()  # [bs, 1, nvars]
    x_enc = x - means
    stdev = torch.sqrt(
        torch.var(x_enc.to(torch.float64) if fp64 else x_enc, 
                  dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev /= norm_const
    x_enc /= stdev
    
    # Channel Independent: [bs, seq_len, nvars] -> [bs, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')
    
    # 2. Calculate padding
    pad_left = 0
    if seq_len % periodicity != 0:
        pad_left = periodicity - (seq_len % periodicity)
    
    # 3. Segmentation
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> b n f p', f=periodicity)
    
    # 4. Resize to image_size x image_size
    # Interpolate [batch, nvars, f, p] to [batch, nvars, image_size, image_size]
    x_resized = F.interpolate(
        x_2d, 
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False
    )
    x_resized = x_resized.mean(dim=1).unsqueeze(1)

    input_image = einops.repeat(x_resized, 'b 1 h w -> b c h w', c=3)
    
    return input_image
