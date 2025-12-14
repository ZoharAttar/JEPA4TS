# import torch
# import torch.nn.functional as F
# import einops

# def visionTS_plot (x, norm_const=0.4, pad_left=0, periodicity=1, fp64=False): #add defult like visionTS
#     # 1. Normalization
#     means = x.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
#     x_enc = x - means
#     stdev = torch.sqrt(
#         torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [bs x 1 x nvars]
#     stdev /= norm_const
#     x_enc /= stdev
#     # Channel Independent
#     x_enc = einops.rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]

#     # 2. Segmentation
#     x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate') # [b n s]
#     x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
#     return x_2d


import torch
import torch.nn.functional as F
import einops

def visionTS_plot(x, norm_const=0.4, periodicity=1, image_size=224, 
                  clip_range=(-5, 5), fp64=False):
    """
    Convert time series to image for Vision Transformer input
    
    Args:
        x: Time series [batch, seq_len, nvars]
        norm_const: Normalization constant (default: 0.4)
        periodicity: Periodicity for segmentation (default: 1)
        image_size: Output image size (default: 224)
        clip_range: Clipping range (default: (-5, 5))
        fp64: Use float64 for variance (default: False)
    
    Returns:
        image: [batch, 3, 224, 224] RGB image for ViT
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
    
    # 5. Average across variables to create grayscale image
    x_gray = x_resized.mean(dim=1, keepdim=True)  # [batch, 1, 224, 224]
    
    # 6. Clip values
    x_gray = torch.clamp(x_gray, clip_range[0], clip_range[1])
    
    # 7. Convert to 3-channel RGB (repeat grayscale)
    x_rgb = x_gray.repeat(1, 3, 1, 1)  # [batch, 3, 224, 224]
    
    return x_rgb
