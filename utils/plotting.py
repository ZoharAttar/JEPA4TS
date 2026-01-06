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


##############################################################################
##############################################################################
##############################################################################


def visionTS_plot_v2(x, norm_const=0.4, periodicity=1, image_size=224, fp64=False):
    """
    Convert time series to image for Vision Transformer input
    """
    batch_size, seq_len, nvars = x.shape

    # Normalize per-variable
    means = x.mean(1, keepdim=True).detach()
    x_enc = x - means
    stdev = torch.sqrt(
        torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
    )
    stdev /= norm_const
    x_enc /= stdev

    # Rearrange to [batch, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # Padding if needed
    pad_left = (periodicity - (seq_len % periodicity)) % periodicity
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')

    # Segment to 2D [batch, nvars, f, p]
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> b n f p', f=periodicity)

    # Resize to image
    x_resized = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # If nvars > 3, map to 3 channels (mean or simple linear)
    if x_resized.shape[1] == 3:
        input_image = x_resized
    elif x_resized.shape[1] > 3:
        input_image = x_resized[:, :3]  # take first 3 variables
    else:  # nvars < 3
        input_image = einops.repeat(x_resized, 'b c h w -> b 3 h w', c=x_resized.shape[1])

    return input_image



##############################################################################
##############################################################################
##############################################################################



import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

def visionTS_plot_heatmap(x, image_size=224):
    """
    x: [batch, seq_len, nvars]
    Returns: [batch, 3, image_size, image_size] images
    """
    images = []
    for ts in x:
        # Create heatmap (time on y, features on x)
        hm = ts.T.unsqueeze(0)  # [nvars, seq_len] -> [1, nvars, seq_len]
        hm = torch.nn.functional.interpolate(
            hm.unsqueeze(0), size=(image_size, image_size), mode='bilinear'
        ).squeeze(0)
        # Normalize
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-5)
        # Repeat channels
        img = hm.repeat(3, 1, 1)
        images.append(img)
    return torch.stack(images)




##############################################################################
##############################################################################
##############################################################################



import torch
import torch.nn.functional as F
from pyts.image import GramianAngularField
import einops

def transform_GAF_batch(x, image_size=224, method='summation', fp64=False):
    """
    Convert batch of time series to GAF images for ViT input.

    Args:
        x: Time series tensor [batch, seq_len, nvars] (torch.Tensor)
        image_size: Desired output image size for ViT (int)
        method: 'summation' or 'difference' for GAF
        fp64: Use float64 for variance (optional)

    Returns:
        torch.Tensor: GAF images [batch, 3, image_size, image_size]
    """
    batch_size, seq_len, nvars = x.shape

    # Ensure GAF image_size <= seq_len
    gaf_image_size = min(seq_len, image_size)

    # Normalize each series
    means = x.mean(1, keepdim=True)
    x_enc = x - means
    stdev = torch.sqrt(
        torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
    )
    x_enc = x_enc / stdev

    # Move to CPU for pyts
    x_cpu = x_enc.detach().cpu().numpy()

    # Merge batch and variables for pyts GAF
    x_merged = x_cpu.reshape(-1, seq_len)  # [batch*nvars, seq_len]

    # Create GAF transformer
    gaf = GramianAngularField(image_size=gaf_image_size, method=method)
    gaf_images = gaf.fit_transform(x_merged)  # [batch*nvars, H, W]

    # Convert back to torch
    gaf_images = torch.tensor(gaf_images, dtype=x_enc.dtype)  # [B*nvars, H, W]
    gaf_images = gaf_images.unsqueeze(1)  # [B*nvars, 1, H, W]

    # Resize to target ViT image_size
    if gaf_image_size != image_size:
        gaf_images = F.interpolate(gaf_images, size=(image_size, image_size),
                                   mode='bilinear', align_corners=False)

    # Reshape back to [batch, nvars, H, W] and take mean over vars for RGB
    gaf_images = gaf_images.reshape(batch_size, nvars, image_size, image_size)
    gaf_images = gaf_images.mean(dim=1, keepdim=True)  # [batch, 1, H, W]

    # Repeat channels for RGB
    gaf_images = einops.repeat(gaf_images, 'b 1 h w -> b 3 h w')

    return gaf_images




##############################################################################
##############################################################################
##############################################################################



import torch
import torch.nn.functional as F
from pyts.image import RecurrencePlot
import einops

def transform_RP_batch(x, image_size=224, device=None):
    """
    Convert batch of time series to recurrence plot (RP) images,
    in the same style as transform_GAF_batch.
    
    Args:
        x: Tensor [batch, seq_len, nvars]
        image_size: Output image size (int)
        device: Torch device. If None, use x.device.
    
    Returns:
        Tensor [batch, 3, image_size, image_size]
    """
    if device is None:
        device = x.device

    batch_size, seq_len, nvars = x.shape
    rp = RecurrencePlot()

    # Move to CPU for pyts processing
    x_cpu = x.detach().cpu()
    
    # Flatten batch and variables: [batch*nvars, seq_len]
    x_flat = x_cpu.permute(0, 2, 1).reshape(-1, seq_len).numpy()
    
    # Apply Recurrence Plot transform
    rp_images = rp.fit_transform(x_flat)  # [batch*nvars, seq_len, seq_len]
    
    # Convert back to tensor
    rp_images = torch.tensor(rp_images, dtype=torch.float32)  # [batch*nvars, seq_len, seq_len]
    
    # Reshape to [batch, nvars, H, W]
    rp_images = einops.rearrange(rp_images, '(b n) h w -> b n h w', b=batch_size)
    
    # Resize to image_size x image_size
    rp_images = F.interpolate(rp_images, size=(image_size, image_size), mode='bilinear', align_corners=False)
    
    # Combine variables into 1 channel and repeat to 3 channels
    rp_images = rp_images.mean(dim=1, keepdim=True)  # [batch, 1, H, W]
    rp_images = rp_images.repeat(1, 3, 1, 1)  # [batch, 3, H, W]
    
    return rp_images.to(device)

