import torch
import torch.nn.functional as F
import einops

def visionTS_plot (x, norm_const=0.4, pad_left=0, periodicity=1, fp64=False): #add defult like visionTS
    # 1. Normalization
    means = x.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
    x_enc = x - means
    stdev = torch.sqrt(
        torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [bs x 1 x nvars]
    stdev /= norm_const
    x_enc /= stdev
    # Channel Independent
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]

    # 2. Segmentation
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate') # [b n s]
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    return x_2d
