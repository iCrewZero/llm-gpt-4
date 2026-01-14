import torch

def yarn_scale(dim, rope_factor, beta_fast=32, beta_slow=1):
    half_dim = dim // 2
    idx = torch.arange(half_dim, dtype=torch.float32)

    low = half_dim / beta_fast
    high = half_dim / beta_slow

    scale = torch.ones_like(idx)

    scale[idx > high] = rope_factor

    mask = (idx >= low) & (idx <= high)
    scale[mask] = 1 + (rope_factor - 1) * (
        (idx[mask] - low) / (high - low)
    )

    return scale
