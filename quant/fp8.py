import torch

def fake_fp8(x):
    scale = x.abs().amax() / 127
    q = (x / scale).round().clamp(-128,127)
    return q * scale
