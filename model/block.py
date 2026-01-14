import torch.nn as nn
from .attention import CausalSelfAttention
from .rmsnorm import RMSNorm

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, use_flash=True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, use_flash)

        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, rope, start_pos=0):
        x = x + self.attn(self.norm1(x), rope, start_pos)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x, rope, start_pos=0):
        x = x + self.attn(self.norm1(x), rope, start_pos)
        x = x + self.mlp(self.norm2(x))
        return x
