import torch.nn as nn
from .rmsnorm import RMSNorm
from .attention import CausalSelfAttention
from .mla import MultiHeadLatentAttention
from .moe import MoE

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(
            cfg.dim, cfg.n_head, cfg.n_kv_head, cfg.rope_factor
        )
        self.norm2 = RMSNorm(cfg.dim)
        self.mla = MultiHeadLatentAttention(cfg.dim, cfg.n_head)
        self.moe = MoE(cfg.dim, cfg.moe_experts, cfg.moe_topk)

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mla(self.norm2(x))
        x = x + self.moe(x)
        return x
