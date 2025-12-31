import torch.nn as nn
from model.rmsnorm import RMSNorm
from model.attention import Attention
from model.moe import MoE

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.moe = MoE(cfg.n_embd, cfg.moe_experts, cfg.moe_top_k)

    def forward(self, x, rope=None, cache=None):
        x = x + self.attn(self.ln1(x), rope, cache)
        x = x + self.moe(self.ln2(x))
        return x
