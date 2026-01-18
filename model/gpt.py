import torch
import torch.nn as nn
from .block import Block
from .config import ModelConfig
from .rmsnorm import RMSNorm
from .verifier import VerifierHead

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.dim)

        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.verifier = VerifierHead(cfg.dim)

    def forward(self, input_ids):
        B,T = input_ids.shape
        x = self.embed(input_ids)
        for i,block in enumerate(self.blocks):
            x = block(x, pos=0)
        x = self.norm(x)
        logits = self.lm_head(x)
        value = self.verifier(x)
        return logits, value
