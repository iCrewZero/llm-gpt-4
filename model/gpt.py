import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks import Block
from model.rmsnorm import RMSNorm
from model.rope import RotaryEmbedding

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = RMSNorm(cfg.n_embd)
        self.lm = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm.weight
        self.rope = RotaryEmbedding(cfg.n_embd // cfg.n_head, factor=cfg.rope_factor)

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        x = self.wte(idx)
        rope = self.rope.get(T, x.device, x.dtype)

        for i, blk in enumerate(self.blocks):
            blk_cache = cache[i] if cache else None
            x = blk(x, rope, blk_cache)

        logits = self.lm(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss
