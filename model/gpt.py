import torch.nn as nn
from .block import Block, RMSNorm
from .rope import RoPE

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.rope = RoPE(cfg.n_embd // cfg.n_head)

        self.blocks = nn.ModuleList([
            Block(cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.use_flash)
            for _ in range(cfg.n_layer)
        ])

        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, start_pos=0, kv_cache=None):
        x = self.embed(idx)
        new_cache = []

        for i, blk in enumerate(self.blocks):
            past = None if kv_cache is None else kv_cache[i]
            x, kv = blk(x, self.rope, start_pos, past)
            new_cache.append(kv)

        return self.head(self.norm(x)), new_cache
