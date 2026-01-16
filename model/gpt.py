import torch
import torch.nn as nn
from .block import Block

class GPT(nn.Module):
    def __init__(self, vocab_size, dim, layers, heads, max_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)

        self.blocks = nn.ModuleList([
            Block(dim, heads) for _ in range(layers)
        ])

        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        x = self.embed(input_ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)

        x = self.ln(x)
        return self.head(x)
