import torch.nn as nn

class MTPHead(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.proj = nn.Linear(dim, vocab)

    def forward(self, h):
        return self.proj(h)
