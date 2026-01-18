import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.latent = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        latent = self.latent(x)
        return x + latent
