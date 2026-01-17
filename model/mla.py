import torch
import torch.nn as nn

class MLA(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(n_head, dim//n_head))
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B,T,C = x.shape
        h = x.view(B,T,-1,self.latent.size(-1))
        scores = (h * self.latent).sum(-1, keepdim=True)
        h = h * scores
        return self.proj(h.reshape(B,T,C))
