import torch, torch.nn as nn

class MoE(nn.Module):
    def __init__(self, dim, experts, top_k):
        super().__init__()
        self.router = nn.Linear(dim, experts, False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim,4*dim), nn.GELU(), nn.Linear(4*dim,dim))
             for _ in range(experts)]
        )
        self.top_k = top_k

    def forward(self, x):
        scores = self.router(x)
        top = scores.topk(self.top_k, dim=-1).indices
        out = 0
        for i in range(self.top_k):
            out += self.experts[top[...,i]](x)
        return out / self.top_k
