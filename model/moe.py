import torch
import torch.nn as nn
from .router import TopKRouter

class MoE(nn.Module):
    def __init__(self, dim, n_experts, topk):
        super().__init__()
        self.router = TopKRouter(dim, n_experts, topk)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim*4),
                nn.GELU(),
                nn.Linear(dim*4, dim)
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        probs, idx = self.router(x)
        out = torch.zeros_like(x)

        for i in range(self.router.k):
            expert = idx[..., i]
            weight = probs[..., i].unsqueeze(-1)
            for e_id, expert_net in enumerate(self.experts):
                mask = expert == e_id
                if mask.any():
                    out[mask] += expert_net(x[mask]) * weight[mask]
        return out
