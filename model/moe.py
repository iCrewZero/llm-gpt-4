import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = int(8 * d / 3)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, d, experts, top_k):
        super().__init__()
        self.router = nn.Linear(d, experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d) for _ in range(experts)])
        self.k = top_k

    def forward(self, x):
        B, T, C = x.shape
        flat = x.view(-1, C)
        probs = F.softmax(self.router(flat), -1)
        topv, topi = probs.topk(self.k, -1)
        out = torch.zeros_like(flat)

        for e, expert in enumerate(self.experts):
            mask = topi == e
            if mask.any():
                idx, slot = mask.nonzero(as_tuple=True)
                out[idx] += expert(flat[idx]) * topv[idx, slot].unsqueeze(-1)

        return out.view(B, T, C)
