import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, use_flash=True):
        super().__init__()
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_flash = use_flash

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope, start_pos=0):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        q, k = rope(q, k, start_pos)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_flash and torch.cuda.is_available():
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            att = att.masked_fill(~mask, float("-inf"))
            att = att.softmax(dim=-1)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
