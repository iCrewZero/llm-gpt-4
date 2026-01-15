import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.paged_kv import KVState, KVAllocator, append_token_state, write_kv_token, gather_kv
from inference.attention import PagedAttention

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn = PagedAttention(n_heads, self.head_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, state: KVState, allocator: KVAllocator):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        for t in range(T):
            append_token_state(state, allocator, allocator.page_size)
            write_kv_token(allocator.K, allocator.V, state, k[:, t], v[:, t], allocator.page_size)

        K, V = gather_kv(allocator.K, allocator.V, state, state.seqlen, allocator.page_size)
        K = K.repeat_interleave(self.n_heads // allocator.n_kv_head, dim=1)
        V = V.repeat_interleave(self.n_heads // allocator.n_kv_head, dim=1)

        out = self.attn(q, K, V)
        out = out.transpose(1, 2).reshape(B, T, C)
        return x + self.proj(out) + self.ff(x)
