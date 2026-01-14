import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        n_kv_head,
        rope,
        use_flash=False,
    ):
        super().__init__()

        assert dim % n_head == 0
        assert n_head % n_kv_head == 0

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.kv_repeat = n_head // n_kv_head
        self.use_flash = use_flash
        self.rope = rope

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, start_pos=0):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        q, k = self.rope.apply_rotary(q, k, start_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        if self.use_flash and torch.cuda.is_available():
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            att = att.softmax(dim=-1)
            out = att @ v

        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, C)

        return self.o_proj(out)
