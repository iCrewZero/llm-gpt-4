import torch, torch.nn as nn, torch.nn.functional as F
from model.rope import RoPE
from kv.paged_kv import append_token

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.h = cfg.n_heads
        self.dh = d // cfg.n_heads
        self.qkv = nn.Linear(d, d*3, False)
        self.o = nn.Linear(d,d,False)
        self.rope = RoPE(self.dh)

    def forward(self, x, state, alloc, page):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3,-1)
        q = q.view(B,T,self.h,self.dh).transpose(1,2)
        k = k.view(B,T,self.h,self.dh).transpose(1,2)
        v = v.view(B,T,self.h,self.dh).transpose(1,2)
        q,k = self.rope.apply(q,k,state.seq_len)

        for _ in range(T):
            append_token(state, alloc, page)

        out = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(out.transpose(1,2).reshape(B,T,C))
