import torch.nn as nn
from .attention import CausalSelfAttention

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=CausalSelfAttention()
        self.ff=nn.Sequential(nn.Linear(2048,8192),nn.GELU(),nn.Linear(8192,2048))
        self.ln1=nn.LayerNorm(2048)
        self.ln2=nn.LayerNorm(2048)
    def forward(self,x,state,K_pool,V_pool,allocator):
        x=x+self.attn(self.ln1(x),state,K_pool,V_pool,allocator)
        x=x+self.ff(self.ln2(x))
        return x
