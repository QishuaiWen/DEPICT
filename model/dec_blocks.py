"""
Adapted from https://github.com/Ma-Lab-Berkeley/CRATE
"""

import torch
from torch import nn

from einops import rearrange

from segm.model.utils import ortho


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Parameter((dim ** -0.5) * torch.randn(inner_dim, dim))
        self.step_size = nn.Parameter(torch.randn(1))

    def forward(self, x, query=None):
        proj = self.proj
        proj = ortho(proj, self.heads, self.dim_head, operation=None)  # operation: None, ortho_trans, head_ortho, head_trans
        
        w = rearrange(x @ proj.t(), 'b n (h d) -> b h n d', h=self.heads)
        
        if query is not None:
            query = rearrange(query @ proj.t(), 'b n (h d) -> b h n d', h=self.heads)
            dots = torch.matmul(query, w.transpose(-1, -2)) * self.scale
        else:
            dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = out @ proj
        return self.step_size * out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)))
            
    def forward(self, x, query=None):
        if query is not None:
            for attn in self.layers:
                query = attn(x, query=query) + query
            return query
        else:
            for attn in self.layers:
                x = attn(x) + x
            return x