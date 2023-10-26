from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from typing import Optional

class Prefix(nn.Module):
    def __init__(
        self,
        length: int = 10,
        dim: int = 768,
        position: int = 1,
        compensatory: bool = True,
    ):
        super().__init__()

        self.compensatory = compensatory

        args = (length, dim, position, False)
        self.key = Prompt(*args)
        self.val = Prompt(*args)

    def forward(self, key: torch.Tensor, val: torch.Tensor):
        return self.key(key), self.val(val)

    def compensate(self, attn):
        if not self.compensatory:
            return attn

        position, length = self.key.position, self.key.length
        s, t = position, position + length
        lamb = attn[..., s:t].sum(dim=-1, keepdim=True)
        attn1 = attn[..., :s]
        attn2 = attn[..., s:t] / lamb.clamp(min=1e-12)
        attn3 = attn[..., t:]
        attn = torch.cat([attn1, attn2, attn3], dim=-1)

        return attn
class Prompt(nn.Module):
    def __init__(
        self, 
        length: int = 10,
        dim: int = 768,
        position: int = 1,
        reducible: bool = False,
    ):
        super().__init__()
        self.length = length
        self.dim = dim
        self.position = position
        self.reducible = reducible

        tokens = nn.Parameter(torch.zeros(length, dim))
        self.register_parameter("tokens", tokens)
        nn.init.uniform_(self.tokens.data, 0, 0.01)

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.dim
        tokens = (self.tokens).expand(x.shape[0], -1, -1)
        if self.position > 0:
            x1, x2 = x[:, : self.position], x[:, self.position :]
            return torch.cat([x1, tokens, x2], dim=1)
        return torch.cat([tokens, x], dim=1)

    def reduce(self, x: torch.Tensor):
        if not self.reducible:
            return x

        if self.position > 0:
            x1, x2 = x[:, : self.position], x[:, self.position + self.length :]
            return torch.cat([x1, x2], dim=1)
        return x[:, self.length :]
class Prefixattention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        num_frames = 16,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))#nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.num_frames= num_frames
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.T_prefix = Prefix(2,dim,1,True)
        self.S_prefix = Prefix(10,dim,1,True)
    def forward(self, x):
        # x = self.add_prompt(x)
        x = x.permute(1,0,2) # -> B*L, 16, 768
        qkv = F.linear(input=x, weight=self.in_proj_weight, bias=self.in_proj_bias)
        B, N, C = x.shape
        # qkv = self.adapt_module("qkv", x)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.chunk(3, dim=-1)# q: B*L, 16, 768
        k,v = self.T_prefix(k,v) if N==self.num_frames else self.S_prefix(k,v)
        # k, v = self.add_prefix(k, v)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        k = k.reshape(B, -1, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)

        v = v.reshape(B, -1, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)#BL, head, N+Prefixlen , dim/head  ( 2364,12,18,64)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn =self.T_prefix.compensate(attn) if N==self.num_frames else self.S_prefix.compensate(attn)
        # self.compensate_prefix(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.adapt_module("proj", x)  # x = self.proj(x)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        x = x.permute(1,0,2)

        # x = self.reduce_prompt(x)

        return x