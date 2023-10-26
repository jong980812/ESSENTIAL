from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from typing import Optional
from einops import rearrange

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
    
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float16))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
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
        k,v = self.T_prefix(k,v) if N==16 else self.S_prefix(k,v)
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
        # attn =self.T_prefix.compensate(attn) if N==16 else self.S_prefix.compensate(attn)
        # self.compensate_prefix(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.adapt_module("proj", x)  # x = self.proj(x)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        x = x.permute(1,0,2)

        # x = self.reduce_prompt(x)

        return x
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0.):
        super().__init__()
        

        self.num_tadapter = num_tadapter
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Attention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        # self.MLP_Adapter_2 = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        # temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attn(self.T_Adapter_in(self.ln_1(xt))))
        else:
            xt = self.T_Adapter(self.attn(self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + self.drop_path(xt)
        
        # spatial adaptation
        x = x + self.S_Adapter(self.attn(self.ln_1(x)))
        # xn_1 =self.ln_1(x)
        # x = x + self.attn(xn_1)+self.S_Adapter(xn_1)
        # joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        
        
        
        
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class ViT_CLIP_jong(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None,num_classes=400,init_scale=0.001,spatial_type='avg',dropout_ratio=0.2):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.prompt_num = 2
        # self.prompt=[nn.Parameter(scale * torch.randn(width)) for i in range(self.prompt_num)]
        self.prompt1 =nn.Parameter(scale * torch.randn(width))
        self.prompt2 =nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate)

        self.ln_post = LayerNorm(width)

        embed_dim = 768
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=.02)

        self.init_weights(pretrained='clip')
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        if spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None
        
        
        
        
    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            print(f'load model from: {self.pretrained}')
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            print('Missing keys: {}'.format(msg.missing_keys))
            print('Unexpected keys: {}'.format(msg.unexpected_keys))
            print(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
 

        n = x.shape[1]
        # x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        # x = x + self.temporal_embedding
        # x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        
       
        
        # x=x[:,0:3].mean(dim=1)#PT
        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head
        
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.head(x)
        # [N, num_classes]
        return cls_score
        
   