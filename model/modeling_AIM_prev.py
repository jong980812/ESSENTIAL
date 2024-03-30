from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from einops import rearrange


class Adapter(nn.Module):
    def __init__(self, D_features, dim_mlp=192, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(dim_mlp)
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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0.,dim_mlp=192):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        self.dim_mlp = dim_mlp
        self.scale = scale
        self.MLP_Adapter = Adapter(d_model, dim_mlp=self.dim_mlp,skip_connect=False)
        self.MLP_Adapter_p = Adapter(d_model, dim_mlp=self.dim_mlp,skip_connect=False)
        self.S_Adapter = Adapter(d_model,dim_mlp=self.dim_mlp)
        self.S_Adapter_p = Adapter(d_model,dim_mlp=self.dim_mlp)
        self.T_Adapter = Adapter(d_model, skip_connect=False,dim_mlp=self.dim_mlp)
        self.T_Adapter_p = Adapter(d_model, skip_connect=False,dim_mlp=self.dim_mlp)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model,dim_mlp=dim_mlp)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def c_to_p(self):
        msg=self.T_Adapter_p.load_state_dict(self.T_Adapter.state_dict())
        msg=self.S_Adapter_p.load_state_dict(self.S_Adapter.state_dict())
        msg=self.MLP_Adapter_p.load_state_dict(self.MLP_Adapter.state_dict())
        for adapter in self.T_Adapter_p.parameters():#!과거 어답터 freeze
            adapter.requires_grad = False
        for adapter in self.S_Adapter_p.parameters():
            adapter.requires_grad = False
        for adapter in self.MLP_Adapter_p.parameters():
            adapter.requires_grad = False
        # self.MLP_Adapter_p = self.MLP_Adapter.detach().clone()
        # for n, m in self.named_modules():#! 현재 어답터 up projection zero로 
        #     if n=='MLP_Adapter' or n=='S_Adapter' or n=='T_Adapter':
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0)
        #                     nn.init.constant_(m2.bias, 0)
        
        print(msg)
    def attention(self, q,k,v):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q,k,v, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, first = False):
        if first:
            # ## x shape [HW+1, BT, D]
            # x,_=x
            # n, bt, d = x.shape
            # ## temporal adaptation
            # xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
            # xt_ln1=self.ln_1(xt)
            # xt = self.T_Adapter(self.attention(xt_ln1,xt_ln1,xt_ln1))
            # xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            # x = x + self.drop_path(xt)
            # ## spatial adaptation
            # x_ln1 = self.ln_1(x)
            # x = x + self.S_Adapter(self.attention(x_ln1,x_ln1,x_ln1))
            # ## joint adaptation
            # xn = self.ln_2(x)
            # x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
            x1,x2 = x
            n, bt, d = x1.shape
            B=bt//self.num_frames
            xt1 = rearrange(x1, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xt2 = rearrange(x2, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xn1 = self.ln_1(xt1)
            xn2 = self.ln_1(xt2)
            xt1_attn = self.attention(xn1,xn1,xn1)
            xt2_xattn = self.attention(xn2,xn1,xn1)
            #   xt1 = self.T_Adapter_p(xt1_attn)
            xt2 = self.T_Adapter(xt2_xattn)#sum([adapter(xtln) for adapter in self.T_Adapter])
            xt1 = rearrange(xt1_attn, 't (b n) d -> n (b t) d', n=n)
            xt2 = rearrange(xt2, 't (b n) d -> n (b t) d', n=n)
            xt1 = x1 + self.drop_path(xt1)
            xt2 = x2 + self.drop_path(xt2)
            ## spatial adaptation
            xn1=self.ln_1(xt1)
            xn2=self.ln_1(xt2)
            xt1 = xt1 +(self.attention(xn1,xn1,xn1))
            xt2 = xt2 + self.S_Adapter(self.attention(xn2,xn2,xn2))
            ## joint adaptation
            xn1 = self.ln_2(xt1)
            xn2 = self.ln_2(xt2)
            xt1 = xt1 + self.mlp(xn1)
            xt2 = xt2 + self.mlp(xn2) + self.drop_path(self.scale * self.MLP_Adapter(xn2))
            return xt1,xt2

            return x,x
        else:
            x1,x2 = x
            n, bt, d = x1.shape
            B=bt//self.num_frames
            xt1 = rearrange(x1, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xt2 = rearrange(x2, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xn1 = self.ln_1(xt1)
            xn2 = self.ln_1(xt2)
            xt1_attn = self.attention(xn1,xn1,xn1)
            xt2_xattn = self.attention(xn2,xn1,xn1)
            xt1 = self.T_Adapter_p(xt1_attn)
            xt2 = self.T_Adapter(xt2_xattn)#sum([adapter(xtln) for adapter in self.T_Adapter])
            xt1 = rearrange(xt1, 't (b n) d -> n (b t) d', n=n)
            xt2 = rearrange(xt2, 't (b n) d -> n (b t) d', n=n)
            xt1 = x1 + self.drop_path(xt1)
            xt2 = x2 + self.drop_path(xt2)
            ## spatial adaptation
            xn1=self.ln_1(xt1)
            xn2=self.ln_1(xt2)
            xt1 = xt1 + self.S_Adapter_p(self.attention(xn1,xn1,xn1))
            xt2 = xt2 + self.S_Adapter(self.attention(xn2,xn2,xn2))
            ## joint adaptation
            xn1 = self.ln_2(xt1)
            xn2 = self.ln_2(xt2)
            xt1 = xt1 + self.mlp(xn1) + self.drop_path(self.scale * self.MLP_Adapter_p(xn1))
            xt2 = xt2 + self.mlp(xn2) + self.drop_path(self.scale * self.MLP_Adapter(xn2))
            return xt1,xt2



class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1,dim_mlp=192):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i],dim_mlp=dim_mlp) for i in range(layers)])
        self.first = True
    def forward(self, x: torch.Tensor):
        x1 = x
        x2 = x[:]
        for layer, block in enumerate(self.resblocks):
            x1,x2 = block((x1,x2),self.first) 
        return x2
    def set_first(self,set):
        self.first = set
    def initial_adapter(self,init_scale):
        for n, m in self.resblocks.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            # nn.init.constant_(m2.weight, 0)
                            # nn.init.constant_(m2.bias, 0)
                            m2.weight.data.mul_(init_scale)
                            m2.bias.data.mul_(init_scale)
    def transfer_c_to_p(self):
        for block in self.resblocks:
            block.c_to_p()
    def down_freeze(self):
        for n, m in self.resblocks.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc1' in n2:
                        if isinstance(m2, nn.Linear):
                            # nn.init.constant_(m2.weight, 0)
                            # nn.init.constant_(m2.bias, 0)
                            m2.weight.requires_grad_(False)
                            m2.bias.requires_grad_(False)
    def down_unfreeze(self):
        for n, m in self.resblocks.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc1' in n2:
                        if isinstance(m2, nn.Linear):
                            # nn.init.constant_(m2.weight, 0)
                            # nn.init.constant_(m2.bias, 0)
                            m2.weight.requires_grad_(True)
                            m2.bias.requires_grad_(True)

class AIM_prev(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None,num_classes=400,init_scale=0.001,spatial_type='avg',dropout_ratio=0.2,dim_mlp=192):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate,dim_mlp=dim_mlp)

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
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
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
        
   