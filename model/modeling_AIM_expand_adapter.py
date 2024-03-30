from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Optional
import clip
from einops import rearrange
class Scaler(nn.Module):
    def __init__(self, scale: Optional[float] = None):
        super().__init__()

        if scale is None:
            self.register_parameter("scale", nn.Parameter(torch.tensor(1.0)))
        else:
            self.scale = scale

    def forward(self, input):
        return input * self.scale

    def extra_repr(self):
        learnable = isinstance(self.scale, nn.Parameter)
        return f"scale={self.scale:.4f}, learnable={learnable}"


class Adapter(nn.Module):
    def __init__(self, D_features, dim_mlp=192, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(dim_mlp)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        # self.scaler = Scaler(scale=1.0)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        # xs = self.scaler(xs)
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
    
# class Cross_T_Adapter(nn.Module):
#     def __init__(self, dim:int, n_head:int, num_frames:int, attn_mask:torch.Tensor=None):
        
#         self.cross_s_down = nn.Linear(dim, dim//self.down_ratio)
#         self.cross_t_down = nn.Linear(dim, dim//self.down_ratio)
#         self.ln_s_cross = norm_layer(dim//self.down_ratio)
#         self.t2s_cross = CrossAttentionP2C(dim//self.down_ratio, num_heads, num_frames)
#         self.cross_s_up = nn.Linear(dim//self.down_ratio, dim)
# class CrossAttentionP2C(nn.Module):
#     def __init__(self, dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
#         super().__init__()

#         # add for cross-attn
#         self.num_frames = num_frames
#         self.num_head = n_head
#         head_dim = dim // self.num_head
#         self.scale = head_dim ** -0.5
#         all_head_dim = head_dim * self.num_head
#         # self.clip_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
#         # self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
        

#         self.p2c_q = nn.Linear(dim, all_head_dim, bias=False)
#         self.p2c_q_bias = nn.Parameter(torch.zeros(all_head_dim))
#         self.p2c_kv = nn.Linear(dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
#         self.p2c_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
#         self.p2c_proj = nn.Linear(all_head_dim, dim)
        
#         self.attn_mask = attn_mask
    
#     def s2t_cross_attn(self, s_x, t_x): # s_x=[n (b t) d], t_x=[b n d]
#         B, _, _ = t_x.shape
#         t = s_x.shape[1] // t_x.shape[0]
#         s_x_pat = s_x[1:, :, :]
#         s_x_pat = rearrange(s_x_pat, 'n b d -> b n d') # batch -> token
#         s_x_pat = s_x_pat + self.clip_space_pos
#         t_x = rearrange(t_x, 'b (t n) d -> (b t) n d', t=t)
#         t_x = t_x + self.vmae_space_pos
#         s2t_q_bias = self.s2t_q_bias
#         s2t_kv_bias = self.s2t_kv_bias
        
#         s2t_q = F.linear(input=t_x, weight=self.s2t_q.weight, bias=s2t_q_bias)
#         s2t_q = rearrange(s2t_q, 'b n (h d) -> b h n d', h=self.num_head)
#         s2t_kv = F.linear(input=s_x_pat, weight=self.s2t_kv.weight, bias=s2t_kv_bias)
#         s2t_kv = rearrange(s2t_kv, 'b n (e h d) -> e b h n d',e=2, h=self.num_head)
#         s2t_k, s2t_v = s2t_kv[0], s2t_kv[1]
        
#         s2t_q = s2t_q * self.scale
#         s2t_attn = (s2t_q @ s2t_k.transpose(-2, -1))
        
#         s2t_attn = s2t_attn.softmax(dim=-1)
        
#         t_x = (s2t_attn @ s2t_v)
#         t_x = rearrange(t_x, 'b h t d -> b t (h d)')
#         t_x = self.t2s_proj(t_x)
#         t_x = rearrange(t_x, '(b t) n d -> b (t n) d', b=B)
#         return t_x

#     def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
#         return self.s2t_cross_attn(s_x, t_x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1,layer=0, num_frames=8, drop_path=0.,dim_mlp=192,cross=False):
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
        self.layer=layer
        self.cross=cross

        self.n_head = n_head
        self.d_model = d_model
        self.dim_mlp = dim_mlp
        # self.MLP_Adapter = Adapter(d_model, dim_mlp=dim_mlp,skip_connect=False)
        # self.S_Adapter = Adapter(d_model,dim_mlp=dim_mlp)
        self.scale = scale
        self.T_Adapter = nn.ModuleList([Adapter(d_model, skip_connect=False,dim_mlp=dim_mlp)])
        self.S_Adapter = nn.ModuleList([Adapter(d_model, skip_connect=True,dim_mlp=dim_mlp)])
        self.MLP_Adapter = nn.ModuleList([Adapter(d_model, skip_connect=False,dim_mlp=dim_mlp)])
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model,dim_mlp=dim_mlp)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, q,k,v):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q,k,v, need_weights=False, attn_mask=self.attn_mask)[0]
    
    
    
    def add_adapter(self,mode='copy'):
        new_T_adapter = Adapter(self.d_model, skip_connect=False,dim_mlp=self.dim_mlp)
        new_S_adapter = Adapter(self.d_model, skip_connect=True,dim_mlp=self.dim_mlp)
        new_MLP_adapter = Adapter(self.d_model, skip_connect=False,dim_mlp=self.dim_mlp)
        if mode=='new':
            # for new_adapter in [new_T_adapter,new_S_adapter,new_MLP_adapter]:
            # msg=new_T_adapter.load_state_dict(self.T_Adapter[-1].state_dict())
            # msg=new_S_adapter.load_state_dict(self.S_Adapter[-1].state_dict())
            # msg=new_MLP_adapter.load_state_dict(self.MLP_Adapter[-1].state_dict())
            # print(msg)
            for new_adapter in [new_T_adapter,new_S_adapter,new_MLP_adapter]:
                for n, m in new_adapter.named_modules():
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)
                        # elif 'D_fc1' in n2:
                        #     if isinstance(m2,nn.Linear):
                        #         trunc_normal_(m2.weight, std=.02)
                        #         nn.init.constant_(m2.bias, 0)
        elif mode == 'copy':
            msg=new_T_adapter.load_state_dict(self.T_Adapter[-1].state_dict())
            msg=new_S_adapter.load_state_dict(self.S_Adapter[-1].state_dict())
            msg=new_MLP_adapter.load_state_dict(self.MLP_Adapter[-1].state_dict())
            print(msg)
        elif mode =='copy_without_T':
            msg=new_T_adapter.load_state_dict(self.T_Adapter[-1].state_dict())
            msg=new_S_adapter.load_state_dict(self.S_Adapter[-1].state_dict())
            msg=new_MLP_adapter.load_state_dict(self.MLP_Adapter[-1].state_dict())
            for adapter in [new_S_adapter, new_MLP_adapter]:
                for param in adapter.parameters():
                    param.requires_grad = False
            for n, m in new_T_adapter.named_modules():
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
        
        self.T_Adapter.append(new_T_adapter)
        self.S_Adapter.append(new_S_adapter)
        self.MLP_Adapter.append(new_MLP_adapter)
    def del_adapter(self):
        adapter_len = len(self.T_Adapter)
        if adapter_len>2:
            del self.T_Adapter[0]
            del self.MLP_Adapter[0]
            del self.S_Adapter[0]

    
    def forward(self, x,prompt):
        ## x shape [HW+1, BT, D]
        x1,x2 = x # x1==x2 when task1
        n, bt, d = x1.shape
        B=bt//self.num_frames
        adapter_len = len(self.T_Adapter)
        if adapter_len > 1:
            ## temporal adaptation
            # xt1 = rearrange(x1, 'n (b t) d -> t (b n) d', t=self.num_frames)
            # xt2 = rearrange(x2, 'n (b t) d -> t (b n) d', t=self.num_frames)
            # with torch.no_grad():xn1 = self.ln_1(xt1)
            # xn2 = self.ln_1(xt2)
            # with torch.no_grad():xt1_attn = self.attention(xn1,xn1,xn1)
            # xt2_xattn = self.attention(xn2,xn1,xn1) if self.cross else self.attention(xn2,xn2,xn2)
            
            # with torch.no_grad():xt1 = self.T_Adapter[0](xt1_attn)
            # xt2 = self.T_Adapter[1](xt2_xattn)#sum([adapter(xtln) for adapter in self.T_Adapter])
            # xt1 = rearrange(xt1, 't (b n) d -> n (b t) d', n=n)
            # xt2 = rearrange(xt2, 't (b n) d -> n (b t) d', n=n)
            # with torch.no_grad():xt1 = x1 + self.drop_path(xt1)
            # xt2 = x2 + self.drop_path(xt2)
            # ## spatial adaptation
            # with torch.no_grad():xn1=self.ln_1(xt1)
            # xn2=self.ln_1(xt2)
            # with torch.no_grad():xt1 = xt1 + self.S_Adapter[0](self.attention(xn1,xn1,xn1))
            # xt2 = xt2 + self.S_Adapter[1](self.attention(xn2,xn2,xn2))
            # ## joint adaptation
            # with torch.no_grad():xn1 = self.ln_2(xt1)
            # xn2 = self.ln_2(xt2)
            # with torch.no_grad():xt1 = xt1 + self.mlp(xn1) + self.drop_path(self.scale * self.MLP_Adapter[0](xn1))
            # xt2 = xt2 + self.mlp(xn2) + self.drop_path(self.scale * self.MLP_Adapter[1](xn2))
            xt1 = rearrange(x1, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xt2 = rearrange(x2, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xn1 = self.ln_1(xt1)
            xn2 = self.ln_1(xt2)
            xt1_attn = self.attention(xn1,xn1,xn1)
            xt2_xattn = self.attention(xn2,xn1,xn1) if self.cross else self.attention(xn2,xn2,xn2)
            
            xt1 = self.T_Adapter[0](xt1_attn)
            xt2 = self.T_Adapter[1](xt2_xattn)#sum([adapter(xtln) for adapter in self.T_Adapter])
            xt1 = rearrange(xt1, 't (b n) d -> n (b t) d', n=n)
            xt2 = rearrange(xt2, 't (b n) d -> n (b t) d', n=n)
            xt1 = x1 + self.drop_path(xt1)
            xt2 = x2 + self.drop_path(xt2)
            ## spatial adaptation
            xn1=self.ln_1(xt1)
            xn2=self.ln_1(xt2)
            xt1 = xt1 + self.S_Adapter[0](self.attention(xn1,xn1,xn1))
            xt2 = xt2 + self.S_Adapter[1](self.attention(xn2,xn2,xn2))
            ## joint adaptation
            xn1 = self.ln_2(xt1)
            xn2 = self.ln_2(xt2)
            xt1 = xt1 + self.mlp(xn1) + self.drop_path(self.scale * self.MLP_Adapter[0](xn1))
            xt2 = xt2 + self.mlp(xn2) + self.drop_path(self.scale * self.MLP_Adapter[1](xn2))
            return xt1,xt2
        else:# Task1
            xt1 = rearrange(x1, 'n (b t) d -> t (b n) d', t=self.num_frames)
                # xt = sum([self.T_Adapter[i](self.attention(self.ln_1(xt))) for i in range(adapter_len)])
            xt1 = self.ln_1(xt1)
            xt1 = self.attention(xt1,xt1,xt1)
            xt1 = self.T_Adapter[0](xt1)
            xt1 = rearrange(xt1, 't (b n) d -> n (b t) d', n=n)
            xt1 = x1 + self.drop_path(xt1)
            ## spatial adaptation
            xn1=self.ln_1(xt1)
            xt1 = xt1 + self.S_Adapter[0](self.attention(xn1,xn1,xn1))
            ## joint adaptation
            xn1 = self.ln_2(xt1)
            xt1 = xt1 + self.mlp(xn1) + self.drop_path(self.scale * self.MLP_Adapter[0](xn1))
            return xt1,xt1
            


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1,dim_mlp=192,cross=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.cross=cross
        # self.prompt = nn.Parameter((width ** -0.5)*torch.randn((5,2,10,768)))
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter,i, num_frames, dpr[i],dim_mlp=dim_mlp,cross=self.cross) for i in range(layers)])
    def add_adapters(self,mode):
        for i in range(self.layers):
            self.resblocks[i].add_adapter(mode)
    def del_adapters(self):
        for i in range(self.layers):
            self.resblocks[i].del_adapter()
    def freeze_adapters(self):
        for block in self.resblocks:
            for adapter in block.T_Adapter.parameters():
                adapter.requires_grad = False
            for adapter in block.S_Adapter.parameters():
                adapter.requires_grad = False
            for adapter in block.MLP_Adapter.parameters():
                adapter.requires_grad = False
    def forward(self, x: torch.Tensor):
        x1 = x
        x2 = x.clone().detach()#
        for layer, block in enumerate(self.resblocks):
            # if layer<5:
            #     x1,x2= block((x1,x2),self.prompt[layer])
            # else:
            x1,x2= block((x1,x2),None)
            
        return x2
        # return self.resblocks((x1,x2))

class AIM_adapter(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None,num_classes=400,init_scale=0.001,spatial_type='avg',dropout_ratio=0.2,dim_mlp=192,cross=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)  
        self.cross = cross
        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.init_scale = init_scale
        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate,dim_mlp=dim_mlp,cross=self.cross)

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
    def head_initial(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.head.apply(_init_weights)
        self.head.weight.data.mul_(self.init_scale)
        self.head.bias.data.mul_(self.init_scale)
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
        '''
        태스크 1 끝나고 
        태스크 2 시작할때 어답터 세트를 추가 (T,MLP,S)
        추가된 new adapter들한테 load state로 태스크1weight받아옴.
        태스크 3일때는 어답터가 3세트잖아요. 여기서부터는 첫번째꺼 del
        '''
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
        
   