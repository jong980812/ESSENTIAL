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


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float16))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class Adapter(nn.Module):
    def __init__(self, D_features=768, dim_mlp=192, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, dim_mlp)#!
        self.D_fc2 = nn.Linear(dim_mlp, D_features)
        
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
class Attn_Adapter(nn.Module):
    def __init__(self, D_features=768, dim_mlp=192, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(dim_mlp)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.attn = nn.MultiheadAttention(dim_mlp, 6)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        # self.scaler = Scaler(scale=1.0)
        
    def attention(self, q,k,v):
        # self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q,k,v, need_weights=False, attn_mask=None)[0]
    
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.attention(xs,xs,xs)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        # xs = self.scaler(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0.,dim_mlp=192,adapter=True):
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
        self.adapter = adapter
        self.n_head = n_head
        self.d_model = d_model
        self.dim_mlp = dim_mlp

        self.scale = scale
        
        if self.adapter:
            self.Attn_Adapter = Attn_Adapter(d_model,dim_mlp=dim_mlp,skip_connect=False)
        # self.MLP_Adapter = Adapter(d_model, dim_mlp=dim_mlp,skip_connect=False)
        # self.S_Adapter = Adapter(d_model,dim_mlp=dim_mlp)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, q,k,v):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q,k,v, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        # B=bt//self.num_frames
        if self.adapter:
            xn1=self.ln_1(x)
            xt1 = rearrange(xn1, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xt1 = self.Attn_Adapter(xt1) 
            xt1 = rearrange(xt1, 't (b n) d -> n (b t) d', n=n)
            # xt1 = x + self.drop_path(xt1)
            ## spatial adaptation
            # xt1 = xt1 + self.S_Adapter(self.attention(xn1,xn1,xn1))
            x = x + self.attention(xn1,xn1,xn1) + self.drop_path(0.5*xt1)
            ## joint adaptation
            xn2 = self.ln_2(x)
            x = x + self.mlp(xn2) #+ self.drop_path(self.scale * self.MLP_Adapter(xn1))
        else:
            xn1 = self.ln_1(x)
            x = x + self.attention(xn1,xn1,xn1)
            x = x + self.mlp(self.ln_2(x))
        return x
            


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1,dim_mlp=192,adapter_layers=[]):
        super().__init__()
        self.width = width
        self.layers = layers
        # self.prompt = nn.Parameter((width ** -0.5)*torch.randn((5,2,10,768)))
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.adapter_layers = adapter_layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i],dim_mlp=dim_mlp,adapter = (i) in self.adapter_layers) for i in range(layers)])
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
        for layer, block in enumerate(self.resblocks):
            # if layer<5:
            #     x1,x2= block((x1,x2),self.prompt[layer])
            # else:
            x= block(x)
            
        return x
        # return self.resblocks((x1,x2))

class AIM_attn_adapter(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None,num_classes=400,init_scale=0.001,spatial_type='avg',dropout_ratio=0.2,dim_mlp=192,adapter_layers=[],class_mask=None,args=None):
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
        # self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.init_scale = init_scale
        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate,dim_mlp=dim_mlp,adapter_layers=adapter_layers)
        self.ln_post = LayerNorm(width)

        embed_dim = 768
        self.each_head = args.each_head
        if self.each_head:
            self.head = nn.ModuleList()
            for mask in class_mask:
                n_class = len(mask)
                head= nn.Linear(embed_dim,n_class)
                trunc_normal_(head.weight, std=.02)
                self.head.append(head)
            self.init_weights(pretrained='clip')
            for head in self.head:
                head.weight.data.mul_(init_scale)
                head.bias.data.mul_(init_scale)
        else:
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

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'Attn_Adapter' in n:
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

    def forward(self, x: torch.Tensor, train=False,task_id =-1):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        # x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        # x = x + self.temporal_embedding
        # x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
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
        
        
        if not self.each_head:#* each head아니면 그냥 원래대로 return
            cls_score = self.head(x)
            return cls_score
        if train:
        # [N, in_channels]
            cls_score = self.head[task_id](x)#! 학습 중에는 현재 태스크 알 수 있음.
        else:
            logits = [self.head[t](x) for t in range(task_id+1)]
            cls_score = torch.cat(logits,1)
        # [N, num_classes]
        return cls_score
        # [N, in_channels]
        # cls_score = self.head(x)
        # # [N, num_classes]
        # return cls_score
        
   