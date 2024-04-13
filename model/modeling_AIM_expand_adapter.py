import random
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
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, adapter = None):
        if adapter is not None:
            ## x shape [HW+1, BT, D]
            T_Adapter = adapter['T']
            S_Adapter = adapter['S']
            MLP_Adapter = adapter['MLP']
            n, bt, d = x.shape
            ## temporal adaptation
            xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
            if self.num_tadapter == 2:
                xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
            else:
                xt = T_Adapter(self.attention(self.ln_1(xt)))
            xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            x = x + self.drop_path(xt)
            ## spatial adaptation
            x = x + S_Adapter(self.attention(self.ln_1(x)))
            ## joint adaptation
            xn = self.ln_2(x)
            x = x + self.mlp(xn) + self.drop_path(self.scale * MLP_Adapter(xn))
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1,dim_mlp=192,adapter_layers=[]):
        super().__init__()
        self.width = width
        self.layers = layers
        self.adapter_layers = adapter_layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.dim_mlp = dim_mlp
        self.scale = scale
        self.Adapter_pool = nn.ModuleList()
        self.Adapters = nn.ModuleList([
            nn.ModuleDict({
            'T':Adapter(width, skip_connect=False,dim_mlp=self.dim_mlp),
            'S':Adapter(width,dim_mlp=self.dim_mlp),
            'MLP':Adapter(width, dim_mlp=self.dim_mlp,skip_connect=False)
        }) for i in adapter_layers
            ])
        self.task_id = 0
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i],dim_mlp=dim_mlp) for i in range(layers)])
    def add_task(self):
        self.task_id+=1
        print('**Add task id**')
        for n, m in self.Adapters.named_modules():
            for n2, m2 in m.named_modules():
                m2.requires_grad_(False)
        print('**Freeze previous adapter**')
        self.Adapter_pool.append(self.Adapters)
    def make_new_adapter(self):
        self.Adapters = nn.ModuleList([
            nn.ModuleDict({
            'T':Adapter(self.width, skip_connect=False,dim_mlp=self.dim_mlp),
            'S':Adapter(self.width,dim_mlp=self.dim_mlp),
            'MLP':Adapter(self.width, dim_mlp=self.dim_mlp,skip_connect=False)
        }) for i in self.adapter_layers
            ])
        for n, m in self.Adapters.named_modules():
            for n2, m2 in m.named_modules():
                if 'D_fc1' in n2:
                    if isinstance(m2, nn.Linear):
                        trunc_normal_(m2.weight, std=.02)
                        nn.init.constant_(m2.bias,0)
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)
                        # m2.weight.data.mul_(init_scale)
                        # m2.bias.data.mul_(init_scale)
        print('**New adapte **')


    def forward(self, x: torch.Tensor,task_id = None):
        if task_id is not None:
            self.Adapters = self.Adapter_pool[task_id]
        for i,block in enumerate(self.resblocks):
            if i in self.adapter_layers:
                x = block(x,self.Adapters[i])
            else:
                x = block(x)
        return x
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

class AIM_expand(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None,num_classes=400,init_scale=0.001,spatial_type='avg',dropout_ratio=0.2,dim_mlp=192,adapter_layers=[]):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.adapter_layers = adapter_layers
        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate,dim_mlp=dim_mlp,adapter_layers=self.adapter_layers)

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
            if 'Adapter' in n:
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

    def head_scailing(self,first_task,class_per_task,task_id):
        data=self.head.weight.clone().permute(1,0)
        # 스케일링할 열 범위 설정
        current=first_task+int(task_id-1*class_per_task)
        # 스케일링할 열 범위 설정
        cols_to_scale = data[:, first_task:first_task+task_id*class_per_task]  # (768, 6:12) 범위
        cols_reference = data[:, 0:first_task]  # (768, 0:7) 범위, 이 범위에 맞추려고 함

        # Norm 조정을 위한 함수 정의
        

        # Norm 조정 실행
        adjusted_cols = adjust_norm(cols_to_scale, cols_reference)
        # 조정된 열을 원본 데이터에 다시 삽입
        #self.head.weight[current:,: ] = torch.nn.Parameter(adjusted_cols.permute(1,0))
        data[:,first_task:first_task+task_id*class_per_task ] = adjusted_cols
        self.head.weight = nn.Parameter(data.permute(1,0))
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor,task_id=None,selection_results=None):
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
        if selection_results is not None:
            x = rearrange(x, 'n (b t) d -> n b t d', b=B)
            xs = []
            for b in range(B):
                x_ = self.transformer(x[:,b,:,:],selection_results[b])
                xs.append(x_.unsqueeze(0))
            x = torch.cat(xs,0)
            x = rearrange(x, 'b n t d -> n (b t) d', b=B,t=T)
        else:
            x = self.transformer(x,task_id) if task_id is not None else self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        # x = x.mean(1)
        
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head
        
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.head(x)
        # [N, num_classes]
        return cls_score
        
def adjust_norm(input_tensor, ref_tensor):
    # input_tensor와 ref_tensor의 norm 계산
    input_norm = input_tensor.norm(p=2, dim=0, keepdim=True)
    ref_norm = ref_tensor.norm(p=2, dim=0, keepdim=True)

    # ref_tensor의 평균 norm 계산
    ref_mean_norm = ref_norm.mean()
    
    # input_tensor의 각 열을 조정하여 ref_mean_norm과 비슷하게 만듦
    adjusted_tensor = input_tensor * (ref_mean_norm / input_norm)
    
    return adjusted_tensor