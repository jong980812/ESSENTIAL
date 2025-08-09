from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from einops import rearrange
np.random.seed(0)

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
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


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
        self.n_head = n_head
        self.adapter = adapter
        if self.adapter:
            self.dim_mlp = dim_mlp
            self.MLP_Adapter = Adapter(d_model, dim_mlp=self.dim_mlp,skip_connect=False)
            self.S_Adapter = Adapter(d_model,dim_mlp=self.dim_mlp)
            self.scale = scale
            self.T_Adapter = Adapter(d_model, skip_connect=False,dim_mlp=self.dim_mlp)
            if num_tadapter == 2:
                self.T_Adapter_in = Adapter(d_model,dim_mlp=dim_mlp)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def temporal_attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.temporal_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def forward(self, x: torch.Tensor, b = 1):
        if self.adapter:
            ## x shape [HW+1, BT, D]
            B =b
            n, bt, d = x.shape
            T = bt//B
            ## temporal adaptation
            xt = rearrange(x, 'n (b t) d -> t (b n) d', t=T)
            if self.num_tadapter == 2:
                xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
            else:
                xt = self.T_Adapter(self.attention(self.ln_1(xt)))
            xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            x = x + self.drop_path(xt)
            ## spatial adaptation
            x = x + self.S_Adapter(self.attention(self.ln_1(x)))
            ## joint adaptation
            xn = self.ln_2(x)
            x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
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
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i],dim_mlp=dim_mlp,adapter = (i) in self.adapter_layers) for i in range(layers)])

    def forward(self, x: torch.Tensor,b = 1):
        for i,block in enumerate(self.resblocks):
            x = block(x=x,b=b)
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
class MR_module(nn.Module):
    def __init__(self, temp_mode:str,d_model: int, attn_mask: torch.Tensor = None, num_frames=8, drop_path=0.2,fs_topk=8,len_sem_prompt=8,class_id=-1,task_id=-1,mode= 'cross'):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fs_topk = fs_topk
        self.task_id = task_id #! global is set -1
        self.class_id = class_id
        d_model = 768
        n_head = 12
        # self.temporal_encoding = nn.Parameter(torch.zeros(1, num_frames, d_model))
        self.len_sem_prompt = len_sem_prompt
        self.mode = mode
        if mode == 'cross' or mode=='self':
            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.attn_mask = attn_mask
            self.ln_1 = LayerNorm(d_model)
            self.ln_2 = LayerNorm(d_model)
            self.ln_tokens = LayerNorm(d_model)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                # ("c_fc", nn.Linear(d_model, d_model )),
                ("gelu", QuickGELU()),
                # ("c_proj", nn.Linear(d_model , d_model))
                ("c_proj", nn.Linear(d_model * 4 , d_model))
                ]))
            self.prompt = nn.Parameter(torch.FloatTensor(len_sem_prompt, d_model), requires_grad=True)
            nn.init.uniform_(self.prompt)
        elif mode =='3_layer_mlp':
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc1", nn.Linear(d_model, d_model)),
                ("gelu", QuickGELU()),
                ("c_fc2", nn.Linear(d_model , d_model)),
                ("gelu", QuickGELU()),
                ("c_fc3", nn.Linear(d_model,d_model))
                ]))
            self.prompt = nn.Parameter(torch.FloatTensor(len_sem_prompt, d_model), requires_grad=True)
            nn.init.uniform_(self.prompt)
    def forward(self, x):
        B,kv_T,D = x.shape

        x = rearrange(x, 'b t d -> t b d',b=B,t=kv_T)
        if self.mode=='general':
            pass
            # frame_token= frame_token + self.mlp(frame_token)
        elif self.mode =='cross':
            frame_token = self.prompt.expand(B,-1,-1)
            frame_token = rearrange(frame_token, 'b t d -> t b d',b=B,t=self.len_sem_prompt)
            ln_tokens = self.ln_tokens(frame_token)#!T,b,d
            frame_token = frame_token + self.drop_path(self.attention(ln_tokens,self.ln_1(x)))
            frame_token = frame_token + self.drop_path(self.mlp(self.ln_2(frame_token)))
            frame_token = rearrange(frame_token, 't b d -> b t d',b=B,t=self.len_sem_prompt)

        elif self.mode =='self':
            frame_token = self.prompt.expand(B,-1,-1)
            frame_token = rearrange(frame_token, 'b t d -> t b d',b=B,t=self.len_sem_prompt)
            ln_tokens = self.ln_tokens(frame_token)#!T,b,d
            ln_x = self.ln_1(x)
            new_x = torch.cat([ln_x,ln_tokens],dim=0)# (kv_T+len prompt, B, D)
            new_x = new_x+self.drop_path(self.attention(new_x,new_x))
            new_x = new_x + self.drop_path(self.mlp(self.ln_2(new_x)))
            frame_token = new_x[kv_T:,:,:]
            frame_token = rearrange(frame_token, 't b d -> b t d',b=B,t=self.len_sem_prompt)
            
        elif self.mode =='3_layer_mlp':
            frame_token = self.prompt.expand(B,-1,-1)
            frame_token = rearrange(frame_token, 'b t d -> t b d',b=B,t=self.len_sem_prompt)
            x = self.mlp(x)
            frame_token = frame_token+x
            frame_token = rearrange(frame_token, 't b d -> b t d',b=B,t=self.len_sem_prompt)
            
        elif self.mode =='linear':
            x = rearrange(x, 't b d -> b ( t d )', b = B, t = kv_T)
            x = self.mlp(x)
            frame_token = rearrange(x, 'b ( t d ) -> b t d ', b = B, t = self.len_sem_prompt)
        # frame_token = rearrange(frame_token, 't b d -> b t d',b=B,t=self.len_sem_prompt)
        return frame_token
    def freeze(self,block_list=None):
        freeze_list = []
        if block_list is None:
            print(f'MR_module ({self.task_id}) is all freezed')
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                for block in block_list:#if block in block_list
                    if block in name:
                        param.requires_grad = False
                        freeze_list.append(name)
                        break
                    else:
                        param.requires_grad = True
            print(f'freeze_list:{freeze_list}')
        return
    def unfreeze(self,block_list=None):
        if block_list is None:
            print(f'MR_module ({self.task_id}) is all activated')
            for param in self.parameters():
                param.requires_grad = True
        else:
            unfreeze_list = []
            for name, param in self.named_parameters():
                for block in block_list:#if block in block_list
                    if block in name:
                        param.requires_grad = True
                        unfreeze_list.append(name)
                        break
                    else:
                        param.requires_grad = False
            print(f'unfreeze_list:{unfreeze_list}')
        return
    def attention(self, q: torch.Tensor,kv:torch.Tensor, need_weights=False):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, kv, kv, need_weights=need_weights, attn_mask=self.attn_mask)[0] if not need_weights else self.attn(q, kv, kv, need_weights=need_weights, attn_mask=self.attn_mask)[1]
class Decoder_ResidualAttentionBlock_time(nn.Module):
    def __init__(self, temp_mode:str,d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0.2,dim_mlp=192,fs_topk=8):
        super().__init__()
        self.temp_mode = temp_mode
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fs_topk = fs_topk
        if self.temp_mode=='self':
            d_model = 768
            n_head = 12
            self.attn = nn.MultiheadAttention(d_model, n_head)
            # self.mlp = nn.Sequential(OrderedDict([
            #     ("c_fc", nn.Linear(d_model, d_model * 4)),
            #     ("gelu", QuickGELU()),
            #     ("c_proj", nn.Linear(d_model * 4, d_model))
            # ]))
            self.ln_1 = LayerNorm(d_model)
            self.ln_2 = LayerNorm(d_model)
            # self.ln_cls = LayerNorm(d_model)
            self.attn_mask = attn_mask
        elif self.temp_mode=='attention':
            d_model = 768
            n_head = 12
            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.attn_mask = attn_mask
            self.ln_1 = LayerNorm(d_model)
            self.ln_cls = LayerNorm(d_model)
        elif self.temp_mode =='ba':
            '''
            Bottleneck cross attention 
            https://arxiv.org/abs/2311.18825
            '''
            self.attn = nn.MultiheadAttention(dim_mlp, n_head)
            self.ln_1 = LayerNorm(dim_mlp)
            self.ln_cls = LayerNorm(dim_mlp)
            self.attn_mask = attn_mask
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.kv_down = nn.Linear(d_model,dim_mlp)
            self.time_down = nn.Linear(d_model,dim_mlp)
            self.time_up = nn.Linear(dim_mlp,d_model)
            self.time_act = nn.GELU()
            self.initial_adapter()
    def initial_adapter(self):
        for n, m in self.time_up.named_modules():
            for n2, m2 in m.named_modules():
                if isinstance(m2, nn.Linear):
                    nn.init.constant_(m2.weight, 0)
                    nn.init.constant_(m2.bias, 0)
    def attention(self, q: torch.Tensor,kv:torch.Tensor, need_weights=False):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, kv, kv, need_weights=need_weights, attn_mask=self.attn_mask)[0] if not need_weights else self.attn(q, kv, kv, need_weights=need_weights, attn_mask=self.attn_mask)[1]
    def forward(self, cls: torch.Tensor,x: torch.Tensor,get_frame=False):
        #입력 cls_token 1,B,D
        B = x.shape[1]# X: T,B,D
        # cls = self.decoder_cls.expand(B,-1).unsqueeze(1) # B,1,D
        if self.temp_mode=='self':
            new_x = torch.cat([cls,x],dim=0) # T+1, B,D
            ln1 = self.ln_1(new_x)
            if get_frame:
                return self.attention(ln_cls,ln1,need_weights=True)
            new_x = new_x + self.drop_path(self.attention(ln1,ln1))
            cls = new_x[0:1,:,:]
            # cls = cls + self.drop_path(self.mlp(self.ln_2(cls)))
        elif self.temp_mode=='attention':
            ln_cls = self.ln_cls(cls)
            ln1 = self.ln_1(x)
            cls = cls + self.drop_path(self.attention(ln_cls,ln1))
            if get_frame:
                attention_map = self.attention(ln_cls,ln1,need_weights=True)
                return attention_map,cls
        elif self.temp_mode =='ba':
            ln_cls = self.ln_cls(self.time_down(cls))
            ln1 = self.ln_1(self.kv_down(x))
            x_cls= self.time_act(self.attention(ln_cls,ln1))
            cls = self.time_up(x_cls)+cls
        return cls

class CLIPs(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int,
                 num_frames: int,
                 patch_size: int,
                width: int,
                layers: int,
                heads: int,
                drop_path_rate,
                num_tadapter=1,
                 adapter_scale=0.5,
                 pretrained=None,
                num_classes=400,
                init_scale=0.001,
                spatial_type='avg',
                dropout_ratio=0.2,
                dim_mlp=192,
                adapter_layers=[],
                class_mask=None,
                args=None):
        super().__init__()
        self.args = args
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.embed_dim = 768
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.imagenet=args.imagenet
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=True if self.imagenet else False)
        self.ln_pre = LayerNorm(width) if not self.imagenet else nn.Identity()
        self.adapter_layers = adapter_layers
        self.num_frames = num_frames
        self.decoder_temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.use_aim_weight = args.use_aim_weight
        self.replay_token = args.replay_token
        self.memory_mode = args.memory_mode
        self.oracle = args.fine_tune_path
        # if args.use_aim_weight:
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.temporal_layer =args.temporal_layer
        self.fs_topk = args.fs_topk
        self.selected_selection = args.selected_selection
        self.data_set = args.data_set
        if self.replay_token:
            self.len_sem_prompt = args.len_sem_prompt
            if self.memory_mode=='global':
                self.mr_module = nn.ModuleList([MR_module(args.temp_mode, width, None, num_frames, drop_path=drop_path_rate,fs_topk=self.fs_topk,len_sem_prompt=self.len_sem_prompt,task_id=-1,mode=args.prompt_mode)])
            elif self.memory_mode=='task':
                self.mr_module = nn.ModuleList([MR_module(args.temp_mode, width, None, num_frames, drop_path=drop_path_rate,fs_topk=self.fs_topk,len_sem_prompt=self.len_sem_prompt,task_id=i,mode=args.prompt_mode) for i in range(args.num_tasks)])
            elif self.memory_mode =='identity':
                self.mr_module = nn.Identity()
        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=2 if args.data_set=='SSV2' else 1, scale=adapter_scale, drop_path=drop_path_rate,dim_mlp=dim_mlp,adapter_layers=self.adapter_layers)
        self.decoder_cls = nn.Parameter(scale * torch.randn(width))
        self.decoder_transformer_for_cls = nn.Sequential(*[Decoder_ResidualAttentionBlock_time(args.temp_mode, width, args.temporal_heads, None,0.2, num_tadapter, num_frames, drop_path=drop_path_rate,dim_mlp=dim_mlp,fs_topk=self.fs_topk) for _ in range(args.temporal_layer)])
        self.ln_post = LayerNorm(width)
        self.cos = args.cos
        self.static_matching = args.static_matching
        self.temporal_matching = args.temporal_matching
        self.class_mask = class_mask
                
        #!!
        

        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=.02)
        self.init_weights(pretrained='clip')
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)
        if self.cos:
            self.cos_loss = AngularPenaltySMLoss('cosface')
            self.cos_temp = args.cos_temp
            init_scale = 1.0
            self.head = nn.Linear(self.embed_dim, num_classes,bias=False) if num_classes > 0 else nn.Identity()
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
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
        
        
        
    def unfreeze(self,block_list):
        unfreeze_list = []
        for name, param in self.named_parameters():
            for block in block_list:#if block in block_list
                if block in name:
                    param.requires_grad = True
                    unfreeze_list.append(name)
                    break
                else:
                    param.requires_grad = False
        print(f'unfreeze_list:{unfreeze_list}')
    def freeze_all_MR(self):
        if self.memory_mode=='task':
            for asso in self.mr_module:
                print(f'MR_module ({asso.task_id}) is all freezed')
                for param in asso.parameters():
                    param.requires_grad = False
        elif self.memory_mode=='class':
            for asso in self.mr_module:
                print(f'MR_module ({asso.class_id}) is all freezed')
                for param in asso.parameters():
                    param.requires_grad = False
        elif self.memory_mode=='global':
            for asso in self.mr_module:
                print(f'MR_module global ({asso.task_id}) is all freezed')
                for param in asso.parameters():
                    param.requires_grad = False
        
        return
    def unfreeze_MR(self,task_id = -1):
        if self.memory_mode=='task':
            cur_asso = self.mr_module[task_id]
            print(f'MR_module ({cur_asso.task_id}) is activated')
            for param in cur_asso.parameters():
                param.requires_grad = True
        elif self.memory_mode =='class':
            cur_classes = self.class_mask[task_id]
            for cls in cur_classes:
                cur_asso = self.mr_module[cls]
                print(f'MR_module class ({cur_asso.class_id}) is activated')
                for param in cur_asso.parameters():
                    param.requires_grad = True
        elif self.memory_mode =='global':
            cur_asso = self.mr_module[0]
            print(f'MR_module (global) is activated')
            for param in cur_asso.parameters():
                param.requires_grad = True
        return
    def update_from_previous_associ(self,task_id = -1):
        cur_asso = self.mr_module[task_id]
        pre_asso = self.mr_module[task_id-1]
        pre_weight = pre_asso.state_dict()
        print(cur_asso.load_state_dict(pre_weight,strict=True))
        print(f'MR_module ({cur_asso.task_id}) is updated from previous mr_module')
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
            if self.imagenet:
                weight= torch.load('{your_path}/vit_1k.pt')
                pretrain_dict = convert_clip(weight)
                del weight
                print("IN1K weight !!")
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
        return {'absolute_pos_embed', 'temporal_embedding','decoder_temporal_embedding'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}
    def get_cls_tokens(self,x):
        if len(x.shape)==4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.shape 
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) #! Positional embedding, (8*10), 197, 768
        if self.use_aim_weight:
            temporal_embedding=self.temporal_embedding 
            if temporal_embedding.shape[1]!=(T):
                temporal_embedding = F.interpolate(
                    temporal_embedding.unsqueeze(1), size=(T,768), mode='bilinear', align_corners=False
                ).squeeze(1)
            n = x.shape[1]
            x = rearrange(x, '(b t) n d -> (b n) t d', t=T)
            x = x + temporal_embedding
            x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x,B)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b t d',b=B,t=T)
        return x
    def forward(self, x: torch.Tensor, train=False,class_id=-1,task_id =-1,sample_task_id=-1,
                get_frame=False,rehearsal = False,inference = False,selected_frame=None):
            
        B, C, T, H, W = x.shape 
        if get_frame:
            # Uniform selection
            frame_index = torch.tensor(np.sort(np.random.choice(range(T),self.fs_topk,False)), dtype=torch.int32).unsqueeze(0).unsqueeze(0)

            return frame_index,T,None
        # x = x[:,:,3,:,:].unsqueeze(2)#! single frame
        if rehearsal:
            with torch.no_grad():
                x = self.get_cls_tokens(x)
        else:
            x = self.get_cls_tokens(x)
                # x = torch.randn(1, 8, 768, device=x.device, dtype=x.dtype)
                # task_id=0
        # final = self.head(x.mean(1))
        # return (final,None),None,None
        decoder_temporal_embedding=self.decoder_temporal_embedding 
        if decoder_temporal_embedding.shape[1]!=(T):
            decoder_temporal_embedding = F.interpolate(
                decoder_temporal_embedding.unsqueeze(1), size=(T,768), mode='bilinear', align_corners=False
            ).squeeze(1)

        cls_origin = self.decoder_cls.expand(B,-1).unsqueeze(1) # B,1,D
        cls_origin = rearrange(cls_origin, 'b t d -> t b d',b=B,t=1)
        cls_virtual = self.decoder_cls.expand(B,-1).unsqueeze(1) # B,1,D
        cls_virtual = rearrange(cls_virtual, 'b t d -> t b d',b=B,t=1)
        x = x + decoder_temporal_embedding
        if inference:
            return self.inference(cls_origin,x,cls_virtual,task_id)
        elif rehearsal:
            x_selected = x[torch.arange(B)[:, None], selected_frame]
            return self.rehearsal(cls_origin,cls_virtual,x_selected,sample_task_id,class_id)
        
        new_x = torch.zeros(B,self.fs_topk,self.embed_dim).to(x.device)
        for i in range(B):
            new_x[i] = x[i,np.sort(np.random.choice(range(8),self.fs_topk,False))]
            # new_x[i] = x[i,np.array([4,5,6,7])]
        if self.memory_mode=='task':
            cur_mr_module = self.mr_module[task_id]
            frame_prompt = cur_mr_module(new_x) # frame_token is b len_p d #? debugging으로 req grad check
        elif self.memory_mode =='global':
            cur_mr_module = self.mr_module[0]
            frame_prompt = cur_mr_module(new_x) # frame_token is b len_p d #? debugging으로 req grad check
        elif self.memory_mode =='class':
            batch = list()
            for i in range(B):
                cur_mr_module=self.mr_module[class_id[i].item()]
                each_x = x[i:i+1,:,:]# 1, t, d
                frame_token = cur_mr_module(each_x) # frame_token is (1,len_p,d) 
                batch.append(frame_token)
            frame_prompt = torch.cat(batch, dim=0)# B, len_p, d
        elif self.memory_mode =='identity':
            # cur_mr_module = self.mr_module[task_id]
            # frame_prompt = cur_mr_module(new_x) # frame_token is b len_p d #? debugging으로 req grad check
            x = rearrange(x, 'b t d -> t b d',b=B,t=T)
            for i, decoder in enumerate(self.decoder_transformer_for_cls):
                cls_origin = decoder(cls_origin,x)
            cls_len = cls_origin.shape[0]
                
            cls_origin = rearrange(cls_origin, 't b d -> b d t',b=B,t=cls_len)#! B,D,cls_len
            cls_origin = cls_origin.unsqueeze(-1).unsqueeze(-1)
            static_matching_loss=None
            temporal_matching_loss=None
        if self.memory_mode != 'identity':
            x = rearrange(x, 'b t d -> t b d',b=B,t=T)
            frame_prompt = rearrange(frame_prompt, 'b t d -> t b d',b=B,t=self.len_sem_prompt)
            static_matching_loss = F.mse_loss(x,frame_prompt) if self.static_matching else None

            for i, decoder in enumerate(self.decoder_transformer_for_cls):
                cls_origin = decoder(cls_origin,x)
            # for i, decoder in enumerate(self.decoder_transformer_for_cls):
                # cls_virtual = decoder(cls_virtual,frame_prompt)
            cls_len = cls_origin.shape[0]
            cls_origin = rearrange(cls_origin, 't b d -> b d t',b=B,t=cls_len)#! B,D,cls_len
            cls_virtual = rearrange(cls_virtual, 't b d -> b d t',b=B,t=cls_len)#! B,D,cls_len
            temporal_matching_loss = F.mse_loss(cls_origin, cls_virtual) if self.temporal_matching else None
            cls_origin = cls_origin.unsqueeze(-1).unsqueeze(-1)
            # cls_virtual = cls_virtual.unsqueeze(-1).unsqueeze(-1)
        
        if self.avg_pool is not None:
            cls_origin = self.avg_pool(cls_origin)
            # cls_virtual = self.avg_pool(cls_virtual)  

        if self.dropout is not None:
            cls_origin = self.dropout(cls_origin)
            # cls_virtual = self.dropout(cls_virtual)

        cls_origin = cls_origin.view(cls_origin.shape[0], -1)
        # cls_virtual = cls_virtual.view(cls_virtual.shape[0], -1)
        
        if self.cos:
            cls_origin = F.linear(F.normalize(cls_origin, p=2, dim=-1), F.normalize(self.head.weight, p=2, dim=-1))
            cls_origin = self.cos_temp * cls_origin  # temperature set as 16
            cls_virtual = None#F.linear(F.normalize(cls_virtual, p=2, dim=-1), F.normalize(self.head.weight, p=2, dim=-1))
            # cls_virtual = self.cos_temp * cls_virtual  # temperature set as 16
        else:
        # [N, in_channels]
            cls_origin = self.head(cls_origin)
            cls_virtual = None#self.head(cls_virtual)# if self.data_set !='SSV2' else None
        return (cls_origin,cls_virtual),static_matching_loss,temporal_matching_loss
    
    
    def inference(self,cls_origin,x,cls_virtual,task_id):
        '''
        cls shape -> t, b, d
        x -> b,t,d
        '''
        frame_prompt =None
        self.eval()
        static_matching_loss=None
        B=cls_origin.shape[1]
        cls_sparse = cls_origin.clone().detach()
        B,T,D = x.shape
        if task_id is not None:
            batch = list()
            # for i in range(B):
            cur_mr_module=self.mr_module[0]
            new_x = torch.zeros(B,self.fs_topk,self.embed_dim).to(x.device)
            for i in range(B):
                new_x[i] = x[i,np.array([3])]
                # each_x = x[i:i+1,:,:]# 1, t, d
            frame_prompt = cur_mr_module(new_x) # frame_token is (1,len_p,d) 
            # batch.append(frame_token)
            # frame_prompt = torch.cat(batch, dim=0)# B, len_p, d
        # # else:
        x = rearrange(x, 'b t d -> t b d',b=B,t=T)
        if task_id is not None:#! oracle mode
            static_matching_loss =F.pairwise_distance(x.reshape(B,8,-1),frame_prompt.reshape(B,8,-1),p=2).mean() if self.static_matching else None

        for i, decoder in enumerate(self.decoder_transformer_for_cls):
            cls_origin = decoder(cls_origin,x)
        cls_len = cls_origin.shape[0]
        cls_origin = rearrange(cls_origin, 't b d -> b d t',b=B,t=cls_len)#! B,D,cls_len
        cls_origin = cls_origin.unsqueeze(-1).unsqueeze(-1)
        if self.avg_pool is not None:
            cls_origin = self.avg_pool(cls_origin)
        if self.dropout is not None:
            cls_origin = self.dropout(cls_origin)
        cls_origin = cls_origin.view(cls_origin.shape[0], -1)

        
        if self.cos:
            origin = F.linear(F.normalize(cls_origin, p=2, dim=-1), F.normalize(self.head.weight, p=2, dim=-1))
            origin = self.cos_temp * cls_origin  # temperature set as 16
        else:
            # virtural = self.head(cls_virtual)
            origin = self.head(cls_origin) #+ virtural
            # origin = (origin+virtural)/2.0
        return (origin ,frame_prompt),(static_matching_loss)
    
    
    
    def rehearsal(self,cls_origin,cls_virtual,x,sample_task_id,class_id):
        '''
        cls shape -> t, b, d
        x -> b,t,d
        '''
        B,kv_T,D = x.shape
        assert kv_T == self.fs_topk, "kv_T and self.fs_topk must be equal"
        #?Making bath for rehearsal
        if self.memory_mode=='task':
            batch = list()
            for i in range(B):
                cur_mr_module=self.mr_module[sample_task_id[i]]
                each_x = x[i:i+1,:,:]# 1, t, d
                frame_token = cur_mr_module(each_x) # frame_token is (1,len_p,d) 
                batch.append(frame_token)
            frame_prompt = torch.cat(batch, dim=0)# B, len_p, d
        elif self.memory_mode =='class':
            batch = list()
            for i in range(B):
                cur_mr_module=self.mr_module[class_id[i]]
                each_x = x[i:i+1,:,:]# 1, t, d
                frame_token = cur_mr_module(each_x) # frame_token is (1,len_p,d) 
                batch.append(frame_token)
            frame_prompt = torch.cat(batch, dim=0)# B, len_p, d
        elif self.memory_mode =='global':
            cur_mr_module = self.mr_module[0]
            frame_prompt = cur_mr_module(x) # frame_token is b len_p d #? debugging으로 req grad check
    
        x = rearrange(x, 'b t d -> t b d',b=B,t=kv_T)
        if self.memory_mode=='identity':
            x = rearrange(x, 't b d -> b d t')
            # frame_prompt = F.interpolate(x, size=self.len_sem_prompt, mode='linear', align_corners=False)  # 선형 보간
            # frame_prompt = rearrange(frame_prompt, 'b d t -> t b d')
            x = rearrange(x, 'b d t -> t b d')
            frame_prompt = x.clone()
        else:
            frame_prompt = rearrange(frame_prompt, 'b t d -> t b d',b=B,t=self.len_sem_prompt)

    # 다시 원래 형태 (T, B, D)로 변환
        for i, decoder in enumerate(self.decoder_transformer_for_cls):
            cls_origin = decoder(cls_origin,x)
        for i, decoder in enumerate(self.decoder_transformer_for_cls):
            cls_virtual = decoder(cls_virtual,frame_prompt)
        cls_len = cls_origin.shape[0]
        cls_origin = rearrange(cls_origin, 't b d -> b d t',b=B,t=cls_len)#! B,D,cls_len
        cls_virtual = rearrange(cls_virtual, 't b d -> b d t',b=B,t=cls_len)#! B,D,cls_len
        # token_loss =F.mse_loss(cls_origin, cls_virtual)s
        cls_origin = cls_origin.unsqueeze(-1).unsqueeze(-1)
        cls_virtual = cls_virtual.unsqueeze(-1).unsqueeze(-1)
        if self.avg_pool is not None:
            cls_origin = self.avg_pool(cls_origin)
            cls_virtual = self.avg_pool(cls_virtual)

        if self.dropout is not None:
            cls_origin = self.dropout(cls_origin)
            cls_virtual = self.dropout(cls_virtual)

        cls_origin = cls_origin.view(cls_origin.shape[0], -1)
        cls_virtual = cls_virtual.view(cls_virtual.shape[0], -1)
        
        if self.cos:
            cls_origin = F.linear(F.normalize(cls_origin, p=2, dim=-1), F.normalize(self.head.weight, p=2, dim=-1))
            cls_origin = self.cos_temp * cls_origin  # temperature set as 16
            cls_virtual = F.linear(F.normalize(cls_virtual, p=2, dim=-1), F.normalize(self.head.weight, p=2, dim=-1))
            cls_virtual = self.cos_temp * cls_virtual  # temperature set as 16
        else:
        # [N, in_channels]
            cls_origin = self.head(cls_origin)
            cls_virtual = self.head(cls_virtual)
        return (cls_origin,cls_virtual),None,None



class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''

        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        # wf = wf.transpose(0, 1)
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)


def convert_clip(state_dict):
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'), ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'), ('ln_pre', 'norm_pre'), ('ln_1', 'norm1'),('ln_2', 'norm2'),
        ('in_proj_', 'qkv.'), ('out_proj', 'proj'), ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if k.startswith('head.'):
            continue
        for sp in swaps:
            k = k.replace(sp[1], sp[0])
            if 'in_proj_' in k:
                break
        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'cls_token':
            k = 'class_embedding'
            v = v.squeeze(1).squeeze(0)
        elif k == 'positional_embedding':
            v = v.squeeze(0)
            # if v.shape[1] != model.pos_embed.shape[1]:
            #     # To resize pos embedding when using model at different size from pretrained weights
            #     v = resize_pos_embed(
            #         v,
            #         model.pos_embed,
            #         0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
            #         model.patch_embed.grid_size
            #     )
        
        out_dict[k] = v
        out_dict['ln_post.weight']= state_dict['norm.weight']
        out_dict['ln_post.bias']=state_dict['norm.bias']
    return out_dict