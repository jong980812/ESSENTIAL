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
class Decoder_ResidualAttentionBlock_time(nn.Module):
    def __init__(self, temp_mode:str,d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0.2,dim_mlp=192,fs_topk=8):
        super().__init__()
        self.temp_mode = temp_mode
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fs_topk = fs_topk
        if self.temp_mode=='transformer':
            d_model = 768
            n_head = 12
            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]))
            self.ln_1 = LayerNorm(d_model)
            self.ln_2 = LayerNorm(d_model)
            self.ln_cls = LayerNorm(d_model)
            self.attn_mask = attn_mask
        elif self.temp_mode=='attention':
            d_model = 768
            n_head = 12
            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.attn_mask = attn_mask
            self.ln_1 = LayerNorm(d_model)
            self.ln_cls = LayerNorm(d_model)
        elif self.temp_mode =='ba':
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
        #입력 cls_token B,T,D
        B = x.shape[1]# X: T,B,D
        # cls = self.decoder_cls.expand(B,-1).unsqueeze(1) # B,1,D
        if self.temp_mode=='transformer':
            ln_cls = self.ln_cls(cls)
            ln1 = self.ln_1(x)
            if get_frame:
                return self.attention(ln_cls,ln1,need_weights=True)
            cls = cls + self.drop_path(self.attention(ln_cls,ln1))
            cls = cls + self.drop_path(self.mlp(self.ln_2(cls)))
        elif self.temp_mode=='attention':
            ln_cls = self.ln_cls(cls)
            ln1 = self.ln_1(x)
            cls = cls + self.drop_path(self.attention(ln_cls,ln1))
            if get_frame:
                attention_map = self.attention(ln_cls,ln1,need_weights=True)
                topk_index = attention_map.topk(self.fs_topk,-1).indices.sort().values
                top_section = attention_map>(1/x.shape[0])
                top_section = torch.nonzero(top_section[0,0], as_tuple=True)[0]
                first_true_index = top_section[0].item()
                last_true_index = top_section[-1].item()
                frame_index = torch.linspace(first_true_index, last_true_index,8).long().unsqueeze(0).unsqueeze(0)
                return frame_index,cls
        elif self.temp_mode =='ba':
            ln_cls = self.ln_cls(self.time_down(cls))
            ln1 = self.ln_1(self.kv_down(x))
            x_cls= self.time_act(self.attention(ln_cls,ln1))
            cls = self.time_up(x_cls)+cls
        return cls

class AIM_base_decoder(nn.Module):
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
        self.adapter_layers = adapter_layers
        self.num_frames = num_frames
        self.decoder_temporal_embedding = nn.Parameter(torch.zeros(1, num_frames+1, width))
        self.use_aim_weight = args.use_aim_weight
        if args.use_aim_weight:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
            
        self.order = args.order
        if self.order:
            self.temp_head = nn.Linear(width, num_frames)
            trunc_normal_(self.temp_head.weight, std=.02)
            self.temp_head.weight.data.mul_(init_scale)
            self.temp_head.bias.data.mul_(init_scale)
        self.embed_dim = 768
        self.ba_layers =args.ba_layers
        self.fs_topk = args.fs_topk
        self.n_token_rehearsal = args.n_token_rehearsal
        self.handcrafted_selection = args.handcrafted_selection
        self.selected_selection = args.selected_selection
        self.fs_density = args.fs_density
        self.cls_aug = args.cls_aug
        if self.cls_aug:self.aug_adapter = Adapter(width,192) 
        if self.order:
            self.temp_head = nn.Linear(self.embed_dim, num_frames)
            trunc_normal_(self.temp_head.weight, std=.02)
            self.temp_head.weight.data.mul_(init_scale)
            self.temp_head.bias.data.mul_(init_scale)
        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=2 if args.data_set=='SSV2' else 1, scale=adapter_scale, drop_path=drop_path_rate,dim_mlp=dim_mlp,adapter_layers=self.adapter_layers)
        # self.transformer_for_cls = Decoder_ResidualAttentionBlock_time(width, heads, None,0., num_tadapter, num_frames, drop_path=drop_path_rate,dim_mlp=dim_mlp)
        self.decoder_cls = nn.Parameter(scale * torch.randn(width))
        self.decoder_transformer_for_cls = nn.Sequential(*[Decoder_ResidualAttentionBlock_time(args.temp_mode, width, args.ba_heads, None,0.2, num_tadapter, num_frames, drop_path=drop_path_rate,dim_mlp=dim_mlp,fs_topk=self.fs_topk) for _ in range(args.ba_layers)])
        self.ln_post = LayerNorm(width)
        self.cos = args.cos
        self.mse = torch.nn.MSELoss()
        
        #!!
        self.each_head = args.each_head
        if self.each_head:
            self.head = nn.ModuleList()
            for mask in class_mask:
                n_class = len(mask)
                head= nn.Linear(self.embed_dim,n_class)
                trunc_normal_(head.weight, std=.02)
                self.head.append(head)
            self.init_weights(pretrained='clip')
            for head in self.head:
                head.weight.data.mul_(init_scale)
                head.bias.data.mul_(init_scale)
        else:
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

    def forward(self, x: torch.Tensor, train=False,task_id =-1,get_frame=False):
            
        # x = x[:,:,3,:,:].unsqueeze(2)#! single frame
        if len(x.shape)==4:#! 이미지 입력 들어왔을 떄 대비
            x = x.unsqueeze(2)
        B, C, T, H, W = x.shape 
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        #! Add classification token-> 각 프레임당 1개씩 ex) (8*10), 196+1, 768 
        x = x + self.positional_embedding.to(x.dtype) #! Positional embedding, (8*10), 197, 768
        if self.use_aim_weight:#! AIM은 temporal embedding들어옴.
            temporal_embedding=self.temporal_embedding 
            if self.cls_aug:
                temporal_embedding = torch.cat([temporal_embedding,temporal_embedding],1)
            elif temporal_embedding.shape[1]!=(T):
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
        
        '''
        x_norm = x / x.norm(dim=1, keepdim=True)
        
         코사인 유사도 맵 계산
        cosine_similarity_map = torch.matmul(x_norm, x_norm.transpose(0, 1))
        '''
        # 
        # if T<8:
        #     x = torch.repeat_interleave(x, 8//T, dim=1)
        #     T=8
        decoder_temporal_embedding=self.decoder_temporal_embedding 
        if decoder_temporal_embedding.shape[1]!=(T+1):
            decoder_temporal_embedding = F.interpolate(
                decoder_temporal_embedding.unsqueeze(1), size=(T+1,768), mode='bilinear', align_corners=False
            ).squeeze(1)
        # #!
        # if get_frame:
        #     decoder_temporal_embedding = F.interpolate(
        #         decoder_temporal_embedding.unsqueeze(1), size=(T+1,768), mode='bilinear', align_corners=False
        #     ).squeeze(1)
    
        # #!
        '''
        x는 원래 CLIP으로 부터 나온 CLS 토큰들
        cls는 decoder를 위한 새로운 CLS token. 
        '''
        cls = self.decoder_cls.expand(B,-1).unsqueeze(1) # B,1,D
        cls_and_x = torch.cat([cls,x],1)# B, T+1, D
        cls_and_x = cls_and_x + decoder_temporal_embedding
        cls_and_x = rearrange(cls_and_x, 'b t d -> t b d',b=B,t=T+1)
        cls,x = cls_and_x[0,:,:],cls_and_x[1:,:,:]
        cls = cls.unsqueeze(0)
        if get_frame:
            for i, decoder in enumerate(self.decoder_transformer_for_cls):
                if i < (self.ba_layers-1):
                    cls = decoder(cls,x)
                else:
                    frame_index,cls = decoder(cls,x,get_frame)
            if self.fs_density:
                density = calculate_density(frame_index,T)
                if density>30.0:
                    new_frame_index = expand_indices_around_center(frame_index,T, density/30.0)
                    # print(f'Index: {frame_index},New Index : {new_frame_index}, T:{T},Den:{density}')
                    frame_index = new_frame_index
                # elif self.selected_selection:
                    # pass
                else:
                    average_duration = T // 8
                    frame_index = torch.tensor(list(np.multiply(list(range(8)), average_duration)),dtype=torch.int32).unsqueeze(0).unsqueeze(0)#torch.tensor([int(i) for i in range(8)],dtype=torch.int32).unsqueeze(0).unsqueeze(0)
            if self.handcrafted_selection:
                # str_idx = int(T * 1/3)
                # end_idx = int(T * 2/3)
                # frame_index = torch.tensor([str_idx, end_idx], dtype=torch.int32).unsqueeze(0).unsqueeze(0)
                average_duration = T // 8
                uniform_index = np.multiply(list(range(8)), average_duration)
                frame_index = torch.tensor(list(np.sort(np.random.choice(uniform_index,2,False))),dtype = torch.int32).unsqueeze(0).unsqueeze(0)

                # frame_index = torch.tensor([uniform_index[0,0,2], uniform_index[0,0,5]], dtype=torch.int32).unsqueeze(0).unsqueeze(0)
                
                # frame_index 텐서를 생성합니다.
            elif self.selected_selection:
                selected_indices = torch.randperm(frame_index.size(2))[:4]

                # 선택된 인덱스를 사용하여 정렬된 텐서에서 값을 선택
                frame_index = torch.sort(torch.index_select(frame_index, dim=2, index=selected_indices),dim=2)[0]
                # frame_index = torch.tensor([frame_index[0,0,2], frame_index[0,0,5]], dtype=torch.int32).unsqueeze(0).unsqueeze(0)
            
            #!
            # cls_len = cls.shape[0]
            # cls = rearrange(cls, 't b d -> b d t',b=B,t=cls_len)#! B,D,cls_len
            # #
            # cls = cls.unsqueeze(-1).unsqueeze(-1)
            
            # if self.avg_pool is not None:
            #     cls = self.avg_pool(cls)
            # # [N, in_channels, 1, 1, 1]
            # if self.dropout is not None:
            #     cls = self.dropout(cls)
            # # [N, in_channels, 1, 1, 1]
            # cls = cls.view(cls.shape[0], -1)
            # logit = self.head(cls)
            #!
            # indices = frame_index[0, 0]
            # gaps = indices[1:] - indices[:-1]
            # average_gap = gaps.float().mean()
            # density = T / average_gap
            return frame_index,T,_
        else:
            for i, decoder in enumerate(self.decoder_transformer_for_cls):
                cls = decoder(cls,x)
            
        cls_len = cls.shape[0]
        cls = rearrange(cls, 't b d -> b d t',b=B,t=cls_len)#! B,D,cls_len
        x_final = rearrange(x,'t b d -> b t d',b=B,t=T)
        
        # x_final = rearrange(cls,'b d t -> b t d',b=B,t=)
        #
        cls = cls.unsqueeze(-1).unsqueeze(-1)
        
        if self.avg_pool is not None:
            cls = self.avg_pool(cls)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            cls = self.dropout(cls)
        # [N, in_channels, 1, 1, 1]
        cls = cls.view(cls.shape[0], -1)
        
        if self.cos:
            cls = F.linear(F.normalize(cls, p=2, dim=-1), F.normalize(self.head.weight, p=2, dim=-1))
            cls = self.cos_temp * cls  # temperature set as 16
        else:
        # [N, in_channels]
            cls = self.head(cls)
        x_final = (self.temp_head(x_final)) if self.order else None
        return cls,x_final
    
def adjust_norm(input_tensor, ref_tensor):
    # input_tensor와 ref_tensor의 norm 계산
    input_norm = input_tensor.norm(p=2, dim=0, keepdim=True)
    ref_norm = ref_tensor.norm(p=2, dim=0, keepdim=True)

    # ref_tensor의 평균 norm 계산
    ref_mean_norm = ref_norm.mean()
    
    # input_tensor의 각 열을 조정하여 ref_mean_norm과 비슷하게 만듦
    adjusted_tensor = input_tensor * (ref_mean_norm / input_norm)
    
    return adjusted_tensor


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



def expand_indices_around_center(frame_indices, T, expand_factor):
    indices = frame_indices[0, 0].cpu()
    first, last = indices[0], indices[-1]
    length = len(indices)
    # 첫 인덱스가 시작에 가까운지 확인
    if first < T * 0.1:
        # 첫 인덱스가 전체의 10% 이내일 경우: 뒤로 확장
        offsets = torch.arange(length).float() * expand_factor
        new_indices = indices.float() + offsets
    # 마지막 인덱스가 끝에 가까운지 확인
    elif last > T * 0.9:
        # 마지막 인덱스가 전체의 90% 이상일 경우: 앞으로 확장
        offsets = torch.arange(length).float() * expand_factor
        new_indices = indices.float() - offsets.flip(0)  # 뒤집힌 순서로 감소
    else:
        # 중심점을 기준으로 확장
        center = indices.float().mean()
        offsets = (torch.arange(length) - length // 2).float() * expand_factor
        new_indices = (indices - center) + offsets + center

    # 결과를 다시 정수로 변환하고, 0과 T-1 범위 내로 제한
    new_indices = new_indices.round().int()
    new_indices = torch.clamp(new_indices, 0, T-1)
    return new_indices.unsqueeze(0).unsqueeze(0)
def calculate_density(frame_indices, T):
    indices = frame_indices[0, 0]
    gaps = indices[1:] - indices[:-1]
    average_gap = gaps.float().mean()
    density = T / average_gap
    return density.item()