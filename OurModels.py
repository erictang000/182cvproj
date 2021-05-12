import timm
import torch.nn as nn
import torch
from torchvision import transforms

from torch import einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


##From CrossViT Architecture
class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)



        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
##Also from CrossViT    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

##From ViT
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

##Our Architecture
class DoubleViT(nn.Module):
    def __init__(self, image_size, dropout=0.4, cross_attn_heads=2, small_dim_head=64, large_dim_head=32, cross_dropout = 0, cross_depth=-3, num_cross_attn=1):
        super().__init__()
        self.cross_depth = cross_depth
        self.im_size = image_size
        self.model_s = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.model_s.blocks = nn.ModuleList([self.model_s.blocks[i] for i in range(len(self.model_s.blocks))])
        self.model_l = timm.create_model("vit_base_patch32_224_in21k", pretrained=True)
        self.model_l.blocks = nn.ModuleList([self.model_l.blocks[i] for i in range(len(self.model_l.blocks))])

        self.dropout = nn.Dropout(dropout)
        self.lin_in_features = self.model_l.head.in_features + self.model_s.head.in_features
        self.linear1 = nn.Linear(self.lin_in_features, self.lin_in_features // 2)
        self.linear2 = nn.Linear(self.lin_in_features // 2, 200)
        self.head = nn.Sequential(
            self.dropout, 
            self.linear1,
            nn.ReLU(), 
            self.linear2
        )
        small_dim = self.model_s.blocks[-2].mlp.fc2.out_features
        large_dim = self.model_l.blocks[-2].mlp.fc2.out_features
        self.cross_attn_layers = nn.ModuleList([])
        for i in range(num_cross_attn):
            self.cross_attn_layers.append(nn.ModuleList([
                            nn.Linear(small_dim,large_dim),
                            nn.Linear(large_dim, small_dim),
                            PreNorm(large_dim, CrossAttention(large_dim, heads = cross_attn_heads, dim_head = large_dim_head, dropout = cross_dropout)),
                            nn.Linear(large_dim, small_dim),
                            nn.Linear(small_dim, large_dim),
                            PreNorm(small_dim, CrossAttention(small_dim, heads = cross_attn_heads, dim_head = small_dim_head, dropout = cross_dropout)),
                        ]))
        
    ##Adapted from ViT code (ross wrightman's timm repo)
    def forward_features(self, x, model):
        x = model.patch_embed(x)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        for block in model.blocks[:self.cross_depth]:
            x = block(x)
        return x
    
    def finish_forward_features(self, x, model):
        for block in model.blocks[self.cross_depth:]:
            x = block(x)
        return x
    
    def get_head_inputs(self, x, model):
        x = model.norm(x)
        return model.pre_logits(x[:, 0])
    
    
    def forward(self, x):
        s_out = self.forward_features(x, self.model_s)
        l_out = self.forward_features(x, self.model_l)

        ##Cross Attention code from CrossViT
        f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l = self.cross_attn_layers
    
        small_class = s_out[:, 0]
        x_small = s_out[:, 1:]
        large_class = l_out[:, 0]
        x_large = l_out[:, 1:]

        # Cross Attn for Large Patch

        cal_q = f_ls(large_class.unsqueeze(1))
        cal_qkv = torch.cat((cal_q, x_small), dim=1)
        cal_out = cal_q + cross_attn_l(cal_qkv)
        cal_out = g_sl(cal_out)
        s_out = torch.cat((cal_out, x_large), dim=1)

        # Cross Attn for Smaller Patch
        cal_q = f_sl(small_class.unsqueeze(1))
        cal_qkv = torch.cat((cal_q, x_large), dim=1)
        cal_out = cal_q + cross_attn_s(cal_qkv)
        cal_out = g_ls(cal_out)
        l_out = torch.cat((cal_out, x_small), dim=1)
        
        
        s_out = self.finish_forward_features(s_out, self.model_s)
        l_out = self.finish_forward_features(l_out, self.model_l)
        
        if len(self.cross_attn_layers) > 1:
            f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l = self.cross_attn_layers[1]

            small_class = s_out[:, 0]
            x_small = s_out[:, 1:]
            large_class = l_out[:, 0]
            x_large = l_out[:, 1:]

            # Cross Attn for Large Patch

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            s_out = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            l_out = torch.cat((cal_out, x_small), dim=1)

        s_out = self.get_head_inputs(s_out, self.model_s)
        l_out = self.get_head_inputs(l_out, self.model_l)
        
        out = torch.cat((s_out, l_out), dim = -1)
        return self.head(out)
    
    
def double_vit_cross_original():
    return DoubleViT(224, small_dim_head = 64, large_dim_head = 32, cross_attn_heads = 2, num_cross_attn=1)

def double_vit_cross_new():
    return DoubleViT(224, small_dim_head = 32, large_dim_head = 64, cross_attn_heads = 4, num_cross_attn=2 )