import torch.nn as nn
import torch

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class SparseAttention(nn.Module):
    def __init__(self, attn, k=0):
        super().__init__()
        self.k = k
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask_value = max_neg_value(attn)
        if self.k > 0:
            top, _ = attn.topk(self.k, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(attn)
            mask = attn < vk
            attn.masked_fill_(mask, mask_value)
            del mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual
    
class GRUGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)
    
