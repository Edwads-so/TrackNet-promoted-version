import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models import register_model
# from timm.models.vision_transformer import _cfg
from typing import Tuple, Union
from functools import partial

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class RelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
        mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  # (n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:

            retention_rel_pos = self.decay.exp()

        elif chunkwise_recurrent:
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = (mask_h, mask_w)

        else:
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)
            retention_rel_pos = mask

        return retention_rel_pos


class MaSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(embed_dim, embed_dim * 3, 1)

        # 新增：局部空间注意力补偿，确保小目标不丢失
        self.local_suppress = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 3, padding=1, groups=embed_dim // 4),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        qkv = self.qkv(x).view(B, 3, self.num_heads, self.head_dim, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # 改良：不再做 mean 压缩，使用点积注意力并保持分辨率
        # 简化版的高效空间自注意力
        attn = (q * k).sum(dim=2, keepdim=True) * self.scaling  # (B, heads, 1, H, W)
        attn = torch.softmax(attn, dim=-1)

        out = attn * v  # (B, heads, d, H, W)
        out = out.view(B, C, H, W)

        # 结合局部增强
        local_mask = self.local_suppress(x)
        out = out * local_mask

        return self.proj(out)

class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn=F.gelu,
            dropout=0.0,
            activation_dropout=0.0,
            layernorm_eps=1e-6,
            subln=False,
            subconv=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class MaSA_Block(nn.Module):
    """
    封装 MaSA 注意力和 FFN 的完整块，处理 TrackNet 所需的 (B C H W) 转换。
    """

    def __init__(self, dim, num_heads, ffn_dim, drop_path=0.0):
        super().__init__()

        self.attn = MaSA(dim, num_heads)
        self.rel_pos_generator = RelPos2d(dim, num_heads, initial_value=2.0, heads_range=3.0)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim, ffn_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x_in: torch.Tensor):
        B, C, H, W = x_in.shape

        # 1. 形状转换 (B C H W -> B H W C)
        x = x_in.permute(0, 2, 3, 1)

        # 2. 生成 rel_pos
        rel_pos = self.rel_pos_generator(slen=(H, W), chunkwise_recurrent=False)

        # 3. MaSA Attention
        x_norm1 = self.norm1(x)
        attn_output = self.attn(x_norm1, rel_pos)
        x = x + self.drop_path(attn_output)  # 残差连接

        # 4. FFN
        x_norm2 = self.norm2(x)
        x = x + self.drop_path(self.ffn(x_norm2))  # 残差连接

        # 5. 形状转换 (B H W C -> B C H W)
        x_out = x.permute(0, 3, 1, 2)
        return x_out