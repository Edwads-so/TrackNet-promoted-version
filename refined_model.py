import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


# ==========================================
# 第一部分：轻量化基础算子 (针对原 Double2DConv/Triple2DConv 的改良)
# 改动点：引入深度可分离卷积 (DS-Conv)，显著降低参数量提升 FPS
# ==========================================
class DSConv2DBlock(nn.Module):
    """ 深度可分离卷积：Depthwise + Pointwise，比标准卷积减少约 8-9 倍计算量 """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise: 空间特征提取
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            # Pointwise: 通道融合
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LightweightDoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            DSConv2DBlock(in_dim, out_dim),
            DSConv2DBlock(out_dim, out_dim)
        )

    def forward(self, x):
        return self.block(x)


# ==========================================
# 第二部分：自适应帧差模块 (Motion Enhancement Module)
# 改动点：新增模块，解决原模型 TP/FP 表现变差的问题。
# 原理：通过显式计算帧间差异，过滤静止背景噪声。
# ==========================================
class AdaptiveMotionModule(nn.Module):
    def __init__(self, in_channels=9):  # 假设输入是 3 帧堆叠 (3x3=9)
        super().__init__()
        self.compress = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # 处理两个差分图
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 假设 x 形状为 (B, 9, H, W)，分为 f1, f2, f3
        f1, f2, f3 = torch.split(x, 3, dim=1)
        diff1 = torch.abs(f2 - f1)
        diff2 = torch.abs(f3 - f2)
        diff_concat = torch.cat([diff1, diff2], dim=1)  # (B, 6, H, W)

        motion_feat = self.compress(diff_concat)
        weight = self.weight_gen(motion_feat).view(-1, 1, 1, 1)
        return motion_feat * weight  # 输出增强的运动线索


# ==========================================
# 第三部分：改良版 MaSA (针对 MaSA_model.py 的改良)
# 改动点：移除过于粗暴的 mean(dim=-1) 全局平均，改为保留空间结构的注意力。
# 解决：原 MaSA 导致小目标（羽毛球）特征被稀释的问题。
# ==========================================
class ImprovedMaSA(nn.Module):
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


# ==========================================
# 第四部分：集成 TrackNet
# 结合了自适应帧差 + 轻量化编码器 + 改良 MaSA
# ==========================================
class RefinedTrackNet(nn.Module):
    def __init__(self, in_dim=9, out_dim=3):
        super().__init__()
        # 1. 运动增强入口
        self.motion_module = AdaptiveMotionModule(in_dim)

        # 2. 轻量化编码器 (DSConv 替代标准 Conv)
        self.down1 = LightweightDoubleConv(in_dim, 64)
        self.down2 = LightweightDoubleConv(64, 128)
        self.down3 = LightweightDoubleConv(128, 256)

        # 3. Bottleneck (MaSA 改良)
        self.bottleneck_proj = nn.Conv2d(256 + 16, 512, 1)  # 融合运动特征后的维度
        self.masa_block = ImprovedMaSA(512, num_heads=8)

        # 4. 解码器 (保持跳跃连接)
        self.up1 = LightweightDoubleConv(512 + 256, 256)
        self.up2 = LightweightDoubleConv(256 + 128, 128)
        self.up3 = LightweightDoubleConv(128 + 64, 64)

        self.predictor = nn.Sequential(
            nn.Conv2d(64, out_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取显式运动特征
        m_feat = self.motion_module(x)  # (B, 16, H, W)

        # Encoder
        x1 = self.down1(x)  # (B, 64, H, W)
        p1 = F.max_pool2d(x1, 2)  # (B, 64, H/2, W/2)
        x2 = self.down2(p1)  # (B, 128, H/2, W/2)
        p2 = F.max_pool2d(x2, 2)  # (B, 128, H/4, W/4)
        x3 = self.down3(p2)  # (B, 256, H/4, W/4)
        p3 = F.max_pool2d(x3, 2)  # (B, 256, H/8, W/8)

        # 融合运动线索 (将运动特征下采样到 bottleneck 尺度)
        m_feat_down = F.adaptive_avg_pool2d(m_feat, (p3.shape[2], p3.shape[3]))
        b_in = torch.cat([p3, m_feat_down], dim=1)

        # Bottleneck + MaSA
        b_feat = self.bottleneck_proj(b_in)
        b_feat = self.masa_block(b_feat)

        # Decoder
        u1 = torch.cat([F.interpolate(b_feat, scale_factor=2), x3], dim=1)
        d1 = self.up1(u1)
        u2 = torch.cat([F.interpolate(d1, scale_factor=2), x2], dim=1)
        d2 = self.up2(u2)
        u3 = torch.cat([F.interpolate(d2, scale_factor=2), x1], dim=1)
        d3 = self.up3(u3)

        return self.predictor(d3)