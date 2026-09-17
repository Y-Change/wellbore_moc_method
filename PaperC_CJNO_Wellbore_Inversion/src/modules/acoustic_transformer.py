# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.modules.acoustic_transformer

声学时延偏置簇间解耦注意力网络 (Acoustic-Biased Decoupling Transformer):
针对密集多簇相邻水击反射波相互交织干涉的难题，显式融入声学双程传播时差物理偏置:
A_{ij} = Softmax( (q_i * k_j^T) / sqrt(d) - gamma * |x_i - x_j| / a )
其中 gamma >= 0 为可学习物理距离衰减惩罚系数，强制自注意力抑制声学因果无关/远距虚假相关簇，
实现逐簇几何解耦与精确流量份额/水力顺应性表征重构。
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        two_pi = 2.0 * torch.pi
        scaled = two_pi * x * self.freqs
        sin_f = torch.sin(scaled)
        cos_f = torch.cos(scaled)
        return torch.cat([x, sin_f, cos_f], dim=-1)


class AcousticBiasedMultiheadAttention(nn.Module):
    """
    带声学时延物理偏置的多头注意力层:
    Bias_{ij} = - gamma * |x_i - x_j| / a
    """
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_acoustic_bias: bool = True,
        init_gamma: float = 10.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) 必须被 n_heads ({n_heads}) 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / np.sqrt(self.d_k)
        self.use_acoustic_bias = use_acoustic_bias

        # 线性投影矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 可学习物理偏置系数 gamma >= 0 (通过 softplus 保证非负)
        # softplus 逆变换初始化，默认 init_gamma = 10.0 s^-1 (特征尺度对齐水击脉冲持续期 ~50ms)
        init_raw = np.log(np.exp(init_gamma) - 1.0 + 1e-6)
        self.raw_gamma = nn.Parameter(torch.tensor(float(init_raw), dtype=torch.float32))

    def get_gamma(self) -> torch.Tensor:
        """获取物理惩罚系数 gamma >= 0"""
        return F.softplus(self.raw_gamma)

    def forward(
        self,
        x: torch.Tensor,                                # (B, M, d_model)
        positions: torch.Tensor,                        # (B, M) [m]
        wavespeed: torch.Tensor,                        # (B, 1) [m/s]
        mask: Optional[torch.Tensor] = None,            # (B, M) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            out: (B, M, d_model)
            attn_weights: (B, n_heads, M, M)
        """
        B, M, _ = x.shape

        # 1. 计算 Q, K, V
        q = self.w_q(x).view(B, M, self.n_heads, self.d_k).transpose(1, 2) # (B, H, M, d_k)
        k = self.w_k(x).view(B, M, self.n_heads, self.d_k).transpose(1, 2) # (B, H, M, d_k)
        v = self.w_v(x).view(B, M, self.n_heads, self.d_k).transpose(1, 2) # (B, H, M, d_k)

        # 2. 点乘自注意力得分: (B, H, M, M)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. 显式叠加声学时延偏置: - gamma * |x_i - x_j| / a
        if self.use_acoustic_bias:
            # pos_diff: (B, M, M)
            pos_diff = torch.abs(positions.unsqueeze(2) - positions.unsqueeze(1)) # (B, M, M)
            # a: (B, 1, 1)
            a_exp = wavespeed.view(B, 1, 1).clamp(min=500.0)
            tau_diff = pos_diff / a_exp # (B, M, M) 声学传播时差 (s)

            gamma = self.get_gamma()
            bias = - gamma * tau_diff # (B, M, M)
            bias = bias.unsqueeze(1)  # (B, 1, M, M)
            scores = scores + bias

        # 4. 掩码处理 (Key Mask: 将未激活簇的得分压至极小)
        if mask is not None:
            # key_mask: (B, 1, 1, M)
            key_mask = mask.view(B, 1, 1, M)
            scores = scores.masked_fill(~key_mask, -1e9)

        # 5. Softmax 归一化与 Dropout
        attn_weights = F.softmax(scores, dim=-1) # (B, H, M, M)
        # 防止整行全为 mask 时的数值 NaN
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        # 清理未激活 Query 簇的行权重
        if mask is not None:
            query_mask = mask.view(B, 1, M, 1)
            attn_weights = torch.where(query_mask, attn_weights, torch.zeros_like(attn_weights))
        attn_weights = self.dropout(attn_weights)

        # 6. 加权上下文与输出投影
        out = torch.matmul(attn_weights, v) # (B, H, M, d_k)
        out = out.transpose(1, 2).contiguous().view(B, M, self.d_model)
        out = self.w_out(out)

        # 7. 屏蔽未激活簇
        if mask is not None:
            out = torch.where(mask.unsqueeze(-1), out, torch.zeros_like(out))

        return out, attn_weights


class AcousticBiasedTransformerLayer(nn.Module):
    """单层声学时延偏置 Transformer 编码块 (Pre-LN 架构)"""
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        use_acoustic_bias: bool = True,
        init_gamma: float = 10.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AcousticBiasedMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_acoustic_bias=use_acoustic_bias,
            init_gamma=init_gamma,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        wavespeed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN Self-Attention
        x_norm = self.norm1(x)
        attn_out, weights = self.attn(x_norm, positions, wavespeed, mask=mask)
        x = x + attn_out

        # Pre-LN FFN
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x, weights


class AcousticBiasedTransformer(nn.Module):
    """
    声学时延偏置 Transformer 编码器主体:
    1. 坐标位置傅里叶编码 (Fourier Positional Encoding) 融合;
    2. 全局工况条件注入 (Condition FiLM/Add);
    3. 多层 AcousticBiasedTransformerLayer 堆叠;
    4. 输出解耦簇表征 H in R^{B, M, d_model} 与各层注意力权重 maps。
    """
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        cond_dim: int = 3,
        num_pos_freqs: int = 8,
        dropout: float = 0.0,
        use_acoustic_bias: bool = True,
        init_gamma: float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_acoustic_bias = use_acoustic_bias

        # 射孔簇坐标高频傅里叶嵌入
        self.pos_embed = FourierPositionalEncoding(num_freqs=num_pos_freqs)
        self.pos_proj = nn.Linear(2 * num_pos_freqs + 1, d_model)

        # 全局工况条件 MLP 投影
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 融合前 LayerNorm
        self.in_norm = nn.LayerNorm(d_model)

        # 编码层堆叠
        self.layers = nn.ModuleList([
            AcousticBiasedTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_acoustic_bias=use_acoustic_bias,
                init_gamma=init_gamma,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        cluster_tokens: torch.Tensor,       # (B, M, d_model) 来自到时提取
        positions: torch.Tensor,            # (B, M) [m]
        norm_positions: torch.Tensor,       # (B, M) [0, 1]
        cond: torch.Tensor,                 # (B, cond_dim)
        mask: Optional[torch.Tensor] = None,# (B, M) bool
        wavespeed: Optional[torch.Tensor] = None,  # (B,) or (B, 1) [m/s]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            h: (B, M, d_model) 解耦簇表征
            attn_maps: (B, n_layers, n_heads, M, M) 注意力热力图

        wavespeed 若给定则直接用于声学距离偏置；否则回退到旧 Pilot
        约定 cond[:, 1] = (a - 1450) / 20。
        """
        B, M, _ = cluster_tokens.shape

        # 1. 声速解算
        if wavespeed is None:
            wavespeed = cond[:, 1:2] * 20.0 + 1450.0  # (B, 1)
        else:
            wavespeed = wavespeed.reshape(B, 1).to(dtype=cluster_tokens.dtype)

        # 2. 坐标编码与工况注入
        pos_feat = self.pos_proj(self.pos_embed(norm_positions.unsqueeze(-1))) # (B, M, d_model)
        cond_feat = self.cond_proj(cond).unsqueeze(1)                          # (B, 1, d_model)

        h = cluster_tokens + pos_feat + cond_feat
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), h, torch.zeros_like(h))
        h = self.in_norm(h)

        # 3. 逐层前向传播
        attn_maps = []
        for layer in self.layers:
            h, weights = layer(h, positions, wavespeed, mask=mask)
            attn_maps.append(weights)

        # 堆叠所有层注意力权重: (B, n_layers, n_heads, M, M)
        all_attn = torch.stack(attn_maps, dim=1)
        return h, all_attn
