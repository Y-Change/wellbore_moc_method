# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.models.fno1d

1D-FNO 傅里叶神经算子基线网络：
1. 实现 1D 谱卷积算子 (SpectralConv1d)，频域复数核截断前 k_max=32 阶模式；
2. 频域全局模式提取 + 局部 1x1 跳跃卷积；
3. 学习全时程周期性波形混响与共振阻抗特征；
4. 结合完井簇连续坐标输出物理守恒流动分配与顺应性。
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1D 傅里叶谱卷积层：
    在频域与可学习复数核执行 Einstein 求和相乘，保留前 modes 阶主频模式。
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        # 1. 实数快速傅里叶变换
        x_ft = torch.fft.rfft(x, dim=-1) # (B, in_c, N//2 + 1), complex

        # 2. 复数乘法 (截断前 modes 模式)
        k_modes = min(self.modes, x_ft.shape[-1])
        w_complex = torch.view_as_complex(self.weights)[:, :, :k_modes] # (in_c, out_c, k_modes)

        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :k_modes] = torch.einsum("bix,iox->box", x_ft[:, :, :k_modes], w_complex)

        # 3. 傅里叶逆变换还原时域
        x_out = torch.fft.irfft(out_ft, n=N, dim=-1)
        return x_out


class FNOBlock1D(nn.Module):
    def __init__(self, channels: int, modes: int = 32):
        super().__init__()
        self.spectral_conv = SpectralConv1d(channels, channels, modes)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral_conv(x) + self.skip_conv(x)))


class FNO1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        cond_dim: int = 3,
        width: int = 64,
        modes: int = 32,
        max_nc: int = 6,
        cf_ref: float = 0.01,
        L: float = 5000.0,
    ):
        super().__init__()
        self.max_nc = max_nc
        self.cf_ref = cf_ref
        self.L = L

        # 1. 升维投影与下采样层 (4096 -> 1024 点，大幅加速频域谱卷积运算)
        self.lift = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )

        # 2. 4 层 1D FNO 算子块
        self.fno1 = FNOBlock1D(width, modes=modes)
        self.fno2 = FNOBlock1D(width, modes=modes)
        self.fno3 = FNOBlock1D(width, modes=modes)
        self.fno4 = FNOBlock1D(width, modes=modes)

        # 3. 降维与池化
        self.proj = nn.Sequential(
            nn.Conv1d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 4. 工况条件编码
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        # 5. 簇回归头 (128 + 32 + 5 = 165)
        self.cluster_head = nn.Sequential(
            nn.Linear(165, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def _pos_features(self, norm_pos: torch.Tensor) -> torch.Tensor:
        s = norm_pos.unsqueeze(-1)
        two_pi = 2.0 * torch.pi
        return torch.cat([
            s,
            torch.sin(two_pi * s),
            torch.cos(two_pi * s),
            torch.sin(2.0 * two_pi * s),
            torch.cos(2.0 * two_pi * s),
        ], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wave = batch["wave"]             # (B, 2, 4096)
        cond = batch["cond"]             # (B, 3)
        norm_pos = batch["norm_positions"] # (B, M)
        mask = batch["mask"]             # (B, M)
        B, M = norm_pos.shape

        # A. FNO 谱卷积前向传播
        x = self.lift(wave)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.fno3(x)
        x = self.fno4(x)
        z_fno = self.proj(x).squeeze(-1) # (B, 128)

        # B. 工况条件
        z_cond = self.cond_mlp(cond)     # (B, 32)
        z_global = torch.cat([z_fno, z_cond], dim=-1) # (B, 160)

        # C. 簇特征融合
        z_global_expanded = z_global.unsqueeze(1).expand(-1, M, -1)
        pos_feat = self._pos_features(norm_pos)
        cluster_in = torch.cat([z_global_expanded, pos_feat], dim=-1)

        cluster_out = self.cluster_head(cluster_in)
        logits_alpha = cluster_out[..., 0]
        pred_log_cf = cluster_out[..., 1]

        # D. Masked Softmax 物理守恒
        masked_logits = logits_alpha.masked_fill(~mask, -1e9)
        pred_alpha = F.softmax(masked_logits, dim=-1)
        pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))

        # E. 顺应性
        pred_cf = self.cf_ref * torch.exp(pred_log_cf)
        pred_cf = torch.where(mask, pred_cf, torch.zeros_like(pred_cf))
        pred_log_cf = torch.where(mask, pred_log_cf, torch.zeros_like(pred_log_cf))

        # F. 连续重构场
        pos_m = norm_pos * self.L
        grid_x = torch.linspace(0, self.L, 500, device=wave.device)
        sigma = 20.0
        diff = grid_x.unsqueeze(0).unsqueeze(0) - pos_m.unsqueeze(-1)
        gauss = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * torch.exp(-(diff ** 2) / (2.0 * sigma ** 2))
        m_grid = (pred_alpha.unsqueeze(-1) * gauss).sum(dim=1)
        dx = self.L / 499.0
        m_grid = m_grid / (m_grid.sum(dim=-1, keepdim=True) * dx).clamp(min=1e-6)

        return {
            "alpha": pred_alpha,
            "cf": pred_cf,
            "log_cf": pred_log_cf,
            "m_alpha_grid": m_grid,
        }
