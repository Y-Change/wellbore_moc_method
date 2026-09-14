# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.models.resnet1d

1D-ResNet 深度残差时序卷积基线网络：
1. 采用多尺度 1D 残差卷积块提取高频水击波形与波前跳变特征；
2. 融合井口激励工况元数据 (tc, a, Hext)；
3. 结合射孔簇连续坐标傅里叶嵌入，经过变长簇头回归流动分配份额与水力顺应性；
4. Masked Softmax 保证物理守恒单纯形约束。
"""
from __future__ import annotations

import math
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + res)
        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        cond_dim: int = 3,
        base_channels: int = 32,
        latent_dim: int = 128,
        max_nc: int = 6,
        cf_ref: float = 0.01,
        L: float = 5000.0,
    ):
        super().__init__()
        self.max_nc = max_nc
        self.cf_ref = cf_ref
        self.L = L

        # 1. 1D 时序卷积残差主干
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.layer1 = nn.Sequential(
            ResBlock1D(c1),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.layer2 = nn.Sequential(
            ResBlock1D(c2),
            nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.layer3 = nn.Sequential(
            ResBlock1D(c3),
            nn.Conv1d(c3, latent_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # 2. 工况条件 MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # 3. 簇坐标位置傅里叶嵌入与回归头
        # pos_feat: s, sin(2pi*s), cos(2pi*s), sin(4pi*s), cos(4pi*s) (5 dims)
        cluster_in_dim = latent_dim + 32 + 5
        self.cluster_head = nn.Sequential(
            nn.Linear(cluster_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # (logit_alpha, log_cf)
        )

    def _pos_features(self, norm_pos: torch.Tensor) -> torch.Tensor:
        """norm_pos: (B, M) -> (B, M, 5)"""
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

        # A. 主干特征
        x = self.stem(wave)
        x = self.layer1(x)
        x = self.layer2(x)
        z_wave = self.layer3(x).squeeze(-1) # (B, latent_dim)

        # B. 工况特征
        z_cond = self.cond_mlp(cond)        # (B, 32)
        z_global = torch.cat([z_wave, z_cond], dim=-1) # (B, latent_dim + 32)

        # C. 簇特征融合
        z_global_expanded = z_global.unsqueeze(1).expand(-1, M, -1) # (B, M, D)
        pos_feat = self._pos_features(norm_pos)                      # (B, M, 5)
        cluster_in = torch.cat([z_global_expanded, pos_feat], dim=-1) # (B, M, D+5)

        cluster_out = self.cluster_head(cluster_in) # (B, M, 2)
        logits_alpha = cluster_out[..., 0]          # (B, M)
        pred_log_cf = cluster_out[..., 1]           # (B, M)

        # D. 严格物理守恒 Masked Softmax
        masked_logits = logits_alpha.masked_fill(~mask, -1e9)
        pred_alpha = F.softmax(masked_logits, dim=-1)
        pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))

        # E. 顺应性恢复
        pred_cf = self.cf_ref * torch.exp(pred_log_cf)
        pred_cf = torch.where(mask, pred_cf, torch.zeros_like(pred_cf))
        pred_log_cf = torch.where(mask, pred_log_cf, torch.zeros_like(pred_log_cf))

        # F. 连续重构场 (高斯加权)
        pos_m = norm_pos * self.L
        grid_x = torch.linspace(0, self.L, 500, device=wave.device) # (500,)
        sigma = 20.0
        diff = grid_x.unsqueeze(0).unsqueeze(0) - pos_m.unsqueeze(-1) # (B, M, 500)
        gauss = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * torch.exp(-(diff ** 2) / (2.0 * sigma ** 2))
        m_grid = (pred_alpha.unsqueeze(-1) * gauss).sum(dim=1) # (B, 500)
        dx = self.L / 499.0
        m_grid = m_grid / (m_grid.sum(dim=-1, keepdim=True) * dx).clamp(min=1e-6)

        return {
            "alpha": pred_alpha,
            "cf": pred_cf,
            "log_cf": pred_log_cf,
            "m_alpha_grid": m_grid,
        }
