# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.models.deeponet

Vanilla DeepONet (深度算子网络) 基线模型：
1. Branch Network: 编码离散井口水头时程与工况条件，输出 p 维算子基系数 b_alpha 与 b_C；
2. Trunk Network: 编码连续井深坐标 x in [0, L]，结合高频傅里叶位置编码输出 p 维空间基函数 t(x)；
3. 内积映射: G(u)(x) = <b(u), t(x)> + b0 连续生成流体贡献场与顺应性场；
4. 离散簇查询: 在已知簇坐标 xj 处查询并经 Masked Softmax 保证守恒。
"""
from __future__ import annotations

from typing import Dict
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
        scaled = two_pi * x * self.freqs # (..., num_freqs)
        sin_f = torch.sin(scaled)
        cos_f = torch.cos(scaled)
        return torch.cat([x, sin_f, cos_f], dim=-1) # (..., 2*num_freqs + 1)


class BranchNet(nn.Module):
    def __init__(self, in_channels: int = 2, cond_dim: int = 3, p: int = 64):
        super().__init__()
        self.p = p

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )

        self.wave_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 128),
            nn.GELU(),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        # 分别输出 b_alpha 与 b_C (各 p 维 + 1 偏置)
        self.fc_alpha = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.GELU(),
            nn.Linear(128, p + 1),
        )
        self.fc_c = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.GELU(),
            nn.Linear(128, p + 1),
        )

    def forward(self, wave: torch.Tensor, cond: torch.Tensor):
        z_wave = self.wave_proj(self.conv(wave)) # (B, 128)
        z_cond = self.cond_mlp(cond)             # (B, 32)
        z = torch.cat([z_wave, z_cond], dim=-1)  # (B, 160)

        out_alpha = self.fc_alpha(z) # (B, p+1)
        out_c = self.fc_c(z)         # (B, p+1)

        b_alpha, bias_alpha = out_alpha[:, :self.p], out_alpha[:, self.p:]
        b_c, bias_c = out_c[:, :self.p], out_c[:, self.p:]
        return b_alpha, bias_alpha, b_c, bias_c


class TrunkNet(nn.Module):
    def __init__(self, p: int = 64, num_freqs: int = 10):
        super().__init__()
        self.embed = FourierPositionalEncoding(num_freqs=num_freqs)
        in_dim = 2 * num_freqs + 1

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, p),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, N_pts) 或 (N_pts,)
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)
        coords = coords.unsqueeze(-1) # (B, N_pts, 1)
        feat = self.embed(coords)     # (B, N_pts, 2*num_freqs+1)
        return self.mlp(feat)         # (B, N_pts, p)


class VanillaDeepONet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        cond_dim: int = 3,
        p: int = 64,
        max_nc: int = 6,
        cf_ref: float = 0.01,
        L: float = 5000.0,
    ):
        super().__init__()
        self.p = p
        self.max_nc = max_nc
        self.cf_ref = cf_ref
        self.L = L

        self.branch = BranchNet(in_channels=in_channels, cond_dim=cond_dim, p=p)
        self.trunk = TrunkNet(p=p, num_freqs=10)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wave = batch["wave"]             # (B, 2, 4096)
        cond = batch["cond"]             # (B, 3)
        norm_pos = batch["norm_positions"] # (B, M)
        mask = batch["mask"]             # (B, M)
        B, M = norm_pos.shape

        # A. Branch 算子系数
        b_alpha, bias_alpha, b_c, bias_c = self.branch(wave, cond) # (B, p), (B, 1)

        # B. Trunk 离散簇查询点计算
        t_clusters = self.trunk(norm_pos) # (B, M, p)

        # C. 算子内积: <b, t(xj)> + bias
        # b_alpha: (B, 1, p), t_clusters: (B, M, p) -> (B, M)
        logits_alpha = torch.sum(b_alpha.unsqueeze(1) * t_clusters, dim=-1) + bias_alpha
        pred_log_cf = torch.sum(b_c.unsqueeze(1) * t_clusters, dim=-1) + bias_c

        # D. Masked Softmax 守恒投影
        masked_logits = logits_alpha.masked_fill(~mask, -1e9)
        pred_alpha = F.softmax(masked_logits, dim=-1)
        pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))

        # E. 顺应性恢复
        pred_cf = self.cf_ref * torch.exp(pred_log_cf)
        pred_cf = torch.where(mask, pred_cf, torch.zeros_like(pred_cf))
        pred_log_cf = torch.where(mask, pred_log_cf, torch.zeros_like(pred_log_cf))

        # F. 连续空间全域网格查询 (N_grid = 500)
        grid_norm = torch.linspace(0, 1.0, 500, device=wave.device) # (500,)
        t_grid = self.trunk(grid_norm) # (1, 500, p)
        # 内积生成连续进液密度场 m_alpha(x) = Softplus(<b, t(x)> + b0)
        m_grid = F.softplus(torch.sum(b_alpha.unsqueeze(1) * t_grid, dim=-1) + bias_alpha) # (B, 500)
        dx = self.L / 499.0
        m_grid = m_grid / (m_grid.sum(dim=-1, keepdim=True) * dx).clamp(min=1e-6)

        return {
            "alpha": pred_alpha,
            "cf": pred_cf,
            "log_cf": pred_log_cf,
            "m_alpha_grid": m_grid,
        }
