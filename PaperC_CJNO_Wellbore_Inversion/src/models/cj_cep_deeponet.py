# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.models.cj_cep_deeponet

CJ-Cep-DeepONet (Characteristic-Jump Cepstrum Deep Operator Network) 预研原型网络：
1. 双通道物理特征分支：
   - 波形分支 (Wave Branch): 提取高频水击波形与一阶差分波前特征；
   - 倒谱分支 (Cepstrum Branch): 提取 1D 空间深度反射倒谱时延特征；
2. 工况与射孔簇集合编码器 (Cluster Set Encoder):
   - 采用置换不变性 DeepSets/Set-Transformer 聚合任意变长簇分布 {xj}；
3. 多模态门控融合层 (Cross-Modal Fusion):
   - 融合波形、倒谱、工况与集合特征，构建潜空间全局井筒表征 z_well；
4. 连续坐标 Trunk 解码器:
   - 连续解码流体进入贡献密度场 m_alpha(x) 与水力顺应性场 c_H(x)；
5. Voronoi 守恒空间积分池化层 (Voronoi Conservative Pooling):
   - 基于已知簇坐标划分 Voronoi 积分区间，数值积分并经 Simplex 投影严格守恒输出 alpha_j 与 Cf_j。
"""
from __future__ import annotations

from typing import Dict, Tuple
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


class WaveEncoder(nn.Module):
    """波形时域特征分支 (1D CNN + 多尺度时间池化)"""
    def __init__(self, in_channels: int = 2, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, out_dim),
            nn.GELU(),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        feat = self.conv(wave)
        return self.proj(feat) # (B, out_dim)


class CepstrumEncoder(nn.Module):
    """1D 空间反射倒谱特征分支"""
    def __init__(self, in_channels: int = 1, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )

    def forward(self, cepstrum: torch.Tensor) -> torch.Tensor:
        return self.net(cepstrum) # (B, out_dim)


class ClusterSetEncoder(nn.Module):
    """射孔簇位置集合编码器 (Permutation-Invariant DeepSets)"""
    def __init__(self, num_freqs: int = 6, out_dim: int = 32):
        super().__init__()
        self.embed = FourierPositionalEncoding(num_freqs=num_freqs)
        in_dim = 2 * num_freqs + 1

        self.elem_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.set_mlp = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.GELU(),
        )

    def forward(self, norm_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # norm_pos: (B, M), mask: (B, M)
        pos_expand = norm_pos.unsqueeze(-1) # (B, M, 1)
        feat = self.embed(pos_expand)       # (B, M, D)
        elem = self.elem_mlp(feat)         # (B, M, 64)

        # 掩码平均池化
        mask_expand = mask.unsqueeze(-1).float() # (B, M, 1)
        masked_elem = elem * mask_expand
        num_valid = mask_expand.sum(dim=1).clamp(min=1.0)
        pooled = masked_elem.sum(dim=1) / num_valid # (B, 64)

        return self.set_mlp(pooled) # (B, out_dim)


class VoronoiPooling1D(nn.Module):
    """
    可微 1D Voronoi 守恒空间积分池化层：
    基于已知射孔簇坐标序列构造局部 Voronoi 单元 [b_left, b_right]，
    通过 1D 规则插值采样对连续场 m_alpha(x) 与 c_H(x) 进行局部守恒数值积分，
    严格保证单纯形物理约束 sum(alpha_hat) = 1.0，并彻底消除空井筒背景泄漏与密集簇网格丢失假阳性。
    """
    def __init__(self, L: float = 5000.0, n_grid: int = 500, cf_ref: float = 0.01, n_quad: int = 5):
        super().__init__()
        self.L = L
        self.n_grid = n_grid
        self.cf_ref = cf_ref
        self.n_quad = n_quad
        self.register_buffer("offsets", torch.linspace(0.1, 0.9, n_quad))

    def _compute_bounds(self, pos_m: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M = pos_m.shape
        b_left = torch.zeros_like(pos_m)
        b_right = torch.zeros_like(pos_m)
        for b in range(B):
            m_b = mask[b]
            nc = int(m_b.sum().item())
            if nc <= 0:
                continue
            elif nc == 1:
                p0 = pos_m[b, 0]
                b_left[b, 0] = torch.clamp(p0 - 50.0, min=0.0)
                b_right[b, 0] = torch.clamp(p0 + 50.0, max=self.L)
            else:
                p = pos_m[b, :nc]
                mids = (p[1:] + p[:-1]) / 2.0
                d0 = (p[1] - p[0]) / 2.0
                d_end = (p[-1] - p[-2]) / 2.0
                b_left[b, 0] = torch.clamp(p[0] - d0, min=0.0)
                b_left[b, 1:nc] = mids
                b_right[b, :nc-1] = mids
                b_right[b, nc-1] = torch.clamp(p[-1] + d_end, max=self.L)
        return b_left, b_right

    def forward(
        self,
        m_alpha_grid: torch.Tensor, # (B, n_grid)
        c_grid: torch.Tensor,       # (B, n_grid)
        pos_m: torch.Tensor,        # (B, M)
        mask: torch.Tensor,         # (B, M)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M = pos_m.shape
        b_left, b_right = self._compute_bounds(pos_m, mask)
        cell_widths = (b_right - b_left).clamp(min=1.0) # (B, M)

        quad_pts = b_left.unsqueeze(-1) + cell_widths.unsqueeze(-1) * self.offsets.view(1, 1, -1)
        quad_norm = (quad_pts / self.L).clamp(0.0, 1.0)

        # 1D Grid Sample 插值采样
        grid_x = quad_norm * 2.0 - 1.0
        grid_y = torch.zeros_like(grid_x)
        sample_grid = torch.stack([grid_x.view(B, M * self.n_quad, 1), grid_y.view(B, M * self.n_quad, 1)], dim=-1)

        inp_m = m_alpha_grid.unsqueeze(1).unsqueeze(2)
        inp_c = c_grid.unsqueeze(1).unsqueeze(2)

        sampled_m = F.grid_sample(inp_m, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled_c = F.grid_sample(inp_c, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)

        sampled_m = sampled_m.view(B, M, self.n_quad)
        sampled_c = sampled_c.view(B, M, self.n_quad)

        # 局部 Voronoi 守恒积分: alpha_j = int_{Omega_j} m_alpha(x) dx
        raw_alpha = sampled_m.mean(dim=-1) * cell_widths # (B, M)
        raw_alpha = torch.where(mask, raw_alpha, torch.zeros_like(raw_alpha))
        alpha_sum = raw_alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pred_alpha = raw_alpha / alpha_sum
        pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))

        # 局部顺应性池化
        pred_cf = sampled_c.mean(dim=-1)
        pred_cf = torch.where(mask, pred_cf, torch.zeros_like(pred_cf))
        pred_log_cf = torch.log(torch.clamp(pred_cf, min=1e-7) / self.cf_ref)
        pred_log_cf = torch.where(mask, pred_log_cf, torch.zeros_like(pred_log_cf))

        return pred_alpha, pred_cf, pred_log_cf


class CJCepDeepONet(nn.Module):
    """
    CJ-Cep-DeepONet 主网络结构。
    """
    def __init__(
        self,
        in_channels: int = 2,
        cond_dim: int = 3,
        p: int = 64,
        max_nc: int = 6,
        cf_ref: float = 0.01,
        L: float = 5000.0,
        n_grid: int = 500,
    ):
        super().__init__()
        self.p = p
        self.max_nc = max_nc
        self.cf_ref = cf_ref
        self.L = L
        self.n_grid = n_grid

        # 1. 特征分支
        self.wave_encoder = WaveEncoder(in_channels=in_channels, out_dim=64)
        self.cep_encoder = CepstrumEncoder(in_channels=1, out_dim=64)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.set_encoder = ClusterSetEncoder(num_freqs=8, out_dim=32)

        # 2. 多模态门控融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # 3. Branch 投影头 (生成 p 维空间基系数)
        self.branch_alpha = nn.Sequential(
            nn.Linear(128, p + 1),
        )
        self.branch_c = nn.Sequential(
            nn.Linear(128, p + 1),
        )

        # 4. Trunk 连续坐标网络 (高频 10 阶傅里叶编码，解析至 5m 簇间距)
        self.trunk_embed = FourierPositionalEncoding(num_freqs=10)
        self.trunk_mlp = nn.Sequential(
            nn.Linear(2 * 10 + 1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, p),
        )

        # 5. 内置局部 Voronoi 守恒空间积分池化层
        self.voronoi_pool = VoronoiPooling1D(L=L, n_grid=n_grid, cf_ref=cf_ref)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wave = batch["wave"]               # (B, 2, 4096)
        cepstrum = batch["cepstrum"]       # (B, 1, 1024)
        cond = batch["cond"]               # (B, 3)
        norm_pos = batch["norm_positions"] # (B, M)
        mask = batch["mask"]               # (B, M)
        B, M = norm_pos.shape

        # A. 多模态特征提取
        z_wave = self.wave_encoder(wave)
        z_cep = self.cep_encoder(cepstrum)
        z_cond = self.cond_mlp(cond)
        z_set = self.set_encoder(norm_pos, mask)

        # B. 潜空间融合
        z_fused = torch.cat([z_wave, z_cep, z_cond, z_set], dim=-1) # (B, 192)
        z_well = self.fusion(z_fused)                               # (B, 128)

        # C. 连续场算子基系数
        out_b_alpha = self.branch_alpha(z_well)
        b_alpha, bias_alpha = out_b_alpha[:, :self.p], out_b_alpha[:, self.p:]

        out_b_c = self.branch_c(z_well)
        b_c, bias_c = out_b_c[:, :self.p], out_b_c[:, self.p:]

        # D. Trunk 在空间网格 x_grid 上连续查询 (n_grid = 500)
        grid_norm = torch.linspace(0, 1.0, self.n_grid, device=wave.device) # (n_grid,)
        t_grid = self.trunk_mlp(self.trunk_embed(grid_norm.unsqueeze(-1)))   # (n_grid, p)

        # 连续流体进入贡献密度场 m_alpha(x) = Softplus(<b_alpha, t(x)> + bias)
        inner_alpha = torch.matmul(b_alpha, t_grid.t()) + bias_alpha
        m_alpha_grid = F.softplus(inner_alpha) # (B, n_grid)

        # 连续水力顺应性场 c_H(x) = Cf0 * exp(<b_c, t(x)> + bias)
        inner_c = torch.matmul(b_c, t_grid.t()) + bias_c
        c_grid = self.cf_ref * torch.exp(inner_c) # (B, n_grid)

        # E. Voronoi 守恒空间积分池化 (alpha 从连续场局部积分解出)
        pos_m = norm_pos * self.L
        pred_alpha, _, _ = self.voronoi_pool(
            m_alpha_grid=m_alpha_grid,
            c_grid=c_grid,
            pos_m=pos_m,
            mask=mask,
        )

        # F. Trunk 离散簇坐标直接查询 (精准解耦每簇顺应性 Cf_j)
        t_clusters = self.trunk_mlp(self.trunk_embed(norm_pos.unsqueeze(-1))) # (B, M, p)
        pred_log_cf_raw = torch.sum(b_c.unsqueeze(1) * t_clusters, dim=-1) + bias_c # (B, M)
        pred_cf_raw = self.cf_ref * torch.exp(pred_log_cf_raw)
        pred_cf = torch.where(mask, pred_cf_raw, torch.zeros_like(pred_cf_raw))
        pred_log_cf = torch.where(mask, pred_log_cf_raw, torch.zeros_like(pred_log_cf_raw))

        # 归一化连续场以匹配积分 1.0 (Riemann sum)
        dx = self.L / (self.n_grid - 1)
        m_alpha_norm = m_alpha_grid / (m_alpha_grid.sum(dim=-1, keepdim=True) * dx).clamp(min=1e-8)

        return {
            "alpha": pred_alpha,
            "cf": pred_cf,
            "log_cf": pred_log_cf,
            "m_alpha_grid": m_alpha_norm,
            "c_grid": c_grid,
        }
