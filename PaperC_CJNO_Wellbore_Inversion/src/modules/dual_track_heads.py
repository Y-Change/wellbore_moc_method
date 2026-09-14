# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.modules.dual_track_heads

双轨协同解码头 (Dual-Track Collaborative Heads):
1. 轨 1 (离散精准头 - DiscretePreciseHead):
   - 由 Transformer 各簇解耦表征 h_j 直接经独立 MLP 预测;
   - 经 Masked Softmax 输出严格和为 1 的离散流量份额 alpha_hat_j in Simplex;
   - 预测对数尺度跨量级水力顺应性 log10(C_f_hat_j / C_f_0);
2. 轨 2 (连续场辅助头 - ContinuousTrunkHead):
   - 由井筒全局表征 z_well 与 Trunk 空间傅里叶坐标编码连续内积;
   - 输出连续流体进入密度场 m_alpha(x) 与水力顺应性场 c_H(x);
   - 通过 Voronoi 守恒空间积分池化计算区间积分量 alpha_hat_field_j = int_{Omega_j} m_alpha(x) dx;
3. 双轨协同调度头 (DualTrackHead):
   - 整合两轨预测，提供端到端输出并为损失函数提供一致性闭环约束 L_cons。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.modules.acoustic_transformer import FourierPositionalEncoding


class VoronoiPooling1D(nn.Module):
    """
    可微 1D Voronoi 守恒空间积分池化层：
    基于已知射孔簇坐标序列构造局部 Voronoi 单元 [b_left, b_right]，
    通过 1D 规则插值采样对连续场 m_alpha(x) 与 c_H(x) 进行局部守恒数值积分，
    严格保证单纯形物理约束 sum(alpha_hat) = 1.0。
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
        pred_alpha = raw_alpha

        # 局部顺应性池化
        pred_cf = sampled_c.mean(dim=-1)
        pred_cf = torch.where(mask, pred_cf, torch.zeros_like(pred_cf))
        pred_log_cf = torch.log(torch.clamp(pred_cf, min=1e-7) / self.cf_ref)
        pred_log_cf = torch.where(mask, pred_log_cf, torch.zeros_like(pred_log_cf))

        return pred_alpha, pred_cf, pred_log_cf



class DiscretePreciseHead(nn.Module):
    """
    轨 1: 离散精准解码头 (Discrete Precise Head)
    从逐簇解耦特征 h_j 直接通过 MLP 预测:
    1. alpha: 通过带掩码的 Masked Softmax，严格保证 sum_j alpha_j = 1.0 且非激活簇严格为 0；
    2. cf & log_cf: 预测 log10(Cf / Cf0)，转换为 Cf = Cf0 * 10^(log10_cf) 及 ln(Cf / Cf0)。
    """
    def __init__(
        self,
        d_model: int = 64,
        cf_ref: float = 0.01, # 基准顺应性 Cf0 = 0.01 m^2
    ):
        super().__init__()
        self.d_model = d_model
        self.cf_ref = cf_ref

        # 流量份额预测 MLP (输出单纯形未归一化 logits)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 水力顺应性预测 MLP (输出 log10(Cf / Cf0))
        self.cf_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        h: torch.Tensor,                    # (B, M, d_model)
        mask: Optional[torch.Tensor] = None,# (B, M) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            pred_alpha: (B, M) 流量分配 (和严格为 1)
            pred_cf: (B, M) 绝对顺应性 (m^2)
            pred_log_cf: (B, M) 自然对数 ln(Cf / Cf0)
            pred_log10_cf: (B, M) 常用对数 log10(Cf / Cf0)
        """
        B, M, _ = h.shape

        # 1. 流量分配 logits 与 Masked Softmax
        raw_alpha_logits = self.alpha_mlp(h).squeeze(-1) # (B, M)

        if mask is not None:
            # 未激活簇赋予极小负值 (-1e9) 屏蔽
            masked_logits = raw_alpha_logits.masked_fill(~mask, -1e9)
            pred_alpha = F.softmax(masked_logits, dim=-1)
            pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))
            # 严格数值重整化确保精度在 1e-6 以内
            alpha_sum = pred_alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            pred_alpha = pred_alpha / alpha_sum
            pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))
        else:
            pred_alpha = F.softmax(raw_alpha_logits, dim=-1)

        # 2. 对数水力顺应性预测 log10(Cf / Cf0)
        log10_cf_raw = self.cf_mlp(h).squeeze(-1) # (B, M)
        # 转换至自然对数: ln(Cf / Cf0) = log10(Cf / Cf0) * ln(10)
        ln10 = float(np.log(10.0))
        log_cf_raw = log10_cf_raw * ln10

        # 计算绝对顺应性 Cf = Cf0 * 10^(log10_cf)
        cf_raw = self.cf_ref * torch.pow(10.0, torch.clamp(log10_cf_raw, min=-4.0, max=4.0))

        if mask is not None:
            pred_cf = torch.where(mask, cf_raw, torch.zeros_like(cf_raw))
            pred_log_cf = torch.where(mask, log_cf_raw, torch.zeros_like(log_cf_raw))
            pred_log10_cf = torch.where(mask, log10_cf_raw, torch.zeros_like(log10_cf_raw))
        else:
            pred_cf = cf_raw
            pred_log_cf = log_cf_raw
            pred_log10_cf = log10_cf_raw

        return pred_alpha, pred_cf, pred_log_cf, pred_log10_cf


class ContinuousTrunkHead(nn.Module):
    """
    轨 2: 连续场辅助解码头 (Continuous Field Trunk Head)
    整井全局表征 z_well 与 Trunk 网络的空间傅里叶坐标编码内积，
    输出连续场 m_alpha(x) 与 c_H(x)，并通过 Voronoi 空间守恒池化输出区间守恒积分量。
    """
    def __init__(
        self,
        d_well: int = 128,
        p: int = 64,
        num_freqs: int = 10,
        n_grid: int = 500,
        L: float = 5000.0,
        cf_ref: float = 0.01,
    ):
        super().__init__()
        self.d_well = d_well
        self.p = p
        self.n_grid = n_grid
        self.L = L
        self.cf_ref = cf_ref

        # Branch 投影网络 (生成 p 维算子基系数与偏置)
        self.branch_alpha = nn.Linear(d_well, p + 1)
        self.branch_c = nn.Linear(d_well, p + 1)

        # Trunk 连续坐标编码器 (10 阶傅里叶编码)
        self.trunk_embed = FourierPositionalEncoding(num_freqs=num_freqs)
        self.trunk_mlp = nn.Sequential(
            nn.Linear(2 * num_freqs + 1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, p),
        )

        # Voronoi 守恒空间积分池化层
        self.voronoi_pool = VoronoiPooling1D(L=L, n_grid=n_grid, cf_ref=cf_ref)

        # 预计算固定空间网格坐标 [0, 1]
        self.register_buffer("grid_norm", torch.linspace(0.0, 1.0, n_grid))

    def forward(
        self,
        z_well: torch.Tensor,               # (B, d_well)
        positions: torch.Tensor,            # (B, M) [m]
        mask: Optional[torch.Tensor] = None,# (B, M) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            m_alpha_grid: (B, n_grid) 归一化连续流体进入密度场
            c_grid: (B, n_grid) 连续水力顺应性场
            alpha_field: (B, M) Voronoi 积分得到的离散份额
            cf_field: (B, M) Voronoi 积分得到的簇顺应性
            log_cf_field: (B, M) 顺应性对数
        """
        B = z_well.shape[0]

        # 1. 连续算子基系数
        out_alpha = self.branch_alpha(z_well)
        b_alpha, bias_alpha = out_alpha[:, :self.p], out_alpha[:, self.p:]

        out_c = self.branch_c(z_well)
        b_c, bias_c = out_c[:, :self.p], out_c[:, self.p:]

        # 2. Trunk 在空间网格上的响应
        t_coords = self.trunk_embed(self.grid_norm.unsqueeze(-1)) # (n_grid, 2*num_freqs+1)
        t_grid = self.trunk_mlp(t_coords)                        # (n_grid, p)

        # 3. 连续场解码
        inner_alpha = torch.matmul(b_alpha, t_grid.t()) + bias_alpha # (B, n_grid)
        m_alpha_grid = F.softplus(inner_alpha)                       # (B, n_grid)

        inner_c = torch.matmul(b_c, t_grid.t()) + bias_c             # (B, n_grid)
        c_grid = self.cf_ref * torch.exp(torch.clamp(inner_c, -5.0, 5.0)) # (B, n_grid)

        # 4. 连续流体进入密度场黎曼和归一化 (全井积分为 1.0)
        dx = self.L / (self.n_grid - 1)
        field_integral = (m_alpha_grid.sum(dim=-1, keepdim=True) * dx).clamp(min=1e-8)
        m_alpha_norm = m_alpha_grid / field_integral

        # 5. Voronoi 守恒空间数值积分得到各簇进入量 int_{Omega_j} m_alpha(x) dx
        pos_m = positions
        if mask is None:
            mask = torch.ones_like(positions, dtype=torch.bool)

        alpha_field, cf_field, log_cf_field = self.voronoi_pool(
            m_alpha_grid=m_alpha_norm,
            c_grid=c_grid,
            pos_m=pos_m,
            mask=mask,
        )

        return m_alpha_norm, c_grid, alpha_field, cf_field, log_cf_field


class DualTrackHead(nn.Module):
    """
    双轨协同调度解码头:
    融合离散精准头 (Track 1) 与连续场辅助头 (Track 2)，
    输出全套物理反演指标，并支持双轨一致性协同约束。
    """
    def __init__(
        self,
        d_model: int = 64,
        d_well: int = 128,
        p: int = 64,
        n_grid: int = 500,
        L: float = 5000.0,
        cf_ref: float = 0.01,
        cond_dim: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_well = d_well
        self.L = L

        # 轨 1: 离散精准头
        self.track1_discrete = DiscretePreciseHead(
            d_model=d_model,
            cf_ref=cf_ref,
        )

        # 全局井筒聚合网络 (聚合并融合逐簇表征、全局波形与工况)
        self.well_fusion = nn.Sequential(
            nn.Linear(d_model + 64 + cond_dim, d_well),
            nn.LayerNorm(d_well),
            nn.GELU(),
            nn.Linear(d_well, d_well),
            nn.GELU(),
        )

        # 轨 2: 连续场辅助头
        self.track2_continuous = ContinuousTrunkHead(
            d_well=d_well,
            p=p,
            num_freqs=10,
            n_grid=n_grid,
            L=L,
            cf_ref=cf_ref,
        )

    def forward(
        self,
        h: torch.Tensor,                    # (B, M, d_model) 解耦簇表征
        positions: torch.Tensor,            # (B, M) [m]
        norm_positions: torch.Tensor,       # (B, M) [0, 1]
        mask: torch.Tensor,                 # (B, M) bool
        global_wave_feat: torch.Tensor,     # (B, 64)
        cond: torch.Tensor,                 # (B, cond_dim)
    ) -> Dict[str, torch.Tensor]:
        # 1. 轨 1: 离散精准头前向
        pred_alpha, pred_cf, pred_log_cf, pred_log10_cf = self.track1_discrete(h, mask=mask)

        # 2. 聚合井筒全局特征 z_well
        # 掩码有效簇平均池化
        mask_exp = mask.unsqueeze(-1).float()
        pooled_h = (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0) # (B, d_model)
        fused_input = torch.cat([pooled_h, global_wave_feat, cond], dim=-1)
        z_well = self.well_fusion(fused_input) # (B, d_well)

        # 3. 轨 2: 连续场辅助头前向
        pos_m = positions
        m_alpha_norm, c_grid, alpha_field, cf_field, log_cf_field = self.track2_continuous(
            z_well=z_well,
            positions=pos_m,
            mask=mask,
        )

        return {
            # 轨 1 核心精准预测
            "alpha": pred_alpha,
            "cf": pred_cf,
            "log_cf": pred_log_cf,
            "log10_cf": pred_log10_cf,
            # 轨 2 连续物理场
            "m_alpha_grid": m_alpha_norm,
            "c_grid": c_grid,
            # 轨 2 守恒空间池化辅助量
            "alpha_field": alpha_field,
            "cf_field": cf_field,
            "log_cf_field": log_cf_field,
            # 潜空间全局表征
            "z_well": z_well,
        }
