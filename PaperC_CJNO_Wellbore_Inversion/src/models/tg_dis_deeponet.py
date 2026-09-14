# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet

TG-DIS-DeepONet (Time-Gated Differentiable Inverse Scattering Deep Operator Network):
面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演旗舰网络。

核心架构:
1. 物理到时对齐提取 (Time-Gating Module):
   tau_j = t_s + 2*x_j/a，以相对窗 (relative_window) 消除井深绝对时延；
2. 波动方程可微逆散射层剥离算子 (Differentiable Layer-Stripping Layer, DIS Layer):
   严格沿井筒由跟端到趾端 (x_1 -> x_{N_c}) 逐级消除上游裂缝累积透射功率损耗与多程混响，
   显式输出物理反射率 Gamma_j in [-1, 0] 与支路导纳 Y_{b,j} >= 0；
3. 声学时延偏置簇间解耦注意力 (Acoustic-Biased Decoupling Transformer):
   显式注入声学格林函数时延物理偏置 A_{ij} = Softmax(q_i k_j^T / sqrt(d) - gamma * |x_i - x_j| / a)；
4. 倒谱与宏观全波多模态分支 (Multi-Modal Cepstrum & Wave Branch Encoders):
   抽取全井宏观流体动力学与频域能量特征；
5. 多任务解耦解码头 (Multi-Task Decoupling Heads):
   - alpha: Masked Softmax 严格物理单纯形流量分配 (max |sum(alpha) - 1.0| < 10^{-6})；
   - cf & log_cf: 对数水力储集顺应性回归；
   - p_exist: 裂缝起裂存在性概率分类头 in [0, 1]；
   - delta_x: 亚米级射孔位置偏移细化回归头 in [-10.0, 10.0] m；
   - m_alpha_grid & c_grid: 全井连续流体密度场与顺应性场 (含 Voronoi 守恒空间积分)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.modules.time_gating import TimeGatingModule
from PaperC_CJNO_Wellbore_Inversion.src.modules.layer_stripping import DifferentiableLayerStripping
from PaperC_CJNO_Wellbore_Inversion.src.modules.acoustic_transformer import AcousticBiasedTransformer
from PaperC_CJNO_Wellbore_Inversion.src.modules.dual_track_heads import ContinuousTrunkHead


class GlobalWaveEncoder(nn.Module):
    """全局水锤波形宏观衰减特征提取器 (1D CNN + 多尺度池化)"""

    def __init__(self, in_channels: int = 2, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
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
        return self.proj(feat)


class CepstrumCNNEncoder(nn.Module):
    """空间反射倒谱 CNN 特征编码器 (从 1024 点倒谱中提取多尺度谐波周期特征)"""

    def __init__(self, in_channels: int = 1, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, out_dim),
            nn.GELU(),
        )

    def forward(self, cepstrum: torch.Tensor) -> torch.Tensor:
        feat = self.conv(cepstrum)
        return self.proj(feat)


class TGDISDeepONet(nn.Module):
    """
    TG-DIS-DeepONet: 融合可微逆散射层剥离算子的高精度水击物理反演模型。
    """

    def __init__(
        self,
        window_mode: str = "relative_window",
        use_acoustic_bias: bool = True,
        in_channels: int = 2,
        cond_dim: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        p: int = 64,
        max_nc: int = 6,
        cf_ref: float = 0.01,
        L: float = 5000.0,
        n_grid: int = 500,
        t_total: float = 60.0,
        ts: float = 1.0,
        delta_x_max: float = 10.0,
        d_well: int = 128,
    ):
        super().__init__()
        self.window_mode = window_mode
        self.use_acoustic_bias = use_acoustic_bias
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.d_model = d_model
        self.max_nc = max_nc
        self.cf_ref = cf_ref
        self.L = L
        self.n_grid = n_grid
        self.t_total = t_total
        self.ts = ts
        self.delta_x_max = delta_x_max
        self.d_well = d_well

        # 1. 物理到时对齐提取模块 (Trunk 前端波前对齐)
        self.time_gating = TimeGatingModule(
            window_mode=window_mode,
            in_channels=in_channels,
            d_model=d_model,
            max_nc=max_nc,
            t_total=t_total,
            L_total=L,
            ts=ts,
        )

        # 2. 波动方程可微逆散射层剥离算子 (DIS Layer)
        self.layer_stripping = DifferentiableLayerStripping(
            in_dim=d_model,
            d_model=d_model,
            max_nc=max_nc,
            a_ref=1450.0,
        )

        # 3. 声学时延偏置 Transformer 簇间解耦模块
        self.transformer = AcousticBiasedTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=128,
            cond_dim=cond_dim,
            num_pos_freqs=8,
            dropout=0.0,
            use_acoustic_bias=use_acoustic_bias,
            init_gamma=10.0,
        )

        # 4. 分支编码器: 全局波形编码器与倒谱编码器
        self.global_wave_encoder = GlobalWaveEncoder(
            in_channels=in_channels,
            out_dim=64,
        )
        self.cepstrum_encoder = CepstrumCNNEncoder(
            in_channels=1,
            out_dim=64,
        )

        # 5. 全局井筒聚合融合网络
        self.well_fusion = nn.Sequential(
            nn.Linear(d_model + 64 + 64 + cond_dim, d_well),
            nn.LayerNorm(d_well),
            nn.GELU(),
            nn.Linear(d_well, d_well),
            nn.GELU(),
        )

        # 6. 多任务解耦预测头
        # 6.1 流量分配 MLP: 输入融合解耦波形表征与逆散射层剥离显式物理量 [h, Gamma, ln(Y_b), T_cum]
        self.alpha_mlp = nn.Sequential(
            nn.Linear(d_model + 3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # 可学习 Logit 敏感度温度缩放系数 (有效打破密集多簇均摊陷阱)
        self.alpha_scale = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

        # 6.2 水力顺应性 MLP (预测 log10(Cf / Cf0))
        self.cf_mlp = nn.Sequential(
            nn.Linear(d_model + 3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 6.3 裂缝起裂存在性分类头 (p_exist in [0, 1])
        self.exist_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 6.4 亚米级射孔位置修正头 (delta_x in [-delta_x_max, +delta_x_max] m)
        self.pos_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        # 6.5 连续 Trunk 场解码头 (含 Voronoi 守恒空间积分池化)
        self.continuous_trunk_head = ContinuousTrunkHead(
            d_well=d_well,
            p=p,
            num_freqs=10,
            n_grid=n_grid,
            L=L,
            cf_ref=cf_ref,
        )

    def forward(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        wave: Optional[torch.Tensor] = None,
        cepstrum: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        norm_positions: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播计算:
        支持接收 batch 字典或单独传入张量。
        返回:
            包含 alpha, cf, log_cf, log10_cf, p_exist, delta_x, gamma, admittance,
            attenuation_factors, m_alpha_grid, c_grid, alpha_field, tau, attn_weights 等的字典。
        """
        if batch is not None:
            wave = batch["wave"]
            cepstrum = batch["cepstrum"]
            cond = batch["cond"]
            norm_positions = batch.get("norm_positions")
            mask = batch["mask"]
            if "positions" in batch and batch["positions"] is not None:
                positions = batch["positions"]
            elif norm_positions is not None:
                positions = norm_positions * self.L
            else:
                raise KeyError("batch 必须包含 'positions' 或 'norm_positions'")
            if norm_positions is None:
                norm_positions = positions / self.L
        else:
            if wave is None or cepstrum is None or cond is None:
                raise ValueError("必须传入 batch 字典或完整的 wave, cepstrum, cond 张量")
            if positions is None and norm_positions is not None:
                positions = norm_positions * self.L
            elif positions is not None and norm_positions is None:
                norm_positions = positions / self.L
            elif positions is None and norm_positions is None:
                raise ValueError("必须传入 positions 或 norm_positions")
            if mask is None:
                mask = torch.ones_like(positions, dtype=torch.bool)

        B, M = positions.shape

        # A. 物理到时对齐波前特征提取 (Time-Gating)
        cluster_tokens, tau = self.time_gating(
            wave=wave,
            cepstrum=cepstrum,
            positions=positions,
            norm_positions=norm_positions,
            cond=cond,
            mask=mask,
        )  # cluster_tokens: (B, M, d_model), tau: (B, M)

        # B. 波动方程可微逆散射层剥离算子 (DIS Layer)
        dis_out = self.layer_stripping(
            h_patches=cluster_tokens,
            positions=positions,
            wavespeed=cond,
            mask=mask,
        )
        h_stripped = dis_out.h_stripped                # (B, M, d_model)
        gamma = dis_out.gamma                          # (B, M)
        admittance = dis_out.admittance                # (B, M)
        attenuation_factors = dis_out.attenuation_factors  # (B, M)

        # C. 声学时延偏置自注意力解耦
        h_decoupled, attn_maps = self.transformer(
            cluster_tokens=h_stripped,
            positions=positions,
            norm_positions=norm_positions,
            cond=cond,
            mask=mask,
        )  # h_decoupled: (B, M, d_model), attn_maps: (B, L, H, M, M)

        # D. 分支特征抽取: 宏观波形特征与倒谱特征
        global_wave_feat = self.global_wave_encoder(wave)         # (B, 64)
        cepstrum_feat = self.cepstrum_encoder(cepstrum)           # (B, 64)

        # E. 全局井筒潜空间融合
        mask_exp = mask.unsqueeze(-1).float()
        pooled_h = (h_decoupled * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)  # (B, d_model)
        fused_input = torch.cat([pooled_h, global_wave_feat, cepstrum_feat, cond], dim=-1)
        z_well = self.well_fusion(fused_input)                    # (B, d_well)

        # F. 离散精准解码头
        # 融合逆散射层剥离显式物理量: [gamma, ln(admittance+eps), attenuation_factors]
        phys_features = torch.stack([
            gamma,
            torch.log(torch.clamp(admittance, min=1e-8)),
            attenuation_factors,
        ], dim=-1)  # (B, M, 3)
        h_phys = torch.cat([h_decoupled, phys_features], dim=-1)  # (B, M, d_model + 3)

        # 1. 流量分配 alpha: Masked Softmax + 严格单纯形守恒微修正
        raw_alpha_logits = self.alpha_mlp(h_phys).squeeze(-1)  # (B, M)
        scaled_logits = raw_alpha_logits * F.softplus(self.alpha_scale)
        masked_logits = scaled_logits.masked_fill(~mask, -1e9)
        pred_alpha = F.softmax(masked_logits, dim=-1)
        pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))
        alpha_sum = pred_alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pred_alpha = pred_alpha / alpha_sum
        pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))

        # 2. 水力顺应性 cf: 预测 log10(Cf / Cf0)
        log10_cf_raw = self.cf_mlp(h_phys).squeeze(-1)       # (B, M)
        log10_cf_clamped = torch.clamp(log10_cf_raw, min=-4.0, max=4.0)
        ln10 = float(np.log(10.0))
        log_cf_raw = log10_cf_clamped * ln10
        cf_raw = self.cf_ref * torch.pow(10.0, log10_cf_clamped)

        pred_cf = torch.where(mask, cf_raw, torch.zeros_like(cf_raw))
        pred_log_cf = torch.where(mask, log_cf_raw, torch.zeros_like(log_cf_raw))
        pred_log10_cf = torch.where(mask, log10_cf_clamped, torch.zeros_like(log10_cf_clamped))

        # 3. 裂缝存在性分类头 p_exist
        p_exist_raw = self.exist_mlp(h_decoupled).squeeze(-1)     # (B, M)
        pred_exist = torch.where(mask, p_exist_raw, torch.zeros_like(p_exist_raw))

        # 4. 亚米级位置细化微调 delta_x in [-delta_x_max, +delta_x_max]
        delta_x_raw = self.pos_mlp(h_decoupled).squeeze(-1) * self.delta_x_max  # (B, M)
        pred_delta_x = torch.where(mask, delta_x_raw, torch.zeros_like(delta_x_raw))

        # G. 连续场辅助头 (Continuous Field Trunk Head)
        m_alpha_norm, c_grid, alpha_field, cf_field, log_cf_field = self.continuous_trunk_head(
            z_well=z_well,
            positions=positions,
            mask=mask,
        )

        return {
            # 离散核心物理预测
            "alpha": pred_alpha,
            "cf": pred_cf,
            "log_cf": pred_log_cf,
            "log10_cf": pred_log10_cf,
            "p_exist": pred_exist,
            "delta_x": pred_delta_x,
            # 可微逆散射层剥离显式物理量
            "gamma": gamma,
            "admittance": admittance,
            "attenuation_factors": attenuation_factors,
            # 连续物理场
            "m_alpha_grid": m_alpha_norm,
            "c_grid": c_grid,
            # 连续场 Voronoi 守恒空间池化量
            "alpha_field": alpha_field,
            "cf_field": cf_field,
            "log_cf_field": log_cf_field,
            # 中间物理引导量与可解释性
            "tau": tau,
            "attn_weights": attn_maps,
            "z_well": z_well,
            "h_stripped": h_stripped,
            "h_decoupled": h_decoupled,
        }
