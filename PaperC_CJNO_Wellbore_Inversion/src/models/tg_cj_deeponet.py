# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.models.tg_cj_deeponet

TG-CJ-DeepONet (Time-Gated Characteristic-Jump Deep Operator Network):
第二阶段密集多簇 (5~40m) 水击波智能物理反演核心主干网络。

核心三层架构:
1. 物理到时对齐提取 (Time-Gating Module):
   tau_j = t_s + 2*x_j/a，支持 3 种多模态窗提取机制 (relative_window, gaussian_gating, ceps_patch)；
2. 声学时延偏置簇间解耦注意力 (Acoustic-Biased Decoupling Transformer):
   显式融入声学物理传播时差偏置 A_{ij} = Softmax(q k^T / sqrt(d) - gamma * |x_i - x_j| / a)；
3. 双轨协同解码头 (Dual-Track Collaborative Heads):
   - 轨 1 (离散精准头): Masked Softmax 严格保障 sum(alpha) = 1.0，直接回归 log10(Cf/Cf0)；
   - 轨 2 (连续场辅助头): 连续解码 m_alpha(x) 与 c_H(x)，提供 Voronoi 守恒积分与一致性协同约束 L_cons。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.modules.time_gating import TimeGatingModule
from PaperC_CJNO_Wellbore_Inversion.src.modules.acoustic_transformer import AcousticBiasedTransformer
from PaperC_CJNO_Wellbore_Inversion.src.modules.dual_track_heads import DualTrackHead
class GlobalWaveEncoder(nn.Module):
    """全局波形时域宏观特征编码器 (1D CNN + 多尺度池化)"""
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


class TGCJDeepONet(nn.Module):

    """
    TG-CJ-DeepONet 完整网络模型。
    支持 window_mode 消融与 use_acoustic_bias 消融。
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

        # 1. 物理到时对齐提取模块
        self.time_gating = TimeGatingModule(
            window_mode=window_mode,
            in_channels=in_channels,
            d_model=d_model,
            max_nc=max_nc,
            t_total=t_total,
            L_total=L,
            ts=ts,
        )

        # 2. 声学时延偏置 Transformer 簇间解耦模块
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

        # 3. 全局波形宏观衰减特征提取器 (为轨 2 提供整井宏观阻尼信息)
        self.global_wave_encoder = GlobalWaveEncoder(
            in_channels=in_channels,
            out_dim=64,
        )

        # 4. 双轨协同解码头 (轨 1: 离散精准头, 轨 2: 连续场辅助头)
        self.dual_track_head = DualTrackHead(
            d_model=d_model,
            d_well=128,
            p=p,
            n_grid=n_grid,
            L=L,
            cf_ref=cf_ref,
            cond_dim=cond_dim,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播计算:
        输入 batch 字典:
            - wave: (B, 2, 4096)
            - cepstrum: (B, 1, 1024)
            - cond: (B, 3)
            - norm_positions: (B, M)
            - mask: (B, M)
            - positions (可选): (B, M)
        返回:
            预测字典，包含 alpha, cf, log_cf, log10_cf, m_alpha_grid, c_grid,
            alpha_field, tau, attn_weights 等。
        """
        wave = batch["wave"]
        cepstrum = batch["cepstrum"]
        cond = batch["cond"]
        norm_positions = batch["norm_positions"]
        mask = batch["mask"]

        if "positions" in batch and batch["positions"] is not None:
            positions = batch["positions"]
        else:
            positions = norm_positions * self.L

        # A. 物理到时对齐波前特征提取
        cluster_tokens, tau = self.time_gating(
            wave=wave,
            cepstrum=cepstrum,
            positions=positions,
            norm_positions=norm_positions,
            cond=cond,
            mask=mask,
        ) # tokens: (B, M, d_model), tau: (B, M)

        # B. 声学时延偏置自注意力解耦
        h, attn_maps = self.transformer(
            cluster_tokens=cluster_tokens,
            positions=positions,
            norm_positions=norm_positions,
            cond=cond,
            mask=mask,
        ) # h: (B, M, d_model), attn_maps: (B, L, H, M, M)

        # C. 提取宏观波形特征
        global_wave = self.global_wave_encoder(wave) # (B, 64)

        # D. 双轨协同解码
        out = self.dual_track_head(
            h=h,
            positions=positions,
            norm_positions=norm_positions,
            mask=mask,
            global_wave_feat=global_wave,
            cond=cond,
        )

        # 注入中间物理量方便调试、可解释性分析与消融制图
        out["tau"] = tau
        out["attn_weights"] = attn_maps
        return out
