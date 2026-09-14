# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_embedding_variants

统一模型变体类 TGDISEmbeddingModel:
在最新旗舰 TG-DIS-DeepONet 的核心骨干 (DIS-Op 逆散射层剥离算子 + 声学时延偏置注意力 + 双轨解码头) 保持严格一致的前提下，
提供四种前端时空 Embedding 模块的一键切换，进行严格控制变量的横向消融对标:

支持的 embedding_type:
1. 'baseline_resample': 离线 4096 点降采样 + 相对到时窗采样基准 (Phase 3 原型);
2. 'physics_query':     方案一 (物理到时锚定 Query + 原生 1000 Hz 波包声学偏置 Cross-Attention);
3. 'fourier_feature':   方案二 (多尺度高频连续傅里叶字典嵌入, 覆盖 0.0725 Hz ~ 500 Hz);
4. 'sinc_filterbank':   方案三 (井筒声学奇数谐波参数化 Sinc 带通可微滤波器组)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.modules.layer_stripping import DifferentiableLayerStripping
from PaperC_CJNO_Wellbore_Inversion.src.modules.acoustic_transformer import AcousticBiasedTransformer
from PaperC_CJNO_Wellbore_Inversion.src.modules.dual_track_heads import ContinuousTrunkHead
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import GlobalWaveEncoder, CepstrumCNNEncoder
from PaperC_CJNO_Wellbore_Inversion.src.modules.embedding_modules import (
    PhysicsQueryEmbedding,
    FourierFeatureEmbedding,
    SincNetAcousticFilterbank,
    BaselineResampleEmbedding,
)


class TGDISEmbeddingModel(nn.Module):
    """
    统一端到端连续时空嵌入 TG-DIS 模型:
    保持后端的 DIS-Op 逆散射算子、声学注意力及双轨多任务头 100% 一致，
    仅替换前端波形到时特征提取层。
    """
    VALID_EMBEDDING_TYPES = (
        "baseline_resample",
        "physics_query",
        "fourier_feature",
        "sinc_filterbank",
    )

    def __init__(
        self,
        embedding_type: str = "physics_query",
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
        # Embedding 模块专用超参数
        n_win_pts: int = 301,
        t_pre: float = 0.05,
        t_post: float = 0.25,
    ):
        super().__init__()
        self.embedding_type = str(embedding_type).strip().lower()
        if self.embedding_type not in self.VALID_EMBEDDING_TYPES:
            raise ValueError(
                f"不支持的 embedding_type: '{embedding_type}', "
                f"必须在 {self.VALID_EMBEDDING_TYPES} 中选择。"
            )

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

        # 1. 严格单变量消融: 前端波形连续时空嵌入层替换
        if self.embedding_type == "baseline_resample":
            self.embedding_module = BaselineResampleEmbedding(
                in_channels=in_channels,
                d_model=d_model,
                max_nc=max_nc,
                n_win_pts=128,
                t_pre=t_pre,
                t_post=t_post,
                t_total=t_total,
                L_total=L,
                ts=ts,
            )
        elif self.embedding_type == "physics_query":
            self.embedding_module = PhysicsQueryEmbedding(
                in_channels=in_channels,
                d_model=d_model,
                max_nc=max_nc,
                n_win_pts=n_win_pts,
                t_pre=t_pre,
                t_post=t_post,
                t_total=t_total,
                L_total=L,
                ts=ts,
                init_lambda=10.0,
            )
        elif self.embedding_type == "fourier_feature":
            self.embedding_module = FourierFeatureEmbedding(
                in_channels=in_channels,
                d_model=d_model,
                max_nc=max_nc,
                n_win_pts=n_win_pts,
                num_freqs=16,
                f_min=0.0725,
                f_max=500.0,
                t_pre=t_pre,
                t_post=t_post,
                t_total=t_total,
                L_total=L,
                ts=ts,
            )
        elif self.embedding_type == "sinc_filterbank":
            self.embedding_module = SincNetAcousticFilterbank(
                in_channels=in_channels,
                d_model=d_model,
                max_nc=max_nc,
                n_filters=32,
                kernel_size=65,
                fs=1000.0,
                f_min=0.0725,
                f_max=450.0,
                a_ref=1450.0,
                L_total=L,
                t_pre=t_pre,
                t_post=t_post,
                t_total=t_total,
                ts=ts,
            )

        # 2. 波动方程可微逆散射层剥离算子 (DIS Layer) - 完全一致
        self.layer_stripping = DifferentiableLayerStripping(
            in_dim=d_model,
            d_model=d_model,
            max_nc=max_nc,
            a_ref=1450.0,
        )

        # 3. 声学时延偏置 Transformer 簇间解耦模块 - 完全一致
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

        # 4. 分支宏观编码器 - 完全一致
        self.global_wave_encoder = GlobalWaveEncoder(
            in_channels=in_channels,
            out_dim=64,
        )
        self.cepstrum_encoder = CepstrumCNNEncoder(
            in_channels=1,
            out_dim=64,
        )

        # 5. 全局井筒聚合融合网络 - 完全一致
        self.well_fusion = nn.Sequential(
            nn.Linear(d_model + 64 + 64 + cond_dim, d_well),
            nn.LayerNorm(d_well),
            nn.GELU(),
            nn.Linear(d_well, d_well),
            nn.GELU(),
        )

        # 6. 多任务解耦预测头 - 完全一致
        # 6.1 流量分配 MLP
        self.alpha_mlp = nn.Sequential(
            nn.Linear(d_model + 3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.alpha_scale = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

        # 6.2 水力顺应性 MLP
        self.cf_mlp = nn.Sequential(
            nn.Linear(d_model + 3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 6.3 裂缝起裂存在性分类头
        self.exist_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 6.4 亚米级射孔位置修正头
        self.pos_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        # 6.5 连续 Trunk 场解码头
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
        raw_wave: Optional[torch.Tensor] = None,
        cepstrum: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        norm_positions: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if batch is not None:
            wave = batch.get("wave", None)
            raw_wave = batch.get("raw_wave", None)
            cepstrum = batch["cepstrum"]
            cond = batch.get("cond", None)
            norm_positions = batch.get("norm_positions", None)
            mask = batch.get("mask", None)
            if "positions" in batch and batch["positions"] is not None:
                positions = batch["positions"]
            elif norm_positions is not None:
                positions = norm_positions * self.L
            else:
                raise KeyError("batch 必须包含 'positions' 或 'norm_positions'")
            if norm_positions is None:
                norm_positions = positions / self.L
        else:
            if wave is None and raw_wave is None:
                raise ValueError("必须传入 batch 字典或至少传入 wave 或 raw_wave")
            if cepstrum is None:
                raise ValueError("必须传入 cepstrum 张量")
            if positions is None and norm_positions is not None:
                positions = norm_positions * self.L
            elif positions is not None and norm_positions is None:
                norm_positions = positions / self.L
            elif positions is None and norm_positions is None:
                raise ValueError("必须传入 positions 或 norm_positions")

        if wave is None and raw_wave is not None:
            wave = raw_wave
        if raw_wave is None and wave is not None:
            raw_wave = wave

        B, M = positions.shape
        if cond is None:
            cond = torch.zeros((B, 3), device=positions.device, dtype=positions.dtype)
        if mask is None:
            mask = torch.ones_like(positions, dtype=torch.bool)

        # A. 前端波形到时特征提取 (执行当前选定的 Embedding 操作)
        cluster_tokens, tau = self.embedding_module(
            wave=wave,
            raw_wave=raw_wave,
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
        global_wave_feat = self.global_wave_encoder(wave)  # (B, 64)
        cepstrum_feat = self.cepstrum_encoder(cepstrum)    # (B, 64)

        # E. 全局井筒潜空间融合
        mask_exp = mask.unsqueeze(-1).float()
        pooled_h = (h_decoupled * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
        fused_input = torch.cat([pooled_h, global_wave_feat, cepstrum_feat, cond], dim=-1)
        z_well = self.well_fusion(fused_input)             # (B, d_well)

        # F. 离散精准解码头
        phys_features = torch.stack([
            gamma,
            torch.log(torch.clamp(admittance, min=1e-8)),
            attenuation_factors,
        ], dim=-1)  # (B, M, 3)
        h_phys = torch.cat([h_decoupled, phys_features], dim=-1)  # (B, M, d_model + 3)

        # 1. 流量分配 alpha: Masked Softmax + 严格单纯形守恒
        raw_alpha_logits = self.alpha_mlp(h_phys).squeeze(-1)  # (B, M)
        scaled_logits = raw_alpha_logits * F.softplus(self.alpha_scale)
        masked_logits = scaled_logits.masked_fill(~mask, -1e9)
        pred_alpha = F.softmax(masked_logits, dim=-1)
        pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))
        alpha_sum = pred_alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pred_alpha = pred_alpha / alpha_sum
        pred_alpha = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))

        # 2. 水力顺应性 cf: 预测 log10(Cf / Cf0)
        log10_cf_raw = self.cf_mlp(h_phys).squeeze(-1)        # (B, M)
        log10_cf_clamped = torch.clamp(log10_cf_raw, min=-4.0, max=4.0)
        ln10 = float(np.log(10.0))
        log_cf_raw = log10_cf_clamped * ln10
        cf_raw = self.cf_ref * torch.pow(10.0, log10_cf_clamped)

        pred_cf = torch.where(mask, cf_raw, torch.zeros_like(cf_raw))
        pred_log_cf = torch.where(mask, log_cf_raw, torch.zeros_like(log_cf_raw))
        pred_log10_cf = torch.where(mask, log10_cf_clamped, torch.zeros_like(log10_cf_clamped))

        # 3. 裂缝存在性分类头 p_exist
        p_exist_raw = self.exist_mlp(h_decoupled).squeeze(-1)  # (B, M)
        pred_exist = torch.where(mask, p_exist_raw, torch.zeros_like(p_exist_raw))

        # 4. 亚米级位置细化微调 delta_x
        delta_x_raw = self.pos_mlp(h_decoupled).squeeze(-1) * self.delta_x_max
        pred_delta_x = torch.where(mask, delta_x_raw, torch.zeros_like(delta_x_raw))

        # G. 连续场辅助头
        m_alpha_norm, c_grid, alpha_field, cf_field, log_cf_field = self.continuous_trunk_head(
            z_well=z_well,
            positions=positions,
            mask=mask,
        )

        return {
            "alpha": pred_alpha,
            "cf": pred_cf,
            "log_cf": pred_log_cf,
            "log10_cf": pred_log10_cf,
            "p_exist": pred_exist,
            "delta_x": pred_delta_x,
            "gamma": gamma,
            "admittance": admittance,
            "attenuation_factors": attenuation_factors,
            "m_alpha_grid": m_alpha_norm,
            "c_grid": c_grid,
            "alpha_field": alpha_field,
            "cf_field": cf_field,
            "log_cf_field": log_cf_field,
            "tau": tau,
            "attn_weights": attn_maps,
            "z_well": z_well,
            "h_stripped": h_stripped,
            "h_decoupled": h_decoupled,
            "embedding_type": self.embedding_type,
        }
