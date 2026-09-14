# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.modules.embedding_modules

网络端到端连续时空 Embedding 架构模块库:
针对离线固定等间隔下采样 (从 60,001 点截断到 4,096 点导致 Delta_t = 14.65ms > Delta_tau_10m = 13.79ms 的采样瓶颈)，
将机械降采样升级为网络内部端到端可学习的连续时空嵌入操作，支持三大对标方案:

1. 方案一 (Physics-Anchored Query Embedding, TG-DIS-QueryEmbed):
   - 连续物理 Query: q_j = MLP([x_j, tau_j, a]) in R^d;
   - 在原生 1000 Hz 高频波包 Token 序列上执行带时空声学衰减偏置的 Cross-Attention:
     h_j = sum_t Softmax( (q_j k(t)^T) / sqrt(d) - lambda * |t - tau_j| ) v(t)
   - 充当自适应高频连续积分滤波器，消除到时截断误差。

2. 方案二 (Continuous Fourier Feature Embedding, TG-DIS-FourierEmbed):
   - 多尺度高频连续傅里叶字典: gamma(t) = [sin(omega_m t), cos(omega_m t)]^T，覆盖 0.0725 Hz 至 500 Hz;
   - 将连续时间傅里叶坐标嵌入注入波形时序特征中，克服波前突变拟合中的“低频偏置 (Spectral Bias)”，赋予模型亚毫秒级时间感知。

3. 方案三 (SincNet-style Acoustic Filterbank Embedding, TG-DIS-SincEmbed):
   - 可学习连续带通 Sinc 滤波器组: g(t, f_low, f_high) = 2*f_high*sinc(2*f_high*t) - 2*f_low*sinc(2*f_low*t);
   - 截止频率 [f_low, f_high] 可学习，初始频段严格绑定井筒声学奇数次驻波谐频 f_k = (2k-1)*a / (4L);
   - 实现无损声学带通滤波与物理包络提取。

4. 基准方案 (BaselineResampleEmbedding):
   - 离线 4096 点降采样 + RelativeWindowGating 机制，用于严格控制变量横向消融对标。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.modules.time_gating import (
    compute_theoretical_arrival_times,
    RelativeWindowGating,
)


def extract_raw_wave_packet(
    raw_wave: torch.Tensor,       # (B, C, N_raw) or (B, N_raw)
    tau: torch.Tensor,            # (B, M) [s]
    t_pre: float = 0.05,          # 50 ms
    t_post: float = 0.25,         # 250 ms
    n_pts: int = 301,             # 301 点 (1000 Hz, 1ms 间隔)
    t_total: float = 60.0,        # 总时程 (s)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    以可微双线性网格采样 (F.grid_sample) 从原生高频波形中精确截取
    各簇局部毫秒级连续波包 [tau_j - t_pre, tau_j + t_post]。

    返回:
        sampled: (B, M, C, n_pts) 连续重采样波包
        rel_offsets: (n_pts,) 相对时间偏移 [s]
    """
    if raw_wave.dim() == 2:
        raw_wave = raw_wave.unsqueeze(1)
    B, C, N_raw = raw_wave.shape
    _, M = tau.shape
    device = raw_wave.device
    dtype = raw_wave.dtype

    # 构造相对时间网格: [-t_pre, +t_post], 保证与 raw_wave 的 device 和 dtype 匹配
    rel_offsets = torch.linspace(-t_pre, t_post, n_pts, device=device, dtype=dtype)  # (n_pts,)

    # 各簇采样时刻: t_pts[b, j, k] = tau[b, j] + rel_offsets[k]
    tau_aligned = tau.to(device=device, dtype=dtype)
    t_pts = tau_aligned.unsqueeze(-1) + rel_offsets.view(1, 1, -1)  # (B, M, n_pts)

    # 归一化至 grid_sample 所需 [-1, 1] 坐标: x = 2 * t / t_total - 1
    grid_x = (2.0 * t_pts / t_total) - 1.0
    grid_x = torch.clamp(grid_x, -1.0, 1.0)
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, M, n_pts, 2)

    wave_4d = raw_wave.unsqueeze(2)  # (B, C, 1, N_raw)
    sampled = F.grid_sample(
        wave_4d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, C, M, n_pts)
    sampled = sampled.permute(0, 2, 1, 3).contiguous()  # (B, M, C, n_pts)

    return sampled, rel_offsets


class PhysicsQueryEmbedding(nn.Module):
    """
    方案一 (Physics-Anchored Query Embedding, TG-DIS-QueryEmbed):
    - 连续物理 Query: q_j = MLP([x_j, tau_j, a]) in R^d;
    - 在原生 1000 Hz 高频波包 Token 序列上执行带时空声学衰减偏置的 Cross-Attention:
      h_j = sum_t Softmax( (q_j k(t)^T) / sqrt(d) - lambda * |t - tau_j| ) v(t)
    - 充当自适应高频连续积分滤波器，消除波前截断误差。
    """
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        max_nc: int = 6,
        n_win_pts: int = 301,
        t_pre: float = 0.05,
        t_post: float = 0.25,
        t_total: float = 60.0,
        L_total: float = 5000.0,
        ts: float = 1.0,
        init_lambda: float = 10.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.max_nc = max_nc
        self.n_win_pts = n_win_pts
        self.t_pre = t_pre
        self.t_post = t_post
        self.t_total = t_total
        self.L_total = L_total
        self.ts = ts

        # 1. 物理坐标 Query 映射网络: [x_norm, tau_norm, a_norm] -> d_model
        self.query_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 2. 原生高频波包 Token 特征抽取 (1D Conv)
        self.token_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

        # 3. Key & Value 投影
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        # 4. 可学习声学衰减率参数 lambda = softplus(raw_lambda) + 0.5
        # 初始值 init_lambda = 10.0 s^-1
        init_raw = math.log(math.exp(max(init_lambda - 0.5, 1e-4)) - 1.0)
        self.raw_lambda = nn.Parameter(torch.tensor(float(init_raw), dtype=torch.float32))

        # 5. 残差与前馈层
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def get_lambda(self) -> torch.Tensor:
        """获取物理尺度的正数声学衰减系数 lambda (s^-1)"""
        return F.softplus(self.raw_lambda) + 0.5

    def forward(
        self,
        wave: torch.Tensor,                            # (B, C, N_t)
        raw_wave: Optional[torch.Tensor] = None,       # (B, C, N_raw) 原生 1000 Hz 波形
        positions: Optional[torch.Tensor] = None,      # (B, M) [m]
        norm_positions: Optional[torch.Tensor] = None, # (B, M) [0, 1]
        cond: Optional[torch.Tensor] = None,           # (B, 3)
        mask: Optional[torch.Tensor] = None,           # (B, M) bool
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions is None and norm_positions is None:
            raise ValueError("必须提供 positions 或 norm_positions 之一")
        elif positions is None and norm_positions is not None:
            positions = norm_positions * self.L_total
        elif norm_positions is None and positions is not None:
            norm_positions = positions / self.L_total

        B, M = positions.shape
        if cond is None:
            cond = torch.zeros((B, 3), device=positions.device, dtype=positions.dtype)

        tau = compute_theoretical_arrival_times(
            positions=positions,
            cond=cond,
            ts=self.ts,
            mask=mask,
        )  # (B, M)
        source_wave = raw_wave if raw_wave is not None else wave

        # 提取 301 点高频波包
        packets, rel_offsets = extract_raw_wave_packet(
            raw_wave=source_wave,
            tau=tau,
            t_pre=self.t_pre,
            t_post=self.t_post,
            n_pts=self.n_win_pts,
            t_total=self.t_total,
        )  # packets: (B, M, C, K), rel_offsets: (K,)
        K = self.n_win_pts

        # 构造物理 Query 向量 q_j = MLP([x_norm, tau_norm, a_norm])
        a_norm = cond[:, 1:2].expand(-1, M).unsqueeze(-1)      # (B, M, 1)
        x_norm = norm_positions.unsqueeze(-1)                  # (B, M, 1)
        tau_norm = (tau / self.t_total).unsqueeze(-1)          # (B, M, 1)
        query_input = torch.cat([x_norm, tau_norm, a_norm], dim=-1)  # (B, M, 3)
        q = self.query_mlp(query_input)                        # (B, M, d_model)

        # 波包 Token 特征提取
        flat_packets = packets.view(B * M, self.in_channels, K)
        flat_tokens = self.token_conv(flat_packets)            # (B*M, d_model, K)
        tokens_seq = flat_tokens.permute(0, 2, 1).contiguous()  # (B*M, K, d_model)
        tokens_seq = tokens_seq.view(B, M, K, self.d_model)    # (B, M, K, d_model)

        # Key 与 Value
        k = self.to_k(tokens_seq)                              # (B, M, K, d_model)
        v = self.to_v(tokens_seq)                              # (B, M, K, d_model)

        # 带声学衰减偏置的连续 Cross-Attention
        scale = 1.0 / math.sqrt(self.d_model)
        sim = (q.unsqueeze(2) * k).sum(dim=-1) * scale         # (B, M, K)

        decay_lambda = self.get_lambda()
        time_penalty = decay_lambda * torch.abs(rel_offsets).view(1, 1, K)  # (1, 1, K)
        attn_logits = sim - time_penalty                       # (B, M, K)
        attn_weights = F.softmax(attn_logits, dim=-1)          # (B, M, K)

        # 连续加权积分: h_j = sum_k attn[k] * v[k]
        h = (attn_weights.unsqueeze(-1) * v).sum(dim=2)        # (B, M, d_model)

        # 残差与前馈融合
        h = self.norm(h + q)
        h = self.norm_ffn(h + self.ffn(h))                     # (B, M, d_model)

        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), h, torch.zeros_like(h))

        return h, tau


class FourierFeatureEmbedding(nn.Module):
    """
    方案二 (Continuous Fourier Feature Embedding, TG-DIS-FourierEmbed):
    - 多尺度高频连续傅里叶字典:
      gamma(t) = [sin(omega_0 t), cos(omega_0 t), ..., sin(omega_{M-1} t), cos(omega_{M-1} t)]^T
      覆盖基频 0.0725 Hz (井筒声学基频 a/(4L)) 至 500 Hz (1000 Hz 采样奈奎斯特截止频);
    - 将连续时间傅里叶坐标嵌入注入波形时序特征中，克服神经网络在波前突变拟合中的“低频偏置”，
      赋予模型亚毫秒级连续时间感知。
    """
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        max_nc: int = 6,
        n_win_pts: int = 301,
        num_freqs: int = 16,
        f_min: float = 0.0725,   # a / (4L) ~ 1450 / 20000 = 0.0725 Hz
        f_max: float = 500.0,    # 1000 Hz 采样 Nyquist 频率
        t_pre: float = 0.05,
        t_post: float = 0.25,
        t_total: float = 60.0,
        L_total: float = 5000.0,
        ts: float = 1.0,
        use_relative_time: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.max_nc = max_nc
        self.n_win_pts = n_win_pts
        self.num_freqs = num_freqs
        self.t_pre = t_pre
        self.t_post = t_post
        self.t_total = t_total
        self.L_total = L_total
        self.ts = ts
        self.use_relative_time = bool(use_relative_time)

        # 几何级数频带划分 (从 0.0725 Hz 对数均匀延伸至 500.0 Hz)
        log_f = torch.linspace(math.log(f_min), math.log(f_max), num_freqs)
        freqs = torch.exp(log_f)  # (num_freqs,)
        omegas = 2.0 * math.pi * freqs  # (num_freqs,) [rad/s]
        self.register_buffer("omegas", omegas)

        fourier_dim = 2 * num_freqs  # sin + cos

        # 傅里叶高频坐标特征投影层
        self.fourier_proj = nn.Sequential(
            nn.Linear(fourier_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # 局部波形特征抽取器
        self.wave_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )

        # 融合卷积与池化
        self.joint_conv = nn.Sequential(
            nn.Conv1d(32 + 32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 物理几何辅助注入
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(64 + 32, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_fourier_features(self, t_pts: torch.Tensor) -> torch.Tensor:
        """
        计算连续时间点的多频带傅里叶基 gamma(t):
        t_pts: (B, M, K)
        返回: (B, M, K, 2*num_freqs)
        """
        angles = t_pts.unsqueeze(-1) * self.omegas.view(1, 1, 1, -1)
        sins = torch.sin(angles)
        coss = torch.cos(angles)
        return torch.cat([sins, coss], dim=-1)  # (B, M, K, 2*num_freqs)

    def forward(
        self,
        wave: torch.Tensor,                            # (B, C, N_t)
        raw_wave: Optional[torch.Tensor] = None,       # (B, C, N_raw)
        positions: Optional[torch.Tensor] = None,      # (B, M) [m]
        norm_positions: Optional[torch.Tensor] = None, # (B, M) [0, 1]
        cond: Optional[torch.Tensor] = None,           # (B, 3)
        mask: Optional[torch.Tensor] = None,           # (B, M) bool
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions is None and norm_positions is None:
            raise ValueError("必须提供 positions 或 norm_positions 之一")
        elif positions is None and norm_positions is not None:
            positions = norm_positions * self.L_total
        elif norm_positions is None and positions is not None:
            norm_positions = positions / self.L_total

        B, M = positions.shape
        if cond is None:
            cond = torch.zeros((B, 3), device=positions.device, dtype=positions.dtype)

        tau = compute_theoretical_arrival_times(
            positions=positions,
            cond=cond,
            ts=self.ts,
            mask=mask,
        )

        source_wave = raw_wave if raw_wave is not None else wave

        packets, rel_offsets = extract_raw_wave_packet(
            raw_wave=source_wave,
            tau=tau,
            t_pre=self.t_pre,
            t_post=self.t_post,
            n_pts=self.n_win_pts,
            t_total=self.t_total,
        )
        K = self.n_win_pts

        if self.use_relative_time:
            t_pts = rel_offsets.view(1, 1, -1).expand(B, M, -1)
        else:
            t_pts = tau.unsqueeze(-1) + rel_offsets.view(1, 1, -1)  # (B, M, K)
        fourier_feats = self.compute_fourier_features(t_pts)    # (B, M, K, 2*num_freqs)
        fourier_emb = self.fourier_proj(fourier_feats)          # (B, M, K, 32)
        fourier_emb = fourier_emb.permute(0, 1, 3, 2).contiguous()  # (B, M, 32, K)

        flat_packets = packets.view(B * M, self.in_channels, K)
        wave_feats = self.wave_conv(flat_packets).view(B, M, 32, K)  # (B, M, 32, K)

        joint_in = torch.cat([wave_feats, fourier_emb], dim=2).view(B * M, 64, K)
        h_wave = self.joint_conv(joint_in).squeeze(-1).view(B, M, 64)  # (B, M, 64)

        a_norm = cond[:, 1:2].expand(-1, M).unsqueeze(-1)
        x_norm = norm_positions.unsqueeze(-1)
        tau_norm = (tau / self.t_total).unsqueeze(-1)
        coord_feat = self.coord_mlp(torch.cat([x_norm, tau_norm, a_norm], dim=-1))  # (B, M, 32)

        tokens = self.out_proj(torch.cat([h_wave, coord_feat], dim=-1))  # (B, M, d_model)

        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        return tokens, tau


class SincNetAcousticFilterbank(nn.Module):
    """
    方案三 (SincNet-style Acoustic Filterbank Embedding, TG-DIS-SincEmbed):
    - 可学习连续带通 Sinc 滤波器组:
      g(t, f_low, f_high) = 2*f_high*sinc(2*f_high*t) - 2*f_low*sinc(2*f_low*t)
    - 截止频率 [f_low, f_high] 设为可学习参数;
    - 初始频段严格绑定井筒声学奇数次驻波谐频 f_k = (2k-1)*a / (4L);
    - 端到端完成声学物理包络提取与无损降维。
    """
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        max_nc: int = 6,
        n_filters: int = 32,
        kernel_size: int = 65,
        fs: float = 1000.0,      # 1000 Hz 原生采样率
        f_min: float = 0.0725,   # 一阶基频 a/(4L)
        f_max: float = 450.0,    # 上限截止
        a_ref: float = 1450.0,
        L_total: float = 5000.0,
        t_pre: float = 0.05,
        t_post: float = 0.25,
        t_total: float = 60.0,
        ts: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.max_nc = max_nc
        self.n_filters = n_filters
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.fs = fs
        self.t_pre = t_pre
        self.t_post = t_post
        self.t_total = t_total
        self.L_total = L_total
        self.ts = ts

        half_len = (self.kernel_size - 1) // 2
        t_axis = torch.arange(-half_len, half_len + 1, dtype=torch.float32) / fs  # (kernel_size,)
        self.register_buffer("t_axis", t_axis)

        window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * torch.arange(self.kernel_size, dtype=torch.float32) / (self.kernel_size - 1))
        self.register_buffer("window", window.view(1, 1, -1))

        init_f_low, init_band = self._init_acoustic_harmonic_bands(
            n_filters=n_filters,
            a_ref=a_ref,
            L=L_total,
            f_min=f_min,
            f_max=f_max,
        )

        # 数值稳定逆 softplus，防止大频率下 exp(f) 溢出为 inf/NaN
        def _inv_softplus(val: torch.Tensor, threshold: float = 20.0) -> torch.Tensor:
            return torch.where(
                val > threshold,
                val,
                torch.log(torch.clamp(torch.exp(torch.clamp(val, max=threshold)) - 1.0, min=1e-6))
            )

        raw_f_low = _inv_softplus(torch.clamp(init_f_low - 0.01, min=1e-4))
        raw_band = _inv_softplus(torch.clamp(init_band - 0.5, min=1e-4))
        self.raw_f_low = nn.Parameter(raw_f_low)
        self.raw_band = nn.Parameter(raw_band)

        self.envelope_conv = nn.Sequential(
            nn.Conv1d(n_filters * in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Sequential(
            nn.Linear(64 + 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _init_acoustic_harmonic_bands(
        self,
        n_filters: int,
        a_ref: float,
        L: float,
        f_min: float,
        f_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = a_ref / (4.0 * L)  # ~0.0725 Hz
        k_indices = 1.0 + (3000.0 - 1.0) * (torch.linspace(0.0, 1.0, n_filters) ** 1.8)
        f_centers = (2.0 * k_indices - 1.0) * f1
        f_centers = torch.clamp(f_centers, min=f_min, max=f_max)

        bands = torch.clamp(f_centers * 0.35 + 2.0, min=2.0, max=80.0)
        f_low = torch.clamp(f_centers - 0.5 * bands, min=0.05, max=f_max - 5.0)
        return f_low, bands

    def get_filter_kernels(self) -> torch.Tensor:
        f_low = F.softplus(self.raw_f_low) + 0.01                  # (n_filters,)
        band = F.softplus(self.raw_band) + 0.5                     # (n_filters,)
        f_high = torch.clamp(f_low + band, max=self.fs * 0.495)    # (n_filters,)

        f_low_hz = f_low.view(self.n_filters, 1, 1)
        f_high_hz = f_high.view(self.n_filters, 1, 1)
        t = self.t_axis.view(1, 1, -1)  # (1, 1, kernel_size)

        high_sinc = 2.0 * f_high_hz * torch.sinc(2.0 * f_high_hz * t)
        low_sinc = 2.0 * f_low_hz * torch.sinc(2.0 * f_low_hz * t)
        bandpass = (high_sinc - low_sinc) * self.window  # (n_filters, 1, kernel_size)

        norm_factor = torch.sqrt(torch.sum(bandpass ** 2, dim=-1, keepdim=True) + 1e-8)
        return bandpass / norm_factor

    def forward(
        self,
        wave: torch.Tensor,                            # (B, C, N_t)
        raw_wave: Optional[torch.Tensor] = None,       # (B, C, N_raw)
        positions: Optional[torch.Tensor] = None,      # (B, M) [m]
        norm_positions: Optional[torch.Tensor] = None, # (B, M) [0, 1]
        cond: Optional[torch.Tensor] = None,           # (B, 3)
        mask: Optional[torch.Tensor] = None,           # (B, M) bool
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions is None and norm_positions is None:
            raise ValueError("必须提供 positions 或 norm_positions 之一")
        elif positions is None and norm_positions is not None:
            positions = norm_positions * self.L_total
        elif norm_positions is None and positions is not None:
            norm_positions = positions / self.L_total

        B, M = positions.shape
        if cond is None:
            cond = torch.zeros((B, 3), device=positions.device, dtype=positions.dtype)

        tau = compute_theoretical_arrival_times(
            positions=positions,
            cond=cond,
            ts=self.ts,
            mask=mask,
        )
        source_wave = raw_wave if raw_wave is not None else wave

        packets, _ = extract_raw_wave_packet(
            raw_wave=source_wave,
            tau=tau,
            t_pre=self.t_pre,
            t_post=self.t_post,
            n_pts=301,
            t_total=self.t_total,
        )
        K = packets.shape[-1]

        filters = self.get_filter_kernels()  # (N_f, 1, L_k)
        pad = (self.kernel_size - 1) // 2

        flat_packets = packets.view(B * M, self.in_channels, K)
        filtered_channels = []
        for c in range(self.in_channels):
            ch_wave = flat_packets[:, c : c + 1, :]
            ch_filtered = F.conv1d(ch_wave, filters, padding=pad)
            filtered_channels.append(ch_filtered)
        all_filtered = torch.cat(filtered_channels, dim=1)

        envelope = torch.sqrt(all_filtered ** 2 + 1e-6)  # (B*M, N_f * C, K)
        h_env = self.envelope_conv(envelope).squeeze(-1).view(B, M, 64)  # (B, M, 64)

        a_norm = cond[:, 1:2].expand(-1, M).unsqueeze(-1)
        x_norm = norm_positions.unsqueeze(-1)
        tau_norm = (tau / self.t_total).unsqueeze(-1)
        phys_in = torch.cat([h_env, x_norm, tau_norm, a_norm], dim=-1)  # (B, M, 67)

        tokens = self.proj(phys_in)  # (B, M, d_model)

        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        return tokens, tau


class BaselineResampleEmbedding(nn.Module):
    """
    基准方案 (BaselineResampleEmbedding):
    复用前期研究中的 RelativeWindowGating 机制，在 4096 点离线降采样波形上执行网格采样与 1D 卷积，
    用于严格单变量消融对标基准。
    """
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        max_nc: int = 6,
        n_win_pts: int = 128,
        t_pre: float = 0.05,
        t_post: float = 0.25,
        t_total: float = 60.0,
        L_total: float = 5000.0,
        ts: float = 1.0,
    ):
        super().__init__()
        self.L_total = L_total
        self.ts = ts
        self.extractor = RelativeWindowGating(
            in_channels=in_channels,
            d_model=d_model,
            n_win_pts=n_win_pts,
            t_pre=t_pre,
            t_post=t_post,
            t_total=t_total,
        )

    def forward(
        self,
        wave: torch.Tensor,
        raw_wave: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        norm_positions: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions is None and norm_positions is None:
            raise ValueError("必须提供 positions 或 norm_positions 之一")
        elif positions is None and norm_positions is not None:
            positions = norm_positions * self.L_total
        elif norm_positions is None and positions is not None:
            norm_positions = positions / self.L_total

        B, M = positions.shape
        if cond is None:
            cond = torch.zeros((B, 3), device=positions.device, dtype=positions.dtype)

        tau = compute_theoretical_arrival_times(
            positions=positions,
            cond=cond,
            ts=self.ts,
            mask=mask,
        )

        tokens = self.extractor(wave=wave, tau=tau, mask=mask)
        return tokens, tau
