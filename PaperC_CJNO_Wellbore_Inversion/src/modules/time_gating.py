# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.modules.time_gating

物理到时对齐提取模块 (Time-Gating Modules):
针对密集多簇 (5~40m) 回波时差小、频域严重混叠的瓶颈，实现以声学往返到时 tau_j = t_s + 2*x_j/a 为引导的三大多模态到时窗提取机制:
1. RelativeWindowGating (模式 1):
   时域截取 [tau_j - 50ms, tau_j + 250ms]，可微一维重采样 (F.grid_sample) 归一化至固定点数 (128点)，
   使每簇首波到达严格对齐到 t=0 (即窗口 50ms 处)；
2. GaussianGating (模式 2):
   在全时程波形上施加连续可微软高斯窗 w_j(t) = exp(-(t - tau_j)^2 / (2*sigma_j^2))，窗宽 sigma_j 可学习；
3. CepstrumPatchGating (模式 3):
   在 2D 连续倒谱时空切片上提取以 (tau_j, x_j) 为中心的局部图块，经 2D 卷积/Patch 投影为特征；
4. TimeGatingModule: 统一门控调度模块，支持一键消融切换。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_theoretical_arrival_times(
    positions: torch.Tensor,       # (B, M) [m]
    cond: torch.Tensor,            # (B, 3)
    ts: float = 1.0,
    a_ref: float = 1450.0,
    a_scale: float = 20.0,
    mask: Optional[torch.Tensor] = None, # (B, M) bool
) -> torch.Tensor:
    """
    依据实测声速 a、关泵时刻 t_s 及完井已知深度 x_j 精确计算理论往返到时:
    tau_j = t_s + 2 * x_j / a
    """
    # a = cond[:, 1:2] * 20.0 + 1450.0 (m/s)
    a = cond[:, 1:2] * a_scale + a_ref
    a = torch.clamp(a, min=1000.0, max=2000.0)
    tau = ts + (2.0 * positions) / a  # (B, M)
    if mask is not None:
        tau = torch.where(mask, tau, torch.full_like(tau, ts))
    return tau


class RelativeWindowGating(nn.Module):
    """
    模式 1: 相对到时窗提取机制 (relative_window)
    截取 [tau_j - 50ms, tau_j + 250ms]，以可微 1D grid_sample 归一化至 N_w=128 点，
    严格对齐波前起始并经 1D ConvNet 投影为 d_model 维簇表征。
    """
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        n_win_pts: int = 128,
        t_pre: float = 0.05,   # 50 ms
        t_post: float = 0.25,  # 250 ms
        t_total: float = 60.0, # 全波形总时程 (s，对齐真实 MOC 60s 仿真)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_win_pts = n_win_pts
        self.t_pre = t_pre
        self.t_post = t_post
        self.t_total = t_total
        self.win_len = t_pre + t_post

        # 归一化网格相对偏移 [0, 1]
        self.register_buffer("offset_grid", torch.linspace(0.0, 1.0, n_win_pts))

        # 1D 局部时序特征编码器 (GroupNorm 保证变 batch 下的数值稳定性)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, d_model)

    def extract_windows(
        self,
        wave: torch.Tensor,  # (B, C, N_t)
        tau: torch.Tensor,   # (B, M)
    ) -> torch.Tensor:
        """
        利用可微 F.grid_sample 提取所有样本与簇的重采样窗口:
        返回: (B, M, C, n_win_pts)
        """
        B, C, N_t = wave.shape
        _, M = tau.shape

        # t_start: (B, M, 1), t_end: (B, M, 1)
        t_start = (tau - self.t_pre).unsqueeze(-1)
        t_pts = t_start + self.win_len * self.offset_grid.view(1, 1, -1) # (B, M, n_win_pts)

        # 归一化到 [-1, 1] 坐标 (grid_sample 要求)
        grid_x = (2.0 * t_pts / self.t_total) - 1.0
        grid_x = torch.clamp(grid_x, -1.0, 1.0)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1) # (B, M, n_win_pts, 2)

        # 直接输入 (B, C, 1, N_t) 与 (B, M, n_win_pts, 2)，grid_sample 产生 (B, C, M, n_win_pts)
        wave_4d = wave.unsqueeze(2) # (B, C, 1, N_t)
        sampled = F.grid_sample(
            wave_4d,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ) # (B, C, M, n_win_pts)
        sampled = sampled.permute(0, 2, 1, 3).contiguous() # (B, M, C, n_win_pts)
        return sampled

    def forward(
        self,
        wave: torch.Tensor,                # (B, C, N_t)
        tau: torch.Tensor,                 # (B, M)
        mask: Optional[torch.Tensor] = None, # (B, M) bool
    ) -> torch.Tensor:
        B, M = tau.shape
        # 1. 可微网格采样提取窗口
        windows = self.extract_windows(wave, tau) # (B, M, C, n_win_pts)

        # 2. 局部卷积特征抽取
        flat_windows = windows.view(B * M, self.in_channels, self.n_win_pts)
        feats = self.conv(flat_windows).squeeze(-1) # (B*M, 64)
        tokens = self.proj(feats).view(B, M, self.d_model)

        # 3. 掩码清零
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return tokens


class GaussianGating(nn.Module):
    """
    模式 2: 可微软高斯窗到时门控机制 (gaussian_gating)
    在全时程波形上施加连续可微软高斯窗:
    w_j(t) = exp( - (t - tau_j)^2 / (2 * sigma_j^2) )
    窗宽 sigma_j 可学习 (初始化为 0.08s, 即约 80ms)，经卷积与自适应池化映射为簇表征。
    """
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        max_nc: int = 6,
        init_sigma: float = 0.08,
        t_total: float = 60.0,
        n_time: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.max_nc = max_nc
        self.t_total = t_total
        self.n_time = n_time

        # 可学习窗宽参数 raw_sigma, 约束 sigma = softplus(raw_sigma) + 0.01s
        # 逆 softplus: log(exp(init_sigma - 0.01) - 1.0) ~ -2.624 严格精确对应 init_sigma = 0.08s
        init_raw = np.log(np.exp(max(init_sigma - 0.01, 1e-4)) - 1.0)
        self.raw_sigma = nn.Parameter(torch.full((max_nc,), float(init_raw), dtype=torch.float32))

        # 时间轴网格 [0, T]
        self.register_buffer("time_grid", torch.linspace(0.0, t_total, n_time).view(1, 1, 1, n_time))

        # 全波形门控特征提取器 (多级大步长 1D 卷积降采样)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=4, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, d_model)

    def get_sigma(self) -> torch.Tensor:
        """获取物理尺度的正数高斯窗宽 (秒)"""
        return F.softplus(self.raw_sigma) + 0.01

    def forward(
        self,
        wave: torch.Tensor,                  # (B, C, N_t)
        tau: torch.Tensor,                   # (B, M)
        mask: Optional[torch.Tensor] = None, # (B, M) bool
    ) -> torch.Tensor:
        B, C, N_t = wave.shape
        _, M = tau.shape

        # 1. 计算各簇高斯软窗权重: (B, M, 1, N_t)
        sigma = self.get_sigma()[:M].view(1, M, 1, 1) # (1, M, 1, 1)
        tau_exp = tau.view(B, M, 1, 1)
        diff_sq = (self.time_grid - tau_exp) ** 2
        weights = torch.exp(-diff_sq / (2.0 * (sigma ** 2))) # (B, M, 1, N_t)

        # 2. 施加软高斯门控: (B, M, C, N_t)
        gated_wave = wave.unsqueeze(1) * weights # (B, M, C, N_t)

        # 3. 卷积特征提取
        flat_wave = gated_wave.view(B * M, C, N_t)
        feats = self.conv(flat_wave).squeeze(-1) # (B*M, 64)
        tokens = self.proj(feats).view(B, M, self.d_model)

        # 4. 掩码清零
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return tokens


class CepstrumPatchGating(nn.Module):
    """
    模式 3: 2D 倒谱时空切片图块提取机制 (ceps_patch)
    融合时域波形与空间反射倒谱，构建 2D 时空连续特征图 (时间轴 t x 空间轴 x)，
    并在 (tau_j, x_j) 处以双线性可微网格采样提取局部图块 (Patch)，投影为簇表征。
    """
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        patch_size: int = 16,
        n_time_2d: int = 64,
        n_space_2d: int = 64,
        t_total: float = 60.0,
        L_total: float = 5000.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_time_2d = n_time_2d
        self.n_space_2d = n_space_2d
        self.t_total = t_total
        self.L_total = L_total

        # 时域与倒谱特征投影至 2D 底图
        self.time_proj = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(n_time_2d),
        )
        self.space_proj = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(n_space_2d),
        )
        # 2D 时空混合卷积 (H=n_time_2d, W=n_space_2d)
        self.joint_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )

        # 局部 Patch 采样相对网格:
        # 时间窗跨度 +/- 150ms (Height), 空间窗跨度 +/- 150m (Width)
        self.du = (2.0 * 0.15) / self.t_total   # 时间轴归一化半窗 (Height/y)
        self.dv = (2.0 * 150.0) / self.L_total # 空间轴归一化半窗 (Width/x)
        rel_u = torch.linspace(-self.du, self.du, patch_size) # row offset (time)
        rel_v = torch.linspace(-self.dv, self.dv, patch_size) # col offset (space)
        grid_u, grid_v = torch.meshgrid(rel_u, rel_v, indexing="ij")
        # PyTorch grid_sample 要求坐标格式: (x=col=space, y=row=time)
        self.register_buffer("local_patch_grid", torch.stack([grid_v, grid_u], dim=-1).unsqueeze(0).unsqueeze(0))

        # Patch 编码器
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, d_model)

    def forward(
        self,
        wave: torch.Tensor,                  # (B, C, N_t)
        cepstrum: torch.Tensor,              # (B, 1, N_ceps)
        tau: torch.Tensor,                   # (B, M)
        positions: torch.Tensor,             # (B, M)
        mask: Optional[torch.Tensor] = None, # (B, M) bool
    ) -> torch.Tensor:
        B, C, N_t = wave.shape
        _, M = tau.shape

        # 1. 投影构建 2D 时空底图: (B, 16, N_t_2d, N_x_2d)
        ft = self.time_proj(wave)      # (B, 16, n_time_2d)
        fx = self.space_proj(cepstrum) # (B, 16, n_space_2d)
        # 外积融合: (B, 16, n_time_2d, n_space_2d)
        feat_2d = ft.unsqueeze(-1) + fx.unsqueeze(2)
        s_2d = self.joint_conv(feat_2d) # (B, 32, n_time_2d, n_space_2d)

        # 2. 计算各簇在 2D 坐标图中的归一化中心 (v_j: 空间轴 x, u_j: 时间轴 y) in [-1, 1]
        u_j = (2.0 * tau / self.t_total) - 1.0        # (B, M) 时间轴 (对应 grid y/row)
        v_j = (2.0 * positions / self.L_total) - 1.0  # (B, M) 空间轴 (对应 grid x/col)
        center = torch.stack([v_j, u_j], dim=-1)      # (B, M, 2) (x=space, y=time)
        center = center.view(B, M, 1, 1, 2)

        # 3. 构造各簇 Patch 采样网格: (B, M*P, P, 2) 直接在批次维度切片
        sample_grid = center + self.local_patch_grid
        sample_grid = torch.clamp(sample_grid, -1.0, 1.0)
        sample_grid_flat = sample_grid.view(B, M * self.patch_size, self.patch_size, 2)

        # 4. grid_sample 直接对 s_2d 提取 Patch，无需膨胀复制底图
        patches_flat = F.grid_sample(
            s_2d,
            sample_grid_flat,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ) # (B, 32, M*P, P)
        patches = patches_flat.view(B, 32, M, self.patch_size, self.patch_size).permute(0, 2, 1, 3, 4).contiguous()
        flat_patches = patches.view(B * M, 32, self.patch_size, self.patch_size)

        # 5. Patch 投影为簇表征
        patch_feats = self.patch_encoder(flat_patches).squeeze(-1).squeeze(-1) # (B*M, 64)
        tokens = self.proj(patch_feats).view(B, M, self.d_model)

        # 6. 掩码清零
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return tokens


class TimeGatingModule(nn.Module):
    """
    统一时间门控提取调度模块:
    支持 window_mode in ['relative_window', 'gaussian_gating', 'ceps_patch']
    通过统一的 forward 接口输出 (B, M, d_model) 的簇级波前解耦表征。
    """
    VALID_MODES = ("relative_window", "gaussian_gating", "ceps_patch")

    def __init__(
        self,
        window_mode: str = "relative_window",
        in_channels: int = 2,
        d_model: int = 64,
        max_nc: int = 6,
        t_total: float = 60.0,
        L_total: float = 5000.0,
        ts: float = 1.0,
    ):
        super().__init__()
        self.window_mode = str(window_mode).strip().lower()
        if self.window_mode not in self.VALID_MODES:
            raise ValueError(f"不支持的 window_mode: '{window_mode}'，必须在 {self.VALID_MODES} 中选择。")

        self.in_channels = in_channels
        self.d_model = d_model
        self.max_nc = max_nc
        self.t_total = t_total
        self.L_total = L_total
        self.ts = ts

        if self.window_mode == "relative_window":
            self.extractor = RelativeWindowGating(
                in_channels=in_channels,
                d_model=d_model,
                n_win_pts=128,
                t_pre=0.05,
                t_post=0.25,
                t_total=t_total,
            )
        elif self.window_mode == "gaussian_gating":
            self.extractor = GaussianGating(
                in_channels=in_channels,
                d_model=d_model,
                max_nc=max_nc,
                init_sigma=0.08,
                t_total=t_total,
            )
        elif self.window_mode == "ceps_patch":
            self.extractor = CepstrumPatchGating(
                in_channels=in_channels,
                d_model=d_model,
                patch_size=16,
                t_total=t_total,
                L_total=L_total,
            )

    def forward(
        self,
        wave: torch.Tensor,                            # (B, C, N_t)
        cepstrum: torch.Tensor,                        # (B, 1, N_ceps)
        positions: torch.Tensor,                       # (B, M) [m]
        norm_positions: torch.Tensor,                  # (B, M) [0, 1]
        cond: torch.Tensor,                            # (B, 3)
        mask: Optional[torch.Tensor] = None,           # (B, M) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            tokens: (B, M, d_model) 各簇波前特征向量
            tau: (B, M) 各簇理论到达时刻 (s)
        """
        # 1. 理论往返到时计算
        tau = compute_theoretical_arrival_times(
            positions=positions,
            cond=cond,
            ts=self.ts,
            mask=mask,
        )

        # 2. 执行对应到时窗提取
        if self.window_mode == "relative_window":
            tokens = self.extractor(wave=wave, tau=tau, mask=mask)
        elif self.window_mode == "gaussian_gating":
            tokens = self.extractor(wave=wave, tau=tau, mask=mask)
        elif self.window_mode == "ceps_patch":
            tokens = self.extractor(
                wave=wave,
                cepstrum=cepstrum,
                tau=tau,
                positions=positions,
                mask=mask,
            )
        else:
            raise RuntimeError(f"Unknown window_mode {self.window_mode}")

        return tokens, tau
