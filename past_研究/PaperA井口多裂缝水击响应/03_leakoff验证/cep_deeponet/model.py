# -*- coding: utf-8 -*-
"""
model.py — 倒谱 Cep-DeepONet 神经网络模型定义

设计核心：
1. 参照 CMAME 2023 论文《Fourier-DeepONet》的参数空间算子回归思想：
   G: (H_wh, xi) -> y(x), x in [0, L]
2. Branch 网络：
   - 包含 DifferentiableCepstrum 可微倒谱层，提取倒频域回波时滞
   - 双流融合：物理时距映射（将倒谱 quefrency 与初至时间物理重采样至空间网格 x）
   - 多尺度时序卷积与残差特征提取，升维至潜空间特征 b in [B, C, Nx]
3. Trunk 网络：
   - 参数空间 MLP 编码器，将物理工况参数 xi 编码为潜向量 t in [B, C]
4. Merger 融合：
   - 广播逐点乘法 z0 = b ⊙ t (点乘调制)
5. 1D 傅里叶-残差增强解码器 (1D Fourier-Residual Decoder)：
   - 4层 1D 谱卷积 (SpectralConv1d) + 1D 残差卷积块 (ResConv1d)
   - 捕获全井全局多途谐振与局部裂缝奇异跳跃
   - 投影头 Q 输出空间反射率连续剖面 y_hat in [B, Nx]
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableCepstrum(nn.Module):
    """
    可微实倒谱 (Differentiable Real Cepstrum)
    嵌入 Clamp-Log 防爆机制，打通时域与倒频域端到端梯度。
    """
    def __init__(self, eps_threshold: float = 1e-6):
        super().__init__()
        self.eps_threshold = eps_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x shape [B, C, T] 时域信号
        输出: c_q shape [B, C, T] 倒频率序列 (Quefrency domain)
        """
        T = x.size(-1)
        # 1. 快速傅里叶变换到频域
        X_f = torch.fft.rfft(x, dim=-1)
        # 2. 功率谱
        P_f = torch.abs(X_f) ** 2
        # 3. 对数谱与防溢出截断
        P_safe = torch.clamp(P_f, min=self.eps_threshold)
        L_f = torch.log(P_safe)
        # 4. 离散傅里叶逆变换回倒频域
        c_q = torch.fft.irfft(L_f, n=T, dim=-1)
        return c_q


class SpectralConv1d(nn.Module):
    """一维傅里叶谱卷积层 (1D Fourier Spectral Convolution)"""
    def __init__(self, in_channels: int, out_channels: int, modes1: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_fp32 = x.to(torch.float32)

        # 空间傅里叶变换
        x_ft = torch.fft.rfft(x_fp32, dim=-1)

        # 截断并应用频域权重乘法
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        modes = min(self.modes1, x_ft.size(-1))
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])

        # 逆傅里叶变换回到空间域
        x_out = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x_out


class FourierResidualBlock1d(nn.Module):
    """
    1D 傅里叶增强残差块：
    并联 1D 谱卷积 (全局谐振) + 1D 局部卷积残差 (局部跳跃奇异性) + 线性捷径
    """
    def __init__(self, channels: int, modes: int = 32):
        super().__init__()
        self.spectral_conv = SpectralConv1d(channels, channels, modes1=modes)
        self.local_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels)
        )
        self.shortcut = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spectral = self.spectral_conv(x)
        x_local = self.local_conv(x)
        x_sc = self.shortcut(x)
        return self.act(x_spectral + x_local + x_sc)


class BranchNet(nn.Module):
    """
    Branch 分支网络：
    1. 前端可微实倒谱算子
    2. 物理声速空间网格坐标连续重采样 (Grid Sample)
    3. 1D 多尺度卷积特征融合
    输出: b in [B, C, Nx]
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        T_steps: int = 16384,
        Nx: int = 1000,
        L: float = 5000.0,
        dt: float = 0.001,
        default_a: float = 1450.0,
        default_ts: float = 1.0
    ):
        super().__init__()
        self.T_steps = T_steps
        self.Nx = Nx
        self.L = L
        self.dt = dt
        self.a = default_a
        self.ts = default_ts
        self.out_channels = out_channels

        self.cepstrum = DifferentiableCepstrum(eps_threshold=1e-6)

        # 物理空间采样网格预计算 (按 a=1450 m/s)
        # x in [0, L]
        x_grid = torch.linspace(0.0, L, Nx)
        # quefrency q = 2x / a
        q_grid = 2.0 * x_grid / self.a
        idx_q = q_grid / dt
        norm_q = 2.0 * (idx_q / (T_steps - 1)) - 1.0

        # primary arrival time t = ts + 2x / a
        t_grid = self.ts + 2.0 * x_grid / self.a
        idx_t = t_grid / dt
        norm_t = 2.0 * (idx_t / (T_steps - 1)) - 1.0

        # 构造 grid_sample 坐标 [1, 1, Nx, 2] (x, y)，y恒为0
        zeros_y = torch.zeros_like(norm_q)
        grid_q_tensor = torch.stack([norm_q, zeros_y], dim=-1).unsqueeze(0).unsqueeze(0)
        grid_t_tensor = torch.stack([norm_t, zeros_y], dim=-1).unsqueeze(0).unsqueeze(0)

        self.register_buffer('grid_q', grid_q_tensor)
        self.register_buffer('grid_t', grid_t_tensor)

        # 全序列多尺度时序编码网络 (将 T=16384 降采样压缩到 Nx=1000)
        self.time_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),  # 8192
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), # 4096
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3), # 2048
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(Nx)                                # 自适应拉到 Nx=1000
        )

        # 空间物理重采样通道特征卷积
        self.spatial_sample_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )

        # 融合升维
        self.fusion = nn.Sequential(
            nn.Conv1d(64 + 64, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [B, 1, T] 井口水头时程
        返回: b: [B, C, Nx]
        """
        B = H.size(0)
        # 1. 计算可微实倒谱
        C_cep = self.cepstrum(H)  # [B, 1, T]

        # 2. 物理重采样至空间网格 Nx
        H_4d = H.unsqueeze(2)      # [B, 1, 1, T]
        C_4d = C_cep.unsqueeze(2)  # [B, 1, 1, T]

        grid_t_batch = self.grid_t.expand(B, -1, -1, -1)
        grid_q_batch = self.grid_q.expand(B, -1, -1, -1)

        H_sampled = F.grid_sample(H_4d, grid_t_batch, mode='bilinear', align_corners=True).squeeze(2) # [B, 1, Nx]
        C_sampled = F.grid_sample(C_4d, grid_q_batch, mode='bilinear', align_corners=True).squeeze(2) # [B, 1, Nx]

        phys_feat = torch.cat([H_sampled, C_sampled], dim=1) # [B, 2, Nx]
        phys_encoded = self.spatial_sample_encoder(phys_feat) # [B, 64, Nx]

        # 3. 全序列多尺度编码
        full_signal = torch.cat([H, C_cep], dim=1)           # [B, 2, T]
        time_encoded = self.time_encoder(full_signal)         # [B, 64, Nx]

        # 4. 空间特征拼接与升维
        b = self.fusion(torch.cat([phys_encoded, time_encoded], dim=1)) # [B, C, Nx]
        return b


class TrunkNet(nn.Module):
    """
    Trunk 主干网络：参数空间物理调制器
    将物理工况参数 xi (8维) 映射为潜空间特征调制向量 t in [B, C]
    """
    def __init__(self, in_dim: int = 8, out_channels: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
            nn.GELU()
        )

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        xi: [B, D_param]
        返回: t: [B, C]
        """
        return self.net(xi)


class CepDeepONet(nn.Module):
    """
    倒谱 Cep-DeepONet 整体反演模型架构 (CMAME 2023 Fourier-DeepONet 变体)
    """
    def __init__(
        self,
        T_steps: int = 16384,
        Nx: int = 1000,
        L: float = 5000.0,
        param_dim: int = 8,
        latent_channels: int = 64,
        fourier_modes: int = 32,
        num_fourier_layers: int = 4
    ):
        super().__init__()
        self.Nx = Nx
        self.latent_channels = latent_channels

        # 1. Branch 网络
        self.branch_net = BranchNet(
            in_channels=1,
            out_channels=latent_channels,
            T_steps=T_steps,
            Nx=Nx,
            L=L
        )

        # 2. Trunk 网络
        self.trunk_net = TrunkNet(
            in_dim=param_dim,
            out_channels=latent_channels,
            hidden_dim=latent_channels
        )

        # 3. 解码网络：多层 1D 傅里叶-残差增强块
        decoder_blocks = []
        for _ in range(num_fourier_layers):
            decoder_blocks.append(
                FourierResidualBlock1d(channels=latent_channels, modes=fourier_modes)
            )
        self.decoder = nn.Sequential(*decoder_blocks)

        # 4. 最终投影头 Q: 映射为裂缝反射强度分布 [0, 1]
        self.projection = nn.Sequential(
            nn.Conv1d(latent_channels, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()  # 约束在 [0, 1] 高斯平滑峰值区间
        )

    def forward(self, H: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """
        输入:
            H:  [B, 1, T] 井口水击水头时程
            xi: [B, param_dim] 物理工况参数向量
        输出:
            y_pred: [B, Nx] 空间深度裂缝反射率剖面
        """
        # Branch 编码
        b = self.branch_net(H)               # [B, C, Nx]

        # Trunk 编码
        t = self.trunk_net(xi)               # [B, C]

        # Merger 操作：逐点乘法调制 (Pointwise Multiplication)
        z0 = b * t.unsqueeze(-1)             # [B, C, Nx]

        # 傅里叶-残差解码
        z = self.decoder(z0)                 # [B, C, Nx]

        # 投影输出
        y_pred = self.projection(z).squeeze(1) # [B, Nx]
        return y_pred


if __name__ == '__main__':
    # 架构自检与维度测试
    B, T, Nx = 2, 16384, 1000
    H = torch.randn(B, 1, T, requires_grad=True)
    xi = torch.randn(B, 8, requires_grad=True)

    model = CepDeepONet(T_steps=T, Nx=Nx)
    y_pred = model(H, xi)

    loss = y_pred.sum()
    loss.backward()

    print("Model forward success!")
    print("y_pred shape:", y_pred.shape)
    print("H grad norm:", H.grad.norm().item())
    print("xi grad norm:", xi.grad.norm().item())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")
    print("model.py self-test PASSED!")
