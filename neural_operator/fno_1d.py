# -*- coding: utf-8 -*-
"""
neural_operator/fno_1d.py — 一维傅里叶神经算子 (Fourier Neural Operator, FNO-1D) 物理正演代理模型

物理理论基础：
在井筒水击波动力学中，井口压力响应 H_wh(t) 可以看作初始停泵阶跃激励与井筒-多缝系统传递函数（或者空间格林函数）的卷积。
傅里叶神经算子 (FNO) 核心思想是在频域通过截断的高频傅里叶模式 (Modes) 学习这种非局部、全局反射混响的算子映射：
    (F_w * v)(x) = F^{-1}( R_w · F(v) )(x)
    
输入特征映射 (in_channels = 4)：
    Channel 0: 归一化时间步序号 t / T_max
    Channel 1: 停泵激励指示信号 (t < ts 为 1，t >= ts 为 0)
    Channel 2: 柔度反射特征脉冲序列 —— 在预测反射时间 t_arr(k) 处分布由 log10(Cf) 加权的高斯脉冲
    Channel 3: 滤失反射特征脉冲序列 —— 在预测反射时间 t_arr(k) 处分布由 log10(kleak) 加权的高斯脉冲

输出：
    预测的井口水头时序信号 H_wh(t) [m]
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """一维傅里叶谱卷积层 (1D Fourier Spectral Convolution Layer)"""
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 保留的低频傅里叶模式数 (水击波主要能量与低中频包络)

        self.scale = 1.0 / (in_channels * out_channels)
        # 复数权重矩阵：(in_channels, out_channels, modes1)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """复数张量乘法: (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)"""
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        orig_dtype = x.dtype
        # 防御 PyTorch 2.4+ AMP 下 ComplexHalf 的 einsum 不兼容问题
        x_fp32 = x.to(torch.float32)
        
        # 1. 快速傅里叶变换到频域 (rfft 返回正频率部分)
        x_ft = torch.fft.rfft(x_fp32)

        # 2. 截断低频模式并应用自适应复数权重乘法
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # 3. 逆快速傅里叶变换回到时域
        x_out = torch.fft.irfft(out_ft, n=x.size(-1))
        return x_out.to(orig_dtype)


class FNO1dBlock(nn.Module):
    """FNO 基础残差模块：谱卷积分支 + 局部线性分支 + GeLU 激活"""
    def __init__(self, width: int, modes: int):
        super(FNO1dBlock, self).__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm1d(width)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 双流残差：全局频域反射响应 + 局部瞬时波动
        x1 = self.conv(x)
        x2 = self.w(x)
        out = self.norm(x1 + x2)
        return self.act(out) + x  # 残差连接加速收敛


class FNO1dSurrogate(nn.Module):
    """
    一维傅里叶神经算子正演代理模型主网络
    参数：
        in_channels : 输入物理脉冲编码通道数 (默认 4)
        out_channels: 输出物理场数 (默认 1，对应 H_wh)
        width       : FNO 隐层通道数 (默认 64)
        modes       : 保留的傅里叶模式数 (默认 32)
        num_layers  : FNO 级联残差层数 (默认 4)
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        width: int = 64,
        modes: int = 32,
        num_layers: int = 4,
    ):
        super(FNO1dSurrogate, self).__init__()
        self.width = width
        self.modes = modes

        # 将物理特征脉冲提升至高维特征空间
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO 级联算子层
        self.fno_blocks = nn.ModuleList([
            FNO1dBlock(width, modes) for _ in range(num_layers)
        ])
        
        # 投影回实际物理场值 [H_wh(t)]
        self.fc1 = nn.Linear(width, 128)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入格式 x: (Batch, In_Channels, Time_Steps)
        输出格式 out: (Batch, Out_Channels, Time_Steps)
        """
        # 转为 (Batch, Time_Steps, In_Channels) 进行通道映射
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # 回到 (Batch, Width, Time_Steps)

        for block in self.fno_blocks:
            x = block(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


if __name__ == "__main__":
    if sys.platform.startswith('win') and hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # 本地冒烟测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在 {device} 上测试 FNO1dSurrogate 网络...")
    model = FNO1dSurrogate(in_channels=4, out_channels=1, width=64, modes=32, num_layers=4).to(device)
    
    # 模拟 Batch=8, 通道=4, 采样点=4096 (对齐 50s 降采样到 4096 点)
    dummy_input = torch.randn(8, 4, 4096, device=device)
    out = model(dummy_input)
    print(f"输入 Tensor 形状: {dummy_input.shape}")
    print(f"输出 Tensor 形状: {out.shape}")
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} 个")
    print("冒烟测试通过 [PASS]")
