import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步 (Timestep) 的正弦波位置编码。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CrossAttention1D(nn.Module):
    """一维交叉注意力：U-Net 特征作 Query，物理条件作 Key/Value。"""

    def __init__(self, query_dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.to_q = nn.Conv1d(query_dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv1d(context_dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv1d(context_dim, inner_dim, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, query_dim, 1),
            nn.GroupNorm(8, query_dim),
        )

    def forward(self, x, context):
        """
        x: [B, query_dim, T]
        context: [B, context_dim, T_cond]
        """
        batch, _, n_time = x.shape
        q = self.to_q(x).view(batch, self.heads, -1, n_time).transpose(-1, -2)
        context_time = context.shape[-1]
        k = self.to_k(context).view(batch, self.heads, -1, context_time).transpose(-1, -2)
        v = self.to_v(context).view(batch, self.heads, -1, context_time).transpose(-1, -2)

        # 避免显式构造 [B, H, T, T_cond] 相似度矩阵。
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(batch, -1, n_time)
        return self.to_out(out) + x


class ConditionalUNet1D(nn.Module):
    """带倒谱先验交叉注意力的 1D 扩散 U-Net。"""

    def __init__(self, in_channels=1, out_channels=1, context_dim=1, base_dim=64):
        super().__init__()
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_conv = nn.Conv1d(in_channels, base_dim, kernel_size=7, padding=3)

        self.t_proj1 = nn.Linear(time_dim, base_dim)
        self.t_proj2 = nn.Linear(time_dim, base_dim * 2)
        self.t_proj_m = nn.Linear(time_dim, base_dim * 2)
        self.t_proj_u2 = nn.Linear(time_dim, base_dim)
        self.t_proj_u1 = nn.Linear(time_dim, base_dim)

        self.down1 = ConvBlock1D(base_dim, base_dim)
        self.attn1 = CrossAttention1D(base_dim, context_dim)
        self.pool1 = nn.MaxPool1d(2)

        self.down2 = ConvBlock1D(base_dim, base_dim * 2)
        self.attn2 = CrossAttention1D(base_dim * 2, context_dim)
        self.pool2 = nn.MaxPool1d(2)

        self.mid1 = ConvBlock1D(base_dim * 2, base_dim * 2)
        self.mid_attn = CrossAttention1D(base_dim * 2, context_dim)
        self.mid2 = ConvBlock1D(base_dim * 2, base_dim * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv2 = ConvBlock1D(base_dim * 4, base_dim)
        self.up_attn2 = CrossAttention1D(base_dim, context_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv1 = ConvBlock1D(base_dim * 2, base_dim)
        self.up_attn1 = CrossAttention1D(base_dim, context_dim)
        self.out_conv = nn.Conv1d(base_dim, out_channels, kernel_size=1)

    def forward(self, x, time, context):
        """
        x: [B, C, T] 带噪序列
        time: [B] 时间步
        context: [B, context_dim, T_cond] 倒谱物理先验
        """
        if x.shape[-1] % 4:
            raise ValueError("sequence length must be divisible by 4")
        t_emb = self.time_mlp(time)
        x0 = self.init_conv(x)

        d1 = self.down1(x0) + self.t_proj1(t_emb).unsqueeze(-1)
        d1 = self.attn1(d1, context)
        p1 = self.pool1(d1)

        d2 = self.down2(p1) + self.t_proj2(t_emb).unsqueeze(-1)
        d2 = self.attn2(d2, context)
        p2 = self.pool2(d2)

        middle = self.mid1(p2) + self.t_proj_m(t_emb).unsqueeze(-1)
        middle = self.mid_attn(middle, context)
        middle = self.mid2(middle)

        u2 = self.up2(middle)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_conv2(u2) + self.t_proj_u2(t_emb).unsqueeze(-1)
        u2 = self.up_attn2(u2, context)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_conv1(u1) + self.t_proj_u1(t_emb).unsqueeze(-1)
        u1 = self.up_attn1(u1, context)
        return self.out_conv(u1)


class GaussianDiffusion1D(nn.Module):
    """DDPM 前向加噪、噪声预测损失与反向采样管理器。"""

    def __init__(self, model, seq_length, timesteps=1000):
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        self.timesteps = timesteps

        betas = torch.linspace(1.0e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """前向过程：采样 q(x_t | x_0)。"""
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sigma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return alpha * x_start + sigma * noise

    def p_losses(self, x_start, t, context, noise=None):
        """计算去噪损失 MSE(epsilon, epsilon_theta)。"""
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, context)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, context, batch_size=None, generator=None, initial_noise=None):
        """从纯噪声生成 raw 归一化裂缝图，不在采样器内裁剪。"""
        device = self.betas.device
        context_batch = context.shape[0]
        if batch_size is None:
            batch_size = context_batch
        if batch_size != context_batch:
            raise ValueError(
                f"batch_size={batch_size} 与 context batch={context_batch} 不一致"
            )
        if not torch.isfinite(context).all():
            raise FloatingPointError("context contains NaN or Inf")

        shape = (batch_size, 1, self.seq_length)
        if initial_noise is None:
            x = torch.randn(shape, device=device, generator=generator)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(f"initial_noise shape {tuple(initial_noise.shape)} != {shape}")
            x = initial_noise.to(device=device, dtype=self.betas.dtype)

        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t, context)

            alpha_t = (1.0 - self.betas[t]).view(-1, 1, 1)
            alpha_cum_t = self.alphas_cumprod[t].view(-1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - beta_t / torch.sqrt(1.0 - alpha_cum_t) * predicted_noise
            )

            if step == 0:
                x = mean
            else:
                previous = self.alphas_cumprod_prev[t].view(-1, 1, 1)
                variance = beta_t * (1.0 - previous) / (1.0 - alpha_cum_t)
                noise = torch.randn(
                    x.shape,
                    device=x.device,
                    dtype=x.dtype,
                    generator=generator,
                )
                x = mean + torch.sqrt(variance) * noise
        return x
