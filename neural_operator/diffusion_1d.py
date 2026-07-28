import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础模块 (位置编码与卷积块)
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    """时间步 (Timestep) 的正弦波位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# ==========================================
# 2. 交叉注意力机制 (Cross-Attention)
# ==========================================
class CrossAttention1D(nn.Module):
    """
    一维交叉注意力层
    Query: 来源于 U-Net 当前层的空间特征 (去噪特征)
    Key/Value: 来源于物理前置算子提取的特征 (如可微倒谱)
    """
    def __init__(self, query_dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv1d(query_dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv1d(context_dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv1d(context_dim, inner_dim, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, query_dim, 1),
            nn.GroupNorm(8, query_dim)
        )

    def forward(self, x, context):
        """
        x: [B, query_dim, T]
        context: [B, context_dim, T_cond]
        """
        b, c, t = x.shape
        # Q: [B, Heads, T, Dim_Head]
        q = self.to_q(x).view(b, self.heads, -1, t).transpose(-1, -2)
        # K, V: [B, Heads, T_cond, Dim_Head]
        k = self.to_k(context).view(b, self.heads, -1, context.shape[-1]).transpose(-1, -2)
        v = self.to_v(context).view(b, self.heads, -1, context.shape[-1]).transpose(-1, -2)

        # 相似度矩阵: [B, Heads, T, T_cond]
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        # 聚合值: [B, Heads, T, Dim_Head]
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        # 还原形状: [B, inner_dim, T]
        out = out.transpose(-1, -2).reshape(b, -1, t)
        
        return self.to_out(out) + x # 残差连接

# ==========================================
# 3. 核心底座: 1D 条件去噪 U-Net
# ==========================================
class ConditionalUNet1D(nn.Module):
    """
    带倒谱先验交叉注意力的 1D 扩散 U-Net
    目标：接收加噪信号 x_t、时间步 t、倒谱特征 context，预测添加的噪声 eps。
    """
    def __init__(self, in_channels=1, out_channels=1, context_dim=1, base_dim=64):
        super().__init__()
        
        # 时间嵌入
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 初始投影
        self.init_conv = nn.Conv1d(in_channels, base_dim, kernel_size=7, padding=3)
        
        # Down 1
        self.down1 = ConvBlock1D(base_dim, base_dim)
        self.attn1 = CrossAttention1D(base_dim, context_dim)
        self.pool1 = nn.MaxPool1d(2)
        
        # Down 2
        self.down2 = ConvBlock1D(base_dim, base_dim * 2)
        self.attn2 = CrossAttention1D(base_dim * 2, context_dim)
        self.pool2 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.mid1 = ConvBlock1D(base_dim * 2, base_dim * 2)
        self.mid_attn = CrossAttention1D(base_dim * 2, context_dim)
        self.mid2 = ConvBlock1D(base_dim * 2, base_dim * 2)
        
        # Up 2
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = ConvBlock1D(base_dim * 4, base_dim) # Concat 下采样的特征
        self.up_attn2 = CrossAttention1D(base_dim, context_dim)
        
        # Up 1
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = ConvBlock1D(base_dim * 2, base_dim)
        self.up_attn1 = CrossAttention1D(base_dim, context_dim)
        
        # 预测头
        self.out_conv = nn.Conv1d(base_dim, out_channels, kernel_size=1)
        
    def forward(self, x, time, context):
        """
        x: [B, C, T] 带噪序列
        time: [B] 时间步
        context: [B, context_dim, T] 倒谱物理先验
        """
        # 1. 提取时间特征，简单地加到空间特征上 (或者可以通过 AdaGN 注入，这里用简单的 broadcast 加法)
        t_emb = self.time_mlp(time) # [B, time_dim]
        # 这里为了简化，我们仅将 t_emb 的部分特征与通道混合
        # 在严谨实现中，通常使用 FiLM 层。这里提供极简结构。
        
        # 2. 初始层
        x0 = self.init_conv(x)
        
        # 3. Down阶段
        d1 = self.down1(x0)
        d1 = self.attn1(d1, context)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)
        d2 = self.attn2(d2, context)
        p2 = self.pool2(d2)
        
        # 4. Bottleneck
        m = self.mid1(p2)
        m = self.mid_attn(m, context)
        m = self.mid2(m)
        
        # 5. Up阶段
        u2 = self.up2(m)
        u2 = torch.cat([u2, d2], dim=1) # 经典 Skip-Connection
        u2 = self.up_conv2(u2)
        u2 = self.up_attn2(u2, context)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_conv1(u1)
        u1 = self.up_attn1(u1, context)
        
        # 6. 输出预测的噪声
        out = self.out_conv(u1)
        return out

# ==========================================
# 4. 扩散过程管理 (Forward SDE & DDPM)
# ==========================================
class GaussianDiffusion1D(nn.Module):
    """
    高斯扩散管理器 (DDPM 框架)
    负责计算前向加噪、计算损失，以及反向采样。
    """
    def __init__(self, model, seq_length, timesteps=1000):
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        self.timesteps = timesteps
        
        # 定义线性的 Beta Schedule
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为 buffer 防止被优化器更新
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
    def q_sample(self, x_start, t, noise=None):
        """前向过程: 采样 q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def p_losses(self, x_start, t, context, noise=None):
        """计算去噪损失 MSE(eps, eps_theta)"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 1. 制造带噪数据
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 2. 网络在物理先验的 condition 下预测噪声
        predicted_noise = self.model(x_noisy, t, context)
        
        # 3. 计算均方误差
        loss = F.mse_loss(noise, predicted_noise)
        return loss
        
    @torch.no_grad()
    def sample(self, context, batch_size=1):
        """反向过程: 从纯噪声生成裂缝脉冲图 p(x_{t-1} | x_t)"""
        device = self.betas.device
        
        # 从纯高斯噪声开始
        shape = (batch_size, 1, self.seq_length)
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.model(x, t, context)
            
            # 根据 DDPM 公式计算均值
            alpha_t = (1 - self.betas[t]).view(-1, 1, 1)
            alpha_cum_t = self.alphas_cumprod[t].view(-1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1)
            
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cum_t)) * predicted_noise)
            
            # 最后一步不加随机噪声
            if i == 0:
                x = mean
            else:
                posterior_variance = beta_t * (1. - self.alphas_cumprod_prev[t].view(-1,1,1)) / (1. - alpha_cum_t)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_variance) * noise
                
        return x
