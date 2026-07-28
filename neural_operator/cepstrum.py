import torch
import torch.nn as nn

class DifferentiableCepstrum(nn.Module):
    """
    可微倒谱 (Differentiable Cepstrum)
    集成了 Clamp-Log 梯度防爆策略，彻底切断水击波高频死寂区的梯度爆炸。
    纯物理算子，无任何可学习参数。
    """
    def __init__(self, eps_threshold=1e-6):
        super().__init__()
        self.eps_threshold = eps_threshold
        
    def forward(self, x):
        """
        x: [B, C, T] 时域信号 (由 De-dispersion 网络锐化后的信号)
        Returns:
            cepstrum: [B, C, T] 倒频率域特征序列 (Quefrency domain)
        """
        # 记录原始的时间序列长度
        T = x.size(-1)
        
        # 1. 傅里叶变换到频域
        # 使用 rfft 因为输入是实数信号，能够节省一半的算力和显存
        X_f = torch.fft.rfft(x, dim=-1)
        
        # 2. 功率谱 (Power Spectrum)
        P_f = torch.abs(X_f) ** 2
        
        # 3. 对数谱与 Clamp-Log 防爆机制 (Log Spectrum)
        # 核心：拦截功率低于阈值的频率，阻断梯度回传，防止 NaN 爆炸
        P_safe = torch.clamp(P_f, min=self.eps_threshold)
        L_f = torch.log(P_safe)
        
        # 4. 离散傅里叶逆变换回倒频域 (Real Cepstrum)
        # 使用 irfft 将实对称的对数谱转换回倒频率域
        # 注意：irfft 需要指定原始序列长度 n=T，否则如果 T 是奇数，irfft 会默认输出偶数长度导致对齐失败
        c_q = torch.fft.irfft(L_f, n=T, dim=-1)
        
        return c_q
