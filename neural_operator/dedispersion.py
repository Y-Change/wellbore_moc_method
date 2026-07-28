import torch
import torch.nn as nn

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_channels) # Group norm 更适合小 Batch Size
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DeDispersionFrontEnd(nn.Module):
    """
    相幅频散逆滤波网络 (De-dispersion Front-end)
    本质：盲反卷积 (Blind Deconvolution) 算子
    目标：补偿 MOC 非定常摩阻带来的相位畸变与高频衰减，在进入可微倒谱前，将水击回波重新锐化为脉冲。
    结构：采用带空洞卷积 (Dilated Convolution) 的 1D 残差网络，以捕捉长距离时序回波关联。
    """
    def __init__(self, in_channels=1, hidden_channels=32, num_layers=4):
        super().__init__()
        # 1. 扩大感受野的词元编码
        self.stem = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3)
        
        # 2. 空洞残差块 (感受野指数增加，抓取长程反射波)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = dilation
            layers.append(ConvBlock1D(hidden_channels, hidden_channels, kernel_size=3, padding=padding, dilation=dilation))
            
        self.res_blocks = nn.Sequential(*layers)
        
        # 3. 输出映射
        self.out_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        x: [B, C, T] 带频散与噪声的水击观测信号
        返回: 锐化后的类脉冲信号
        """
        identity = x
        h = self.stem(x)
        h = self.res_blocks(h)
        out = self.out_conv(h)
        
        # 残差连接：网络只需要学习“如何把被抹平的波形重新捏尖 (高频补偿)”
        return identity + out
