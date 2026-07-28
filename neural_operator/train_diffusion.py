import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from neural_operator.dedispersion import DeDispersionFrontEnd
from neural_operator.cepstrum import DifferentiableCepstrum
from neural_operator.diffusion_1d import ConditionalUNet1D, GaussianDiffusion1D
from neural_operator.dataset_surrogate import FracturingMOCSurrogateDataset

def train_dccdm(data_dir="output/lhs_dataset/data", epochs=100, batch_size=16, seq_length=4096):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 启动可微倒谱条件扩散反演网络 (DCCDM) 训练 | 设备: {device}")
    
    # 1. 准备数据
    try:
        dataset = FracturingMOCSurrogateDataset(data_dir=data_dir, n_time_target=seq_length, split="train")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except FileNotFoundError:
        print("未找到真实数据集，构造假数据用于测试网络连通性。")
        class FakeDataset:
            def __len__(self): return 16
            def __getitem__(self, idx):
                # 返回 (impulse_trains [4, T], pressure_wave [1, T])
                return torch.randn(4, seq_length), torch.randn(1, seq_length)
        dataloader = DataLoader(FakeDataset(), batch_size=4)

    # 2. 实例化四大核心组件
    dedisp = DeDispersionFrontEnd().to(device)
    ceps = DifferentiableCepstrum(eps_threshold=1e-6).to(device)
    unet = ConditionalUNet1D(in_channels=1, out_channels=1, context_dim=1, base_dim=64).to(device)
    diffusion = GaussianDiffusion1D(unet, seq_length=seq_length, timesteps=100).to(device) # 测试时步设小点
    
    # 3. 联合优化器 (同时优化逆滤波前端和 U-Net)
    optimizer = optim.AdamW([
        {'params': dedisp.parameters(), 'lr': 1e-4},
        {'params': unet.parameters(), 'lr': 2e-4}
    ], weight_decay=1e-5)
    
    # 4. 训练循环
    for epoch in range(epochs):
        dedisp.train()
        unet.train()
        
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_input, batch_H in pbar:
            batch_input = batch_input.to(device)
            batch_H = batch_H.to(device)
            
            # --- 构造扩散模型的目标图 ---
            # 反演任务：目标 (x_start) 是预测真实裂缝的高斯脉冲序列
            # 我们取 Cf 柔度高斯脉冲图 (在 channel 2)
            x_start = batch_input[:, 2:3, :]  # [B, 1, T]
            
            # 观测特征 (y_obs) 是压力波 (batch_H)
            y_obs = batch_H # [B, 1, T]
            
            # --- 物理前置处理 (提取条件) ---
            # 1. 逆滤波锐化
            y_sharp = dedisp(y_obs)
            # 2. 可微倒谱 (提取高频反演先验)
            cond_ceps = ceps(y_sharp)
            
            # --- 扩散去噪损失 ---
            # 随机采样时间步 t
            b = y_obs.shape[0]
            t = torch.randint(0, diffusion.timesteps, (b,), device=device).long()
            
            # 计算加噪与去噪损失 MSE(eps, eps_theta)
            loss = diffusion.p_losses(x_start, t, context=cond_ceps)
            
            # --- 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防爆
            torch.nn.utils.clip_grad_norm_(dedisp.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        print(f"✅ Epoch {epoch+1} | Average Loss: {epoch_loss/len(dataloader):.4f}")
        
if __name__ == "__main__":
    print(">>> 开始进行 DCCDM 1-Epoch 试跑打靶...")
    train_dccdm(data_dir="output/lhs_dataset/data", epochs=1, batch_size=4, seq_length=4096)
