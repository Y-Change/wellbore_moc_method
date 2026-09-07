# -*- coding: utf-8 -*-
"""
dataset.py — PaperA 03_leakoff验证 Brunone 数据集加载与预处理模块

功能：
1. 扫描并加载 03_leakoff验证 下全部 brunone 工况（D5, D10, D20, D50, D100 共40个正交案例）
2. 提取井口水头时程 H_wh(t) 并截取前 T 步（默认 16384 步，覆盖约 16.384 s 多周期往返波）
3. 提取工况物理先验参数向量 xi (a, L, V0, H0, ts, Cf, kleak, D) 并标准化
4. 构建空间网格深度 x in [0, L] 上的真实裂缝高斯平滑反射率剖面 y(x)
5. 支持 80/20 随机可复现切分，支持训练时高斯白噪声数据增强（对接论文 Section 4.4 鲁棒性体系）
"""
from __future__ import annotations

import os
import json
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def generate_gaussian_target(
    x_f: List[float],
    Nx: int = 1000,
    L: float = 5000.0,
    sigma: float = 7.5,
    include_toe: bool = False,
    toe_weight: float = 0.5
) -> np.ndarray:
    """
    在井深空间网格 x in [0, L] 上生成高斯平滑裂缝反射率/阻抗突变真值剖面 y(x)。
    
    参数:
        x_f: 真实裂缝深度列表 [m]
        Nx: 空间离散网格点数 (默认 1000 点，dx = 5 m)
        L: 井筒总长 [m]
        sigma: 高斯脉冲半宽度 [m] (默认 7.5 m, 脉冲覆盖约 3~5 个网格点)
        include_toe: 是否在井底 x=L 处添加端点反射高斯峰
        toe_weight: 井底反射高斯峰幅度
    返回:
        y: shape [Nx] 浮点数组，幅值范围 [0, 1]
    """
    x_grid = np.linspace(0.0, L, Nx, endpoint=True)
    y = np.zeros(Nx, dtype=np.float32)
    
    for x_frac in x_f:
        pulse = np.exp(-0.5 * ((x_grid - x_frac) / sigma) ** 2)
        y = np.maximum(y, pulse)  # 保持各裂缝峰值标准化为 1.0
        
    if include_toe:
        toe_pulse = toe_weight * np.exp(-0.5 * ((x_grid - L) / sigma) ** 2)
        y = np.maximum(y, toe_pulse)
        
    return y.astype(np.float32)


class WellboreBrunoneDataset(Dataset):
    """井筒多裂缝水击 Brunone 瞬变反演数据集"""
    def __init__(
        self,
        cases_data: List[Dict],
        T_steps: int = 16384,
        Nx: int = 1000,
        L: float = 5000.0,
        sigma: float = 7.5,
        include_toe: bool = False,
        augment: bool = False,
        noise_std: float = 0.02
    ):
        self.cases_data = cases_data
        self.T_steps = T_steps
        self.Nx = Nx
        self.L = L
        self.sigma = sigma
        self.include_toe = include_toe
        self.augment = augment
        self.noise_std = noise_std
        
    def __len__(self) -> int:
        return len(self.cases_data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.cases_data[idx]
        
        # 1. 原始时序截取与归一化
        H_raw = item['H_raw'][:self.T_steps].copy()
        if len(H_raw) < self.T_steps:
            # 填零保护 (padding)
            H_pad = np.zeros(self.T_steps, dtype=np.float32)
            H_pad[:len(H_raw)] = H_raw
            H_raw = H_pad
        else:
            H_raw = H_raw.astype(np.float32)
            
        # 零均值单位方差归一化
        mean_H = np.mean(H_raw)
        std_H = np.std(H_raw) + 1e-8
        H_norm = (H_raw - mean_H) / std_H
        
        # 数据增强：注入高斯白噪声 (论文 Section 4.4)
        if self.augment and self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=H_norm.shape).astype(np.float32)
            H_norm = H_norm + noise
            
        # 2. 物理先验参数向量 xi (标准化)
        # xi = [a/1500, L/5000, V0/1.0, H0/300, ts/1.0, log10(Cf)/(-5), log10(kleak)/(-4), D/50.0]
        xi = np.array([
            item['a'] / 1500.0,
            item['L'] / 5000.0,
            item['V0'] / 1.0,
            item['H0'] / 300.0,
            item['ts'] / 1.0,
            np.log10(max(item['Cf'], 1e-9)) / (-5.0),
            np.log10(max(item['kleak'], 1e-9)) / (-4.0),
            item['spacing'] / 50.0
        ], dtype=np.float32)
        
        # 3. 目标空间剖面 y(x)
        y = generate_gaussian_target(
            item['x_f'],
            Nx=self.Nx,
            L=self.L,
            sigma=self.sigma,
            include_toe=self.include_toe
        )
        
        return {
            'H': torch.tensor(H_norm, dtype=torch.float32).unsqueeze(0), # [1, T]
            'xi': torch.tensor(xi, dtype=torch.float32),                  # [D_param]
            'y': torch.tensor(y, dtype=torch.float32),                    # [Nx]
            'case_name': f"{item['dir']}/{item['case']}",
            'x_f': torch.tensor(item['x_f'], dtype=torch.float32),
            'n_fracs': torch.tensor(item['n_fracs'], dtype=torch.long)
        }


def collect_brunone_cases(base_dir: str) -> List[Dict]:
    """
    扫描并内存缓存 03_leakoff验证 下全部 brunone 案例
    """
    spacing_dirs = ['brunone_D5', 'brunone_D10', 'brunone_D20', 'brunone_D50', 'brunone_D100']
    cases = ['single', 'dual', 'triple', 'quad', 'quint', 'hex', 'hept', 'oct']
    
    collected = []
    print(f"[Dataset] 正在从 {base_dir} 扫描 Brunone 案例...")
    
    for d in spacing_dirs:
        spacing_val = float(d.replace('brunone_D', ''))
        dir_path = os.path.join(base_dir, d)
        if not os.path.exists(dir_path):
            continue
            
        for c in cases:
            csv_path = os.path.join(dir_path, c, 'moc_timeseries.csv')
            json_path = os.path.join(dir_path, c, 'moc_leakoff.json')
            
            if os.path.exists(csv_path) and os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                cfg = meta.get('config', {})
                x_f = cfg.get('x_f', [])
                
                # 读取时序
                df = pd.read_csv(csv_path)
                H_series = df['H_wh'].values.astype(np.float32)
                
                collected.append({
                    'dir': d,
                    'case': c,
                    'spacing': spacing_val,
                    'n_fracs': len(x_f),
                    'x_f': x_f,
                    'H_raw': H_series,
                    'a': float(cfg.get('a', 1450.0)),
                    'L': float(cfg.get('L', 5000.0)),
                    'V0': float(cfg.get('V0', 1.0)),
                    'H0': float(cfg.get('H0', 300.0)),
                    'ts': float(cfg.get('ts', 1.0)),
                    'Cf': float(cfg.get('Cf', 1e-5)),
                    'kleak': float(cfg.get('kleak', 1e-4))
                })
                
    print(f"[Dataset] 成功加载 {len(collected)} 个有效案例并缓存入内存。")
    return collected


def get_dataloaders(
    base_dir: str,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    seed: int = 42,
    T_steps: int = 16384,
    Nx: int = 1000,
    augment_train: bool = True,
    noise_std: float = 0.01
) -> Tuple[DataLoader, DataLoader, List[Dict], List[Dict]]:
    """
    构建训练集与验证集 DataLoader (80/20 随机可复现划分)
    """
    all_cases = collect_brunone_cases(base_dir)
    assert len(all_cases) > 0, f"未在 {base_dir} 找到有效案例！"
    
    np.random.seed(seed)
    indices = np.random.permutation(len(all_cases))
    split_idx = int(len(all_cases) * train_ratio)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_cases = [all_cases[i] for i in train_indices]
    val_cases = [all_cases[i] for i in val_indices]
    
    print(f"[Dataset] 数据集划分: 训练集 {len(train_cases)} 案例, 验证集 {len(val_cases)} 案例 (seed={seed})")
    
    train_dataset = WellboreBrunoneDataset(
        train_cases,
        T_steps=T_steps,
        Nx=Nx,
        augment=augment_train,
        noise_std=noise_std
    )
    
    val_dataset = WellboreBrunoneDataset(
        val_cases,
        T_steps=T_steps,
        Nx=Nx,
        augment=False
    )
    
    # collate_fn 针对动态长度的 x_f
    def custom_collate(batch):
        H = torch.stack([b['H'] for b in batch], dim=0)
        xi = torch.stack([b['xi'] for b in batch], dim=0)
        y = torch.stack([b['y'] for b in batch], dim=0)
        cases = [b['case_name'] for b in batch]
        x_fs = [b['x_f'] for b in batch]
        n_fracs = torch.stack([b['n_fracs'] for b in batch], dim=0)
        return {
            'H': H,
            'xi': xi,
            'y': y,
            'case_name': cases,
            'x_f': x_fs,
            'n_fracs': n_fracs
        }
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )
    
    return train_loader, val_loader, train_cases, val_cases


if __name__ == '__main__':
    # 单元测试与自检
    base_dir = r"e:\water_hammer_research\wellbore_moc_method\PaperA井口多裂缝水击响应\03_leakoff验证"
    train_loader, val_loader, tr, vl = get_dataloaders(base_dir, batch_size=4)
    for b in train_loader:
        print("Batch H shape:", b['H'].shape)
        print("Batch xi shape:", b['xi'].shape)
        print("Batch y shape:", b['y'].shape)
        print("Sample cases:", b['case_name'])
        break
    print("Dataset module self-test PASSED!")
