# -*- coding: utf-8 -*-
"""
train.py — 倒谱 Cep-DeepONet 训练与收敛验证脚本

参照 CMAME 2023 论文《Fourier-DeepONet》的训练设定：
1. 组合损失函数：相对 L2 误差 (Rel-L2) + 裂缝峰值加权 L1 损失 (Weighted-L1)
2. AdamW 优化器 + CosineAnnealingLR 学习率衰减
3. 训练/验证损失记录与最佳 Checkpoint 保存
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 确保本模块路径优先加载
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from dataset import get_dataloaders
from model import CepDeepONet


def compute_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    lambda_peak: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    组合损失函数：
    1. 相对 L2 误差 (Relative L2 norm loss) — 论文核心度量
       Rel_L2 = ||y_pred - y_true||_2 / (||y_true||_2 + eps)
    2. 裂缝峰值加权 L1 损失 — 强化局部裂缝尖锐度并抑制背景杂波
       Weighted_L1 = mean( |y_pred - y_true| * (1 + 4 * I(y_true > 0.1)) )
    """
    eps = 1e-6
    # 相对 L2 损失 (逐样本计算后取 batch 均值)
    diff_norm = torch.norm(y_pred - y_true, p=2, dim=-1)
    true_norm = torch.norm(y_true, p=2, dim=-1) + eps
    rel_l2 = torch.mean(diff_norm / true_norm)

    # 峰值加权 L1
    weights = 1.0 + 4.0 * (y_true > 0.1).float()
    weighted_l1 = torch.mean(torch.abs(y_pred - y_true) * weights)

    total_loss = rel_l2 + lambda_peak * weighted_l1
    return total_loss, rel_l2, weighted_l1


def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device
) -> Dict[str, float]:
    """在验证集上评估模型整体指标"""
    model.eval()
    total_loss = 0.0
    total_rel_l2 = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            H = batch['H'].to(device)
            xi = batch['xi'].to(device)
            y = batch['y'].to(device)

            y_pred = model(H, xi)
            loss, rel_l2, _ = compute_loss(y_pred, y)

            mae = torch.mean(torch.abs(y_pred - y)).item()
            rmse = torch.sqrt(torch.mean((y_pred - y) ** 2)).item()

            total_loss += loss.item()
            total_rel_l2 += rel_l2.item()
            total_mae += mae
            total_rmse += rmse
            n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'rel_l2': total_rel_l2 / max(n_batches, 1),
        'mae': total_mae / max(n_batches, 1),
        'rmse': total_rmse / max(n_batches, 1),
    }


def train_cep_deeponet(
    base_dir: str,
    output_dir: str,
    epochs: int = 120,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_peak: float = 0.5,
    seed: int = 42,
    device_str: str = "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)
    print(f"[Train] 设备: {device}, 随机种子: {seed}, 目标轮数: {epochs}")

    # 1. 加载数据
    train_loader, val_loader, train_cases, val_cases = get_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        seed=seed,
        augment_train=True,
        noise_std=0.015
    )

    # 2. 实例化模型
    model = CepDeepONet(
        T_steps=16384,
        Nx=1000,
        L=5000.0,
        param_dim=8,
        latent_channels=64,
        fourier_modes=32,
        num_fourier_layers=4
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_rel_l2 = float('inf')
    best_ckpt_path = os.path.join(output_dir, 'best_cep_deeponet.pt')
    history = {
        'train_loss': [],
        'train_rel_l2': [],
        'val_loss': [],
        'val_rel_l2': [],
        'val_mae': [],
        'val_rmse': [],
        'lr': []
    }

    start_time = time.time()
    print("=" * 75)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Rel-L2':^12} | {'Val Loss':^10} | {'Val Rel-L2':^11} | {'Val MAE':^9} | {'Val RMSE':^9}")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_epoch = 0.0
        train_rel_l2_epoch = 0.0
        n_train_batches = 0

        for batch in train_loader:
            H = batch['H'].to(device)
            xi = batch['xi'].to(device)
            y = batch['y'].to(device)

            optimizer.zero_grad()
            y_pred = model(H, xi)
            loss, rel_l2, _ = compute_loss(y_pred, y, lambda_peak=lambda_peak)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_epoch += loss.item()
            train_rel_l2_epoch += rel_l2.item()
            n_train_batches += 1

        scheduler.step()

        avg_train_loss = train_loss_epoch / max(n_train_batches, 1)
        avg_train_rel_l2 = train_rel_l2_epoch / max(n_train_batches, 1)

        # 验证评估
        val_metrics = evaluate(model, val_loader, device)

        # 记录日志
        current_lr = scheduler.get_last_lr()[0]
        history['train_loss'].append(avg_train_loss)
        history['train_rel_l2'].append(avg_train_rel_l2)
        history['val_loss'].append(val_metrics['loss'])
        history['val_rel_l2'].append(val_metrics['rel_l2'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['lr'].append(current_lr)

        # 保存最优模型
        if val_metrics['rel_l2'] < best_val_rel_l2:
            best_val_rel_l2 = val_metrics['rel_l2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'seed': seed,
                'train_cases': [c['dir'] + '/' + c['case'] for c in train_cases],
                'val_cases': [c['dir'] + '/' + c['case'] for c in val_cases]
            }, best_ckpt_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"{epoch:^7d} | {avg_train_loss:^11.4f} | {avg_train_rel_l2:^12.4f} | "
                f"{val_metrics['loss']:^10.4f} | {val_metrics['rel_l2']:^11.4f} | "
                f"{val_metrics['mae']:^9.4f} | {val_metrics['rmse']:^9.4f}"
            )

    total_time = time.time() - start_time
    print("=" * 75)
    print(f"[Train] 训练完成！耗时: {total_time:.2f} 秒 ({total_time/epochs:.3f} s/epoch)")
    print(f"[Train] 最佳验证集 Rel-L2 误差: {best_val_rel_l2:.4f}")
    print(f"[Train] Checkpoint 已落盘: {best_ckpt_path}")

    # 保存训练历史
    history_path = os.path.join(output_dir, 'train_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"[Train] 训练历史已保存至: {history_path}")

    return best_ckpt_path, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练 Cep-DeepONet 反演模型")
    parser.add_argument('--epochs', type=int, default=120, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=8, help="批大小")
    parser.add_argument('--lr', type=float, default=1e-3, help="学习率")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--device', type=str, default="cpu", help="计算设备")
    args = parser.parse_args()

    base_dir = r"e:\water_hammer_research\wellbore_moc_method\PaperA井口多裂缝水击响应\03_leakoff验证"
    output_dir = os.path.join(base_dir, "cep_deeponet", "results")

    train_cep_deeponet(
        base_dir=base_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_str=args.device
    )
