# -*- coding: utf-8 -*-
"""
visualize.py — 倒谱 Cep-DeepONet 反演结果可视化与图表绘制

生成两类学术论文级别的高清图表 (PNG + PDF)：
1. training_convergence.png: 训练与验证收敛曲线 (Loss, Rel-L2, MAE, RMSE)
2. inversion_profiles_comparison.png: 多案例反演对比图（Ground Truth 裂缝真值 vs Cep-DeepONet 反演剖面 vs 原始倒谱），参照 CMAME 2023 论文版式。
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from dataset import collect_brunone_cases, generate_gaussian_target
from model import CepDeepONet

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_convergence(history_json: str, save_dir: str):
    """绘制训练过程收敛四联图"""
    with open(history_json, 'r', encoding='utf-8') as f:
        h = json.load(f)

    epochs = range(1, len(h['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # 1. Total Loss
    axes[0].plot(epochs, h['train_loss'], label='Train Loss', color='#1f77b4', lw=1.8)
    axes[0].plot(epochs, h['val_loss'], label='Val Loss', color='#d62728', lw=1.8, ls='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('(a) Total Loss Convergence', fontsize=13, fontweight='bold')
    axes[0].grid(True, ls=':', alpha=0.6)
    axes[0].legend(frameon=True, fontsize=11)

    # 2. Relative L2 Error
    axes[1].plot(epochs, h['train_rel_l2'], label='Train Rel-L2', color='#1f77b4', lw=1.8)
    axes[1].plot(epochs, h['val_rel_l2'], label='Val Rel-L2', color='#2ca02c', lw=1.8, ls='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Relative L2 Error', fontsize=12)
    axes[1].set_title('(b) Relative $L^2$ Norm Error', fontsize=13, fontweight='bold')
    axes[1].grid(True, ls=':', alpha=0.6)
    axes[1].legend(frameon=True, fontsize=11)

    # 3. MAE & RMSE
    axes[2].plot(epochs, h['val_mae'], label='Val MAE', color='#ff7f0e', lw=1.8)
    axes[2].plot(epochs, h['val_rmse'], label='Val RMSE', color='#9467bd', lw=1.8, ls='-.')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Error Metric', fontsize=12)
    axes[2].set_title('(c) Validation MAE & RMSE', fontsize=13, fontweight='bold')
    axes[2].grid(True, ls=':', alpha=0.6)
    axes[2].legend(frameon=True, fontsize=11)

    for ax in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    out_png = os.path.join(save_dir, 'training_convergence.png')
    out_pdf = os.path.join(save_dir, 'training_convergence.pdf')
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Visualize] 训练收敛图已保存至: {out_png}")


def plot_inversion_comparisons(
    ckpt_path: str,
    base_dir: str,
    save_dir: str,
    sample_cases: List[Tuple[str, str]] = None
):
    """
    绘制典型案例反演对比图（真值 vs Cep-DeepONet 预测 vs 原始实倒谱）
    """
    device = torch.device('cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = CepDeepONet(
        T_steps=16384,
        Nx=1000,
        L=5000.0,
        param_dim=8,
        latent_channels=64,
        fourier_modes=32,
        num_fourier_layers=4
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_cases = collect_brunone_cases(base_dir)
    case_map = {f"{c['dir']}/{c['case']}": c for c in all_cases}

    if sample_cases is None:
        # 挑选 4 个典型工况覆盖不同缝数与间距 (单缝、双缝、四缝、八缝)
        sample_cases = [
            ('brunone_D50', 'single'),
            ('brunone_D20', 'dual'),
            ('brunone_D50', 'quad'),
            ('brunone_D10', 'oct')
        ]

    fig, axes = plt.subplots(len(sample_cases), 1, figsize=(15, 3.2 * len(sample_cases)), dpi=300, sharex=True)
    if len(sample_cases) == 1:
        axes = [axes]

    x_grid = np.linspace(0.0, 5000.0, 1000, endpoint=True)
    dt = 0.001
    a = 1450.0

    for i, (d, c) in enumerate(sample_cases):
        ax = axes[i]
        key = f"{d}/{c}"
        if key not in case_map:
            continue
        item = case_map[key]
        x_f = item['x_f']
        n_frac = len(x_f)
        spacing = item['spacing']

        # 准备输入张量
        H_raw = item['H_raw'][:16384].copy()
        mean_H = np.mean(H_raw)
        std_H = np.std(H_raw) + 1e-8
        H_norm = (H_raw - mean_H) / std_H
        H_t = torch.tensor(H_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

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
        xi_t = torch.tensor(xi, dtype=torch.float32).unsqueeze(0)

        # 真值剖面
        y_true = generate_gaussian_target(x_f, Nx=1000, L=5000.0, sigma=7.5)

        # 模型推理
        with torch.no_grad():
            y_pred = model(H_t, xi_t)[0].cpu().numpy()
            # 提取可微倒谱用于基线对比
            raw_cep = model.branch_net.cepstrum(H_t)[0, 0].cpu().numpy()

        # 原始倒谱按声速映射到空间深度: x = q * a / 2
        quefrency = np.arange(len(raw_cep)) * dt
        cep_depth = quefrency * a / 2.0
        # 截取到 5000 m
        mask = cep_depth <= 5000.0
        cep_depth_sub = cep_depth[mask]
        cep_sub = np.abs(raw_cep[mask])
        # 归一化便于在同一尺度观察峰形
        if np.max(cep_sub) > 0:
            cep_sub = cep_sub / np.max(cep_sub)

        # 绘制曲线
        # 1. 原始倒谱 (灰蓝虚线背景)
        ax.plot(cep_depth_sub, cep_sub, label='Raw Quefrency Cepstrum', color='#7f7f7f', lw=1.2, ls=':', alpha=0.7)

        # 2. 真实目标分布 (灰色填充脉冲)
        ax.fill_between(x_grid, 0, y_true, color='#2ca02c', alpha=0.35, label='Ground Truth Fracs')
        ax.plot(x_grid, y_true, color='#2ca02c', lw=1.5)

        # 3. Cep-DeepONet 反演预测 (深红实线)
        ax.plot(x_grid, y_pred, label='Cep-DeepONet Inversion', color='#d62728', lw=2.0)

        # 标记真实裂缝垂直线
        for idx, x_val in enumerate(x_f):
            ax.axvline(x=x_val, color='#1f77b4', ls='--', lw=1.0, alpha=0.7)

        ax.set_xlim(3800, 4800)  # 聚焦裂缝重点反射区
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel(f"Profile (a.u.)\n[{c}, D={spacing}m]", fontsize=11)
        ax.grid(True, ls=':', alpha=0.5)

        # 计算该案例局部指标
        rel_l2 = np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + 1e-8)
        mae = np.mean(np.abs(y_pred - y_true))
        ax.text(
            0.02, 0.88,
            f"Case: {d}/{c} ($n={n_frac}, S={spacing}$m) | Rel-$L^2$={rel_l2:.3f} | MAE={mae:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc')
        )

        if i == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=10)

    axes[-1].set_xlabel("Wellbore Depth $x$ (m)", fontsize=12)
    plt.suptitle("Cep-DeepONet Full Waveform Inversion vs Ground Truth & Raw Cepstrum", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_png = os.path.join(save_dir, 'inversion_profiles_comparison.png')
    out_pdf = os.path.join(save_dir, 'inversion_profiles_comparison.pdf')
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Visualize] 反演对比多联图已保存至: {out_png}")


if __name__ == '__main__':
    base_dir = r"e:\water_hammer_research\wellbore_moc_method\PaperA井口多裂缝水击响应\03_leakoff验证"
    results_dir = os.path.join(base_dir, "cep_deeponet", "results")
    history_file = os.path.join(results_dir, "train_history.json")
    ckpt_file = os.path.join(results_dir, "best_cep_deeponet.pt")

    if os.path.exists(history_file):
        plot_training_convergence(history_file, results_dir)
    if os.path.exists(ckpt_file):
        plot_inversion_comparisons(ckpt_file, base_dir, results_dir)
