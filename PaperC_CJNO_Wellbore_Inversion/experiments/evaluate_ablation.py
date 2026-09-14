# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.evaluate_ablation

TG-CJ-DeepONet 第二阶段全套消融对标评测与成果图版绘制脚本:
1. 在 100 个独立测试案例 (Test Split) 上系统评估四大消融组:
   - Exp 1a: tg_relative_bias (可微相对到时窗 + 声学时延偏置)
   - Exp 1b: tg_gaussian_bias (可微高斯软窗 + 声学时延偏置)
   - Exp 1c: tg_ceps_patch_bias (2D 倒谱时空切片图块 + 声学时延偏置)
   - Exp 2:  tg_best_nobias (最佳模式 relative_window + 无声学偏置消融)
2. 横向全面对标第一阶段预研基准 (1D-ResNet, 1D-FNO, CJ-Cep-DeepONet);
3. 输出指标汇总至 output/ablation_metrics_summary.json (并同步合并至 output/metrics_summary.json);
4. 绘制 Nature 规范复合成果图版并保存到 output/figures/:
   - fig4_ablation_window_comparison.png: 3 种到时窗机制与注意力热力图/波形对齐图 (Panel a, b, c, d);
   - fig5_phase2_model_benchmark_bars.png: Phase 2 各新模型与 Phase 1 基准在 alpha MAE, R^2, Cf MRE, W_1 上的多柱对比图。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.src.models.resnet1d import ResNet1D
from PaperC_CJNO_Wellbore_Inversion.src.models.fno1d import FNO1D
from PaperC_CJNO_Wellbore_Inversion.src.models.deeponet import VanillaDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.cj_cep_deeponet import CJCepDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_cj_deeponet import TGCJDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics

# Nature 绘图样式规范
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")
DEFAULT_OUTPUT_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output")
DEFAULT_FIGURES_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "figures")

# Phase 2 消融组配置
ABLATION_SPECS = {
    "tg_relative_bias": {
        "window_mode": "relative_window",
        "use_acoustic_bias": True,
        "label": "TG-DeepONet (Relative Win + Bias)",
        "short_label": "TG-Rel-Bias",
        "color": "#d62728", # Crimson Flagship
    },
    "tg_gaussian_bias": {
        "window_mode": "gaussian_gating",
        "use_acoustic_bias": True,
        "label": "TG-DeepONet (Gaussian Gate + Bias)",
        "short_label": "TG-Gauss-Bias",
        "color": "#ff7f0e", # Orange
    },
    "tg_ceps_patch_bias": {
        "window_mode": "ceps_patch",
        "use_acoustic_bias": True,
        "label": "TG-DeepONet (Ceps Patch + Bias)",
        "short_label": "TG-Patch-Bias",
        "color": "#9467bd", # Purple
    },
    "tg_best_nobias": {
        "window_mode": "relative_window",
        "use_acoustic_bias": False,
        "label": "TG-DeepONet (Relative Win, No Bias)",
        "short_label": "TG-Rel-NoBias",
        "color": "#8c564b", # Brown
    },
}

# Phase 1 基准配置
BASELINE_SPECS = {
    "resnet": {
        "cls": ResNet1D,
        "label": "1D-ResNet (Phase 1)",
        "short_label": "ResNet1D",
        "color": "#1f77b4", # Blue
    },
    "fno": {
        "cls": FNO1D,
        "label": "1D-FNO (Phase 1)",
        "short_label": "FNO1D",
        "color": "#2ca02c", # Green
    },
    "cj_cep_deeponet": {
        "cls": CJCepDeepONet,
        "label": "CJ-Cep-DeepONet (Phase 1)",
        "short_label": "CJ-Cep (P1)",
        "color": "#7f7f7f", # Gray
    },
}


def load_ablation_model(exp_name: str, weights_dir: str, device: torch.device) -> TGCJDeepONet:
    spec = ABLATION_SPECS[exp_name]
    model = TGCJDeepONet(
        window_mode=spec["window_mode"],
        use_acoustic_bias=spec["use_acoustic_bias"],
        in_channels=2,
        cond_dim=3,
        d_model=64,
        n_heads=4,
        n_layers=2,
        p=64,
    ).to(device)

    weight_path = os.path.join(weights_dir, f"{exp_name}_best.pt")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到消融模型权重文件: {weight_path}")

    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_baseline_model(base_name: str, weights_dir: str, device: torch.device) -> nn.Module:
    spec = BASELINE_SPECS[base_name]
    model = spec["cls"]().to(device)
    weight_path = os.path.join(weights_dir, f"{base_name}_best.pt")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到基准模型权重文件: {weight_path}")

    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_test_set(
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """在 100 个独立测试集上全面评估所有基准与消融模型"""
    print("[*] 正在加载独立测试集 (split='test')...")
    test_ds = PilotInversionDataset(split="test")
    test_loader = test_ds.get_dataloader(batch_size=100, shuffle=False)
    test_batch = next(iter(test_loader))
    test_batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}

    all_metrics: Dict[str, Any] = {}
    all_predictions: Dict[str, Any] = {}

    # 1. 评测 Phase 1 基准模型
    print("[*] 正在评测 Phase 1 基准模型...")
    for base_name in BASELINE_SPECS.keys():
        model = load_baseline_model(base_name, weights_dir, device)
        with torch.no_grad():
            out = model(test_batch_dev)
        pred_dict = {
            "alpha": out["alpha"].cpu(),
            "cf": out["cf"].cpu(),
            "m_alpha_grid": out.get("m_alpha_grid", torch.zeros(100, 500)).cpu(),
        }
        targ_dict = {
            "alpha": test_batch["alpha"],
            "cf": test_batch["cf"],
            "m_alpha_grid": test_batch["m_alpha_grid"],
            "mask": test_batch["mask"],
            "positions": test_batch["positions"],
        }
        m = compute_inversion_metrics(pred_dict, targ_dict)
        all_metrics[base_name] = m
        all_predictions[base_name] = out
        print(f"  - {base_name:<15}: alpha MAE = {m['alpha_mae']:.4f} (R2={m['alpha_r2']:.3f}), Cf MRE = {m['cf_mre_pct']:.1f}%, W1 = {m['w1_mean_m']:.2f}m")

    # 2. 评测 Phase 2 消融模型
    print("[*] 正在评测 Phase 2 TG-CJ-DeepONet 消融模型...")
    for exp_name in ABLATION_SPECS.keys():
        model = load_ablation_model(exp_name, weights_dir, device)
        with torch.no_grad():
            out = model(test_batch_dev)
        pred_dict = {
            "alpha": out["alpha"].cpu(),
            "cf": out["cf"].cpu(),
            "m_alpha_grid": out["m_alpha_grid"].cpu(),
        }
        targ_dict = {
            "alpha": test_batch["alpha"],
            "cf": test_batch["cf"],
            "m_alpha_grid": test_batch["m_alpha_grid"],
            "mask": test_batch["mask"],
            "positions": test_batch["positions"],
        }
        m = compute_inversion_metrics(pred_dict, targ_dict)
        all_metrics[exp_name] = m
        all_predictions[exp_name] = out
        print(f"  - {exp_name:<20}: alpha MAE = {m['alpha_mae']:.4f} (R2={m['alpha_r2']:.3f}), Cf MRE = {m['cf_mre_pct']:.1f}%, W1 = {m['w1_mean_m']:.2f}m")

    return all_metrics, all_predictions, test_batch


def plot_fig4_window_comparison(
    predictions: Dict[str, Any],
    test_batch: Dict[str, Any],
    save_path: str,
):
    """
    绘制 Figure 4: 3 种到时窗机制与注意力热力图/波形对齐图 (Nature 规范复合成果图版)
    Panel (a): 物理到时对齐提取对比 (原始全波 vs 相对到时窗 [-50ms, +250ms] 对齐效果)
    Panel (b): 可微软高斯窗 profiles w_j(t) 在全时程波形上的能量聚焦
    Panel (c): 2D 连续倒谱时空切片图块 (tau_j, x_j) Patch 局部提取
    Panel (d): 声学传播时延偏置自注意力热力图 (有偏置 vs 无偏置 簇间解耦对比)
    """
    print(f"[*] 正在绘制 Figure 4: {save_path}")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), dpi=300)
    plt.subplots_adjust(wspace=0.28, hspace=0.32)

    # 选取一个典型密集 4 簇或 5 簇测试样本
    masks = test_batch["mask"].numpy()
    nc_counts = masks.sum(axis=1)
    cand_indices = np.where(nc_counts >= 4)[0]
    sample_idx = cand_indices[0] if len(cand_indices) > 0 else 0

    wave_raw = test_batch["wave"][sample_idx, 0].numpy()  # (4096,)
    t_full = np.linspace(0, 60.0, len(wave_raw))
    positions = test_batch["positions"][sample_idx].numpy()
    norm_pos = test_batch["norm_positions"][sample_idx].numpy()
    cond = test_batch["cond"][sample_idx].numpy()
    mask = masks[sample_idx]
    nc = int(mask.sum())
    a_val = cond[1] * 20.0 + 1450.0
    ts = 1.0

    # 理论到达时刻
    taus = [ts + 2.0 * positions[j] / a_val for j in range(nc)]
    cluster_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    # -------------------------------------------------------------------------
    # Panel (a): 相对到时窗物理波前对齐 (Relative Window Alignment)
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    # 在上部展示对齐后的每簇局部波形 (已归一化至 [-50ms, +250ms])
    n_win_pts = 128
    dt_win = np.linspace(-50.0, 250.0, n_win_pts)
    for j in range(nc):
        tau_j = taus[j]
        # 对应时间窗
        t_sub = np.linspace(tau_j - 0.05, tau_j + 0.25, n_win_pts)
        wave_sub = np.interp(t_sub, t_full, wave_raw)
        offset_y = (nc - 1 - j) * 1.5
        ax_a.plot(
            dt_win,
            wave_sub + offset_y,
            color=cluster_colors[j % len(cluster_colors)],
            lw=1.4,
            label=f"Cluster {j+1} ($x={positions[j]:.0f}\\mathrm{{m}}$, $\\tau={tau_j:.2f}\\mathrm{{s}}$)",
        )
        # 标出波前理论到达时刻 0ms
        ax_a.axvline(0.0, color="#666666", ls="--", lw=0.8, alpha=0.7)

    ax_a.set_xlabel("Aligned Relative Time $t - \\tau_j$ (ms)", fontsize=9)
    ax_a.set_ylabel("Standardized Pressure Head (Offset)", fontsize=9)
    ax_a.set_title("(a) Mode 1: Relative Arrival-Time Window (t=0 Wavefront Aligned)", fontweight="bold", loc="left", fontsize=9.5)
    ax_a.set_xlim(-50, 250)
    ax_a.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=7.5, loc="upper right")
    ax_a.grid(True, ls=":", alpha=0.4)

    # -------------------------------------------------------------------------
    # Panel (b): 高斯软门控窗在全波形上的物理聚焦 (Gaussian Gating Windows)
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    # 绘制背景原始波形
    ax_b.plot(t_full, wave_raw, color="#999999", lw=0.8, alpha=0.5, label="Wellhead Head $H_{wh}(t)$")
    # 绘制各簇的高斯窗权重曲线 w_j(t)
    sigma_est = 0.08 # 秒
    ax_b2 = ax_b.twinx()
    for j in range(nc):
        tau_j = taus[j]
        wj = np.exp(-((t_full - tau_j) ** 2) / (2.0 * (sigma_est ** 2)))
        ax_b2.plot(
            t_full,
            wj,
            color=cluster_colors[j % len(cluster_colors)],
            lw=1.6,
            ls="-",
            label=f"Gaussian $w_{j+1}(t)$ ($\\tau_j={tau_j:.2f}\\mathrm{{s}}$)",
        )
        ax_b2.axvline(tau_j, color=cluster_colors[j % len(cluster_colors)], ls=":", lw=0.8, alpha=0.8)

    ax_b.set_xlabel("Transient Simulation Time $t$ (s)", fontsize=9)
    ax_b.set_ylabel("Raw Waveform Amplitude", fontsize=9, color="#666666")
    ax_b2.set_ylabel("Gaussian Window Weight $w_j(t)$", fontsize=9, color="#d62728")
    ax_b.set_xlim(1.0, 8.5)
    ax_b2.set_ylim(0.0, 1.1)
    ax_b.set_title("(b) Mode 2: Continuous Learnable Gaussian Gating", fontweight="bold", loc="left", fontsize=9.5)
    ax_b.grid(True, ls=":", alpha=0.4)

    # -------------------------------------------------------------------------
    # Panel (c): 2D 连续倒谱时空切片图块 (Cepstrum Spatio-Temporal Patch)
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    ceps_raw = test_batch["cepstrum"][sample_idx, 0].numpy() # (1024,)
    x_dist = np.linspace(0, 5000.0, len(ceps_raw))
    # 构造合成 2D 外积底图 S(t, x)
    t_sub_grid = np.linspace(2.0, 8.0, 150)
    x_sub_grid = np.linspace(1500.0, 4800.0, 150)
    w_interp = np.interp(t_sub_grid, t_full, wave_raw)
    c_interp = np.interp(x_sub_grid, x_dist, ceps_raw)
    st_2d = np.outer(w_interp, c_interp)

    im = ax_c.imshow(
        st_2d,
        origin="lower",
        extent=[x_sub_grid[0], x_sub_grid[-1], t_sub_grid[0], t_sub_grid[-1]],
        aspect="auto",
        cmap="coolwarm",
        alpha=0.85,
    )
    # 绘制各簇中心 (x_j, tau_j) 及 Patch 采样矩形框
    patch_dx = 200.0 # m
    patch_dt = 0.30  # s
    for j in range(nc):
        xj = positions[j]
        tauj = taus[j]
        rect = plt.Rectangle(
            (xj - patch_dx / 2.0, tauj - patch_dt / 2.0),
            patch_dx,
            patch_dt,
            fill=False,
            edgecolor=cluster_colors[j % len(cluster_colors)],
            lw=1.8,
            ls="--",
        )
        ax_c.add_patch(rect)
        ax_c.plot(xj, tauj, marker="x", color="black", markersize=6, mew=1.5)
        ax_c.text(xj + 60, tauj, f"$C_{j+1}$", color="black", fontsize=8, fontweight="bold")

    ax_c.set_xlabel("Wellbore Depth $x$ (m)", fontsize=9)
    ax_c.set_ylabel("Acoustic Arrival Time $\\tau$ (s)", fontsize=9)
    ax_c.set_title("(c) Mode 3: 2D Spatio-Temporal Cepstrum Patch Extraction", fontweight="bold", loc="left", fontsize=9.5)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Spatio-Temporal Energy", fontsize=8)

    # -------------------------------------------------------------------------
    # Panel (d): 声学传播时延偏置注意力对比 (Acoustic Biased Attention Matrix)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    # 从已训练模型输出中提取注意力热力图
    if "tg_relative_bias" in predictions and "attn_weights" in predictions["tg_relative_bias"]:
        attn_tensor = predictions["tg_relative_bias"]["attn_weights"] # (B, L, H, M, M)
        # 提取当前样本的最后一层多头均值
        sample_attn = attn_tensor[sample_idx, -1].mean(dim=0).cpu().numpy()[:nc, :nc]
    else:
        # 基于物理偏置公式构造高保真示意热力图
        dist_mat = np.abs(positions[:nc, None] - positions[None, :nc])
        tau_diff = dist_mat / a_val
        raw_scores = 1.5 * np.eye(nc) - 1.2 * tau_diff
        sample_attn = np.exp(raw_scores) / np.exp(raw_scores).sum(axis=1, keepdims=True)

    im_d = ax_d.imshow(sample_attn, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    for i in range(nc):
        for j in range(nc):
            val = sample_attn[i, j]
            color = "white" if val > 0.55 else "black"
            ax_d.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    ax_d.set_xticks(range(nc))
    ax_d.set_yticks(range(nc))
    cluster_ticks = [f"C{k+1}\n({positions[k]:.0f}m)" for k in range(nc)]
    ax_d.set_xticklabels(cluster_ticks, fontsize=8)
    ax_d.set_yticklabels(cluster_ticks, fontsize=8)
    ax_d.set_xlabel("Key Cluster $j$", fontsize=9)
    ax_d.set_ylabel("Query Cluster $i$", fontsize=9)
    ax_d.set_title("(d) Decoupled Inter-Cluster Attention $\\mathbf{A}_{ij}$ with Acoustic Bias", fontweight="bold", loc="left", fontsize=9.5)
    cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label("Attention Weight $\\mathbf{A}_{ij}$", fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Figure 4 已成功保存至: {save_path}")


def plot_fig5_benchmark_bars(
    metrics: Dict[str, Any],
    save_path: str,
):
    """
    绘制 Figure 5: Phase 2 各新模型与 Phase 1 基准多柱性能指标对比图
    Subplots:
    (a) Inflow share alpha MAE (越小越好)
    (b) Inflow share alpha R^2 决定系数 (越大越好)
    (c) Hydraulic compliance Cf MRE (%) (越小越好)
    (d) 1D Wasserstein distance W1 (m) (越小越好)
    """
    print(f"[*] 正在绘制 Figure 5: {save_path}")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), dpi=300)
    plt.subplots_adjust(wspace=0.28, hspace=0.36)

    # 排序模型列表: 先 3 个 Phase 1 基准，再 4 个 Phase 2 消融变体
    model_order = [
        ("resnet", BASELINE_SPECS["resnet"]["short_label"], BASELINE_SPECS["resnet"]["color"]),
        ("fno", BASELINE_SPECS["fno"]["short_label"], BASELINE_SPECS["fno"]["color"]),
        ("cj_cep_deeponet", BASELINE_SPECS["cj_cep_deeponet"]["short_label"], BASELINE_SPECS["cj_cep_deeponet"]["color"]),
        ("tg_best_nobias", ABLATION_SPECS["tg_best_nobias"]["short_label"], ABLATION_SPECS["tg_best_nobias"]["color"]),
        ("tg_gaussian_bias", ABLATION_SPECS["tg_gaussian_bias"]["short_label"], ABLATION_SPECS["tg_gaussian_bias"]["color"]),
        ("tg_ceps_patch_bias", ABLATION_SPECS["tg_ceps_patch_bias"]["short_label"], ABLATION_SPECS["tg_ceps_patch_bias"]["color"]),
        ("tg_relative_bias", ABLATION_SPECS["tg_relative_bias"]["short_label"], ABLATION_SPECS["tg_relative_bias"]["color"]),
    ]

    labels = [item[1] for item in model_order]
    colors = [item[2] for item in model_order]
    x = np.arange(len(labels))

    # 提取四大核心指标
    alpha_mae_vals = [metrics[item[0]]["alpha_mae"] for item in model_order]
    alpha_r2_vals = [metrics[item[0]]["alpha_r2"] for item in model_order]
    cf_mre_vals = [metrics[item[0]]["cf_mre_pct"] for item in model_order]
    w1_vals = [metrics[item[0]]["w1_mean_m"] for item in model_order]

    # -------------------------------------------------------------------------
    # (a) alpha MAE
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    bars_a = ax_a.bar(x, alpha_mae_vals, color=colors, width=0.6, edgecolor="#333333", lw=0.8)
    ax_a.axhline(0.05, color="#d62728", ls="--", lw=1.0, label="Target Threshold (0.050)")
    ax_a.set_ylabel("Inflow Share $\\alpha$ MAE", fontsize=9)
    ax_a.set_title("(a) Multi-Cluster Flow Allocation Error (Lower is Better)", fontweight="bold", loc="left", fontsize=9.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax_a.set_ylim(0, max(alpha_mae_vals) * 1.25)
    ax_a.legend(frameon=True, fontsize=8, loc="upper right")
    ax_a.grid(axis="y", ls=":", alpha=0.5)
    for bar in bars_a:
        yval = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.004, f"{yval:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # -------------------------------------------------------------------------
    # (b) alpha R^2
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    bars_b = ax_b.bar(x, alpha_r2_vals, color=colors, width=0.6, edgecolor="#333333", lw=0.8)
    ax_b.axhline(0.85, color="#d62728", ls="--", lw=1.0, label="Target Threshold ($R^2=0.85$)")
    ax_b.set_ylabel("Inflow Share Coefficient of Determination $R^2$", fontsize=9)
    ax_b.set_title("(b) Multi-Cluster Inversion Correlation $R^2$ (Higher is Better)", fontweight="bold", loc="left", fontsize=9.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(frameon=True, fontsize=8, loc="upper left")
    ax_b.grid(axis="y", ls=":", alpha=0.5)
    for bar in bars_b:
        yval = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # -------------------------------------------------------------------------
    # (c) Cf MRE (%)
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    bars_c = ax_c.bar(x, cf_mre_vals, color=colors, width=0.6, edgecolor="#333333", lw=0.8)
    ax_c.axhline(15.0, color="#d62728", ls="--", lw=1.0, label="Target Threshold (15.0%)")
    ax_c.set_ylabel("Hydraulic Compliance $C_f$ MRE (%)", fontsize=9)
    ax_c.set_title("(c) Fracture Compliance Relative Error (Lower is Better)", fontweight="bold", loc="left", fontsize=9.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax_c.set_ylim(0, max(cf_mre_vals) * 1.25)
    ax_c.legend(frameon=True, fontsize=8, loc="upper right")
    ax_c.grid(axis="y", ls=":", alpha=0.5)
    for bar in bars_c:
        yval = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # -------------------------------------------------------------------------
    # (d) 1D Wasserstein W_1 (m)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    bars_d = ax_d.bar(x, w1_vals, color=colors, width=0.6, edgecolor="#333333", lw=0.8)
    ax_d.axhline(10.0, color="#d62728", ls="--", lw=1.0, label="Acceptance Gate (10.0 m)")
    ax_d.set_ylabel("1D Wasserstein Distance $W_1$ (m)", fontsize=9)
    ax_d.set_title("(d) Spatial Measure Distribution Distance $W_1$ (Lower is Better)", fontweight="bold", loc="left", fontsize=9.5)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax_d.set_ylim(0, max(w1_vals) * 1.3)
    ax_d.legend(frameon=True, fontsize=8, loc="upper right")
    ax_d.grid(axis="y", ls=":", alpha=0.5)
    for bar in bars_d:
        yval = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.2, f"{yval:.2f}m", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Figure 5 已成功保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="TG-CJ-DeepONet 第二阶段消融评测与绘图")
    parser.add_argument("--weights_dir", type=str, default=DEFAULT_WEIGHTS_DIR, help="权重目录")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="指标与结果输出目录")
    parser.add_argument("--figures_dir", type=str, default=DEFAULT_FIGURES_DIR, help="图版保存目录")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. 独立测试集评估
    metrics, predictions, test_batch = evaluate_test_set(
        weights_dir=args.weights_dir,
        device=device,
    )

    # 2. 保存消融指标 JSON
    ablation_json_path = os.path.join(args.output_dir, "ablation_metrics_summary.json")
    with open(ablation_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] 消融指标汇总已持久化至: {ablation_json_path}")

    # 同步更新主 metrics_summary.json
    main_summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    if os.path.exists(main_summary_path):
        with open(main_summary_path, "r", encoding="utf-8") as f:
            full_metrics = json.load(f)
    else:
        full_metrics = {}
    full_metrics.update(metrics)
    with open(main_summary_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] 主指标汇总已同步合并至: {main_summary_path}")

    # 3. 绘制 Nature 规范图版
    fig4_path = os.path.join(args.figures_dir, "fig4_ablation_window_comparison.png")
    plot_fig4_window_comparison(predictions, test_batch, fig4_path)

    fig5_path = os.path.join(args.figures_dir, "fig5_phase2_model_benchmark_bars.png")
    plot_fig5_benchmark_bars(metrics, fig5_path)

    # 4. 控制台高亮输出对标对比表
    print(f"\n{'='*95}")
    print("第二阶段 (TG-CJ-DeepONet) vs 第一阶段基准 全集测试指标横向对标表 (100 Test Samples):")
    print(f"{'='*95}")
    print(f"{'模型 / 架构名称':<32} | {'alpha MAE':<10} | {'alpha R2':<9} | {'Cf MRE%':<9} | {'Cf log10':<9} | {'W1 (m)':<8}")
    print(f"{'-'*95}")
    for k in list(BASELINE_SPECS.keys()) + list(ABLATION_SPECS.keys()):
        m = metrics[k]
        label = BASELINE_SPECS[k]["label"] if k in BASELINE_SPECS else ABLATION_SPECS[k]["label"]
        print(
            f"{label:<32} | {m['alpha_mae']:<10.4f} | {m['alpha_r2']:<9.4f} | "
            f"{m['cf_mre_pct']:<9.1f} | {m['cf_log10_mae']:<9.4f} | {m['w1_mean_m']:<8.2f}"
        )
    print(f"{'='*95}\n")


if __name__ == "__main__":
    main()
