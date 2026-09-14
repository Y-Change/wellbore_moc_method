# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.evaluate_pilot

水击波物理反演小样本预研评估与成果图版生成脚本：
1. 在 100 个未见测试案例 (Test Split) 上全面评估 ResNet1D, FNO1D, VanillaDeepONet, CJCepDeepONet；
2. 计算 alpha MAE/R2, Cf MRE/Log-MAE, 1D Wasserstein 距离及变簇数分层指标；
3. 输出指标汇总表至 output/metrics_summary.json；
4. 绘制 Nature 规范复合图版：
   - output/figures/fig1_pilot_parity_plots.png: 真值 vs 预测散点一致性图；
   - output/figures/fig2_pilot_continuous_profiles.png: 典型多簇案例连续场重建曲线；
   - output/figures/fig3_pilot_model_comparison_bars.png: 四大模型性能指标对比柱状图；
5. 输出阶段验收合格核验报告。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 确保项目根目录在 sys.path 中
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
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics


# Nature 规范绘图全局参数
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

MODEL_REGISTRY = {
    "resnet": (ResNet1D, "1D-ResNet", "#1f77b4"),
    "fno": (FNO1D, "1D-FNO", "#2ca02c"),
    "deeponet": (VanillaDeepONet, "Vanilla DeepONet", "#ff7f0e"),
    "cj_cep_deeponet": (CJCepDeepONet, "CJ-Cep-DeepONet (Ours)", "#d62728"),
}


def load_model(model_key: str, weight_path: str, device: torch.device):
    cls = MODEL_REGISTRY[model_key][0]
    model = cls().to(device)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到模型权重文件: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")
DEFAULT_OUTPUT_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output")


def evaluate_all_models(
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    test_dataset = PilotInversionDataset(split="test")
    test_loader = test_dataset.get_dataloader(batch_size=100, shuffle=False)
    test_batch = next(iter(test_loader))
    test_batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}

    results = {}
    predictions = {}

    for m_key in MODEL_REGISTRY.keys():
        weight_path = os.path.join(weights_dir, f"{m_key}_best.pt")
        model = load_model(m_key, weight_path, device)
        with torch.no_grad():
            out = model(test_batch_dev)

        pred_cpu = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        targ_cpu = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}

        metrics = compute_inversion_metrics(pred_cpu, targ_cpu)
        results[m_key] = metrics
        predictions[m_key] = pred_cpu

    # 保存指标总结 JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Test metrics saved to: {json_path}")

    # 绘制图版
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_fig1_parity(predictions, test_batch, figures_dir)
    plot_fig2_continuous_profiles(predictions, test_batch, figures_dir)
    plot_fig3_model_comparison_bars(results, figures_dir)

    return results


def plot_fig1_parity(predictions: Dict[str, Any], test_batch: Dict[str, torch.Tensor], figures_dir: str):
    """绘制图 1: 真值 vs 预测散点一致性图版 (Parity Plots)"""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=300)

    mask = test_batch["mask"].numpy().astype(bool)
    true_alpha = test_batch["alpha"].numpy()[mask]
    true_cf = test_batch["cf"].numpy()[mask]
    true_log_cf = np.log10(np.maximum(true_cf, 1e-6))

    # A. 流量份额 Parity Plot
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Ideal Parity ($y=x$)")
    ax1.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="#e0e0e0", alpha=0.5, label=r"$\pm 0.05$ Acceptance")

    for m_key, (_, label, color) in MODEL_REGISTRY.items():
        p_alpha = predictions[m_key]["alpha"][mask]
        alpha_mae = np.mean(np.abs(p_alpha - true_alpha))
        ax1.scatter(true_alpha, p_alpha, s=18, color=color, alpha=0.6, edgecolors="none", label=f"{label} (MAE={alpha_mae:.3f})")

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("True Flow Contribution Fraction $\\alpha_j$", fontweight="bold")
    ax1.set_ylabel("Inverted Flow Contribution $\\hat{\\alpha}_j$", fontweight="bold")
    ax1.set_title("Flow Contribution Inversion Consistency", pad=6)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7)
    ax1.text(-0.15, 1.05, "a", transform=ax1.transAxes, fontsize=12, fontweight="bold")

    # B. 水力顺应性 Parity Plot (Log10 scale)
    ax2 = axes[1]
    min_log, max_log = -3.8, -1.1
    ax2.plot([min_log, max_log], [min_log, max_log], "k--", lw=1.2, alpha=0.7, label="Ideal Parity")
    ax2.fill_between([min_log, max_log], [min_log - 0.15, max_log - 0.15], [min_log + 0.15, max_log + 0.15],
                     color="#e0e0e0", alpha=0.5, label=r"$\pm 0.15$ Log Gate")

    for m_key, (_, label, color) in MODEL_REGISTRY.items():
        p_cf = predictions[m_key]["cf"][mask]
        p_log_cf = np.log10(np.maximum(p_cf, 1e-6))
        cf_mre = np.median(np.abs(p_cf - true_cf) / true_cf) * 100.0
        ax2.scatter(true_log_cf, p_log_cf, s=18, color=color, alpha=0.6, edgecolors="none", label=f"{label} (MRE={cf_mre:.1f}%)")

    ax2.set_xlim(min_log, max_log)
    ax2.set_ylim(min_log, max_log)
    ax2.set_xlabel("True $\\log_{10}(C_{f, j} \\; [\\mathrm{m^2}])$", fontweight="bold")
    ax2.set_ylabel("Inverted $\\log_{10}(\\hat{C}_{f, j} \\; [\\mathrm{m^2}])$", fontweight="bold")
    ax2.set_title("Hydraulic Compliance Inversion Consistency", pad=6)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7)
    ax2.text(-0.15, 1.05, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(figures_dir, "fig1_pilot_parity_plots.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Figure 1 generated: {save_path}")


def plot_fig2_continuous_profiles(predictions: Dict[str, Any], test_batch: Dict[str, torch.Tensor], figures_dir: str):
    """绘制图 2: 典型变簇数案例连续重构场剖面与真值对比"""
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.5), dpi=300)
    grid_x = np.linspace(0, 5000, 500)

    # 选取 3 个具有代表性簇数 (Nc=2, 4, 6) 的案例
    n_frac_arr = test_batch["n_frac"].numpy()
    target_ncs = [2, 4, 6]
    case_indices = []
    for t_nc in target_ncs:
        matches = np.where(n_frac_arr == t_nc)[0]
        case_indices.append(matches[0] if len(matches) > 0 else 0)

    titles = [
        "Case 1: Dual-cluster ($N_c=2$) Flow Density Reconstruction",
        "Case 2: Four-cluster ($N_c=4$) Flow Density Reconstruction",
        "Case 3: Six-cluster ($N_c=6$) Multi-cluster Challenge",
    ]
    labels_tag = ["a", "b", "c"]

    for idx, (c_idx, title, tag) in enumerate(zip(case_indices, titles, labels_tag)):
        ax = axes[idx]
        nc = int(n_frac_arr[c_idx])
        pos = test_batch["positions"][c_idx].numpy()[:nc]
        true_w = test_batch["alpha"][c_idx].numpy()[:nc]
        true_m = test_batch["m_alpha_grid"][c_idx].numpy()

        # 绘制真值连续场
        ax.plot(grid_x, true_m * 1000, color="black", lw=1.8, label="Ground Truth Field $m_\\alpha(x)$")

        # 绘制各模型连续预测场
        for m_key, (_, label, color) in MODEL_REGISTRY.items():
            if "m_alpha_grid" in predictions[m_key]:
                pred_m = predictions[m_key]["m_alpha_grid"][c_idx]
                ls = "-" if "cj_cep" in m_key else "--"
                lw = 1.6 if "cj_cep" in m_key else 1.1
                ax.plot(grid_x, pred_m * 1000, color=color, linestyle=ls, lw=lw, label=f"{label} Field")

        # 标注射孔簇真值位置
        for j in range(nc):
            ax.axvline(pos[j], color="crimson", linestyle=":", lw=1.0, alpha=0.8)
            ax.scatter([pos[j]], [true_w[j] * 10], color="crimson", marker="v", s=30, zorder=5)

        ax.set_xlim(3200, 4800)  # 聚集在主要射孔段
        ax.set_ylabel("Density ($\\times 10^{-3} \\, \\mathrm{m^{-1}}$)")
        ax.set_title(title, pad=4)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.text(-0.08, 1.05, tag, transform=ax.transAxes, fontsize=11, fontweight="bold")
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=7, ncol=2)

    axes[-1].set_xlabel("Wellbore Depth Coordinate $x$ (m)", fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "fig2_pilot_continuous_profiles.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Figure 2 generated: {save_path}")


def plot_fig3_model_comparison_bars(results: Dict[str, Any], figures_dir: str):
    """绘制图 3: 四大模型性能指标多柱状对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), dpi=300)

    model_keys = list(MODEL_REGISTRY.keys())
    model_labels = [MODEL_REGISTRY[k][1] for k in model_keys]
    colors = [MODEL_REGISTRY[k][2] for k in model_keys]
    x = np.arange(len(model_keys))
    width = 0.55

    # Panel A: alpha MAE (门槛 < 0.050)
    ax_a = axes[0, 0]
    vals_alpha = [results[k]["alpha_mae"] for k in model_keys]
    bars_a = ax_a.bar(x, vals_alpha, width, color=colors, edgecolor="black", lw=0.8)
    ax_a.axhline(0.050, color="crimson", linestyle="--", lw=1.2, label="Acceptance Gate (< 0.050)")
    ax_a.set_ylabel("$\\alpha$ Mean Absolute Error", fontweight="bold")
    ax_a.set_title("Flow Fraction MAE (Lower is Better)", pad=5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(model_labels, rotation=15, ha="right")
    ax_a.set_ylim(0, max(vals_alpha) * 1.35)
    ax_a.grid(axis="y", linestyle=":", alpha=0.5)
    ax_a.legend(loc="upper right", fontsize=7.5)
    ax_a.text(-0.15, 1.05, "a", transform=ax_a.transAxes, fontsize=11, fontweight="bold")
    for b in bars_a:
        h = b.get_height()
        ax_a.text(b.get_x() + b.get_width()/2., h + 0.002, f"{h:.4f}", ha="center", va="bottom", fontsize=7.5)

    # Panel B: alpha R2 (门槛 > 0.850)
    ax_b = axes[0, 1]
    vals_r2 = [results[k]["alpha_r2"] for k in model_keys]
    bars_b = ax_b.bar(x, vals_r2, width, color=colors, edgecolor="black", lw=0.8)
    ax_b.axhline(0.0, color="#888888", linestyle="-", lw=0.8)
    ax_b.axhline(0.850, color="crimson", linestyle="--", lw=1.2, label="Acceptance Gate (> 0.850)")
    ax_b.set_ylabel(r"$\alpha$ Determination ($R^2$)", fontweight="bold")
    ax_b.set_title(r"Flow Fraction $R^2$ (Higher is Better)", pad=5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(model_labels, rotation=15, ha="right")
    min_r2 = min(vals_r2)
    ax_b.set_ylim(min(-0.25, min_r2 - 0.1), 1.05)
    ax_b.grid(axis="y", linestyle=":", alpha=0.5)
    ax_b.legend(loc="upper right", fontsize=7.5)
    ax_b.text(-0.15, 1.05, "b", transform=ax_b.transAxes, fontsize=11, fontweight="bold")
    for b in bars_b:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.02 if h >= 0 else -0.02
        ax_b.text(b.get_x() + b.get_width()/2., h + offset, f"{h:.3f}", ha="center", va=va, fontsize=7.5)

    # Panel C: Cf MRE % (门槛 < 15.0%)
    ax_c = axes[1, 0]
    vals_cf = [results[k]["cf_mre_pct"] for k in model_keys]
    bars_c = ax_c.bar(x, vals_cf, width, color=colors, edgecolor="black", lw=0.8)
    ax_c.axhline(15.0, color="crimson", linestyle="--", lw=1.2, label="Acceptance Gate (< 15.0%)")
    ax_c.set_ylabel("Compliance MRE [%]", fontweight="bold")
    ax_c.set_title("Compliance Median Rel. Error (Lower is Better)", pad=5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(model_labels, rotation=15, ha="right")
    ax_c.set_ylim(0, max(vals_cf) * 1.35)
    ax_c.grid(axis="y", linestyle=":", alpha=0.5)
    ax_c.legend(loc="upper right", fontsize=7.5)
    ax_c.text(-0.15, 1.05, "c", transform=ax_c.transAxes, fontsize=11, fontweight="bold")
    for b in bars_c:
        h = b.get_height()
        ax_c.text(b.get_x() + b.get_width()/2., h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    # Panel D: 1D Wasserstein distance (门槛 < 10.0m)
    ax_d = axes[1, 1]
    vals_w1 = [results[k]["w1_mean_m"] for k in model_keys]
    bars_d = ax_d.bar(x, vals_w1, width, color=colors, edgecolor="black", lw=0.8)
    ax_d.axhline(10.0, color="crimson", linestyle="--", lw=1.2, label="Acceptance Gate (< 10.0 m)")
    ax_d.set_ylabel("1D Wasserstein Distance [m]", fontweight="bold")
    ax_d.set_title("Spatial Depth Equiv. Error (Lower is Better)", pad=5)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(model_labels, rotation=15, ha="right")
    ax_d.set_ylim(0, max(vals_w1) * 1.35)
    ax_d.grid(axis="y", linestyle=":", alpha=0.5)
    ax_d.legend(loc="upper right", fontsize=7.5)
    ax_d.text(-0.15, 1.05, "d", transform=ax_d.transAxes, fontsize=11, fontweight="bold")
    for b in bars_d:
        h = b.get_height()
        ax_d.text(b.get_x() + b.get_width()/2., h + 0.1, f"{h:.2f}m", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, "fig3_pilot_model_comparison_bars.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Figure 3 generated: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 开始在测试集 (100 Cases) 上执行综合模型评估...")
    results = evaluate_all_models(device=device)

    print("\n" + "=" * 80)
    print("                    四大模型测试集验收指标全景对比表                    ")
    print("=" * 80)
    fmt_header = f"{'Model':<20} | {'alpha MAE':<10} | {'alpha R2':<10} | {'Cf MRE (%)':<11} | {'Cf Log10':<9} | {'W1 (m)':<8} | {'Gate Status':<11}"
    print(fmt_header)
    print("-" * 80)

    for m_key, (_, label, _) in MODEL_REGISTRY.items():
        m = results[m_key]
        pass_alpha = m["alpha_mae"] < 0.050
        pass_r2 = m["alpha_r2"] > 0.850
        pass_cf = m["cf_mre_pct"] < 15.0
        pass_w1 = m["w1_mean_m"] < 10.0
        all_pass = pass_alpha and pass_r2 and pass_cf and pass_w1
        status = "[PASSED]" if all_pass else "[MARGINAL]"

        print(
            f"{label:<20} | "
            f"{m['alpha_mae']:<10.4f} | "
            f"{m['alpha_r2']:<10.4f} | "
            f"{m['cf_mre_pct']:<11.2f} | "
            f"{m['cf_log10_mae']:<9.4f} | "
            f"{m['w1_mean_m']:<8.2f} | "
            f"{status:<11}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
