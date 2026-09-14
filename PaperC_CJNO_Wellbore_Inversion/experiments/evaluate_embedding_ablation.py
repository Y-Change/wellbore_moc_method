# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.evaluate_embedding_ablation

端到端连续时空 Embedding 架构消融评测与 Nature 规范图版生成脚本:
1. 在 100 个独立盲测样本 (Test Split) 上系统评测四大 Embedding 变体:
   - 'baseline_resample': 离线 4096 点机械降采样基准 (TG-DIS-Baseline);
   - 'physics_query':     方案一 (物理到时锚定 Query + 原生 1000 Hz 波包声学偏置 Cross-Attention);
   - 'fourier_feature':   方案二 (多尺度高频连续傅里叶字典嵌入 0.0725 Hz ~ 500 Hz);
   - 'sinc_filterbank':   方案三 (井筒声学奇数次驻波谐频 Sinc 带通可微滤波器组);
2. 计算密集多簇 (Nc>=4) R^2、总体 R^2、alpha MAE、Cf Log10 MAE、空间 W1 距离、F1 分数、
   单纯形守恒最大偏差及单样本 CPU 推理耗时 (ms);
3. 输出结构化指标 JSON: output/embedding_ablation_summary.json;
4. 绘制 Nature 规范四联横向对标图版 output/figures/fig_embedding_comparison.png (300 DPI) 与 .svg:
   - Panel a: 50-Epoch 收敛演化曲线 (训练损失与验证损失);
   - Panel b: 密集多簇 (Nc>=4) R^2 与全集 R^2 对比柱状图;
   - Panel c: 流量分配 MAE 与空间 Wasserstein-1 距离对比;
   - Panel d: 水力顺应性 Log10 MAE 与单样本 CPU 推理延迟。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
import torch.nn as nn

from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_embedding_variants import TGDISEmbeddingModel
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics

# Nature 规范绘图参数
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DEFAULT_OUTPUT_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output")
DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")
DEFAULT_FIGURES_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "figures")

VARIANT_META = {
    "baseline_resample": {
        "label": "TG-DIS-Baseline\n(4096-pts Resample)",
        "short_name": "Baseline (4096)",
        "color": "#7f7f7f",  # 灰色基准
    },
    "physics_query": {
        "label": "TG-DIS-QueryEmbed\n(Physics Query Attn)",
        "short_name": "QueryEmbed (Ours)",
        "color": "#d62728",  # 旗舰红
    },
    "fourier_feature": {
        "label": "TG-DIS-FourierEmbed\n(Continuous Fourier)",
        "short_name": "FourierEmbed",
        "color": "#1f77b4",  # 科技蓝
    },
    "sinc_filterbank": {
        "label": "TG-DIS-SincEmbed\n(Acoustic SincNet)",
        "short_name": "SincEmbed",
        "color": "#2ca02c",  # 物理绿
    },
}


def evaluate_variant_on_test(
    embedding_type: str,
    test_loader: torch.utils.data.DataLoader,
    weight_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """在 100 个盲测样本上运行模型推理并计算完整评估指标"""
    model = TGDISEmbeddingModel(
        embedding_type=embedding_type,
        d_model=64,
        max_nc=6,
        n_grid=500,
        L=5000.0,
    ).to(device)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件: {weight_path}")

    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_preds_alpha = []
    all_preds_cf = []
    all_preds_exist = []
    all_preds_dx = []
    all_preds_field = []

    all_targets_alpha = []
    all_targets_cf = []
    all_targets_field = []
    all_masks = []
    all_positions = []

    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            t0 = time.perf_counter()
            out = model(batch_dev)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / batch["wave"].shape[0])

            all_preds_alpha.append(out["alpha"].cpu())
            all_preds_cf.append(out["cf"].cpu())
            if "p_exist" in out:
                all_preds_exist.append(out["p_exist"].cpu())
            if "delta_x" in out:
                all_preds_dx.append(out["delta_x"].cpu())
            all_preds_field.append(out["m_alpha_grid"].cpu())

            all_targets_alpha.append(batch["alpha"])
            all_targets_cf.append(batch["cf"])
            all_targets_field.append(batch["m_alpha_grid"])
            all_masks.append(batch["mask"])
            all_positions.append(batch["positions"])

    pred_alpha = torch.cat(all_preds_alpha, dim=0)
    pred_cf = torch.cat(all_preds_cf, dim=0)
    pred_field = torch.cat(all_preds_field, dim=0)

    targ_alpha = torch.cat(all_targets_alpha, dim=0)
    targ_cf = torch.cat(all_targets_cf, dim=0)
    targ_field = torch.cat(all_targets_field, dim=0)
    masks = torch.cat(all_masks, dim=0)
    positions = torch.cat(all_positions, dim=0)

    pred_dict: Dict[str, Any] = {
        "alpha": pred_alpha,
        "cf": pred_cf,
        "m_alpha_grid": pred_field,
    }
    if all_preds_exist:
        pred_dict["p_exist"] = torch.cat(all_preds_exist, dim=0)
    if all_preds_dx:
        pred_dict["delta_x"] = torch.cat(all_preds_dx, dim=0)

    targ_dict = {
        "alpha": targ_alpha,
        "cf": targ_cf,
        "m_alpha_grid": targ_field,
        "mask": masks,
        "positions": positions,
    }

    metrics = compute_inversion_metrics(pred_dict, targ_dict)

    # 计算单纯形守恒最大偏差
    alpha_sums = pred_alpha.sum(dim=-1)
    simplex_max_err = float(torch.max(torch.abs(alpha_sums - 1.0)).item())

    # 平均 CPU 推理延迟 (ms/样本)
    mean_latency_ms = float(np.mean(latencies) * 1000.0)

    metrics["simplex_max_violation"] = simplex_max_err
    metrics["latency_ms_per_sample"] = mean_latency_ms
    metrics["embedding_type"] = embedding_type

    return metrics


def plot_embedding_comparison_figures(
    summary_data: Dict[str, Any],
    histories: Dict[str, List[Dict[str, Any]]],
    output_png: str,
    output_svg: str,
):
    """绘制 Nature 规范四联横向对标图版"""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

    variants = list(VARIANT_META.keys())

    # -------------------------------------------------------------
    # Panel a: 50-Epoch 收敛演化曲线 (验证损失)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    for var in variants:
        hist = histories.get(var, [])
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        val_losses = [h["val_loss"] for h in hist]
        meta = VARIANT_META[var]
        ax_a.plot(epochs, val_losses, label=meta["short_name"], color=meta["color"], lw=1.5)

    ax_a.set_xlabel("Training Epochs")
    ax_a.set_ylabel("Validation Loss")
    ax_a.set_title("a  Training Convergence Evolution", loc="left", fontweight="bold")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(frameon=False, loc="upper right")

    # -------------------------------------------------------------
    # Panel b: Dense R^2 (Nc>=4) 与 Overall R^2 对比柱状图
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    x_indices = np.arange(len(variants))
    bar_width = 0.35

    dense_r2s = [summary_data[v].get("alpha_r2_dense", 0.0) for v in variants]
    all_r2s = [summary_data[v].get("alpha_r2", 0.0) for v in variants]

    colors = [VARIANT_META[v]["color"] for v in variants]
    bars1 = ax_b.bar(x_indices - bar_width/2, dense_r2s, width=bar_width, label=r"Dense ($N_c \geq 4$)", color=colors, alpha=0.9, edgecolor="#333333", lw=0.6)
    bars2 = ax_b.bar(x_indices + bar_width/2, all_r2s, width=bar_width, label=r"Overall ($N_c \in [1..6])$", color=colors, alpha=0.45, hatch="//", edgecolor="#333333", lw=0.6)

    # 柱顶数值标注
    for b in bars1:
        h = b.get_height()
        ax_b.annotate(f"{h:+.3f}",
                      xy=(b.get_x() + b.get_width() / 2, h),
                      xytext=(0, 2 if h >= 0 else -10),
                      textcoords="offset points",
                      ha="center", va="bottom" if h >= 0 else "top",
                      fontsize=6.5)

    ax_b.set_xticks(x_indices)
    ax_b.set_xticklabels([VARIANT_META[v]["short_name"] for v in variants], rotation=12, ha="right")
    ax_b.set_ylabel("Flow Allocation $R^2$ Score")
    ax_b.set_title("b  Multi-Cluster Inversion Accuracy ($R^2$)", loc="left", fontweight="bold")
    ax_b.axhline(0, color="#666666", lw=0.8, linestyle="--")
    ax_b.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax_b.legend(frameon=False, loc="lower right")

    # -------------------------------------------------------------
    # Panel c: Flow allocation MAE 与 Spatial W1 距离对比
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    maes = [summary_data[v].get("alpha_mae", 0.0) for v in variants]
    w1s = [summary_data[v].get("w1_mean_m", 0.0) for v in variants]

    bars_mae = ax_c.bar(x_indices - bar_width/2, maes, width=bar_width, label="Flow Fraction MAE", color=colors, alpha=0.9, edgecolor="#333333", lw=0.6)

    ax_c_twin = ax_c.twinx()
    bars_w1 = ax_c_twin.bar(x_indices + bar_width/2, w1s, width=bar_width, label="Spatial $W_1$ (m)", color=colors, alpha=0.45, hatch="\\\\", edgecolor="#333333", lw=0.6)

    ax_c.set_xticks(x_indices)
    ax_c.set_xticklabels([VARIANT_META[v]["short_name"] for v in variants], rotation=12, ha="right")
    ax_c.set_ylabel("Allocation MAE (Unitless)")
    ax_c_twin.set_ylabel("Wasserstein-1 Distance $W_1$ (m)")
    ax_c.set_title("c  Allocation Error & Spatial Metric $W_1$", loc="left", fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax_c.set_ylim(0, max(maes) * 1.35)
    ax_c_twin.set_ylim(0, max(w1s) * 1.35)

    # 联合图例
    lines, labels = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c_twin.get_legend_handles_labels()
    ax_c.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")

    # -------------------------------------------------------------
    # Panel d: Compliance Log10 MAE 与 CPU 推理延迟
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    cf_maes = [summary_data[v].get("cf_log10_mae", 0.0) for v in variants]
    latencies = [summary_data[v].get("latency_ms_per_sample", 0.0) for v in variants]

    bars_cf = ax_d.bar(x_indices - bar_width/2, cf_maes, width=bar_width, label="$C_f$ Log10 MAE", color=colors, alpha=0.9, edgecolor="#333333", lw=0.6)

    ax_d_twin = ax_d.twinx()
    bars_lat = ax_d_twin.bar(x_indices + bar_width/2, latencies, width=bar_width, label="CPU Latency (ms)", color=colors, alpha=0.45, hatch="xx", edgecolor="#333333", lw=0.6)

    ax_d.set_xticks(x_indices)
    ax_d.set_xticklabels([VARIANT_META[v]["short_name"] for v in variants], rotation=12, ha="right")
    ax_d.set_ylabel("Compliance $\\log_{10}(C_f/C_0)$ MAE")
    ax_d_twin.set_ylabel("Latency per Sample (ms)")
    ax_d.set_title("d  Hydraulic Compliance & CPU Latency", loc="left", fontweight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax_d.set_ylim(0, max(cf_maes) * 1.35)
    ax_d_twin.set_ylim(0, max(latencies) * 1.45)

    lines, labels = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d_twin.get_legend_handles_labels()
    ax_d.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_svg)
    plt.close(fig)
    print(f"[*] 成果图版已保存至:\n    - PNG: {output_png}\n    - SVG: {output_svg}")


def run_embedding_ablation_evaluation(
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    figures_dir: str = DEFAULT_FIGURES_DIR,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """主评测执行入口"""
    device = torch.device(device_str)
    print(f"\n{'#'*80}")
    print(f"[*] 启动端到端连续时空 Embedding 四大变体测试集盲测评测")
    print(f"[*] 设备: {device} | 权重目录: {weights_dir}")
    print(f"{'#'*80}\n")

    # 1. 加载 100 个独立盲测样本
    print("[*] 正在加载独立盲测集 (100 个真实物理仿真案例)...")
    test_dataset = PilotInversionDataset(split="test", load_raw_wave=True)
    test_loader = test_dataset.get_dataloader(batch_size=32, shuffle=False)
    print(f"[*] 盲测样本数: {len(test_dataset)}")

    variants = list(VARIANT_META.keys())
    summary: Dict[str, Any] = {}
    histories: Dict[str, List[Dict[str, Any]]] = {}

    for var in variants:
        weight_path = os.path.join(weights_dir, f"embedding_ablation_{var}_best.pt")
        history_path = os.path.join(weights_dir, f"embedding_ablation_{var}_history.json")

        print(f"\n[*] 正在评测变体: {var} ({VARIANT_META[var]['short_name']})...")
        metrics = evaluate_variant_on_test(var, test_loader, weight_path, device)
        summary[var] = metrics

        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                h_data = json.load(f)
                histories[var] = h_data.get("history", [])

        print(f"    - Dense R^2: {metrics.get('alpha_r2_dense', 0.0):+.4f}")
        print(f"    - Overall R^2: {metrics.get('alpha_r2', 0.0):+.4f}")
        print(f"    - Alpha MAE: {metrics.get('alpha_mae', 0.0):.4f}")
        print(f"    - Cf Log10 MAE: {metrics.get('cf_log10_mae', 0.0):.4f}")
        print(f"    - W1 距离: {metrics.get('w1_mean_m', 0.0):.2f} m")
        print(f"    - F1-score: {metrics.get('f1_score', 0.0):.4f}")
        print(f"    - 单纯形最大偏差: {metrics.get('simplex_max_violation', 0.0):.2e}")
        print(f"    - CPU 延迟: {metrics.get('latency_ms_per_sample', 0.0):.2f} ms/样本")

    # 保存指标摘要 JSON
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "embedding_ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 综合评测指标 JSON 已保存至: {summary_path}")

    # 绘制对比图版
    os.makedirs(figures_dir, exist_ok=True)
    fig_png = os.path.join(figures_dir, "fig_embedding_comparison.png")
    fig_svg = os.path.join(figures_dir, "fig_embedding_comparison.svg")
    plot_embedding_comparison_figures(summary, histories, fig_png, fig_svg)

    # 打印对比汇总表
    print(f"\n{'='*95}")
    print(f"{'Embedding 变体名称':<22} {'Dense R^2':<12} {'Overall R^2':<12} {'Alpha MAE':<10} {'W1 距离 (m)':<12} {'F1-score':<10} {'CPU 耗时(ms)':<10}")
    print(f"{'-'*95}")
    for var, m in summary.items():
        print(
            f"{VARIANT_META[var]['short_name']:<22} "
            f"{m.get('alpha_r2_dense', 0.0):+10.4f}  "
            f"{m.get('alpha_r2', 0.0):+10.4f}  "
            f"{m.get('alpha_mae', 0.0):8.4f}  "
            f"{m.get('w1_mean_m', 0.0):10.2f}  "
            f"{m.get('f1_score', 0.0):8.4f}  "
            f"{m.get('latency_ms_per_sample', 0.0):8.2f}"
        )
    print(f"{'='*95}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="端到端连续时空 Embedding 消融盲测评测与绘图")
    parser.add_argument("--weights-dir", type=str, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=str, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_embedding_ablation_evaluation(
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
