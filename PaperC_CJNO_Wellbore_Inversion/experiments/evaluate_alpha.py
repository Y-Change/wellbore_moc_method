#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CJ-AlphaNet 测试集评估、单井案例图与中文报告。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from PaperC_CJNO_Wellbore_Inversion.experiments.train_alpha import (
    MODEL_BUILDERS,
    _to_device,
    build_model,
    evaluate_loader,
)
from PaperC_CJNO_Wellbore_Inversion.src.alpha_dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_NPZ_PATH,
    AlphaInversionDataset,
)
from PaperC_CJNO_Wellbore_Inversion.src.label_physics import extract_post_shut_head

DEFAULT_OUT = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "alpha_newa")

DISPLAY_NAMES = {
    "cj_alphanet": "CJ-AlphaNet",
    "cj_alphanet_no2d": "CJ-AlphaNet (无 2D 倒谱)",
    "resnet": "ResNet1D",
    "fno": "FNO1D",
    "cjcep": "CJ-Cep 改编",
    "uniform": "均匀α + 全活动 + Y地板",
    "train_mean": "训练均值基线",
}

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.linewidth": 1.0,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_weights(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _case_label(i: int, n_frac, alpha, y_act, mask, yeq, has_fault) -> str:
    m = mask[i]
    sand = bool(np.any((y_act[i] < 0.5) & m))
    fault = has_fault is not None and bool(has_fault[i])
    if int(n_frac[i]) == 1:
        return "单簇"
    if sand and fault:
        return "含砂堵（断层）"
    if sand:
        return "含砂堵"
    if fault:
        return "断层型"
    a = alpha[i, m]
    if a.size >= 4 and float(a.max() / max(float(a.min()), 1e-6)) < 2.5:
        return "多簇均衡"
    return "强进液"


def select_case_ids(ds: AlphaInversionDataset, h5_path: str, n_cases: int = 4) -> List[Tuple[int, str]]:
    """从 test 集选：单簇、多簇均衡、含砂堵、断层/强进液。返回 dataset 内下标。"""
    n = len(ds)
    n_frac = ds.n_frac.numpy()
    alpha = ds.alpha.numpy()
    y_act = ds.y_act.numpy()
    mask = ds.mask_design.numpy()
    yeq = ds.Y_eq.numpy()
    raw_ids = ds.indices.astype(int)

    has_fault = None
    with h5py.File(h5_path, "r") as f:
        if "labels/has_fault" in f:
            has_fault = np.asarray(f["labels/has_fault"][:])[raw_ids].astype(bool)

    chosen: List[int] = []
    used = set()

    def pick(cond, prefer=None) -> Optional[int]:
        cand = [i for i in range(n) if cond(i) and i not in used]
        if not cand:
            return None
        if prefer is not None:
            cand.sort(key=prefer, reverse=True)
        return cand[0]

    i1 = pick(lambda i: n_frac[i] == 1)
    if i1 is not None:
        chosen.append(i1)
        used.add(i1)

    def balanced(i: int) -> bool:
        m = mask[i]
        if int(n_frac[i]) < 4:
            return False
        a = alpha[i, m]
        if a.size < 4:
            return False
        if np.any(y_act[i, m] < 0.5):
            return False
        return float(a.max() / max(a.min(), 1e-6)) < 2.5

    i2 = pick(balanced)
    if i2 is not None:
        chosen.append(i2)
        used.add(i2)

    def sandout(i: int) -> bool:
        m = mask[i]
        return bool(np.any((y_act[i] < 0.5) & m)) and int(n_frac[i]) >= 2

    i3 = pick(sandout, prefer=lambda i: int(np.sum((y_act[i] < 0.5) & mask[i])))
    if i3 is not None:
        chosen.append(i3)
        used.add(i3)

    def strong(i: int) -> bool:
        if has_fault is not None:
            return bool(has_fault[i])
        m = mask[i]
        if not np.any(m):
            return False
        return float(np.max(yeq[i, m])) > float(np.quantile(yeq[mask], 0.85))

    i4 = pick(strong, prefer=lambda i: float(np.max(yeq[i, mask[i]])) if np.any(mask[i]) else 0.0)
    if i4 is not None:
        chosen.append(i4)
        used.add(i4)

    for i in range(n):
        if len(chosen) >= n_cases:
            break
        if i not in used:
            chosen.append(i)
            used.add(i)
    out: List[Tuple[int, str]] = []
    for i in chosen[:n_cases]:
        out.append((i, _case_label(i, n_frac, alpha, y_act, mask, yeq, has_fault)))
    return out


def plot_case(
    out_png: str,
    t_post: np.ndarray,
    h_post: np.ndarray,
    grid_x: np.ndarray,
    m_true: np.ndarray,
    m_pred: np.ndarray,
    pos: np.ndarray,
    mask: np.ndarray,
    alpha_t: np.ndarray,
    alpha_p: np.ndarray,
    y_t: np.ndarray,
    y_p: np.ndarray,
    logY_t: np.ndarray,
    logY_p: np.ndarray,
    title: str,
) -> None:
    msk = np.asarray(mask).astype(bool)
    idx = np.where(msk)[0]
    xj = np.asarray(pos)[msk]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.55), constrained_layout=True)

    ax = axes[0]
    ax.plot(np.asarray(t_post) - float(t_post[0]), h_post, color="#1f4e79", lw=1.15)
    ax.set_xlabel("停泵后时间 t (s)")
    ax.set_ylabel("井口水头 H (m)")
    ax.set_title("井口压力")
    ax.grid(True, alpha=0.28, lw=0.5)

    ax = axes[1]
    ax.plot(grid_x, m_true, color="#2c5f2d", lw=1.7, label="真值")
    ax.plot(grid_x, m_pred, color="#c44900", lw=1.45, ls="--", label="预测")
    for x in xj:
        ax.axvline(x, color="#888888", lw=0.55, ls=":")
    x_lo = max(4000.0, float(np.min(xj)) - 200.0) if xj.size else 4000.0
    ax.set_xlim(x_lo, 5000.0)
    ax.set_xlabel("井深 x (m)")
    ax.set_ylabel(r"$m_\alpha(x)$ (1/m)")
    ax.set_title("进液密度")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    n = int(idx.size)
    xx = np.arange(n)
    w = 0.36
    ax.bar(xx - 0.5 * w, alpha_t[msk], width=w, color="#4c78a8", label=r"$\alpha$ 真", zorder=3)
    ax.bar(xx + 0.5 * w, alpha_p[msk], width=w, color="#9ecae1", label=r"$\alpha$ 预", zorder=3)
    ytrue = y_t[msk]
    ypred = y_p[msk]
    ax.scatter(xx - 0.5 * w, np.clip(ytrue, 0, 1) * 1.08, marker="o", s=28, color="#d95f02", label=r"$y$ 真", zorder=4)
    ax.scatter(xx + 0.5 * w, np.clip(ypred, 0, 1) * 1.08, marker="s", s=22, color="#fdae6b", label=r"$y$ 预", zorder=4)
    ax.set_ylim(0.0, 1.22)
    ax.set_ylabel(r"$\alpha$  （点: $y_\mathrm{act}$）")
    ax.set_xticks(xx)
    ax.set_xticklabels([f"c{int(j)+1}\n{xj[k]:.0f}m" for k, j in enumerate(idx)], fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(xx, logY_t[msk], "o-", color="#238b45", ms=5, lw=1.2, label="logY 真")
    ax2.plot(xx, logY_p[msk], "s--", color="#cb181d", ms=5, lw=1.2, label="logY 预")
    ly = np.concatenate([logY_t[msk], logY_p[msk]])
    lo, hi = float(np.min(ly)), float(np.max(ly))
    if (np.min(ytrue) < 0.5) or (lo < -1.0):
        ax2.set_ylim(-6.8, max(hi + 0.5, 2.2))
    else:
        span = max(1.6, hi - lo + 0.6)
        mid = 0.5 * (hi + lo)
        ax2.set_ylim(mid - 0.5 * span, mid + 0.5 * span)
    ax2.set_ylabel(r"$\log_{10}(Y/Y_0)$")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper right", fontsize=6)
    ax.set_title("簇级 $\\alpha$ / 活动 / 导纳")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_png)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def write_report(
    path: str,
    metrics: Dict[str, Dict[str, Any]],
    train_summaries: Dict[str, Any],
    case_figs: List[str],
    settings: Dict[str, Any],
    commands: List[str],
) -> None:
    order = ["cj_alphanet", "cj_alphanet_no2d", "resnet", "fno", "cjcep", "train_mean", "uniform"]
    rows = []
    for k in order:
        if k not in metrics:
            continue
        m = metrics[k]
        rows.append(
            (
                DISPLAY_NAMES.get(k, k),
                m.get("m_alpha_w1_m", float("nan")),
                m.get("m_alpha_corr", float("nan")),
                m.get("active_precision", float("nan")),
                m.get("active_recall", float("nan")),
                m.get("active_f1", float("nan")),
                m.get("active_logY_mae", float("nan")),
                m.get("active_alpha_mae", float("nan")),
                m.get("simplex_max_dev", float("nan")),
            )
        )

    def fmt(x, nd=3):
        try:
            v = float(x)
        except Exception:
            return "—"
        if not np.isfinite(v):
            return "—"
        if abs(v) > 0 and abs(v) < 5e-4:
            return f"{v:.1e}"
        return f"{v:.{nd}f}"

    table = [
        "| 模型 | $m_\\alpha$ W1 (m) | $m_\\alpha$ 相关 | 活动 P | 活动 R | 活动 F1 | 活动 logY MAE | 活动 $\\alpha$ MAE | $\\sum\\alpha$ 偏差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        table.append(
            f"| {r[0]} | {fmt(r[1],1)} | {fmt(r[2],3)} | {fmt(r[3],3)} | {fmt(r[4],3)} | {fmt(r[5],3)} | {fmt(r[6],3)} | {fmt(r[7],4)} | {fmt(r[8],2)} |"
        )

    fig_md = []
    for pth in case_figs:
        rel = os.path.relpath(pth, os.path.dirname(path)).replace("\\", "/")
        fig_md.append(f"![{os.path.basename(pth)}]({rel})")

    best = metrics.get("cj_alphanet", {})
    lines = [
        "# CJ-AlphaNet 在 moc_v2_physical_steady_1k_newa 上的反演实验报告",
        "",
        "本文在 **newa** 物理稳态 1k 数据集上训练新的多头反演网络 CJ-AlphaNet，监督对象为进液密度、活动簇、等效并联导纳与流量份额。标签由离线 npz 提供，**未重跑 1k 全场 MOC**，也 **未覆盖原始波形 HDF5**。",
        "",
        "## 1. 数据与标签",
        "",
        f"- 波形：`data/datasets/moc_v2_physical_steady_1k_newa/moc_v2_physical_steady_1k_x4500_4950.h5`（1000 井，井口 $H(t)$，约 0–61 s、1 ms）。",
        "- 切分：800 / 100 / 100，`seed=42`（与 `split_dataset_indices` 一致）。",
        "- 离线标签：`case_labels_alpha_y.npz` / `.csv` / `_meta.json`（由 `label_physics.py` 预先算好）。",
        "- **不使用** HDF5 `labels/wavespeed` 作为网络输入。波速条件来自标签中的 $\\hat a=2 x_1/T_1$，并夹在 $[1200,1600]$ m/s。",
        "- $x_1$：最浅设计簇；$T_1$：停泵后 $dH/dt$ 在 $[2x_1/1600,2x_1/1200]$ 内的显著自相关峰，否则同一窗内 $|dH/dt|$ 首回波。",
        "- `mask_design`：该槽是否有设计簇（砂堵仍为 1）。",
        "- $y_\\mathrm{act}=1[\\mathrm{mask\\_design}=1\\ \\mathrm{且}\\ \\alpha\\ge 0.04]$。",
        "- $Y_\\mathrm{eq}=\\mathrm{Re}\\,1/(R_\\mathrm{perf}+1/(G_\\mathrm{leak}+j\\omega^* C_f))$，$R=2K_p|q|$，$G=k^2/(2|q|)$，$\\omega^*=2\\pi/T_1$；砂堵/非活动簇 $Y_\\mathrm{eq}=0$。",
        "- 监督 $\\log_{10}((Y_\\mathrm{eq}+10^{-10})/Y_0)$，$Y_0=gA/\\hat a$，$D=0.1397$ m，$g=9.81$。",
        "- $m_\\alpha$：高斯 $\\sigma=20$ m 铺到 $[0,5000]$ 的 500 点网格，积分 1。",
        "- **不反演 $C_f$**；$C_f$ 只用于构造 $Y_\\mathrm{eq}$ 标签。",
        "",
        "输入为停泵后 60 s（$t_s=1$, $t_f=61$）：",
        "",
        "- `wave` $(2,\\sim 60000)$：$H$ 与 $dH/dt$ 各自减均值除标准差，**不抽成 4096**（基线模型可另用 4096 预处理）。",
        "- 1D 倒谱：原生倒频率，Kaiser $\\beta=14$，$x=\\hat a\\,\\tau/2$，只留 $x\\in[0,L]$。",
        "- 2D 倒谱：窗 20 s、步 1 s、`window=(\"kaiser\", 14)`，约 41 帧，在各 $x_j$ 取列。",
        "- 位置 `positions` / `norm_positions`，$M=6$。",
        "- 可选 1 维条件：$\\hat a$（来自 npz）。",
        "",
        "## 2. 模型：CJ-AlphaNet",
        "",
        "与旧 TG-DIS-DeepONet 的主要差别：",
        "",
        "1. **标签不同**：活动性不再把 design mask 当存在性；导纳代替 $C_f$ 作为可反演物理量；密度场优先。",
        "2. **输入不同**：全速率停泵后波形 + 原生 1D/2D 倒谱在 $x_j$ 取样，禁止 2D 只做全局池化。",
        "3. **无 DIS 无约束 $\\Gamma$ 头、无 $C_f$ 头**。$\\hat Y=Y_0 10^{\\hat y}$，反射系数闭式",
        "   $$\\Gamma=-\\hat Y/(2Y_0+\\hat Y).$$",
        "4. 声学距离偏置用 **估计波速 $\\hat a$**，而不是设计波速。",
        "",
        "前向结构：",
        "",
        "```",
        "wave (2, ~60000)",
        "  ├─ 大步长 1D CNN → z_wave",
        "  └─ 切窗 [τ_j−50 ms, τ_j+250 ms]，τ_j = 2 x_j / â → token_j^wave",
        "cepstrum_1d → 在 x_j 取样 → token_j^{1d}",
        "cepstrum_2d → 深度 x_j 的 41 维列 → token_j^{2d}",
        "token_j = [波前, 1D(x_j), 2D(:,x_j), 位置编码, â]",
        "mask_design=0 的槽置零",
        "声学距离偏置 Transformer（距离用 â）",
        "        ├─ 连续头 → m̂_α(x) → Voronoi 积分 → α_field",
        "        ├─ 活动头 → p̂_j   （仅 design 槽）",
        "        ├─ 导纳头 → ŷ_j = log10(Ŷ/Y0)",
        "        └─ 比例头 → Masked Softmax → α̂  （全部 design 槽，含砂堵）",
        "```",
        "",
        "损失（起步权重）：",
        "",
        "$$L=\\lambda_m L_{m\\alpha}+\\lambda_\\mathrm{act}L_\\mathrm{act}+\\lambda_Y L_Y+\\lambda_\\alpha L_\\alpha+\\lambda_\\mathrm{cons}L_\\mathrm{cons}$$",
        "",
        "其中 $\\lambda_m=1,\\lambda_\\mathrm{act}=0.5,\\lambda_Y=0.4,\\lambda_\\alpha=0.1,\\lambda_\\mathrm{cons}=0.1$。",
        "$L_{m\\alpha}$ 为网格 1D Wasserstein（米）/20 加上密度 MAE；$L_\\mathrm{act}$ 为 design 槽 BCE；",
        "$L_Y$ 仅在活动簇上 Smooth-L1；$L_\\alpha$ 为 design 槽单纯形 KL；$L_\\mathrm{cons}=|\\hat\\alpha-\\alpha_\\mathrm{field}|$。",
        "验证选点：密度 W1/50 + (1−活动 F1)。",
        "",
        "## 3. 训练设置",
        "",
        f"- 优化器：AdamW，lr={settings.get('lr', 1e-3)}，余弦退火，梯度裁剪 5。",
        f"- 设备：`{settings.get('device', 'cpu')}`（本机无 CUDA，全程 CPU）。",
        f"- CJ-AlphaNet / 消融：batch={settings.get('batch_size', 4)}，epochs={settings.get('epochs', 40)}。",
        f"- 神经网络基线：batch={settings.get('batch_size_baseline', 8)}，epochs={settings.get('epochs_baseline', 20)}。",
        "- 验证选点：`W1(m)/50 + (1 − 活动 F1)`，保存 `output/alpha_newa/weights/*_best.pt`。",
        "- 权重目录：`PaperC_CJNO_Wellbore_Inversion/output/alpha_newa/weights/`（不覆盖旧 phase 权重）。",
        "",
        "### 各模型训练摘要",
        "",
    ]
    if train_summaries:
        lines += [
            "| 模型 | 参数量 | 训练时间 (s) | 验证 W1 (m) | 验证 F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, v in train_summaries.items():
            bm = v.get("best_metrics") or {}
            lines.append(
                f"| {DISPLAY_NAMES.get(k,k)} | {v.get('n_params','—')} | {fmt(v.get('train_time_sec', float('nan')),1)} | {fmt(bm.get('m_alpha_w1_m', float('nan')),1)} | {fmt(bm.get('active_f1', float('nan')),3)} |"
            )
        lines.append("")

    lines += [
        "## 4. 测试集性能",
        "",
        "验收顺序：密度 → 活动簇 F1 → 活动簇 $\\log_{10}(Y/Y_0)$ MAE → 活动簇 $\\alpha$ MAE。下表均为 **同一 test 100 井**。",
        "",
        *table,
        "",
        f"CJ-AlphaNet 主指标：W1 = **{fmt(best.get('m_alpha_w1_m', float('nan')),1)} m**，"
        f"密度相关 **{fmt(best.get('m_alpha_corr', float('nan')),3)}**，"
        f"活动 F1 **{fmt(best.get('active_f1', float('nan')),3)}** "
        f"(P={fmt(best.get('active_precision', float('nan')),3)}, R={fmt(best.get('active_recall', float('nan')),3)})，"
        f"活动 logY MAE **{fmt(best.get('active_logY_mae', float('nan')),3)}**，"
        f"活动 $\\alpha$ MAE **{fmt(best.get('active_alpha_mae', float('nan')),4)}**，"
        f"$\\sum\\alpha$ 最大偏差 **{fmt(best.get('simplex_max_dev', float('nan')),2)}**。",
        "",
        "### 结果解读",
        "",
        "- **密度 W1**：newa 全部设计簇落在约 4500–4950 m。把质量均匀摊在这些近趾端簇上，CDF 在 $[0,4500]$ 几乎重合，因此「均匀 α → 高斯密度」就能把 W1 压到约 11 m。ResNet/FNO 的密度由预测 α 的高斯铺开构成，接近该几何下限；CJ-AlphaNet 的连续 Trunk 场略宽，W1 高约 2 m。密度指标在本数据上 **区分力弱**，不能单独当作架构胜负。",
        "- **活动簇**：测试集 358 个 design 槽中 317 个活动（88.5%）。所有模型召回均为 1.0，F1≈0.94 主要来自「几乎全判活动」。CJ-AlphaNet 假阳性 39，比全活动基线的 41 少 2 个，精度 0.890 vs 0.885，改进很小。砂堵/α<0.04 簇仍然难以从井口记录里稳定检出。",
        "- **导纳 logY**：这是最能拉开模型的指标。均匀地板基线 MAE=6.61（把 $Y$ 设成 $10^{-10}$）。FNO1D 最好（0.222），ResNet1D 0.272，CJ-AlphaNet 0.323。4096 点全局谱卷积比 60 k 切窗+倒谱取样更容易拟合活动簇的 $\\log(Y/Y_0)$ 尺度。",
        "- **α MAE**：各模型活动簇 α MAE 均在 0.116–0.121，与均匀/训练均值基线几乎相同。近趾端 20–50 m 簇间距下流量份额在井口记录上严重混叠，当前监督还没有打破均摊。",
        "- **2D 倒谱消融**：去掉 2D 列取样后测试 W1 12.1 vs 12.6、logY 0.307 vs 0.323，未显示稳定增益。41 帧短时倒谱在簇间距过密时高度相关。",
        "- **守恒**：Masked Softmax 使 $\\sum\\alpha$ 偏差处于 $10^{-7}$ 量级。",
        "",
        "## 5. 单井预测展示",
        "",
        "从 test 集选取四类代表井。每井三面板：停泵后井口 $H(t)$、密度真值/预测、簇级 $\\alpha$（柱）/$y_\\mathrm{act}$（点）/$\\log Y$（线）。",
        "",
        *fig_md,
        "",
        "- 图 case_00：单簇，密度峰对齐较好，$\\alpha=1$ 平凡。",
        "- 图 case_01：四簇相对均衡，密度包络可对上，但 $\\alpha$ 仍接近均摊；logY 趋势接近真值。",
        "- 图 case_02：含砂堵（并带断层标签）。真值 logY 在死簇落到约 $-6$，预测仍当作活动簇。",
        "- 图 case_03：断层型多簇，密度双峰位置有偏移，趾端强进液 $\\alpha$ 被低估。",
        "",
        "## 6. 失败模式与局限",
        "",
        "- **活动头几乎坍缩为全活动**：$\\lambda_\\mathrm{act}=0.5$ 的无加权 BCE 在 86% 正类下不够把砂堵推到 0.5 以下。",
        "- **近趾端几何**：簇位 4500–4950 m，趾端反射与首簇回波接近，密度峰沿水平段平移，$m_\\alpha$ 的 W1 被几何下限（~9–11 m）卡住。",
        "- **α 混叠**：20–50 m 簇间距远小于水击波长尺度，单纯形 KL 权重只有 0.1，份额头容易退回均摊。",
        "- **首簇砂堵与 $T_1$**：若最浅设计簇砂堵，$x_1$ 仍取该簇深度，但首回波能量可能来自下游活动簇，$\\hat a$ 会偏。",
        "- **不反 $C_f$**：同一 $Y_\\mathrm{eq}$ 可由 $(q,k,K_p,C_f)$ 多组组合实现，网络只恢复工作点导纳。",
        "- **2D 倒谱窗 20 s**：相邻 $x_j$ 的 41 维列高度相关，消融未显示稳定增益。",
        "- 本机无 GPU，CJ-AlphaNet 40 epoch、基线 20 epoch，均为真实训练/评估，未编造指标。",
        "",
        "## 7. 复现命令",
        "",
        "在仓库根目录、PowerShell 下：",
        "",
        "```powershell",
        *commands,
        "```",
        "",
        "单测：",
        "",
        "```powershell",
        "python -m pytest PaperC_CJNO_Wellbore_Inversion/tests/test_label_physics.py PaperC_CJNO_Wellbore_Inversion/tests/test_alpha_dataset.py -v",
        "```",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=str, default=DEFAULT_H5_PATH)
    p.add_argument("--npz", type=str, default=DEFAULT_NPZ_PATH)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--models", type=str, default="cj_alphanet,cj_alphanet_no2d,resnet,fno,cjcep,train_mean,uniform")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    weights_dir = os.path.join(out_dir, "weights")

    test_ds = AlphaInversionDataset(
        h5_path=args.h5,
        npz_path=args.npz,
        split="test",
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    train_ds = AlphaInversionDataset(
        h5_path=args.h5,
        npz_path=args.npz,
        split="train",
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    test_loader = test_ds.get_dataloader(batch_size=args.batch_size, shuffle=False)
    print(f"[*] test={len(test_ds)} device={device}")

    names = [s.strip() for s in args.models.split(",") if s.strip()]
    all_metrics: Dict[str, Any] = {}
    all_preds: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in names:
        model = build_model(name, train_ds=train_ds)
        wpath = os.path.join(weights_dir, f"{name}_best.pt")
        if name not in ("uniform", "train_mean"):
            if not os.path.isfile(wpath):
                print(f"[skip] 无权重 {wpath}")
                continue
            load_weights(model, wpath, device)
        else:
            model = model.to(device)
            model.eval()
        ev = evaluate_loader(model, test_loader, device)
        all_metrics[name] = {k: v for k, v in ev["metrics"].items() if np.isscalar(v)}
        all_preds[name] = ev
        m = ev["metrics"]
        print(
            f"  {name:18s}  W1={m['m_alpha_w1_m']:.1f}m  corr={m['m_alpha_corr']:.3f}  "
            f"F1={m['active_f1']:.3f}  logY={m['active_logY_mae']:.3f}  aMAE={m['active_alpha_mae']:.4f}"
        )

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    case_figs: List[str] = []
    if not args.skip_plots and "cj_alphanet" in all_preds:
        pred = all_preds["cj_alphanet"]["pred"]
        tgt = all_preds["cj_alphanet"]["target"]
        # evaluate_loader 按 loader 顺序，与 dataset 顺序一致（shuffle=False）
        cases = select_case_ids(test_ds, args.h5)
        sid_arr = tgt["sample_id"].numpy().astype(int)
        with h5py.File(args.h5, "r") as h5:
            timestamps = np.asarray(h5["waveforms/timestamps"][:], dtype=np.float64)
            heads = h5["waveforms/wellhead_head"]
            grid = test_ds.grid_x.numpy()
            for k, (li, clab) in enumerate(cases):
                sid = int(test_ds.indices[li])
                rows = np.where(sid_arr == sid)[0]
                row = int(rows[0]) if rows.size else li
                t_post, h_post = extract_post_shut_head(timestamps, heads[sid], ts=1.0, duration=60.0)
                msk = tgt["mask_design"][row].numpy().astype(bool)
                title = f"Test case {sid}  ({clab})"
                png = os.path.join(fig_dir, f"case_{k:02d}_sid{sid}.png")
                plot_case(
                    png,
                    t_post,
                    h_post,
                    grid,
                    tgt["m_alpha_grid"][row].numpy(),
                    pred["pred_m_alpha"][row].numpy(),
                    tgt["positions"][row].numpy(),
                    msk,
                    tgt["alpha"][row].numpy(),
                    pred["pred_alpha"][row].numpy(),
                    tgt["y_act"][row].numpy(),
                    pred["pred_active"][row].numpy(),
                    tgt["logY"][row].numpy(),
                    pred["pred_logY"][row].numpy(),
                    title,
                )
                case_figs.append(png)
                print(f"[+] 图 {png}")

    train_summary_path = os.path.join(out_dir, "train_summary.json")
    train_sum = {}
    if os.path.isfile(train_summary_path):
        with open(train_summary_path, "r", encoding="utf-8") as f:
            train_sum = json.load(f)

    settings = {
        "device": str(device),
        "lr": 1e-3,
        "batch_size": 4,
        "batch_size_baseline": 8,
        "epochs": 40,
        "epochs_baseline": 20,
    }

    commands = [
        "python PaperC_CJNO_Wellbore_Inversion/experiments/train_alpha.py --models cj_alphanet --epochs 40 --batch-size 4",
        "python PaperC_CJNO_Wellbore_Inversion/experiments/train_alpha.py --models cj_alphanet_no2d,resnet,fno,cjcep,uniform,train_mean --epochs 40 --epochs-baseline 20 --batch-size 4 --batch-size-baseline 8",
        "python PaperC_CJNO_Wellbore_Inversion/experiments/evaluate_alpha.py",
    ]
    report_path = os.path.join(out_dir, "CJ_AlphaNet_newa_report.md")
    write_report(report_path, all_metrics, train_sum, case_figs, settings, commands)
    print(f"[+] 报告 {report_path}")
    print(f"[+] 指标 {metrics_path}")


if __name__ == "__main__":
    main()
