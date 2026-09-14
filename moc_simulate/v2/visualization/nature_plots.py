# -*- coding: utf-8 -*-
"""
moc_simulate.v2.visualization.nature_plots

Nature / High-Impact 期刊标准科研制图模块：
波形时程演化、2D 倒谱照亮、参数敏感性标准出图工具。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


def setup_nature_style():
    """配置符合顶级期刊排版规范的 matplotlib 样式"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True


def plot_waveform_evolution(
    timestamps: np.ndarray,
    wellhead_head: np.ndarray,
    toe_head: Optional[np.ndarray] = None,
    fracture_heads: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Wellhead Transient Pressure Waveform",
) -> plt.Figure:
    """绘制全井水头瞬态时程演化图版"""
    setup_nature_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    ax.plot(timestamps, wellhead_head, label="Wellhead Head", color="#1f77b4", lw=1.2)
    if toe_head is not None:
        ax.plot(timestamps, toe_head, label="Toe Head", color="#7f7f7f", lw=0.9, ls="--")
    if fracture_heads is not None and fracture_heads.ndim == 2 and fracture_heads.shape[1] > 0:
        colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for k in range(min(4, fracture_heads.shape[1])):
            ax.plot(
                timestamps,
                fracture_heads[:, k],
                label=f"Fracture {k+1}",
                color=colors[k % len(colors)],
                lw=1.0,
                alpha=0.8,
            )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Pressure Head (m)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", ls=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cepstrogram_heatmap(
    time_centers: np.ndarray,
    distances: np.ndarray,
    cepstrogram: np.ndarray,
    fracture_positions: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    title: str = "2D Cepstrogram Spatial-Temporal Illumination",
) -> plt.Figure:
    """绘制 2D 连续时空倒谱热图"""
    setup_nature_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    t_grid, d_grid = np.meshgrid(time_centers, distances)
    im = ax.pcolormesh(
        t_grid,
        d_grid,
        cepstrogram.T,
        shading="auto",
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cepstral Amplitude", fontsize=10)

    if fracture_positions:
        for k, xf in enumerate(fracture_positions):
            ax.axhline(xf, color="red", ls="--", lw=1.0, alpha=0.7, label=f"Frac {k+1} ({xf:.0f}m)" if k == 0 else "")

    ax.set_xlabel("Window Center Time (s)", fontsize=11)
    ax.set_ylabel("Reflective Depth (m)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
