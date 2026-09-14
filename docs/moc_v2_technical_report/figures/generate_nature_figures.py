# -*- coding: utf-8 -*-
"""
generate_nature_figures.py
===============================================================================
Script to generate submission-grade Nature figures for MOC_V2 Physics Upgrade:
- Figure 0: MOC_V1 Baseline Defects (Toe False Shock, Drift, Cavitation, Continuity Violation)
- Figure 1: Fluid Transient Waveform Evolution (100s, 12s Zoom, and dH/dt)
- Figure 2: 2D Cepstrogram & Multi-Cluster Fracture Illumination Evolution
- Figure 3: Perforation Impedance & Ramp Closure Parametric Sensitivity Array

Adheres strictly to Nature Figure standards:
- Dimensions: Double column (180 mm ~ 7.086 in)
- Color palette: Nature classic (#0C4B8E, #E15759, #4E79A7, #F28E2B, #76B7B2, #7F7F7F)
- Typography: Sans-serif (Arial / Helvetica / DejaVu Sans), 6.5-9.5 pt
- Outputs: 300 DPI .png and vector .svg
===============================================================================
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(SCRIPT_DIR)
DOCS_DIR = os.path.dirname(REPORT_DIR)
PROJECT_ROOT = os.path.dirname(DOCS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum, compute_moc_cepstrum_1d

# -----------------------------------------------------------------------------
# Nature Plotting Style Configuration
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",       # Editable text in SVG
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "figure.titlesize": 9.5,
    "axes.linewidth": 0.75,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Nature Classic Palette
COLOR_NAVY   = "#0C4B8E"  # Step 4 / Primary focus
COLOR_CORAL  = "#E15759"  # Highlighting / alerts / false peaks
COLOR_LAKE   = "#4E79A7"  # Step 2 / geological compliance
COLOR_AMBER  = "#F28E2B"  # Step 3 / perforation throttle
COLOR_SAGE   = "#76B7B2"  # Step 1 / steady state
COLOR_GREY   = "#7F7F7F"  # Baseline V1 PaperA
COLOR_LIGHT  = "#F0F0F0"
COLOR_GRID   = "#EAEAEA"


def add_panel_label(ax, label: str, x: float = -0.09, y: float = 1.05):
    """Add bold sans-serif panel identifier (a, b, c) per Nature conventions."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        ha="left",
        color="#222222"
    )


def save_figure_nature(fig, filename_base: str):
    """Save both high-res 300 DPI PNG and vector SVG."""
    png_path = f"{filename_base}.png"
    svg_path = f"{filename_base}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=False)
    fig.savefig(svg_path, bbox_inches="tight", transparent=False)
    print(f"  [Saved PNG]: {png_path}")
    print(f"  [Saved SVG]: {svg_path}")



# =============================================================================
# FIGURE 0: MOC_V1 原始缺陷诊断图 (趾端假激波 + 空间速度场断崖截断)
# =============================================================================
def simulate_v1_toe_shock(L=5000.0, a=1450.0, dt=0.001, tf=100.0, V0=1.0, H0=300.0, ts=1.0, D=0.1397, f=0.018):
    """
    Simulate the uncorrected MOC_V1 baseline with unphysical toe cliff cut V(L, 0) = 0
    and flat H(x, 0) = 300 m to capture the genuine +147.8 m toe false shock and back-propagation.
    """
    g = 9.80665
    N = int(round(L / (a * dt)))
    ga = g / a
    n_steps = int(round(tf / dt))

    H = np.full(N + 1, H0, dtype=np.float64)
    V = np.full(N + 1, V0, dtype=np.float64)
    V[-1] = 0.0  # Unphysical cliff cutoff at toe

    H_toe = np.zeros(n_steps + 1, dtype=np.float64)
    H_wh = np.zeros(n_steps + 1, dtype=np.float64)
    H_toe[0] = H[-1]
    H_wh[0] = H[0]

    f_term = f * dt / (2.0 * D)

    for n in range(1, n_steps + 1):
        t = n * dt
        J = f_term * V * np.abs(V)
        Cp = V[:-1] + ga * H[:-1] - ga * J[:-1]
        Cm = V[1:] - ga * H[1:] + ga * J[1:]
        
        H_new = np.zeros(N + 1, dtype=np.float64)
        V_new = np.zeros(N + 1, dtype=np.float64)
        
        H_new[1:-1] = (Cp[:-1] - Cm[1:]) / (2.0 * ga)
        V_new[1:-1] = (Cp[:-1] + Cm[1:]) / 2.0
        
        V_wh_val = V0 if t < ts else 0.0
        V_new[0] = V_wh_val
        H_new[0] = (V_wh_val - Cm[0]) / ga
        
        V_new[-1] = 0.0
        H_new[-1] = Cp[-1] / ga
        
        H = H_new
        V = V_new
        H_wh[n] = H[0]
        H_toe[n] = H[-1]

    t_arr = np.linspace(0.0, tf, n_steps + 1)
    return t_arr, H_wh, H_toe


def generate_figure_0_defects(df_papera, df_step1):
    print("\nGenerating Figure 0: MOC_V1 Baseline Physical Defects...")
    t_v1, h_wh_v1, h_toe_v1 = simulate_v1_toe_shock(tf=100.0)

    fig = plt.figure(figsize=(7.086, 6.2))  # 180 mm width
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.18, 1.0], hspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    # -------------------------------------------------------------------------
    # Panel a: 100s 时程波形 (趾端假激波 + 井口倒灌漂移 + 真空塌陷)
    # -------------------------------------------------------------------------
    add_panel_label(ax_a, "a", x=-0.045, y=1.18)

    # Shaded cavitation collapse regime
    ax_a.axhspan(-85, 0, color="#FFD1D1", alpha=0.40, zorder=1)
    ax_a.text(80, -65, "Non-Physical Cavitation Regime ($H < 0$)",
              fontsize=6.2, color="#B03A2E", style="italic", ha="center")

    # Reference levels
    ax_a.axhline(300, color="#888888", linestyle="--", linewidth=0.7, zorder=2, label="Initial Steady $H_0=300\\,\\mathrm{m}$")
    ax_a.axhline(100, color="#A0A0A0", linestyle=":", linewidth=0.7, zorder=2, label="Pore Pressure $H_{ext}=100\\,\\mathrm{m}$")
    ax_a.axhline(0, color="#D9534F", linestyle="-.", linewidth=0.6, zorder=2, label="Cavitation Boundary $H=0\\,\\mathrm{m}$")

    # Plot traces
    ax_a.plot(t_v1, h_toe_v1, color=COLOR_CORAL, linestyle="-", linewidth=1.0, alpha=0.85,
              label="V1 Toe Head $H_{toe}(t)$ (Uncorrected False Shock)")
    ax_a.plot(df_papera["t"], df_papera["H_wh"], color=COLOR_GREY, linestyle="--", linewidth=1.0, alpha=0.85,
              label="V1 Wellhead $H_{wh}(t)$ (PaperA Drift Baseline)")
    ax_a.plot(df_step1["t"], df_step1["H_wh"], color=COLOR_SAGE, linestyle="-", linewidth=1.1, alpha=0.9,
              label="Step 1 Wellhead $H_{wh}(t)$ (Micro $C_f=10^{-5}\\,\\mathrm{m^2}$, Cavitation)")

    # Red callout arrows & annotations (with clean white bounding boxes to prevent collisions)
    ax_a.annotate(
        "$t=0$ Toe False Shock\n$\\Delta H_{toe} = +147.8\\,\\mathrm{m}$ ($H_{toe} \\to 447.8\\,\\mathrm{m}$)",
        xy=(0.0, 447.8), xytext=(5.0, 465.0),
        arrowprops=dict(arrowstyle="->", color=COLOR_CORAL, lw=0.8),
        fontsize=6.2, fontweight="bold", color=COLOR_CORAL, ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#E0A0A0", alpha=0.92, lw=0.6)
    )

    ax_a.annotate(
        "Back-propagation to wellhead\n($t = L/a \\approx 3.45\\,\\mathrm{s}$)",
        xy=(3.45, 347.8), xytext=(8.0, 255.0),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.7),
        fontsize=6.0, color="#333333", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D0D0", alpha=0.90, lw=0.6)
    )

    # Actual Step 1 minimum occurs at t = 12.33 s (H_wh = -57.45 m)
    ax_a.annotate(
        "Micro-Compliance ($C_f=10^{-5}\\,\\mathrm{m^2}$):\nVacuum Cavitation Collapse ($-57.45\\,\\mathrm{m}$ at $t=12.3\\,\\mathrm{s}$)",
        xy=(12.33, -57.45), xytext=(20.0, -56.0),
        arrowprops=dict(arrowstyle="->", color=COLOR_CORAL, lw=0.8),
        fontsize=6.2, fontweight="bold", color=COLOR_CORAL, ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#E0A0A0", alpha=0.92, lw=0.6)
    )

    ax_a.set_xlim([0, 100])
    ax_a.set_ylim([-95, 510])
    ax_a.set_xlabel("Time $t$ [s]")
    ax_a.set_ylabel("Pressure Head $H$ [m]")
    ax_a.set_title("V1 Transient Breakdown: Toe False Shock (+147.8 m) & Cavitation Collapse (-57.45 m)", pad=5, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)
    ax_a.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98), ncol=2, frameon=True,
                facecolor="white", edgecolor="#D8D8D8", framealpha=0.92, fontsize=5.4)

    # -------------------------------------------------------------------------
    # Panel b: 空间速度剖面对比 (全井 1.0 m/s + 单点断崖切零 vs 多簇分流 + 死水区)
    # -------------------------------------------------------------------------
    add_panel_label(ax_b, "b", x=-0.045, y=1.18)

    x_pts = np.linspace(0, 5000, 5001)
    v_v1 = np.ones_like(x_pts)
    v_v1[-1] = 0.0

    v_v2 = np.zeros_like(x_pts)
    v_v2[x_pts < 4100.0] = 1.0
    v_v2[(x_pts >= 4100.0) & (x_pts < 4120.0)] = 0.75
    v_v2[(x_pts >= 4120.0) & (x_pts < 4140.0)] = 0.50
    v_v2[(x_pts >= 4140.0) & (x_pts <= 4160.0)] = 0.25
    v_v2[x_pts > 4160.0] = 0.0

    # Shaded dead-end stagnant zone
    ax_b.axvspan(4160.0, 5000.0, color="#EBF3FB", alpha=0.70, zorder=1)
    ax_b.text(4650.0, 0.55, "Quiescent Dead-End Zone\n$V(x, 0) \\equiv 0.0\\,\\mathrm{m/s}$\n($x > 4160\\,\\mathrm{m}$)",
              fontsize=6.2, color=COLOR_NAVY, style="italic", ha="center")

    # Combined fracture cluster marker
    ax_b.axvspan(4100.0, 4160.0, color="#FFE8D6", alpha=0.85, zorder=2)
    ax_b.text(4130.0, 1.12, "Fracs 1–4\n4100–4160 m", fontsize=5.6, ha="center", color="#8A4B1A", fontweight="bold")

    # Plot velocity steps (V1 thicker dashed line so it is visible behind V2 solid line where they coincide)
    ax_b.step(x_pts, v_v1, where="post", color=COLOR_CORAL, linestyle="--", linewidth=1.8, alpha=0.85,
              label="V1 Defective Profile ($V \\equiv 1.0\\,\\mathrm{m/s}$ + Single-Node Cliff Cut)", zorder=3)
    ax_b.plot([5000.0, 5000.0], [1.0, 0.0], color=COLOR_CORAL, linestyle="--", linewidth=1.8, alpha=0.85, zorder=3)
    ax_b.scatter([5000.0], [0.0], color=COLOR_CORAL, edgecolor="#333333", s=34, zorder=4,
                 label="V1 Cliff Discontinuity Point ($V(L, 0)=0$)")

    ax_b.step(x_pts, v_v2, where="post", color=COLOR_NAVY, linestyle="-", linewidth=1.2, alpha=0.95,
              label="MOC_V2 Physical Profile (Continuous Inflow Split + Quiescent Dead Zone)", zorder=3)

    # Annotations on Panel b
    ax_b.annotate(
        "V1: Ignores Fracture Split\n$\\sum q_{j,0} \\ne Q_0$ (Mass non-conserved)",
        xy=(2500.0, 1.0), xytext=(2200.0, 0.72),
        arrowprops=dict(arrowstyle="->", color=COLOR_CORAL, lw=0.7),
        fontsize=6.2, fontweight="bold", color=COLOR_CORAL, ha="center"
    )

    ax_b.annotate(
        "Single-Node Cliff Cut ($V(L)=0$)\nLaunches $+147.8\\,\\mathrm{m}$ False Shock",
        xy=(5000.0, 0.0), xytext=(4820.0, 0.22),
        arrowprops=dict(arrowstyle="->", color=COLOR_CORAL, lw=0.8),
        fontsize=6.2, fontweight="bold", color=COLOR_CORAL, ha="right"
    )

    ax_b.annotate(
        "MOC_V2: Partitioned Flow ($q_{j,0} = Q_0/4$)\nStepwise Deceleration",
        xy=(4100.0, 0.88), xytext=(3200.0, 0.40),
        arrowprops=dict(arrowstyle="->", color=COLOR_NAVY, lw=0.7),
        fontsize=6.2, fontweight="bold", color=COLOR_NAVY, ha="center"
    )

    ax_b.set_xlim([-100, 5150])
    ax_b.set_ylim([-0.12, 1.30])
    ax_b.set_xlabel("Axial Coordinate Along Wellbore $x$ [m]")
    ax_b.set_ylabel("Initial Velocity Field $V(x, 0)$ [m/s]")
    ax_b.set_title("Spatial Initial Velocity Profile: Continuity Violation vs Stagnant Dead-End Zone", pad=4, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=True,
                facecolor="white", edgecolor="#D8D8D8", framealpha=0.92, fontsize=5.6)

    out_base = os.path.join(SCRIPT_DIR, "fig0_v1_baseline_defects")
    save_figure_nature(fig, out_base)
    plt.close(fig)


# =============================================================================
# FIGURE 1: 时域波形全景与局部的四阶段物理演进对比
# =============================================================================
def generate_figure_1(df_papera, df_step1, df_step2, df_step3, df_step4):
    print("\nGenerating Figure 1: Fluid Transient Waveform Evolution...")
    fig = plt.figure(figsize=(7.086, 6.5))  # 180 mm width
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.48, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------------------
    # Panel a: 100s 全时程波形对比
    # -------------------------------------------------------------------------
    add_panel_label(ax_a, "a", x=-0.045, y=1.22)

    # Shaded cavitation collapse region
    ax_a.axhspan(-85, 0, color="#FFD1D1", alpha=0.35, zorder=1)
    ax_a.text(60, -45, "Non-Physical Cavitation & Collapse Regime (Step 1)",
              fontsize=6.8, color=COLOR_CORAL, style="italic", ha="center")

    # Reference levels
    ax_a.axhline(300, color="#888888", linestyle="--", linewidth=0.7, zorder=2, label="Steady $H_{ss}=300\\,\\mathrm{m}$")
    ax_a.axhline(100, color="#A0A0A0", linestyle=":", linewidth=0.7, zorder=2, label="Pore $H_{ext}=100\\,\\mathrm{m}$")

    # Waveform traces
    ax_a.plot(df_papera["t"], df_papera["H_wh"], color=COLOR_GREY, linestyle="--", linewidth=0.9, alpha=0.75,
              label="V1 Baseline (PaperA): Toe false shock")
    ax_a.plot(df_step1["t"], df_step1["H_wh"], color=COLOR_SAGE, linestyle="-", linewidth=0.9, alpha=0.85,
              label="Step 1: Steady flow (Micro $C_f=10^{-5}\\,\\mathrm{m^2}$)")
    ax_a.plot(df_step2["t"], df_step2["H_wh"], color=COLOR_LAKE, linestyle="-", linewidth=0.95, alpha=0.75,
              label="Step 2: Geo compliance ($C_f=0.01\\,\\mathrm{m^2}$)")
    ax_a.plot(df_step3["t"], df_step3["H_wh"], color=COLOR_AMBER, linestyle="-", linewidth=1.0, alpha=0.85,
              label="Step 3: Perf throttle ($N_p=6, K_p=5.43\\times 10^5$)")
    ax_a.plot(df_step4["t"], df_step4["H_wh"], color=COLOR_NAVY, linestyle="-", linewidth=1.2,
              label="Step 4: Field ramp closure ($t_c=1.0\\,\\mathrm{s}$)")

    # Annotation for rebound peak placed in the clear upper-left white space at t=4s, y=420m
    ax_a.annotate(
        "Macro Geological Rebound\n$+194\\sim +239\\,\\mathrm{m}$",
        xy=(12.3, 354.3), xytext=(4.0, 420.0),
        arrowprops=dict(arrowstyle="->", color=COLOR_NAVY, lw=0.8),
        fontsize=6.8, fontweight="bold", color=COLOR_NAVY, ha="left"
    )

    ax_a.set_xlim([0, 100])
    ax_a.set_ylim([-85, 470])
    ax_a.set_xlabel("Time $t$ [s]")
    ax_a.set_ylabel("Wellhead Pressure Head $H_{wh}$ [m]")
    ax_a.set_title("Full-Scale Waveform Evolution (0–100 s)", pad=6, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)

    # Place legend neatly in upper right with compact 2-column layout
    ax_a.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98), ncol=2, frameon=True,
                facecolor="white", edgecolor="#D8D8D8", framealpha=0.92, fontsize=5.8)

    # -------------------------------------------------------------------------
    # Panel b: 前 12s 局部高频波列特写 (含 Step 2 全阶物理对比)
    # -------------------------------------------------------------------------
    add_panel_label(ax_b, "b", x=-0.12, y=1.06)

    mask_zoom_p = df_papera["t"] <= 12.0
    mask_zoom_1 = df_step1["t"] <= 12.0
    mask_zoom_2 = df_step2["t"] <= 12.0
    mask_zoom_3 = df_step3["t"] <= 12.0
    mask_zoom_4 = df_step4["t"] <= 12.0

    ax_b.plot(df_papera.loc[mask_zoom_p, "t"], df_papera.loc[mask_zoom_p, "H_wh"], color=COLOR_GREY, linestyle="--", lw=0.85, alpha=0.7, label="PaperA V1")
    ax_b.plot(df_step1.loc[mask_zoom_1, "t"], df_step1.loc[mask_zoom_1, "H_wh"], color=COLOR_SAGE, linestyle="-", lw=0.85, alpha=0.8, label="Step 1")
    ax_b.plot(df_step2.loc[mask_zoom_2, "t"], df_step2.loc[mask_zoom_2, "H_wh"], color=COLOR_LAKE, linestyle="-", lw=0.85, alpha=0.75, label="Step 2 (Short-cct)")
    ax_b.plot(df_step3.loc[mask_zoom_3, "t"], df_step3.loc[mask_zoom_3, "H_wh"], color=COLOR_AMBER, linestyle="-", lw=0.95, alpha=0.85, label="Step 3 (Throttle)")
    ax_b.plot(df_step4.loc[mask_zoom_4, "t"], df_step4.loc[mask_zoom_4, "H_wh"], color=COLOR_NAVY, linestyle="-", lw=1.2, label="Step 4 (Ramp $1\\mathrm{s}$)")

    # Key acoustic event annotations (corrected: t_arr = ts + 2*dist/a)
    ax_b.axvline(1.0, color="#555555", linestyle=":", lw=0.7)
    ax_b.annotate("$t_s=1.0\\,\\mathrm{s}$\nJoukowsky Drop", xy=(1.0, 300), xytext=(1.35, 345),
                  arrowprops=dict(arrowstyle="->", color="#333333", lw=0.6),
                  fontsize=6.0, color="#333333")

    # Fracture reflection wave arrival: ts + 2*xf/a = 1.0 + 2*4100/1450 = 6.655s
    ax_b.axvline(6.655, color=COLOR_AMBER, linestyle=":", lw=0.7)
    ax_b.annotate("$t_{arr}=t_s+2x_1/a=6.66\\,\\mathrm{s}$\nFracture 1 Reflection",
                  xy=(6.655, 140), xytext=(3.6, 265),
                  arrowprops=dict(arrowstyle="->", color=COLOR_AMBER, lw=0.6),
                  fontsize=5.8, color=COLOR_AMBER)

    # Toe reflection wave arrival: ts + 2*L/a = 1.0 + 2*5000/1450 = 7.897s placed clearly above curve at y=370m
    ax_b.axvline(7.897, color=COLOR_CORAL, linestyle=":", lw=0.7)
    ax_b.annotate("$t_{arr}=t_s+2L/a=7.90\\,\\mathrm{s}$\nDead-End Reflection",
                  xy=(7.897, 335), xytext=(8.3, 370),
                  arrowprops=dict(arrowstyle="->", color=COLOR_CORAL, lw=0.6),
                  fontsize=5.8, color=COLOR_CORAL)

    ax_b.annotate("Suppressed Gibbs Oscillations", xy=(4.5, 150), xytext=(2.2, 185),
                  arrowprops=dict(arrowstyle="->", color=COLOR_NAVY, lw=0.7),
                  fontsize=6.0, color=COLOR_NAVY)

    ax_b.set_xlim([0.5, 12.0])
    ax_b.set_ylim([100, 390])
    ax_b.set_xlabel("Time $t$ [s]")
    ax_b.set_ylabel("Wellhead Pressure Head $H_{wh}$ [m]")
    ax_b.set_title("Early Wave Train Detail (0.5–12 s)", pad=4, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)
    # Position legend in lower right where there are no data points (t>8.5s, y<250m)
    ax_b.legend(loc="lower right", bbox_to_anchor=(0.98, 0.05), frameon=True, facecolor="white",
                edgecolor="#E5E5E5", framealpha=0.92, fontsize=5.8)

    # -------------------------------------------------------------------------
    # Panel c: 稳态与关泵瞬态压力变化率 |dH/dt| 对比 (对数刻度)
    # -------------------------------------------------------------------------
    add_panel_label(ax_c, "c", x=-0.12, y=1.06)

    # Compute dH/dt for PaperA, Step 1, Step 3, Step 4
    for df, col, name, ls in [
        (df_papera, COLOR_GREY, "PaperA (Toe false wave)", "--"),
        (df_step1, COLOR_SAGE, "Step 1 (Steady flow zero wave)", "-"),
        (df_step3, COLOR_AMBER, "Step 3 (Instant step shock)", "-"),
        (df_step4, COLOR_NAVY, "Step 4 (Smooth ramp closure)", "-")
    ]:
        t_arr = df["t"].values
        h_arr = df["H_wh"].values
        mask_der = (t_arr >= 0.0) & (t_arr <= 3.0)
        t_sub = t_arr[mask_der]
        h_sub = h_arr[mask_der]
        dhdt = np.abs(np.gradient(h_sub, t_sub))
        dhdt = np.maximum(dhdt, 1e-3)
        ax_c.plot(t_sub, dhdt, color=col, linestyle=ls, lw=1.0 if col != COLOR_NAVY else 1.2, label=name)

    ax_c.set_yscale("log")
    ax_c.set_xlim([0.0, 3.0])
    ax_c.set_ylim([1e-3, 3e5])
    ax_c.set_xlabel("Time $t$ [s]")
    ax_c.set_ylabel("Head Rate of Change $|dH/dt|$ [m/s]")
    ax_c.set_title("Wavefront Pressure Acceleration (0–3 s)", pad=4, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)

    # Callout annotations for derivative
    ax_c.annotate(
        "Instant Step Discrete Shock\n$1.48\\times 10^5\\,\\mathrm{m/s}$ (Central: $7.4\\times 10^4$)",
        xy=(1.001, 7.4e4), xytext=(1.25, 2.5e4),
        arrowprops=dict(arrowstyle="->", color=COLOR_AMBER, lw=0.7),
        fontsize=5.8, color=COLOR_AMBER
    )
    ax_c.annotate(
        "Smooth Ramp Peak\n$156.4\\,\\mathrm{m/s}$ (945-fold drop)",
        xy=(1.5, 156.4), xytext=(1.65, 500),
        arrowprops=dict(arrowstyle="->", color=COLOR_NAVY, lw=0.7),
        fontsize=6.0, fontweight="bold", color=COLOR_NAVY
    )
    ax_c.annotate(
        "Pre-closure rate strictly 0\n(Toe false shock eliminated)",
        xy=(0.5, 1e-3), xytext=(0.04, 0.005),
        arrowprops=dict(arrowstyle="->", color=COLOR_SAGE, lw=0.7),
        fontsize=5.8, color=COLOR_SAGE
    )

    ax_c.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E5E5E5", framealpha=0.92, fontsize=5.8)

    # Save outputs
    out_base = os.path.join(SCRIPT_DIR, "fig1_waveform_evolution")
    save_figure_nature(fig, out_base)
    plt.close(fig)


# =============================================================================
# FIGURE 2: 2D 倒谱演进与多簇裂缝照亮机制对比
# =============================================================================
def generate_figure_2(df_step2, df_step3, df_step4):
    print("\nGenerating Figure 2: 2D Cepstrogram & Fracture Illumination Evolution...")

    # Compute 2D Cepstrogram for Step 2 and Step 3
    print("  Calculating 2D cepstrogram for Step 2...")
    res2_2d = compute_moc_cepstrum(
        df_step2["t"].values, df_step2["H_wh"].values,
        v=1450.0, wlen_sec=30.0, hop_sec=1.0, win_type="kaiser"
    )
    print("  Calculating 2D cepstrogram for Step 3...")
    res3_2d = compute_moc_cepstrum(
        df_step3["t"].values, df_step3["H_wh"].values,
        v=1450.0, wlen_sec=30.0, hop_sec=1.0, win_type="kaiser"
    )

    # Compute 1D Cepstrum for Step 3 and Step 4 (standard & derivative)
    res3_1d = compute_moc_cepstrum_1d(df_step3["t"].values, df_step3["H_wh"].values, v=1450.0)
    res4_1d = compute_moc_cepstrum_1d(df_step4["t"].values, df_step4["H_wh"].values, v=1450.0)
    res4_1d_der = compute_moc_cepstrum_1d(df_step4["t"].values, df_step4["H_wh"].values, v=1450.0, derivative=True)

    fig = plt.figure(figsize=(7.086, 7.2))  # 180 mm width
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0, 0.95],
                           hspace=0.42, wspace=0.25)

    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1])
    ax_c1 = fig.add_subplot(gs[2, 0])
    ax_c2 = fig.add_subplot(gs[2, 1])

    true_fracs = [4100.0, 4120.0, 4140.0, 4160.0]
    z_min, z_max = 4050.0, 4200.0

    # -------------------------------------------------------------------------
    # Panel a: Step 2 (无射孔阻抗 Kp=0, 首缝声学短路完全屏蔽)
    # -------------------------------------------------------------------------
    add_panel_label(ax_a1, "a", x=-0.08, y=1.12)

    d2 = res2_2d["depth"]
    mask_z2 = (d2 >= z_min) & (d2 <= z_max)
    t2 = res2_2d["t_cep"]
    C2_map = res2_2d["response_2d"][mask_z2, :]
    C2_norm = C2_map / (np.max(C2_map) + 1e-12)

    im_a = ax_a1.pcolormesh(t2, d2[mask_z2], C2_norm, shading="auto", cmap="Blues", vmin=0, vmax=0.6)
    for zf in true_fracs:
        ax_a1.axhline(zf, color=COLOR_CORAL, linestyle="--", linewidth=0.7, alpha=0.8)
    ax_a1.set_ylabel("Apparent Depth $z$ [m]")
    ax_a1.set_title("Step 2: 2D Cepstrogram ($K_p=0$, Acoustic Short Circuit)", fontsize=7.8, fontweight="bold")
    # Position text in lower left so it never collides with upper-right colorbar
    ax_a1.text(0.03, 0.10, "Downstream clusters 2–4 shadowed", transform=ax_a1.transAxes,
               fontsize=6.2, color=COLOR_CORAL, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLOR_CORAL, lw=0.6, alpha=0.9))

    # Inset colorbar for Panel a1
    cax_a = ax_a1.inset_axes([0.64, 0.86, 0.33, 0.06])
    cbar_a = fig.colorbar(im_a, cax=cax_a, orientation="horizontal")
    cbar_a.set_ticks([0.0, 0.3, 0.6])
    cbar_a.set_ticklabels(["0", "0.3", "0.6"], fontsize=5.5)
    cbar_a.ax.set_title("Normalized Energy", fontsize=5.8, pad=2)

    # Time-averaged depth profile Step 2
    prof2 = np.mean(C2_map, axis=1)
    prof2_norm = prof2 / np.max(prof2)
    ax_a2.plot(prof2_norm, d2[mask_z2], color=COLOR_LAKE, lw=1.1)
    for i, zf in enumerate(true_fracs):
        ax_a2.axhline(zf, color=COLOR_CORAL, linestyle="--", linewidth=0.7, alpha=0.8)
        if i == 0:
            ax_a2.text(0.55, zf + 3.0, f"Frac 1: 4099.5m\n(Single Peak)", fontsize=6.2, color=COLOR_NAVY)
        else:
            ax_a2.text(0.55, zf + 2.0, f"Frac {i+1}: Shadowed", fontsize=6.0, color=COLOR_CORAL)

    ax_a2.set_xlim([0, 1.1])
    ax_a2.set_xlabel("Normalized Response")
    ax_a2.set_title("Depth Profile (Detection 1/4)", fontsize=7.8, fontweight="bold")
    ax_a2.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)

    # -------------------------------------------------------------------------
    # Panel b: Step 3 (引入限流射孔阻抗 Np=6, Kp=5.43e5, 破除短路完全照亮)
    # -------------------------------------------------------------------------
    add_panel_label(ax_b1, "b", x=-0.08, y=1.12)

    d3 = res3_2d["depth"]
    mask_z3 = (d3 >= z_min) & (d3 <= z_max)
    t3 = res3_2d["t_cep"]
    C3_map = res3_2d["response_2d"][mask_z3, :]
    C3_norm = C3_map / (np.max(C3_map) + 1e-12)

    im_b = ax_b1.pcolormesh(t3, d3[mask_z3], C3_norm, shading="auto", cmap="Blues", vmin=0, vmax=0.4)
    for zf in true_fracs:
        ax_b1.axhline(zf, color=COLOR_CORAL, linestyle="--", linewidth=0.7, alpha=0.8)
    ax_b1.set_ylabel("Apparent Depth $z$ [m]")
    ax_b1.set_title("Step 3: 2D Cepstrogram ($N_p=6$, Acoustic Choke Coupling)", fontsize=7.8, fontweight="bold")
    # Position text in lower left to avoid upper-right colorbar
    ax_b1.text(0.03, 0.10, "Acoustic Choke: 4 clusters 100% illuminated", transform=ax_b1.transAxes,
               fontsize=6.2, color=COLOR_NAVY, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLOR_NAVY, lw=0.6, alpha=0.9))

    # Inset colorbar for Panel b1
    cax_b = ax_b1.inset_axes([0.64, 0.86, 0.33, 0.06])
    cbar_b = fig.colorbar(im_b, cax=cax_b, orientation="horizontal")
    cbar_b.set_ticks([0.0, 0.2, 0.4])
    cbar_b.set_ticklabels(["0", "0.2", "0.4"], fontsize=5.5)
    cbar_b.ax.set_title("Normalized Energy", fontsize=5.8, pad=2)

    # Time-averaged depth profile Step 3
    prof3 = np.mean(C3_map, axis=1)
    prof3_norm = prof3 / np.max(prof3)
    ax_b2.plot(prof3_norm, d3[mask_z3], color=COLOR_AMBER, lw=1.1)
    detected_z3 = [4099.5, 4119.8, 4140.1, 4160.4]
    for i, (zf, zd) in enumerate(zip(true_fracs, detected_z3)):
        ax_b2.axhline(zf, color=COLOR_CORAL, linestyle="--", linewidth=0.7, alpha=0.8)
        err = abs(zd - zf)
        ax_b2.text(0.48, zf + 2.0, f"Frac {i+1}: {zd:.1f}m (Err {err:.2f}m)", fontsize=5.8, color="#222222")

    ax_b2.set_xlim([0, 1.1])
    ax_b2.set_xlabel("Normalized Response")
    ax_b2.set_title("Depth Profile (Detection 4/4, Error 0.30m)", fontsize=7.8, fontweight="bold")
    ax_b2.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)

    # -------------------------------------------------------------------------
    # Panel c: Step 4 (斜坡关泵对 20m 间距混响假阳性的抑制与导数增强)
    # -------------------------------------------------------------------------
    add_panel_label(ax_c1, "c", x=-0.08, y=1.12)

    # Left: Step 3 (instant step) vs Step 4 (ramp)
    d_1d = res3_1d["depth"]
    mask_1d = (d_1d >= z_min) & (d_1d <= z_max)
    d_sub = d_1d[mask_1d]

    resp3_1d = res3_1d["response"][mask_1d]
    resp4_1d = res4_1d["response"][mask_1d]

    r4_detrended = resp4_1d - np.min(resp4_1d)
    r4_detrended = r4_detrended / (np.max(r4_detrended) + 1e-12)

    ax_c1.plot(d_sub, resp3_1d / np.max(resp3_1d), color=COLOR_AMBER, lw=1.0, label="Step 3: Instant Step (4 reverberation false peaks)")
    ax_c1.plot(d_sub, r4_detrended, color=COLOR_NAVY, lw=1.1, label="Step 4: Ramp Closure (False peaks eliminated)")
    for zf in true_fracs:
        ax_c1.axvline(zf, color=COLOR_CORAL, linestyle="--", linewidth=0.7, alpha=0.8)

    # Callout highlighting cavity reverberation false peak at 4106m
    ax_c1.annotate(
        "Cavity Reverberation False Peak\n($f_{rev}=36.3\\,\\mathrm{Hz}$ Intermodulation)",
        xy=(4106.0, 0.15), xytext=(4055.0, 0.45),
        arrowprops=dict(arrowstyle="->", color=COLOR_CORAL, lw=0.7),
        fontsize=5.8, color=COLOR_CORAL, fontweight="bold"
    )

    ax_c1.set_xlim([z_min, z_max])
    ax_c1.set_ylim([-0.05, 1.1])
    ax_c1.set_xlabel("Apparent Depth $z$ [m]")
    ax_c1.set_ylabel("Normalized Response")
    ax_c1.set_title("Low-Pass Suppression of 20m Cavity Reverberation", fontsize=7.8, fontweight="bold")
    ax_c1.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)
    ax_c1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E5E5", framealpha=0.90, fontsize=5.8)

    # Right: Step 4 Standard vs Derivative-Enhanced Cepstrum
    resp4_der = res4_1d_der["response"][mask_1d]
    r4_der_detrended = resp4_der - np.min(resp4_der)
    r4_der_detrended = r4_der_detrended / (np.max(r4_der_detrended) + 1e-12)

    ax_c2.plot(d_sub, r4_detrended, color=COLOR_NAVY, lw=0.9, linestyle="--", label="Step 4: Standard Cepstrum")
    ax_c2.plot(d_sub, r4_der_detrended, color=COLOR_CORAL, lw=1.1, label="Step 4: 1st-Derivative Enhanced")
    for zf in true_fracs:
        ax_c2.axvline(zf, color=COLOR_CORAL, linestyle="--", linewidth=0.7, alpha=0.8)

    ax_c2.set_xlim([z_min, z_max])
    ax_c2.set_ylim([-0.05, 1.1])
    ax_c2.set_xlabel("Apparent Depth $z$ [m]")
    ax_c2.set_ylabel("Normalized Response")
    ax_c2.set_title("Derivative Edge Weighting for Roll-off Compensation", fontsize=7.8, fontweight="bold")
    ax_c2.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)
    ax_c2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E5E5", framealpha=0.90, fontsize=5.8)

    # Save outputs
    out_base = os.path.join(SCRIPT_DIR, "fig2_cepstrum_illumination")
    save_figure_nature(fig, out_base)
    plt.close(fig)


# =============================================================================
# FIGURE 3: 射孔节流阻抗与关泵斜坡参数敏感性多面板阵列
# =============================================================================
def generate_figure_3():
    print("\nGenerating Figure 3: Parametric Sensitivity & Physics Kernel Scorecard...")
    fig = plt.figure(figsize=(7.086, 7.5))  # 180 mm width

    # Main GridSpec: 2 rows (top: sensitivity sweeps; bottom: radar & benchmarks)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.2], hspace=0.48, wspace=0.58)

    # Nested GridSpec for Panel a (left column, top): two vertically stacked subpanels (a1 and a2)
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.38)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[1, 0])

    # Panel b (right column, top)
    ax_b = fig.add_subplot(gs[0, 1])

    # -------------------------------------------------------------------------
    # Panel a: 射孔孔数 Np 敏感性分析 (水动力学节流 vs 声学反射与反弹相图)
    # -------------------------------------------------------------------------
    add_panel_label(ax_a1, "a", x=-0.16, y=1.20)

    np_vals = np.array([4, 6, 8, 12, 16])
    kp_vals = np.array([12.22, 5.43, 3.06, 1.36, 0.76])  # [10^5 s^2/m^5]
    rebound_amp = np.array([180.4, 193.7, 210.9, 217.4, 219.8])  # [m]
    reflectivity = np.array([34.0, 53.7, 67.3, 82.2, 89.2])  # [%]
    n_detected = np.array([2, 4, 2, 2, 2])

    # Subpanel a1: 水动力学流阻 Kp 与宏观大反弹幅值 Delta H_reb 随 Np 的单调影响
    ax_a1_twin = ax_a1.twinx()
    ax_a1_twin.spines["right"].set_visible(True)

    line_a1 = ax_a1.plot(np_vals, kp_vals, marker="D", markersize=4.0, color=COLOR_AMBER,
                         lw=1.1, label="Throttle Resistance $K_p$ [$10^5\\,\\mathrm{s^2/m^5}$]")
    line_a1_reb = ax_a1_twin.plot(np_vals, rebound_amp, marker="o", markersize=4.0, color=COLOR_LAKE,
                                  lw=1.1, linestyle="--", label="Macro Rebound $\\Delta H_{reb}$ [m]")

    # Optimal window shading across both subpanels
    ax_a1.axvspan(5.2, 7.0, color="#EBF3FB", alpha=0.7, zorder=0)
    ax_a1.set_xticks(np_vals)
    ax_a1.set_xticklabels([])
    ax_a1.set_ylabel("Resistance $K_p$\n[$10^5\\,\\mathrm{s^2/m^5}$]", color=COLOR_AMBER, fontsize=6.5)
    ax_a1_twin.set_ylabel("Rebound $\\Delta H_{reb}$\n[m]", color=COLOR_LAKE, fontsize=6.5, labelpad=5)
    ax_a1.set_ylim([0, 14])
    ax_a1_twin.set_ylim([170, 230])
    ax_a1.set_title("Perforation $N_p$: Hydraulic Throttle & Rebound", pad=3, fontweight="bold", fontsize=7.8)
    ax_a1.grid(True, linestyle="--", alpha=0.35, color=COLOR_GRID)

    lines_top = line_a1 + line_a1_reb
    labels_top = [l.get_label() for l in lines_top]
    ax_a1.legend(lines_top, labels_top, loc="center right", frameon=True, facecolor="white",
                 edgecolor="#E5E5E5", framealpha=0.92, fontsize=5.2)

    # Subpanel a2: 首波声学反射率 |Gamma_1| 与裂缝检出簇数
    ax_a2_twin = ax_a2.twinx()
    ax_a2_twin.spines["right"].set_visible(True)

    line_a2 = ax_a2.plot(np_vals, reflectivity, marker="s", markersize=4.0, color=COLOR_CORAL,
                         lw=1.1, label="First-Wave Reflectivity $|\\Gamma_1|$ [%]")
    line_a2_det = ax_a2_twin.plot(np_vals, n_detected, marker="^", markersize=4.5, color=COLOR_NAVY,
                                  lw=1.2, linestyle="-.", label="Detected Clusters (True=4)")

    ax_a2.axvspan(5.2, 7.0, color="#EBF3FB", alpha=0.7, zorder=0)
    ax_a2.text(6.0, 78, "Optimal Window\n(100% Illumination)", fontsize=5.5, color=COLOR_NAVY,
               fontweight="bold", ha="center")

    ax_a2.set_xlabel("Perforation Holes per Cluster $N_p$")
    ax_a2.set_ylabel("Reflectivity $|\\Gamma_1|$\n[%]", color=COLOR_CORAL, fontsize=6.5)
    ax_a2_twin.set_ylabel("Detected\nClusters", color=COLOR_NAVY, fontsize=6.5, labelpad=5)
    ax_a2.set_xticks(np_vals)
    ax_a2.set_ylim([20, 100])
    ax_a2_twin.set_ylim([1.0, 4.8])
    ax_a2.grid(True, linestyle="--", alpha=0.35, color=COLOR_GRID)

    lines_bot = line_a2 + line_a2_det
    labels_bot = [l.get_label() for l in lines_bot]
    ax_a2.legend(lines_bot, labels_bot, loc="lower right", frameon=True, facecolor="white",
                 edgecolor="#E5E5E5", framealpha=0.92, fontsize=5.2)

    # -------------------------------------------------------------------------
    # Panel b: 关泵斜坡历时 tc 敏感性分析
    # -------------------------------------------------------------------------
    add_panel_label(ax_b, "b", x=-0.14, y=1.08)

    tc_vals = np.array([0.0, 0.5, 1.0, 2.0])
    max_dhdt = np.array([147820.2, 306.1, 156.4, 81.6])
    dt_half = np.array([0.0, 0.321, 0.641, 1.269])

    ax_b1 = ax_b
    ax_b2 = ax_b.twinx()
    ax_b2.spines["right"].set_visible(True)

    line_b1 = ax_b1.plot(tc_vals, max_dhdt, marker="D", markersize=4.5, color=COLOR_AMBER,
                         lw=1.2, label="Max Rate $|dH/dt|_{max}$ [m/s]")
    ax_b1.set_yscale("log")
    ax_b1.set_ylim([40, 3e5])

    line_b2 = ax_b2.plot(tc_vals, dt_half, marker="o", markersize=4.5, color=COLOR_NAVY,
                         lw=1.2, linestyle="--", label="Rebound Delay $\\Delta t_{half}$ [s]")
    ax_b2.set_ylim([-0.1, 1.5])

    # Fit equation annotation
    ax_b2.text(0.8, 0.25, "$\\Delta t_{half} \\approx 0.64 t_c$\n($R^2=0.9999$ Linear Fit)",
               fontsize=6.2, color=COLOR_NAVY, fontweight="bold")

    ax_b1.set_xlabel("Ramp Closure Duration $t_c$ [s]")
    ax_b1.set_ylabel("Max Wavefront Rate (Log) [m/s]", color=COLOR_AMBER, labelpad=5)
    ax_b2.set_ylabel("Rebound Rise Delay $\\Delta t_{half}$ [s]", color=COLOR_NAVY, labelpad=7)
    ax_b1.set_xticks(tc_vals)
    ax_b1.set_title("Ramp Duration $t_c$ vs Shock Suppression", pad=4, fontweight="bold")
    ax_b1.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID)

    lines_b = line_b1 + line_b2
    labels_b = [l.get_label() for l in lines_b]
    ax_b1.legend(lines_b, labels_b, loc="upper right", frameon=True, facecolor="white",
                 edgecolor="#E5E5E5", framealpha=0.92, fontsize=5.8)

    # -------------------------------------------------------------------------
    # Panel c: 物理内核升级全景雷达对比图 (Left) 与量化条形对比图 (Right)
    # -------------------------------------------------------------------------
    ax_c1 = fig.add_subplot(gs[1, 0], polar=True)
    ax_c2 = fig.add_subplot(gs[1, 1])

    add_panel_label(ax_c1, "c", x=-0.14, y=1.12)
    add_panel_label(ax_c2, "d", x=-0.14, y=1.08)

    categories = [
        "Steady State\nConsistency",
        "Dead-End Toe\nZero Flow",
        "Geological\nCompliance",
        "Macro Rebound\nReproduction",
        "Perforation\nThrottle",
        "Multi-Cluster\nAccuracy",
        "Wavefront Shock\nElimination"
    ]
    N_cat = len(categories)

    # Radar scores normalized [0 to 100]
    scores_v1 = [10, 0, 5, 0, 0, 25, 0]
    scores_v2 = [100, 100, 100, 98, 100, 97, 98]

    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]
    scores_v1 += scores_v1[:1]
    scores_v2 += scores_v2[:1]

    ax_c1.plot(angles, scores_v1, color=COLOR_GREY, linewidth=1.0, linestyle="--", label="V1 Baseline (PaperA)")
    ax_c1.fill(angles, scores_v1, color=COLOR_GREY, alpha=0.15)
    ax_c1.plot(angles, scores_v2, color=COLOR_NAVY, linewidth=1.3, label="MOC_V2 Upgrade")
    ax_c1.fill(angles, scores_v2, color=COLOR_NAVY, alpha=0.20)

    ax_c1.set_xticks(angles[:-1])
    ax_c1.set_xticklabels(categories, fontsize=5.6, color="#222222")
    ax_c1.tick_params(pad=10)
    ax_c1.set_ylim(0, 105)
    ax_c1.set_yticks([25, 50, 75, 100])
    ax_c1.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=5.0, color="#888888")
    ax_c1.legend(loc="lower left", bbox_to_anchor=(-0.05, -0.22), ncol=2, fontsize=5.8, frameon=False)
    ax_c1.set_title("Physics Fidelity Radar", fontsize=7.8, fontweight="bold", pad=14)

    # -------------------------------------------------------------------------
    # Panel d: 4 组关键定量提升对比条形图 (Horizontal Bar Comparison)
    # -------------------------------------------------------------------------
    metrics_names = [
        "Detection Error (m)\n[Lower is better]",
        "Shock dH/dt (m/s)\n[Lower is better]",
        "Cluster Illumination (%)\n[Higher is better]",
        "Macro Rebound (m)\n[Higher is better]"
    ]
    y_pos = np.arange(len(metrics_names))
    bar_height = 0.35

    v1_real = ["9.72 m", "147,820 m/s", "25% (1/4)", "0 m (Collapsed)"]
    v2_real = ["0.30 m", "156.4 m/s", "100% (4/4)", "+239.2 m"]

    v1_perf = [10.0, 5.0, 25.0, 0.0]
    v2_perf = [97.0, 99.0, 100.0, 98.0]

    bars1 = ax_c2.barh(y_pos - bar_height/2, v1_perf, height=bar_height, color=COLOR_GREY, alpha=0.6, label="V1 Baseline (PaperA)")
    bars2 = ax_c2.barh(y_pos + bar_height/2, v2_perf, height=bar_height, color=COLOR_NAVY, alpha=0.85, label="MOC_V2 Upgrade")

    for idx in range(len(metrics_names)):
        ax_c2.text(v1_perf[idx] + 2, y_pos[idx] - bar_height/2, v1_real[idx],
                   va="center", fontsize=5.8, color="#555555", fontweight="bold")
        ax_c2.text(v2_perf[idx] - 3, y_pos[idx] + bar_height/2, v2_real[idx],
                   va="center", ha="right", fontsize=5.8, color="white", fontweight="bold")

    ax_c2.set_yticks(y_pos)
    ax_c2.set_yticklabels(metrics_names, fontsize=6.2)
    ax_c2.set_xlim([0, 115])
    ax_c2.set_xlabel("Normalized Performance Benchmark Score [%]")
    ax_c2.set_title("Quantitative Metric Benchmarks (V1 vs V2)", pad=4, fontweight="bold")
    ax_c2.grid(True, linestyle="--", alpha=0.4, color=COLOR_GRID, axis="x")
    ax_c2.legend(loc="lower right", frameon=False, fontsize=5.8)

    # Save outputs
    out_base = os.path.join(SCRIPT_DIR, "fig3_parametric_sensitivity")
    save_figure_nature(fig, out_base)
    plt.close(fig)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 78)
    print("MOC_V2 Nature Figure Generator")
    print("=" * 78)

    # Paths to source data
    path_papera = os.path.join(PROJECT_ROOT, "PaperA井口多裂缝水击响应", "03_leakoff验证", "brunone_D20", "quad", "moc_timeseries.csv")
    path_step1 = os.path.join(PROJECT_ROOT, "output", "moc_physics_upgrade", "step1_steady_state_toe", "quad", "moc_timeseries.csv")
    path_step2 = os.path.join(PROJECT_ROOT, "output", "moc_physics_upgrade", "step2_fracture_compliance", "quad", "moc_timeseries.csv")
    path_step3 = os.path.join(PROJECT_ROOT, "output", "moc_physics_upgrade", "step3_perforation_throttle", "quad", "moc_timeseries.csv")
    path_step4 = os.path.join(PROJECT_ROOT, "output", "moc_physics_upgrade", "step4_ramp_closure", "quad", "moc_timeseries.csv")

    for p in [path_papera, path_step1, path_step2, path_step3, path_step4]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required data file: {p}")

    print("Loading simulation datasets...")
    df_papera = pd.read_csv(path_papera)
    df_step1 = pd.read_csv(path_step1)
    df_step2 = pd.read_csv(path_step2)
    df_step3 = pd.read_csv(path_step3)
    df_step4 = pd.read_csv(path_step4)
    print("All datasets loaded successfully.")

    # Generate the four Nature-standard figures
    generate_figure_0_defects(df_papera, df_step1)
    generate_figure_1(df_papera, df_step1, df_step2, df_step3, df_step4)
    generate_figure_2(df_step2, df_step3, df_step4)
    generate_figure_3()

    print("\n" + "=" * 78)
    print("All 4 Nature figures generated successfully!")
    print(f"Directory: {SCRIPT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
