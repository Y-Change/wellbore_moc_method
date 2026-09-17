"""
Analysis and Plotting Script for Section 5.5: Primary Depth and Spacing Interaction Phase Map (X1 x S)
Author: Antigravity (Pair Programming with User)
Target: Academic Journal Paper A (Steady Friction MOC Transient Wavefield)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

# Ensure repository root is on sys.path
REPO_ROOT = r"E:\water_hammer_research\wellbore_moc_method"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Matplotlib rcParams Configuration: Strict Paper A Specification ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.75,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.8,
    # Full box border and inward ticks
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "legend.frameon": False,
    "figure.dpi": 300,
})

# --- Curated Color Palettes ---
COLOR_BLUE_MAIN = "#1B4F72"
COLOR_TEAL      = "#117864"
COLOR_ORANGE    = "#D35400"
COLOR_RED       = "#900C3F"
COLOR_PURPLE    = "#6C3483"
COLOR_SLATE     = "#2C3E50"
COLOR_GOLD      = "#B7950B"
COLOR_GREY      = "#7F8C8D"

PALETTE_DEPTHS = ["#1B4F72", "#117864", "#D35400", "#900C3F", "#6C3483", "#2C3E50"]

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")
SEC_DIR  = os.path.join(BASE_DIR, "文章撰写", "5.5_深度与间距交互相图分析")
FIG_DIR  = os.path.join(SEC_DIR, "figures")
TAB_DIR  = os.path.join(SEC_DIR, "data")


def extract_interaction_X1_S_metrics():
    """
    Extracts 2D grid metrics across X1 in [2000..4500m] and S in [10..100m] for n_total=4.
    Computes P1..P4 and alpha2..alpha4.
    """
    print("[1/3] Extracting 2D interaction grid metrics for X1 x S (n_total=4)...")
    csv_table = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
    df = pd.read_csv(csv_table)
    df_n4 = df[(df['friction_model'] == 'steady') & (df['n_total'] == 4)]
    
    grid_records = []
    for (x1_val, sp_val), group in df_n4.groupby(['x1', 'spacing_m']):
        group_sorted = group.sort_values('frac_idx')
        p_vals = group_sorted['P_2d'].values
        a_vals = group_sorted['alpha_2d'].values
        
        p1 = p_vals[0] if len(p_vals) > 0 else np.nan
        p2 = p_vals[1] if len(p_vals) > 1 else np.nan
        p3 = p_vals[2] if len(p_vals) > 2 else np.nan
        p4 = p_vals[3] if len(p_vals) > 3 else np.nan
        
        a2 = a_vals[1] if len(a_vals) > 1 else np.nan
        a3 = a_vals[2] if len(a_vals) > 2 else np.nan
        a4 = a_vals[3] if len(a_vals) > 3 else np.nan
        
        grid_records.append({
            'friction_model': 'steady',
            'n_total': 4,
            'x1': float(x1_val),
            'spacing_m': float(sp_val),
            'P_1': p1,
            'P_2': p2,
            'P_3': p3,
            'P_4': p4,
            'alpha_2': a2,
            'alpha_3': a3,
            'alpha_4': a4,
        })
        
    df_grid = pd.DataFrame(grid_records)
    out_csv = os.path.join(TAB_DIR, "interaction_X1_S_metrics.csv")
    df_grid.to_csv(out_csv, index=False)
    print(f"[Done] Interaction metrics saved to: {out_csv} ({len(df_grid)} grid points)")
    return df_grid


def plot_figure_5_5(df_grid):
    """
    Figure 5.5 Main Publication Figure:
    3-Panel (Double column 7.35 x 2.35 in, aspect 4:5):
    (a) 2D Heatmap of P1(S, X1) with vertical colorbar on the right
    (b) 2D Heatmap of alpha2(S, X1) with vertical colorbar on the right
    (c) Orthogonal Decoupling: alpha2 vs S across all 6 depths X1 in [2000, 4500m]
    """
    print("[2/3] Generating Figure 5.5 (Main X1 x S Interaction Figure)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)
    
    # Grid interpolation mesh
    S_dense = np.linspace(10, 100, 120)
    X_dense = np.linspace(2000, 4500, 120)
    SS, XX = np.meshgrid(S_dense, X_dense)
    points = df_grid[['spacing_m', 'x1']].values
    
    # Custom high-contrast nature palettes
    cmap_blue_warm = LinearSegmentedColormap.from_list("nature_blue_warm", ["#EBF5FB", "#AED6F1", "#5DADE2", "#F5B041", "#E74C3C"])
    cmap_teal_warm = LinearSegmentedColormap.from_list("nature_teal_warm", ["#E8F8F5", "#A3E4D7", "#48C9B0", "#E67E22", "#C0392B"])
    
    # --- (a) 2D Contour Phase Map: P1(S, X1) ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    PP1 = griddata(points, df_grid['P_1'].values, (SS, XX), method='cubic')
    cnt1 = ax.contourf(SS, XX, PP1, levels=14, cmap=cmap_blue_warm, alpha=0.95)
    ax.scatter(points[:, 0], points[:, 1], color=COLOR_SLATE, s=6, alpha=0.5, edgecolors='none')
    
    cbar1 = fig.colorbar(cnt1, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label("Primary Peak $P_1$ (a.u.)", fontsize=7.0, labelpad=4)
    cbar1.ax.tick_params(labelsize=6.0, length=2)
    
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Primary Depth $X_1$ (m)")
    ax.set_xlim(10, 100)
    ax.set_ylim(2000, 4500)
    ax.set_yticks([2000, 2500, 3000, 3500, 4000, 4500])
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) 2D Contour Phase Map: alpha2(S, X1) ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    AA2 = griddata(points, df_grid['alpha_2'].values, (SS, XX), method='cubic')
    cnt2 = ax.contourf(SS, XX, AA2, levels=14, cmap=cmap_teal_warm, alpha=0.95)
    ax.scatter(points[:, 0], points[:, 1], color=COLOR_SLATE, s=6, alpha=0.5, edgecolors='none')
    
    cbar2 = fig.colorbar(cnt2, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label("Relative Ratio $\\alpha_2 = P_2/P_1$", fontsize=7.0, labelpad=4)
    cbar2.ax.tick_params(labelsize=6.0, length=2)
    
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Primary Depth $X_1$ (m)")
    ax.set_xlim(10, 100)
    ax.set_ylim(2000, 4500)
    ax.set_yticks([2000, 2500, 3000, 3500, 4000, 4500])
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) Orthogonal Decoupling: alpha2(S) across X1 ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    depths = [2000, 2500, 3000, 3500, 4000, 4500]
    for d_val, col in zip(depths, PALETTE_DEPTHS):
        sub_d = df_grid[df_grid['x1'] == d_val].sort_values('spacing_m')
        ax.plot(sub_d['spacing_m'], sub_d['alpha_2'], marker='o', label=f"$X_1={d_val}$ m",
                color=col, lw=1.0, markersize=3.2)
        
    ax.axhline(0.5, color="#BDC3C7", ls=":", lw=0.8)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Relative Ratio $\\alpha_2 = P_2/P_1$")
    ax.set_xlim(5, 105)
    # Headroom expansion to 0.85 to ensure ZERO overlap with legend
    ax.set_ylim(0.25, 0.85)
    ax.legend(loc="upper right", fontsize=6.0, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_5_Interaction_X1_S_Phase_Maps")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.5: {out_prefix}.png / .svg / .pdf")


def plot_figure_5_5_supp(df_grid):
    """
    Figure 5.5 Supplementary / Deep Fracture Coupling across X1 and S:
    (a) 2D Heatmap of alpha3(S, X1)
    (b) 2D Heatmap of alpha4(S, X1)
    (c) Universal Preservation of U-Valley in alpha3(S) across Depths
    """
    print("[3/3] Generating Figure 5.5 Supplementary (Deep Fracture Coupling)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)
    
    S_dense = np.linspace(10, 100, 120)
    X_dense = np.linspace(2000, 4500, 120)
    SS, XX = np.meshgrid(S_dense, X_dense)
    points = df_grid[['spacing_m', 'x1']].values
    
    cmap_purple = LinearSegmentedColormap.from_list("nature_purple", ["#F4ECF7", "#D7BDE2", "#AF7AC5", "#7D3C98", "#4A235A"])
    cmap_orange = LinearSegmentedColormap.from_list("nature_orange", ["#FEF5E7", "#FAD7A0", "#F8C471", "#E67E22", "#BA4A00"])
    
    # --- (a) 2D Heatmap: alpha3(S, X1) ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    AA3 = griddata(points, df_grid['alpha_3'].values, (SS, XX), method='cubic')
    cnt_a = ax.contourf(SS, XX, AA3, levels=14, cmap=cmap_purple, alpha=0.95)
    ax.scatter(points[:, 0], points[:, 1], color=COLOR_SLATE, s=6, alpha=0.5, edgecolors='none')
    
    cbar_a = fig.colorbar(cnt_a, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_a.set_label("Relative Ratio $\\alpha_3 = P_3/P_1$", fontsize=7.0, labelpad=4)
    cbar_a.ax.tick_params(labelsize=6.0, length=2)
    
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Primary Depth $X_1$ (m)")
    ax.set_xlim(10, 100)
    ax.set_ylim(2000, 4500)
    ax.set_yticks([2000, 2500, 3000, 3500, 4000, 4500])
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) 2D Heatmap: alpha4(S, X1) ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    AA4 = griddata(points, df_grid['alpha_4'].values, (SS, XX), method='cubic')
    cnt_b = ax.contourf(SS, XX, AA4, levels=14, cmap=cmap_orange, alpha=0.95)
    ax.scatter(points[:, 0], points[:, 1], color=COLOR_SLATE, s=6, alpha=0.5, edgecolors='none')
    
    cbar_b = fig.colorbar(cnt_b, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_b.set_label("Relative Ratio $\\alpha_4 = P_4/P_1$", fontsize=7.0, labelpad=4)
    cbar_b.ax.tick_params(labelsize=6.0, length=2)
    
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Primary Depth $X_1$ (m)")
    ax.set_xlim(10, 100)
    ax.set_ylim(2000, 4500)
    ax.set_yticks([2000, 2500, 3000, 3500, 4000, 4500])
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) U-Valley Stability across Selected Depths ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    sel_depths = [2000, 3000, 4000, 4500]
    sel_cols = [COLOR_BLUE_MAIN, COLOR_TEAL, COLOR_ORANGE, COLOR_RED]
    
    for d_val, col in zip(sel_depths, sel_cols):
        sub_d = df_grid[df_grid['x1'] == d_val].sort_values('spacing_m')
        ax.plot(sub_d['spacing_m'], sub_d['alpha_3'], marker='s', label=f"$X_1={d_val}$ m",
                color=col, lw=1.1, markersize=3.4)
        
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Relative Ratio $\\alpha_3 = P_3/P_1$")
    ax.set_xlim(5, 105)
    # Headroom expansion to 0.65 to ensure ZERO overlap
    ax.set_ylim(0.10, 0.65)
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.25)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_5_Supp_Deep_Fracture_Coupling")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.5 Supp: {out_prefix}.png / .svg / .pdf")


if __name__ == "__main__":
    df_grid = extract_interaction_X1_S_metrics()
    plot_figure_5_5(df_grid)
    plot_figure_5_5_supp(df_grid)
