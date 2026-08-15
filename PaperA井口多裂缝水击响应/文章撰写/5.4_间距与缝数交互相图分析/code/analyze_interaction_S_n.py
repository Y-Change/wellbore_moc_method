"""
Analysis and Plotting Script for Section 5.4: Spacing-Multiplicity Interaction Phase Map (S x n_total)
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

PALETTE_N = ["#1B4F72", "#117864", "#D35400", "#900C3F", "#6C3483", "#2874A6", "#B7950B"]

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")
SEC_DIR  = os.path.join(BASE_DIR, "文章撰写", "5.4_间距与缝数交互相图分析")
FIG_DIR  = os.path.join(SEC_DIR, "figures")
TAB_DIR  = os.path.join(SEC_DIR, "data")


def extract_interaction_metrics():
    """
    Extracts 2D grid metrics across S in 10..100m and n_total in 2..8 at X1=3000m.
    Computes P1, P_end, R_end, sum_P, and physical cluster span L_span = (n-1)*S.
    """
    print("[1/3] Extracting 2D interaction grid metrics for S x n_total...")
    csv_table = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
    df = pd.read_csv(csv_table)
    df_sub = df[(df['friction_model'] == 'steady') & (df['x1'] == 3000.0) & (df['n_total'] >= 2)]
    
    grid_records = []
    for (n, sp), group in df_sub.groupby(['n_total', 'spacing_m']):
        group_sorted = group.sort_values('frac_idx')
        p1 = group_sorted[group_sorted['frac_idx'] == 1]['P_2d'].values[0]
        pn = group_sorted[group_sorted['frac_idx'] == n]['P_2d'].values[0]
        sum_p = group_sorted['P_2d'].sum()
        r_end = pn / p1
        l_span = (n - 1) * sp
        
        grid_records.append({
            'friction_model': 'steady',
            'x1': 3000.0,
            'n_total': int(n),
            'spacing_m': float(sp),
            'L_span_m': float(l_span),
            'P_1': p1,
            'P_end': pn,
            'R_end': r_end,
            'sum_P': sum_p,
        })
        
    df_grid = pd.DataFrame(grid_records)
    out_csv = os.path.join(TAB_DIR, "interaction_S_n_metrics.csv")
    df_grid.to_csv(out_csv, index=False)
    print(f"[Done] Interaction metrics saved to: {out_csv} ({len(df_grid)} grid points)")
    return df_grid


def plot_figure_5_4(df_grid):
    """
    Figure 5.4 Main Publication Figure:
    3-Panel (Double column 7.35 x 2.35 in, aspect 4:5):
    (a) 2D Heatmap of P1(S, n_total) with vertical colorbar on the right
    (b) 2D Heatmap of R_end(S, n_total) with vertical colorbar on the right
    (c) Iso-Cluster Span Test: R_end vs Physical Cluster Span L_span = (n-1)*S
    """
    print("[2/3] Generating Figure 5.4 (Main S x n_total Interaction Figure)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)
    
    # Grid interpolation mesh
    S_dense = np.linspace(10, 100, 120)
    n_dense = np.linspace(2, 8, 120)
    SS, NN = np.meshgrid(S_dense, n_dense)
    points = df_grid[['spacing_m', 'n_total']].values
    
    # Custom high-contrast nature palettes
    cmap_blue_warm = LinearSegmentedColormap.from_list("nature_blue_warm", ["#EBF5FB", "#AED6F1", "#5DADE2", "#F5B041", "#E74C3C"])
    cmap_teal_warm = LinearSegmentedColormap.from_list("nature_teal_warm", ["#E8F8F5", "#A3E4D7", "#48C9B0", "#E67E22", "#C0392B"])
    
    # --- (a) 2D Contour Phase Map: P1(S, n_total) ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    PP1 = griddata(points, df_grid['P_1'].values, (SS, NN), method='cubic')
    cnt1 = ax.contourf(SS, NN, PP1, levels=14, cmap=cmap_blue_warm, alpha=0.95)
    ax.scatter(points[:, 0], points[:, 1], color=COLOR_SLATE, s=6, alpha=0.5, edgecolors='none')
    
    cbar1 = fig.colorbar(cnt1, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label("Primary Peak $P_1$ (a.u.)", fontsize=7.0, labelpad=4)
    cbar1.ax.tick_params(labelsize=6.0, length=2)
    
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Fracture Multiplicity $n_{total}$")
    ax.set_xlim(10, 100)
    ax.set_ylim(2, 8)
    ax.set_yticks([2, 3, 4, 5, 6, 7, 8])
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) 2D Contour Phase Map: R_end(S, n_total) ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    RR = griddata(points, df_grid['R_end'].values, (SS, NN), method='cubic')
    cnt2 = ax.contourf(SS, NN, RR, levels=14, cmap=cmap_teal_warm, alpha=0.95)
    ax.scatter(points[:, 0], points[:, 1], color=COLOR_SLATE, s=6, alpha=0.5, edgecolors='none')
    
    cbar2 = fig.colorbar(cnt2, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label("End Ratio $R_{end} = P_n/P_1$", fontsize=7.0, labelpad=4)
    cbar2.ax.tick_params(labelsize=6.0, length=2)
    
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Fracture Multiplicity $n_{total}$")
    ax.set_xlim(10, 100)
    ax.set_ylim(2, 8)
    ax.set_yticks([2, 3, 4, 5, 6, 7, 8])
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) Iso-Cluster Span Test: R_end vs L_span = (n-1)*S ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    n_list = [2, 3, 4, 5, 6, 7, 8]
    cmap_n = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_list)))
    
    for n_val, col in zip(n_list, cmap_n):
        sub_n = df_grid[df_grid['n_total'] == n_val].sort_values('L_span_m')
        ax.plot(sub_n['L_span_m'], sub_n['R_end'], marker='o', label=f"$n_{{tot}}={n_val}$",
                color=col, lw=1.0, markersize=3.2)
        
    ax.axhline(0.0, color="#BDC3C7", ls=":", lw=0.8)
    ax.set_xlabel("Cluster Physical Span $L_{span} = (n-1)S$ (m)")
    ax.set_ylabel("End-Member Ratio $R_{end} = P_n/P_1$")
    ax.set_xlim(-15, 720)
    # Headroom expansion to 0.70 to ensure ZERO overlap with legend
    ax.set_ylim(0.0, 0.70)
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_4_Interaction_Phase_Maps")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.4: {out_prefix}.png / .svg / .pdf")


def plot_figure_5_4_supp(df_grid):
    """
    Figure 5.4 Supplementary / Extended Interaction Analysis:
    (a) Direct Comparison of Matched Span Configurations (L_span = 60, 100, 120, 200m)
    (b) 2D Heatmap of Total Detected Peak Sum \Sigma P(S, n_total)
    (c) Spacing Sensitivity Trajectories of P1(S) grouped by Multiplicity
    """
    print("[3/3] Generating Figure 5.4 Supplementary (Matched Span & Energy Phase Map)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)
    
    # --- (a) Matched Span Divergence: R_end for identical L_span ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    
    # 4 matched span sets
    matched_sets = [
        {'L': 60,  'cfgs': [(2, 60), (3, 30), (4, 20), (7, 10)],  'col': COLOR_BLUE_MAIN},
        {'L': 100, 'cfgs': [(2, 100), (3, 50), (6, 20)],           'col': COLOR_TEAL},
        {'L': 120, 'cfgs': [(3, 60), (4, 40), (5, 30), (7, 20)],  'col': COLOR_ORANGE},
        {'L': 200, 'cfgs': [(3, 100), (5, 50), (6, 40)],          'col': COLOR_RED},
    ]
    
    for s_info in matched_sets:
        L_target = s_info['L']
        col = s_info['col']
        n_vals, r_vals = [], []
        for (n_val, sp_val) in s_info['cfgs']:
            match = df_grid[(df_grid['n_total'] == n_val) & (df_grid['spacing_m'] == sp_val)]
            if len(match) > 0:
                n_vals.append(n_val)
                r_vals.append(match['R_end'].values[0])
        ax.plot(n_vals, r_vals, marker='o', label=f"$L_{{span}}={L_target}$ m", color=col, lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Fracture Multiplicity $n_{total}$")
    ax.set_ylabel("End-Member Ratio $R_{end} = P_n/P_1$")
    ax.set_xlim(1.5, 7.5)
    ax.set_xticks([2, 3, 4, 5, 6, 7])
    ax.set_ylim(0.0, 0.65)
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) 2D Heatmap: Total Detected Energy Sum \Sigma P(S, n_total) ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    
    S_dense = np.linspace(10, 100, 120)
    n_dense = np.linspace(2, 8, 120)
    SS, NN = np.meshgrid(S_dense, n_dense)
    points = df_grid[['spacing_m', 'n_total']].values
    
    cmap_energy = LinearSegmentedColormap.from_list("nature_energy", ["#FADBD8", "#F5B7B1", "#EDBB99", "#A569BD", "#2C3E50"])
    SS_P = griddata(points, df_grid['sum_P'].values, (SS, NN), method='cubic')
    cnt_e = ax.contourf(SS, NN, SS_P, levels=14, cmap=cmap_energy, alpha=0.95)
    ax.scatter(points[:, 0], points[:, 1], color=COLOR_SLATE, s=6, alpha=0.5, edgecolors='none')
    
    cbar_e = fig.colorbar(cnt_e, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_e.set_label("Total Peak Sum $\\sum P_i$ (a.u.)", fontsize=7.0, labelpad=4)
    cbar_e.ax.tick_params(labelsize=6.0, length=2)
    
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Fracture Multiplicity $n_{total}$")
    ax.set_xlim(10, 100)
    ax.set_ylim(2, 8)
    ax.set_yticks([2, 3, 4, 5, 6, 7, 8])
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) P1 vs S grouped by Multiplicity ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    n_sel = [2, 4, 6, 8]
    colors_sel = [COLOR_TEAL, COLOR_ORANGE, COLOR_PURPLE, COLOR_RED]
    
    for n_val, col in zip(n_sel, colors_sel):
        sub_n = df_grid[df_grid['n_total'] == n_val].sort_values('spacing_m')
        ax.plot(sub_n['spacing_m'], sub_n['P_1'], marker='o', label=f"$n_{{tot}}={n_val}$",
                color=col, lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Primary Peak $P_1$ (a.u.)")
    ax.set_xlim(5, 105)
    # Headroom expansion to 4.8 to guarantee ZERO overlap
    ax.set_ylim(2.0, 4.8)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.25)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_4_Supp_Iso_Span_Divergence")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.4 Supp: {out_prefix}.png / .svg / .pdf")


if __name__ == "__main__":
    df_grid = extract_interaction_metrics()
    plot_figure_5_4(df_grid)
    plot_figure_5_4_supp(df_grid)
