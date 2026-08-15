"""
Analysis and Plotting Script for Section 5.3: Primary Fracture Depth X1 Main Effect
Author: Antigravity (Pair Programming with User)
Target: Academic Journal Paper A (Steady Friction MOC Transient Wavefield)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
REPO_ROOT = r"E:\water_hammer_research\wellbore_moc_method"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum

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
    "legend.fontsize": 6.8,
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

# --- Palette ---
COLOR_BLUE_MAIN = "#1B4F72"
COLOR_TEAL      = "#117864"
COLOR_ORANGE    = "#D35400"
COLOR_RED       = "#900C3F"
COLOR_PURPLE    = "#6C3483"
COLOR_SLATE     = "#2C3E50"
COLOR_GOLD      = "#B7950B"
COLOR_GREY      = "#7F8C8D"

PALETTE_FRACTURES = ["#1B4F72", "#117864", "#D35400", "#900C3F", "#6C3483", "#2874A6", "#B7950B", "#5D6D7E"]

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")
SEC_DIR  = os.path.join(BASE_DIR, "文章撰写", "5.3_首缝深度X1主效应分析")
FIG_DIR  = os.path.join(SEC_DIR, "figures")
TAB_DIR  = os.path.join(SEC_DIR, "data")


def extract_depth_metrics():
    """
    Extracts metrics for X1 in [2000, 2500, 3000, 3500, 4000, 4500]m across spacings and n_total.
    """
    print("[1/3] Extracting depth effect metrics across all X1 depths...")
    csv_table = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
    df = pd.read_csv(csv_table)
    df_steady = df[df['friction_model'] == 'steady'].copy()
    
    depth_records = []
    x1_list = np.sort(df_steady['x1'].unique())
    
    for x1 in x1_list:
        sub_x1 = df_steady[df_steady['x1'] == x1]
        for (n, sp), group in sub_x1.groupby(['n_total', 'spacing_m']):
            group_sorted = group.sort_values('frac_idx')
            p1 = group_sorted[group_sorted['frac_idx'] == 1]['P_2d'].values[0]
            for idx, row in group_sorted.iterrows():
                f_idx = int(row['frac_idx'])
                p2d = float(row['P_2d'])
                alpha2d = float(row['alpha_2d'])
                x_f = float(row['x_f'])
                tx = x1 + (f_idx - 1) * sp
                loc_err = abs(x_f - tx)
                
                depth_records.append({
                    'friction_model': 'steady',
                    'x1': float(x1),
                    'n_total': int(n),
                    'spacing_m': float(sp),
                    'frac_idx': f_idx,
                    'x_true': tx,
                    'x_extracted': x_f,
                    'loc_error_m': loc_err,
                    'P_2d': p2d,
                    'alpha_2d': alpha2d,
                    'tau_round_sec': 2.0 * x1 / 1450.0,
                })
                
    df_metrics = pd.DataFrame(depth_records)
    out_csv = os.path.join(TAB_DIR, "depth_effect_metrics.csv")
    df_metrics.to_csv(out_csv, index=False)
    print(f"[Done] Depth metrics saved to: {out_csv} ({len(df_metrics)} rows)")
    return df_steady, df_metrics


def plot_figure_5_3(df_steady, df_metrics):
    """
    Figure 5.3 Main Publication Figure:
    3-Panel (Double column 7.2 x 2.35 in, aspect 4:5):
    (a) Aligned Spectral Profiles for X1 in {2000, 3000, 4000}m at S=20m, n=4
    (b) Absolute Peak Trajectories P_i(X1) vs Depth X1
    (c) Relative Ratio Invariance alpha_i(X1) vs Depth X1
    """
    print("[2/3] Generating Figure 5.3 (Main Depth Effect Figure)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.35), constrained_layout=True)
    
    # --- (a) Aligned Profiles Delta x = x - X1 at S=20m, n=4 ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    
    x1_sample = [2000, 3000, 4000]
    colors_x1 = [COLOR_TEAL, COLOR_BLUE_MAIN, COLOR_RED]
    sp = 20
    n = 4
    
    for x1_val, col in zip(x1_sample, colors_x1):
        csv_wave = os.path.join(DATA_SRC, "波形", f"steady_x1_{x1_val}_sp_{sp}_n_{n}", "moc_timeseries.csv")
        if os.path.exists(csv_wave):
            df_w = pd.read_csv(csv_wave)
            t, H_wh = df_w['t'].values, df_w['H_wh'].values
            v, fs, ts, L = 1450.0, 1.0 / (t[1] - t[0]), 1.0, x1_val + n * sp + 500.0
            out = compute_moc_cepstrum(t, H_wh, v, fs=fs, ts=ts, wellbore_length=L, wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
            depth = out['depth']
            prof = -np.sum(out['C'], axis=1)
            
            # Offset coordinate delta x = depth - x1_val
            delta_x = depth - x1_val
            mask = (delta_x >= -20) & (delta_x <= 100)
            ax.plot(delta_x[mask], prof[mask], label=f"$X_1 = {x1_val}$ m", color=col, lw=1.0, alpha=0.9)
            
    # Mark true fracture relative positions
    for k in range(4):
        tx_rel = k * sp
        ax.axvline(tx_rel, color="#BDC3C7", ls="--", lw=0.6, zorder=0)
        ax.text(tx_rel, 3.45, f"$f_{k+1}$", color=COLOR_SLATE, fontsize=6.5, ha='center', va='bottom')
        
    ax.set_xlabel("Relative Offset $\\Delta x = x - X_1$ (m)")
    ax.set_ylabel("Accumulated Cepstrum $P_{2D}$ (a.u.)")
    ax.set_xlim(-20, 100)
    ax.set_ylim(-0.2, 4.2)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) Absolute Peaks P_i(X1) at S=20m, n=4 ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    
    df_n4_sp20 = df_metrics[(df_metrics['n_total'] == 4) & (df_metrics['spacing_m'] == 20.0)]
    
    for i in range(1, 5):
        sub_i = df_n4_sp20[df_n4_sp20['frac_idx'] == i].sort_values('x1')
        ax.plot(sub_i['x1'], sub_i['P_2d'], marker='o', label=f"Frac {i} ($P_{i}$)",
                color=PALETTE_FRACTURES[i-1], lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Primary Fracture Depth $X_1$ (m)")
    ax.set_ylabel("Peak Amplitude $P_i$ (a.u.)")
    ax.set_xlim(1800, 4700)
    ax.set_xticks([2000, 2500, 3000, 3500, 4000, 4500])
    # Headroom expansion to 4.8 to ensure ZERO overlap
    ax.set_ylim(0.0, 4.8)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.3)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) Relative Ratio Invariance alpha_i(X1) at S=20m, n=4 ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    for i in range(2, 5):
        sub_i = df_n4_sp20[df_n4_sp20['frac_idx'] == i].sort_values('x1')
        ax.plot(sub_i['x1'], sub_i['alpha_2d'], marker='s', label=f"$\\alpha_{i} = P_{i}/P_1$",
                color=PALETTE_FRACTURES[i-1], lw=1.1, markersize=3.6)
        
    ax.axhline(1.0, color="#BDC3C7", ls=":", lw=0.8)
    ax.set_xlabel("Primary Fracture Depth $X_1$ (m)")
    ax.set_ylabel("Relative Ratio $\\alpha_i = P_i/P_1$")
    ax.set_xlim(1800, 4700)
    ax.set_xticks([2000, 2500, 3000, 3500, 4000, 4500])
    # Headroom expansion to 1.35 to ensure ZERO overlap
    ax.set_ylim(0.0, 1.35)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_3_Depth_Main_Effect")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.3: {out_prefix}.png / .svg / .pdf")


def plot_figure_5_3_supp(df_steady, df_metrics):
    """
    Figure 5.3 Supplementary / Extended Depth Invariance Analysis:
    (a) P1 vs Depth X1 across Spacings S in {10, 20, 50, 100}m
    (b) Relative ratio alpha_2 vs Depth X1 across Spacings
    (c) Spatial Localization Error vs Depth X1
    """
    print("[3/3] Generating Figure 5.3 Supplementary (Depth Invariance across Spacings & Accuracy)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.35), constrained_layout=True)
    
    sel_spacings = [10, 20, 30, 50, 100]
    palette_sp = [COLOR_RED, COLOR_ORANGE, COLOR_TEAL, COLOR_BLUE_MAIN, COLOR_PURPLE]
    
    # --- (a) P1 vs Depth X1 across Spacings (n=4) ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    df_n4 = df_metrics[df_metrics['n_total'] == 4]
    
    for sp_val, col in zip(sel_spacings, palette_sp):
        sub_sp = df_n4[(df_n4['spacing_m'] == sp_val) & (df_n4['frac_idx'] == 1)].sort_values('x1')
        ax.plot(sub_sp['x1'], sub_sp['P_2d'], marker='o', label=f"$S = {sp_val}$ m",
                color=col, lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Primary Fracture Depth $X_1$ (m)")
    ax.set_ylabel("Primary Peak $P_1$ (a.u.)")
    ax.set_xlim(1800, 4700)
    ax.set_xticks([2000, 2500, 3000, 3500, 4000, 4500])
    ax.set_ylim(1.2, 4.8)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) alpha_2 vs Depth X1 across Spacings (n=4) ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    
    for sp_val, col in zip(sel_spacings, palette_sp):
        sub_sp = df_n4[(df_n4['spacing_m'] == sp_val) & (df_n4['frac_idx'] == 2)].sort_values('x1')
        ax.plot(sub_sp['x1'], sub_sp['alpha_2d'], marker='s', label=f"$S = {sp_val}$ m",
                color=col, lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Primary Fracture Depth $X_1$ (m)")
    ax.set_ylabel("Relative Ratio $\\alpha_2 = P_2/P_1$")
    ax.set_xlim(1800, 4700)
    ax.set_xticks([2000, 2500, 3000, 3500, 4000, 4500])
    ax.set_ylim(0.35, 0.75)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) Max Localization Error vs Depth X1 ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    # Filter valid within-wellbore fractures (x_true <= 5000m)
    df_valid = df_metrics[df_metrics['x_true'] <= 5000.0]
    mean_errs = df_valid.groupby('x1')['loc_error_m'].mean()
    max_errs = df_valid.groupby('x1')['loc_error_m'].max()
    x1_vals = mean_errs.index.values
    
    ax.plot(x1_vals, max_errs.values, marker='^', color=COLOR_RED, lw=1.1, markersize=3.8, label="Max Error")
    ax.plot(x1_vals, mean_errs.values, marker='o', color=COLOR_BLUE_MAIN, lw=1.1, markersize=3.8, label="Mean Error")
    
    ax.axhline(0.725, color="#BDC3C7", ls=":", lw=0.8, label="MOC Grid $\\Delta x$")
    ax.set_xlabel("Primary Fracture Depth $X_1$ (m)")
    ax.set_ylabel("Localization Error $|\\Delta x_{err}|$ (m)")
    ax.set_xlim(1800, 4700)
    ax.set_xticks([2000, 2500, 3000, 3500, 4000, 4500])
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_3_Supp_Depth_Invariance")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.3 Supp: {out_prefix}.png / .svg / .pdf")


if __name__ == "__main__":
    df_steady, df_metrics = extract_depth_metrics()
    plot_figure_5_3(df_steady, df_metrics)
    plot_figure_5_3_supp(df_steady, df_metrics)
