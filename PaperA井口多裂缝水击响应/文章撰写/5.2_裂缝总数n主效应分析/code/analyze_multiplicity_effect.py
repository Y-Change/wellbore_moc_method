"""
Analysis and Plotting Script for Section 5.2: Fracture Multiplicity n_total Main Effect
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

PALETTE_N = ["#1B4F72", "#117864", "#D35400", "#900C3F", "#6C3483", "#2874A6", "#B7950B", "#5D6D7E"]

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")
SEC_DIR  = os.path.join(BASE_DIR, "文章撰写", "5.2_裂缝总数n主效应分析")
FIG_DIR  = os.path.join(SEC_DIR, "figures")
TAB_DIR  = os.path.join(SEC_DIR, "data")


def extract_multiplicity_metrics():
    """
    Extracts metrics for n_total in 1..8 across spacings at X1=3000m.
    Computes P1, P_end, R_end, sum_P, and decay parameters.
    """
    print("[1/3] Extracting metrics across all n_total in 1..8 and spacings...")
    csv_table = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
    df = pd.read_csv(csv_table)
    df_sub = df[(df['friction_model'] == 'steady') & (df['x1'] == 3000.0)]
    
    summary_records = []
    for (n, sp), group in df_sub.groupby(['n_total', 'spacing_m']):
        group_sorted = group.sort_values('frac_idx')
        p1 = group_sorted[group_sorted['frac_idx'] == 1]['P_2d'].values[0]
        pn = group_sorted[group_sorted['frac_idx'] == n]['P_2d'].values[0]
        sum_p = group_sorted['P_2d'].sum()
        mean_p = group_sorted['P_2d'].mean()
        r_end = pn / p1
        
        # Exponential fit P_i = P_1 * exp(-b * (i-1))
        idxs = group_sorted['frac_idx'].values
        p_vals = group_sorted['P_2d'].values
        if len(idxs) >= 2:
            # log(P_i/P_1) = -b * (i-1)
            y = np.log(np.maximum(p_vals / p1, 1e-4))
            x = idxs - 1
            # Linear regression without intercept: y = -b * x
            b_fit = -float(np.sum(x * y) / np.sum(x**2)) if np.sum(x**2) > 0 else 0.0
        else:
            b_fit = 0.0
            
        summary_records.append({
            'friction_model': 'steady',
            'x1': 3000.0,
            'n_total': int(n),
            'spacing_m': float(sp),
            'P_1': p1,
            'P_end': pn,
            'R_end': r_end,
            'sum_P': sum_p,
            'mean_P': mean_p,
            'b_topological': b_fit,
        })
        
    df_metrics = pd.DataFrame(summary_records)
    out_csv = os.path.join(TAB_DIR, "multiplicity_effect_metrics.csv")
    df_metrics.to_csv(out_csv, index=False)
    print(f"[Done] Multiplicity metrics saved to: {out_csv} ({len(df_metrics)} rows)")
    return df_sub, df_metrics


def plot_figure_5_2(df_sub, df_metrics):
    """
    Figure 5.2 Main Publication Figure:
    3-Panel (Double column 7.2 x 2.35 in, aspect 4:5):
    (a) Spectral Profiles for n_total in {1, 2, 4, 6, 8} at S=20m
    (b) First Fracture Peak P1(n_total) across Spacings S in {10, 20, 30, 50, 100}m
    (c) End-member Relative Ratio R_end = P_n / P_1 vs n_total across Spacings
    """
    print("[2/3] Generating Figure 5.2 (Main Multiplicity Effect Figure)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.35), constrained_layout=True)
    
    # --- (a) Spectral Profiles at X1=3000m, S=20m for n in {1, 2, 4, 8} ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    
    n_sample = [1, 2, 4, 8]
    colors_n = [COLOR_SLATE, COLOR_TEAL, COLOR_ORANGE, COLOR_RED]
    sp = 20
    
    for n, col in zip(n_sample, colors_n):
        csv_wave = os.path.join(DATA_SRC, "波形", f"steady_x1_3000_sp_{sp}_n_{n}", "moc_timeseries.csv")
        if os.path.exists(csv_wave):
            df_w = pd.read_csv(csv_wave)
            t, H_wh = df_w['t'].values, df_w['H_wh'].values
            v, fs, ts, L = 1450.0, 1.0 / (t[1] - t[0]), 1.0, 3000.0 + n * sp + 500.0
            out = compute_moc_cepstrum(t, H_wh, v, fs=fs, ts=ts, wellbore_length=L, wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
            depth = out['depth']
            prof = -np.sum(out['C'], axis=1)
            
            mask = (depth >= 2980) & (depth <= 3200)
            ax.plot(depth[mask], prof[mask], label=f"$n_{{tot}} = {n}$", color=col, lw=1.0, alpha=0.9)
            
    ax.set_xlabel("Apparent Depth $x$ (m)")
    ax.set_ylabel("Accumulated Cepstrum $P_{2D}$ (a.u.)")
    ax.set_xlim(2980, 3200)
    # Headroom expansion for legend
    ax.set_ylim(-0.2, 6.2)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) Primary Peak P1 vs n_total across Spacings ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    
    sel_spacings = [10, 20, 30, 50, 100]
    palette_sp = [COLOR_RED, COLOR_ORANGE, COLOR_TEAL, COLOR_BLUE_MAIN, COLOR_PURPLE]
    
    for sp_val, col in zip(sel_spacings, palette_sp):
        sub_sp = df_metrics[df_metrics['spacing_m'] == sp_val].sort_values('n_total')
        ax.plot(sub_sp['n_total'], sub_sp['P_1'], marker='o', label=f"$S = {sp_val}$ m",
                color=col, lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Fracture Multiplicity $n_{total}$")
    ax.set_ylabel("Primary Peak $P_1$ (a.u.)")
    ax.set_xlim(0.5, 8.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    # Headroom expansion
    ax.set_ylim(1.8, 6.4)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) End-member Relative Ratio R_end = P_n / P_1 vs n_total ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    for sp_val, col in zip(sel_spacings, palette_sp):
        sub_sp = df_metrics[df_metrics['spacing_m'] == sp_val].sort_values('n_total')
        ax.plot(sub_sp['n_total'], sub_sp['R_end'], marker='s', label=f"$S = {sp_val}$ m",
                color=col, lw=1.1, markersize=3.6)
        
    ax.axhline(0.0, color="#BDC3C7", ls=":", lw=0.8)
    ax.set_xlabel("Fracture Multiplicity $n_{total}$")
    ax.set_ylabel("End-Member Ratio $R_{end} = P_n/P_1$")
    ax.set_xlim(0.5, 8.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    # Headroom expansion
    ax.set_ylim(0.0, 1.35)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_2_Multiplicity_Main_Effect")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.2: {out_prefix}.png / .svg / .pdf")


def plot_figure_5_2_supp(df_sub, df_metrics):
    """
    Figure 5.2 Supplementary / Extended Multiplicity Analysis:
    (a) Total Detected Peak Sum \Sigma P vs n_total across Spacings
    (b) Sub-fracture decay trajectories P_i at S=20m for n in {2, 4, 6, 8}
    (c) Topological decay exponent b vs Spacing across n_total
    """
    print("[3/3] Generating Figure 5.2 Supplementary (Cumulative Energy and Decay Rates)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.35), constrained_layout=True)
    
    sel_spacings = [10, 20, 30, 50, 100]
    palette_sp = [COLOR_RED, COLOR_ORANGE, COLOR_TEAL, COLOR_BLUE_MAIN, COLOR_PURPLE]
    
    # --- (a) Total Detected Energy Sum vs n_total ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    for sp_val, col in zip(sel_spacings, palette_sp):
        sub_sp = df_metrics[df_metrics['spacing_m'] == sp_val].sort_values('n_total')
        ax.plot(sub_sp['n_total'], sub_sp['sum_P'], marker='o', label=f"$S = {sp_val}$ m",
                color=col, lw=1.1, markersize=3.6)
    ax.set_xlabel("Fracture Multiplicity $n_{total}$")
    ax.set_ylabel("Total Cumulative Peak $\\sum P_i$ (a.u.)")
    ax.set_xlim(0.5, 8.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_ylim(2.5, 12.5)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) P_i decay curves for n_total in {2, 4, 6, 8} at S=20m ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    n_sample = [2, 4, 6, 8]
    colors_n = [COLOR_TEAL, COLOR_ORANGE, COLOR_PURPLE, COLOR_RED]
    
    for n_val, col in zip(n_sample, colors_n):
        sub_n = df_sub[(df_sub['spacing_m'] == 20.0) & (df_sub['n_total'] == n_val)].sort_values('frac_idx')
        ax.plot(sub_n['frac_idx'], sub_n['P_2d'], marker='o', label=f"$n_{{tot}} = {n_val}$",
                color=col, lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Fracture Index $\\mathrm{idx} = i$")
    ax.set_ylabel("Peak Amplitude $P_i$ (a.u.)")
    ax.set_xlim(0.5, 8.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    # Headroom expansion to 5.8 to ensure ZERO overlap
    ax.set_ylim(0.0, 5.8)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) Empirical index-envelope slope b vs n_total ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    for sp_val, col in zip(sel_spacings, palette_sp):
        sub_sp = df_metrics[(df_metrics['spacing_m'] == sp_val) & (df_metrics['n_total'] >= 2)].sort_values('n_total')
        ax.plot(sub_sp['n_total'], sub_sp['b_topological'], marker='^', label=f"$S = {sp_val}$ m",
                color=col, lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Fracture Multiplicity $n_{total}$")
    ax.set_ylabel(r"Empirical envelope slope $b$")
    ax.set_xlim(1.5, 8.5)
    ax.set_xticks([2, 3, 4, 5, 6, 7, 8])
    # Headroom expansion to 1.15 to ensure ZERO overlap with legend
    ax.set_ylim(0.15, 1.15)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_2_Supp_Cumulative_Energy_Decay")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.2 Supp: {out_prefix}.png / .svg / .pdf")


if __name__ == "__main__":
    df_sub, df_metrics = extract_multiplicity_metrics()
    plot_figure_5_2(df_sub, df_metrics)
    plot_figure_5_2_supp(df_sub, df_metrics)
