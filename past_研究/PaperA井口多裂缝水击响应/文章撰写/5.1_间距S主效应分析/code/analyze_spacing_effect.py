"""
Analysis and Plotting Script for Section 5.1: Fracture Spacing S Main Effect
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
COLOR_GREY      = "#7F8C8D"
PALETTE_FRACTURES = ["#1B4F72", "#117864", "#D35400", "#900C3F", "#6C3483", "#2874A6", "#B7950B", "#5D6D7E"]

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")
SEC_DIR  = os.path.join(BASE_DIR, "文章撰写", "5.1_间距S主效应分析")
FIG_DIR  = os.path.join(SEC_DIR, "figures")
TAB_DIR  = os.path.join(SEC_DIR, "data")


def extract_detailed_metrics():
    """
    Extracts peak metrics, noise floor, PSR, and localization errors for X1=3000m, n in {2, 4, 8}, S in 10..100m.
    """
    print("[1/3] Extracting metrics and analyzing noise floors across all 30 spacing cases...")
    
    csv_table = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
    df_decay = pd.read_csv(csv_table)
    
    records = []
    x1 = 3000.0
    spacings = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_list = [2, 4, 8]
    
    for n in n_list:
        for sp in spacings:
            # Theoretical fracture coordinates
            true_coords = [x1 + k * sp for k in range(n)]
            
            # Load waveform if available to calculate background coda floor and PSR
            csv_wave = os.path.join(DATA_SRC, "波形", f"steady_x1_{int(x1)}_sp_{sp}_n_{n}", "moc_timeseries.csv")
            if os.path.exists(csv_wave):
                df_w = pd.read_csv(csv_wave)
                t, H_wh = df_w['t'].values, df_w['H_wh'].values
                v, fs, ts, L = 1450.0, 1.0 / (t[1] - t[0]), 1.0, x1 + n * sp + 500.0
                out = compute_moc_cepstrum(t, H_wh, v, fs=fs, ts=ts, wellbore_length=L, wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
                depth = out['depth']
                prof = -np.sum(out['C'], axis=1)
                
                # Identify coda noise zone (beyond the last fracture + 50m to L)
                coda_mask = (depth >= true_coords[-1] + 60.0) & (depth <= true_coords[-1] + 300.0)
                if np.any(coda_mask):
                    mu_coda = float(np.mean(prof[coda_mask]))
                    sigma_coda = float(np.std(prof[coda_mask])) + 1e-6
                else:
                    mu_coda, sigma_coda = 0.0, 0.01
            else:
                mu_coda, sigma_coda = 0.0, 0.01
                
            # Filter from decay_table
            sub = df_decay[(df_decay['friction_model'] == 'steady') & 
                           (df_decay['x1'] == x1) & 
                           (df_decay['n_total'] == n) & 
                           (df_decay['spacing_m'] == float(sp))].sort_values('frac_idx')
            
            p1_val = sub[sub['frac_idx'] == 1]['P_2d'].values[0] if len(sub) > 0 else 1.0
            
            for idx, row in sub.iterrows():
                f_idx = int(row['frac_idx'])
                p2d = float(row['P_2d'])
                p1d = float(row['P_1d'])
                alpha2d = float(row['alpha_2d'])
                x_f = float(row['x_f'])
                tx = true_coords[f_idx - 1]
                loc_err = abs(x_f - tx)
                psr_db = 20.0 * np.log10(max(p2d - mu_coda, 1e-4) / sigma_coda) if sigma_coda > 0 else np.nan
                
                records.append({
                    'friction_model': 'steady',
                    'x1': x1,
                    'n_total': n,
                    'spacing_m': sp,
                    'frac_idx': f_idx,
                    'x_true': tx,
                    'x_extracted': x_f,
                    'loc_error_m': loc_err,
                    'P_1d': p1d,
                    'P_2d': p2d,
                    'alpha_2d': alpha2d,
                    'mu_coda': mu_coda,
                    'sigma_coda': sigma_coda,
                    'PSR_dB': psr_db,
                    'tau_s_ms': 2.0 * sp / 1450.0 * 1000.0,
                })
                
    df_out = pd.DataFrame(records)
    out_csv = os.path.join(TAB_DIR, "spacing_effect_metrics.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[Done] Metrics extracted and saved to: {out_csv} ({len(df_out)} rows)")
    return df_out


def plot_figure_5_1(df_metrics):
    """
    Figure 5.1 Main Publication Figure:
    3-Panel (Double column 7.2 x 2.35 in, aspect 4:5):
    (a) Spectral Profiles at X1=3000m, n=4 across S in {10, 30, 50, 100}m
    (b) Sub-fracture peak trajectories P_i(S)
    (c) Relative energy ratios alpha_i(S)
    """
    print("[2/3] Generating Figure 5.1 (Main Spacing Effect Figure)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.35), constrained_layout=True)
    
    # --- (a) Spectral Profiles ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    spacings = [10, 30, 50, 100]
    colors_prof = [COLOR_RED, COLOR_ORANGE, COLOR_TEAL, COLOR_BLUE_MAIN]
    
    for sp, col in zip(spacings, colors_prof):
        csv_wave = os.path.join(DATA_SRC, "波形", f"steady_x1_3000_sp_{sp}_n_4", "moc_timeseries.csv")
        if os.path.exists(csv_wave):
            df_w = pd.read_csv(csv_wave)
            t, H_wh = df_w['t'].values, df_w['H_wh'].values
            v, fs, ts, L = 1450.0, 1.0 / (t[1] - t[0]), 1.0, 3000.0 + 4 * sp + 500.0
            out = compute_moc_cepstrum(t, H_wh, v, fs=fs, ts=ts, wellbore_length=L, wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
            depth = out['depth']
            prof = -np.sum(out['C'], axis=1)
            
            mask = (depth >= 2980) & (depth <= 3350)
            ax.plot(depth[mask], prof[mask], label=f"$S = {sp}$ m", color=col, lw=1.0, alpha=0.9)
            
    # Mark true fracture positions for S=50m
    true_x = [3000, 3050, 3100, 3150]
    for k, tx in enumerate(true_x):
        ax.axvline(tx, color="#BDC3C7", ls="--", lw=0.6, zorder=0)
        ax.text(tx, 3.45, f"$f_{k+1}$", color=COLOR_SLATE, fontsize=6.5, ha='center', va='bottom')
        
    ax.set_xlabel("Apparent Depth $x$ (m)")
    ax.set_ylabel("Accumulated Cepstrum $P_{2D}$ (a.u.)")
    ax.set_xlim(2980, 3350)
    ax.set_ylim(-0.2, 3.8)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) Sub-fracture Peak Trajectories P_i(S) ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    df_n4 = df_metrics[df_metrics['n_total'] == 4]
    
    for i in range(1, 5):
        df_i = df_n4[df_n4['frac_idx'] == i].sort_values('spacing_m')
        ax.plot(df_i['spacing_m'], df_i['P_2d'], marker='o', label=f"Frac {i} ($P_{i}$)",
                color=PALETTE_FRACTURES[i-1], lw=1.1, markersize=3.6)
        
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Peak Amplitude $P_i$ (a.u.)")
    ax.set_xlim(5, 105)
    ax.set_ylim(0, 4.8)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.3)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) Relative Ratios alpha_i(S) ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    for i in range(2, 5):
        df_i = df_n4[df_n4['frac_idx'] == i].sort_values('spacing_m')
        ax.plot(df_i['spacing_m'], df_i['alpha_2d'], marker='s', label=f"$\\alpha_{i} = P_{i}/P_1$",
                color=PALETTE_FRACTURES[i-1], lw=1.1, markersize=3.6)
        
    ax.axhline(1.0, color="#BDC3C7", ls=":", lw=0.8)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Relative Ratio $\\alpha_i = P_i/P_1$")
    ax.set_xlim(5, 105)
    ax.set_ylim(0, 1.35)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_1_Spacing_Main_Effect")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.1: {out_prefix}.png / .svg / .pdf")


def plot_figure_5_1_supp(df_metrics):
    """
    Figure 5.1 Supplementary / Extended Multiplicity Comparison:
    3-Panel comparing Spacing Effect across n_total in {2, 4, 8}:
    (a) n_total = 2: P1, P2 vs S
    (b) n_total = 4: P1..P4 vs S
    (c) n_total = 8: P1, P2, P4, P6, P8 vs S
    """
    print("[3/3] Generating Figure 5.1 Supplementary (Multiplicity Comparison across S)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.35), constrained_layout=True)
    
    # --- (a) n=2 ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    df_n2 = df_metrics[df_metrics['n_total'] == 2]
    for i in [1, 2]:
        sub_i = df_n2[df_n2['frac_idx'] == i].sort_values('spacing_m')
        ax.plot(sub_i['spacing_m'], sub_i['P_2d'], marker='o', label=f"Frac {i} ($P_{i}$)",
                color=PALETTE_FRACTURES[i-1], lw=1.1, markersize=3.6)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Peak Amplitude $P_i$ (a.u.)")
    ax.set_xlim(5, 105)
    ax.set_ylim(0, 6.4)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.25)
    ax.set_title("(a) $n_{total} = 2$", loc="left", fontsize=9.0, fontweight="bold", pad=5)
    
    # --- (b) n=4 ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    df_n4 = df_metrics[df_metrics['n_total'] == 4]
    for i in range(1, 5):
        sub_i = df_n4[df_n4['frac_idx'] == i].sort_values('spacing_m')
        ax.plot(sub_i['spacing_m'], sub_i['P_2d'], marker='o', label=f"Frac {i} ($P_{i}$)",
                color=PALETTE_FRACTURES[i-1], lw=1.1, markersize=3.6)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Peak Amplitude $P_i$ (a.u.)")
    ax.set_xlim(5, 105)
    ax.set_ylim(0, 6.4)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.25)
    ax.set_title("(b) $n_{total} = 4$", loc="left", fontsize=9.0, fontweight="bold", pad=5)
    
    # --- (c) n=8 ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    df_n8 = df_metrics[df_metrics['n_total'] == 8]
    sel_indices = [1, 2, 4, 6, 8]
    for k, i in enumerate(sel_indices):
        sub_i = df_n8[df_n8['frac_idx'] == i].sort_values('spacing_m')
        ax.plot(sub_i['spacing_m'], sub_i['P_2d'], marker='o', label=f"Frac {i} ($P_{i}$)",
                color=PALETTE_FRACTURES[k], lw=1.1, markersize=3.6)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Peak Amplitude $P_i$ (a.u.)")
    ax.set_xlim(5, 105)
    # Headroom expansion to 6.4 provides generous, pristine whitespace
    ax.set_ylim(0, 6.4)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.25)
    ax.set_title("(c) $n_{total} = 8$", loc="left", fontsize=9.0, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_5_1_Supp_Multiplicity_Spacing")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.1 Supp: {out_prefix}.png / .svg / .pdf")


if __name__ == "__main__":
    df_metrics = extract_detailed_metrics()
    plot_figure_5_1(df_metrics)
    plot_figure_5_1_supp(df_metrics)
