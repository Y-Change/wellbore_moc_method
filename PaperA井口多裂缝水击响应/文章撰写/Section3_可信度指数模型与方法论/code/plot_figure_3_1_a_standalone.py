"""
Standalone 4-Row 1-Column Plot for Section 3: Representative Waveform Archetypes
Author: Antigravity (Pair Programming with User)
Target: SPE Journal - Multi-Cluster Water-Hammer Diagnostics
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

# --- Matplotlib rcParams Configuration: Strict Paper A & SPE Journal Specification ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
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
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "legend.frameon": False,
    "figure.dpi": 300,
})

# Curated Color Palettes
COLOR_BLUE_MAIN = "#1B4F72"
COLOR_TEAL      = "#117864"
COLOR_ORANGE    = "#D35400"
COLOR_RED       = "#900C3F"
COLOR_PURPLE    = "#6C3483"
COLOR_SLATE     = "#2C3E50"
COLOR_GREY      = "#7F8C8D"

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
SEC_DIR  = os.path.join(BASE_DIR, "文章撰写", "Section3_可信度指数模型与方法论")
FIG_DIR  = os.path.join(SEC_DIR, "figures")
DATA_BASE = os.path.join(BASE_DIR, "01_几何网格")
NPZ_DIRNAME = [d for d in os.listdir(DATA_BASE) if "npz" in d][0]
NPZ_DIR = os.path.join(DATA_BASE, NPZ_DIRNAME)


def load_moc_profile(fn):
    """Loads MOC simulation from npz and computes 2D marginal cepstrum profile along depth x."""
    p = os.path.join(NPZ_DIR, fn)
    data = np.load(p)
    t = data['t_sim']
    H_wh = data['H_wh']
    v = float(data['v'][0]) if hasattr(data['v'], '__len__') else float(data['v'])
    fs = float(data['fs'][0]) if hasattr(data['fs'], '__len__') else float(data['fs'])
    ts = float(data['ts'][0]) if hasattr(data['ts'], '__len__') else float(data['ts'])
    L = float(data['L'][0]) if hasattr(data['L'], '__len__') else float(data['L'])
    x_f_aligned = data['x_f_aligned']
    
    out = compute_moc_cepstrum(t, H_wh, v, fs=fs, ts=ts, wellbore_length=L, wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
    depth = out['depth']
    prof = -np.sum(out['C'], axis=1)
    return depth, prof, x_f_aligned


def plot_waveform_archetypes_4x1():
    """
    Plots the 4 representative waveform archetypes in a 4-row 1-column format
    with complete upper and lower boundaries and strict SPE Journal typography.
    """
    print("[1/1] Plotting Standalone 4x1 Waveform Archetype Figure...")
    
    cases = [
        {
            'panel': '(a)',
            'fn': 'steady_x1_3000_sp_80_n_3.npz',
            'title': 'Isolated decaying peaks ($S = 80$ m, $n_{total} = 3$)',
            'desc': 'Later peaks recovered and spatially separated',
            'color': COLOR_TEAL,
            'ylim': (-0.35, 3.85),
            'yticks': [0.0, 1.0, 2.0, 3.0]
        },
        {
            'panel': '(b)',
            'fn': 'steady_x1_3000_sp_10_n_4.npz',
            'title': 'Peak coalescence ($S = 10$ m, $n_{total} = 4$)',
            'desc': 'Densest spacing on the grid:\nadjacent peaks begin to overlap',
            'color': COLOR_SLATE,
            'ylim': (-0.35, 3.85),
            'yticks': [0.0, 1.0, 2.0, 3.0]
        },
        {
            'panel': '(c)',
            'fn': 'steady_x1_3000_sp_30_n_4.npz',
            'title': 'Later-peak suppression ($S = 30$ m, $n_{total} = 4$)',
            'desc': 'Same mid-spacing depression as Sec. 3.2:\n3rd and 4th apparent peaks are low',
            'color': COLOR_ORANGE,
            'ylim': (-0.35, 3.95),
            'yticks': [0.0, 1.0, 2.0, 3.0]
        },
        {
            'panel': '(d)',
            'fn': 'steady_x1_3000_sp_20_n_8.npz',
            'title': 'Weak tail peaks ($S = 20$ m, $n_{total} = 8$)',
            'desc': 'After many interfaces, tail $P_i$ are small;\nnot a proven physical blind zone',
            'color': COLOR_RED,
            'ylim': (-0.35, 4.25),
            'yticks': [0.0, 1.0, 2.0, 3.0, 4.0]
        }
    ]
    
    fig, axes = plt.subplots(4, 1, figsize=(5.20, 5.80), sharex=True, constrained_layout=True)
    
    for ax, c in zip(axes, cases):
        depth, prof, x_f = load_moc_profile(c['fn'])
        mask = (depth >= 2960) & (depth <= 3320)
        
        # Plot full waveform with complete bounds
        ax.plot(depth[mask], prof[mask], color=c['color'], lw=1.1, label="Cepstrum $P_{2D}(x)$", zorder=3)
        
        # Zero baseline
        ax.axhline(0.0, color="#D5D8DC", lw=0.6, ls="-", zorder=1)
        
        # Noise floor reference line
        ax.axhline(0.20, color=COLOR_GREY, lw=0.75, ls=":", zorder=2)
        ax.text(3315, 0.30, "ref. $0.20$ a.u.", fontsize=5.8, color=COLOR_GREY, ha="right", va="bottom")
        
        # Mark true fracture positions and place fk right above local peak
        for k, tx in enumerate(x_f):
            ax.axvline(tx, color="#BDC3C7", ls="--", lw=0.65, zorder=1)
            # Find actual local peak near tx
            local_mask = (depth >= tx - 5.0) & (depth <= tx + 5.0)
            local_peak = np.max(prof[local_mask]) if np.any(local_mask) else 0.5
            # Label anchored above peak tip with clearance
            y_label = min(local_peak + 0.35, c['ylim'][1] - 0.35)
            ax.text(tx, y_label, f"$f_{{{k+1}}}$", fontsize=6.2, color=COLOR_SLATE, 
                    ha="center", va="bottom", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85))
            
        ax.set_xlim(2960, 3325)
        ax.set_ylim(c['ylim'])
        ax.set_yticks(c['yticks'])
        ax.set_ylabel("$P_{2D}$ (a.u.)", fontsize=8.0)
        
        # Subplot Title with bold panel letter on top left
        ax.set_title(f"{c['panel']} {c['title']}", loc="left", fontsize=8.3, fontweight="bold", pad=4)
        
        # Descriptive subtitle strictly in the right-hand blank region (x >= 3180m)
        ax.text(3315, c['ylim'][1] * 0.90, c['desc'], fontsize=5.8, color=COLOR_SLATE,
                ha="right", va="top", style="italic")
        
    axes[-1].set_xlabel("Apparent Depth $x$ (m)", fontsize=8.5)
    
    out_png = os.path.join(FIG_DIR, "Figure_3_1_a_Waveform_Archetypes_4x1.png")
    out_svg = os.path.join(FIG_DIR, "Figure_3_1_a_Waveform_Archetypes_4x1.svg")
    out_pdf = os.path.join(FIG_DIR, "Figure_3_1_a_Waveform_Archetypes_4x1.pdf")
    
    fig.savefig(out_png, dpi=400)
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved 4x1 Figure: {out_png} / .svg / .pdf")


if __name__ == "__main__":
    plot_waveform_archetypes_4x1()
