"""Forward evidence matrix for the four requested engineering geometries.

This figure deliberately does *not* plot the old target-equalized curve.  The
previous implementation divided each local peak into 3.12 a.u. and supplied
the true fracture coordinates to the mask, which is an oracle reconstruction
and cannot support a blind diagnostic claim.  The replacement reports the
stored MOC/cepstrum profile, nominal perforation priors, predicted
non-perforation path depths and a finite transmission-sensitivity proxy.
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

NATURE_FIGURE_SCRIPTS = r"C:\Users\Change\.agents\skills\nature-figure\scripts"
if NATURE_FIGURE_SCRIPTS not in sys.path:
    sys.path.insert(0, NATURE_FIGURE_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment

# --- Matplotlib rcParams Configuration: Strict SPE Journal Specification ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.8,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.75,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.8,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "legend.frameon": False,
    "figure.dpi": 300,
})

# Curated Color Palettes
COLOR_RAW       = "#D35400"
COLOR_CORRECTED = "#117864"
COLOR_TEAL      = "#117864"
COLOR_ORANGE    = "#D35400"
COLOR_RED       = "#900C3F"
COLOR_GHOST     = "#900C3F"
COLOR_SLATE     = "#2C3E50"
COLOR_BLUE_MAIN = "#1B4F72"
COLOR_GREY      = "#7F8C8D"
COLOR_GREEN_BG  = "#D4EFDF"
COLOR_YELLOW_BG = "#FCF3CF"
COLOR_RED_BG    = "#FADBD8"

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


def prior_geometry(x1, S_nom, n_nom, x_max):
    """Return nominal design depths and non-perforation path-depth candidates."""
    x_perf = x1 + np.arange(n_nom, dtype=float) * S_nom
    ghosts = []
    for m in range(2, int(np.floor((x_max - x1) / S_nom)) + 1):
        x_g = x1 + m * S_nom
        if not np.any(np.isclose(x_g, x_perf, atol=1e-6)):
            ghosts.append(x_g)
    return x_perf, ghosts


def transmission_proxy(n_nom, T_eff=0.784):
    """Finite forward transmission factor used only as a sensitivity proxy."""
    return np.power(T_eff, 2.0 * np.arange(n_nom, dtype=float))


def plot_multi_case_matrix():
    """
    Plots the publication-grade 4-scenario validation matrix (2x2 grid layout).
    """
    print("[1/1] Plotting Multi-Case Physics-Informed Correction Matrix...")
    cases_cfg = [
        {
            'id': '(a) Case 1: Phase Cancellation & Ghost Echoes ($S = 30$ m, $n = 4$)',
            'fn': 'steady_x1_3000_sp_30_n_4.npz',
            'S': 30.0, 'n': 4,
            'xlim': (2985, 3175),
            'desc': 'Forward trap: $P_3,P_4$ are attenuated\n• Candidate paths at 3120 and 3150 m\n• No blind recovery claimed'
        },
        {
            'id': '(b) Case 2: Sub-Resolution Spacing & Coalescence ($S = 10$ m, $n = 4$)',
            'fn': 'steady_x1_3000_sp_10_n_4.npz',
            'S': 10.0, 'n': 4,
            'xlim': (2995, 3065),
            'desc': 'Sub-resolution spacing\n• Nominal windows overlap\n• De-aliasing is unresolved here'
        },
        {
            'id': '(c) Case 3: Deep Multi-Interface Depletion ($S = 20$ m, $n = 8$)',
            'fn': 'steady_x1_3000_sp_20_n_8.npz',
            'S': 20.0, 'n': 8,
            'xlim': (2985, 3165),
            'desc': 'Deep cascade depletion\n• Finite transmission proxy decays with $k$\n• Late-cluster activity is low-information'
        },
        {
            'id': '(d) Case 4: High-Spacing Raw Baseline (S = 80 m, n = 4)',
            'fn': 'steady_x1_3000_sp_80_n_4.npz',
            'S': 80.0, 'n': 4,
            'xlim': (2970, 3270),
            'desc': 'High-spacing raw baseline\n• Raw peaks are separated\n• Forward evidence only; no recovery claim'
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.50), constrained_layout=False)
    # Reserve a dedicated top strip for the shared legend; this prevents it
    # from colliding with the long case titles at manuscript size.
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.095, top=0.84,
                        wspace=0.16, hspace=0.30)
    axes_flat = axes.flatten()
    
    for ax, cfg in zip(axes_flat, cases_cfg):
        depth, prof_raw, _xf = load_moc_profile(cfg['fn'])
        
        x_min, x_max = cfg['xlim']
        mask = (depth >= x_min) & (depth <= x_max)
        d = depth[mask]
        x_perf, ghost_locations = prior_geometry(3000.0, cfg['S'], cfg['n'], x_max)
        conf = transmission_proxy(cfg['n'])

        ax.plot(d, prof_raw[mask], color=COLOR_RAW, lw=1.0,
                label="Raw MOC/cepstrum profile", zorder=3)
        ax.axhline(0.20, color=COLOR_GREY, ls=":", lw=0.6, label="Noise Floor ($0.20$ a.u.)", zorder=1)
        win_span = max(min(cfg['S'] * 0.22, 6.0), 1.2)
        for k, tx in enumerate(x_perf):
            band_color = COLOR_TEAL if conf[k] >= 0.70 else ("#F1C40F" if conf[k] >= 0.20 else "#AAB7B8")
            ax.axvspan(tx - win_span, tx + win_span, color=band_color, alpha=0.14, zorder=0)
            ax.axvline(tx, color=band_color, lw=0.65, ls="--", zorder=1)
            ax.text(tx, 3.30, f"$x_{{p,{k+1}}}$", fontsize=5.8, color=COLOR_SLATE,
                    ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.85))
        for x_g in ghost_locations:
            if x_min <= x_g <= x_max:
                ax.axvspan(x_g - 5.5, x_g + 5.5, color=COLOR_RED, alpha=0.10, zorder=0)
                ax.axvline(x_g, color=COLOR_GHOST, lw=0.7, ls=":", zorder=1)
                ax.text(x_g, 0.40, "path\ncandidate", fontsize=5.0, color=COLOR_GHOST, ha='center', va='bottom', style='italic',
                        bbox=dict(boxstyle="round,pad=0.06", fc="#FDEDEC", ec=COLOR_GHOST, lw=0.4))
                
        # Layout and Typography
        ax.set_title(cfg['id'], loc="left", fontsize=8.0, fontweight="bold", pad=3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.30, 4.40)
        ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
        ax.set_ylabel("Raw cepstrum $P_{2D}$ (a.u.)", fontsize=7.5)
        ax.set_xlabel("Apparent Wellbore Depth $x$ (m)", fontsize=7.8)
        
        # Case summary note inside subplot (Clean compact upper right box)
        ax.text(0.98, 0.96, cfg['desc'], transform=ax.transAxes, fontsize=5.5, color=COLOR_SLATE,
                ha='right', va='top', bbox=dict(boxstyle="round,pad=0.20", fc="#F8F9F9", ec="#BDC3C7", lw=0.5))
        
    # Shared Top Legend across the entire figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    # Add proxy handles for nominal prior and predicted path candidates.
    patch_true = mpl.patches.Patch(facecolor=COLOR_TEAL, alpha=0.18, label="Nominal prior (high k-proxy)")
    patch_amb = mpl.patches.Patch(facecolor="#F1C40F", alpha=0.18, label="Nominal prior (ambiguous k-proxy)")
    patch_low = mpl.patches.Patch(facecolor="#AAB7B8", alpha=0.18, label="Nominal prior (low-information)")
    patch_ghost = mpl.patches.Patch(facecolor=COLOR_RED, alpha=0.15, label="Non-perforation path candidate")
    all_handles = handles + [patch_true, patch_amb, patch_low, patch_ghost]
    all_labels = labels + ["Nominal prior (high k-proxy)", "Nominal prior (ambiguous k-proxy)", "Nominal prior (low-information)", "Non-perforation path candidate"]
    
    fig.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=5, fontsize=6.0, frameon=False, handlelength=1.2)

    # Render-time gate required by the Nature figure workflow.  The measured
    # axes rectangles are written alongside the figure for independent QA.
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(FIG_DIR, "Figure_3_4_MultiCase_Physics_Correction_Matrix.panel-alignment.json"),
        require_panel_labels=False,
    )
        
    out_png = os.path.join(FIG_DIR, "Figure_3_4_MultiCase_Physics_Correction_Matrix.png")
    out_svg = os.path.join(FIG_DIR, "Figure_3_4_MultiCase_Physics_Correction_Matrix.svg")
    out_pdf = os.path.join(FIG_DIR, "Figure_3_4_MultiCase_Physics_Correction_Matrix.pdf")
    
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Multi-Case Matrix Figure: {out_png} / .svg / .pdf")


if __name__ == "__main__":
    plot_multi_case_matrix()
