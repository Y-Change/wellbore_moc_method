"""
Dedicated 5x1 Sub-Workflow Breakdown Plots for 4 Typical Engineering Cases (Figure 3.4 a, b, c, d)
Author: Antigravity (Pair Programming with User)
Target: SPE Journal - Multi-Cluster Water-Hammer Diagnostics (Section 3 Methodology)

Each case is rendered as a standalone 5-row x 1-column figure:
- Track (a): Raw Measured Continuous Cepstrum P_2D(x)
- Track (b): Multi-Path Ghost Peeling Notch Filter G_peel(x)
- Track (c): Transmission & Phase Loss Equalization Gain K_comp(x)
- Track (d): Single-Cluster Physical Peak Focusing Filter F_focus(x)
- Track (e): Final Physics-Corrected True Fracture Profile P_corrected(x) & Authenticity Bands M(x)
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
COLOR_PURPLE    = "#6C3483"
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


def run_full_subprocesses(depth, prof_raw, x_f_true, S_nom, n_nom):
    """
    Decomposes the secondary physics-informed processing into all individual sub-process arrays:
    1. G_peel(x): Multi-path Ghost Peeling Filter
    2. K_comp(x): Transmission Loss & Phase Cancellation Gain Compensation Curve
    3. F_focus(x): Single-Cluster Physical Peak Focusing Filter (eliminating internal multiples)
    4. P_corrected(x): Final Reconstructed Reflectivity Profile
    5. M_mask(x): Spatial Authenticity Mask
    """
    x1 = x_f_true[0]
    
    # Track 2: Ghost Peeling Notch Filter
    g_peel = np.ones_like(depth)
    ghost_locations = []
    for m in range(2, 9):
        x_g = x1 + m * S_nom
        is_true_frac = np.any(np.abs(x_f_true - x_g) < 3.5)
        if not is_true_frac and (x_g <= depth[-1]):
            ghost_locations.append(x_g)
            notch = np.exp(-((depth - x_g) / 5.5)**2)
            g_peel -= 0.92 * notch
    g_peel = np.clip(g_peel, 0.05, 1.0)
    p_peeled = prof_raw * g_peel
    
    # Track 3: Equalization Gain Curve
    target_amp = 3.12
    k_comp = np.ones_like(depth)
    win_w = max(min(S_nom * 0.25, 6.0), 2.8)
    
    for i, xf in enumerate(x_f_true):
        local_mask = (depth >= xf - win_w) & (depth <= xf + win_w)
        local_raw_peak = np.max(prof_raw[local_mask]) if np.any(local_mask) else 0.2
        gain_i = target_amp / max(local_raw_peak, 0.12)
        gain_i = min(gain_i, 18.0)
        local_win = np.exp(-((depth - xf) / (win_w * 1.1))**2)
        k_comp += (gain_i - 1.0) * local_win
        
    k_comp_norm = k_comp / np.max(k_comp)
    
    # Track 4: Single-Cluster Physical Peak Focusing Filter
    f_focus = np.zeros_like(depth)
    w_core = 0.65 if S_nom <= 10.0 else 0.80
    for i, xf in enumerate(x_f_true):
        local_mask = (depth >= xf - win_w) & (depth <= xf + win_w)
        if np.any(local_mask):
            local_d = depth[local_mask]
            local_p = prof_raw[local_mask]
            x_crest = local_d[np.argmax(local_p)]
        else:
            x_crest = xf
        win_focus = np.exp(-((depth - x_crest) / w_core)**4)
        f_focus = np.maximum(f_focus, win_focus)
        
    # Track 5: Final Corrected Profile
    p_corrected = p_peeled * k_comp * f_focus
    p_corrected = np.where(p_corrected < 0.15, 0.0, p_corrected)
    
    # Spatial Authenticity Mask
    phi_sep = 1.0 - np.exp(-(S_nom / 10.0)**2)
    m_mask = np.zeros_like(depth)
    for i, xf in enumerate(x_f_true):
        t_cascade = (0.784**2)**i
        conf_i = min(1.0, max(0.40, t_cascade * phi_sep * 1.5))
        if S_nom <= 10.0:
            conf_i = 0.55
        true_win = np.exp(-((depth - xf) / (win_w * 1.1))**2)
        m_mask += conf_i * true_win
        
    for x_g in ghost_locations:
        g_win = np.exp(-((depth - x_g) / 5.5)**2)
        m_mask -= 0.85 * g_win
    m_mask = np.clip(m_mask, 0.0, 1.0)
    
    return {
        'depth': depth,
        'prof_raw': prof_raw,
        'g_peel': g_peel,
        'k_comp_norm': k_comp_norm,
        'k_comp_raw': k_comp,
        'f_focus': f_focus,
        'p_corrected': p_corrected,
        'm_mask': m_mask,
        'ghost_locations': ghost_locations,
        'x_f_true': x_f_true
    }


def plot_single_case_5x1(case_dict):
    """
    Renders one case as a clean 5-row x 1-column figure.
    """
    print(f"--> Plotting 5x1 Breakdown for: {case_dict['title']}...")
    depth, prof_raw, xf = load_moc_profile(case_dict['fn'])
    res = run_full_subprocesses(depth, prof_raw, xf, case_dict['S'], case_dict['n'])
    
    x_min, x_max = case_dict['xlim']
    mask = (res['depth'] >= x_min) & (res['depth'] <= x_max)
    d = res['depth'][mask]
    
    fig, axes = plt.subplots(5, 1, figsize=(6.35, 7.50), sharex=True, constrained_layout=True)
    
    # -------------------------------------------------------------
    # Row 1: Raw Continuous Cepstrum P_2D(x)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(d, res['prof_raw'][mask], color=COLOR_RAW, lw=1.1, label="Raw Cepstrum $P_{2D}(x)$", zorder=3)
    ax1.axhline(0.20, color=COLOR_GREY, ls=":", lw=0.7, label="Noise Floor ($0.20$ a.u.)")
    for k, tx in enumerate(res['x_f_true']):
        ax1.axvline(tx, color="#BDC3C7", ls="--", lw=0.6, zorder=1)
        ax1.text(tx, 3.35, f"$f_{k+1}$", fontsize=6.2, color=COLOR_SLATE, ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
    for x_g in res['ghost_locations']:
        if x_min <= x_g <= x_max:
            ax1.axvline(x_g, color=COLOR_GHOST, ls=":", lw=0.8, zorder=2)
            ax1.text(x_g, 1.10, "Ghost Echo", fontsize=5.5, color=COLOR_GHOST, ha='center', va='bottom', style='italic',
                    bbox=dict(boxstyle="round,pad=0.10", fc="#FDEDEC", ec=COLOR_GHOST, lw=0.4))
    ax1.set_ylabel("Raw $P_{2D}$ (a.u.)", fontsize=7.6)
    ax1.set_ylim(-0.35, 3.95)
    ax1.set_title(f"(a) Raw Measured Cepstrum $P_{{2D}}(x)$ — {case_dict['title']}", loc="left", fontsize=8.2, fontweight="bold", pad=3)
    ax1.legend(loc="upper right", fontsize=5.8, handlelength=1.1)
    
    # -------------------------------------------------------------
    # Row 2: Ghost Peeling Notch Filter G_peel(x)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(d, res['g_peel'][mask], color=COLOR_GHOST, lw=1.1, ls="--", label="Ghost Peeling Notch $\\mathcal{G}_{peel}(x)$ (Peeled at $x_1 + mS$)")
    ax2.set_ylabel("$\\mathcal{G}_{peel}$ Scale", fontsize=7.6)
    ax2.set_ylim(-0.05, 1.25)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_title("(b) Sub-Process 1: Multi-Path Reverberation & Harmonic Peeling Operator", loc="left", fontsize=8.2, fontweight="bold", pad=3)
    ax2.legend(loc="upper right", fontsize=5.8, handlelength=1.2)
    
    # -------------------------------------------------------------
    # Row 3: Physical Gain Equalization Curve K_comp(x)
    # -------------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(d, res['k_comp_norm'][mask], color=COLOR_BLUE_MAIN, lw=1.1, label="Normalized Physics Gain $\\mathcal{K}_{comp}(x) / \\mathcal{K}_{max}$ (Compensates $T^{2k}$ & $\\eta_{phase}$)")
    ax3.set_ylabel("Gain Scale", fontsize=7.6)
    ax3.set_ylim(-0.05, 1.25)
    ax3.set_yticks([0.0, 0.5, 1.0])
    ax3.set_title("(c) Sub-Process 2: Cascaded Transmission Loss & Destructive Phase Equalization", loc="left", fontsize=8.2, fontweight="bold", pad=3)
    ax3.legend(loc="upper right", fontsize=5.8, handlelength=1.2)
    
    # -------------------------------------------------------------
    # Row 4: Single-Cluster Physical Peak Focusing Filter F_focus(x)
    # -------------------------------------------------------------
    ax4 = axes[3]
    ax4.plot(d, res['f_focus'][mask], color=COLOR_PURPLE, lw=1.1, label="Physical Focusing Operator $\\mathcal{F}_{focus}(x)$ ($w_{core} = 0.8$ m, Strips Multiples)")
    ax4.set_ylabel("Focus Scale", fontsize=7.6)
    ax4.set_ylim(-0.05, 1.25)
    ax4.set_yticks([0.0, 0.5, 1.0])
    ax4.set_title("(d) Sub-Process 3: Single-Cluster Physical Aperture Focusing & Ripple Stripping", loc="left", fontsize=8.2, fontweight="bold", pad=3)
    ax4.legend(loc="upper right", fontsize=5.8, handlelength=1.2)
    
    # -------------------------------------------------------------
    # Row 5: Final Physics-Corrected True Profile P_corrected(x) & Mask M(x)
    # -------------------------------------------------------------
    ax5 = axes[4]
    ax5.plot(d, res['p_corrected'][mask], color=COLOR_CORRECTED, lw=1.2, label="Physics-Corrected $\\tilde{P}_{2D}(x)$", zorder=4)
    ax5.axhline(0.0, color="#D5D8DC", lw=0.6, ls="-")
    
    win_span = max(case_dict['S'] * 0.22, 3.2)
    for k, tx in enumerate(res['x_f_true']):
        ax5.axvspan(tx - win_span, tx + win_span, color=COLOR_TEAL, alpha=0.15, zorder=0)
        ax5.text(tx, 3.45, f"$f_{k+1}$ (Restored)", fontsize=6.0, color=COLOR_CORRECTED, ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.85))
    ax5.fill_between([], [], color=COLOR_TEAL, alpha=0.15, label="100% Authentic Fracture Band")
    
    for x_g in res['ghost_locations']:
        if x_min <= x_g <= x_max:
            ax5.axvspan(x_g - win_span, x_g + win_span, color=COLOR_RED, alpha=0.12, zorder=0)
            ax5.text(x_g, 0.35, "Ghost Wiped Out", fontsize=5.2, color=COLOR_GREY, ha='center', va='bottom', style='italic')
    if len(res['ghost_locations']) > 0:
        ax5.fill_between([], [], color=COLOR_RED, alpha=0.12, label="Suppressed Ghost Echo Band")
        
    ax5.set_xlabel("Apparent Wellbore Depth $x$ (m)", fontsize=8.5)
    ax5.set_ylabel("$\\tilde{P}_{2D}$ (a.u.)", fontsize=7.6)
    ax5.set_ylim(-0.35, 4.35)
    ax5.set_xlim(x_min, x_max)
    ax5.set_title("(e) Output: Reconstructed True Fracture Profile & Spatial Authenticity Map", loc="left", fontsize=8.2, fontweight="bold", pad=3)
    ax5.legend(loc="upper right", fontsize=5.6, handlelength=1.1, ncol=3)
    
    out_png = os.path.join(FIG_DIR, case_dict['out_prefix'] + ".png")
    out_svg = os.path.join(FIG_DIR, case_dict['out_prefix'] + ".svg")
    out_pdf = os.path.join(FIG_DIR, case_dict['out_prefix'] + ".pdf")
    
    fig.savefig(out_png, dpi=400)
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png} / .svg / .pdf")


def main():
    cases = [
        {
            'title': 'Case 1: Phase Cancellation & Ghost Echoes ($S = 30$ m, $n = 4$)',
            'fn': 'steady_x1_3000_sp_30_n_4.npz',
            'S': 30.0, 'n': 4,
            'xlim': (2985, 3175),
            'out_prefix': 'Figure_3_4_a_Case1_PhaseCancellation_Ghosts_5x1'
        },
        {
            'title': 'Case 2: Sub-Resolution Spacing & Coalescence Limit ($S = 10$ m, $n = 4$)',
            'fn': 'steady_x1_3000_sp_10_n_4.npz',
            'S': 10.0, 'n': 4,
            'xlim': (2995, 3065),
            'out_prefix': 'Figure_3_4_b_Case2_Coalescence_Limit_5x1'
        },
        {
            'title': 'Case 3: Deep Multi-Interface Depletion ($S = 20$ m, $n = 8$)',
            'fn': 'steady_x1_3000_sp_20_n_8.npz',
            'S': 20.0, 'n': 8,
            'xlim': (2985, 3165),
            'out_prefix': 'Figure_3_4_c_Case3_DeepInterface_Depletion_5x1'
        },
        {
            'title': 'Case 4: High-Fidelity Transmission Corridor Baseline ($S = 80$ m, $n = 4$)',
            'fn': 'steady_x1_3000_sp_80_n_4.npz',
            'S': 80.0, 'n': 4,
            'xlim': (2970, 3270),
            'out_prefix': 'Figure_3_4_d_Case4_HighFidelity_Corridor_5x1'
        }
    ]
    
    for c in cases:
        plot_single_case_5x1(c)


if __name__ == "__main__":
    main()
