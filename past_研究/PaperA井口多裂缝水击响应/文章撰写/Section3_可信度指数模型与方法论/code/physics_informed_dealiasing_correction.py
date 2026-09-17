"""
Physics-Informed De-aliasing, Ghost Peeling and Waveform Reconstruction Model for Multi-Cluster Water-Hammer Cepstrum
Author: Antigravity (Pair Programming with User)
Target: SPE Journal - Multi-Cluster Water-Hammer Diagnostics (Section 3 Methodology)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.75,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.0,
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


class PhysicsInformedDealiasingCorrector:
    """
    Generates operator kernels and a prior-based sensitivity proxy.  It does
    not estimate a fracture state and therefore does not produce an
    "equalized" or "true" profile.
    """
    def __init__(self, T_eff=0.784, S_null=30.0, w_null=22.0, kappa=0.65, S_min=10.0):
        self.T_eff = T_eff
        self.S_null = S_null
        self.w_null = w_null
        self.kappa = kappa
        self.S_min = S_min

    def process(self, depth, prof_raw, x_f_true, S_nom, n_nom):
        x1 = 3000.0
        x_perf = x1 + np.arange(n_nom, dtype=float) * S_nom
        
        # 1. Multi-path Ghost Harmonic Peeling Filter G_peel(x)
        # Theoretical ghost positions occur at x1 + m * S (m >= 2) due to inter-cluster reverberation
        g_peel = np.ones_like(depth)
        ghost_locations = []
        for m in range(2, 7):
            x_g = x1 + m * S_nom
            # Check if this ghost position coincides with a designed true fracture
            is_true_frac = np.any(np.isclose(x_perf, x_g, atol=1e-6))
            if not is_true_frac and (x_g <= depth[-1]):
                ghost_locations.append(x_g)
                # Apply peeling notch filter at ghost echo position
                notch = np.exp(-((depth - x_g) / 5.5)**2)
                g_peel -= 0.92 * notch
        g_peel = np.clip(g_peel, 0.05, 1.0)
        
        # 2. Physics-derived finite transmission gain.  No local peak or
        # target amplitude enters this operator.
        k_comp = np.ones_like(depth)
        for i, xf in enumerate(x_perf):
            gain_i = 1.0 / (self.T_eff ** (2.0 * i))
            local_win = np.exp(-((depth - xf) / max(0.8, min(6.0, 0.25 * S_nom)))**2)
            k_comp += (gain_i - 1.0) * local_win

        # 3. Super-Gaussian core kernels at the nominal design depths.
        f_focus = np.zeros_like(depth)
        for xf in x_perf:
            win_focus = np.exp(-((depth - xf) / 0.80)**4)
            f_focus = np.maximum(f_focus, win_focus)

        # 4. Prior/transmission sensitivity proxy, not a probability of
        # breakdown.  Residual and coefficient uncertainty are unavailable in
        # this forward-only figure.
        m_mask = np.zeros_like(depth)
        for i, xf in enumerate(x_perf):
            conf_i = self.T_eff ** (2.0 * i)
            m_mask += conf_i * np.exp(-((depth - xf) / max(1.5, min(6.0, 0.25 * S_nom)))**2)
        for x_g in ghost_locations:
            g_win = np.exp(-((depth - x_g) / 5.5)**2)
            m_mask -= 0.85 * g_win
            
        m_mask = np.clip(m_mask, 0.0, 1.0)
        
        return {
            'depth': depth,
            'prof_raw': prof_raw,
            'g_peel': g_peel,
            'k_comp': k_comp,
            'f_focus': f_focus,
            'm_mask': m_mask,
            'ghost_locations': ghost_locations,
            'x_perf': x_perf
        }


def plot_physics_correction_workflow():
    """
    Plots the full Physics-Informed De-aliasing & Waveform Correction Workflow:
    Track 1: Raw Measured Cepstrum P_2D(x) (showing phase cancellation & ghost peaks)
    Track 2: Physical Correction Operators (Ghost Peeling Filter + Physical Gain Restorer)
    Track 3: finite physics-derived K_comp and F_focus kernels
    Track 4: prior/transmission sensitivity proxy (not a probability)
    """
    print("[1/1] Plotting Physics-Informed De-aliasing & Correction Workflow Figure...")
    corrector = PhysicsInformedDealiasingCorrector()
    
    # Case: S=30m, n=4 (Strong Phase Cancellation & Ghost Peak at 3120m)
    depth, prof_raw, xf = load_moc_profile("steady_x1_3000_sp_30_n_4.npz")
    res = corrector.process(depth, prof_raw, xf, S_nom=30.0, n_nom=4)
    
    # Clean display window: 2985m to 3175m (centers all 4 fractures and 2 ghost echoes)
    x_disp_min, x_disp_max = 2985, 3175
    mask = (res['depth'] >= x_disp_min) & (res['depth'] <= x_disp_max)
    d = res['depth'][mask]
    
    fig, axes = plt.subplots(4, 1, figsize=(6.2, 6.2), sharex=True, constrained_layout=True)
    
    # -------------------------------------------------------------
    # Track 1: Raw Continuous Cepstrum P_2D(x) (Raw Waveform with Traps)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(d, res['prof_raw'][mask], color=COLOR_RAW, lw=1.2, label="Raw Cepstrum $P_{2D}(x)$", zorder=3)
    ax1.axhline(0.20, color=COLOR_GREY, ls=":", lw=0.75, label="Noise Floor ($0.20$ a.u.)")
    
    # Mark nominal design prior only; activation remains unknown.
    for k, tx in enumerate(res['x_perf']):
        ax1.axvline(tx, color="#BDC3C7", ls="--", lw=0.65, zorder=1)
        ax1.text(tx, 3.35, f"$x_{{p,{k+1}}}$", fontsize=6.2, color=COLOR_SLATE, ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85))
        
    # Mark Ghost Peak at 3120m and 3150m
    for x_g in res['ghost_locations']:
        if x_disp_min <= x_g <= x_disp_max:
            ax1.axvline(x_g, color=COLOR_GHOST, ls=":", lw=0.85, zorder=2)
            ax1.text(x_g, 1.10, "Path\ncandidate", fontsize=5.8, color=COLOR_GHOST, 
                    ha='center', va='bottom', style='italic', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.15", fc="#FDEDEC", ec=COLOR_GHOST, lw=0.5))
            
    ax1.set_ylabel("Raw $P_{2D}$ (a.u.)", fontsize=7.8)
    ax1.set_ylim(-0.35, 3.95)
    ax1.set_title("(a) Raw continuous cepstrum $P_{2D}(x)$ (nonlocal attenuation and path candidates)", loc="left", fontsize=8.2, fontweight="bold", pad=4)
    ax1.legend(loc="upper right", fontsize=5.8, handlelength=1.1)
    
    # -------------------------------------------------------------
    # Track 2: Physics Correction Operators (Peeling & Gain Restoration)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(d, res['g_peel'][mask], color=COLOR_GHOST, lw=1.1, ls="--", label="Candidate-path notch $\\mathcal{G}_{peel}(x)$ (operator template)")
    # Normalize the physical gain curve to max = 1.0 for the plotted scale.
    k_comp_norm = res['k_comp'][mask] / np.max(res['k_comp'])
    ax2.plot(d, k_comp_norm, color=COLOR_BLUE_MAIN, lw=1.1, label="Finite transmission gain $\\mathcal{K}_{comp}/\\mathcal{K}_{max}$ (unit phase factor)")
    
    ax2.set_ylabel("Operator Scale", fontsize=7.8)
    ax2.set_ylim(-0.05, 1.28)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_title("(b) Physics-Informed Correction Operators Derived from Forward Wavefield Laws", loc="left", fontsize=8.2, fontweight="bold", pad=4)
    ax2.legend(loc="upper right", fontsize=5.8, handlelength=1.2, ncol=2)
    
    # -------------------------------------------------------------
    # Track 3: finite K_comp and F_focus kernels (no corrected profile)
    # -------------------------------------------------------------
    ax3 = axes[2]
    k_norm = res['k_comp'][mask] / np.max(res['k_comp'])
    ax3.plot(d, k_norm, color=COLOR_BLUE_MAIN, lw=1.1,
             label="Finite $\\mathcal{K}_{comp}(x)$ (normalized for display)")
    ax3.plot(d, res['f_focus'][mask], color=COLOR_PURPLE, lw=1.1,
             label="Candidate $\\mathcal{F}_{focus}$ kernel, $w_{core}=0.80$ m")
    ax3.set_ylabel("Operator scale (normalized)", fontsize=7.8)
    ax3.set_ylim(-0.05, 1.25)
    ax3.set_title("(c) Physics-derived gain and candidate focusing kernels", loc="left", fontsize=8.2, fontweight="bold", pad=4)
    ax3.legend(loc="upper right", fontsize=5.8, handlelength=1.1)
    
    # -------------------------------------------------------------
    # Track 4: Spatial Authenticity & Confidence Mask M(x)
    # -------------------------------------------------------------
    ax4 = axes[3]
    m_sub = res['m_mask'][mask]
    ax4.plot(d, m_sub, color=COLOR_SLATE, lw=1.1, label="Prior/transmission proxy $M_{\\mathrm{sens}}(x)$", zorder=3)
    
    # Fill confidence zones
    for tx in res['x_perf']:
        ax4.axvspan(tx - 6.5, tx + 6.5, color=COLOR_TEAL, alpha=0.16)
    # Add one proxy fill for legend
    ax4.fill_between([], [], color=COLOR_TEAL, alpha=0.16, label="Nominal prior window")
    
    for x_g in res['ghost_locations']:
        if x_disp_min <= x_g <= x_disp_max:
            ax4.axvspan(x_g - 6.5, x_g + 6.5, color=COLOR_RED, alpha=0.12)
    ax4.fill_between([], [], color=COLOR_RED, alpha=0.12, label="Non-perforation path candidate")
    
    ax4.set_xlabel("Apparent Wellbore Depth $x$ (m)", fontsize=8.5)
    ax4.set_ylabel("Transmission sensitivity proxy", fontsize=7.8)
    ax4.set_ylim(-0.05, 1.28)
    ax4.set_yticks([0.0, 0.5, 1.0])
    ax4.set_xlim(x_disp_min, x_disp_max)
    ax4.set_title("(d) Prior/transmission sensitivity proxy (not a fracture probability)", loc="left", fontsize=8.2, fontweight="bold", pad=4)
    ax4.legend(loc="upper right", fontsize=5.8, handlelength=1.1, ncol=3)

    # Render-time gate required by the Nature figure workflow.  The measured
    # axes rectangles are written alongside the figure for independent QA.
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(FIG_DIR, "Figure_3_3_Physics_Informed_Dealiasing_Workflow.panel-alignment.json"),
        require_panel_labels=False,
    )
    
    out_png = os.path.join(FIG_DIR, "Figure_3_3_Physics_Informed_Dealiasing_Workflow.png")
    out_svg = os.path.join(FIG_DIR, "Figure_3_3_Physics_Informed_Dealiasing_Workflow.svg")
    out_pdf = os.path.join(FIG_DIR, "Figure_3_3_Physics_Informed_Dealiasing_Workflow.pdf")
    
    fig.savefig(out_png, dpi=400)
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Physics-Informed Dealiasing Figure: {out_png} / .svg / .pdf")


if __name__ == "__main__":
    plot_physics_correction_workflow()
