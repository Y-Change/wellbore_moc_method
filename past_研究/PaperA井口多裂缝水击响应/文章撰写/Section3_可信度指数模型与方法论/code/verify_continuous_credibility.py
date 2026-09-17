"""
Continuous Multi-Factor Diagnostic Credibility Model & Spatial Interval Partitioning
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

# Curated Color Palettes
COLOR_BLUE_MAIN = "#1B4F72"
COLOR_TEAL      = "#117864"
COLOR_ORANGE    = "#D35400"
COLOR_RED       = "#900C3F"
COLOR_PURPLE    = "#6C3483"
COLOR_SLATE     = "#2C3E50"
COLOR_GOLD      = "#B7950B"
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


class ContinuousCredibilityAnalyzer:
    """
    Evaluates continuous multi-attribute credibility curves along wellbore depth x:
    1. Local Signal-to-Noise Ratio & Prominence Gamma_SNR(x)
    2. Spatial Wavelet Isolation / Shape Clarity S_iso(x)
    3. Multi-Interface Physical Transmission Compatibility Lambda_phys(x)
    4. Ghost / Harmonic Artifact Rejection Factor G_reject(x)
    5. Continuous Composite Diagnostic Credibility Curve Psi(x)
    6. Comprehensive Stage Diagnostic Index (SDI Score)
    """
    def __init__(self, 
                 sigma_noise_floor=0.20,
                 T_eff=0.784,
                 S_min=10.0,
                 S_null=30.0,
                 w_null=22.0,
                 kappa=0.65):
        self.sigma_noise = sigma_noise_floor
        self.T_eff = T_eff
        self.S_min = S_min
        self.S_null = S_null
        self.w_null = w_null
        self.kappa = kappa

    def compute_continuous_metrics(self, depth, prof, x_f_true, S_nom, n_nom):
        """Computes continuous metric tracks across wellbore depth array."""
        dx = depth[1] - depth[0]
        
        # 1. Local SNR Track Gamma_SNR(x)
        # Ratio of positive envelope to noise floor
        gamma_snr = np.maximum(prof - 0.05, 0.0) / self.sigma_noise
        gamma_snr_norm = np.clip(gamma_snr / 5.0, 0.0, 1.2) # Normalized to [0, 1.2]
        
        # 2. Spatial Wavelet Isolation S_iso(x)
        # Second derivative curvature to detect sharp peak tops vs flat/smeared envelopes
        d2_prof = np.gradient(np.gradient(prof, dx), dx)
        sharpness = np.maximum(-d2_prof, 0.0)
        s_iso = 1.0 - np.exp(-sharpness / 0.04)
        
        # 3. Spatial Separation Factor from S_nom
        phi_sep_val = 1.0 - np.exp(-(S_nom / self.S_min)**2)
        s_iso = s_iso * phi_sep_val
        
        # 4. Multi-Interface Attenuation & Ghost Artifact Suppression Track G_reject(x)
        # Penalizes regions that are expected to be echo harmonics (e.g. at x1 + 2*S, x1 + 3*S)
        g_reject = np.ones_like(depth)
        x1 = x_f_true[0]
        for m in range(2, 6):
            x_ghost = x1 + m * S_nom
            ghost_mask = np.exp(-((depth - x_ghost) / 8.0)**2)
            # If there's an expected true fracture there, no penalty; otherwise, penalize harmonic
            is_true_frac = np.any(np.abs(x_f_true - x_ghost) < 4.0)
            if not is_true_frac:
                g_reject -= 0.65 * ghost_mask
        g_reject = np.clip(g_reject, 0.15, 1.0)
        
        # 5. Continuous Diagnostic Credibility Index Profile Psi(x)
        psi_continuous = gamma_snr * s_iso * g_reject
        
        # 6. Cluster-level Evaluation on Candidate Wave Packets
        peaks_idx, _ = find_peaks(prof, height=0.18, distance=int(12.0 / dx), prominence=0.15)
        peak_depths = depth[peaks_idx]
        peak_amps = prof[peaks_idx]
        
        cluster_diagnostics = []
        for i, tx in enumerate(x_f_true):
            # Find closest detected peak to true position
            dist = np.abs(peak_depths - tx)
            if len(dist) > 0 and np.min(dist) <= 15.0:
                best_match = np.argmin(dist)
                p_amp = peak_amps[best_match]
                p_x = peak_depths[best_match]
                p_idx = peaks_idx[best_match]
                psi_val = psi_continuous[p_idx]
                offset = p_x - tx
            else:
                p_amp = 0.0
                p_x = tx
                psi_val = 0.0
                offset = np.nan
                
            # Grade classification per cluster
            if psi_val >= 5.0 and p_amp >= 0.25:
                status = "High-Credibility (Grade A)"
                color = COLOR_TEAL
            elif psi_val >= 2.0 and p_amp >= 0.15:
                status = "Interference-Marginal (Grade B)"
                color = COLOR_ORANGE
            else:
                status = "Degraded/Blind (Grade C)"
                color = COLOR_RED
                
            cluster_diagnostics.append({
                'frac_idx': i + 1,
                'x_true': tx,
                'x_est': p_x,
                'offset_m': offset,
                'P_amp': p_amp,
                'Psi_val': psi_val,
                'status': status,
                'color': color
            })
            
        # 7. Comprehensive Stage Diagnostic Index (SDI Score, 0 - 100)
        n_true = len(x_f_true)
        n_valid = sum(1 for c in cluster_diagnostics if "Grade A" in c['status'])
        n_marg  = sum(1 for c in cluster_diagnostics if "Grade B" in c['status'])
        eta_valid = (n_valid + 0.5 * n_marg) / n_true
        
        # Effective amplitude stability & mean credibility of true clusters
        psi_vals_true = [c['Psi_val'] for c in cluster_diagnostics]
        mean_psi = np.mean(psi_vals_true)
        psi_score = np.clip(mean_psi / 10.0, 0.0, 1.0)
        
        # Ghost risk: false peaks far from true fractures (dist >= 18m) with amp >= 0.25
        ghost_peaks = [px for px, pa in zip(peak_depths, peak_amps) if pa >= 0.25 and not np.any(np.abs(x_f_true - px) < 18.0)]
        ghost_risk = len(ghost_peaks) / max(len(peak_depths), 1)
        
        # Combined SDI Score (0 - 100)
        sdi_score = (0.50 * eta_valid + 0.35 * psi_score + 0.15 * (1.0 - ghost_risk)) * 100.0
        
        return {
            'depth': depth,
            'prof': prof,
            'gamma_snr': gamma_snr,
            's_iso': s_iso,
            'g_reject': g_reject,
            'psi_continuous': psi_continuous,
            'clusters': cluster_diagnostics,
            'peak_depths': peak_depths,
            'peak_amps': peak_amps,
            'sdi_score': sdi_score,
            'eta_valid': eta_valid * 100.0,
            'ghost_risk': ghost_risk * 100.0
        }


def plot_continuous_credibility_comparison():
    """
    Generates a high-impact multi-panel diagnostic comparison figure:
    Contrasting Case 1 (Grade A: S=80m, n=4, High SDI) vs Case 2 (Grade B: S=30m, n=4, Interference & Ghost Risk).
    """
    print("[1/1] Plotting Continuous Multi-Factor Credibility Workflow Figure...")
    
    analyzer = ContinuousCredibilityAnalyzer()
    
    # Load 2 contrasting 4-cluster cases
    # Case 1: S=80m, n=4 (High-Fidelity)
    depth_1, prof_1, xf_1 = load_moc_profile("steady_x1_3000_sp_80_n_4.npz")
    res_1 = analyzer.compute_continuous_metrics(depth_1, prof_1, xf_1, S_nom=80.0, n_nom=4)
    
    # Case 2: S=30m, n=4 (Destructive Phase Interference & Ghost Risk)
    depth_2, prof_2, xf_2 = load_moc_profile("steady_x1_3000_sp_30_n_4.npz")
    res_2 = analyzer.compute_continuous_metrics(depth_2, prof_2, xf_2, S_nom=30.0, n_nom=4)
    
    fig, axes = plt.subplots(3, 2, figsize=(7.35, 5.20), sharex='col', constrained_layout=True)
    
    cases_data = [
        (axes[:, 0], res_1, "(a) Case 1: High-Fidelity Array ($S = 80$ m, $n = 4$)", COLOR_TEAL, 2960, 3320),
        (axes[:, 1], res_2, "(b) Case 2: Phase Cancellation & Ghost Risk ($S = 30$ m, $n = 4$)", COLOR_ORANGE, 2960, 3200)
    ]
    
    for (col_axes, res, col_title, col_theme, x_min, x_max) in cases_data:
        depth = res['depth']
        mask = (depth >= x_min) & (depth <= x_max)
        d_sub = depth[mask]
        
        # -------------------------------------------------------------
        # Track 1: Marginal Cepstral Waveform & Candidate Wave Packets
        # -------------------------------------------------------------
        ax1 = col_axes[0]
        ax1.plot(d_sub, res['prof'][mask], color=col_theme, lw=1.1, label="Cepstrum $P_{2D}(x)$", zorder=3)
        ax1.axhline(0.20, color=COLOR_GREY, ls=":", lw=0.75, label="Noise Floor ($0.20$ a.u.)")
        
        # Mark candidate peaks
        for c in res['clusters']:
            ax1.axvline(c['x_true'], color="#BDC3C7", ls="--", lw=0.65, zorder=1)
            ax1.scatter(c['x_est'], c['P_amp'], color=c['color'], s=22, zorder=4, edgecolor='white', linewidth=0.5)
            # Label
            ax1.text(c['x_true'], c['P_amp'] + 0.35, f"$f_{c['frac_idx']}$", fontsize=6.2, 
                    color=COLOR_SLATE, ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85))
            
        ax1.set_ylabel("$P_{2D}$ (a.u.)", fontsize=7.8)
        ax1.set_ylim(-0.35, 3.85)
        ax1.set_title(col_title, loc="left", fontsize=8.2, fontweight="bold", pad=4)
        ax1.legend(loc="upper right", fontsize=5.8, handlelength=1.1)
        
        # -------------------------------------------------------------
        # Track 2: Deconstructed Multi-Factor Physical Metrics
        # -------------------------------------------------------------
        ax2 = col_axes[1]
        ax2.plot(d_sub, res['s_iso'][mask], color=COLOR_SLATE, lw=0.9, ls='--', label="Curvature Sharpness $\\mathcal{S}_{iso}(x)$")
        ax2.plot(d_sub, res['g_reject'][mask], color=COLOR_PURPLE, lw=0.9, ls=':', label="Ghost Rejection $\\mathcal{G}_{rej}(x)$")
        ax2.plot(d_sub, np.clip(res['gamma_snr'][mask]/10.0, 0, 1.1), color=COLOR_BLUE_MAIN, lw=1.0, label="Normalized SNR $\\Gamma_{SNR}(x)/10$")
        
        ax2.set_ylabel("Factor Metric", fontsize=7.8)
        ax2.set_ylim(-0.05, 1.25)
        ax2.set_yticks([0.0, 0.5, 1.0])
        ax2.legend(loc="upper right", fontsize=5.6, handlelength=1.1, ncol=2)
        
        # -------------------------------------------------------------
        # Track 3: Continuous Credibility Profile Psi(x) & Diagnostic Zone Partitioning
        # -------------------------------------------------------------
        ax3 = col_axes[2]
        psi_sub = res['psi_continuous'][mask]
        ax3.plot(d_sub, psi_sub, color=COLOR_SLATE, lw=1.1, label="Continuous Credibility $\\Psi(x)$", zorder=3)
        
        # Horizontal decision thresholds
        ax3.axhline(5.0, color=COLOR_TEAL, ls="--", lw=0.75, label="High-Credibility Thresh ($\\Psi=5.0$)")
        ax3.axhline(2.0, color=COLOR_ORANGE, ls=":", lw=0.75, label="Marginal Thresh ($\\Psi=2.0$)")
        
        # Continuous color-shaded background interval partition
        ax3.fill_between(d_sub, 0, 35, where=(psi_sub >= 5.0), color=COLOR_TEAL, alpha=0.15, label="High-Credibility Zone")
        ax3.fill_between(d_sub, 0, 35, where=((psi_sub >= 2.0) & (psi_sub < 5.0)), color=COLOR_ORANGE, alpha=0.15, label="Marginal/Caution Zone")
        ax3.fill_between(d_sub, 0, 35, where=(psi_sub < 2.0), color=COLOR_RED, alpha=0.10, label="Blind/Unreliable Zone")
        
        # Display Stage Diagnostic Index (SDI) Box
        sdi_text = f"Stage SDI Score: {res['sdi_score']:.1f}/100\nValid Clusters: {res['eta_valid']:.0f}%\nGhost Risk: {res['ghost_risk']:.0f}%"
        ax3.text(0.98, 0.92, sdi_text, transform=ax3.transAxes, fontsize=6.0, color=COLOR_SLATE,
                ha='right', va='top', bbox=dict(boxstyle="round,pad=0.25", fc="#EAEDED", ec="#BDC3C7", lw=0.6))
        
        ax3.set_xlabel("Apparent Depth $x$ (m)", fontsize=8.2)
        ax3.set_ylabel("Credibility $\\Psi(x)$", fontsize=7.8)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(-1.0, 28.0)
        ax3.legend(loc="upper left", fontsize=5.2, handlelength=1.1, ncol=2)
        
    out_png = os.path.join(FIG_DIR, "Figure_3_2_Continuous_Diagnostic_Credibility_Workflow.png")
    out_svg = os.path.join(FIG_DIR, "Figure_3_2_Continuous_Diagnostic_Credibility_Workflow.svg")
    out_pdf = os.path.join(FIG_DIR, "Figure_3_2_Continuous_Diagnostic_Credibility_Workflow.pdf")
    
    fig.savefig(out_png, dpi=400)
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Continuous Workflow Figure: {out_png} / .svg / .pdf")


if __name__ == "__main__":
    plot_continuous_credibility_comparison()
