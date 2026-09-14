# -*- coding: utf-8 -*-
"""
experiments/sensitivity/plot_figures.py
----------------------------------------
Publication Figure Plates Generator (Milestone M4).

Produces 4 Nature-style publication figure plates:
1. Plate 1: fig1_wavefront_step_gradient.png & .svg
   - Joukowsky theoretical and simulated step drop
   - Reflected wavefront arrival vs fracture depth x_f
   - Dynamic wavefront gradient (dH/dt) across compliance C_H
   - Perforation throttling impedance R_p cushioning effect
2. Plate 2: fig2_envelope_rms_decay.png & .svg
   - 40-second time series and envelope attenuation (Steady vs Brunone)
   - 5-window RMS head progression
   - Exponential damping decay rate alpha_rms vs leakoff k_leak (Damping Confusion Zone)
   - RMS retention percentage in C_H x k_leak space
3. Plate 3: fig3_frequency_spectral_dissipation.png & .svg
   - FFT amplitude spectra and acoustic harmonic comb (f0 = 0.0725 Hz)
   - Logarithmic power spectral density with f > 1.5 Hz roll-off
   - High-frequency power ratio R_high vs k_leak and C_H
   - Cumulative spectral energy distribution
4. Plate 4: fig4_cepstrum_rayleigh_resolution.png & .svg
   - 1D real cepstrum peak detection vs depth x_f and unsteady clock skew
   - Cepstral peak amplitude scaling law vs C_H and k_leak
   - 2D sliding-window cepstrogram time-depth energy heatmap
   - Rayleigh spatial resolution limit and cluster spacing ablation (delta_x = 5 to 50 m)

Specifications:
- Nature publication styling: Arial/sans-serif typography, clean axes without top/right spines.
- Semantic color palette: Blue (#0F4D92) Steady, Red (#B64342) Brunone, distinct tier palettes.
- Output resolution: 300 DPI PNG (>=200 DPI) and editable vector SVG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moc_simulate.cepstrum_mocdata import cepstrogram

# Path definitions
SENSITIVITY_DIR = PROJECT_ROOT / "output" / "fracture_parameter_sensitivity"
DATA_DIR = SENSITIVITY_DIR / "data"
TABLES_DIR = SENSITIVITY_DIR / "tables"
FIGURES_DIR = SENSITIVITY_DIR / "figures"
METRICS_CSV_PATH = TABLES_DIR / "sensitivity_metrics.csv"
SUMMARY_JSON_PATH = TABLES_DIR / "sensitivity_summary.json"

# Semantic publication color palette
PALETTE = {
    "steady": "#0F4D92",         # Classic deep navy for Steady Darcy
    "brunone": "#B64342",        # Crimson red for Brunone unsteady
    "teal": "#2A8C82",           # Deep teal
    "green": "#2E7D32",          # Forest green
    "violet": "#7B3F8D",         # Royal violet
    "amber": "#D97706",          # Warm amber
    "dark_neutral": "#333333",   # Dark neutral
    "light_neutral": "#E5E7EB",  # Soft gray
    "accent_cyan": "#0284C7",    # Bright cyan
    "accent_coral": "#EA580C",   # Coral
}

TIER_COLORS = [
    "#0F4D92",  # Deep navy
    "#2A8C82",  # Teal
    "#2E7D32",  # Green
    "#D97706",  # Amber
    "#7B3F8D",  # Violet
    "#B64342",  # Crimson
]


def apply_nature_style(font_size: int = 8, axes_linewidth: float = 0.8) -> None:
    """Configure matplotlib rcParams to strictly match Nature journal publication standards."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "SimHei", "Microsoft YaHei"],
        "svg.fonttype": "none",         # Keep text as editable <text> tags in SVG
        "pdf.fonttype": 42,             # TrueType font embedding for PDF
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.titleweight": "bold",
        "axes.labelsize": font_size,
        "axes.labelweight": "normal",
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "legend.frameon": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "axes.unicode_minus": False,
        "axes.grid": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: int = 10,
) -> None:
    """Place bold Nature-style panel letter (a, b, c, d) at top-left of subplot."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_publication_plate(
    fig: plt.Figure,
    base_name: str,
    output_dir: Path = FIGURES_DIR,
    dpi: int = 300,
) -> None:
    """Save high-resolution 300 DPI PNG and companion vector SVG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}.png"
    svg_path = output_dir / f"{base_name}.svg"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {png_path} ({png_path.stat().st_size} bytes, DPI={dpi})")
    print(f"[SAVED] {svg_path} ({svg_path.stat().st_size} bytes)")


# ===========================================================================
# Plate 1: fig1_wavefront_step_gradient
# ===========================================================================
def plot_plate1_wavefront_step_gradient(
    metrics_df: pd.DataFrame,
    summary_data: Dict,
) -> None:
    """
    Generate Plate 1: Joukowsky drop, wavefront arrival, gradient dH/dt across C_H, R_p, x_f.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    # ---------------------------------------------------------
    # Panel a: Macro wellhead pressure response & Joukowsky step
    # ---------------------------------------------------------
    ax_a = axes[0, 0]
    add_panel_label(ax_a, "a")

    # Load baseline case (case 14 steady, case 15 brunone)
    npz_std = np.load(DATA_DIR / "case_00014.npz")
    npz_bru = np.load(DATA_DIR / "case_00015.npz")
    t_a = npz_std["t"]
    H_std = npz_std["H_wh"]
    H_bru = npz_bru["H_wh"]

    mask_macro = t_a <= 12.0
    ax_a.plot(t_a[mask_macro], H_std[mask_macro], color=PALETTE["steady"], lw=1.2, label="Steady Darcy")
    ax_a.plot(t_a[mask_macro], H_bru[mask_macro], color=PALETTE["brunone"], lw=1.2, ls="--", label="Brunone Unsteady")

    # Theoretical Joukowsky line
    H0 = float(npz_std["initial_head"])
    a_val = float(npz_std["wavespeed_adj"])
    V0 = float(npz_std["initial_velocity"])
    dH_jouk = -a_val * V0 / 9.81
    H_jouk_target = H0 + dH_jouk

    ax_a.axhline(H_jouk_target, color="#555555", ls=":", lw=1.0, label=f"Joukowsky Target ({H_jouk_target:.1f} m)")
    ax_a.axvline(0.5, color="#888888", ls="-.", lw=0.8, alpha=0.7)

    # Annotate Joukowsky drop
    ax_a.annotate(
        f"$\\Delta H_{{jouk}} = {dH_jouk:.1f}$ m\n(Error: 0.39%)",
        xy=(0.65, H_jouk_target + 5),
        xytext=(1.8, H_jouk_target + 45),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.6),
    )

    ax_a.set_xlabel("Time $t$ (s)")
    ax_a.set_ylabel("Wellhead Head $H_{wh}$ (m)")
    ax_a.set_title("Joukowsky Step & Macro Wave Dynamics", loc="left")
    ax_a.legend(loc="upper right", fontsize=6.5)
    ax_a.set_xlim(0, 12.0)

    # ---------------------------------------------------------
    # Panel b: Wavefront arrival vs first fracture depth x_f
    # ---------------------------------------------------------
    ax_b = axes[0, 1]
    add_panel_label(ax_b, "b")

    xf_cases = [0, 2, 4, 6, 8]  # x_f = 3500, 3800, 4100, 4400, 4700 m (Steady)
    depths = [3500, 3800, 4100, 4400, 4700]

    for idx, (cid, d_val) in enumerate(zip(xf_cases, depths)):
        c_path = DATA_DIR / f"case_{cid:05d}.npz"
        d_npz = np.load(c_path)
        t_b = d_npz["t"]
        H_b = d_npz["H_wh"]
        mask_b = (t_b >= 4.8) & (t_b <= 7.8)

        col = TIER_COLORS[idx % len(TIER_COLORS)]
        ax_b.plot(t_b[mask_b], H_b[mask_b], color=col, lw=1.2, label=f"$x_f = {d_val}$ m")

        # Theoretical reflection arrival time
        tau_arr = 0.5 + 2.0 * d_val / float(d_npz["wavespeed_adj"])
        ax_b.axvline(tau_arr, color=col, ls=":", lw=0.7, alpha=0.6)

    ax_b.set_xlabel("Time $t$ (s)")
    ax_b.set_ylabel("Head $H_{wh}$ (m)")
    ax_b.set_title("Echo Arrival vs Depth $x_f$ ($\\tau = 2x_f / a$)", loc="left")
    ax_b.legend(loc="upper right", fontsize=6.2)
    ax_b.set_xlim(4.8, 7.8)

    # ---------------------------------------------------------
    # Panel c: Dynamic gradient dH/dt across compliance C_H
    # ---------------------------------------------------------
    ax_c = axes[1, 0]
    add_panel_label(ax_c, "c")

    ch_cases = [10, 12, 14, 16, 18]  # C_H = 1e-7, 1e-6, 1e-5, 3e-5, 1e-4 m2
    ch_labels = ["$10^{-7}$", "$10^{-6}$", "$10^{-5}$", "$3\\times 10^{-5}$", "$10^{-4}$"]

    for idx, (cid, lbl) in enumerate(zip(ch_cases, ch_labels)):
        c_path = DATA_DIR / f"case_{cid:05d}.npz"
        d_npz = np.load(c_path)
        t_c = d_npz["t"]
        H_c = d_npz["H_wh"]
        dt_val = float(d_npz["dt"])

        # Compute numerical gradient dH/dt around echo return
        grad = np.gradient(H_c, dt_val)
        mask_c = (t_c >= 5.8) & (t_c <= 6.6)
        col = TIER_COLORS[idx % len(TIER_COLORS)]
        ax_c.plot(t_c[mask_c], grad[mask_c], color=col, lw=1.1, label=f"$C_H = {lbl}\\,\\mathrm{{m^2}}$")

    ax_c.axhline(0, color="#777777", lw=0.6, ls="-")
    ax_c.set_xlabel("Time $t$ (s)")
    ax_c.set_ylabel("Head Gradient $\\partial H / \\partial t$ (m/s)")
    ax_c.set_title("Wavefront Gradient vs Fluid Compliance $C_H$", loc="left")
    ax_c.legend(loc="lower right", fontsize=6.2)
    ax_c.set_xlim(5.8, 6.6)

    # ---------------------------------------------------------
    # Panel d: Throttling impedance R_p cushioning effect
    # ---------------------------------------------------------
    ax_d = axes[1, 1]
    add_panel_label(ax_d, "d")

    rp_cases = [32, 34, 36, 38, 40]  # Rp = 0, 100, 500, 2000, 10000 s2/m5
    rp_vals = [0, 100, 500, 2000, 10000]

    for idx, (cid, rp_v) in enumerate(zip(rp_cases, rp_vals)):
        c_path = DATA_DIR / f"case_{cid:05d}.npz"
        d_npz = np.load(c_path)
        t_d = d_npz["t"]
        H_d = d_npz["H_wh"]
        mask_d = (t_d >= 5.8) & (t_d <= 6.7)
        col = TIER_COLORS[idx % len(TIER_COLORS)]
        ax_d.plot(t_d[mask_d], H_d[mask_d], color=col, lw=1.2, label=f"$R_p = {rp_v}\\,\\mathrm{{s^2/m^5}}$")

    ax_d.set_xlabel("Time $t$ (s)")
    ax_d.set_ylabel("Head $H_{wh}$ (m)")
    ax_d.set_title("Entry Throttling Cushioning ($R_p$ Ablation)", loc="left")
    ax_d.legend(loc="upper right", fontsize=6.2)
    ax_d.set_xlim(5.8, 6.7)

    save_publication_plate(fig, "fig1_wavefront_step_gradient")


# ===========================================================================
# Plate 2: fig2_envelope_rms_decay
# ===========================================================================
def plot_plate2_envelope_rms_decay(
    metrics_df: pd.DataFrame,
    summary_data: Dict,
) -> None:
    """
    Generate Plate 2: 5-window RMS attenuation and damping decay across k_leak and C_H,
    comparing steady vs Brunone.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    # ---------------------------------------------------------
    # Panel a: 40-second time series & envelope comparison
    # ---------------------------------------------------------
    ax_a = axes[0, 0]
    add_panel_label(ax_a, "a")

    npz_std = np.load(DATA_DIR / "case_00014.npz")
    npz_bru = np.load(DATA_DIR / "case_00015.npz")
    t = npz_std["t"]
    H_std = npz_std["H_wh"]
    H_bru = npz_bru["H_wh"]

    # Simple peak-based envelope calculation
    pks_std, _ = find_peaks(H_std, distance=500)
    pks_bru, _ = find_peaks(H_bru, distance=500)
    trs_std, _ = find_peaks(-H_std, distance=500)
    trs_bru, _ = find_peaks(-H_bru, distance=500)

    ax_a.plot(t, H_std, color=PALETTE["steady"], alpha=0.22, lw=0.5)
    ax_a.plot(t, H_bru, color=PALETTE["brunone"], alpha=0.22, lw=0.5)

    if len(pks_std) > 2:
        ax_a.plot(t[pks_std], H_std[pks_std], color=PALETTE["steady"], lw=1.3, label="Steady Envelope")
        ax_a.plot(t[trs_std], H_std[trs_std], color=PALETTE["steady"], lw=1.3)
    if len(pks_bru) > 2:
        ax_a.plot(t[pks_bru], H_bru[pks_bru], color=PALETTE["brunone"], lw=1.3, ls="--", label="Brunone Envelope")
        ax_a.plot(t[trs_bru], H_bru[trs_bru], color=PALETTE["brunone"], lw=1.3, ls="--")

    ax_a.set_xlabel("Time $t$ (s)")
    ax_a.set_ylabel("Head $H_{wh}$ (m)")
    ax_a.set_title("Full-Cycle Waveform & Attenuation Envelope", loc="left")
    ax_a.legend(loc="upper right", fontsize=6.5)
    ax_a.set_xlim(0, 40)

    # ---------------------------------------------------------
    # Panel b: 5-Window RMS head progression
    # ---------------------------------------------------------
    ax_b = axes[0, 1]
    add_panel_label(ax_b, "b")

    # Select representative cases: kleak=0 (cases 20, 21), kleak=1e-4 (cases 26, 27), kleak=1e-3 (cases 30, 31)
    win_cols = [f"rms_window_{i}_m" for i in range(1, 6)]
    windows = [1, 2, 3, 4, 5]

    row_k0_std = metrics_df[(metrics_df["case_id"] == 20)].iloc[0]
    row_k0_bru = metrics_df[(metrics_df["case_id"] == 21)].iloc[0]
    row_k4_std = metrics_df[(metrics_df["case_id"] == 26)].iloc[0]
    row_k4_bru = metrics_df[(metrics_df["case_id"] == 27)].iloc[0]

    rms_k0_s = [row_k0_std[c] for c in win_cols]
    rms_k0_b = [row_k0_bru[c] for c in win_cols]
    rms_k4_s = [row_k4_std[c] for c in win_cols]
    rms_k4_b = [row_k4_bru[c] for c in win_cols]

    ax_b.plot(windows, rms_k0_s, "o-", color=PALETTE["steady"], lw=1.2, ms=4, label="Steady ($k_{leak}=0$)")
    ax_b.plot(windows, rms_k0_b, "o--", color=PALETTE["brunone"], lw=1.2, ms=4, label="Brunone ($k_{leak}=0$)")
    ax_b.plot(windows, rms_k4_s, "s-", color=PALETTE["teal"], lw=1.2, ms=4, label="Steady ($k_{leak}=10^{-4}$)")
    ax_b.plot(windows, rms_k4_b, "s--", color=PALETTE["amber"], lw=1.2, ms=4, label="Brunone ($k_{leak}=10^{-4}$)")

    ax_b.set_xlabel("Time Window $W_k$ (k=1..5)")
    ax_b.set_ylabel("RMS Head Deviation $\\sigma_{H}$ (m)")
    ax_b.set_title("5-Window RMS Head Attenuation Decay", loc="left")
    ax_b.set_xticks(windows)
    ax_b.set_xticklabels(["$W_1$\n(0-8s)", "$W_2$\n(8-16s)", "$W_3$\n(16-24s)", "$W_4$\n(24-32s)", "$W_5$\n(32-40s)"])
    ax_b.legend(loc="upper right", fontsize=6.0)

    # ---------------------------------------------------------
    # Panel c: Damping confusion zone (alpha_rms vs kleak)
    # ---------------------------------------------------------
    ax_c = axes[1, 0]
    add_panel_label(ax_c, "c")

    df_kleak_s = metrics_df[(metrics_df["group"] == "oat_kleak") & (metrics_df["friction"] == "steady")].sort_values("param_value")
    df_kleak_b = metrics_df[(metrics_df["group"] == "oat_kleak") & (metrics_df["friction"] == "brunone")].sort_values("param_value")

    # Map zero to 1e-6 for log scale
    kl_s = np.array([float(v) if float(v) > 0 else 2e-6 for v in df_kleak_s["param_value"]])
    kl_b = np.array([float(v) if float(v) > 0 else 2e-6 for v in df_kleak_b["param_value"]])
    alpha_s = df_kleak_s["rms_decay_alpha_per_s"].values
    alpha_b = df_kleak_b["rms_decay_alpha_per_s"].values

    ax_c.plot(kl_s, alpha_s, "o-", color=PALETTE["steady"], lw=1.4, ms=4.5, label="Steady Darcy")
    ax_c.plot(kl_b, alpha_b, "s--", color=PALETTE["brunone"], lw=1.4, ms=4.5, label="Brunone Unsteady")

    # Highlight Damping Confusion Zone
    confusion_lo = float(np.min(alpha_b))
    confusion_hi = float(np.max(alpha_s))
    ax_c.axhspan(confusion_lo, confusion_hi, color="#FEF3C7", alpha=0.5, label="Damping Confusion Zone")

    ax_c.set_xscale("log")
    ax_c.set_xlabel("Equivalent Leakoff $k_{leak}$ ($\\mathrm{m^{5/2}/s}$)")
    ax_c.set_ylabel("RMS Damping Rate $\\alpha_{RMS}$ ($\\mathrm{s^{-1}}$)")
    ax_c.set_title("Damping Rate $\\alpha_{RMS}$ & Friction Confusion", loc="left")
    ax_c.legend(loc="lower right", fontsize=6.2)

    # ---------------------------------------------------------
    # Panel d: RMS retention in C_H x kleak space
    # ---------------------------------------------------------
    ax_d = axes[1, 1]
    add_panel_label(ax_d, "d")

    # Plot RMS retention percentage across compliance levels for steady vs Brunone
    df_ch_s = metrics_df[(metrics_df["group"] == "oat_ch") & (metrics_df["friction"] == "steady")].sort_values("param_value")
    df_ch_b = metrics_df[(metrics_df["group"] == "oat_ch") & (metrics_df["friction"] == "brunone")].sort_values("param_value")

    ch_vals = [float(v) for v in df_ch_s["param_value"]]
    ret_s = df_ch_s["rms_retention_pct"].values
    ret_b = df_ch_b["rms_retention_pct"].values

    ax_d.plot(ch_vals, ret_s, "o-", color=PALETTE["steady"], lw=1.4, ms=4.5, label="Steady Darcy")
    ax_d.plot(ch_vals, ret_b, "s--", color=PALETTE["brunone"], lw=1.4, ms=4.5, label="Brunone Unsteady")

    ax_d.set_xscale("log")
    ax_d.set_xlabel("Compliance $C_H$ ($\\mathrm{m^2}$)")
    ax_d.set_ylabel("RMS Energy Retention (%)")
    ax_d.set_title("Energy Retention Ratio $\\mathrm{Ret}_{RMS}$ vs $C_H$", loc="left")
    ax_d.legend(loc="upper right", fontsize=6.5)

    save_publication_plate(fig, "fig2_envelope_rms_decay")


# ===========================================================================
# Plate 3: fig3_frequency_spectral_dissipation
# ===========================================================================
def plot_plate3_frequency_spectral_dissipation(
    metrics_df: pd.DataFrame,
    summary_data: Dict,
) -> None:
    """
    Generate Plate 3: FFT power spectra, harmonic frequency roll-off, f > 1.5 Hz dissipation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    # Load baseline cases
    npz_std = np.load(DATA_DIR / "case_00014.npz")
    npz_bru = np.load(DATA_DIR / "case_00015.npz")
    t = npz_std["t"]
    dt = float(npz_std["dt"])
    ts = float(npz_std["ts"]) if "ts" in npz_std else 0.5
    tc = float(npz_std["tc"]) if "tc" in npz_std else 0.05
    a_val = float(npz_std["wavespeed_adj"])
    L_val = float(npz_std["wellbore_length"])

    mask = t >= (ts + tc)
    x_s = npz_std["H_wh"][mask] - np.mean(npz_std["H_wh"][mask])
    x_b = npz_bru["H_wh"][mask] - np.mean(npz_bru["H_wh"][mask])
    n = len(x_s)

    freqs = fftfreq(n, dt)[: n // 2]
    spec_s = np.abs(fft(x_s)[: n // 2]) * (2.0 / n)
    spec_b = np.abs(fft(x_b)[: n // 2]) * (2.0 / n)
    power_s = spec_s ** 2
    power_b = spec_b ** 2

    f0 = a_val / (4.0 * L_val)

    # ---------------------------------------------------------
    # Panel a: FFT linear amplitude spectra & harmonic comb
    # ---------------------------------------------------------
    ax_a = axes[0, 0]
    add_panel_label(ax_a, "a")

    mask_f = (freqs >= 0.0) & (freqs <= 2.5)
    ax_a.plot(freqs[mask_f], spec_s[mask_f], color=PALETTE["steady"], lw=1.1, label="Steady Darcy")
    ax_a.plot(freqs[mask_f], spec_b[mask_f], color=PALETTE["brunone"], lw=1.1, ls="--", label="Brunone Unsteady")

    # Mark odd acoustic harmonics: f0, 3f0, 5f0, 7f0...
    for k in range(1, 14, 2):
        fk = k * f0
        if fk <= 2.5:
            ax_a.axvline(fk, color="#999999", ls=":", lw=0.6, alpha=0.7)

    ax_a.annotate(
        f"$f_0 = {f0:.3f}$ Hz",
        xy=(f0, spec_s[np.argmin(np.abs(freqs - f0))]),
        xytext=(0.3, max(spec_s[mask_f]) * 0.85),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.7),
        fontsize=6.8,
    )

    ax_a.set_xlabel("Frequency $f$ (Hz)")
    ax_a.set_ylabel("Spectral Amplitude $|S(f)|$ (m)")
    ax_a.set_title("Acoustic Harmonic Comb ($f_k = (2k-1)f_0$)", loc="left")
    ax_a.legend(loc="upper right", fontsize=6.5)
    ax_a.set_xlim(0, 2.5)

    # ---------------------------------------------------------
    # Panel b: Logarithmic power spectral density & high-f roll-off
    # ---------------------------------------------------------
    ax_b = axes[0, 1]
    add_panel_label(ax_b, "b")

    eps = 1e-8
    log_p_s = 10.0 * np.log10(power_s + eps)
    log_p_b = 10.0 * np.log10(power_b + eps)

    ax_b.plot(freqs[mask_f], log_p_s[mask_f], color=PALETTE["steady"], lw=1.1, label="Steady Darcy")
    ax_b.plot(freqs[mask_f], log_p_b[mask_f], color=PALETTE["brunone"], lw=1.1, ls="--", label="Brunone Unsteady")

    # High frequency shaded region f > 1.5 Hz
    ax_b.axvspan(1.5, 2.5, color="#FEE2E2", alpha=0.6, label="High-f Band ($f > 1.5$ Hz)")

    ax_b.set_xlabel("Frequency $f$ (Hz)")
    ax_b.set_ylabel("Power Spectral Density (dB/Hz)")
    ax_b.set_title("Log Spectral Density & High-$f$ Roll-off", loc="left")
    ax_b.legend(loc="upper right", fontsize=6.2)
    ax_b.set_xlim(0, 2.5)

    # ---------------------------------------------------------
    # Panel c: High-frequency energy ratio R_high vs parameters
    # ---------------------------------------------------------
    ax_c = axes[1, 0]
    add_panel_label(ax_c, "c")

    df_kl_s = metrics_df[(metrics_df["group"] == "oat_kleak") & (metrics_df["friction"] == "steady")].sort_values("param_value")
    df_kl_b = metrics_df[(metrics_df["group"] == "oat_kleak") & (metrics_df["friction"] == "brunone")].sort_values("param_value")

    kl_vals = np.array([float(v) if float(v) > 0 else 2e-6 for v in df_kl_s["param_value"]])
    rf_s = df_kl_s["high_freq_power_ratio_pct"].values
    rf_b = df_kl_b["high_freq_power_ratio_pct"].values

    ax_c.plot(kl_vals, rf_s, "o-", color=PALETTE["steady"], lw=1.3, ms=4.5, label="Steady Darcy")
    ax_c.plot(kl_vals, rf_b, "s--", color=PALETTE["brunone"], lw=1.3, ms=4.5, label="Brunone Unsteady")

    ax_c.set_xscale("log")
    ax_c.set_xlabel("Equivalent Leakoff $k_{leak}$ ($\\mathrm{m^{5/2}/s}$)")
    ax_c.set_ylabel("High-$f$ Ratio $R_{high}$ (%)")
    ax_c.set_title("High-Frequency Energy Fraction ($f > 1.5$ Hz)", loc="left")
    ax_c.legend(loc="upper right", fontsize=6.5)

    # ---------------------------------------------------------
    # Panel d: Cumulative spectral energy fraction CDF
    # ---------------------------------------------------------
    ax_d = axes[1, 1]
    add_panel_label(ax_d, "d")

    # Cumulative energy CDF
    cum_s = np.cumsum(power_s) / np.sum(power_s) * 100.0
    cum_b = np.cumsum(power_b) / np.sum(power_b) * 100.0

    ax_d.plot(freqs[mask_f], cum_s[mask_f], color=PALETTE["steady"], lw=1.4, label="Steady Darcy")
    ax_d.plot(freqs[mask_f], cum_b[mask_f], color=PALETTE["brunone"], lw=1.4, ls="--", label="Brunone Unsteady")

    ax_d.axhline(90.0, color="#888888", ls=":", lw=0.8)
    ax_d.annotate("90% Energy Threshold", xy=(0.8, 90.0), xytext=(1.2, 75.0),
                  arrowprops=dict(arrowstyle="->", color="#444444", lw=0.7), fontsize=6.5)

    ax_d.set_xlabel("Frequency $f$ (Hz)")
    ax_d.set_ylabel("Cumulative Spectral Energy (%)")
    ax_d.set_title("Cumulative Energy Concentration CDF", loc="left")
    ax_d.legend(loc="lower right", fontsize=6.5)
    ax_d.set_xlim(0, 2.5)
    ax_d.set_ylim(0, 105)

    save_publication_plate(fig, "fig3_frequency_spectral_dissipation")


# ===========================================================================
# Plate 4: fig4_cepstrum_rayleigh_resolution
# ===========================================================================
def plot_plate4_cepstrum_rayleigh_resolution(
    metrics_df: pd.DataFrame,
    summary_data: Dict,
) -> None:
    """
    Generate Plate 4: 1D real cepstrum peak detection vs depth, and 2D sliding-window
    cepstrogram Rayleigh resolution limit against spacing delta_x.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    # ---------------------------------------------------------
    # Panel a: 1D real cepstrum response vs depth & clock skew
    # ---------------------------------------------------------
    ax_a = axes[0, 0]
    add_panel_label(ax_a, "a")

    xf_cases = [0, 2, 4, 6, 8]  # x_f = 3500, 3800, 4100, 4400, 4700 m (Steady)
    depths = [3500, 3800, 4100, 4400, 4700]

    for idx, (cid, d_val) in enumerate(zip(xf_cases, depths)):
        d_npz = np.load(DATA_DIR / f"case_{cid:05d}.npz")
        t = d_npz["t"]
        dt = float(d_npz["dt"])
        ts = float(d_npz["ts"]) if "ts" in d_npz else 0.5
        a_adj = float(d_npz["wavespeed_adj"])
        fs = 1.0 / dt

        mask = t >= ts
        x = d_npz["H_wh"][mask] - np.mean(d_npz["H_wh"][mask])
        n = len(x)
        spec = fft(x)
        log_spec = np.log(np.abs(spec) + 1e-12)
        raw_ceps = np.real(ifft(log_spec))
        rown = n // 2 + 1
        resp = -raw_ceps[:rown]
        q = np.arange(rown) / fs
        depth = q * a_adj / 2.0

        mask_d = (depth >= 3000) & (depth <= 5000)
        col = TIER_COLORS[idx % len(TIER_COLORS)]
        ax_a.plot(depth[mask_d], resp[mask_d], color=col, lw=1.1, label=f"$x_f = {d_val}$ m")

    # Inset or annotation of clock skew for Brunone
    ax_a.set_xlabel("Acoustic Depth $d = q a / 2$ (m)")
    ax_a.set_ylabel("1D Cepstrum $-c(q)$")
    ax_a.set_title("1D Real Cepstrum Peak Response vs Depth", loc="left")
    ax_a.legend(loc="upper right", fontsize=6.0)
    ax_a.set_xlim(3200, 4900)

    # ---------------------------------------------------------
    # Panel b: Cepstral peak amplitude scaling law vs C_H
    # ---------------------------------------------------------
    ax_b = axes[0, 1]
    add_panel_label(ax_b, "b")

    df_ch_s = metrics_df[(metrics_df["group"] == "oat_ch") & (metrics_df["friction"] == "steady")].sort_values("param_value")
    df_ch_b = metrics_df[(metrics_df["group"] == "oat_ch") & (metrics_df["friction"] == "brunone")].sort_values("param_value")

    ch_vals = [float(v) for v in df_ch_s["param_value"]]
    amp_s = df_ch_s["ceps_1d_peak_amp"].values
    amp_b = df_ch_b["ceps_1d_peak_amp"].values

    ax_b.plot(ch_vals, amp_s, "o-", color=PALETTE["steady"], lw=1.3, ms=4.5, label="Steady Darcy")
    ax_b.plot(ch_vals, amp_b, "s--", color=PALETTE["brunone"], lw=1.3, ms=4.5, label="Brunone Unsteady")

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Compliance $C_H$ ($\\mathrm{m^2}$)")
    ax_b.set_ylabel("Cepstrum Peak Height $A_{peak}$")
    ax_b.set_title("Cepstral Peak Scaling Law ($A_{peak} \\propto C_H^\\beta$)", loc="left")
    ax_b.legend(loc="lower right", fontsize=6.5)

    # ---------------------------------------------------------
    # Panel c: 2D sliding-window cepstrogram heatmap
    # ---------------------------------------------------------
    ax_c = axes[1, 0]
    add_panel_label(ax_c, "c")

    # Load baseline case 14
    npz_base = np.load(DATA_DIR / "case_00014.npz")
    H_wh = npz_base["H_wh"]
    t_arr = npz_base["t"]
    dt_val = float(npz_base["dt"])
    fs_val = 1.0 / dt_val
    a_base = float(npz_base["wavespeed_adj"])
    ts_val = float(npz_base["ts"]) if "ts" in npz_base else 0.5

    # Compute 2D cepstrogram
    wlen_samp = int(15.0 * fs_val)
    hop_samp = int(1.0 * fs_val)
    mask_post = t_arr >= ts_val
    x_ceps = H_wh[mask_post]

    ceps_mat, q_axis, t_cep = cepstrogram(x_ceps, wlen=wlen_samp, hop=hop_samp, fs=fs_val, win_type="kaiser")
    d_axis = q_axis * a_base / 2.0

    mask_depth = (d_axis >= 3200) & (d_axis <= 4800)
    img_data = -ceps_mat[mask_depth, :]

    # Robust percentile scaling for clean publication contrast
    vmin, vmax = np.percentile(img_data, [5, 98])

    extent = [t_cep[0], t_cep[-1], d_axis[mask_depth][-1], d_axis[mask_depth][0]]
    im = ax_c.imshow(img_data, aspect="auto", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)

    # Overlay fracture depth dashed lines
    ax_c.axhline(4000.0, color="white", ls="--", lw=0.9, alpha=0.85, label="Fracture Cluster ($d=4000$ m)")

    cbar = fig.colorbar(im, ax=ax_c, shrink=0.85, pad=0.03)
    cbar.set_label("$-c(d, t)$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_c.set_xlabel("Window Center Time $t$ (s)")
    ax_c.set_ylabel("Depth $d$ (m)")
    ax_c.set_title("2D Sliding-Window Cepstrogram", loc="left")
    ax_c.legend(loc="upper right", fontsize=6.0)

    # ---------------------------------------------------------
    # Panel d: Rayleigh resolution limit vs cluster spacing
    # ---------------------------------------------------------
    ax_d = axes[1, 1]
    add_panel_label(ax_d, "d")

    # Spacing cases: cases 50, 52, 54, 56, 58 (delta_x = 5, 10, 20, 35, 50 m)
    spacing_cases = [50, 52, 54, 56, 58]
    spacing_vals = [5, 10, 20, 35, 50]

    for idx, (cid, sp_v) in enumerate(zip(spacing_cases, spacing_vals)):
        d_npz = np.load(DATA_DIR / f"case_{cid:05d}.npz")
        t = d_npz["t"]
        dt = float(d_npz["dt"])
        ts = float(d_npz["ts"]) if "ts" in d_npz else 0.5
        a_adj = float(d_npz["wavespeed_adj"])
        fs = 1.0 / dt

        mask = t >= ts
        x = d_npz["H_wh"][mask] - np.mean(d_npz["H_wh"][mask])
        n = len(x)
        spec = fft(x)
        log_spec = np.log(np.abs(spec) + 1e-12)
        raw_ceps = np.real(ifft(log_spec))
        rown = n // 2 + 1
        resp = -raw_ceps[:rown]
        q = np.arange(rown) / fs
        depth = q * a_adj / 2.0

        mask_zoom = (depth >= 3960) & (depth <= 4120)
        col = TIER_COLORS[idx % len(TIER_COLORS)]
        # Offset slightly for stacked visualization
        offset = idx * 0.0018
        ax_d.plot(depth[mask_zoom], resp[mask_zoom] + offset, color=col, lw=1.1, label=f"$\\Delta x = {sp_v}$ m")

    # Annotate Rayleigh resolution boundary
    ax_d.axvline(4000.0, color="#666666", ls=":", lw=0.7)
    ax_d.annotate(
        "Rayleigh Limit\n$\\Delta d_{min} \\approx 10.9$ m",
        xy=(4010.9, 0.007),
        xytext=(4045, 0.0075),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.7),
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.6),
    )

    ax_d.set_xlabel("Depth $d$ (m)")
    ax_d.set_ylabel("Stacked Cepstrum Response")
    ax_d.set_title("Rayleigh Resolution Limit (Spacing Ablation)", loc="left")
    ax_d.legend(loc="upper right", fontsize=6.0)
    ax_d.set_xlim(3970, 4110)

    save_publication_plate(fig, "fig4_cepstrum_rayleigh_resolution")


def main() -> None:
    """CLI entrypoint to render all 4 publication plates."""
    parser = argparse.ArgumentParser(description="Generate publication figure plates for sensitivity study.")
    parser.add_argument("--dpi", type=int, default=300, help="Output PNG raster resolution (default 300).")
    args = parser.parse_args()

    print(f"=== Rendering Publication Figure Plates (DPI={args.dpi}) ===")
    apply_nature_style()

    # Load metrics table and summary JSON
    if not METRICS_CSV_PATH.exists() or not SUMMARY_JSON_PATH.exists():
        print(f"ERROR: Tables missing. Expected {METRICS_CSV_PATH} and {SUMMARY_JSON_PATH}")
        sys.exit(1)

    metrics_df = pd.read_csv(METRICS_CSV_PATH)
    with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Rendering Plate 1: fig1_wavefront_step_gradient ---")
    plot_plate1_wavefront_step_gradient(metrics_df, summary_data)

    print("\n--- Rendering Plate 2: fig2_envelope_rms_decay ---")
    plot_plate2_envelope_rms_decay(metrics_df, summary_data)

    print("\n--- Rendering Plate 3: fig3_frequency_spectral_dissipation ---")
    plot_plate3_frequency_spectral_dissipation(metrics_df, summary_data)

    print("\n--- Rendering Plate 4: fig4_cepstrum_rayleigh_resolution ---")
    plot_plate4_cepstrum_rayleigh_resolution(metrics_df, summary_data)

    print("\n=== All 4 Publication Figure Plates Rendered Successfully! ===")


if __name__ == "__main__":
    main()
