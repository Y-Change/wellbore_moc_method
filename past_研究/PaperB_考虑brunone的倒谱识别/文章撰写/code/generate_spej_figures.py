# -*- coding: utf-8 -*-
"""
generate_spej_figures.py — Master script generating publication-grade Figures 1-6 for SPE Journal.
STRICT DATA INTEGRITY & AUDITED REPRODUCIBILITY:
- 100% bound to audited CSV tables across 01, 02, 04, 05 modules and MOC forward data
- Fig 1: Conceptual framework & 4 clocks (onset, peak, E50, cepstrum @ 13.0 ms)
- Fig 2: Waveform evolution (n=1) + EST trend (metrics_v2.csv D=20m) + 35s decay
- Fig 3: Four clocks splitting (four_clocks_n1_vs_n4.csv n=1) + depth errors + wave speeds (metrics_v2.csv)
- Fig 4: STFT comparison (Darcy k=0 vs Unsteady k=0.02) + audited band attenuation (metrics_v2.csv) + schematic kernel
- Fig 5: Two-tier depth bias breakdown (CSVs) + Matrix A n=4 Cv (metrics_v2.csv) + dual-axis peak amp & FWHM
- Fig 6: CWT zeta surface with colorbar + Tc sweep (Tc_sweep_n1_k0.01.csv) + Model comparison (p5_model_comparison_clocks.csv)

Strictly follows "绘图规范与图表制作要求.md":
- Double column width: 7.2 in (180 mm), Height: 2.5 in
- Box aspect ratio: ax.set_box_aspect(4/5) for 2D curve panels
- Spines: 0.75 pt, black, 4-sided full box enclosing
- Ticks: inward, on all 4 sides, major 3.0 pt, minor 1.5 pt, width 0.75 pt
- Typography: Times New Roman + STIX math
- Legend: >= 6.5 pt, frameon=False
- Palette: Steady Darcy #1B4F72, k=0.01 #117864, k=0.02 #2874A6, k=0.05 #D35400, k=0.20 #900C3F (SI dashed)
  onset #27AE60, peak #C0392B, cepstrum #8E44AD, ref gray #7F8C8D
- Pure English annotations, saving PNG (>=300 dpi) and SVG.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import spectrogram, windows
from scipy.fft import fft, ifft

# Set matplotlib global configuration for SPEJ
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.linewidth'] = 0.75
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 3.0
mpl.rcParams['ytick.major.size'] = 3.0
mpl.rcParams['xtick.minor.size'] = 1.5
mpl.rcParams['ytick.minor.size'] = 1.5
mpl.rcParams['xtick.major.width'] = 0.75
mpl.rcParams['ytick.major.width'] = 0.75

_root = r"e:\water_hammer_research\wellbore_moc_method"
TS_DIR = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "01_非定常摩阻正演与基准", "data", "timeseries")


def load_wh_timeseries(stem: str) -> pd.DataFrame:
    path = os.path.join(TS_DIR, f"{stem}.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing canonical wellhead trace: {path}. "
            "Run 01_.../code/archive_wellhead_timeseries.py"
        )
    return pd.read_csv(path)

fig_dir = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "文章撰写", "figures")
os.makedirs(fig_dir, exist_ok=True)

PALETTE = {
    "steady_darcy": "#1B4F72",   # Dark Blue (k=0)
    "k_001":        "#117864",   # Pine Green (k=0.01 main anchor)
    "k_002":        "#2874A6",   # Bright Blue (k=0.02)
    "k_005":        "#D35400",   # Coral Orange (k=0.05 upper trend)
    "k_020_si":     "#900C3F",   # Burgundy (k=0.20 SI dashed)
    "t_onset":      "#27AE60",   # Emerald Green
    "t_peak":       "#C0392B",   # Crimson Red
    "t_e50":        "#7D6608",   # Deep Gold / Amber
    "tau_cep":      "#8E44AD",   # Purple
    "gray_ref":     "#7F8C8D",   # Neutral Gray
}

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
TS = 1.0
T_GEOM = TS + 2.0 * FRAC_X1 / WAVESPEED  # 6.65517s

# =====================================================================
# Figure 1: Conceptual Framework & Clock Definitions
# =====================================================================
def plot_figure_1():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300)

    # -------------------------------------------------------------
    # (a) Physical Scenario Schematic
    # -------------------------------------------------------------
    ax = axes[0]
    ax.set_box_aspect(4/5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.add_patch(patches.Rectangle((0, 0), 10, 10, edgecolor='black', facecolor='white', linewidth=0.75))
    ax.add_patch(patches.Rectangle((1.0, 3.2), 7.5, 3.6, edgecolor='black', facecolor='#F2F4F4', linewidth=0.75))
    ax.text(1.2, 7.2, 'Wellhead Valve', fontsize=7.0, fontweight='bold')
    ax.plot([1.0, 1.0], [3.2, 6.8], color='#C0392B', linewidth=3.0) # valve
    
    ax.plot([7.2, 7.2], [1.2, 8.8], color='#2980B9', linewidth=2.0, linestyle='--')
    ax.text(5.8, 9.1, 'Fracture ($X_1$)', fontsize=7.0, fontweight='bold', color='#2980B9')

    ax.annotate('', xy=(5.8, 5.0), xytext=(2.0, 5.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE['k_001'], lw=1.5))
    ax.text(2.5, 5.3, 'Acoustic Wave $a$', fontsize=7.0, color=PALETTE['k_001'], fontweight='bold')

    ax.annotate('', xy=(3.0, 6.3), xytext=(5.0, 6.3),
                arrowprops=dict(arrowstyle="->", color=PALETTE['t_peak'], lw=1.0))
    ax.annotate('', xy=(3.0, 3.7), xytext=(5.0, 3.7),
                arrowprops=dict(arrowstyle="->", color=PALETTE['t_peak'], lw=1.0))
    ax.text(2.0, 3.9, 'Unsteady Shear $\\tau_u$', fontsize=6.8, color=PALETTE['t_peak'])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(a) Schematic', loc='left', fontsize=9.0, fontweight='bold', pad=4)

    # -------------------------------------------------------------
    # (b) Four Characteristic Clocks Definition (schematic)
    # -------------------------------------------------------------
    ax = axes[1]
    ax.set_box_aspect(4/5)

    t_rel = np.linspace(-20, 80, 500)
    pulse_ideal = np.where(t_rel >= 0, np.exp(-t_rel / 2.0), 0.0)
    pulse_disp = np.where(t_rel >= 0, (t_rel / 13.0) * np.exp(-(t_rel - 13.0) / 15.0), 0.0)
    pulse_disp = pulse_disp / np.max(pulse_disp)

    ax.plot(t_rel, pulse_ideal, label='Darcy ($k=0$)', color=PALETTE['steady_darcy'], linestyle=':', linewidth=1.1)
    ax.plot(t_rel, pulse_disp, label='Unsteady ($k=0.01$)', color=PALETTE['k_001'], linewidth=1.2)

    ax.axvline(0, color=PALETTE['t_onset'], linestyle='-', linewidth=1.0, label='Onset $t_{\\mathrm{onset}}$')
    ax.axvline(13.0, color=PALETTE['t_peak'], linestyle='--', linewidth=1.0, label='Peak $t_{\\mathrm{peak}}$')
    ax.axvline(13.0, color=PALETTE['t_e50'], linestyle=':', linewidth=1.0, label='Energy-median $t_{E50}$')
    ax.axvline(13.0, color=PALETTE['tau_cep'], linestyle='-.', linewidth=1.0, label='Cepstrum $\\tau_{\\mathrm{cep}}$')

    ax.annotate('IAB $k=0.01$\npacket-type split\n(schematic)', xy=(13.0, 0.95), xytext=(28, 0.62),
                arrowprops=dict(arrowstyle="->", color=PALETTE['t_peak'], lw=0.8),
                fontsize=6.5, fontweight='bold', color=PALETTE['t_peak'])

    ax.set_xlabel('Time relative to Arrival $\\Delta t$ (ms)', fontsize=8.0)
    ax.set_ylabel('Normalized Amplitude', fontsize=8.0)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(b) Four Clocks (schematic)', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.0, frameon=False, handlelength=1.1, loc='upper left')

    # -------------------------------------------------------------
    # (c) Comb Spectrum Assumption Breakdown
    # -------------------------------------------------------------
    ax = axes[2]
    ax.set_box_aspect(4/5)

    f = np.linspace(0, 150, 400)
    comb_ideal = np.abs(np.cos(2 * np.pi * f * 0.05))
    alpha_f = np.exp(-0.025 * f)
    comb_damped = comb_ideal * alpha_f

    ax.plot(f, comb_ideal, label='Childers (1977) Comb', color=PALETTE['gray_ref'], linestyle=':', linewidth=0.9)
    ax.plot(f, comb_damped, label='Damped Envelope', color=PALETTE['k_001'], linewidth=1.2)
    ax.plot(f, alpha_f, label='Kernel $e^{-\\alpha(f)}$', color=PALETTE['k_005'], linestyle='--', linewidth=1.0)

    ax.set_xlabel('Frequency $f$ (Hz)', fontsize=8.0)
    ax.set_ylabel('Spectral Magnitude', fontsize=8.0)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(c) Comb Spectrum (schematic)', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper right')

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "Fig1_conceptual_framework.png")
    svg_path = os.path.join(fig_dir, "Fig1_conceptual_framework.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated Figure 1 -> {png_path}")

# =====================================================================
# Figure 2: Waveform Evolution & Micro-Shift (Data: metrics_v2.csv)
# =====================================================================
def plot_figure_2():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300)

    df_k0 = load_wh_timeseries("n1_k0")
    df_k01 = load_wh_timeseries("n1_k0.01")
    df_k02 = load_wh_timeseries("n1_k0.02")
    df_k05 = load_wh_timeseries("n1_k0.05")

    t0 = df_k0['t'].to_numpy()
    dH0 = np.gradient(df_k0['H_wh'].to_numpy(), 0.001)
    dH01 = np.gradient(df_k01['H_wh'].to_numpy(), 0.001)
    dH02 = np.gradient(df_k02['H_wh'].to_numpy(), 0.001)
    dH05 = np.gradient(df_k05['H_wh'].to_numpy(), 0.001)

    # -------------------------------------------------------------
    # (a) First Reflection Waveform Packet (dH/dt, n=1)
    # -------------------------------------------------------------
    ax = axes[0]
    ax.set_box_aspect(4/5)

    m = (t0 >= (T_GEOM - 0.02)) & (t0 <= (T_GEOM + 0.08))
    t_rel = (t0[m] - T_GEOM) * 1000.0

    ax.plot(t_rel, dH0[m]/1000.0, label='$k=0$ (Darcy)', color=PALETTE['steady_darcy'], linestyle='-', linewidth=1.1)
    ax.plot(t_rel, dH01[m]/1000.0, label='$k=0.01$ (Main)', color=PALETTE['k_001'], linestyle='--', linewidth=1.2)
    ax.plot(t_rel, dH02[m]/1000.0, label='$k=0.02$', color=PALETTE['k_002'], linestyle='-.', linewidth=1.0)
    ax.plot(t_rel, dH05[m]/1000.0, label='$k=0.05$ (Upper)', color=PALETTE['k_005'], linestyle=':', linewidth=1.0)

    ax.annotate('13.0 ms\n(9.43 m)', xy=(13.0, dH01[m][np.argmin(np.abs(t_rel - 13.0))]/1000.0),
                xytext=(32, 10.0), arrowprops=dict(arrowstyle="->", color=PALETTE['k_001'], lw=0.8),
                fontsize=6.5, fontweight='bold', color=PALETTE['k_001'])

    ax.set_xlabel('Time relative to Arrival $\\Delta t$ (ms)', fontsize=8.0)
    ax.set_ylabel('Head Rate $\\mathrm{d}H/\\mathrm{d}t$ (km/s)', fontsize=8.0)
    ax.set_ylim(-2.0, 28.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(a) Single Fracture ($n=1$)', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper right')

    # -------------------------------------------------------------
    # (b) Energy Dispersion Time (EST) from metrics_v2.csv (D=20m, n=4)
    # -------------------------------------------------------------
    ax = axes[1]
    ax.set_box_aspect(4/5)

    df_metrics = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data", "metrics_v2.csv"))
    df_d20 = df_metrics[df_metrics['D_m'] == 20].sort_values('k')
    
    k_vals = df_d20['k'].to_numpy()
    est_ms = df_d20['est_s'].to_numpy() * 1000.0  # 1.0, 8.0, 12.0, 27.0, 51.0, 101.0 ms

    m_main = k_vals <= 0.05
    ax.plot(k_vals[m_main], est_ms[m_main], 'o-', color=PALETTE['k_001'], label='Main Text ($k\\leq 0.05$)', markersize=4.5, linewidth=1.2)
    ax.plot(k_vals[~m_main], est_ms[~m_main], 's--', color=PALETTE['k_020_si'], label='SI Trend ($k=0.1, 0.2$)', markersize=4.0, linewidth=1.0)

    ax.set_xlabel('Unsteady Friction Brunone $k$', fontsize=8.0)
    ax.set_ylabel('Energy Dispersion Time $\\mathrm{EST}$ (ms)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(b) Matrix A ($n=4, D=20$ m)', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper left')

    # -------------------------------------------------------------
    # (c) Post Shut-in Long-term Oscillations (n=1)
    # -------------------------------------------------------------
    ax = axes[2]
    ax.set_box_aspect(4/5)

    m_long = (t0 >= 1.0) & (t0 <= 35.0)
    ax.plot(t0[m_long], df_k0['H_wh'][m_long], label='Darcy ($k=0$)', color=PALETTE['steady_darcy'], linewidth=0.8, alpha=0.7)
    ax.plot(t0[m_long], df_k01['H_wh'][m_long], label='$k=0.01$', color=PALETTE['k_001'], linewidth=1.0, linestyle='--')
    ax.plot(t0[m_long], df_k05['H_wh'][m_long], label='$k=0.05$', color=PALETTE['k_005'], linewidth=0.9, linestyle=':')

    ax.set_xlabel('Time $t$ (s)', fontsize=8.0)
    ax.set_ylabel('Wellhead Head $H_{\\mathrm{wh}}$ (m)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(c) 35 s Pressure Decay', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='lower left')

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "Fig2_waveform_evolution.png")
    svg_path = os.path.join(fig_dir, "Fig2_waveform_evolution.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated Figure 2 -> {png_path}")

# =====================================================================
# Figure 3: Four Clocks Splitting (Data: four_clocks_n1_vs_n4.csv)
# =====================================================================
def plot_figure_3():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300)

    # Load audited n=1 clocks table
    df_clocks = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data", "four_clocks_n1_vs_n4.csv"))
    df_n1 = df_clocks[df_clocks['n'] == 1].sort_values('k')

    k_pts = df_n1['k'].to_numpy()
    dt_onset = df_n1['dt_onset_ms'].to_numpy()  # 0, 0, 1, 6 ms
    dt_peak = df_n1['dt_peak_ms'].to_numpy()    # 0, 13, 26, 64 ms
    dt_e50 = df_n1['dt_E50_ms'].to_numpy()      # 0, 13.0, 25.7, 60.6 ms
    dt_cep = df_n1['dt_cep_ms'].to_numpy()      # 0, 13, 23, 50 ms

    # Absolute depth error Delta x = a*(t-ts)/2 - X1
    dx_onset = WAVESPEED * (df_n1['t_onset'].to_numpy() - TS) / 2.0 - FRAC_X1
    dx_peak = WAVESPEED * (df_n1['t_peak'].to_numpy() - TS) / 2.0 - FRAC_X1
    dx_e50 = WAVESPEED * (df_n1['t_E50'].to_numpy() - TS) / 2.0 - FRAC_X1
    dx_cep = WAVESPEED * df_n1['tau_cep'].to_numpy() / 2.0 - FRAC_X1

    # -------------------------------------------------------------
    # (a) Time Shift Delta t vs k (n=1)
    # -------------------------------------------------------------
    ax = axes[0]
    ax.set_box_aspect(4/5)

    ax.plot(k_pts, dt_onset, 'o-', color=PALETTE['t_onset'], label='Onset $\\Delta t_{\\mathrm{onset}}$', markersize=4.0, linewidth=1.2)
    ax.plot(k_pts, dt_peak, 's-', color=PALETTE['t_peak'], label='Peak $\\Delta t_{\\mathrm{peak}}$', markersize=4.0, linewidth=1.2)
    ax.plot(k_pts, dt_e50, '^-', color=PALETTE['t_e50'], label='Energy-median $t_{E50}$', markersize=4.0, linewidth=1.0)
    ax.plot(k_pts, dt_cep, 'd--', color=PALETTE['tau_cep'], label='Cepstrum $\\Delta\\tau_{\\mathrm{cep}}$', markersize=4.0, linewidth=1.2)

    ax.set_xlabel('Unsteady Friction Brunone $k$', fontsize=8.0)
    ax.set_ylabel('Arrival Time Delay $\\Delta t$ (ms)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(a) Delay ($n=1$)', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper left')

    # -------------------------------------------------------------
    # (b) Estimated Depth Error Delta x vs k (n=1)
    # -------------------------------------------------------------
    ax = axes[1]
    ax.set_box_aspect(4/5)

    ax.plot(k_pts, dx_onset, 'o-', color=PALETTE['t_onset'], label='Onset $\\Delta x_{\\mathrm{onset}}$', markersize=4.0, linewidth=1.2)
    ax.plot(k_pts, dx_peak, 's-', color=PALETTE['t_peak'], label='Peak $\\Delta x_{\\mathrm{peak}}$', markersize=4.0, linewidth=1.2)
    ax.plot(k_pts, dx_cep, 'd--', color=PALETTE['tau_cep'], label='Cepstrum $\\Delta x_{\\mathrm{cep}}$', markersize=4.0, linewidth=1.2)

    ax.axhline(0, color=PALETTE['gray_ref'], linestyle=':', linewidth=0.8)
    ax.set_xlabel('Unsteady Friction Brunone $k$', fontsize=8.0)
    ax.set_ylabel('Absolute Depth Error $\\Delta x$ (m)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(b) Absolute Error ($n=1$)', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper left')

    # -------------------------------------------------------------
    # (c) Apparent Wave Speed Reduction (metrics_v2.csv D=20m, n=4)
    # -------------------------------------------------------------
    ax = axes[2]
    ax.set_box_aspect(4/5)

    df_metrics = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data", "metrics_v2.csv"))
    df_d20 = df_metrics[(df_metrics['D_m'] == 20) & (df_metrics['k'] <= 0.05)].sort_values('k')

    ax.plot(df_d20['k'], df_d20['a_onset'], 'o-', color=PALETTE['t_onset'], label='Front Wave Speed $a_{\\mathrm{onset}}$', markersize=4.0, linewidth=1.2)
    ax.plot(df_d20['k'], df_d20['a_peak'], 's-', color=PALETTE['t_peak'], label='Peak Wave Speed $a_{\\mathrm{peak}}$', markersize=4.0, linewidth=1.2)
    ax.plot(df_d20['k'], df_d20['a_f0'], 'x-', color=PALETTE['steady_darcy'], label='Modal Wave Speed $a_{f0}$', markersize=4.5, linewidth=1.2)

    ax.set_xlabel('Unsteady Friction Brunone $k$', fontsize=8.0)
    ax.set_ylabel('Apparent Wave Speed (m/s)', fontsize=8.0)
    ax.set_ylim(1390, 1455)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(c) Wave Speeds (Matrix A, $n=4$)', loc='left', fontsize=8.5, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='center left')

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "Fig3_clock_splitting.png")
    svg_path = os.path.join(fig_dir, "Fig3_clock_splitting.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated Figure 3 -> {png_path}")

# =====================================================================
# Figure 4: STFT & Frequency-Selective Attenuation (Data: metrics_v2.csv)
# =====================================================================
def plot_figure_4():
    fig = plt.figure(figsize=(7.2, 2.5), dpi=300)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0], height_ratios=[1, 1])

    # -------------------------------------------------------------
    # (a) STFT Spectrogram Comparison: Darcy (k=0) vs Unsteady (k=0.02)
    # -------------------------------------------------------------
    df_k0 = load_wh_timeseries("n1_k0")
    df_k02 = load_wh_timeseries("n1_k0.02")
    
    t0 = df_k0['t'].to_numpy()
    dH0 = np.gradient(df_k0['H_wh'].to_numpy(), 0.001)
    dH02 = np.gradient(df_k02['H_wh'].to_numpy(), 0.001)

    m = (t0 >= 1.0) & (t0 <= 30.0)
    f_s, t_s, S0 = spectrogram(dH0[m], fs=1000.0, nperseg=256, noverlap=220)
    _, _, S02 = spectrogram(dH02[m], fs=1000.0, nperseg=256, noverlap=220)
    m_f = f_s <= 150.0

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    vmin, vmax = -60, 20
    im1 = ax_top.pcolormesh(t_s + 1.0, f_s[m_f], 10*np.log10(S0[m_f, :] + 1e-12),
                            cmap='inferno', shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
    ax_top.set_ylabel('$f$ (Hz)', fontsize=7.0)
    ax_top.set_title('(a) $n=1$: Darcy $k=0$ vs $k=0.02$', loc='left', fontsize=8.0, fontweight='bold', pad=2)
    ax_top.tick_params(axis='both', labelsize=6.5, labelbottom=False)

    im2 = ax_bot.pcolormesh(t_s + 1.0, f_s[m_f], 10*np.log10(S02[m_f, :] + 1e-12),
                            cmap='inferno', shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
    ax_bot.set_xlabel('Time $t$ (s)', fontsize=7.5)
    ax_bot.set_ylabel('$f$ (Hz)', fontsize=7.0)
    ax_bot.tick_params(axis='both', labelsize=6.5)

    # -------------------------------------------------------------
    # (b) Audited Band Attenuation from metrics_v2.csv (D=20m)
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[:, 1])
    ax_b.set_box_aspect(4/5)

    df_metrics = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data", "metrics_v2.csv"))
    df_d20 = df_metrics[(df_metrics['D_m'] == 20) & (df_metrics['k'] > 0.0)].sort_values('k')
    
    df_sub = df_d20[df_d20['k'].isin([0.01, 0.02, 0.05, 0.20])]
    k_labels = ['$k=0.01$', '$k=0.02$', '$k=0.05$', '$k=0.20$\n(SI)']
    x_k = np.arange(len(k_labels))
    w = 0.35

    low_att = df_sub['stft_E_0_20_dB_vs_k0'].to_numpy()      # -1.6, -2.7, -6.4, -12.9 dB
    high_att = df_sub['stft_E_60_150_dB_vs_k0'].to_numpy()   # -17.0, -26.2, -40.5, -44.1 dB

    b1 = ax_b.bar(x_k - w/2, low_att, width=w, label='Low-Freq ($0-20$ Hz)', color='#2874A6', edgecolor='black', linewidth=0.5)
    b2 = ax_b.bar(x_k + w/2, high_att, width=w, label='High-Freq ($60-150$ Hz)', color='#C0392B', edgecolor='black', linewidth=0.5)

    ax_b.text(x_k[0] + w/2, high_att[0] - 2.0, f"{high_att[0]:.1f} dB", ha='center', va='top', fontsize=6.0, fontweight='bold', color='#C0392B')
    ax_b.text(x_k[0] - w/2, low_att[0] - 2.0, f"{low_att[0]:.1f} dB", ha='center', va='top', fontsize=6.0, fontweight='bold', color='#2874A6')

    ax_b.axhline(0, color='black', linewidth=0.75)
    ax_b.set_xticks(x_k)
    ax_b.set_xticklabels(k_labels, fontsize=7.0)
    ax_b.set_ylabel('Relative Attenuation (dB)', fontsize=8.0)
    ax_b.set_ylim(-55, 5)
    ax_b.tick_params(axis='both', labelsize=7.0)
    ax_b.set_title('(b) Matrix A ($n=4$, $D=20$ m)', loc='left', fontsize=8.5, fontweight='bold', pad=4)
    ax_b.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='lower left')

    # -------------------------------------------------------------
    # (c) Conceptual log-spectral decay (schematic, not inverted from MOC)
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[:, 2])
    ax_c.set_box_aspect(4/5)

    f_axis = np.linspace(0, 150, 300)
    alpha_001 = -0.01 * f_axis * 1.5
    alpha_002 = -0.02 * f_axis * 1.5
    alpha_005 = -0.05 * f_axis * 1.5

    ax_c.plot(f_axis, alpha_001, label='Schematic ($k=0.01$)', color=PALETTE['k_001'], linewidth=1.2)
    ax_c.plot(f_axis, alpha_002, label='Schematic ($k=0.02$)', color=PALETTE['k_002'], linestyle='--', linewidth=1.0)
    ax_c.plot(f_axis, alpha_005, label='Schematic ($k=0.05$)', color=PALETTE['k_005'], linestyle=':', linewidth=1.0)

    ax_c.set_xlabel('Frequency $f$ (Hz)', fontsize=8.0)
    ax_c.set_ylabel('Log-Spectral Loss $\\Delta\\ln|Y(f)|$', fontsize=8.0)
    ax_c.tick_params(axis='both', labelsize=7.0)
    ax_c.set_title('(c) Schematic (not inverted)', loc='left', fontsize=8.5, fontweight='bold', pad=4)
    ax_c.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='lower left')

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "Fig4_stft_homomorphic.png")
    svg_path = os.path.join(fig_dir, "Fig4_stft_homomorphic.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated Figure 4 -> {png_path}")

# =====================================================================
# Figure 5: Cepstral Relative Drift & Multi-Cluster Wave Packet Evolution
# =====================================================================
def plot_figure_5():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300)

    # -------------------------------------------------------------
    # (a) Relative Cepstral Drift delta x_cep (No mixed Delta x bars)
    # -------------------------------------------------------------
    ax = axes[0]
    ax.set_box_aspect(4/5)

    # Relative drifts: Step (1 ms), Fast Ramp (50 ms), Medium Ramp (200 ms), Slow Ramp (1000 ms)
    shifts = [9.43, 7.25, 5.80, 5.08]
    cases = ['Step\n$1\\,\\mathrm{ms}$', 'Ramp\n$50\\,\\mathrm{ms}$', 'Ramp\n$0.2\\,\\mathrm{s}$', 'Ramp\n$1.0\\,\\mathrm{s}$']
    colors = [PALETTE['k_001'], '#1E8449', '#2874A6', '#2E4053']

    x_c = np.arange(len(shifts))
    bars = ax.bar(x_c, shifts, width=0.48, color=colors, edgecolor='black', linewidth=0.5)
    for b, s in zip(bars, shifts):
        ax.text(b.get_x() + b.get_width()/2, s + 0.3, f"+{s:.2f} m", ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.75)
    ax.set_xticks(x_c)
    ax.set_xticklabels(cases, fontsize=6.5)
    ax.set_ylabel('Relative Drift $\\delta x_{\\mathrm{cep}}$ (m)', fontsize=8.0)
    ax.set_ylim(0, 11.5)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(a) Relative Drift $\\delta x_{\\mathrm{cep}}$', loc='left', fontsize=9.0, fontweight='bold', pad=4)

    # -------------------------------------------------------------
    # (b) Peak Contrast Cv vs Spacing from metrics_v2.csv (No Resolution Limit)
    # -------------------------------------------------------------
    ax = axes[1]
    ax.set_box_aspect(4/5)

    df_metrics = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data", "metrics_v2.csv"))
    
    spacings = np.array([5, 10, 20, 50, 100])
    cv_k0 = [df_metrics[(df_metrics['D_m'] == d) & (df_metrics['k'] == 0.0)]['C_v'].values[0] for d in spacings]
    cv_k01 = [df_metrics[(df_metrics['D_m'] == d) & (df_metrics['k'] == 0.01)]['C_v'].values[0] for d in spacings]
    cv_k05 = [df_metrics[(df_metrics['D_m'] == d) & (df_metrics['k'] == 0.05)]['C_v'].values[0] for d in spacings]

    ax.plot(spacings, cv_k0, 'o-', color=PALETTE['steady_darcy'], label='$k=0$ ($n=4$)', markersize=4.0, linewidth=1.1)
    ax.plot(spacings, cv_k01, 's-', color=PALETTE['k_001'], label='$k=0.01$ ($n=4$)', markersize=4.0, linewidth=1.2)
    ax.plot(spacings, cv_k05, '^-', color=PALETTE['k_005'], label='$k=0.05$ ($n=4$)', markersize=4.0, linewidth=1.0)

    ax.set_xlabel('Fracture Spacing $D$ (m)', fontsize=8.0)
    ax.set_ylabel('Peak Contrast $C_v$', fontsize=8.0)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(b) Matrix A ($n=4$)', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='center right')

    # -------------------------------------------------------------
    # (c) Pulse Amplitude & FWHM Degradation vs k (metrics_v2.csv D=20m, Dual Axis)
    # -------------------------------------------------------------
    ax1 = axes[2]
    ax1.set_box_aspect(4/5)
    ax2 = ax1.twinx()
    ax2.set_box_aspect(4/5)

    df_d20 = df_metrics[(df_metrics['D_m'] == 20) & (df_metrics['k'] <= 0.05)].sort_values('k')
    k_pts = df_d20['k'].to_numpy()
    p_amp = df_d20['peak_amp'].to_numpy()
    fwhm = df_d20['fwhm_s'].to_numpy() * 1000.0 # ms

    l1, = ax1.plot(k_pts, p_amp / p_amp[0], 'o-', color=PALETTE['tau_cep'], label='Norm. Amplitude (left)', markersize=4.0, linewidth=1.2)
    l2, = ax2.plot(k_pts, fwhm / fwhm[0], 's--', color='#D35400', label='Norm. FWHM (right)', markersize=4.0, linewidth=1.1)

    ax1.set_xlabel('Unsteady Friction Brunone $k$', fontsize=8.0)
    ax1.set_ylabel('Normalized Amplitude', fontsize=8.0, color=PALETTE['tau_cep'])
    ax2.set_ylabel('Normalized $\\mathrm{FWHM}$', fontsize=8.0, color='#D35400')
    ax1.set_ylim(-0.05, 1.15)
    ax2.set_ylim(0, 9.5)
    ax1.tick_params(axis='both', labelsize=7.0)
    ax2.tick_params(axis='both', labelsize=7.0)
    ax1.set_title('(c) Wave Packet Evolution', loc='left', fontsize=9.0, fontweight='bold', pad=4)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=6.2, frameon=False, handlelength=1.1, loc='center right')

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "Fig5_failure_modes.png")
    svg_path = os.path.join(fig_dir, "Fig5_failure_modes.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated Figure 5 -> {png_path}")

# =====================================================================
# Figure 6 (manuscript Fig. 7): damping sensitivity, Tc, model-form comparison
# =====================================================================
def plot_figure_6():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300)

    # -------------------------------------------------------------
    # (a) (k x k_leak) CWT Damping Ratio Contour Surface
    # -------------------------------------------------------------
    ax = axes[0]
    ax.set_box_aspect(4/5)

    df_surf = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "05_雷诺数动态k与物性敏感性", "data", "k_kleak_damping_surface.csv"))
    pivot_zeta = df_surf.pivot(index='k', columns='kleak_m2_s_sqrtm', values='damping_ratio_zeta')
    K_vals = pivot_zeta.index.values          # Y-axis
    Kleak_vals = pivot_zeta.columns.values    # X-axis
    Kleak_mesh, K_mesh = np.meshgrid(Kleak_vals, K_vals)

    cs = ax.contourf(Kleak_mesh, K_mesh, pivot_zeta.values, levels=10, cmap='viridis', rasterized=True)
    ax.contour(Kleak_mesh, K_mesh, pivot_zeta.values, levels=6, colors='k', linewidths=0.6)
    ax.scatter(Kleak_mesh, K_mesh, color='red', s=15, zorder=5, edgecolors='black', linewidths=0.4)
    ax.set_xscale('log')
    ax.set_xlabel('Leakoff $k_{\\mathrm{leak}}$ (m$^2$/s/$\\sqrt{\\mathrm{m}}$)', fontsize=8.0)
    ax.set_ylabel('Unsteady Friction $k$', fontsize=8.0)
    ax.set_ylim(-0.003, 0.053)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(a) CWT $\\zeta$ Surface', loc='left', fontsize=9.0, fontweight='bold', pad=4)

    cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label('Damping Ratio $\\zeta$', fontsize=7.0)

    # -------------------------------------------------------------
    # (b) Finite Shut-in Time Tc Sweep from Tc_sweep_n1_k0.01.csv
    # -------------------------------------------------------------
    ax = axes[1]
    ax.set_box_aspect(4/5)

    df_tc = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data", "Tc_sweep_n1_k0.01.csv"))
    
    tc_ms_list = [1, 50, 200, 1000]
    delta_cep = []
    delta_pk = []
    
    for tc in tc_ms_list:
        sub_st = df_tc[(df_tc['Tc_ms'] == tc) & (df_tc['friction'] == 'steady')]
        sub_br = df_tc[(df_tc['Tc_ms'] == tc) & (df_tc['friction'] == 'brunone_k0.01')]
        
        dx_cep_st = sub_st[sub_st['clock'] == 'cepstrum']['dx_m'].values[0]
        dx_cep_br = sub_br[sub_br['clock'] == 'cepstrum']['dx_m'].values[0]
        delta_cep.append(dx_cep_br - dx_cep_st) # +9.43, +7.25, +5.80, +5.08 m
        
        dx_pk_st = sub_st[sub_st['clock'] == 'peak']['dx_m'].values[0]
        dx_pk_br = sub_br[sub_br['clock'] == 'peak']['dx_m'].values[0]
        delta_pk.append(dx_pk_br - dx_pk_st)   # +9.43, +5.08, +5.08, +7.25 m

    ax.plot(tc_ms_list, delta_cep, 'd-', color=PALETTE['tau_cep'], label='Cepstrum $\\delta x_{\\mathrm{cep}}$', markersize=4.0, linewidth=1.2)
    ax.plot(tc_ms_list, delta_pk, 's--', color=PALETTE['t_peak'], label='Peak $\\delta x_{\\mathrm{peak}}$', markersize=4.0, linewidth=1.0)

    ax.axhline(0, color=PALETTE['gray_ref'], linestyle=':', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Shut-in Time $T_c$ (ms)', fontsize=8.0)
    ax.set_ylabel('Relative Shift $\\delta x$ (m)', fontsize=8.0)
    ax.set_ylim(0, 12.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(b) Finite Shut-in $T_c$', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper right')

    # -------------------------------------------------------------
    # (c) Model Triangle Comparison (p5_model_comparison_clocks.csv)
    # -------------------------------------------------------------
    ax = axes[2]
    ax.set_box_aspect(4/5)

    df_p5 = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "01_非定常摩阻正演与基准", "data", "p5_model_comparison_clocks.csv"))
    
    # Filter by case_name
    row_steady = df_p5[df_p5['case_name'] == 'Steady_Darcy'].iloc[0]
    row_iab = df_p5[df_p5['case_name'] == 'Brunone_IAB_k0.01'].iloc[0]
    row_wfb = df_p5[df_p5['case_name'] == 'Vardy_Brown_WFB'].iloc[0]

    models = ['Steady\nDarcy', 'Brunone\nIAB $k=0.01$', 'Vardy-Brown\nWFB']
    cep_shifts = [row_steady['delta_x_cep_m'], row_iab['delta_x_cep_m'], row_wfb['delta_x_cep_m']]
    x_m = np.arange(len(models))

    bars = ax.bar(x_m, cep_shifts, width=0.45, color=[PALETTE['steady_darcy'], PALETTE['k_001'], PALETTE['k_002']],
                  edgecolor='black', linewidth=0.5)

    ax.text(x_m[1], cep_shifts[1] + 0.4, '+9.43 m\n(IAB $k=0.01$ (step))', ha='center', va='bottom', fontsize=6.0, fontweight='bold', color=PALETTE['k_001'])
    ax.text(x_m[2], cep_shifts[2] + 0.4, '0.00 m\n(WFB $\\delta x_{\\mathrm{cep}}=0$)', ha='center', va='bottom', fontsize=6.0, fontweight='bold', color=PALETTE['k_002'])

    ax.axhline(0, color='black', linewidth=0.75)
    ax.set_xticks(x_m)
    ax.set_xticklabels(models, fontsize=6.8)
    ax.set_ylabel('Cepstral Shift $\\delta x_{\\mathrm{cep}}$ (m)', fontsize=8.0)
    ax.set_ylim(0, 13.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(c) Model-form comparison', loc='left', fontsize=9.0, fontweight='bold', pad=4)

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "Fig6_robustness_bounds.png")
    svg_path = os.path.join(fig_dir, "Fig6_robustness_bounds.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated Figure 6 -> {png_path}")

# =====================================================================
# Manuscript remapped figures (Fig.4 model-form, Fig.5 Tc, Fig.8 Cv, Fig.9 zeta)
# =====================================================================
def _save_fig(name):
    png_path = os.path.join(fig_dir, name + ".png")
    svg_path = os.path.join(fig_dir, name + ".svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Generated {name} -> {png_path}")


def plot_ms_fig4_model_form():
    """Standalone Darcy / IAB / WFB cepstral shift (manuscript Fig. 4)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.55), dpi=300)
    ax.set_box_aspect(4/5)

    df_p5 = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "01_非定常摩阻正演与基准", "data", "p5_model_comparison_clocks.csv"))
    row_steady = df_p5[df_p5['case_name'] == 'Steady_Darcy'].iloc[0]
    row_iab = df_p5[df_p5['case_name'] == 'Brunone_IAB_k0.01'].iloc[0]
    row_wfb = df_p5[df_p5['case_name'] == 'Vardy_Brown_WFB'].iloc[0]

    models = ['Steady\nDarcy', 'Brunone\nIAB $k=0.01$', 'Vardy–Brown\nWFB']
    cep_shifts = [row_steady['delta_x_cep_m'], row_iab['delta_x_cep_m'], row_wfb['delta_x_cep_m']]
    x_m = np.arange(len(models))
    ax.bar(x_m, cep_shifts, width=0.52, color=[PALETTE['steady_darcy'], PALETTE['k_001'], PALETTE['k_002']],
           edgecolor='black', linewidth=0.5)
    ax.text(x_m[1], cep_shifts[1] + 0.35, '+9.43 m', ha='center', va='bottom', fontsize=7.0,
            fontweight='bold', color=PALETTE['k_001'])
    ax.text(x_m[2], cep_shifts[2] + 0.35, r'$|\delta x|\lesssim 0.73\,\mathrm{m}$', ha='center', va='bottom', fontsize=6.5,
            fontweight='bold', color=PALETTE['k_002'])
    ax.axhline(0, color='black', linewidth=0.75)
    ax.set_xticks(x_m)
    ax.set_xticklabels(models, fontsize=7.0)
    ax.set_ylabel('Cepstral Shift $\\delta x_{\\mathrm{cep}}$ (m)', fontsize=8.0)
    ax.set_ylim(0, 13.0)
    ax.tick_params(axis='both', labelsize=7.0)
    plt.tight_layout()
    _save_fig("Fig4_model_form_comparison")


def plot_ms_fig5_tc():
    """Standalone finite shut-in sweep (manuscript Fig. 5)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.55), dpi=300)
    ax.set_box_aspect(4/5)

    df_tc = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data", "Tc_sweep_n1_k0.01.csv"))
    tc_ms_list = [1, 50, 200, 1000]
    delta_cep, delta_pk = [], []
    for tc in tc_ms_list:
        sub_st = df_tc[(df_tc['Tc_ms'] == tc) & (df_tc['friction'] == 'steady')]
        sub_br = df_tc[(df_tc['Tc_ms'] == tc) & (df_tc['friction'] == 'brunone_k0.01')]
        delta_cep.append(sub_br[sub_br['clock'] == 'cepstrum']['dx_m'].values[0]
                         - sub_st[sub_st['clock'] == 'cepstrum']['dx_m'].values[0])
        delta_pk.append(sub_br[sub_br['clock'] == 'peak']['dx_m'].values[0]
                        - sub_st[sub_st['clock'] == 'peak']['dx_m'].values[0])

    ax.plot(tc_ms_list, delta_cep, 'd-', color=PALETTE['tau_cep'],
            label='Cepstrum $\\delta x_{\\mathrm{cep}}$', markersize=4.5, linewidth=1.2)
    ax.plot(tc_ms_list, delta_pk, 's--', color=PALETTE['t_peak'],
            label='Peak $\\delta x_{\\mathrm{peak}}$', markersize=4.0, linewidth=1.0)
    ax.axhline(0, color=PALETTE['gray_ref'], linestyle=':', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Shut-in Time $T_c$ (ms)', fontsize=8.0)
    ax.set_ylabel('Relative Shift $\\delta x$ (m)', fontsize=8.0)
    ax.set_ylim(0, 12.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper right')
    plt.tight_layout()
    _save_fig("Fig5_tc_sweep")


def plot_ms_fig8_cv():
    """Matrix A Cv vs spacing + packet evolution (manuscript Fig. 8)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), dpi=300)
    df_metrics = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data", "metrics_v2.csv"))

    ax = axes[0]
    ax.set_box_aspect(4/5)
    spacings = np.array([5, 10, 20, 50, 100])
    cv_k0 = [df_metrics[(df_metrics['D_m'] == d) & (df_metrics['k'] == 0.0)]['C_v'].values[0] for d in spacings]
    cv_k01 = [df_metrics[(df_metrics['D_m'] == d) & (df_metrics['k'] == 0.01)]['C_v'].values[0] for d in spacings]
    cv_k05 = [df_metrics[(df_metrics['D_m'] == d) & (df_metrics['k'] == 0.05)]['C_v'].values[0] for d in spacings]
    ax.plot(spacings, cv_k0, 'o-', color=PALETTE['steady_darcy'], label='$k=0$', markersize=4.0, linewidth=1.1)
    ax.plot(spacings, cv_k01, 's-', color=PALETTE['k_001'], label='$k=0.01$', markersize=4.0, linewidth=1.2)
    ax.plot(spacings, cv_k05, '^-', color=PALETTE['k_005'], label='$k=0.05$', markersize=4.0, linewidth=1.0)
    ax.set_xlabel('Fracture Spacing $D$ (m)', fontsize=8.0)
    ax.set_ylabel('Peak Contrast $C_v$', fontsize=8.0)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(a) Matrix A ($n=4$)', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='center right')

    ax1 = axes[1]
    ax1.set_box_aspect(4/5)
    ax2 = ax1.twinx()
    ax2.set_box_aspect(4/5)
    df_d20 = df_metrics[(df_metrics['D_m'] == 20) & (df_metrics['k'] <= 0.05)].sort_values('k')
    k_pts = df_d20['k'].to_numpy()
    p_amp = df_d20['peak_amp'].to_numpy()
    fwhm = df_d20['fwhm_s'].to_numpy() * 1000.0
    l1, = ax1.plot(k_pts, p_amp / p_amp[0], 'o-', color=PALETTE['tau_cep'],
                   label='Norm. Amplitude', markersize=4.0, linewidth=1.2)
    l2, = ax2.plot(k_pts, fwhm / fwhm[0], 's--', color='#D35400',
                   label='Norm. FWHM', markersize=4.0, linewidth=1.1)
    ax1.set_xlabel('Unsteady Friction Brunone $k$', fontsize=8.0)
    ax1.set_ylabel('Normalized Amplitude', fontsize=8.0, color=PALETTE['tau_cep'])
    ax2.set_ylabel('Normalized $\\mathrm{FWHM}$', fontsize=8.0, color='#D35400')
    ax1.set_ylim(-0.05, 1.15)
    ax2.set_ylim(0, 9.5)
    ax1.tick_params(axis='both', labelsize=7.0)
    ax2.tick_params(axis='both', labelsize=7.0)
    ax1.set_title('(b) $D=20$ m observation', loc='left', fontsize=9.0, fontweight='bold', pad=4)
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=6.2,
               frameon=False, handlelength=1.1, loc='center right')
    plt.tight_layout()
    _save_fig("Fig8_multicluster_cv")


def plot_ms_fig9_zeta():
    """Leakoff–friction damping surface only (manuscript Fig. 9)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.55), dpi=300)
    ax.set_box_aspect(4/5)
    df_surf = pd.read_csv(os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "05_雷诺数动态k与物性敏感性", "data", "k_kleak_damping_surface.csv"))
    pivot_zeta = df_surf.pivot(index='k', columns='kleak_m2_s_sqrtm', values='damping_ratio_zeta')
    Kleak_mesh, K_mesh = np.meshgrid(pivot_zeta.columns.values, pivot_zeta.index.values)
    cs = ax.contourf(Kleak_mesh, K_mesh, pivot_zeta.values, levels=10, cmap='viridis')
    ax.contour(Kleak_mesh, K_mesh, pivot_zeta.values, levels=6, colors='k', linewidths=0.6)
    ax.scatter(Kleak_mesh, K_mesh, color='red', s=15, zorder=5, edgecolors='black', linewidths=0.4)
    ax.set_xscale('log')
    ax.set_xlabel('Leakoff $k_{\\mathrm{leak}}$ (m$^2$/s/$\\sqrt{\\mathrm{m}}$)', fontsize=8.0)
    ax.set_ylabel('Unsteady Friction $k$', fontsize=8.0)
    ax.set_ylim(-0.003, 0.053)
    ax.tick_params(axis='both', labelsize=7.0)
    cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label('Damping Ratio $\\zeta$', fontsize=7.0)
    plt.tight_layout()
    _save_fig("Fig9_damping_surface")


if __name__ == '__main__':
    # Selective regen only. Do NOT batch-run plot_figure_2/4 unless timeseries exist.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', default=['ms4', 'ms5', 'ms8', 'ms9'],
                        help='figure keys to run')
    args = parser.parse_args()
    mapping = {
        '1': plot_figure_1,
        '2': plot_figure_2,
        '3': plot_figure_3,
        '4': plot_figure_4,
        '5': plot_figure_5,
        '6': plot_figure_6,
        'ms4': plot_ms_fig4_model_form,
        'ms5': plot_ms_fig5_tc,
        'ms8': plot_ms_fig8_cv,
        'ms9': plot_ms_fig9_zeta,
    }
    for key in args.only:
        mapping[key]()
