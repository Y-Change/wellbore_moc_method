# -*- coding: utf-8 -*-
"""
Gantt p6: Comprehensive Numerical Robustness & Boundary Sensitivity Analysis.
Evaluates stability of IAB k=0.01 clock splitting and delta_x_cep across:
1. Grid refinement (dt = 0.5 ms vs 1.0 ms)
2. Window type (Hann, Hamming, Rectangular) x Window duration (10s, 30s, 50s)
3. Input channel (H, dH/dt, Bandpassed H)
4. Additive Gaussian white noise (SNR = 20, 40, 60 dB, 5 seeds each)

Complies strictly with SPE Journal plotting standards and repository palette.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, butter, sosfiltfilt
from scipy.fft import fft, ifft

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
dir_01 = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "01_非定常摩阻正演与基准")
dir_02 = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽")
dir_04 = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "04_倒谱模糊与系统性深度偏差")

sim_p6 = os.path.join(_root, "output", "analysis", "brunone_spacing_effect", "p6_robustness")
sim_p5 = os.path.join(_root, "output", "analysis", "brunone_spacing_effect", "p5_models")

os.makedirs(os.path.join(dir_02, "code"), exist_ok=True)
os.makedirs(os.path.join(dir_02, "data"), exist_ok=True)
os.makedirs(os.path.join(dir_02, "figures"), exist_ok=True)
os.makedirs(os.path.join(dir_04, "data"), exist_ok=True)

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
TS = 1.0
TAU_GEOM = 2.0 * FRAC_X1 / WAVESPEED  # 5.65517s
T_GEOM = TS + TAU_GEOM                # 6.65517s

# -------------------------------------------------------------
# Base Helper: Clock & Cepstrum Extractors
# -------------------------------------------------------------
def extract_onset_peak(t: np.ndarray, dH: np.ndarray):
    dt = float(np.median(np.diff(t)))
    m_win = (t >= (T_GEOM - 0.5)) & (t <= (T_GEOM + 0.5))
    t_win = t[m_win]
    dH_win = dH[m_win]
    max_dH = float(np.max(dH_win)) if len(dH_win) > 0 else 0.0

    onset_mask = dH_win >= (0.01 * max_dH)
    t_onset = float(t_win[np.argmax(onset_mask)]) if np.any(onset_mask) else np.nan

    peaks, _ = find_peaks(dH_win, height=0.10 * max_dH, distance=int(0.002 / dt))
    t_peak = float(t_win[peaks[0]]) if len(peaks) > 0 else float(t_win[np.argmax(dH_win)])

    x_onset = WAVESPEED * (t_onset - TS) / 2.0
    x_peak = WAVESPEED * (t_peak - TS) / 2.0
    dx_onset = x_onset - FRAC_X1
    dx_peak = x_peak - FRAC_X1
    return t_onset, x_onset, dx_onset, t_peak, x_peak, dx_peak

def compute_custom_cepstrum(t: np.ndarray, sig: np.ndarray, win_type: str = 'hann', t_win_sec: float = 49.0):
    dt = float(np.median(np.diff(t)))
    m_post = (t >= TS) & (t <= (TS + t_win_sec))
    s_post = sig[m_post]
    x = s_post - np.mean(s_post)
    N = len(x)
    
    if win_type == 'hann':
        w = windows.hann(N)
    elif win_type == 'hamming':
        w = windows.hamming(N)
    elif win_type == 'boxcar' or win_type == 'rect':
        w = windows.boxcar(N)
    else:
        w = np.ones(N)

    xw = x * w
    spec = fft(xw)
    log_mag = np.log(np.abs(spec) + np.finfo(np.float64).eps)
    ceps = np.real(ifft(log_mag))
    fs = 1.0 / dt
    quefrency = np.arange(N) / fs

    m_q = (quefrency >= (TAU_GEOM - 0.08)) & (quefrency <= (TAU_GEOM + 0.08))
    q_sub = quefrency[m_q]
    ceps_sub = ceps[m_q]
    idx_max = np.argmax(np.abs(ceps_sub))
    tau_cep = float(q_sub[idx_max])
    x_cep = WAVESPEED * tau_cep / 2.0
    dx_cep = x_cep - FRAC_X1
    return tau_cep, x_cep, dx_cep

# -------------------------------------------------------------
# 1. Grid Refinement Analysis (dt = 0.5 ms vs 1.0 ms)
# -------------------------------------------------------------
def analyze_grid_sensitivity():
    # 1.0 ms baseline
    df_1ms_k0 = pd.read_csv(os.path.join(sim_p5, "Steady_Darcy.csv"))
    df_1ms_k01 = pd.read_csv(os.path.join(sim_p5, "Brunone_IAB_k0.01.csv"))
    
    # 0.5 ms baseline
    df_05ms_k0 = pd.read_csv(os.path.join(sim_p6, "dt05_k0.0.csv"))
    df_05ms_k01 = pd.read_csv(os.path.join(sim_p6, "dt05_k0.01.csv"))

    runs = [
        ("dt_1.0ms_k0.0", df_1ms_k0, 1.0, 0.0),
        ("dt_1.0ms_k0.01", df_1ms_k01, 1.0, 0.01),
        ("dt_0.5ms_k0.0", df_05ms_k0, 0.5, 0.0),
        ("dt_0.5ms_k0.01", df_05ms_k01, 0.5, 0.01),
    ]

    rows = []
    for label, df, dt_ms, k in runs:
        t = df['t'].to_numpy()
        H = df['H_wh'].to_numpy()
        dt_s = dt_ms * 1e-3
        dH = np.gradient(H, dt_s)
        
        t_on, x_on, dx_on, t_pk, x_pk, dx_pk = extract_onset_peak(t, dH)
        tau_cp, x_cp, dx_cp = compute_custom_cepstrum(t, dH, win_type='hann', t_win_sec=49.0)
        
        rows.append({
            'case_label': label,
            'dt_ms': dt_ms,
            'k': k,
            't_onset_s': t_on,
            'dx_onset_m': dx_on,
            't_peak_s': t_pk,
            'dx_peak_m': dx_pk,
            'tau_cep_s': tau_cp,
            'dx_cep_m': dx_cp,
        })
    
    df_grid = pd.DataFrame(rows)
    
    # Compute relative shifts vs steady at respective grid
    dx_1ms_steady_pk = df_grid.loc[df_grid['case_label'] == 'dt_1.0ms_k0.0', 'dx_peak_m'].values[0]
    dx_1ms_steady_cp = df_grid.loc[df_grid['case_label'] == 'dt_1.0ms_k0.0', 'dx_cep_m'].values[0]
    dx_05ms_steady_pk = df_grid.loc[df_grid['case_label'] == 'dt_0.5ms_k0.0', 'dx_peak_m'].values[0]
    dx_05ms_steady_cp = df_grid.loc[df_grid['case_label'] == 'dt_0.5ms_k0.0', 'dx_cep_m'].values[0]

    df_grid['delta_x_peak_m'] = [
        0.0,
        df_grid.loc[1, 'dx_peak_m'] - dx_1ms_steady_pk,
        0.0,
        df_grid.loc[3, 'dx_peak_m'] - dx_05ms_steady_pk,
    ]
    df_grid['delta_x_cep_m'] = [
        0.0,
        df_grid.loc[1, 'dx_cep_m'] - dx_1ms_steady_cp,
        0.0,
        df_grid.loc[3, 'dx_cep_m'] - dx_05ms_steady_cp,
    ]

    out_csv = os.path.join(dir_02, "data", "p6_grid_sensitivity.csv")
    df_grid.to_csv(out_csv, index=False)
    print("\n=== Gantt p6: Grid Sensitivity Analysis ===")
    print(df_grid[['dt_ms', 'k', 'dx_onset_m', 'dx_peak_m', 'dx_cep_m', 'delta_x_peak_m', 'delta_x_cep_m']].to_string())
    return df_grid

# -------------------------------------------------------------
# 2. Window Type & Duration Sensitivity Analysis
# -------------------------------------------------------------
def analyze_window_sensitivity():
    df_k0 = pd.read_csv(os.path.join(sim_p5, "Steady_Darcy.csv"))
    df_k01 = pd.read_csv(os.path.join(sim_p5, "Brunone_IAB_k0.01.csv"))

    t0 = df_k0['t'].to_numpy()
    H0 = df_k0['H_wh'].to_numpy()
    dH0 = np.gradient(H0, 0.001)

    t1 = df_k01['t'].to_numpy()
    H1 = df_k01['H_wh'].to_numpy()
    dH1 = np.gradient(H1, 0.001)

    win_types = ['hann', 'hamming', 'boxcar']
    win_lens = [10.0, 30.0, 50.0]

    rows = []
    for w_type in win_types:
        for w_len in win_lens:
            tau_0, x_0, dx_0 = compute_custom_cepstrum(t0, dH0, win_type=w_type, t_win_sec=w_len-1.0 if w_len==50.0 else w_len)
            tau_1, x_1, dx_1 = compute_custom_cepstrum(t1, dH1, win_type=w_type, t_win_sec=w_len-1.0 if w_len==50.0 else w_len)
            delta_x_cep = x_1 - x_0

            rows.append({
                'window_type': w_type,
                'window_len_s': w_len,
                'tau_cep_steady_s': tau_0,
                'dx_cep_steady_m': dx_0,
                'tau_cep_k0.01_s': tau_1,
                'dx_cep_k0.01_m': dx_1,
                'delta_x_cep_m': delta_x_cep,
            })

    df_win = pd.DataFrame(rows)
    out_csv = os.path.join(dir_04, "data", "p6_window_sensitivity.csv")
    df_win.to_csv(out_csv, index=False)
    print("\n=== Gantt p6: Window Sensitivity Analysis ===")
    print(df_win[['window_type', 'window_len_s', 'dx_cep_steady_m', 'dx_cep_k0.01_m', 'delta_x_cep_m']].to_string())
    return df_win

# -------------------------------------------------------------
# 3. Input Channel Sensitivity Analysis
# -------------------------------------------------------------
def analyze_input_channel_sensitivity():
    df_k0 = pd.read_csv(os.path.join(sim_p5, "Steady_Darcy.csv"))
    df_k01 = pd.read_csv(os.path.join(sim_p5, "Brunone_IAB_k0.01.csv"))

    t = df_k0['t'].to_numpy()
    dt = 0.001
    fs = 1.0 / dt

    H0 = df_k0['H_wh'].to_numpy()
    dH0 = np.gradient(H0, dt)

    H1 = df_k01['H_wh'].to_numpy()
    dH1 = np.gradient(H1, dt)

    # 4th-order Butterworth bandpass filter [0.05, 150] Hz
    sos = butter(4, [0.05, 150.0], btype='bandpass', fs=fs, output='sos')
    H0_bp = sosfiltfilt(sos, H0 - np.mean(H0))
    H1_bp = sosfiltfilt(sos, H1 - np.mean(H1))

    channels = [
        ('Pressure_Head_H', H0, H1),
        ('Head_Derivative_dHdt', dH0, dH1),
        ('Bandpass_Filtered_H', H0_bp, H1_bp),
    ]

    rows = []
    for ch_name, s0, s1 in channels:
        tau_0, x_0, dx_0 = compute_custom_cepstrum(t, s0, win_type='hann', t_win_sec=49.0)
        tau_1, x_1, dx_1 = compute_custom_cepstrum(t, s1, win_type='hann', t_win_sec=49.0)
        delta_x_cep = x_1 - x_0

        rows.append({
            'channel_name': ch_name,
            'tau_cep_steady_s': tau_0,
            'dx_cep_steady_m': dx_0,
            'tau_cep_k0.01_s': tau_1,
            'dx_cep_k0.01_m': dx_1,
            'delta_x_cep_m': delta_x_cep,
        })

    df_ch = pd.DataFrame(rows)
    out_csv = os.path.join(dir_04, "data", "p6_input_channel_sensitivity.csv")
    df_ch.to_csv(out_csv, index=False)
    print("\n=== Gantt p6: Input Channel Sensitivity Analysis ===")
    print(df_ch[['channel_name', 'dx_cep_steady_m', 'dx_cep_k0.01_m', 'delta_x_cep_m']].to_string())
    return df_ch

# -------------------------------------------------------------
# 4. Additive Gaussian White Noise Robustness Analysis
# -------------------------------------------------------------
def analyze_noise_robustness():
    df_k0 = pd.read_csv(os.path.join(sim_p5, "Steady_Darcy.csv"))
    df_k01 = pd.read_csv(os.path.join(sim_p5, "Brunone_IAB_k0.01.csv"))

    t = df_k0['t'].to_numpy()
    dt = 0.001
    H0_clean = df_k0['H_wh'].to_numpy()
    H1_clean = df_k01['H_wh'].to_numpy()

    snr_levels = [60.0, 40.0, 20.0]
    seeds = [42, 43, 44, 45, 46]

    # Signal power based on oscillatory variance
    p_sig0 = np.var(H0_clean)
    p_sig1 = np.var(H1_clean)

    rows = []
    for snr in snr_levels:
        delta_list = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            
            # Noise for Steady
            sigma0 = np.sqrt(p_sig0 / (10.0 ** (snr / 10.0)))
            noise0 = rng.normal(0.0, sigma0, len(H0_clean))
            H0_noisy = H0_clean + noise0
            dH0_noisy = np.gradient(H0_noisy, dt)

            # Noise for Brunone k=0.01
            sigma1 = np.sqrt(p_sig1 / (10.0 ** (snr / 10.0)))
            noise1 = rng.normal(0.0, sigma1, len(H1_clean))
            H1_noisy = H1_clean + noise1
            dH1_noisy = np.gradient(H1_noisy, dt)

            tau_0, x_0, dx_0 = compute_custom_cepstrum(t, dH0_noisy, win_type='hann', t_win_sec=49.0)
            tau_1, x_1, dx_1 = compute_custom_cepstrum(t, dH1_noisy, win_type='hann', t_win_sec=49.0)
            delta_x_cep = x_1 - x_0
            delta_list.append(delta_x_cep)

            rows.append({
                'snr_db': snr,
                'seed': seed,
                'tau_cep_steady_s': tau_0,
                'dx_cep_steady_m': dx_0,
                'tau_cep_k0.01_s': tau_1,
                'dx_cep_k0.01_m': dx_1,
                'delta_x_cep_m': delta_x_cep,
            })

    df_noise = pd.DataFrame(rows)
    out_csv = os.path.join(dir_02, "data", "p6_noise_robustness.csv")
    df_noise.to_csv(out_csv, index=False)
    
    print("\n=== Gantt p6: Additive Noise Robustness Summary ===")
    for snr in snr_levels:
        sub = df_noise[df_noise['snr_db'] == snr]
        med = sub['delta_x_cep_m'].median()
        min_v = sub['delta_x_cep_m'].min()
        max_v = sub['delta_x_cep_m'].max()
        rng_v = max_v - min_v
        print(f"  SNR={snr:2.0f} dB: Median delta_x_cep = {med:+.3f} m, Range = [{min_v:+.3f}, {max_v:+.3f}] m (Span: {rng_v:.3f} m)")
    return df_noise

# -------------------------------------------------------------
# 5. Diagnostic Figure: SPE Journal Compliant
# -------------------------------------------------------------
def plot_p6_diagnostic_figure(df_grid, df_win, df_ch, df_noise):
    PALETTE = {
        "darcy":    "#1B4F72",   # Dark Blue
        "k001":     "#117864",   # Pine Green
        "t_onset":  "#27AE60",   # Emerald Green
        "t_peak":   "#C0392B",   # Crimson Red
        "tau_cep":  "#8E44AD",   # Purple
        "gray_ref": "#7F8C8D",   # Reference Gray
        "accent":   "#D35400",   # Coral
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), dpi=300)

    # ---------------------------------------------------------
    # (a) Grid Convergence (dt = 0.5 ms vs 1.0 ms)
    # ---------------------------------------------------------
    ax = axes[0, 0]
    ax.set_box_aspect(4/5)

    df_1ms_k0 = pd.read_csv(os.path.join(sim_p5, "Steady_Darcy.csv"))
    df_1ms_k01 = pd.read_csv(os.path.join(sim_p5, "Brunone_IAB_k0.01.csv"))
    df_05ms_k0 = pd.read_csv(os.path.join(sim_p6, "dt05_k0.0.csv"))
    df_05ms_k01 = pd.read_csv(os.path.join(sim_p6, "dt05_k0.01.csv"))

    t_geom = TS + 2.0 * FRAC_X1 / WAVESPEED
    m1 = (df_1ms_k01['t'] >= (t_geom - 0.02)) & (df_1ms_k01['t'] <= (t_geom + 0.08))
    m05 = (df_05ms_k01['t'] >= (t_geom - 0.02)) & (df_05ms_k01['t'] <= (t_geom + 0.08))

    dH_1ms = np.gradient(df_1ms_k01['H_wh'].to_numpy(), 0.001)
    dH_05ms = np.gradient(df_05ms_k01['H_wh'].to_numpy(), 0.0005)

    ax.plot((df_1ms_k01['t'][m1] - t_geom)*1000, dH_1ms[m1]/1000,
            label='IAB $k=0.01$ ($\\Delta t=1.0$ ms)', color=PALETTE['k001'], linestyle='-', linewidth=1.2)
    ax.plot((df_05ms_k01['t'][m05] - t_geom)*1000, dH_05ms[m05]/1000,
            label='IAB $k=0.01$ ($\\Delta t=0.5$ ms)', color=PALETTE['accent'], linestyle='--', linewidth=1.1)

    ax.axvline(0, color=PALETTE['gray_ref'], linestyle=':', linewidth=0.8, label='$t_{\\mathrm{geom}}$')
    ax.set_xlabel('Time relative to Arrival $\\Delta t$ (ms)', fontsize=8.0)
    ax.set_ylabel('Head Rate $\\mathrm{d}H/\\mathrm{d}t$ (km/s)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(a)', loc='left', fontsize=9.5, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper right')

    # ---------------------------------------------------------
    # (b) Window Type & Duration Sensitivity
    # ---------------------------------------------------------
    ax = axes[0, 1]
    ax.set_box_aspect(4/5)

    win_labels = ['10 s', '30 s', '50 s']
    x_pos = np.arange(len(win_labels))
    w = 0.25

    hann_vals = df_win[df_win['window_type'] == 'hann']['delta_x_cep_m'].values
    hamm_vals = df_win[df_win['window_type'] == 'hamming']['delta_x_cep_m'].values
    box_vals = df_win[df_win['window_type'] == 'boxcar']['delta_x_cep_m'].values

    ax.bar(x_pos - w, hann_vals, width=w, label='Hann', color=PALETTE['tau_cep'], edgecolor='black', linewidth=0.5)
    ax.bar(x_pos, hamm_vals, width=w, label='Hamming', color='#2874A6', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + w, box_vals, width=w, label='Rectangular', color='#D35400', edgecolor='black', linewidth=0.5)

    ax.axhline(9.425, color=PALETTE['gray_ref'], linestyle='--', linewidth=0.8, label='Reference $+9.43$ m')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(win_labels, fontsize=7.0)
    ax.set_xlabel('Post Shut-in Window Duration $T_{\\mathrm{win}}$ (s)', fontsize=8.0)
    ax.set_ylabel('Relative Shift $\\delta x_{\\mathrm{cep}}$ (m)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(b)', loc='left', fontsize=9.5, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='lower right')

    # ---------------------------------------------------------
    # (c) Input Channel Sensitivity
    # ---------------------------------------------------------
    ax = axes[1, 0]
    ax.set_box_aspect(4/5)

    ch_labels = ['Head $H$', 'Derivative\n$\\mathrm{d}H/\\mathrm{d}t$', 'Bandpassed\n$H$']
    ch_vals = df_ch['delta_x_cep_m'].values
    x_ch = np.arange(len(ch_labels))

    bars = ax.bar(x_ch, ch_vals, width=0.45, color=[PALETTE['darcy'], PALETTE['tau_cep'], PALETTE['k001']],
                  edgecolor='black', linewidth=0.5)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4, f"{ch_vals[i]:+.2f} m",
                ha='center', va='bottom', fontsize=6.8, fontweight='bold')

    ax.axhline(9.425, color=PALETTE['gray_ref'], linestyle='--', linewidth=0.8)
    ax.set_xticks(x_ch)
    ax.set_xticklabels(ch_labels, fontsize=7.0)
    ax.set_ylabel('Relative Shift $\\delta x_{\\mathrm{cep}}$ (m)', fontsize=8.0)
    ax.set_ylim(0, 13.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(c)', loc='left', fontsize=9.5, fontweight='bold', pad=4)

    # ---------------------------------------------------------
    # (d) Additive Gaussian Noise Robustness
    # ---------------------------------------------------------
    ax = axes[1, 1]
    ax.set_box_aspect(4/5)

    snr_levels = [60.0, 40.0, 20.0]
    snr_x = [60, 40, 20]
    medians = []
    mins = []
    maxs = []

    for snr in snr_levels:
        sub = df_noise[df_noise['snr_db'] == snr]
        medians.append(sub['delta_x_cep_m'].median())
        mins.append(sub['delta_x_cep_m'].min())
        maxs.append(sub['delta_x_cep_m'].max())

    err_low = np.array(medians) - np.array(mins)
    err_high = np.array(maxs) - np.array(medians)

    ax.errorbar(snr_x, medians, yerr=[err_low, err_high], fmt='o', color=PALETTE['tau_cep'],
                ecolor='black', elinewidth=1.0, capsize=3.0, capthick=0.75,
                markersize=4.5, label='Median $\\pm$ Range (5 seeds)')
    
    # Scatter individual seeds
    for _, row in df_noise.iterrows():
        ax.scatter(row['snr_db'], row['delta_x_cep_m'], color=PALETTE['accent'], s=15, alpha=0.6, zorder=4)

    ax.axhline(9.425, color=PALETTE['gray_ref'], linestyle='--', linewidth=0.8, label='Clean $+9.43$ m')
    ax.set_xlabel('Signal-to-Noise Ratio $\\mathrm{SNR}$ (dB)', fontsize=8.0)
    ax.set_ylabel('Relative Shift $\\delta x_{\\mathrm{cep}}$ (m)', fontsize=8.0)
    ax.set_xlim(15, 65)
    ax.set_ylim(2.0, 13.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(d)', loc='left', fontsize=9.5, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper left')

    plt.tight_layout()
    fig_path = os.path.join(dir_02, "figures", "fig_p6_robustness_diagnostic.png")
    fig_svg = os.path.join(dir_02, "figures", "fig_p6_robustness_diagnostic.svg")
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_svg)
    plt.close()
    print(f"Saved SPEJ-compliant p6 diagnostic figure to:\n  {fig_path}\n  {fig_svg}")

if __name__ == '__main__':
    df_grid = analyze_grid_sensitivity()
    df_win = analyze_window_sensitivity()
    df_ch = analyze_input_channel_sensitivity()
    df_noise = analyze_noise_robustness()
    plot_p6_diagnostic_figure(df_grid, df_win, df_ch, df_noise)
