# -*- coding: utf-8 -*-
"""
Compute the four clocks (t_onset, t_peak, t_E50, tau_cep) on n=1 and n=4 (D=20m)
under frozen definitions, with unified [1.0, 50.0]s Hann window and |C(tau)| peak picking.
Also generates cepstrum profile plot (n=1/n=4 x k=0/0.01) to verify absence of sidelobe mis-picking.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, windows
from scipy.fft import fft, ifft

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_d = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_d))
if _root not in sys.path:
    sys.path.insert(0, _root)

INPUT_DIR = os.path.join(_root, 'output', 'analysis', 'brunone_spacing_effect')
OUT_DIR = os.path.join(_root, 'PaperB_考虑brunone的倒谱识别', '02_时域波形退化_峰漂移与展宽', '指标表')
IMG_DIR = os.path.join(_root, 'PaperB_考虑brunone的倒谱识别', '04_倒谱模糊与系统性深度偏差', '图')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
T_GEOM_2WAY = 2.0 * FRAC_X1 / WAVESPEED  # 5.6551724...

WIN_FIRST_PACKET = (6.5, 7.5)
PEAK_HEIGHT_FRAC = 0.10
PEAK_MIN_SEP_S = 0.002
ONSET_FRAC = 0.01

SEARCH_CEP_WIN = (5.575, 5.735)  # 5.655s ± 80ms

def load_case_data(n: int, D: int, k: float):
    if n == 1:
        csv_path = os.path.join(INPUT_DIR, f'n1_k{k}', 'moc_timeseries.csv')
    else:
        k_str = '0' if float(k) == 0.0 else str(k)
        csv_path = os.path.join(INPUT_DIR, f'D{D}_k{k_str}', 'moc_timeseries.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Case not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    t = df['t'].to_numpy(dtype=np.float64)
    H = df['H_wh'].to_numpy(dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    dH = np.gradient(H, dt)
    return t, H, dH, dt

def extract_clocks_single_case(t: np.ndarray, H: np.ndarray, dH: np.ndarray, dt: float):
    # 1. First wavepacket window [6.5, 7.5] s
    m_win = (t >= WIN_FIRST_PACKET[0]) & (t <= WIN_FIRST_PACKET[1])
    t_win = t[m_win]
    dH_win = dH[m_win]
    max_dH = float(np.max(dH_win))
    
    # 2. t_onset: first time exceeding 0.01 * max(dH_win)
    onset_mask = dH_win >= (ONSET_FRAC * max_dH)
    if np.any(onset_mask):
        t_onset = float(t_win[np.argmax(onset_mask)])
    else:
        t_onset = np.nan
        
    # 3. t_peak: first peak in window
    dist_pts = max(1, int(round(PEAK_MIN_SEP_S / dt)))
    height_thresh = PEAK_HEIGHT_FRAC * max_dH
    peaks, _ = find_peaks(dH_win, height=height_thresh, distance=dist_pts)
    if len(peaks) > 0:
        t_peak = float(t_win[peaks[0]])
    else:
        t_peak = float(t_win[np.argmax(dH_win)])
        
    # 4. t_E50: positive cumulative energy reaches 50%
    y_pos = np.maximum(0.0, dH_win)
    e = y_pos ** 2
    E = np.cumsum(e)
    if E[-1] > 0:
        E_norm = E / E[-1]
        t_E50 = float(np.interp(0.50, E_norm, t_win))
    else:
        t_E50 = np.nan
        
    # 5. tau_cep: real cepstrum on unified t in [1.0, 50.0]s, Hann windowed dH/dt
    m_post = (t >= 1.0) & (t <= 50.0)
    dH_post = dH[m_post]
    
    # Remove mean and apply Hann window
    x = dH_post - np.mean(dH_post)
    w = windows.hann(len(x))
    xw = x * w
    
    # Real cepstrum: Re{ IFFT( log|FFT(xw)| ) }
    N = len(xw)
    spec = fft(xw)
    log_mag = np.log(np.abs(spec) + np.finfo(np.float64).eps)
    ceps = np.real(ifft(log_mag))
    
    fs = 1.0 / dt
    quefrency = np.arange(N) / fs
    
    # Search for peak in [5.575, 5.735] s (5.655s ± 80ms) using argmax |C(tau)|
    m_q = (quefrency >= SEARCH_CEP_WIN[0]) & (quefrency <= SEARCH_CEP_WIN[1])
    q_sub = quefrency[m_q]
    ceps_sub = ceps[m_q]
    
    idx_max = np.argmax(np.abs(ceps_sub))
    tau_cep = float(q_sub[idx_max])
    
    return t_onset, t_peak, t_E50, tau_cep, (quefrency, ceps)

def run_analysis():
    cases = []
    cep_profiles = {}
    
    # 1. n = 1 cases
    for k in [0, 0.01, 0.02, 0.05]:
        t, H, dH, dt = load_case_data(n=1, D=0, k=k)
        t_on, t_pk, t_e50, tau_cp, prof = extract_clocks_single_case(t, H, dH, dt)
        cep_profiles[(1, k)] = prof
        cases.append({
            'n': 1,
            'D_m': 0,
            'k': k,
            't_onset': t_on,
            't_peak': t_pk,
            't_E50': t_e50,
            'tau_cep': tau_cp,
        })
        
    # 2. n = 4, D = 20m cases
    for k in [0, 0.01, 0.02, 0.05]:
        t, H, dH, dt = load_case_data(n=4, D=20, k=k)
        t_on, t_pk, t_e50, tau_cp, prof = extract_clocks_single_case(t, H, dH, dt)
        cep_profiles[(4, k)] = prof
        cases.append({
            'n': 4,
            'D_m': 20,
            'k': k,
            't_onset': t_on,
            't_peak': t_pk,
            't_E50': t_e50,
            'tau_cep': tau_cp,
        })
        
    df = pd.DataFrame(cases)
    
    # Compute relative deltas against k=0
    base_n1 = df[(df['n'] == 1) & (df['k'] == 0)].iloc[0]
    base_n4 = df[(df['n'] == 4) & (df['k'] == 0)].iloc[0]
    
    dt_onset_list, dt_peak_list, dt_E50_list, dt_cep_list = [], [], [], []
    d_onset_list, d_peak_list, d_E50_list, d_cep_list = [], [], [], []
    
    for _, row in df.iterrows():
        base = base_n1 if row['n'] == 1 else base_n4
        dt_on = (row['t_onset'] - base['t_onset']) * 1000.0  # ms
        dt_pk = (row['t_peak'] - base['t_peak']) * 1000.0    # ms
        dt_e50 = (row['t_E50'] - base['t_E50']) * 1000.0    # ms
        dt_cp = (row['tau_cep'] - base['tau_cep']) * 1000.0  # ms
        
        d_on = WAVESPEED * (dt_on / 1000.0) / 2.0  # m
        d_pk = WAVESPEED * (dt_pk / 1000.0) / 2.0  # m
        d_e50 = WAVESPEED * (dt_e50 / 1000.0) / 2.0 # m
        d_cp = WAVESPEED * (dt_cp / 1000.0) / 2.0   # m
        
        dt_onset_list.append(dt_on)
        dt_peak_list.append(dt_pk)
        dt_E50_list.append(dt_e50)
        dt_cep_list.append(dt_cp)
        
        d_onset_list.append(d_on)
        d_peak_list.append(d_pk)
        d_E50_list.append(d_e50)
        d_cep_list.append(d_cp)
        
    df['dt_onset_ms'] = dt_onset_list
    df['dt_peak_ms'] = dt_peak_list
    df['dt_E50_ms'] = dt_E50_list
    df['dt_cep_ms'] = dt_cep_list
    
    df['d_onset_m'] = d_onset_list
    df['d_peak_m'] = d_peak_list
    df['d_E50_m'] = d_E50_list
    df['d_cep_m'] = d_cep_list
    
    out_csv = os.path.join(OUT_DIR, 'four_clocks_n1_vs_n4.csv')
    df.to_csv(out_csv, index=False)
    print("=== Updated Four Clocks Table ===")
    print(df.to_string())
    print(f"\nSaved table to {out_csv}")
    
    # Generate verification plot: Cepstrum Profiles around 5.655s
    plot_cepstrum_verification(cep_profiles, df)
    
    return df

def plot_cepstrum_verification(cep_profiles, df):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 0.75
    
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), dpi=300, sharex=True)
    
    plot_configs = [
        ((1, 0), axes[0, 0], "$n=1$, $k=0$ (Steady Darcy)", "#1B4F72"),
        ((1, 0.01), axes[0, 1], "$n=1$, $k=0.01$ (Weak Brunone)", "#117864"),
        ((4, 0), axes[1, 0], "$n=4$, $D=20\\,$m, $k=0$ (Steady Darcy)", "#1B4F72"),
        ((4, 0.01), axes[1, 1], "$n=4$, $D=20\\,$m, $k=0.01$ (Weak Brunone)", "#117864"),
    ]
    
    for (n_val, k_val), ax, title, col in plot_configs:
        q, c = cep_profiles[(n_val, k_val)]
        m = (q >= 5.50) & (q <= 5.80)
        
        # Plot real cepstrum and magnitude
        ax.plot(q[m], c[m], color=col, lw=1.2, label="Real Cepstrum $C(\\tau)$")
        ax.plot(q[m], np.abs(c[m]), color="gray", ls="--", lw=0.8, alpha=0.7, label="$|C(\\tau)|$ envelope")
        
        # Mark geometric arrival
        ax.axvline(T_GEOM_2WAY, color="#C0392B", ls=":", lw=1.0, label="Geometric 2-way ($5.655\\,$s)")
        
        # Mark picked peak
        row = df[(df['n'] == n_val) & (df['k'] == k_val)].iloc[0]
        picked_tau = row['tau_cep']
        # Find c value at picked_tau
        idx_p = np.argmin(np.abs(q - picked_tau))
        c_val = c[idx_p]
        
        ax.plot(picked_tau, c_val, marker="o", color="#8E44AD", markersize=5, zorder=5,
                label=f"Picked $\\tau={picked_tau:.3f}\\,$s")
        
        # Annotations
        dt_ms = row['dt_cep_ms']
        d_m = row['d_cep_m']
        ax.annotate(f"$\\tau={picked_tau:.3f}\\,$s\n$\\Delta\\tau={dt_ms:+.1f}\\,$ms\n$\\Delta d={d_m:+.2f}\\,$m",
                    xy=(picked_tau, c_val), xytext=(picked_tau + 0.015, c_val),
                    fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=0.8))
        
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax.tick_params(direction='in', top=True, right=True, labelsize=7)
        ax.set_ylabel("Amplitude", fontsize=8)
        
        if ax == axes[0, 0]:
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
            
    axes[1, 0].set_xlabel("Quefrency $\\tau$ (s)", fontsize=8)
    axes[1, 1].set_xlabel("Quefrency $\\tau$ (s)", fontsize=8)
    
    plt.tight_layout()
    plot_path_png = os.path.join(IMG_DIR, 'cepstrum_profiles_proof.png')
    plot_path_svg = os.path.join(IMG_DIR, 'cepstrum_profiles_proof.svg')
    plt.savefig(plot_path_png, dpi=300)
    plt.savefig(plot_path_svg)
    plt.close()
    print(f"Saved cepstrum verification plot to {plot_path_png} and {plot_path_svg}")

if __name__ == '__main__':
    run_analysis()
