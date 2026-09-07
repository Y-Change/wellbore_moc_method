# -*- coding: utf-8 -*-
"""
Gantt p4: (k x k_leak) Orthogonal Surface Simulation & CWT Damping Confusion Analysis.
Frozen parameter settings:
- Topology: n=1, X1=4100m, Cf=1.0e-5 m^2, H_ext=3000.0m
- Geometry: L=5000m, D=0.1397m, a=1450m/s, V0=2.0m/s, H0=3000.0m
- Friction: constant k in {0, 0.01, 0.02, 0.05} (k=0 uses steady)
- Leakoff: k_leak in {1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4} m^2/s/sqrt(m)
- Wellhead BC: velocity_step, ts=1.0s, tf=50.0s, dt=1.0ms
- Primary metric: CWT envelope damping ratio zeta (Gabry 2025 style)
- Secondary metric: Frozen |C| cepstral peak shift delta_x_cep
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2, windows
from scipy.fft import fft, ifft

_root = r"e:\water_hammer_research\wellbore_moc_method"
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore
from moc_simulate.config import WELL_CONFIG, FRACTURE_CONFIG
import moc_simulate.wellbore_moc as wm

out_dir_05 = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "05_雷诺数动态k与物性敏感性")
code_dir = os.path.join(out_dir_05, "code")
data_dir = os.path.join(out_dir_05, "data")
figs_dir = os.path.join(out_dir_05, "figures")
sim_base = os.path.join(_root, "output", "analysis", "brunone_spacing_effect", "k_kleak_surface")

os.makedirs(code_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(sim_base, exist_ok=True)

# 16-point grid
K_GRID = [0.0, 0.01, 0.02, 0.05]
KLEAK_GRID = [1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4]

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
CF = 1.0e-5
TS = 1.0
TF = 50.0
DT = 0.001

def patch_brunone(k_val):
    def mock_k(Re: float) -> float:
        return k_val
    def mock_k_vec(Re_arr: np.ndarray) -> np.ndarray:
        return np.full_like(Re_arr, k_val, dtype=np.float64)
    wm.brunone_k = mock_k
    wm.brunone_k_vec = mock_k_vec

def run_16_simulations():
    original_brunone_k = wm.brunone_k
    original_brunone_k_vec = wm.brunone_k_vec

    for k in K_GRID:
        for kleak in KLEAK_GRID:
            case_name = f"n1_k{k}_kleak{kleak:.0e}"
            case_dir = os.path.join(sim_base, case_name)
            os.makedirs(case_dir, exist_ok=True)
            csv_path = os.path.join(case_dir, "moc_timeseries.csv")
            
            if os.path.exists(csv_path):
                print(f"Skipping {case_name}, already exists.")
                continue
                
            print(f"Simulating {case_name} (k={k}, kleak={kleak:.1e})...")
            fric_model = 'steady' if k == 0.0 else 'brunone'
            if k > 0:
                patch_brunone(k)
                
            cfg = MocConfig(
                wellbore_length=WELL_CONFIG['L'],
                wellbore_diameter=WELL_CONFIG['wellbore_diameter'],
                fluid_density=WELL_CONFIG['fluid_density'],
                fluid_viscosity=WELL_CONFIG['fluid_viscosity'],
                wavespeed=WAVESPEED,
                roughness_height=WELL_CONFIG['roughness_height'],
                friction_model=fric_model,
                dt=DT,
                tf=TF,
                wellhead_bc="velocity_step",
                pump_shut_time=TS,
                initial_velocity=WELL_CONFIG['V0'],
                initial_head=WELL_CONFIG['H0'],
                theta=WELL_CONFIG['theta'],
                toe_bc="reservoir",
                toe_head=WELL_CONFIG['H0'],
            )
            
            t0 = time.time()
            res = simulate_wellbore(
                cfg,
                fracture_positions=[FRAC_X1],
                fracture_Cf=[CF],
                fracture_kleak=[kleak],
                H_ext=FRACTURE_CONFIG['H_ext'],
                store_full_field=False,
            )
            print(f"  Simulation done in {time.time()-t0:.2f}s")
            
            df = pd.DataFrame({
                't': res["timestamps"],
                'H_wh': res["wellhead_head"],
                'Q_wh': res["wellhead_velocity"] * cfg.area,
            })
            df.to_csv(csv_path, index=False)
            
            wm.brunone_k = original_brunone_k
            wm.brunone_k_vec = original_brunone_k_vec

def compute_cwt_damping_ratio(t: np.ndarray, H: np.ndarray, t_fit_start=2.0, t_fit_end=45.0):
    """
    Compute CWT envelope damping ratio zeta and decay rate alpha (Gabry 2025 style).
    - Signal: H_wh(t) - mean(H_wh) on t in [1.0, 50.0]
    - Dominant frequency f0 from FFT peak
    - CWT ridge envelope A(t) = |CWT(f0, t)| using Morlet wavelet (w=6.0)
    - Linear regression on ln A(t) = -alpha * t + beta over [t_fit_start, t_fit_end]
    - Damping ratio zeta = alpha / (2 * pi * f0)
    """
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    
    # 1. Post shut-in signal
    m_post = (t >= 1.0) & (t <= 50.0)
    t_post = t[m_post]
    H_post = H[m_post]
    y = H_post - np.mean(H_post)
    
    # 2. FFT peak frequency in [0.03, 0.5] Hz
    N = len(y)
    Y = np.abs(fft(y))
    freqs = np.fft.fftfreq(N, dt)
    m_f = (freqs >= 0.03) & (freqs <= 0.50)
    f_sub = freqs[m_f]
    Y_sub = Y[m_f]
    f0 = float(f_sub[np.argmax(Y_sub)])
    
    # 3. CWT using Morlet wavelet at f0
    # For morlet2: scale s = w / (2 * pi * f0) * fs
    w = 6.0
    scale = w * fs / (2.0 * np.pi * f0)
    cwt_matrix = cwt(y, morlet2, [scale], w=w)
    envelope = np.abs(cwt_matrix[0])
    
    # 4. Exponential fit on [t_fit_start, t_fit_end]
    m_fit = (t_post >= t_fit_start) & (t_post <= t_fit_end)
    t_fit = t_post[m_fit]
    env_fit = envelope[m_fit]
    
    # Avoid log of zero
    env_fit = np.maximum(env_fit, 1.0e-12)
    ln_env = np.log(env_fit)
    
    # Linear fit: ln_env = -alpha * t + beta
    poly = np.polyfit(t_fit, ln_env, 1)
    alpha = -float(poly[0])  # decay rate in s^-1
    beta = float(poly[1])
    
    # Goodness of fit R^2
    ln_env_pred = poly[0] * t_fit + poly[1]
    res_ss = np.sum((ln_env - ln_env_pred) ** 2)
    tot_ss = np.sum((ln_env - np.mean(ln_env)) ** 2)
    r2 = float(1.0 - res_ss / tot_ss) if tot_ss > 0 else 0.0
    
    # Damping ratio zeta
    omega0 = 2.0 * np.pi * f0
    zeta = float(alpha / omega0)
    
    return {
        'f0_Hz': f0,
        'alpha_decay_s_inv': alpha,
        'zeta_damping_ratio': zeta,
        'fit_r2': r2,
    }

def compute_frozen_cepstrum(t: np.ndarray, H: np.ndarray):
    """
    Compute frozen |C(tau)| cepstrum clock and estimated depth.
    """
    dt = float(np.median(np.diff(t)))
    dH = np.gradient(H, dt)
    tau_geom = 2.0 * FRAC_X1 / WAVESPEED  # 5.65517s
    
    m_post = (t >= TS) & (t <= (TS + 49.0))
    dH_post = dH[m_post]
    x = dH_post - np.mean(dH_post)
    w = windows.hann(len(x))
    xw = x * w
    N = len(xw)
    spec = fft(xw)
    log_mag = np.log(np.abs(spec) + np.finfo(np.float64).eps)
    ceps = np.real(ifft(log_mag))
    fs = 1.0 / dt
    quefrency = np.arange(N) / fs
    
    # Search window: [tau_geom - 0.08, tau_geom + 0.08]
    m_q = (quefrency >= (tau_geom - 0.08)) & (quefrency <= (tau_geom + 0.08))
    q_sub = quefrency[m_q]
    ceps_sub = ceps[m_q]
    idx_max = np.argmax(np.abs(ceps_sub))
    tau_cep = float(q_sub[idx_max])
    x_cep = WAVESPEED * tau_cep / 2.0
    dx_cep = x_cep - FRAC_X1
    return tau_cep, x_cep, dx_cep

def analyze_16_points():
    rows = []
    
    # Baseline steady at kleak=1e-4 for delta_x_cep reference
    steady_base_dir = os.path.join(sim_base, "n1_k0.0_kleak1e-04", "moc_timeseries.csv")
    df_sb = pd.read_csv(steady_base_dir)
    _, x_cep_base, _ = compute_frozen_cepstrum(df_sb['t'].to_numpy(), df_sb['H_wh'].to_numpy())
    
    for k in K_GRID:
        for kleak in KLEAK_GRID:
            case_name = f"n1_k{k}_kleak{kleak:.0e}"
            csv_path = os.path.join(sim_base, case_name, "moc_timeseries.csv")
            df = pd.read_csv(csv_path)
            t = df['t'].to_numpy()
            H = df['H_wh'].to_numpy()
            
            damp_res = compute_cwt_damping_ratio(t, H)
            tau_cep, x_cep, dx_cep = compute_frozen_cepstrum(t, H)
            
            # Relative shift vs steady at same kleak
            csv_steady_same_kleak = os.path.join(sim_base, f"n1_k0.0_kleak{kleak:.0e}", "moc_timeseries.csv")
            df_ssk = pd.read_csv(csv_steady_same_kleak)
            _, x_cep_steady_same, _ = compute_frozen_cepstrum(df_ssk['t'].to_numpy(), df_ssk['H_wh'].to_numpy())
            delta_x_cep = x_cep - x_cep_steady_same
            
            rows.append({
                'n': 1,
                'friction': 'steady' if k == 0 else f'brunone_k{k}',
                'k': k,
                'kleak_m2_s_sqrtm': kleak,
                'kleak_sci': f"{kleak:.0e}",
                'f0_Hz': damp_res['f0_Hz'],
                'decay_rate_alpha': damp_res['alpha_decay_s_inv'],
                'damping_ratio_zeta': damp_res['zeta_damping_ratio'],
                'fit_r2': damp_res['fit_r2'],
                'tau_cep_s': tau_cep,
                'x_cep_m': x_cep,
                'dx_cep_m': dx_cep,
                'delta_x_cep_m': delta_x_cep,
            })
            
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(data_dir, "k_kleak_damping_surface.csv")
    df_out.to_csv(out_csv, index=False)
    print("\n=== (k x k_leak) 16-Point Orthogonal Damping Surface ===")
    print(df_out[['k', 'kleak_sci', 'decay_rate_alpha', 'damping_ratio_zeta', 'fit_r2', 'dx_cep_m', 'delta_x_cep_m']].to_string())
    print(f"\nSaved 16-point table to {out_csv}")
    
    # Plot diagnostic contour & response surface
    plot_response_surface(df_out)
    return df_out

def plot_response_surface(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Pivot tables (rows: k, cols: kleak)
    pivot_zeta = df.pivot(index='k', columns='kleak_m2_s_sqrtm', values='damping_ratio_zeta')
    pivot_dx = df.pivot(index='k', columns='kleak_m2_s_sqrtm', values='delta_x_cep_m')
    
    K_vals = pivot_zeta.index.values          # [0.0, 0.01, 0.02, 0.05] (Y-axis)
    Kleak_vals = pivot_zeta.columns.values    # [1e-5, 5e-5, 1e-4, 5e-4] (X-axis)
    Kleak_mesh, K_mesh = np.meshgrid(Kleak_vals, K_vals)
    
    # Panel (a): Damping ratio zeta contour
    ax = axes[0]
    cs = ax.contourf(Kleak_mesh, K_mesh, pivot_zeta.values, levels=12, cmap='viridis')
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('CWT Damping Ratio $\\zeta$', fontsize=10)
    ax.contour(Kleak_mesh, K_mesh, pivot_zeta.values, levels=8, colors='k', linewidths=0.8)
    ax.scatter(Kleak_mesh, K_mesh, color='red', s=40, zorder=5, edgecolors='black', label='16 Simulation Points')
    ax.set_xscale('log')
    ax.set_xlabel('Fracture Leakoff Coefficient $k_{\\mathrm{leak}}$ [m$^2$/s/$\\sqrt{\\mathrm{m}}$]', fontsize=10)
    ax.set_ylabel('Unsteady Friction Brunone $k$', fontsize=10)
    ax.set_ylim(-0.003, 0.053)
    ax.set_title('(a) CWT Envelope Damping Ratio $\\zeta(k, k_{\\mathrm{leak}})$', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=9)
    
    # Panel (b): Cepstrum shift delta_x_cep contour
    ax = axes[1]
    cs2 = ax.contourf(Kleak_mesh, K_mesh, pivot_dx.values, levels=12, cmap='plasma')
    cbar2 = fig.colorbar(cs2, ax=ax)
    cbar2.set_label('Cepstral Shift $\\delta x_{\\mathrm{cep}}$ [m]', fontsize=10)
    ax.contour(Kleak_mesh, K_mesh, pivot_dx.values, levels=8, colors='k', linewidths=0.8)
    ax.scatter(Kleak_mesh, K_mesh, color='cyan', s=40, zorder=5, edgecolors='black')
    ax.set_xscale('log')
    ax.set_xlabel('Fracture Leakoff Coefficient $k_{\\mathrm{leak}}$ [m$^2$/s/$\\sqrt{\\mathrm{m}}$]', fontsize=10)
    ax.set_ylabel('Unsteady Friction Brunone $k$', fontsize=10)
    ax.set_ylim(-0.003, 0.053)
    ax.set_title('(b) Cepstral Peak Shift $\\delta x_{\\mathrm{cep}}(k, k_{\\mathrm{leak}})$', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(figs_dir, "fig_k_kleak_damping_surface.png")
    fig_svg = os.path.join(figs_dir, "fig_k_kleak_damping_surface.svg")
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_svg)
    plt.close()
    print(f"Saved corrected diagnostic response surface plots to:\n  {fig_path}\n  {fig_svg}")

if __name__ == '__main__':
    run_16_simulations()
    analyze_16_points()
