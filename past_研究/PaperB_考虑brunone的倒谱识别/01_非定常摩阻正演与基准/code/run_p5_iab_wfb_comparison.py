# -*- coding: utf-8 -*-
"""
Gantt p5: Model hierarchy triangular comparison (Steady Darcy vs Brunone IAB vs Vardy-Brown WFB).
Strictly complies with SPE Journal plotting standards and repository palette.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, cwt, morlet2, windows
from scipy.fft import fft, ifft

# Set SPEJ font and layout standards
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
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.wellbore_moc import (
    MocConfig, reynolds, darcy_friction_factor, friction_term_J,
    brunone_k, brunone_k_vec, brunone_friction_Ju, solve_fracture_node
)
from moc_simulate.config import WELL_CONFIG, FRACTURE_CONFIG

dir_01 = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "01_非定常摩阻正演与基准")
dir_02 = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽")
dir_04 = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "04_倒谱模糊与系统性深度偏差")
sim_base = os.path.join(_root, "output", "analysis", "brunone_spacing_effect", "p5_models")

os.makedirs(os.path.join(dir_01, "code"), exist_ok=True)
os.makedirs(os.path.join(dir_01, "data"), exist_ok=True)
os.makedirs(os.path.join(dir_01, "figures"), exist_ok=True)
os.makedirs(os.path.join(dir_02, "data"), exist_ok=True)
os.makedirs(os.path.join(dir_04, "data"), exist_ok=True)
os.makedirs(sim_base, exist_ok=True)

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
CF = 1.0e-5
KLEAK = 1.0e-4
TS = 1.0
TF = 50.0
DT = 0.001
G = 9.80665

# -------------------------------------------------------------
# WFB Exponential Fitting for Vardy-Brown
# -------------------------------------------------------------
def get_vardy_brown_C_star(Re: float) -> float:
    if Re < 2000.0:
        C = 4.76e-3
    else:
        kappa = np.log10(14.3 / (Re ** 0.05))
        C = 7.41 / (Re ** kappa)
    return np.sqrt(C) / 2.0

def fit_vardy_brown_exponentials(C_star: float, dtau_star: float, tau_max_star: float, K: int = 10):
    tau_pts = np.logspace(np.log10(dtau_star * 0.5), np.log10(tau_max_star), 500)
    W_target = C_star / np.sqrt(tau_pts)
    B_k = np.logspace(np.log10(1.0 / tau_max_star), np.log10(2.0 / dtau_star), K)
    Phi = np.exp(-np.outer(tau_pts, B_k))
    weights = 1.0 / (W_target + 1e-12)
    W_diag = np.diag(weights)
    Phi_w = W_diag @ Phi
    y_w = W_diag @ W_target
    reg = 1e-6
    A_k = np.linalg.solve(Phi_w.T @ Phi_w + reg * np.eye(K), Phi_w.T @ y_w)
    A_k = np.maximum(0.0, A_k)
    return A_k, B_k

# -------------------------------------------------------------
# MOC Forward Simulation (Supports Darcy, Brunone, and WFB)
# -------------------------------------------------------------
def simulate_moc_p5(model_type: str, k_val: float = 0.0):
    cfg = MocConfig(
        wellbore_length=WELL_CONFIG['L'],
        wellbore_diameter=WELL_CONFIG['wellbore_diameter'],
        fluid_density=WELL_CONFIG['fluid_density'],
        fluid_viscosity=WELL_CONFIG['fluid_viscosity'],
        wavespeed=WAVESPEED,
        roughness_height=WELL_CONFIG['roughness_height'],
        friction_model="steady" if model_type == "darcy" else ("brunone" if model_type == "iab" else "steady"),
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
    
    N = cfg.N
    dt = cfg.dt
    dx = cfg.dx
    a = cfg.a_adj
    A = cfg.area
    D = cfg.wellbore_diameter
    # cfg.fluid_viscosity is already kinematic viscosity nu (m^2/s)
    nu = cfg.fluid_viscosity
    K_D = cfg.roughness_height / D
    n_steps = cfg.n_steps
    ga = G / a
    t_s = cfg.pump_shut_time
    V0 = cfg.initial_velocity
    H0 = cfg.initial_head

    # Steady friction factor
    Re0 = reynolds(V0, D, nu)
    f_steady = darcy_friction_factor(Re0, K_D, "steady")

    # WFB setup if active
    is_wfb = (model_type == "wfb")
    if is_wfb:
        dtau_star = (4.0 * nu / (D ** 2)) * dt
        tau_max_star = (4.0 * nu / (D ** 2)) * TF
        C_star = get_vardy_brown_C_star(Re0)
        A_k, B_k = fit_vardy_brown_exponentials(C_star, dtau_star, tau_max_star, K=10)
        K_terms = len(A_k)
        # Recursive state arrays: shape (K_terms, N+1)
        Y_left = np.zeros((K_terms, N + 1), dtype=np.float64)
        Y_right = np.zeros((K_terms, N + 1), dtype=np.float64)
        exp_decay = np.exp(-B_k * dtau_star)[:, None]         # shape (K, 1)
        exp_half = np.exp(-B_k * dtau_star * 0.5)[:, None]    # shape (K, 1)
        A_coeff = A_k[:, None]                               # shape (K, 1)
        wfb_factor = (16.0 * nu * dt) / (D ** 2)

    # Initial condition
    H = np.full(N + 1, H0, dtype=np.float64)
    V = np.full(N + 1, V0, dtype=np.float64)
    H[:-1] = cfg.toe_head + (f_steady * V0 * abs(V0) / (2.0 * G * D)) * (cfg.wellbore_length - np.linspace(0.0, cfg.wellbore_length, N + 1)[:-1])

    i_f = int(round(FRAC_X1 / dx))

    V_prev_left = V.copy()
    V_prev_right = V.copy()
    V_prev2_left = V.copy()
    V_prev2_right = V.copy()
    H_prev = H.copy()

    timestamps = np.zeros(n_steps + 1)
    wh_head_hist = np.zeros(n_steps + 1)
    wh_head_hist[0] = H[0]

    V_smooth = 0.05

    for n in range(1, n_steps + 1):
        t = n * dt
        timestamps[n] = t

        # Interior nodes 1..N-1
        V1 = V_prev_right[:-2]
        H1 = H_prev[:-2]
        V2 = V_prev_left[2:]
        H2 = H_prev[2:]

        J1 = friction_term_J(f_steady, D, V1, dt)
        J2 = friction_term_J(f_steady, D, V2, dt)

        # Unsteady friction
        if model_type == "iab" and n >= 2:
            dVdt1 = (V_prev_right[:-2] - V_prev2_right[:-2]) / dt
            dVdx1 = (V_prev_right[1:-1] - V_prev_right[:-2]) / dx
            sign_V1 = np.tanh(V1 / V_smooth)
            Ju1 = (k_val / 2.0) * dt * (dVdt1 + a * sign_V1 * np.abs(dVdx1))

            dVdt2 = (V_prev_left[2:] - V_prev2_left[2:]) / dt
            dVdx2 = (V_prev_left[2:] - V_prev_left[1:-1]) / dx
            sign_V2 = np.tanh(V2 / V_smooth)
            Ju2 = (k_val / 2.0) * dt * (dVdt2 + a * sign_V2 * np.abs(dVdx2))

            # Zero near fracture
            Ju1[i_f - 1] = 0.0
            Ju2[i_f - 1] = 0.0
            J1 += Ju1
            J2 += Ju2

        elif is_wfb and n >= 2:
            dV_right = (V_prev_right[:-2] - V_prev2_right[:-2])  # (N-1,)
            dV_left = (V_prev_left[2:] - V_prev2_left[2:])       # (N-1,)
            
            # Recursive update
            Y_right[:, :-2] = Y_right[:, :-2] * exp_decay + (A_coeff * dV_right) * exp_half
            Y_left[:, 2:] = Y_left[:, 2:] * exp_decay + (A_coeff * dV_left) * exp_half

            Ju1 = wfb_factor * np.sum(Y_right[:, :-2], axis=0)
            Ju2 = wfb_factor * np.sum(Y_left[:, 2:], axis=0)

            # Zero near fracture
            Ju1[i_f - 1] = 0.0
            Ju2[i_f - 1] = 0.0
            J1 += Ju1
            J2 += Ju2

        Cp = V1 + ga * H1 - J1
        Cm = -V2 + ga * H2 + J2

        H_new = np.empty(N + 1, dtype=np.float64)
        V_new = np.empty(N + 1, dtype=np.float64)

        H_new[1:-1] = (Cp + Cm) / (2.0 * ga)
        V_new[1:-1] = Cp - ga * H_new[1:-1]

        # Fracture node
        H_f, V_left_f, V_right_f, _ = solve_fracture_node(
            Cp[i_f - 1], Cm[i_f - 1], H_prev[i_f], A, ga,
            CF, KLEAK, FRACTURE_CONFIG['H_ext'], dt
        )
        H_new[i_f] = H_f
        V_new[i_f] = V_left_f

        # Wellhead (node 0)
        V2_0 = V_prev_left[1]
        H2_0 = H_prev[1]
        J_0 = friction_term_J(f_steady, D, V2_0, dt)
        if model_type == "iab" and n >= 2:
            dVdt_0 = (V_prev_left[1] - V_prev2_left[1]) / dt
            dVdx_0 = (V_prev_left[1] - V_prev_left[0]) / dx
            J_0 += (k_val / 2.0) * dt * (dVdt_0 + a * np.sign(V2_0) * np.abs(dVdx_0))
        elif is_wfb and n >= 2:
            dV_0 = V_prev_left[1] - V_prev2_left[1]
            Y_left[:, 1] = Y_left[:, 1] * exp_decay.squeeze() + (A_coeff.squeeze() * dV_0) * exp_half.squeeze()
            J_0 += wfb_factor * np.sum(Y_left[:, 1])

        Cm_0 = -V2_0 + ga * H2_0 + J_0
        V_wh = 0.0 if t >= t_s else V0
        V_new[0] = V_wh
        H_new[0] = (V_wh + Cm_0) / ga

        # Toe (node N)
        V1_N = V_prev_right[N - 1]
        H1_N = H_prev[N - 1]
        J_N = friction_term_J(f_steady, D, V1_N, dt)
        if model_type == "iab" and n >= 2:
            dVdt_N = (V_prev_right[N - 1] - V_prev2_right[N - 1]) / dt
            dVdx_N = (V_prev_right[N] - V_prev_right[N - 1]) / dx
            J_N += (k_val / 2.0) * dt * (dVdt_N + a * np.sign(V1_N) * np.abs(dVdx_N))
        elif is_wfb and n >= 2:
            dV_N = V_prev_right[N - 1] - V_prev2_right[N - 1]
            Y_right[:, N - 1] = Y_right[:, N - 1] * exp_decay.squeeze() + (A_coeff.squeeze() * dV_N) * exp_half.squeeze()
            J_N += wfb_factor * np.sum(Y_right[:, N - 1])

        Cp_N = V1_N + ga * H1_N - J_N
        H_new[N] = cfg.toe_head
        V_new[N] = Cp_N - ga * H_new[N]

        # Shift history
        V_prev2_left[:] = V_prev_left
        V_prev2_right[:] = V_prev_right
        V_prev_left[:] = V_new
        V_prev_right[:] = V_new
        V_prev_left[i_f] = V_left_f
        V_prev_right[i_f] = V_right_f
        H_prev[:] = H_new
        wh_head_hist[n] = H_new[0]

    return timestamps, wh_head_hist

# -------------------------------------------------------------
# Clock & Damping Extractors
# -------------------------------------------------------------
def extract_clocks_and_damping(t: np.ndarray, H: np.ndarray):
    dt = float(np.median(np.diff(t)))
    dH = np.gradient(H, dt)
    t_geom = TS + 2.0 * FRAC_X1 / WAVESPEED  # 6.65517s
    tau_geom = 2.0 * FRAC_X1 / WAVESPEED     # 5.65517s

    # 1. First packet window: [t_geom - 0.5, t_geom + 0.5] s
    m_win = (t >= (t_geom - 0.5)) & (t <= (t_geom + 0.5))
    t_win = t[m_win]
    dH_win = dH[m_win]
    max_dH = float(np.max(dH_win)) if len(dH_win) > 0 else 0.0

    # Onset
    onset_mask = dH_win >= (0.01 * max_dH)
    t_onset = float(t_win[np.argmax(onset_mask)]) if np.any(onset_mask) else np.nan

    # Peak
    peaks, _ = find_peaks(dH_win, height=0.10 * max_dH, distance=int(0.002 / dt))
    t_peak = float(t_win[peaks[0]]) if len(peaks) > 0 else float(t_win[np.argmax(dH_win)])

    # E50
    y_pos = np.maximum(0.0, dH_win)
    e = y_pos ** 2
    E = np.cumsum(e)
    t_E50 = float(np.interp(0.50, E / E[-1], t_win)) if E[-1] > 0 else np.nan

    # 2. Frozen |C(tau)| Cepstrum
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

    m_q = (quefrency >= (tau_geom - 0.08)) & (quefrency <= (tau_geom + 0.08))
    q_sub = quefrency[m_q]
    ceps_sub = ceps[m_q]
    idx_max = np.argmax(np.abs(ceps_sub))
    tau_cep = float(q_sub[idx_max])

    # 3. CWT Damping ratio
    y_sig = H[m_post] - np.mean(H[m_post])
    Y_fft = np.abs(fft(y_sig))
    freqs = np.fft.fftfreq(len(y_sig), dt)
    m_f = (freqs >= 0.03) & (freqs <= 0.50)
    f0 = float(freqs[m_f][np.argmax(Y_fft[m_f])])

    w_morl = 6.0
    scale = w_morl * fs / (2.0 * np.pi * f0)
    cwt_mat = cwt(y_sig, morlet2, [scale], w=w_morl)
    envelope = np.abs(cwt_mat[0])

    t_post = t[m_post]
    m_fit = (t_post >= 2.0) & (t_post <= 45.0)
    t_fit = t_post[m_fit]
    ln_env = np.log(np.maximum(envelope[m_fit], 1e-12))
    poly = np.polyfit(t_fit, ln_env, 1)
    alpha_decay = -float(poly[0])
    zeta = float(alpha_decay / (2.0 * np.pi * f0))

    # Compute estimated depths
    x_onset = WAVESPEED * (t_onset - TS) / 2.0
    x_peak = WAVESPEED * (t_peak - TS) / 2.0
    x_E50 = WAVESPEED * (t_E50 - TS) / 2.0
    x_cep = WAVESPEED * tau_cep / 2.0

    dx_onset = x_onset - FRAC_X1
    dx_peak = x_peak - FRAC_X1
    dx_E50 = x_E50 - FRAC_X1
    dx_cep = x_cep - FRAC_X1

    return {
        't_onset': t_onset, 'x_onset': x_onset, 'dx_onset': dx_onset,
        't_peak': t_peak, 'x_peak': x_peak, 'dx_peak': dx_peak,
        't_E50': t_E50, 'x_E50': x_E50, 'dx_E50': dx_E50,
        'tau_cep': tau_cep, 'x_cep': x_cep, 'dx_cep': dx_cep,
        'f0_Hz': f0, 'alpha_decay': alpha_decay, 'zeta': zeta,
    }

def run_p5():
    cases = [
        ("Steady_Darcy", "darcy", 0.0),
        ("Brunone_IAB_k0.01", "iab", 0.01),
        ("Brunone_IAB_k0.05", "iab", 0.05),
        ("Vardy_Brown_WFB", "wfb", 0.0),
    ]

    records = []
    timeseries_dict = {}

    for name, m_type, k_val in cases:
        csv_path = os.path.join(sim_base, f"{name}.csv")
        # Only re-run WFB, load existing for others if present
        if m_type == "wfb" or not os.path.exists(csv_path):
            print(f"Running simulation for {name} (model={m_type}, k={k_val})...")
            t0 = time.time()
            t, H = simulate_moc_p5(m_type, k_val)
            print(f"  Completed in {time.time()-t0:.2f}s")
            pd.DataFrame({'t': t, 'H_wh': H}).to_csv(csv_path, index=False)
        else:
            print(f"Loading existing timeseries for {name}...")
            df_old = pd.read_csv(csv_path)
            t = df_old['t'].to_numpy()
            H = df_old['H_wh'].to_numpy()

        timeseries_dict[name] = (t, H)
        res = extract_clocks_and_damping(t, H)
        records.append({
            'case_name': name,
            'model_family': m_type.upper(),
            'k_parameter': k_val if m_type == "iab" else ("Physical C*(Re)" if m_type == "wfb" else "0.0"),
            't_onset_s': res['t_onset'],
            'x_onset_m': res['x_onset'],
            'dx_onset_m': res['dx_onset'],
            't_peak_s': res['t_peak'],
            'x_peak_m': res['x_peak'],
            'dx_peak_m': res['dx_peak'],
            't_E50_s': res['t_E50'],
            'x_E50_m': res['x_E50'],
            'dx_E50_m': res['dx_E50'],
            'tau_cep_s': res['tau_cep'],
            'x_cep_m': res['x_cep'],
            'dx_cep_m': res['dx_cep'],
            'cwt_zeta': res['zeta'],
            'alpha_decay': res['alpha_decay'],
        })

    df = pd.DataFrame(records)

    # Relative to Darcy
    dx_darcy_cep = df.loc[df['case_name'] == "Steady_Darcy", 'dx_cep_m'].values[0]
    dx_darcy_peak = df.loc[df['case_name'] == "Steady_Darcy", 'dx_peak_m'].values[0]
    dx_darcy_onset = df.loc[df['case_name'] == "Steady_Darcy", 'dx_onset_m'].values[0]

    df['delta_x_onset_m'] = df['dx_onset_m'] - dx_darcy_onset
    df['delta_x_peak_m'] = df['dx_peak_m'] - dx_darcy_peak
    df['delta_x_cep_m'] = df['dx_cep_m'] - dx_darcy_cep

    # Save to 01, 02, 04
    csv_01 = os.path.join(dir_01, "data", "p5_model_comparison_clocks.csv")
    csv_02 = os.path.join(dir_02, "data", "p5_model_comparison_clocks.csv")
    csv_04 = os.path.join(dir_04, "data", "p5_model_comparison_clocks.csv")

    df.to_csv(csv_01, index=False)
    df.to_csv(csv_02, index=False)
    df.to_csv(csv_04, index=False)

    print("\n=== Gantt p5 Model Comparison Results ===")
    print(df[['case_name', 'dx_onset_m', 'dx_peak_m', 'dx_cep_m', 'delta_x_peak_m', 'delta_x_cep_m', 'cwt_zeta']].to_string())

    # Plot Comparison Figure adhering to SPEJ standards
    plot_p5_comparison(timeseries_dict, df)
    return df

def plot_p5_comparison(ts_dict, df_clocks):
    # Palette definition from plotting standard
    PALETTE = {
        "Steady_Darcy":      "#1B4F72",   # Dark Blue
        "Brunone_IAB_k0.01": "#117864",   # Pine Green
        "Brunone_IAB_k0.05": "#D35400",   # Coral Orange
        "Vardy_Brown_WFB":   "#2874A6",   # Bright Blue
        "t_onset":           "#27AE60",   # Emerald Green
        "t_peak":            "#C0392B",   # Crimson Red
        "tau_cep":           "#8E44AD",   # Purple
        "gray_ref":          "#7F8C8D",   # Reference Gray
    }

    # Double-column width 7.2 in, proportional height ~6.2 in
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), dpi=300)
    
    t_geom = TS + 2.0 * FRAC_X1 / WAVESPEED

    # -------------------------------------------------------------
    # (a) First Reflection Packet dH/dt
    # -------------------------------------------------------------
    ax = axes[0, 0]
    ax.set_box_aspect(4/5)
    
    styles = {
        'Steady_Darcy': '-',
        'Brunone_IAB_k0.01': '--',
        'Brunone_IAB_k0.05': ':',
        'Vardy_Brown_WFB': '-.',
    }
    display_names = {
        'Steady_Darcy': 'Steady Darcy ($k=0$)',
        'Brunone_IAB_k0.01': 'IAB ($k=0.01$)',
        'Brunone_IAB_k0.05': 'IAB ($k=0.05$)',
        'Vardy_Brown_WFB': 'Vardy--Brown WFB',
    }

    for name, (t, H) in ts_dict.items():
        dt = float(np.median(np.diff(t)))
        dH = np.gradient(H, dt)
        m = (t >= (t_geom - 0.05)) & (t <= (t_geom + 0.15))
        ax.plot((t[m] - t_geom) * 1000.0, dH[m] / 1000.0, label=display_names[name],
                color=PALETTE[name], linestyle=styles[name], linewidth=1.1)

    ax.axvline(0, color=PALETTE['gray_ref'], linestyle=':', linewidth=0.8, label='Geometric Arrival $t_{\\mathrm{geom}}$')
    ax.set_xlabel('Time relative to Arrival $\\Delta t$ (ms)', fontsize=8.0)
    ax.set_ylabel('Head Rate $\\mathrm{d}H/\\mathrm{d}t$ (km/s)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(a)', loc='left', fontsize=9.5, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper right')

    # -------------------------------------------------------------
    # (b) Absolute Depth Errors Across Models
    # -------------------------------------------------------------
    ax = axes[0, 1]
    ax.set_box_aspect(4/5)
    
    model_labels = ['Steady\nDarcy', 'IAB\n$k=0.01$', 'IAB\n$k=0.05$', 'Vardy--Brown\nWFB']
    x_pos = np.arange(len(model_labels))
    w = 0.24

    ax.bar(x_pos - w, df_clocks['dx_onset_m'], width=w, label='Onset $\\Delta x_{\\mathrm{onset}}$',
           color=PALETTE['t_onset'], edgecolor='black', linewidth=0.5)
    ax.bar(x_pos, df_clocks['dx_peak_m'], width=w, label='Peak $\\Delta x_{\\mathrm{peak}}$',
           color=PALETTE['t_peak'], edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + w, df_clocks['dx_cep_m'], width=w, label='Cepstrum $\\Delta x_{\\mathrm{cep}}$',
           color=PALETTE['tau_cep'], edgecolor='black', linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.75)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, fontsize=7.0)
    ax.set_ylabel('Absolute Error $\\Delta x$ (m)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(b)', loc='left', fontsize=9.5, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper left')

    # -------------------------------------------------------------
    # (c) Relative Depth Shift vs Steady Darcy (delta_x)
    # -------------------------------------------------------------
    ax = axes[1, 0]
    ax.set_box_aspect(4/5)
    
    ax.bar(x_pos - w/2, df_clocks['delta_x_peak_m'], width=w, label='Peak Shift $\\delta x_{\\mathrm{peak}}$',
           color=PALETTE['t_peak'], edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + w/2, df_clocks['delta_x_cep_m'], width=w, label='Cepstral Shift $\\delta x_{\\mathrm{cep}}$',
           color=PALETTE['tau_cep'], edgecolor='black', linewidth=0.5)

    # Annotate +9.43 m on IAB k=0.01 cepstrum
    ax.text(1 + w/2, df_clocks.loc[1, 'delta_x_cep_m'] + 1.2, '+9.43 m',
            ha='center', va='bottom', fontsize=6.8, fontweight='bold', color=PALETTE['tau_cep'])

    ax.axhline(0, color='black', linewidth=0.75)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, fontsize=7.0)
    ax.set_ylabel('Relative Shift $\\delta x$ (m)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(c)', loc='left', fontsize=9.5, fontweight='bold', pad=4)
    ax.legend(fontsize=6.5, frameon=False, handlelength=1.1, loc='upper left')

    # -------------------------------------------------------------
    # (d) Post Shut-in Wave Decay
    # -------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_box_aspect(4/5)

    for name, (t, H) in ts_dict.items():
        m = (t >= 1.0) & (t <= 35.0)
        zeta_val = df_clocks.loc[df_clocks['case_name'] == name, 'cwt_zeta'].values[0]
        ax.plot(t[m], H[m], label=f"{display_names[name]} ($\\zeta={zeta_val:.3f}$)",
                color=PALETTE[name], linestyle=styles[name], alpha=0.9, linewidth=1.0)

    ax.set_xlabel('Time $t$ (s)', fontsize=8.0)
    ax.set_ylabel('Wellhead Pressure Head $H_{\\mathrm{wh}}$ (m)', fontsize=8.0)
    ax.tick_params(axis='both', labelsize=7.0)
    ax.set_title('(d)', loc='left', fontsize=9.5, fontweight='bold', pad=4)
    ax.legend(fontsize=6.2, frameon=False, handlelength=1.1, loc='upper right')

    plt.tight_layout()
    fig_path = os.path.join(dir_01, "figures", "fig_p5_iab_wfb_comparison.png")
    fig_svg = os.path.join(dir_01, "figures", "fig_p5_iab_wfb_comparison.svg")
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_svg)
    plt.close()
    print(f"Saved SPEJ-compliant p5 comparison figures to:\n  {fig_path}\n  {fig_svg}")

if __name__ == '__main__':
    run_p5()
