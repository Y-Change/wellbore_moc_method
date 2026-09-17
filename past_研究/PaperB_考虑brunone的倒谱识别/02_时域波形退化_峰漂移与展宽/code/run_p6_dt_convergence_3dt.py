# -*- coding: utf-8 -*-
"""
Gantt p6: Comprehensive 3-Level Time-Step Convergence Analysis (dt = 0.5 ms, 1.0 ms, 2.0 ms).
Simulates n=1, X1=4100m, Cf=1.0e-5 m^2, kleak=1.0e-4 m^2/s/sqrt(m), H_ext=100.0m
for k=0.0 (Steady Darcy) and k=0.01 (Brunone IAB) with step shut-in across dt in {0.5, 1.0, 2.0} ms.
Extracts all four characteristic clocks (t_onset, t_peak, t_E50, tau_cep).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, windows
from scipy.fft import fft, ifft

_root = r"e:\water_hammer_research\wellbore_moc_method"
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.wellbore_moc import (
    MocConfig, reynolds, darcy_friction_factor, friction_term_J,
    brunone_k, solve_fracture_node
)
from moc_simulate.config import WELL_CONFIG, FRACTURE_CONFIG

sim_base = os.path.join(_root, "output", "analysis", "brunone_spacing_effect", "p6_robustness")
sim_p5 = os.path.join(_root, "output", "analysis", "brunone_spacing_effect", "p5_models")
dir_02 = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽")
os.makedirs(sim_base, exist_ok=True)
os.makedirs(os.path.join(dir_02, "data"), exist_ok=True)

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
CF = 1.0e-5
KLEAK = 1.0e-4
TS = 1.0
TF = 50.0
G = 9.80665
TAU_GEOM = 2.0 * FRAC_X1 / WAVESPEED
T_GEOM = TS + TAU_GEOM

def simulate_dt(dt_val: float, k_val: float, case_label: str):
    csv_path = os.path.join(sim_base, f"{case_label}.csv")
    if os.path.exists(csv_path):
        print(f"Skipping {case_label}, already exists: {csv_path}")
        return

    print(f"Running simulation {case_label} (dt={dt_val*1000:.1f}ms, k={k_val}, tf={TF}s)...")
    t0 = time.time()

    cfg = MocConfig(
        wellbore_length=WELL_CONFIG['L'],
        wellbore_diameter=WELL_CONFIG['wellbore_diameter'],
        fluid_density=WELL_CONFIG['fluid_density'],
        fluid_viscosity=WELL_CONFIG['fluid_viscosity'],
        wavespeed=WAVESPEED,
        roughness_height=WELL_CONFIG['roughness_height'],
        friction_model="steady" if k_val == 0.0 else "brunone",
        dt=dt_val,
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
    nu = cfg.fluid_viscosity
    K_D = cfg.roughness_height / D
    n_steps = cfg.n_steps
    ga = G / a
    t_s = cfg.pump_shut_time
    V0 = cfg.initial_velocity
    H0 = cfg.initial_head

    Re0 = reynolds(V0, D, nu)
    f_steady = darcy_friction_factor(Re0, K_D, "steady")

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
    is_brunone = (k_val > 0.0)

    for n in range(1, n_steps + 1):
        t = n * dt
        timestamps[n] = t

        V1 = V_prev_right[:-2]
        H1 = H_prev[:-2]
        V2 = V_prev_left[2:]
        H2 = H_prev[2:]

        J1 = friction_term_J(f_steady, D, V1, dt)
        J2 = friction_term_J(f_steady, D, V2, dt)

        if is_brunone and n >= 2:
            dVdt1 = (V_prev_right[:-2] - V_prev2_right[:-2]) / dt
            dVdx1 = (V_prev_right[1:-1] - V_prev_right[:-2]) / dx
            sign_V1 = np.tanh(V1 / V_smooth)
            Ju1 = (k_val / 2.0) * dt * (dVdt1 + a * sign_V1 * np.abs(dVdx1))

            dVdt2 = (V_prev_left[2:] - V_prev2_left[2:]) / dt
            dVdx2 = (V_prev_left[2:] - V_prev_left[1:-1]) / dx
            sign_V2 = np.tanh(V2 / V_smooth)
            Ju2 = (k_val / 2.0) * dt * (dVdt2 + a * sign_V2 * np.abs(dVdx2))

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

        H_f, V_left_f, V_right_f, _ = solve_fracture_node(
            Cp[i_f - 1], Cm[i_f - 1], H_prev[i_f], A, ga,
            CF, KLEAK, FRACTURE_CONFIG['H_ext'], dt
        )
        H_new[i_f] = H_f
        V_new[i_f] = V_left_f

        # Wellhead
        V2_0 = V_prev_left[1]
        H2_0 = H_prev[1]
        J_0 = friction_term_J(f_steady, D, V2_0, dt)
        if is_brunone and n >= 2:
            dVdt_0 = (V_prev_left[1] - V_prev2_left[1]) / dt
            dVdx_0 = (V_prev_left[1] - V_prev_left[0]) / dx
            J_0 += (k_val / 2.0) * dt * (dVdt_0 + a * np.sign(V2_0) * np.abs(dVdx_0))

        Cm_0 = -V2_0 + ga * H2_0 + J_0
        V_wh = 0.0 if t >= t_s else V0
        V_new[0] = V_wh
        H_new[0] = (V_wh + Cm_0) / ga

        # Toe
        V1_N = V_prev_right[N - 1]
        H1_N = H_prev[N - 1]
        J_N = friction_term_J(f_steady, D, V1_N, dt)
        if is_brunone and n >= 2:
            dVdt_N = (V_prev_right[N - 1] - V_prev2_right[N - 1]) / dt
            dVdx_N = (V_prev_right[N] - V_prev_right[N - 1]) / dx
            J_N += (k_val / 2.0) * dt * (dVdt_N + a * np.sign(V1_N) * np.abs(dVdx_N))

        Cp_N = V1_N + ga * H1_N - J_N
        H_new[N] = cfg.toe_head
        V_new[N] = Cp_N - ga * H_new[N]

        V_prev2_left[:] = V_prev_left
        V_prev2_right[:] = V_prev_right
        V_prev_left[:] = V_new
        V_prev_right[:] = V_new
        V_prev_left[i_f] = V_left_f
        V_prev_right[i_f] = V_right_f
        H_prev[:] = H_new
        wh_head_hist[n] = H_new[0]

    df_out = pd.DataFrame({'t': timestamps, 'H_wh': wh_head_hist})
    df_out.to_csv(csv_path, index=False)
    print(f"  Completed {case_label} in {time.time()-t0:.2f}s -> {csv_path}")

def extract_four_clocks(t: np.ndarray, H: np.ndarray):
    dt = float(np.median(np.diff(t)))
    dH = np.gradient(H, dt)

    # First wavepacket window [6.5, 7.5] s
    m_win = (t >= (T_GEOM - 0.5)) & (t <= (T_GEOM + 0.5))
    t_win = t[m_win]
    dH_win = dH[m_win]
    max_dH = float(np.max(dH_win))

    # 1. t_onset
    onset_mask = dH_win >= (0.01 * max_dH)
    t_onset = float(t_win[np.argmax(onset_mask)]) if np.any(onset_mask) else np.nan

    # 2. t_peak
    peaks, _ = find_peaks(dH_win, height=0.10 * max_dH, distance=max(1, int(round(0.002 / dt))))
    t_peak = float(t_win[peaks[0]]) if len(peaks) > 0 else float(t_win[np.argmax(dH_win)])

    # 3. t_E50 (positive cumulative energy 50%)
    y_pos = np.maximum(0.0, dH_win)
    E = np.cumsum(y_pos ** 2)
    if E[-1] > 0:
        t_E50 = float(np.interp(0.50, E / E[-1], t_win))
    else:
        t_E50 = np.nan

    # 4. tau_cep (Hann windowed on dH/dt, t in [1.0, 50.0]s, search [5.575, 5.735]s)
    m_post = (t >= 1.0) & (t <= 50.0)
    dH_post = dH[m_post]
    x = dH_post - np.mean(dH_post)
    N = len(x)
    w = windows.hann(N)
    xw = x * w
    spec = fft(xw)
    log_mag = np.log(np.abs(spec) + np.finfo(np.float64).eps)
    ceps = np.real(ifft(log_mag))
    fs = 1.0 / dt
    quefrency = np.arange(N) / fs

    m_q = (quefrency >= (TAU_GEOM - 0.08)) & (quefrency <= (TAU_GEOM + 0.08))
    q_sub = quefrency[m_q]
    ceps_sub = ceps[m_q]
    tau_cep = float(q_sub[np.argmax(np.abs(ceps_sub))])

    # Convert to depths
    x_onset = WAVESPEED * (t_onset - TS) / 2.0
    x_peak = WAVESPEED * (t_peak - TS) / 2.0
    x_E50 = WAVESPEED * (t_E50 - TS) / 2.0
    x_cep = WAVESPEED * tau_cep / 2.0

    dx_onset = x_onset - FRAC_X1
    dx_peak = x_peak - FRAC_X1
    dx_E50 = x_E50 - FRAC_X1
    dx_cep = x_cep - FRAC_X1

    return {
        't_onset_s': t_onset, 'dx_onset_m': dx_onset,
        't_peak_s': t_peak, 'dx_peak_m': dx_peak,
        't_E50_s': t_E50, 'dx_E50_m': dx_E50,
        'tau_cep_s': tau_cep, 'dx_cep_m': dx_cep,
    }

def main():
    # 1. Ensure dt=2.0 ms simulations exist
    simulate_dt(0.002, 0.0, "dt20_k0.0")
    simulate_dt(0.002, 0.01, "dt20_k0.01")

    # 2. Paths to all 6 runs
    runs = [
        ("dt_0.5ms_k0.0", os.path.join(sim_base, "dt05_k0.0.csv"), 0.5, 0.0),
        ("dt_0.5ms_k0.01", os.path.join(sim_base, "dt05_k0.01.csv"), 0.5, 0.01),
        ("dt_1.0ms_k0.0", os.path.join(sim_p5, "Steady_Darcy.csv"), 1.0, 0.0),
        ("dt_1.0ms_k0.01", os.path.join(sim_p5, "Brunone_IAB_k0.01.csv"), 1.0, 0.01),
        ("dt_2.0ms_k0.0", os.path.join(sim_base, "dt20_k0.0.csv"), 2.0, 0.0),
        ("dt_2.0ms_k0.01", os.path.join(sim_base, "dt20_k0.01.csv"), 2.0, 0.01),
    ]

    rows = []
    for label, path, dt_ms, k in runs:
        df = pd.read_csv(path)
        t = df['t'].to_numpy(dtype=np.float64)
        H = df['H_wh'].to_numpy(dtype=np.float64)
        clocks = extract_four_clocks(t, H)
        clocks['case_label'] = label
        clocks['dt_ms'] = dt_ms
        clocks['k'] = k
        rows.append(clocks)

    df_res = pd.DataFrame(rows)

    # Compute relative shifts vs respective steady baseline (k=0.0) at each grid resolution
    base_05 = df_res[df_res['case_label'] == 'dt_0.5ms_k0.0'].iloc[0]
    base_10 = df_res[df_res['case_label'] == 'dt_1.0ms_k0.0'].iloc[0]
    base_20 = df_res[df_res['case_label'] == 'dt_2.0ms_k0.0'].iloc[0]

    bases = {0.5: base_05, 1.0: base_10, 2.0: base_20}

    delta_onset = []
    delta_peak = []
    delta_E50 = []
    delta_cep = []

    for _, row in df_res.iterrows():
        b = bases[row['dt_ms']]
        delta_onset.append(row['dx_onset_m'] - b['dx_onset_m'])
        delta_peak.append(row['dx_peak_m'] - b['dx_peak_m'])
        delta_E50.append(row['dx_E50_m'] - b['dx_E50_m'])
        delta_cep.append(row['dx_cep_m'] - b['dx_cep_m'])

    df_res['delta_x_onset_m'] = delta_onset
    df_res['delta_x_peak_m'] = delta_peak
    df_res['delta_x_E50_m'] = delta_E50
    df_res['delta_x_cep_m'] = delta_cep

    # Reorder columns
    cols = [
        'case_label', 'dt_ms', 'k',
        't_onset_s', 'dx_onset_m', 'delta_x_onset_m',
        't_peak_s', 'dx_peak_m', 'delta_x_peak_m',
        't_E50_s', 'dx_E50_m', 'delta_x_E50_m',
        'tau_cep_s', 'dx_cep_m', 'delta_x_cep_m'
    ]
    df_res = df_res[cols]

    out_csv = os.path.join(dir_02, "data", "p6_grid_sensitivity.csv")
    df_res.to_csv(out_csv, index=False)
    print("\n=== Gantt p6: 3-Level Time-Step Convergence Results ===")
    print(df_res.to_string())

if __name__ == '__main__':
    main()
