# -*- coding: utf-8 -*-
"""
Gantt p3: Shut-in time Tc sweep simulation and clock extraction.
Cases:
- n=1, X1=4100m
- Friction: k=0.01 (Brunone) and k=0.0 (Steady control)
- Tc in {1ms (0.001s), 50ms (0.05s), 200ms (0.2s), 1000ms (1.0s)}
- Wellhead BC: 'ramp' with pump_closure_duration = Tc
- Extract absolute error dx(Tc) = x - X1 for onset, peak, E50, and cepstrum.
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

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore
from moc_simulate.config import WELL_CONFIG, FRACTURE_CONFIG
import moc_simulate.wellbore_moc as wm

out_dir = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data")
sim_out_base = os.path.join(_root, "output", "analysis", "brunone_spacing_effect", "tc_sweep")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(sim_out_base, exist_ok=True)

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
TS = 1.0
TF = 50.0
DT = 0.001

TC_VALUES = [0.001, 0.05, 0.20, 1.00]  # in seconds: 1ms, 50ms, 200ms, 1000ms
K_VALUES = [0.0, 0.01]

def patch_brunone(k_val):
    def mock_k(Re: float) -> float:
        return k_val
    def mock_k_vec(Re_arr: np.ndarray) -> np.ndarray:
        return np.full_like(Re_arr, k_val, dtype=np.float64)
    wm.brunone_k = mock_k
    wm.brunone_k_vec = mock_k_vec

def run_simulations():
    original_brunone_k = wm.brunone_k
    original_brunone_k_vec = wm.brunone_k_vec

    for k in K_VALUES:
        for tc in TC_VALUES:
            tc_ms = int(round(tc * 1000.0))
            case_name = f"n1_k{k}_Tc{tc_ms}ms"
            case_dir = os.path.join(sim_out_base, case_name)
            os.makedirs(case_dir, exist_ok=True)
            csv_path = os.path.join(case_dir, "moc_timeseries.csv")
            
            if os.path.exists(csv_path):
                print(f"Skipping {case_name}, already exists.")
                continue
                
            print(f"Running {case_name} (k={k}, Tc={tc*1000:.1f}ms)...")
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
                wellhead_bc="ramp",
                pump_shut_time=TS,
                pump_closure_duration=tc,
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
                fracture_Cf=[FRACTURE_CONFIG['Cf']],
                fracture_kleak=[FRACTURE_CONFIG['kleak']],
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

def extract_clocks_tc(t: np.ndarray, H: np.ndarray, tc: float, k: float):
    dt = float(np.median(np.diff(t)))
    dH = np.gradient(H, dt)
    t_geom = TS + 2.0 * FRAC_X1 / WAVESPEED  # 6.65517s
    tau_geom = 2.0 * FRAC_X1 / WAVESPEED     # 5.65517s
    
    # First packet window: [t_geom - 0.5, t_geom + 0.5 + tc]
    win_start = t_geom - 0.5
    win_end = t_geom + 0.5 + max(0.5, tc)
    m_win = (t >= win_start) & (t <= win_end)
    t_win = t[m_win]
    dH_win = dH[m_win]
    max_dH = float(np.max(dH_win)) if len(dH_win) > 0 else 0.0
    
    # 1. Onset: first time exceeding 0.01 * max_dH
    onset_mask = dH_win >= (0.01 * max_dH)
    if np.any(onset_mask):
        t_onset = float(t_win[np.argmax(onset_mask)])
    else:
        t_onset = np.nan
        
    # 2. Peak: first peak in window
    # For slow ramp (e.g. Tc=1.0s), peak occurs near t_geom + Tc/2
    dist_pts = max(1, int(round(0.002 / dt)))
    peaks, _ = find_peaks(dH_win, height=0.10 * max_dH, distance=dist_pts)
    if len(peaks) > 0:
        t_peak = float(t_win[peaks[0]])
    else:
        t_peak = float(t_win[np.argmax(dH_win)])
        
    # 3. E50: 50% cumulative positive energy
    y_pos = np.maximum(0.0, dH_win)
    e = y_pos ** 2
    E = np.cumsum(e)
    if E[-1] > 0:
        E_norm = E / E[-1]
        t_E50 = float(np.interp(0.50, E_norm, t_win))
    else:
        t_E50 = np.nan
        
    # 4. Cepstrum: on post shut-in record [TS, TS + 49.0]
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
    
    # Search window: tau in [tau_geom - 0.08, tau_geom + 0.08 + tc/2] (broadened slightly for large Tc)
    search_min = tau_geom - 0.08
    search_max = tau_geom + 0.08 + min(0.5, tc)
    m_q = (quefrency >= search_min) & (quefrency <= search_max)
    q_sub = quefrency[m_q]
    ceps_sub = ceps[m_q]
    idx_max = np.argmax(np.abs(ceps_sub))
    tau_cep = float(q_sub[idx_max])
    
    # Depths
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
    }

def analyze_tc_sweep():
    rows = []
    
    for k in K_VALUES:
        for tc in TC_VALUES:
            tc_ms = int(round(tc * 1000.0))
            case_name = f"n1_k{k}_Tc{tc_ms}ms"
            csv_path = os.path.join(sim_out_base, case_name, "moc_timeseries.csv")
            df_ts = pd.read_csv(csv_path)
            t = df_ts['t'].to_numpy(dtype=np.float64)
            H = df_ts['H_wh'].to_numpy(dtype=np.float64)
            
            res = extract_clocks_tc(t, H, tc, k)
            
            clocks = [
                ('onset', res['t_onset'], res['x_onset'], res['dx_onset']),
                ('peak', res['t_peak'], res['x_peak'], res['dx_peak']),
                ('E50', res['t_E50'], res['x_E50'], res['dx_E50']),
                ('cepstrum', res['tau_cep'], res['x_cep'], res['dx_cep']),
            ]
            for clk_name, clk_time, x_est, dx_est in clocks:
                rows.append({
                    'n': 1,
                    'friction': 'steady' if k == 0 else f'brunone_k{k}',
                    'k_val': k,
                    'Tc_ms': tc_ms,
                    'Tc_s': tc,
                    'X1_m': FRAC_X1,
                    'clock': clk_name,
                    'time_s': clk_time,
                    'x_m': x_est,
                    'dx_m': dx_est,
                })
                
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "Tc_sweep_n1_k0.01.csv")
    df_out.to_csv(out_csv, index=False)
    print("\n=== Tc Sweep Clocks & Error Table ===")
    print(df_out.to_string())
    print(f"\nSaved Tc sweep table to {out_csv}")
    
    return df_out

if __name__ == '__main__':
    run_simulations()
    analyze_tc_sweep()
