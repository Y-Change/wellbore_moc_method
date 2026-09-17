# -*- coding: utf-8 -*-
"""
Gantt p2: Onset decision experiment.
Evaluate absolute depth error dx = x - X1 for:
1. Dataset A: Constant k single fracture (n=1, k in {0, 0.01, 0.02, 0.05})
2. Dataset B: 5 Brunone single-fracture leakoff cases and 5 steady controls.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, windows
from scipy.fft import fft, ifft

_root = r"e:\water_hammer_research\wellbore_moc_method"
p1_input_dir = os.path.join(_root, "output", "analysis", "brunone_spacing_effect")
leakoff_root = os.path.join(_root, "PaperA井口多裂缝水击响应", "03_leakoff验证")
out_dir = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "02_时域波形退化_峰漂移与展宽", "data")
os.makedirs(out_dir, exist_ok=True)

WAVESPEED_DEFAULT = 1450.0
PEAK_HEIGHT_FRAC = 0.10
PEAK_MIN_SEP_S = 0.002
ONSET_FRAC = 0.01

def extract_clocks_from_ts(t: np.ndarray, H: np.ndarray, ts: float, a: float, X1: float):
    dt = float(np.median(np.diff(t)))
    dH = np.gradient(H, dt)
    t_geom = ts + 2.0 * X1 / a
    tau_geom = 2.0 * X1 / a
    
    # 1. First packet window: [t_geom - 0.5, t_geom + 0.5] s
    m_win = (t >= (t_geom - 0.5)) & (t <= (t_geom + 0.5))
    t_win = t[m_win]
    dH_win = dH[m_win]
    max_dH = float(np.max(dH_win)) if len(dH_win) > 0 else 0.0
    
    # Onset
    onset_mask = dH_win >= (ONSET_FRAC * max_dH)
    if np.any(onset_mask):
        t_onset = float(t_win[np.argmax(onset_mask)])
    else:
        t_onset = np.nan
        
    # Peak
    dist_pts = max(1, int(round(PEAK_MIN_SEP_S / dt)))
    height_thresh = PEAK_HEIGHT_FRAC * max_dH
    peaks, _ = find_peaks(dH_win, height=height_thresh, distance=dist_pts)
    if len(peaks) > 0:
        t_peak = float(t_win[peaks[0]])
    else:
        t_peak = float(t_win[np.argmax(dH_win)])
        
    # E50
    y_pos = np.maximum(0.0, dH_win)
    e = y_pos ** 2
    E = np.cumsum(e)
    if E[-1] > 0:
        E_norm = E / E[-1]
        t_E50 = float(np.interp(0.50, E_norm, t_win))
    else:
        t_E50 = np.nan
        
    # 2. Cepstrum on post shut-in record: t in [ts, ts + 49.0] s
    m_post = (t >= ts) & (t <= (ts + 49.0))
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
    
    # Search window: tau in [tau_geom - 0.08, tau_geom + 0.08] s
    m_q = (quefrency >= (tau_geom - 0.08)) & (quefrency <= (tau_geom + 0.08))
    q_sub = quefrency[m_q]
    ceps_sub = ceps[m_q]
    idx_max = np.argmax(np.abs(ceps_sub))
    tau_cep = float(q_sub[idx_max])
    
    # Compute estimated depths
    x_onset = a * (t_onset - ts) / 2.0
    x_peak = a * (t_peak - ts) / 2.0
    x_E50 = a * (t_E50 - ts) / 2.0
    x_cep = a * tau_cep / 2.0
    
    dx_onset = x_onset - X1
    dx_peak = x_peak - X1
    dx_E50 = x_E50 - X1
    dx_cep = x_cep - X1
    
    return {
        't_onset': t_onset, 'x_onset': x_onset, 'dx_onset': dx_onset,
        't_peak': t_peak, 'x_peak': x_peak, 'dx_peak': dx_peak,
        't_E50': t_E50, 'x_E50': x_E50, 'dx_E50': dx_E50,
        'tau_cep': tau_cep, 'x_cep': x_cep, 'dx_cep': dx_cep,
    }

def run_p2():
    records = []
    
    # ==========================================
    # A. Constant k single fracture (n=1)
    # ==========================================
    for k in [0, 0.01, 0.02, 0.05]:
        csv_path = os.path.join(p1_input_dir, f"n1_k{k}", "moc_timeseries.csv")
        df = pd.read_csv(csv_path)
        t = df['t'].to_numpy(dtype=np.float64)
        H = df['H_wh'].to_numpy(dtype=np.float64)
        ts = 1.0
        a = 1450.0
        X1 = 4100.0
        
        res = extract_clocks_from_ts(t, H, ts, a, X1)
        
        # Add clock rows
        clocks = [
            ('onset', res['t_onset'], res['x_onset'], res['dx_onset']),
            ('peak', res['t_peak'], res['x_peak'], res['dx_peak']),
            ('E50', res['t_E50'], res['x_E50'], res['dx_E50']),
            ('cepstrum', res['tau_cep'], res['x_cep'], res['dx_cep']),
        ]
        for clk_name, clk_time, x_est, dx_est in clocks:
            records.append({
                'source': 'Matrix_B_Constant_k',
                'n': 1,
                'friction': 'steady' if k == 0 else f'brunone_k{k}',
                'k_val': k,
                'X1_m': X1,
                'clock': clk_name,
                'time_s': clk_time,
                'x_m': x_est,
                'dx_m': dx_est,
            })
            
    # ==========================================
    # B. Leakoff 5 Brunone + 5 Steady Single-fracture cases
    # ==========================================
    frictions = [
        'brunone_D5', 'brunone_D10', 'brunone_D20', 'brunone_D50', 'brunone_D100',
        'steady_D5', 'steady_D10', 'steady_D20', 'steady_D50', 'steady_D100'
    ]
    
    for fric in frictions:
        case_dir = os.path.join(leakoff_root, fric, 'single')
        meta_path = os.path.join(case_dir, 'moc_leakoff.json')
        csv_path = os.path.join(case_dir, 'moc_timeseries.csv')
        
        if not os.path.exists(meta_path) or not os.path.exists(csv_path):
            print(f"Warning: {case_dir} missing data!")
            continue
            
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        cfg = meta.get('config', {})
        X1 = float(cfg.get('x_f', [4100.0])[0])
        ts = float(cfg.get('ts', 1.0))
        a = float(cfg.get('a', 1450.0))
        
        df = pd.read_csv(csv_path)
        t = df['t'].to_numpy(dtype=np.float64)
        H = df['H_wh'].to_numpy(dtype=np.float64)
        
        res = extract_clocks_from_ts(t, H, ts, a, X1)
        
        clocks = [
            ('onset', res['t_onset'], res['x_onset'], res['dx_onset']),
            ('peak', res['t_peak'], res['x_peak'], res['dx_peak']),
            ('E50', res['t_E50'], res['x_E50'], res['dx_E50']),
            ('cepstrum', res['tau_cep'], res['x_cep'], res['dx_cep']),
        ]
        for clk_name, clk_time, x_est, dx_est in clocks:
            records.append({
                'source': 'Leakoff_Re_dependent',
                'n': 1,
                'friction': fric,
                'k_val': 'k(Re)' if 'brunone' in fric else 0.0,
                'X1_m': X1,
                'clock': clk_name,
                'time_s': clk_time,
                'x_m': x_est,
                'dx_m': dx_est,
            })
            
    df_out = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, "onset_correction_verdict.csv")
    df_out.to_csv(out_csv, index=False)
    print("=== Gantt p2 Onset Decision Experiment Verdict ===")
    print(df_out.to_string())
    print(f"\nSaved verdict table to {out_csv}")
    
    return df_out

if __name__ == '__main__':
    run_p2()
