# -*- coding: utf-8 -*-
"""研究 3：裂缝能量衰减模型数据生成。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time as time_module
from typing import Dict, List

import numpy as np

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

from paths import output_path, SERIES_DECAY_REGRESSION
from wellbore_moc import MocConfig, simulate_wellbore
from validation.config import WELL_CONFIG, SIM_CONFIG, FRACTURE_CONFIG, FRICTION_PARAMS
from cepstrum_mocdata import compute_moc_cepstrum_1d, compute_moc_cepstrum

def _build_moc_config(friction_key: str) -> MocConfig:
    fr_params = FRICTION_PARAMS[friction_key]
    w, s = WELL_CONFIG, SIM_CONFIG
    return MocConfig(
        wellbore_length=w['L'],
        wellbore_diameter=w['wellbore_diameter'],
        fluid_density=w['fluid_density'],
        fluid_viscosity=w['fluid_viscosity'],
        wavespeed=w['wavespeed'],
        roughness_height=w['roughness_height'],
        friction_model=fr_params['friction_model'],
        dt=s['dt'], tf=s['tf'],
        wellhead_bc='velocity_step', pump_shut_time=s['ts'],
        initial_velocity=w['V0'], initial_head=w['H0'],
        theta=w['theta'], toe_bc='reservoir', toe_head=w['H0'],
    )

def _extract_local_peaks(depth: np.ndarray, response: np.ndarray, x_f_list: List[float], search_radius: float = 15.0) -> List[float]:
    peaks = []
    for x_f in x_f_list:
        mask = (depth >= x_f - search_radius) & (depth <= x_f + search_radius)
        if np.any(mask):
            peaks.append(float(np.max(response[mask])))
        else:
            peaks.append(0.0)
    return peaks

def run_one(friction_key: str, x1: float, spacing_m: float, n_fracs: int) -> List[Dict]:
    cfg = _build_moc_config(friction_key)
    x_f_list = [x1 + i * spacing_m for i in range(n_fracs)]
    Cf = float(FRACTURE_CONFIG['Cf'])
    kleak = float(FRACTURE_CONFIG['kleak'])
    H_ext = float(FRACTURE_CONFIG['H_ext'])

    print(f"\n[Decay | {friction_key} | x1={x1}m | D={spacing_m}m | n={n_fracs}]")
    t0 = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * n_fracs,
        fracture_kleak=[kleak] * n_fracs,
        H_ext=H_ext,
        store_full_field=False,
    )
    sim_sec = time_module.time() - t0
    
    x_f_aligned = [float(res['x_grid'][i]) for i in res['fracture_indices']]
    print(f"  仿真 {sim_sec:.1f}s; 实际缝位: {[round(x, 1) for x in x_f_aligned]} m")

    t_sim = res['timestamps']
    H_wh = res['wellhead_head']
    v = cfg.a_adj
    fs = 1.0 / cfg.dt_adj
    ts = cfg.pump_shut_time
    L = cfg.wellbore_length

    # 保存原始仿真数据
    raw_dir = output_path(SERIES_DECAY_REGRESSION, 'raw_data', '')
    os.makedirs(raw_dir, exist_ok=True)
    npz_name = os.path.join(raw_dir, f"{friction_key}_x1_{int(x1)}_sp_{int(spacing_m)}_n_{n_fracs}.npz")
    np.savez_compressed(
        npz_name,
        t_sim=t_sim,
        H_wh=H_wh,
        x_f_aligned=x_f_aligned,
        Cf=np.array([Cf] * n_fracs),
        kleak=np.array([kleak] * n_fracs),
        v=np.array([v]),
        L=np.array([L]),
        fs=np.array([fs]),
        ts=np.array([ts])
    )

    out_1d = compute_moc_cepstrum_1d(t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L)
    depth_1d = out_1d['depth']
    resp_1d = out_1d['response']
    peaks_1d = _extract_local_peaks(depth_1d, resp_1d, x_f_aligned)

    out_2d = compute_moc_cepstrum(t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
                                  wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
    depth_2d = out_2d['depth']
    profile_2d = -np.mean(out_2d['C'], axis=1)
    peaks_2d = _extract_local_peaks(depth_2d, profile_2d, x_f_aligned)

    rows = []
    p1_1d = peaks_1d[0] if peaks_1d[0] > 0 else 1.0
    p1_2d = peaks_2d[0] if peaks_2d[0] > 0 else 1.0
    
    for i in range(n_fracs):
        row = {
            'friction_model': FRICTION_PARAMS[friction_key]['friction_model'],
            'x1': x1,
            'spacing_m': spacing_m,
            'n_total': n_fracs,
            'frac_idx': i + 1,
            'x_f': x_f_aligned[i],
            'delta_x': x_f_aligned[i] - x_f_aligned[0],
            'P_1d': peaks_1d[i],
            'P_2d': peaks_2d[i],
            'alpha_1d': peaks_1d[i] / p1_1d,
            'alpha_2d': peaks_2d[i] / p1_2d,
        }
        rows.append(row)
    
    return rows

CSV_COLUMNS = [
    'friction_model', 'x1', 'spacing_m', 'n_total', 'frac_idx',
    'x_f', 'delta_x', 'P_1d', 'P_2d', 'alpha_1d', 'alpha_2d'
]

def append_csv(rows: List[Dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in CSV_COLUMNS})

def main():
    X1_LIST = [1000.0, 2000.0, 3000.0, 4000.0]
    SPACING_LIST = [10.0, 20.0, 50.0, 100.0]
    N_FRACS_LIST = [2, 3, 4, 5]
    FRICTIONS = ['steady']  # 根据用户要求，目前先在 steady 上开展研究
    
    csv_path = output_path(SERIES_DECAY_REGRESSION, None, 'decay_table.csv')
    if os.path.isfile(csv_path):
        os.remove(csv_path)
        
    all_rows = []
    for fr in FRICTIONS:
        for x1 in X1_LIST:
            for sp in SPACING_LIST:
                for n in N_FRACS_LIST:
                    rows = run_one(fr, x1, sp, n)
                    append_csv(rows, csv_path)
                    all_rows.extend(rows)
    print(f"\nDone. Wrote {len(all_rows)} rows to {csv_path}")

if __name__ == '__main__':
    main()
