# -*- coding: utf-8 -*-
"""研究 4：不同的Cf和Kleak对于裂缝响应的影响 (decay_regression 框架下)"""
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

def _build_moc_config(friction_key: str, L_required: float = 5000.0) -> MocConfig:
    fr_params = FRICTION_PARAMS[friction_key]
    w, s = WELL_CONFIG, SIM_CONFIG
    
    L = max(w['L'], L_required)
    tf = max(s['tf'], 2.0 * L / w['wavespeed'] + 1.0)
    
    return MocConfig(
        wellbore_length=L,
        wellbore_diameter=w['wellbore_diameter'],
        fluid_density=w['fluid_density'],
        fluid_viscosity=w['fluid_viscosity'],
        wavespeed=w['wavespeed'],
        roughness_height=w['roughness_height'],
        friction_model=fr_params['friction_model'],
        dt=s['dt'], tf=tf,
        wellhead_bc='velocity_step', pump_shut_time=s['ts'],
        initial_velocity=w['V0'], initial_head=w['H0'],
        theta=w['theta'], toe_bc='reservoir', toe_head=w['H0'],
    )

def _extract_local_peaks(depth: np.ndarray, response: np.ndarray, x_f_list: List[float], search_radius: float = 15.0) -> List[float]:
    peaks = []
    if len(x_f_list) > 1:
        min_spacing = min(abs(x_f_list[i] - x_f_list[i-1]) for i in range(1, len(x_f_list)))
        search_radius = min(search_radius, min_spacing * 0.49)
        
    for x_f in x_f_list:
        mask = (depth >= x_f - search_radius) & (depth <= x_f + search_radius)
        if np.any(mask):
            peaks.append(float(np.max(response[mask])))
        else:
            peaks.append(0.0)
    return peaks

def run_one(friction_key: str, Cf: float, kleak: float) -> List[Dict]:
    x1 = 3000.0
    spacing_m = 20.0
    n_fracs = 5
    x_f_list = [x1 + i * spacing_m for i in range(n_fracs)]
    L_required = max(x_f_list) + 500.0
    cfg = _build_moc_config(friction_key, L_required)
    
    H_ext = float(FRACTURE_CONFIG['H_ext'])

    print(f"\n[Decay Cf/Kleak | {friction_key} | Cf={Cf:.1e} | Kleak={kleak:.1e}]")
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
    raw_dir = output_path(SERIES_DECAY_REGRESSION, 'cf_kleak_study', 'waves')
    os.makedirs(raw_dir, exist_ok=True)
    npz_name = os.path.join(raw_dir, f"{friction_key}_cf_{Cf:.1e}_kleak_{kleak:.1e}.npz")
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
    profile_2d = -np.sum(out_2d['C'], axis=1)
    peaks_2d = _extract_local_peaks(depth_2d, profile_2d, x_f_aligned)

    rows = []
    p1_1d = peaks_1d[0] if peaks_1d[0] > 0 else 1.0
    p1_2d = peaks_2d[0] if peaks_2d[0] > 0 else 1.0
    
    for i in range(n_fracs):
        row = {
            'friction_model': FRICTION_PARAMS[friction_key]['friction_model'],
            'Cf': Cf,
            'Kleak': kleak,
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
    'friction_model', 'Cf', 'Kleak', 'x1', 'spacing_m', 'n_total', 'frac_idx',
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

def load_completed_cases(csv_path: str) -> set:
    completed = set()
    if not os.path.isfile(csv_path):
        return completed
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            case = (r['friction_model'], float(r['Cf']), float(r['Kleak']))
            completed.add(case)
    return completed

def main():
    CF_LIST = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    KLEAK_LIST = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    FRICTIONS = ['steady', 'brunone']
    
    csv_path = output_path(SERIES_DECAY_REGRESSION, 'cf_kleak_study', 'cf_kleak_table.csv')
    completed_cases = load_completed_cases(csv_path)
        
    all_rows = []
    for fr in FRICTIONS:
        friction_model_name = FRICTION_PARAMS[fr]['friction_model']
        for cf in CF_LIST:
            for kleak in KLEAK_LIST:
                if (friction_model_name, cf, kleak) in completed_cases:
                    print(f"Skipping {fr} Cf={cf:.1e} Kleak={kleak:.1e} (already exists)")
                    continue
                    
                rows = run_one(fr, cf, kleak)
                append_csv(rows, csv_path)
                all_rows.extend(rows)
    print(f"\nDone. Wrote {len(all_rows)} rows to {csv_path}")

if __name__ == '__main__':
    main()
