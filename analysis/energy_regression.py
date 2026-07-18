# -*- coding: utf-8 -*-
"""研究1：裂缝响应能量-参数回归模型 — 仿真与能量提取。

扫描矩阵
--------
- 主网格：friction ∈ {steady, brunone} × spacing ∈ SPACING_PRESETS_M × n_fracs ∈ {1..5}
- 次网格（--sweep-cfkleak）：在 spacing=50, n=3 参考工况扫 Cf × kleak

能量定义（倒谱域深度带内积分，避开 q≈0 源峰）
--------------------------------------------
- E_1d    = ∫_{d_lo}^{d_hi} response(d)^2 dd     （1D 实倒谱）
- E_2d    = ∫_{d_lo}^{d_hi} profile_2d(d)^2 dd   （2D cepstrogram 时间平均剖面）
- E_fft   = Σ_{f0..fmax_eff} |S(f)|^2             （频域归一化参照）
- E_1d_norm = E_1d / E_fft ; E_2d_norm = E_2d / E_fft

运行
----
    python analysis/energy_regression.py --friction steady --spacing 50 --cases dual
    python analysis/energy_regression.py --friction both --spacing all --cases all
    python analysis/energy_regression.py --friction steady --sweep-cfkleak
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time as time_module
from typing import Dict, List, Optional, Tuple

import numpy as np

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

from paths import (
    output_path,
    SERIES_ENERGY_REGRESSION,
    CASE_SINGLE, CASE_DUAL, CASE_TRIPLE, CASE_QUAD, CASE_QUINT,
)
from wellbore_moc import MocConfig, simulate_wellbore
from validation.config import (
    WELL_CONFIG, SIM_CONFIG, FRACTURE_CONFIG, CEPSTRUM_CONFIG,
    SPACING_PRESETS_M, FRAC_FIRST_M, build_cases, FRICTION_PARAMS,
)
from cepstrum_mocdata import (
    compute_moc_cepstrum_1d, compute_moc_cepstrum,
    evaluate_1d_cepstrum_fracture_match, effective_fft_fmax,
    preprocess_moc_head,
)
from validation.cepstrum import _kb_core as kb
from numpy.fft import fft, fftfreq


CASE_KEYS_ALL = [CASE_SINGLE, CASE_DUAL, CASE_TRIPLE, CASE_QUAD, CASE_QUINT]

# 次网格 Cf × kleak（在 spacing=50, n=3 参考工况上扫）
CF_GRID = [0.5e-5, 1.0e-5, 2.0e-5, 5.0e-5]
KLEAK_GRID = [0.5e-4, 1.0e-4, 5.0e-4, 1.0e-3]


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


def _run_simulation(
    friction_key: str,
    x_f_list: List[float],
    Cf: float,
    kleak: float,
) -> Dict:
    cfg = _build_moc_config(friction_key)
    H_ext = float(FRACTURE_CONFIG['H_ext'])
    res = simulate_wellbore(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * len(x_f_list),
        fracture_kleak=[kleak] * len(x_f_list),
        H_ext=H_ext,
        store_full_field=False,
    )
    return res


def _depth_band(x_f_aligned: List[float], margin_m: float) -> Tuple[float, float]:
    """缝群深度带 [d_lo, d_hi]，两侧各扩 margin 米。"""
    d_lo = max(0.0, min(x_f_aligned) - margin_m)
    d_hi = max(x_f_aligned) + margin_m
    return float(d_lo), float(d_hi)


def _band_integral(depth: np.ndarray, response: np.ndarray,
                   d_lo: float, d_hi: float) -> float:
    """带内平方积分 ∫ response^2 dd。"""
    mask = (depth >= d_lo) & (depth <= d_hi)
    if not np.any(mask):
        return 0.0
    d = depth[mask]
    r = response[mask]
    if len(d) < 2:
        return float(np.sum(r ** 2) * (d_hi - d_lo) / max(1, len(d)))
    return float(np.trapz(r ** 2, d))


def _fft_energy(h_detrended: np.ndarray, fs: float, v: float, L: float) -> Dict:
    """井口去均值谱的有效频带能量。"""
    n = len(h_detrended)
    if n < 4:
        return {'E_fft': 0.0, 'fmax_eff': 0.0}
    freqs = fftfreq(n, d=1.0 / fs)[:n // 2]
    mag = np.abs(fft(h_detrended)[:n // 2]) / n
    fmax_eff = effective_fft_fmax(freqs, mag, v, L)
    mask = (freqs > 0) & (freqs <= fmax_eff)
    E_fft = float(np.sum(mag[mask] ** 2))
    return {'E_fft': E_fft, 'fmax_eff': float(fmax_eff)}


def _extract_energies(
    res: Dict,
    x_f_aligned: List[float],
    *,
    wlen_sec: float = 30.0,
    hop_sec: float = 5.0,
    win_type: str = 'hamming',
) -> Dict:
    """对一次仿真结果提取 1D/2D/FFT 三类能量。"""
    cfg = res['cfg']
    v = cfg.a_adj
    L = cfg.wellbore_length
    fs = 1.0 / cfg.dt_adj
    ts = cfg.pump_shut_time

    t_sim = res['timestamps']
    H_wh = res['wellhead_head']

    # 1D 实倒谱
    out_1d = compute_moc_cepstrum_1d(
        t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
    )
    depth_1d = out_1d['depth']
    resp_1d = out_1d['response']
    match_1d = evaluate_1d_cepstrum_fracture_match(
        depth_1d, resp_1d, x_f_aligned, v=v, fs=fs,
    )
    E_1d_peaks = float(sum(
        m['peak_val'] ** 2 for m in match_1d['matches']
        if m.get('matched') and m.get('peak_val') is not None
    ))

    # 2D cepstrogram + 时间平均
    out_2d = compute_moc_cepstrum(
        t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
        wlen_sec=wlen_sec, hop_sec=hop_sec, win_type=win_type,
    )
    C2d = out_2d['C']  # (n_q_kept, n_t)
    q_kept = out_2d['q']
    depth_2d = out_2d['depth']
    profile_2d = -np.sum(C2d, axis=1)  # 沿时间轴累加 → 1D 深度剖面

    metrics_2d = kb.evaluate_multi_fracture_peaks(
        C2d, q_kept, v, x_f_aligned, fs=fs,
        depth_min=100.0, depth_max=L,
    )
    E_2d_peaks = float(sum(
        m['peak_val'] ** 2 for m in metrics_2d['matches']
        if m.get('matched') and m.get('peak_val') is not None
    ))

    # 频域参照
    pre = preprocess_moc_head(t_sim, H_wh, fs=fs, ts=ts)
    fft_info = _fft_energy(pre['h_detrended'], fs, v, L)

    # 缝群带积分
    margin = float(CEPSTRUM_CONFIG['fracture_zoom_margin_m'])
    d_lo, d_hi = _depth_band(x_f_aligned, margin)
    E_1d = _band_integral(depth_1d, resp_1d, d_lo, d_hi)
    E_2d = _band_integral(depth_2d, profile_2d, d_lo, d_hi)

    E_fft = fft_info['E_fft']
    return {
        'E_1d': E_1d,
        'E_2d': E_2d,
        'E_fft': E_fft,
        'E_1d_peaks': E_1d_peaks,
        'E_2d_peaks': E_2d_peaks,
        'E_1d_norm': E_1d / E_fft if E_fft > 0 else 0.0,
        'E_2d_norm': E_2d / E_fft if E_fft > 0 else 0.0,
        'fmax_eff': fft_info['fmax_eff'],
        'n_matched_1d': int(match_1d['n_matched']),
        'n_matched_2d': int(metrics_2d['n_matched']),
        'snr_1d': float(match_1d.get('snr') or 0.0) if match_1d.get('snr') else 0.0,
        'snr_2d': float(metrics_2d.get('snr') or 0.0) if metrics_2d.get('snr') else 0.0,
        'depth_band_m': [d_lo, d_hi],
        'cepstrum_params': {
            'wlen_sec': wlen_sec, 'hop_sec': hop_sec, 'win_type': win_type,
        },
    }


# ── 单工况执行 ─────────────────────────────────────────────
def run_one(
    friction_key: str,
    spacing_m: float,
    case_key: str,
    *,
    Cf: Optional[float] = None,
    kleak: Optional[float] = None,
    wlen_sec: float = 30.0,
    hop_sec: float = 5.0,
    win_type: str = 'hamming',
    save_json: bool = True,
) -> Dict:
    cases = build_cases(spacing_m)
    if case_key not in cases:
        raise ValueError(f"未知 case: {case_key}")
    cfg_case = cases[case_key]
    x_f_list = list(cfg_case['x_f_list'])
    Cf_used = float(Cf) if Cf is not None else float(FRACTURE_CONFIG['Cf'])
    kleak_used = float(kleak) if kleak is not None else float(FRACTURE_CONFIG['kleak'])

    fr_params = FRICTION_PARAMS[friction_key]
    print(f"\n[{friction_key}|D={spacing_m}m|{case_key}|n={len(x_f_list)}] "
          f"Cf={Cf_used:.2e}, kleak={kleak_used:.2e}")

    t0 = time_module.time()
    res = _run_simulation(friction_key, x_f_list, Cf_used, kleak_used)
    sim_sec = time_module.time() - t0
    x_f_aligned = [float(res['x_grid'][i]) for i in res['fracture_indices']]
    print(f"  仿真 {sim_sec:.1f}s; 缝对齐: {[round(x, 1) for x in x_f_aligned]} m")

    t0 = time_module.time()
    energies = _extract_energies(
        res, x_f_aligned,
        wlen_sec=wlen_sec, hop_sec=hop_sec, win_type=win_type,
    )
    print(f"  采能 {time_module.time() - t0:.1f}s; "
          f"E_1d={energies['E_1d']:.4e}, E_2d={energies['E_2d']:.4e}, "
          f"E_fft={energies['E_fft']:.4e}")

    row = {
        'friction': friction_key,
        'friction_model': fr_params['friction_model'],
        'spacing_m': float(spacing_m),
        'case': case_key,
        'n_fracs': len(x_f_list),
        'Cf': Cf_used,
        'kleak': kleak_used,
        'sim_seconds': sim_sec,
        **energies,
    }

    if save_json:
        if Cf is None and kleak is None:
            # 主网格：路径含 spacing，避免同名 case 跨 spacing 互相覆盖
            case_subdir = f"{case_key}_D{int(spacing_m)}"
        else:
            case_subdir = f"{case_key}_cf{Cf_used:.1e}_kl{kleak_used:.1e}"
        out_json = output_path(
            f"{SERIES_ENERGY_REGRESSION}/{friction_key}",
            case_subdir,
            'energy.json',
        )
        serializable = {k: v for k, v in row.items() if isinstance(v, (int, float, str, list, dict))}
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"  → {out_json}")
    return row


# ── CSV 汇总 ───────────────────────────────────────────────
CSV_COLUMNS = [
    'friction', 'friction_model', 'spacing_m', 'case', 'n_fracs',
    'Cf', 'kleak', 'sim_seconds',
    'E_1d', 'E_2d', 'E_fft', 'E_1d_peaks', 'E_2d_peaks',
    'E_1d_norm', 'E_2d_norm', 'fmax_eff',
    'n_matched_1d', 'n_matched_2d', 'snr_1d', 'snr_2d',
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


# ── CLI ────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='研究1：裂缝响应能量-参数回归仿真')
    p.add_argument('--friction', choices=['steady', 'brunone', 'both'], default='steady')
    p.add_argument('--spacing', default='all',
                   help='all 或间距[m]，如 50；默认 all=SPACING_PRESETS_M')
    p.add_argument('--cases', default='all',
                   help='all 或 single/dual/triple/quad/quint，逗号分隔')
    p.add_argument('--sweep-cfkleak', action='store_true',
                   help='次网格：在 spacing=50, n=3 上扫 Cf × kleak')
    p.add_argument('--wlen-sec', type=float, default=30.0)
    p.add_argument('--hop-sec', type=float, default=5.0)
    p.add_argument('--win-type', default='hamming',
                   choices=['rect', 'hamming', 'hanning', 'kaiser', 'gauss'])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frictions = ['steady', 'brunone'] if args.friction == 'both' else [args.friction]
    spacings = (list(SPACING_PRESETS_M) if args.spacing == 'all'
                else [float(args.spacing)])
    case_keys = (CASE_KEYS_ALL if args.cases == 'all'
                 else [c.strip() for c in args.cases.split(',')])

    rows: List[Dict] = []
    csv_path = output_path(SERIES_ENERGY_REGRESSION, None, 'energy_table.csv')

    if args.sweep_cfkleak:
        # 次网格：固定 spacing=50, case=triple
        for fr in frictions:
            for Cf in CF_GRID:
                for kleak in KLEAK_GRID:
                    row = run_one(
                        fr, 50.0, CASE_TRIPLE,
                        Cf=Cf, kleak=kleak,
                        wlen_sec=args.wlen_sec, hop_sec=args.hop_sec,
                        win_type=args.win_type,
                    )
                    rows.append(row)
    else:
        for fr in frictions:
            fr_key = fr  # 主网格用基础 friction 键（steady / brunone）
            for sp in spacings:
                # 主网格需要 friction_D* 键以与 build_cases(spacing) 一致；
                # 但 build_cases 只依赖 spacing_m，friction_key 仅用于 MocConfig。
                # 这里直接用基础 fr_key，spacing 通过 build_cases 参数传入。
                for ck in case_keys:
                    row = run_one(
                        fr_key, sp, ck,
                        wlen_sec=args.wlen_sec, hop_sec=args.hop_sec,
                        win_type=args.win_type,
                    )
                    rows.append(row)

    append_csv(rows, csv_path)
    print(f"\n汇总 CSV: {csv_path}  ({len(rows)} 行)")


if __name__ == '__main__':
    main()
