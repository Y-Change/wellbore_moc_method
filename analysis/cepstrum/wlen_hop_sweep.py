# -*- coding: utf-8 -*-
"""研究2：(wlen, hop) 二维网格扫描 — 评估 2D cepstrogram 窗长 × 滑动步长
对缝响应有效性（位置匹配误差）与分辨率（相邻缝区分能力）的联合影响。

维度：friction × case × wlen × hop_ratio
指标：
- 有效性：n_matched/n_fracs、mean_error_m、max_error_m
- 分辨率：峰间距误差 |Δd_det - Δd_true|、峰宽 FWHM、旁瓣抑制比
- SNR：peak/median

复用：_kb_core.optimized_2d_cepstrogram + evaluate_multi_fracture_peaks、
      wlen_sweep._build_method、_snr_of_profile。

输出：output/cepstrum/wlen_hop_sweep/{friction}/{case}/metrics.json

运行
----
    python validation/cepstrum/wlen_hop_sweep.py --friction steady --case dual \\
        --wlen 20,30,50 --hop 0.1,0.2
    python validation/cepstrum/wlen_hop_sweep.py --friction both --case dual,triple,quad
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks

# bootstrap wellbore_moc_method root
_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'paths.py')) and os.path.isfile(
        os.path.join(_d, 'wellbore_moc.py')
    ):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _d = os.path.dirname(_d)
    if _d == os.path.dirname(_d):
        raise RuntimeError('Cannot find wellbore_moc_method root')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import (
    output_path,
    SERIES_CEPSTRUM_WLEN_HOP,
    CASE_DUAL, CASE_TRIPLE, CASE_QUAD,
)
from cepstrum_mocdata import preprocess_moc_head
from wellbore_moc import MocConfig, simulate_wellbore
from validation.config import (
    WELL_CONFIG, SIM_CONFIG, FRACTURE_CONFIG,
    SPACING_PRESETS_M, FRAC_FIRST_M, build_cases, FRICTION_PARAMS,
)
from analysis.cepstrum import _kb_core as kb
from analysis.cepstrum.wlen_sweep import _build_method, _snr_of_profile, METHOD_PRESETS

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_WLEN_LIST = [15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
DEFAULT_HOP_RATIOS = [0.05, 0.1, 0.2, 0.25, 0.5]
DEFAULT_CASES = [CASE_DUAL, CASE_TRIPLE, CASE_QUAD]
DEFAULT_SPACING = 50.0
DEFAULT_METHOD = 'full'


# ── 分辨率指标 helper ─────────────────────────────────────
def _peak_fwhm(depth: np.ndarray, profile: np.ndarray, peak_idx: int) -> float:
    """单峰半高全宽 [m]（线性插值）。"""
    if len(profile) == 0 or peak_idx >= len(profile):
        return np.nan
    half = profile[peak_idx] * 0.5
    # 左
    li = peak_idx
    while li > 0 and profile[li] > half:
        li -= 1
    if li == 0 and profile[li] > half:
        d_left = float(depth[0])
    elif li < peak_idx:
        # 在 li 与 li+1 之间插值
        d1, d2 = float(depth[li]), float(depth[li + 1])
        p1, p2 = float(profile[li]), float(profile[li + 1])
        if p2 > p1:
            d_left = d1 + (half - p1) * (d2 - d1) / (p2 - p1)
        else:
            d_left = d1
    else:
        d_left = float(depth[peak_idx])
    # 右
    ri = peak_idx
    while ri < len(profile) - 1 and profile[ri] > half:
        ri += 1
    if ri == len(profile) - 1 and profile[ri] > half:
        d_right = float(depth[-1])
    elif ri > peak_idx:
        d1, d2 = float(depth[ri - 1]), float(depth[ri])
        p1, p2 = float(profile[ri - 1]), float(profile[ri])
        if p1 > p2:
            d_right = d2 - (p2 - half) * (d2 - d1) / (p1 - p2)
        else:
            d_right = d2
    else:
        d_right = float(depth[peak_idx])
    return abs(d_right - d_left)


def _sidelobe_suppression(
    depth: np.ndarray, profile: np.ndarray,
    main_depth: float, spacing_m: float,
) -> float:
    """主峰 / 主峰 ±0.5·spacing 外的最大值（旁瓣抑制比）。"""
    if len(profile) == 0:
        return np.nan
    excl_lo = main_depth - 0.5 * spacing_m
    excl_hi = main_depth + 0.5 * spacing_m
    mask = (depth < excl_lo) | (depth > excl_hi)
    if not np.any(mask):
        return np.nan
    side_max = float(np.max(profile[mask]))
    main_val = float(np.max(profile))
    return main_val / max(side_max, 1e-12)


def _spacing_error(
    depth: np.ndarray, profile: np.ndarray,
    true_spacing_m: float, peak_distance_bins: int,
) -> float:
    """检测到的相邻峰间距与真实间距的绝对误差 [m]。"""
    if len(profile) < 5:
        return np.nan
    height_thresh = max(float(np.percentile(profile, 88)), 1e-6)
    peaks, _ = find_peaks(profile, height=height_thresh, distance=peak_distance_bins)
    if len(peaks) < 2:
        return np.nan
    sorted_peaks = np.sort(depth[peaks])
    detected_spacing = float(np.median(np.diff(sorted_peaks)))
    return abs(detected_spacing - true_spacing_m)


# ── 单工况扫描 ────────────────────────────────────────────
def _build_moc_config(friction_key: str, spacing_m: float) -> MocConfig:
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


def sweep_one_case(
    friction_key: str,
    case_key: str,
    spacing_m: float,
    wlen_list: List[float],
    hop_ratios: List[float],
    method: str = DEFAULT_METHOD,
) -> Dict:
    cases = build_cases(spacing_m)
    cfg_case = cases[case_key]
    x_f_list = list(cfg_case['x_f_list'])
    fr_params = FRICTION_PARAMS[friction_key]
    Cf = FRACTURE_CONFIG['Cf']
    kleak = FRACTURE_CONFIG['kleak']
    H_ext = FRACTURE_CONFIG['H_ext']

    print("\n" + "=" * 72)
    print(f"(wlen, hop) 扫描 — {friction_key} | {case_key} | D={spacing_m}m | "
          f"n={len(x_f_list)} | method={method}")
    print("=" * 72)

    cfg = _build_moc_config(friction_key, spacing_m)
    t0 = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * len(x_f_list),
        fracture_kleak=[kleak] * len(x_f_list),
        H_ext=H_ext,
        store_full_field=False,
    )
    print(f"  仿真耗时: {time_module.time() - t0:.1f}s")

    t_sim = res['timestamps']
    H_wh = res['wellhead_head']
    x_f_aligned = [float(res['x_grid'][i]) for i in res['fracture_indices']]
    v = cfg.a_adj
    L = cfg.wellbore_length
    fs = 1.0 / cfg.dt_adj
    target_depth_m = float(np.mean(x_f_aligned))
    true_spacing = (x_f_aligned[1] - x_f_aligned[0]) if len(x_f_aligned) >= 2 else 0.0

    pre = preprocess_moc_head(t_sim, H_wh, fs=fs, ts=cfg.pump_shut_time)
    x_work = pre['h_detrended']
    print(f"  缝对齐: {[round(x, 1) for x in x_f_aligned]} m")

    wlen_min = 4.0 * L / v
    print(f"  wlen_min(4L/v) = {wlen_min:.3f}s")

    results: List[Dict] = []
    for wlen in wlen_list:
        if wlen * fs > len(x_work):
            print(f"  [skip] wlen={wlen}s 超过信号长度 {len(x_work)/fs:.1f}s")
            continue
        for hop_ratio in hop_ratios:
            t0 = time_module.time()
            t_ax, q_ax, C, note = _build_method(
                x_work, fs, wlen, hop_ratio, v, L, target_depth_m, method,
            )
            elapsed = time_module.time() - t0

            metrics = kb.evaluate_multi_fracture_peaks(
                C, q_ax, v, x_f_aligned, fs=fs, depth_min=100.0, depth_max=L,
            )
            depth_prof = metrics['profile_depth']
            profile = metrics['profile']
            snr = _snr_of_profile(profile)

            # 分辨率指标
            match_tol_m, peak_distance, _ = kb.peak_find_params(x_f_aligned, v, fs)
            spacing_err = _spacing_error(depth_prof, profile, true_spacing, peak_distance)
            # FWHM：取最高匹配峰
            fwhm = np.nan
            side_supp = np.nan
            if metrics['n_matched'] > 0:
                best = max(
                    (m for m in metrics['matches'] if m.get('matched')),
                    key=lambda m: m.get('peak_val', 0.0),
                    default=None,
                )
                if best is not None and best.get('peak_depth_m') is not None:
                    pd = float(best['peak_depth_m'])
                    idx = int(np.argmin(np.abs(depth_prof - pd)))
                    fwhm = _peak_fwhm(depth_prof, profile, idx)
                    side_supp = _sidelobe_suppression(
                        depth_prof, profile, pd, true_spacing or 300.0,
                    )

            results.append({
                'wlen_sec': wlen,
                'hop_ratio': hop_ratio,
                'hop_sec': hop_ratio * wlen,
                'note': note,
                'n_matched': int(metrics['n_matched']),
                'n_fracs': int(metrics['n_fracs']),
                'n_matched_ratio': float(metrics['n_matched'] / max(1, metrics['n_fracs'])),
                'mean_error_m': float(metrics['mean_error_m'])
                                 if np.isfinite(metrics['mean_error_m']) else None,
                'max_error_m': float(metrics['max_error_m'])
                               if np.isfinite(metrics['max_error_m']) else None,
                'snr': snr if np.isfinite(snr) else None,
                'spacing_error_m': spacing_err if np.isfinite(spacing_err) else None,
                'fwhm_m': fwhm if np.isfinite(fwhm) else None,
                'sidelobe_suppression': side_supp if np.isfinite(side_supp) else None,
                'elapsed_s': elapsed,
                'n_frames': int(C.shape[1]),
                'n_quefrency': int(C.shape[0]),
            })
            print(f"  wlen={wlen:5.1f}s hop={hop_ratio:.2f} | "
                  f"{metrics['n_matched']}/{metrics['n_fracs']} "
                  f"mean_err={metrics['mean_error_m']:.1f}m "
                  f"spacing_err={spacing_err:.1f}m "
                  f"FWHM={fwhm:.0f}m "
                  f"side={side_supp:.2f} SNR={snr:.2f} "
                  f"| {elapsed:.1f}s, {C.shape[1]}帧")

    out = {
        'friction': friction_key,
        'case': case_key,
        'spacing_m': spacing_m,
        'method': method,
        'x_f_aligned_m': x_f_aligned,
        'true_spacing_m': true_spacing,
        'wlen_min_s': wlen_min,
        'results': results,
    }

    series = f"{SERIES_CEPSTRUM_WLEN_HOP}/{friction_key}"
    out_json = output_path(series, case_key, 'metrics.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"  → {out_json}")
    return out


# ── CLI ──────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='研究2：(wlen, hop) 二维网格扫描')
    p.add_argument('--friction', choices=['steady', 'brunone', 'both'], default='steady')
    p.add_argument('--case', default='dual',
                   help='dual/triple/quad，逗号分隔；或 all=DEFAULT_CASES')
    p.add_argument('--spacing', type=float, default=DEFAULT_SPACING)
    p.add_argument('--wlen', default=None,
                   help='逗号分隔窗长列表[s]；缺省=DEFAULT_WLEN_LIST')
    p.add_argument('--hop', default=None,
                   help='逗号分隔 hop_ratio 列表；缺省=DEFAULT_HOP_RATIOS')
    p.add_argument('--method', default=DEFAULT_METHOD,
                   choices=list(METHOD_PRESETS.keys()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frictions = ['steady', 'brunone'] if args.friction == 'both' else [args.friction]
    case_keys = (DEFAULT_CASES if args.case == 'all'
                 else [c.strip() for c in args.case.split(',')])
    wlen_list = (DEFAULT_WLEN_LIST if args.wlen is None
                 else [float(x) for x in args.wlen.split(',')])
    hop_ratios = (DEFAULT_HOP_RATIOS if args.hop is None
                  else [float(x) for x in args.hop.split(',')])

    for fr in frictions:
        for ck in case_keys:
            sweep_one_case(fr, ck, args.spacing, wlen_list, hop_ratios, args.method)


if __name__ == '__main__':
    main()
