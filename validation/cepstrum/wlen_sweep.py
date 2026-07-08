# -*- coding: utf-8 -*-
"""
窗长 (wlen_sec) 对 2D 倒谱裂缝识别影响的扫描分析。

支持 steady 与 brunone 两种摩阻模型，通过 --friction 切换。
对每个 case（单/双/三/四/五缝）遍历一组 wlen_sec，使用同一倒谱方法
（默认：动态 Kaiser + eps + 预加重 + Lifter，即 _kb_core 的"全复合"方案）
计算 2D cepstrogram，再用时间平均深度剖面 + 峰匹配评估裂缝识别质量。
输出路径: output/cepstrum/wlen_sweep/{friction}/{case}/

运行
----
    python validation/cepstrum/wlen_sweep.py
    python validation/cepstrum/wlen_sweep.py --case quad
    python validation/cepstrum/wlen_sweep.py --friction brunone --case dual
    python validation/cepstrum/wlen_sweep.py --case all --method full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from typing import Dict, List, Optional, Tuple

# bootstrap wellbore_moc_method root
_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'paths.py')) and os.path.isfile(
        os.path.join(_d, 'wellbore_moc.py')
    ):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find wellbore_moc_method root')
    _d = _parent

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from paths import (
    output_path,
    SERIES_CEPSTRUM_KB,
    SERIES_CEPSTRUM_WLEN_SWEEP,
    CASE_DUAL,
    CASE_TRIPLE,
    CASE_QUAD,
    CASE_QUINT,
)
from cepstrum_mocdata import preprocess_moc_head
from wellbore_moc import MocConfig, simulate_wellbore
from validation.cepstrum import _kb_core as kb
from validation.cepstrum.kaiser_bessel_multi import CASES, FRICTION_PARAMS

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出子系列
SERIES_WLEN_SWEEP = SERIES_CEPSTRUM_WLEN_SWEEP

# 默认扫描窗长（秒）——覆盖从 wlen_min(~13.8s) 到接近全长
DEFAULT_WLEN_LIST = [15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

# 方法名 → _kb_core 构建函数的开关组合
METHOD_PRESETS = {
    'baseline_hamming': dict(use_dynamic_kaiser=False, use_epsilon_log=False,
                             use_preemphasis=False, use_lifter=False),
    'baseline_kaiser4': dict(use_dynamic_kaiser=False, use_epsilon_log=False,
                             use_preemphasis=False, use_lifter=False),
    'dynamic_kaiser':   dict(use_dynamic_kaiser=True,  use_epsilon_log=False,
                             use_preemphasis=False, use_lifter=False),
    'eps_preemph':      dict(use_dynamic_kaiser=True,  use_epsilon_log=True,
                             use_preemphasis=True,  use_lifter=False),
    'full':             dict(use_dynamic_kaiser=True,  use_epsilon_log=True,
                             use_preemphasis=True,  use_lifter=True),
}


def _build_method(
    x_work: np.ndarray,
    fs: float,
    wlen_sec: float,
    hop_ratio: float,
    v: float,
    L: float,
    target_depth_m: float,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """按 method preset 调用 _kb_core 的 2D 倒谱。"""
    if method in ('baseline_hamming', 'baseline_kaiser4'):
        win_type = 'hamming' if method == 'baseline_hamming' else 'kaiser'
        t, q, C = kb.baseline_2d_cepstrogram(
            x_work, fs, wlen_sec, hop_ratio=hop_ratio,
            win_type=win_type, kaiser_beta=4.0,
            wavespeed=v, wellbore_length=L,
        )
        return t, q, C, f'基线 {win_type}'
    preset = METHOD_PRESETS[method]
    t, q, C = kb.optimized_2d_cepstrogram(
        x_work, fs, wlen_sec,
        wavespeed=v, wellbore_length=L,
        hop_ratio=hop_ratio,
        target_depth_m=target_depth_m,
        **preset,
    )
    beta = kb.dynamic_kaiser_beta(target_depth_m, fs, v)
    tags = []
    if preset['use_dynamic_kaiser']:
        tags.append(f'β={beta:.1f}')
    if preset['use_epsilon_log']:
        tags.append('eps')
    if preset['use_preemphasis']:
        tags.append('preemph')
    if preset['use_lifter']:
        tags.append('lifter')
    return t, q, C, '+'.join(tags) if tags else 'Kaiser'


def _snr_of_profile(profile: np.ndarray) -> float:
    if len(profile) == 0:
        return float('nan')
    bg = float(np.percentile(profile, 50))
    peak = float(np.max(profile))
    return peak / max(abs(bg), 1e-12)


def _sweep_one_case(
    case_key: str,
    wlen_list: List[float],
    method: str,
    hop_ratio: float,
    friction: str = 'steady',
    save_grid: bool = True,
    save_overlay: bool = True,
) -> Dict:
    cfg_case = CASES[case_key]
    label = cfg_case['label']
    x_f_list = cfg_case['x_f_list']
    fr_params = FRICTION_PARAMS[friction]

    L = 5000.0
    a = 1450.0
    V0 = 1.0
    H0 = 300.0
    ts = 1.0
    dt = 1.0e-3
    tf = 100.0
    Cf = 1.0e-5
    kleak = fr_params['kleak']
    H_ext = fr_params['H_ext']
    fs = 1.0 / dt

    print("\n" + "=" * 72)
    print(f"窗长扫描 — {label} (x_f={x_f_list}m, method={method}, friction={friction})")
    print("=" * 72)

    cfg = MocConfig(
        wellbore_length=L, wellbore_diameter=0.1397,
        fluid_density=1000.0, fluid_viscosity=1.0e-6,
        wavespeed=a, roughness_height=4.5e-5,
        friction_model=fr_params['friction_model'], dt=dt, tf=tf,
        wellhead_bc='velocity_step', pump_shut_time=ts,
        initial_velocity=V0, initial_head=H0,
        theta=0.0, toe_bc='reservoir', toe_head=H0,
    )

    print(f"运行 MOC 仿真 ({fr_params['label']})...")
    t0 = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * len(x_f_list),
        fracture_kleak=[kleak] * len(x_f_list),
        H_ext=H_ext,
        store_full_field=False,
    )
    print(f"  耗时: {time_module.time() - t0:.1f}s")

    t_sim = res['timestamps']
    H_wh = res['wellhead_head']
    x_f_aligned = [float(res['x_grid'][i]) for i in res['fracture_indices']]
    v = cfg.a_adj
    target_depth_m = float(np.mean(x_f_aligned))

    pre = preprocess_moc_head(t_sim, H_wh, fs=fs, ts=ts)
    x_work = pre['h_detrended']
    print(f"  缝对齐: {[round(x, 1) for x in x_f_aligned]} m")

    wlen_min = 4.0 * L / v
    print(f"  wlen_min(4L/v) = {wlen_min:.3f}s, 扫描 {len(wlen_list)} 个窗长")

    sweep_results: List[Dict] = []
    profiles: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    grids: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

    for wlen in wlen_list:
        if wlen * fs > len(x_work):
            print(f"  [skip] wlen={wlen}s 超过信号长度 {len(x_work)/fs:.1f}s")
            continue
        t0 = time_module.time()
        t_ax, q_ax, C, note = _build_method(
            x_work, fs, wlen, hop_ratio, v, L, target_depth_m, method,
        )
        elapsed = time_module.time() - t0

        metrics = kb.evaluate_multi_fracture_peaks(
            C, q_ax, v, x_f_aligned, fs=fs, depth_min=100.0, depth_max=L,
        )
        depth_kept = metrics['profile_depth']
        profile = metrics['profile']
        snr = _snr_of_profile(profile)
        profiles[wlen] = (depth_kept, profile)
        if save_grid:
            grids.append((wlen, t_ax, kb.depth_axis(q_ax, v), C))

        sweep_results.append({
            'wlen_sec': float(wlen),
            'note': note,
            'n_matched': int(metrics['n_matched']),
            'n_fracs': int(metrics['n_fracs']),
            'mean_error_m': float(metrics['mean_error_m']) if np.isfinite(metrics['mean_error_m']) else None,
            'max_error_m': float(metrics['max_error_m']) if np.isfinite(metrics['max_error_m']) else None,
            'snr': float(snr) if np.isfinite(snr) else None,
            'matches': metrics['matches'],
            'elapsed_s': float(elapsed),
            'n_frames': int(C.shape[1]),
            'n_quefrency': int(C.shape[0]),
        })

        status = (
            f"  wlen={wlen:5.1f}s | {metrics['n_matched']}/{metrics['n_fracs']} "
            f"mean_err={metrics['mean_error_m']:.1f}m "
            f"max_err={metrics['max_error_m']:.1f}m SNR={snr:.2f} "
            f"| {note} ({elapsed:.1f}s, {C.shape[1]}帧)"
        )
        print(status)

    if not sweep_results:
        print("  [警告] 无有效 wlen 结果")
        return {'case': case_key, 'wlen_list': [], 'results': []}

    # ── 图1: 指标随 wlen 变化 ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    wlens = [r['wlen_sec'] for r in sweep_results]
    n_match = [r['n_matched'] for r in sweep_results]
    n_frac = sweep_results[0]['n_fracs']
    mean_err = [r['mean_error_m'] if r['mean_error_m'] is not None else np.nan for r in sweep_results]
    max_err = [r['max_error_m'] if r['max_error_m'] is not None else np.nan for r in sweep_results]
    snrs = [r['snr'] if r['snr'] is not None else np.nan for r in sweep_results]
    elapsed = [r['elapsed_s'] for r in sweep_results]

    ax = axes[0, 0]
    ax.plot(wlens, n_match, 'bo-', lw=1.5, markersize=8)
    ax.axhline(n_frac, color='r', ls='--', lw=1.0, label=f'真实缝数 {n_frac}')
    ax.set_xlabel('wlen_sec [s]')
    ax.set_ylabel('匹配数')
    ax.set_title('裂缝匹配数 vs 窗长')
    ax.set_ylim([-0.3, n_frac + 0.5])
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(wlens, mean_err, 'g^-', lw=1.5, markersize=7, label='均误差')
    ax.plot(wlens, max_err, 'rs--', lw=1.2, markersize=6, label='最大误差')
    ax.set_xlabel('wlen_sec [s]')
    ax.set_ylabel('深度误差 [m]')
    ax.set_title('定位误差 vs 窗长')
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(wlens, snrs, 'mD-', lw=1.5, markersize=7)
    ax.set_xlabel('wlen_sec [s]')
    ax.set_ylabel('SNR (peak/median)')
    ax.set_title('时间平均剖面 SNR vs 窗长')
    ax.set_yscale('log')
    ax.grid(True, ls='--', alpha=0.6, which='both')

    ax = axes[1, 1]
    ax.plot(wlens, elapsed, 'c>-', lw=1.5, markersize=7)
    ax.set_xlabel('wlen_sec [s]')
    ax.set_ylabel('计算耗时 [s]')
    ax.set_title('2D 倒谱计算耗时 vs 窗长')
    ax.grid(True, ls='--', alpha=0.6)

    series = f"{SERIES_WLEN_SWEEP}/{friction}"
    fr_label = fr_params['label']

    fig.suptitle(
        f'窗长扫描 — {label} (x_f={[round(x) for x in x_f_aligned]}m, '
        f'method={method}, {fr_label})',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path_metrics = output_path(series, case_key, 'metrics_vs_wlen.png')
    fig.savefig(path_metrics, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  指标图: {path_metrics}")

    # ── 图2: 深度剖面对比叠加 ─────────────────────────────
    if save_overlay and profiles:
        fig2, axp = plt.subplots(figsize=(14, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.95, len(profiles)))
        for (wlen, (depth_kept, profile)), color in zip(profiles.items(), colors):
            axp.plot(depth_kept, profile, lw=1.1, color=color,
                     label=f'wlen={wlen:.0f}s')
        for fd in x_f_aligned:
            axp.axvline(fd, color='k', ls=':', lw=1.0, alpha=0.6)
        axp.set_xlabel('深度 [m]')
        axp.set_ylabel('时间平均倒谱响应 (-C)')
        axp.set_title(f'{label} — 不同窗长时间平均深度剖面 (method={method}, {fr_label})')
        axp.set_xlim([0, L])
        axp.legend(fontsize=8, loc='upper right', ncol=2)
        axp.grid(True, ls='--', alpha=0.4)
        plt.tight_layout()
        path_prof = output_path(series, case_key, 'profile_overlay.png')
        fig2.savefig(path_prof, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  剖面叠加: {path_prof}")

    # ── 图3: 2D 热力图网格 ────────────────────────────────
    if save_grid and grids:
        n = len(grids)
        n_cols = min(3, n)
        n_rows = int(np.ceil(n / n_cols))
        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows),
                                   squeeze=False)
        fig3.suptitle(
            f'{label} — 不同 wlen 2D 倒谱热力图 (method={method}, {fr_label})',
            fontsize=13, fontweight='bold',
        )
        shared_vmin, shared_vmax = kb.compute_shared_vrange(
            [(depth, C) for _, _, depth, C in grids], L,
        )
        for idx, (wlen, t_ax, depth, C) in enumerate(grids):
            r, c = divmod(idx, n_cols)
            ax = axes3[r][c]
            kb.plot_2d_panel_multi(
                ax, t_ax, depth, C, f'wlen={wlen:.0f}s',
                x_f_aligned, L, mark_t_max=10.0,
                vmin=shared_vmin, vmax=shared_vmax,
            )
        for idx in range(n, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes3[r][c].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path_grid = output_path(series, case_key, 'cepstrogram_grid.png')
        fig3.savefig(path_grid, dpi=130, bbox_inches='tight')
        plt.close(fig3)
        print(f"  2D网格: {path_grid}")

    # ── 最优 wlen 选择 ────────────────────────────────────
    fully_matched = [r for r in sweep_results if r['n_matched'] == r['n_fracs']]
    if fully_matched:
        best = min(fully_matched, key=lambda r: (r['mean_error_m'] or 1e9))
        print(
            f"\n  最优 wlen = {best['wlen_sec']:.1f}s  "
            f"({best['n_matched']}/{best['n_fracs']}匹配, "
            f"mean_err={best['mean_error_m']:.1f}m, SNR={best['snr']:.2f})"
        )
    else:
        best = max(sweep_results, key=lambda r: r['n_matched'])
        print(
            f"\n  部分最优 wlen = {best['wlen_sec']:.1f}s  "
            f"({best['n_matched']}/{best['n_fracs']}匹配)"
        )

    summary = {
        'case': case_key,
        'label': label,
        'method': method,
        'friction': friction,
        'x_f_nominal': x_f_list,
        'x_f_aligned': x_f_aligned,
        'wlen_min_sec': float(wlen_min),
        'wlen_list': [r['wlen_sec'] for r in sweep_results],
        'best_wlen_sec': best['wlen_sec'],
        'best_n_matched': best['n_matched'],
        'best_mean_error_m': best['mean_error_m'],
        'results': sweep_results,
    }
    json_path = output_path(series, case_key, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='窗长 (wlen_sec) 对裂缝识别影响扫描（支持 steady/brunone）')
    parser.add_argument(
        '--case', choices=['single', 'dual', 'triple', 'quad', 'quint', 'all'],
        default='all', help='运行哪个 case（默认 all）',
    )
    parser.add_argument(
        '--friction',
        choices=['steady', 'brunone'],
        default='steady',
        help='摩阻模型（默认 steady；brunone 仿真约慢 20×）',
    )
    parser.add_argument(
        '--method', choices=list(METHOD_PRESETS.keys()) + ['baseline_hamming', 'baseline_kaiser4'],
        default='full', help='倒谱方法 preset（默认 full：动态Kaiser+eps+预加重+Lifter）',
    )
    parser.add_argument(
        '--wlen', type=str, default=None,
        help='自定义窗长列表，逗号分隔，如 "14,20,30,45,60"',
    )
    parser.add_argument(
        '--hop-ratio', type=float, default=0.2,
        help='hop = hop_ratio * wlen（默认 0.2）',
    )
    parser.add_argument(
        '--no-grid', action='store_true',
        help='不保存 2D 热力图网格（大规模扫描时省时间）',
    )
    args = parser.parse_args()

    wlen_list = DEFAULT_WLEN_LIST
    if args.wlen:
        wlen_list = [float(x) for x in args.wlen.split(',')]

    keys = list(CASES.keys()) if args.case == 'all' else [args.case]
    all_results = {}
    for key in keys:
        all_results[key] = _sweep_one_case(
            key, wlen_list, args.method, args.hop_ratio,
            friction=args.friction,
            save_grid=not args.no_grid,
        )

    print("\n" + "=" * 72)
    print(f"窗长扫描汇总 ({FRICTION_PARAMS[args.friction]['label']})")
    print("=" * 72)
    for key, res in all_results.items():
        if not res['results']:
            continue
        best_wlen = res.get('best_wlen_sec')
        best_n = res.get('best_n_matched', 0)
        best_err = res.get('best_mean_error_m')
        err_str = f"{best_err:.1f}m" if best_err is not None else "N/A"
        print(
            f"  {res['label']:>4s}: 最优 wlen={best_wlen:.1f}s  "
            f"匹配 {best_n}/{res['results'][0]['n_fracs']}  均误差={err_str}  "
            f"(method={res['method']})"
        )

    return all_results


if __name__ == '__main__':
    main()
