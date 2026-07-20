# -*- coding: utf-8 -*-
"""
多缝 — Kaiser-Bessel 2D 倒谱方案对比（单/双/三/四/五缝）。

支持 steady 与 brunone 两种摩阻模型，通过 --friction 切换。
输出路径: output/cepstrum/kaiser_bessel/{friction}/{case}/

运行:
    python validation/cepstrum/kaiser_bessel_multi.py
    python validation/cepstrum/kaiser_bessel_multi.py --case dual
    python validation/cepstrum/kaiser_bessel_multi.py --friction brunone --case quad
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from typing import Dict, List, Optional, Tuple

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'paths.py')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find wellbore_moc_method root')
    _d = _parent

import matplotlib.pyplot as plt
import numpy as np

from paths import (
    output_path,
    SERIES_CEPSTRUM_KB,
    CASE_SINGLE,
    CASE_DUAL,
    CASE_TRIPLE,
    CASE_QUAD,
    CASE_QUINT,
)

from cepstrum_mocdata import preprocess_moc_head
from wellbore_moc import MocConfig, simulate_wellbore
from analysis.cepstrum import _kb_core as kb

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COMPARE_2D_FRAC_MARK_T_MAX = 5.0  # compare_2d.png 裂缝标注：前 10s 黑色实线

# 缝形态配置（单/双/三/四/五缝）
CASES = {
    CASE_SINGLE: {
        'label': '单缝',
        'x_f_list': [4300.0],
    },
    CASE_DUAL: {
        'label': '双缝',
        'x_f_list': [4300.0, 4600.0],
    },
    CASE_TRIPLE: {
        'label': '三缝',
        'x_f_list': [4100.0, 4300.0, 4500.0],
    },
    CASE_QUAD: {
        'label': '四缝',
        'x_f_list': [3900.0, 4100.0, 4300.0, 4500.0],
    },
    CASE_QUINT: {
        'label': '五缝',
        'x_f_list': [3700.0, 3900.0, 4100.0, 4300.0, 4500.0],
    },
}

# 摩阻模型配置
FRICTION_PARAMS = {
    'steady': {'friction_model': 'steady', 'kleak': 0.0001, 'H_ext': 100.0,
               'label': '稳态达西+滤失'},
    'brunone': {'friction_model': 'brunone', 'kleak': 0.0001, 'H_ext': 100.0,
                'label': 'Brunone非定常+滤失'},
}


def _print_metrics_table(case_label: str, metrics: Dict, true_depths: List[float]) -> None:
    print(f"\n{'=' * 88}")
    print(f"{case_label} — 倒谱缝深匹配")
    print(f"{'=' * 88}")
    print(f"{'方法':<22} {'匹配':>5} {'均误差':>8} {'最大误差':>10}  检测详情")
    print("-" * 88)
    for name, m in metrics.items():
        detail = ' '.join(
            f"F{mt['frac_id']}:{mt['peak_depth_m']:.0f}m({mt['error_m']:.0f})"
            if mt['matched'] else f"F{mt['frac_id']}:×"
            for mt in m['matches']
        )
        print(
            f"{name:<22} {m['n_matched']}/{m['n_fracs']:>3} "
            f"{m['mean_error_m']:8.1f} {m['max_error_m']:10.1f}  {detail}"
        )
    print(f"真实缝深: {[round(d, 1) for d in true_depths]} m")


def _subplot_grid(n_panels: int, figsize=(20, 11)):
    """返回 (fig, axes_flat)，多余子图已隐藏。"""
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flat
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes_flat


def run_case(case_key: str, friction: str = 'steady') -> Dict:
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
    wlen_sec = 30.0
    hop_ratio = 0.5

    print("\n" + "=" * 72)
    print(f"测试 {label} — 2D 倒谱优化方案对比（Kaiser-Bessel）")
    print(f"x_f={x_f_list}m, 摩阻={fr_params['label']}")
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

    print(f"\n运行 MOC {label}仿真 ({friction})...")
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

    methods = kb.build_all_methods(x_work, fs, wlen_sec, hop_ratio, v, L, target_depth_m)
    metrics = {}
    for name, (t_ax, q_ax, C, note) in methods.items():
        metrics[name] = kb.evaluate_multi_fracture_peaks(
            C, q_ax, v, x_f_aligned, fs=fs, depth_min=100.0, depth_max=L,
        )

    _print_metrics_table(label, metrics, x_f_aligned)

    all_panels = [(kb.depth_axis(q_ax, v), C) for _, (_, q_ax, C, _) in methods.items()]
    shared_vmin, shared_vmax = kb.compute_shared_vrange(all_panels, L)

    series = f"{SERIES_CEPSTRUM_KB}/{friction}"
    n_methods = len(methods)

    fig, axes_flat = _subplot_grid(n_methods, figsize=(20, 11))
    frac_label = '/'.join(f'{x:.0f}' for x in x_f_aligned)
    fig.suptitle(
        f'测试 {label} — 2D 倒谱方案对比\n'
        f'x_f=[{frac_label}]m, wlen={wlen_sec}s, {fr_params["label"]}',
        fontsize=13, fontweight='bold',
    )
    for ax, (name, (t_ax, q_ax, C, note)) in zip(axes_flat, methods.items()):
        depth = kb.depth_axis(q_ax, v)
        kb.plot_2d_panel_multi(
            ax, t_ax, depth, C, f'{name}\n{note}', x_f_aligned, L,
            mark_t_max=COMPARE_2D_FRAC_MARK_T_MAX,
            vmin=shared_vmin, vmax=shared_vmax,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path_2d = output_path(series, case_key, 'compare_2d.png')
    fig.savefig(path_2d, dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig2, axp = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for (name, m), color in zip(metrics.items(), colors):
        axp.plot(m['profile_depth'], m['profile'], lw=1.1, label=name, color=color)
        for mt in m['matches']:
            if mt['matched']:
                axp.scatter(mt['peak_depth_m'], mt['peak_val'], color=color, s=35, marker='v', zorder=5)
    for fd in x_f_aligned:
        axp.axvline(fd, color='k', ls=':', lw=1.0, alpha=0.5)
    axp.set_xlabel('深度 [m]')
    axp.set_ylabel('时间平均倒谱响应 (-C)')
    axp.set_title(f'{label} — 1D 深度剖面叠加 ({fr_params["label"]})')
    axp.set_xlim([0, L])
    axp.legend(fontsize=6, loc='upper right', ncol=2)
    axp.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    path_prof = output_path(series, case_key, 'profile_overlay.png')
    fig2.savefig(path_prof, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3, axes3_flat = _subplot_grid(n_methods, figsize=(18, 8))
    fig3.suptitle(f'测试 {label} — 时间平均 1D 深度剖面 ({friction})', fontsize=13, fontweight='bold')
    for ax, (name, m) in zip(axes3_flat, metrics.items()):
        kb.plot_1d_profile_multi(ax, m['profile_depth'], m['profile'], name, x_f_aligned, m['matches'])
        ax.set_xlim([0, L])
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path_1d = output_path(series, case_key, 'profile_grid.png')
    fig3.savefig(path_1d, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    print(f"  2D: {path_2d}")
    print(f"  1D叠加: {path_prof}")
    print(f"  1D分图: {path_1d}")

    valid = [(n, m) for n, m in metrics.items() if m['n_matched'] == m['n_fracs']]
    if valid:
        best_name, best_m = min(valid, key=lambda kv: kv[1]['mean_error_m'])
        print(
            f"  最优: {best_name}  ({best_m['n_matched']}/{best_m['n_fracs']}匹配, "
            f"mean_err={best_m['mean_error_m']:.1f}m)"
        )
    else:
        print("  [注意] 无方法匹配全部裂缝")

    summary = {
        'case': case_key,
        'label': label,
        'friction': friction,
        'x_f_nominal': x_f_list,
        'x_f_aligned': x_f_aligned,
        'methods': {
            name: {
                'n_matched': m['n_matched'],
                'n_fracs': m['n_fracs'],
                'mean_error_m': m['mean_error_m'],
                'max_error_m': m['max_error_m'],
                'matches': m['matches'],
            }
            for name, m in metrics.items()
        },
        'figures': [path_2d, path_prof, path_1d],
    }
    json_path = output_path(series, case_key, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='多缝 Kaiser-Bessel 倒谱验证（支持 steady/brunone）')
    parser.add_argument(
        '--case',
        choices=['single', 'dual', 'triple', 'quad', 'quint', 'all'],
        default='all',
        help='运行单/双/三/四/五缝或全部（默认 all）',
    )
    parser.add_argument(
        '--friction',
        choices=['steady', 'brunone'],
        default='steady',
        help='摩阻模型（默认 steady；brunone 仿真约慢 20×）',
    )
    args = parser.parse_args()

    keys = list(CASES.keys()) if args.case == 'all' else [args.case]
    results = {}
    for key in keys:
        results[key] = run_case(key, friction=args.friction)

    print("\n" + "=" * 72)
    print(f"汇总 ({FRICTION_PARAMS[args.friction]['label']})")
    print("=" * 72)
    for key, res in results.items():
        best = None
        for name, m in res['methods'].items():
            if m['n_matched'] == m['n_fracs']:
                if best is None or m['mean_error_m'] < best[1]:
                    best = (name, m['mean_error_m'], m['n_fracs'])
        if best:
            print(f"  {CASES[key]['label']}: 最佳 {best[0]}  "
                  f"{best[2]}/{best[2]}匹配 mean_err={best[1]:.1f}m")
        else:
            print(f"  {CASES[key]['label']}: 无全匹配方法")

    return results


if __name__ == '__main__':
    main()
