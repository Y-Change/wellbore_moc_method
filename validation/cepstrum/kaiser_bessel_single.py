# -*- coding: utf-8 -*-
"""
测试 B 单缝 — Kaiser-Bessel / AR 2D 倒谱方案对比。

运行（在 wellbore_moc_method/ 目录下）:
    python validation/cepstrum/kaiser_bessel_single.py
    python validate_moc_test_b_Kaiser-Bessel.py   # 兼容 wrapper
"""
from __future__ import annotations

import os
import sys
import time as time_module

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

from paths import output_path, SERIES_CEPSTRUM_KB, CASE_SINGLE

from cepstrum_mocdata import preprocess_moc_head
from wellbore_moc import MocConfig, simulate_wellbore
from validation.cepstrum import _kb_core as kb

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COMPARE_2D_FRAC_MARK_T_MAX = 10.0


def run_validation():
    print("=" * 72)
    print("测试 B — 2D 倒谱优化方案对比（Kaiser-Bessel / AR）")
    print("=" * 72)

    L = 5000.0
    a = 1450.0
    V0 = 1.0
    H0 = 300.0
    ts = 1.0
    dt = 1.0e-3
    tf = 100.0
    x_f = 4300.0
    Cf = 1.0e-5
    kleak = 0.0001
    H_ext = 100.0
    fs = 1.0 / dt
    wlen_sec = 30.0
    hop_ratio = 0.2

    cfg = MocConfig(
        wellbore_length=L, wellbore_diameter=0.1397,
        fluid_density=1000.0, fluid_viscosity=1.0e-6,
        wavespeed=a, roughness_height=4.5e-5,
        friction_model='steady', dt=dt, tf=tf,
        wellhead_bc='velocity_step', pump_shut_time=ts,
        initial_velocity=V0, initial_head=H0,
        theta=0.0, toe_bc='reservoir', toe_head=H0,
    )

    print(f"\n运行 MOC 仿真 (x_f={x_f}m, steady, kleak={kleak})...")
    t0 = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=[x_f],
        fracture_Cf=[Cf],
        fracture_kleak=[kleak],
        H_ext=H_ext,
        store_full_field=False,
    )
    print(f"  耗时: {time_module.time() - t0:.1f}s")

    t_sim = res['timestamps']
    H_wh = res['wellhead_head']
    x_f_aligned = float(res['x_grid'][res['fracture_indices'][0]])
    v = cfg.a_adj

    pre = preprocess_moc_head(t_sim, H_wh, fs=fs, ts=ts)
    x_work = pre['h_detrended']

    methods = kb.build_all_methods(x_work, fs, wlen_sec, hop_ratio, v, L, x_f_aligned)

    print("\n" + "-" * 72)
    print(f"{'方法':<22} {'峰深[m]':>8} {'误差[m]':>8} {'SNR':>8}")
    print("-" * 72)

    metrics = {}
    for name, (t_ax, q_ax, C, note) in methods.items():
        m = kb.evaluate_fracture_peak(C, q_ax, v, x_f_aligned, depth_min=100.0, depth_max=L)
        metrics[name] = m
        print(
            f"{name:<22} {m['peak_depth_m']:8.1f} {m['error_m']:8.1f} {m['snr']:8.2f}"
        )
    print("-" * 72)
    print(f"真实缝深: {x_f_aligned:.1f} m")

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        f'测试 B — 2D 倒谱优化方案对比\n'
        f'x_f={x_f_aligned:.0f}m, v={v:.1f}m/s, wlen={wlen_sec}s',
        fontsize=13, fontweight='bold',
    )
    for ax, (name, (t_ax, q_ax, C, note)) in zip(axes.flat, methods.items()):
        depth = kb.depth_axis(q_ax, v)
        kb.plot_2d_panel_single(
            ax, t_ax, depth, C, f'{name}\n{note}', x_f_aligned, 0.0, L,
            mark_t_max=COMPARE_2D_FRAC_MARK_T_MAX,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path_2d = output_path(SERIES_CEPSTRUM_KB, CASE_SINGLE, 'compare_2d.png')
    fig.savefig(path_2d, dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig2, axp = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for (name, m), color in zip(metrics.items(), colors):
        axp.plot(m['profile_depth'], m['profile'], lw=1.2, label=name, color=color)
        if np.isfinite(m['peak_depth_m']):
            axp.scatter(m['peak_depth_m'], m['peak_val'], c=color, s=40, marker='v', zorder=5)
    axp.axvline(x_f_aligned, color='k', ls='--', lw=1.5, label=f'真实缝 {x_f_aligned:.0f}m')
    axp.set_xlabel('深度 [m]')
    axp.set_ylabel('时间平均倒谱响应 (-C)')
    axp.set_title('各方法 1D 深度剖面对比（2D 倒谱沿时间平均）')
    axp.set_xlim([0, L])
    axp.legend(fontsize=8, loc='upper right')
    axp.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    path_prof = output_path(SERIES_CEPSTRUM_KB, CASE_SINGLE, 'profile_overlay.png')
    fig2.savefig(path_prof, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 8))
    fig3.suptitle(
        f'测试 B — 时间平均 1D 深度剖面\nx_f={x_f_aligned:.0f}m',
        fontsize=13, fontweight='bold',
    )
    for ax, (name, m) in zip(axes3.flat, metrics.items()):
        kb.plot_1d_depth_profile(
            ax, m['profile_depth'], m['profile'], name, x_f_aligned,
            peak_depth=m['peak_depth_m'], peak_val=m['peak_val'],
        )
        ax.set_xlim([0, L])
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path_1d = output_path(SERIES_CEPSTRUM_KB, CASE_SINGLE, 'profile_grid.png')
    fig3.savefig(path_1d, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    print(f"\n2D: {path_2d}")
    print(f"1D叠加: {path_prof}")
    print(f"1D分图: {path_1d}")

    best = min(metrics.items(), key=lambda kv: kv[1]['error_m'])
    print(
        f"\n最优: {best[0]}  峰={best[1]['peak_depth_m']:.1f}m, "
        f"误差={best[1]['error_m']:.1f}m, SNR={best[1]['snr']:.2f}"
    )

    return {'metrics': metrics, 'x_f_aligned': x_f_aligned, 'figures': [path_2d, path_prof, path_1d]}


if __name__ == '__main__':
    run_validation()
