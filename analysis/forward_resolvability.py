# -*- coding: utf-8 -*-
"""
steady 模式：B_coh / N_harm,eff / Δd_min 正推（不用匹配反推）。

定义链
------
瑞利判据（倒谱峰胖瘦 ↔ 对数谱相干带宽）：

    FWHM_τ ≈ 1 / B_coh
    B_coh  = N_harm,eff · f0 ,   f0 = a/(4L)
    Δd_min = a / (2 B_coh) = 2L / N_harm,eff = FWHM_d

B_coh 的正推操作定义（steady 停泵后去均值井口谱）
------------------------------------------------
在幅值谱 |S(f)| 上，取相对门限

    ε = 10^(-DR/20)     （默认 DR=80 dB → ε=1e-4）

从 f≈0 起找 |S(f)| ≥ ε·max|S| 的**最长低频连通支撑**（允许短于 0.5 f0 的空洞），
其右端频率即为相干有效带宽：

    B_coh := f_support_max
    N_harm,eff := B_coh / f0

含义：低于峰值 DR 分贝的谱线视为跌出可用动态范围，不再计入相干谐波梳。
此门限来自谱动态范围约定，**不是**由 dual 匹配间距反推。

默认数据：output/leakoff/steady_D50/single/moc_timeseries.csv

运行
----
    python analysis/forward_resolvability.py
    python analysis/forward_resolvability.py --dr-db 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

from paths import OUTPUT_DIR, SERIES_LEAKOFF
from wellbore_moc import MocConfig
from validation.config import WELL_CONFIG, SIM_CONFIG
from cepstrum_mocdata import preprocess_moc_head

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_CSV = os.path.join(
    OUTPUT_DIR, SERIES_LEAKOFF, 'steady_D50', 'single', 'moc_timeseries.csv',
)


def moc_grid() -> Dict[str, float]:
    w, s = WELL_CONFIG, SIM_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wellbore_diameter=w['wellbore_diameter'],
        fluid_density=w['fluid_density'],
        fluid_viscosity=w['fluid_viscosity'],
        wavespeed=w['wavespeed'],
        roughness_height=w['roughness_height'],
        friction_model='steady',
        dt=s['dt'], tf=s['tf'],
        wellhead_bc='velocity_step', pump_shut_time=s['ts'],
        initial_velocity=w['V0'], initial_head=w['H0'],
        theta=w['theta'], toe_bc='reservoir', toe_head=w['H0'],
    )
    a = float(cfg.a_adj)
    L = float(w['L'])
    return {
        'L': L,
        'a': a,
        'fs': 1.0 / float(s['dt']),
        'ts': float(s['ts']),
        'tf': float(s['tf']),
        'T_1d': float(s['tf']) - float(s['ts']),
        'f0': a / (4.0 * L),
        'T0': 4.0 * L / a,
        'dt_shut': float(s['dt']),
    }


def magnitude_spectrum(x: np.ndarray, fs: float):
    n = len(x)
    spec = fft(x)
    freqs = fftfreq(n, d=1.0 / fs)[: n // 2]
    mag = np.abs(spec[: n // 2])
    log_mag = np.log(mag + np.finfo(float).eps)
    return freqs, mag, log_mag


def coherent_bandwidth(
    freqs: np.ndarray,
    mag: np.ndarray,
    f0: float,
    dr_db: float = 80.0,
) -> Dict:
    """
    正推 B_coh：|S| 相对动态范围门限下的低频连通支撑宽度。
    """
    eps = 10.0 ** (-dr_db / 20.0)
    peak = float(np.max(mag))
    thr = eps * peak
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else f0
    max_gap_bins = max(1, int(round(0.5 * f0 / df)))

    above = mag >= thr
    last = 0
    gap = 0
    for i in range(len(mag)):
        if above[i]:
            last = i
            gap = 0
        else:
            gap += 1
            if i > 0 and gap > max_gap_bins:
                break

    B_coh = float(freqs[last])
    n_eff = B_coh / f0 if f0 > 0 else 0.0
    return {
        'dr_db': dr_db,
        'eps': eps,
        'peak_mag': peak,
        'thr_mag': thr,
        'B_coh_Hz': B_coh,
        'N_harm_eff': n_eff,
        'f_support_max_Hz': B_coh,
        'max_gap_bins': max_gap_bins,
    }


def rayleigh_from_B(a: float, L: float, f0: float, B_coh: float, n_eff: float) -> Dict:
    fwhm_tau = 1.0 / B_coh if B_coh > 0 else np.inf
    fwhm_d = (a / 2.0) * fwhm_tau
    dd = a / (2.0 * B_coh) if B_coh > 0 else np.inf
    dd_N = 2.0 * L / n_eff if n_eff > 0 else np.inf
    return {
        'FWHM_tau_s': fwhm_tau,
        'FWHM_d_m': fwhm_d,
        'delta_d_min_m': dd,
        'delta_d_min_from_N_m': dd_N,
    }


def sensitivity_table(freqs, mag, f0, a, L, dr_list: List[float]) -> List[Dict]:
    rows = []
    for dr in dr_list:
        bw = coherent_bandwidth(freqs, mag, f0, dr_db=dr)
        ray = rayleigh_from_B(a, L, f0, bw['B_coh_Hz'], bw['N_harm_eff'])
        rows.append({**bw, **ray})
    return rows


def plot_forward(freqs, mag, log_mag, grid, bw, ray, sens, save_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    f_show = min(max(bw['B_coh_Hz'] * 2.5, 5.0), 80.0)
    mask = freqs <= f_show

    ax = axes[0, 0]
    ax.semilogy(freqs[mask], mag[mask] / bw['peak_mag'], 'b-', lw=0.9,
                label='|S|/max|S|')
    ax.axhline(
        bw['eps'], color='C1', ls='--', lw=1.2,
        label=f'门限 ε=10^(-DR/20)={bw["eps"]:.1e} (DR={bw["dr_db"]:.0f} dB)',
    )
    ax.axvline(bw['B_coh_Hz'], color='C2', ls='-', lw=1.5,
               label=f'B_coh={bw["B_coh_Hz"]:.2f} Hz')
    ax.axvspan(0, bw['B_coh_Hz'], color='C2', alpha=0.12)
    ax.axvline(grid['f0'], color='orange', ls=':', lw=1.0,
               label=f'f0={grid["f0"]:.4f} Hz')
    ax.set_xlim(0, f_show)
    ax.set_ylim(bw['eps'] * 0.2, 2.0)
    ax.set_xlabel('频率 [Hz]')
    ax.set_ylabel('|S| / max|S|')
    ax.set_title('① 幅值谱与动态范围门限')
    ax.grid(True, which='both', ls='--', alpha=0.45)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(freqs[mask], log_mag[mask], 'b-', lw=0.8, label='log|S|')
    ax.axvline(bw['B_coh_Hz'], color='C2', lw=1.5, label='B_coh')
    ax.axvline(grid['f0'], color='orange', ls=':', lw=1.0,
               label=f'f0={grid["f0"]:.4f} Hz')
    ax.set_xlim(0, f_show)
    ax.set_xlabel('频率 [Hz]')
    ax.set_ylabel('log|S| [neper]')
    ax.set_title('② 对数谱（倒谱 IFFT 输入）')
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    drs = [r['dr_db'] for r in sens]
    dds = [r['delta_d_min_m'] for r in sens]
    ns = [r['N_harm_eff'] for r in sens]
    ax.plot(drs, dds, 'o-', color='C0', label='Δd_min = a/(2 B_coh)')
    ax.axhline(50.0, color='gray', ls='--', lw=1.0, label='参考线 50 m')
    ax.set_xlabel('动态范围门限 DR [dB]')
    ax.set_ylabel('Δd_min [m]')
    ax.set_title('③ Δd_min 对 DR 门限的敏感性')
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(drs, ns, 's--', color='C3', alpha=0.7, label='N_harm,eff')
    ax2.set_ylabel('N_harm,eff')
    ax2.legend(fontsize=8, loc='center right')

    ax = axes[1, 1]
    ax.axis('off')
    text = (
        '正推参数链（steady / velocity_step）\n'
        '────────────────────────────────\n'
        f"L = {grid['L']:.0f} m，a = {grid['a']:.4f} m/s\n"
        f"f0 = a/(4L) = {grid['f0']:.5f} Hz\n"
        f"T_1d = tf-ts = {grid['T_1d']:.1f} s\n"
        f"关断历时 ≈ dt = {grid['dt_shut']*1e3:.1f} ms\n"
        '  （勿用 1/ts 当作 B_coh）\n'
        '\n'
        '瑞利判据\n'
        '  FWHM_τ ≈ 1/B_coh\n'
        '  B_coh = N_harm,eff · f0\n'
        '  Δd_min = a/(2 B_coh) = 2L/N\n'
        '\n'
        '由 |S| 支撑正推 B_coh\n'
        f"  DR = {bw['dr_db']:.0f} dB，ε = {bw['eps']:.1e}\n"
        f"  B_coh = {bw['B_coh_Hz']:.3f} Hz\n"
        f"  N_harm,eff = {bw['N_harm_eff']:.1f}\n"
        '\n'
        '结果\n'
        f"  FWHM_τ = {ray['FWHM_tau_s']:.5f} s\n"
        f"  FWHM_d = {ray['FWHM_d_m']:.2f} m\n"
        f"  Δd_min = {ray['delta_d_min_m']:.2f} m\n"
    )
    ax.text(
        0.04, 0.98, text, transform=ax.transAxes, va='top', ha='left',
        fontsize=10,
        fontfamily=['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
        bbox=dict(boxstyle='round', facecolor='#f7f7f7', edgecolor='#ccc'),
    )
    ax.set_title('④ 正推汇总')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'图已保存: {save_path}')


def render_md(grid, bw, ray, sens, csv_path: str) -> str:
    sens_rows = '\n'.join(
        f"| {r['dr_db']:.0f} | {r['eps']:.1e} | {r['B_coh_Hz']:.2f} | "
        f"{r['N_harm_eff']:.1f} | {r['delta_d_min_m']:.1f} |"
        for r in sens
    )
    return f"""# steady 模式：B_coh / N_harm,eff / Δd_min **正推**

> 数据：`{csv_path}`  
> **不含**用 dual 匹配间距反推 N_harm。

## 1. 瑞利判据（定义）

倒谱峰半高全宽由对数谱中相干谐波梳的有效带宽决定：

$$
\\mathrm{{FWHM}}_\\tau \\approx \\frac{{1}}{{B_{{\\mathrm{{coh}}}}}}
$$

$$
B_{{\\mathrm{{coh}}}} = N_{{\\mathrm{{harm,eff}}}} \\cdot f_0,
\\qquad f_0 = \\frac{{a}}{{4L}}
$$

深度域 $d=q\\cdot a/2$：

$$
\\Delta d_{{\\min}}
= \\frac{{a}}{{2\\,B_{{\\mathrm{{coh}}}}}}
= \\frac{{2L}}{{N_{{\\mathrm{{harm,eff}}}}}}
= \\mathrm{{FWHM}}_d
$$

物理含义：两缝 quefrency 间距 $\\Delta\\tau=2\\Delta d/a$ 须达到约一个 FWHM，才能在倒谱上分开。

## 2. B_coh 的正推操作定义

steady + `velocity_step` 下，停泵后去均值井口水头的幅值谱为 $|S(f)|$。  
取动态范围门限（相对峰值）：

$$
\\varepsilon = 10^{{-\\mathrm{{DR}}/20}}
\\quad(\\text{{默认 DR}}=80\\ \\mathrm{{dB}}\\Rightarrow\\varepsilon=10^{{-4}})
$$

从低频起找 $|S(f)|\\ge\\varepsilon\\,\\max|S|$ 的连通支撑（允许短于 $0.5 f_0$ 的空洞），右端频率定义为：

$$
B_{{\\mathrm{{coh}}}} := f_{{\\mathrm{{support,max}}}},
\\qquad
N_{{\\mathrm{{harm,eff}}}} := B_{{\\mathrm{{coh}}}}/f_0
$$

说明：

- 关断历时 ≈ $dt$（毫秒级），**不能**再用 $1/t_s=1\\,\\mathrm{{Hz}}$ 当作 $B_{{\\mathrm{{coh}}}}$ 上限。
- DR=80 dB 是谱可用动态范围约定（峰值以下 80 dB 视为跌出相干谐波梳），不是由匹配矩阵标定。

## 3. 本算例几何

| 量 | 数值 |
|----|------|
| L | {grid['L']:.0f} m |
| a_adj | {grid['a']:.4f} m/s |
| f0 = a/(4L) | {grid['f0']:.5f} Hz |
| T_1d | {grid['T_1d']:.1f} s |
| 关断 | velocity_step，≈{grid['dt_shut']*1e3:.1f} ms |

## 4. 正推结果（DR = {bw['dr_db']:.0f} dB）

| 量 | 数值 |
|----|------|
| ε | {bw['eps']:.1e} |
| **B_coh** | **{bw['B_coh_Hz']:.3f} Hz** |
| **N_harm,eff** | **{bw['N_harm_eff']:.1f}** |
| FWHM_τ = 1/B_coh | {ray['FWHM_tau_s']:.5f} s |
| FWHM_d = a/(2 B_coh) | {ray['FWHM_d_m']:.2f} m |
| **Δd_min** | **{ray['delta_d_min_m']:.2f} m** |
| Δd_min = 2L/N | {ray['delta_d_min_from_N_m']:.2f} m |

代入验算：

$$
N_{{\\mathrm{{harm,eff}}}} = \\frac{{B_{{\\mathrm{{coh}}}}}}{{f_0}}
= \\frac{{{bw['B_coh_Hz']:.3f}}}{{{grid['f0']:.5f}}}
\\approx {bw['N_harm_eff']:.0f}
$$

$$
\\Delta d_{{\\min}} = \\frac{{2L}}{{N}}
= \\frac{{{2*grid['L']:.0f}}}{{{bw['N_harm_eff']:.1f}}}
\\approx {ray['delta_d_min_from_N_m']:.1f}\\ \\mathrm{{m}}
$$

## 5. DR 门限敏感性（正推族）

| DR [dB] | ε | B_coh [Hz] | N_harm,eff | Δd_min [m] |
|---------|---|------------|------------|------------|
{sens_rows}

## 6. 与匹配实验的关系

匹配实验（`steady_Dall`）用于**验证**正推下界，不参与本计算。  
若正推 Δd_min≈{ray['delta_d_min_m']:.0f} m，则预期 Δd≳该值的 dual 可分、明显更小则不可分。
"""


def main():
    ap = argparse.ArgumentParser(description='正推 B_coh / N_harm,eff / Δd_min')
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--dr-db', type=float, default=80.0,
                    help='相对峰值动态范围门限 [dB]，默认 80')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(
            f'找不到 {args.csv}\n'
            f'请先: python validation/leakoff_multi.py --friction steady_D50 --case single'
        )

    grid = moc_grid()
    data = np.loadtxt(args.csv, delimiter=',', skiprows=1)
    pre = preprocess_moc_head(data[:, 0], data[:, 1], fs=grid['fs'], ts=grid['ts'])
    freqs, mag, log_mag = magnitude_spectrum(pre['h_detrended'], grid['fs'])

    bw = coherent_bandwidth(freqs, mag, grid['f0'], dr_db=args.dr_db)
    ray = rayleigh_from_B(
        grid['a'], grid['L'], grid['f0'], bw['B_coh_Hz'], bw['N_harm_eff'],
    )
    sens = sensitivity_table(
        freqs, mag, grid['f0'], grid['a'], grid['L'],
        dr_list=[60, 70, 80, 90, 100],
    )

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, 'forward_resolvability.png')
    md_path = os.path.join(out_dir, 'FORWARD_RESOLVABILITY.md')
    series_md = os.path.join(OUTPUT_DIR, SERIES_LEAKOFF, 'FORWARD_RESOLVABILITY.md')

    title = (
        '正推：B_coh / N_harm,eff / Δd_min（steady）\n'
        f'{os.path.basename(os.path.dirname(args.csv))}/'
        f'{os.path.basename(args.csv)}'
    )
    plot_forward(freqs, mag, log_mag, grid, bw, ray, sens, png, title)

    md = render_md(grid, bw, ray, sens, args.csv)
    for path in (md_path, series_md):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md)

    payload = {
        'method': 'spectral_support_dynamic_range',
        'grid': grid,
        'bandwidth': bw,
        'rayleigh': ray,
        'sensitivity': sens,
        'csv': args.csv,
    }
    js = os.path.join(out_dir, 'forward_resolvability.json')
    with open(js, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'报告: {md_path}')
    print(f'副本: {series_md}')
    print(
        f"B_coh={bw['B_coh_Hz']:.3f} Hz, "
        f"N_harm,eff={bw['N_harm_eff']:.1f}, "
        f"Δd_min={ray['delta_d_min_m']:.2f} m"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
