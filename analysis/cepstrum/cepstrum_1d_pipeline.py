# -*- coding: utf-8 -*-
"""
1D 实倒谱计算过程可视化（逐步展示）。

链路：原始井口水头 → 频域幅值谱 → 对数谱（log|FFT|）→ 1D 实倒谱

默认读取：
  output/leakoff/steady_D50/quad/moc_timeseries.csv

运行
----
    python analysis/cepstrum_1d_pipeline.py
    python analysis/cepstrum_1d_pipeline.py --csv output/leakoff/steady_D50/quad/moc_timeseries.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft

import os
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'README.md')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find wellbore_moc_method root')
    _d = _parent

from moc_simulate.paths import OUTPUT_DIR, SERIES_LEAKOFF, output_path
from moc_simulate.wellbore_moc import MocConfig
from moc_simulate.config import WELL_CONFIG, SIM_CONFIG
from moc_simulate.cepstrum_mocdata import preprocess_moc_head

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_CSV = os.path.join(
    OUTPUT_DIR, SERIES_LEAKOFF, 'steady_D50', 'quad', 'moc_timeseries.csv',
)
DEFAULT_JSON = os.path.join(
    OUTPUT_DIR, SERIES_LEAKOFF, 'steady_D50', 'quad', 'moc_leakoff.json',
)


def _a_adj() -> float:
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
    return float(cfg.a_adj)


def load_case(csv_path: str, json_path: str | None):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = data[:, 0]
    H_wh = data[:, 1]

    x_f = [4100.0, 4150.0, 4200.0, 4250.0]
    ts = float(SIM_CONFIG['ts'])
    L = float(WELL_CONFIG['L'])
    if json_path and os.path.isfile(json_path):
        with open(json_path, encoding='utf-8') as f:
            meta = json.load(f)
        cfg = meta.get('config') or {}
        x_f = list(cfg.get('x_f') or x_f)
        ts = float(cfg.get('ts', ts))
        L = float(cfg.get('L', L))
    return t, H_wh, x_f, ts, L


def compute_pipeline(t, H_wh, fs: float, ts: float, a: float, L: float):
    """逐步计算，返回绘图用中间量（与 real_cepstrum_1d 一致）。"""
    pre = preprocess_moc_head(t, H_wh, fs=fs, ts=ts)
    x = pre['h_detrended']
    t_after = pre['t_after']
    h_after = pre['h_after']

    n = len(x)
    spec = fft(x)
    freqs_full = fftfreq(n, d=1.0 / fs)
    mag_full = np.abs(spec)
    log_mag_full = np.log(mag_full + np.finfo(float).eps)  # 倒谱所用
    log_power_full = 2.0 * log_mag_full                   # log|S|² = 2 log|S|

    # 正频率半轴（展示用）
    half = n // 2
    freqs = freqs_full[:half]
    mag = mag_full[:half] / n
    log_mag = log_mag_full[:half]
    log_power = log_power_full[:half]

    raw_ceps = np.real(ifft(log_mag_full))
    rown = n // 2 + 1
    q = np.arange(rown) / fs
    C = raw_ceps[:rown]
    depth = q * a / 2.0
    response = -C

    # 有效显示：截到井深附近
    q_max = 2.0 * L / a
    valid = (q > 0) & (q < q_max)

    f0 = a / (4.0 * L)

    return {
        't_after': t_after,
        'h_after': h_after,
        'h_detrended': x,
        'freqs': freqs,
        'mag': mag,
        'log_mag': log_mag,
        'log_power': log_power,
        'q': q[valid],
        'depth': depth[valid],
        'response': response[valid],
        'C': C[valid],
        'fs': fs,
        'f0': f0,
        'a': a,
        'L': L,
        'ts': ts,
    }


def plot_pipeline(pipe: dict, x_f: list, save_path: str, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ① 原始井口水头（停泵后）
    ax = axes[0, 0]
    ax.plot(pipe['t_after'], pipe['h_after'], 'b-', lw=0.9, label='井口水头 $H$')
    ax.axvline(pipe['ts'], color='g', ls='--', lw=1.2, label=f"停泵 $t_s$={pipe['ts']}s")
    ax.set_xlabel('时间 [s]')
    ax.set_ylabel('水头 [m]')
    ax.set_title('① 原始井口压力（停泵后时域）')
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(fontsize=8, loc='best')

    # ② 频域幅值谱
    ax = axes[0, 1]
    f_show = min(2.0, float(pipe['freqs'][-1]) if len(pipe['freqs']) else 2.0)
    mask = pipe['freqs'] <= f_show
    ax.plot(pipe['freqs'][mask], pipe['mag'][mask], 'b-', lw=1.0, label='$|\\mathrm{FFT}|$')
    ax.axvline(pipe['f0'], color='orange', ls=':', lw=1.2,
               label=f"$f_0$=a/(4L)={pipe['f0']:.4f} Hz")
    ax.set_xlabel('频率 [Hz]')
    ax.set_ylabel('幅值 [m]')
    ax.set_title('② 频域信号（幅值谱）')
    ax.set_xlim(0, f_show)
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(fontsize=8, loc='best')

    # ③ 对数功率谱 + 倒谱所用 log|FFT|
    ax = axes[1, 0]
    ax.plot(pipe['freqs'][mask], pipe['log_power'][mask], 'C0-', lw=1.0,
            label='$\\log|S|^2$（对数功率谱）')
    ax.plot(pipe['freqs'][mask], pipe['log_mag'][mask], 'C3--', lw=1.0, alpha=0.85,
            label='$\\log|S|$（倒谱 IFFT 输入）')
    ax.axvline(pipe['f0'], color='orange', ls=':', lw=1.2)
    ax.set_xlabel('频率 [Hz]')
    ax.set_ylabel('对数幅值 [neper]')
    ax.set_title('③ 对数后的功率谱 / 对数幅值谱')
    ax.set_xlim(0, f_show)
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(fontsize=8, loc='best')

    # ④ 1D 实倒谱
    ax = axes[1, 1]
    ax.plot(pipe['depth'], pipe['response'], 'b-', lw=1.0, label='$-C(q)$（实倒谱）')
    for i, xf in enumerate(x_f):
        ax.axvline(xf, color='r', ls='--', lw=1.0, alpha=0.75,
                   label='裂缝位置' if i == 0 else None)
    ax.set_xlabel('深度 $d = q \\cdot a/2$ [m]')
    ax.set_ylabel('倒谱响应 $-C$')
    ax.set_title('④ 1D 实倒谱（IFFT of $\\log|\\mathrm{FFT}|$）')
    ax.set_xlim(0, pipe['L'])
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(fontsize=8, loc='best')

    # 底部公式说明
    fig.text(
        0.5, 0.01,
        r'$c(q)=\mathrm{Re}\{\mathrm{IFFT}(\log|\mathrm{FFT}(x)|)\}$'
        r'  ·  $d=q\cdot a/2$'
        r'  ·  本图 $x$ = 停泵后去均值井口水头',
        ha='center', fontsize=10, style='italic',
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='1D 实倒谱过程可视化')
    parser.add_argument('--csv', default=DEFAULT_CSV, help='moc_timeseries.csv 路径')
    parser.add_argument('--json', default=DEFAULT_JSON, help='moc_leakoff.json（可选）')
    parser.add_argument(
        '--out', default=None,
        help='输出 PNG；默认写到同目录 cepstrum_1d_pipeline.png',
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(
            f'找不到 CSV: {args.csv}\n'
            f'请先运行: python validation/leakoff_multi.py --friction steady_D50 --case quad'
        )

    t, H_wh, x_f, ts, L = load_case(args.csv, args.json)
    fs = 1.0 / float(SIM_CONFIG['dt'])
    a = _a_adj()

    pipe = compute_pipeline(t, H_wh, fs=fs, ts=ts, a=a, L=L)

    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                           'cepstrum_1d_pipeline.png')

    title = (
        f'1D 实倒谱计算过程 — steady_D50 / quad\n'
        f'x_f={x_f} m, a_adj={a:.2f} m/s, fs={fs:.0f} Hz, T≈{pipe["t_after"][-1]-ts:.1f} s'
    )
    plot_pipeline(pipe, x_f, out, title)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
