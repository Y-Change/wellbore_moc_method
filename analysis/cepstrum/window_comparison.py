# -*- coding: utf-8 -*-
"""
2D cepstrogram window function comparison.

Generates a multi-panel figure of 2D cepstrograms computed with
different window functions (kaiser, hamming, hanning, rect, gauss),
with fracture positions marked by side arrows.
"""

import os, sys
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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore, G
from moc_simulate.cepstrum_mocdata import (
    preprocess_moc_head, prepare_cepstrum_signal,
    _resolve_cepstrum_params, cepstrogram,
)
from moc_simulate.paths import output_path, SERIES_ANALYSIS_WINDOW

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def add_side_arrow(ax, depth, label, t_span, side='left', color='#FF4444', lw=2.0):
    """Draw a side arrow at the plot edge pointing to a fracture depth, no in-plot line."""
    t_min, t_max = t_span
    if side == 'left':
        arrow_tip = t_min
        text_t = t_min - (t_max - t_min) * 0.08
        arrow_dir = '->'
        ha = 'right'
    else:
        arrow_tip = t_max
        text_t = t_max + (t_max - t_min) * 0.08
        arrow_dir = '<-'
        ha = 'left'
    ax.annotate(
        '', xy=(arrow_tip, depth), xytext=(text_t, depth),
        arrowprops=dict(arrowstyle=arrow_dir, color=color, lw=2.0,
                        connectionstyle='arc3'),
        annotation_clip=False,
    )
    ax.text(text_t, depth, label,
            color=color, fontsize=9, va='center', ha=ha,
            fontweight='bold')


def run():
    L = 5000.0; a = 1450.0; V0 = 1.0; H0 = 300.0
    ts = 1.0; dt = 1.0e-3; tf = 100.0
    x_f = 4100.0; Cf = 1.0e-5
    kleak = 0.0001; H_ext = 100.0

    cfg = MocConfig(
        wellbore_length=L, wellbore_diameter=0.1397,
        fluid_density=1000.0, fluid_viscosity=1.0e-6,
        wavespeed=a, roughness_height=4.5e-5,
        friction_model='steady', dt=dt, tf=tf,
        wellhead_bc='velocity_step', pump_shut_time=ts,
        initial_velocity=V0, initial_head=H0,
        theta=0.0, toe_bc='reservoir', toe_head=H0,
    )

    print('Running MOC simulation ...')
    res = simulate_wellbore(
        cfg, fracture_positions=[x_f], fracture_Cf=[Cf],
        fracture_kleak=[kleak], H_ext=H_ext, store_full_field=False,
    )
    H_wh = res['wellhead_head']
    t_sim = res['timestamps']
    frac_idx = res['fracture_indices'][0]
    x_f_aligned = float(res['x_grid'][frac_idx])
    v = cfg.a_adj
    print(f'  Simulation done. x_f_aligned = {x_f_aligned:.2f} m, v = {v:.1f} m/s')

    # --- Cepstrum setup ---
    fs = 1000.0
    pre = preprocess_moc_head(t_sim, H_wh, fs=fs, ts=ts)
    p_work = prepare_cepstrum_signal(pre, derivative=True, derivative_order=1)

    cep_params = _resolve_cepstrum_params(
        L, v, wlen_sec=None, lim2=None, lim1=0.0,
        fs=fs, signal_len=len(p_work),
    )
    wlen_sec_used = cep_params['wlen_sec']
    lim1_used = cep_params['lim1']
    lim2_used = cep_params['lim2']
    wlen = int(wlen_sec_used * fs)
    hop = max(1, int(0.1 * fs))

    depth_max = min(L, cep_params['depth_coverage_max'])

    # --- Window types to compare ---
    win_types = [
        ('kaiser',  'Kaiser (β=4)',   'tab:blue'),
        ('hamming', 'Hamming',             'tab:green'),
        ('hanning', 'Hanning',             'tab:orange'),
        ('rect',    'Rectangular',         'tab:red'),
        ('gauss',   'Gaussian (σ=w/6)', 'tab:purple'),
    ]

    # --- Compute cepstrograms for all windows ---
    results = {}
    for win_type, label, color in win_types:
        print(f'  Computing cepstrogram: {win_type} ...')
        C, q, t_cep = cepstrogram(p_work, wlen=wlen, hop=hop, fs=fs, win_type=win_type)
        valid = (q > lim1_used) & (q < lim2_used)
        results[win_type] = {
            'C': C[valid, :], 'q': q[valid], 't_cep': t_cep,
            'label': label, 'color': color,
        }

    # --- Build comparison figure ---
    print('Building comparison figure ...')
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 14))
    axes = axes.flatten()

    Q_depth_all = results['kaiser']['q'] * v / 2.0
    t_cep_all = results['kaiser']['t_cep']
    depth_max_display = min(L, Q_depth_all[-1])

    # Shared vmin/vmax for consistent color scaling
    all_C = np.concatenate([r['C'].ravel() for r in results.values()])
    vmin = float(np.percentile(all_C, 2))
    vmax = float(np.percentile(all_C, 98))
    if vmax <= vmin:
        vmax = vmin + 0.01

    for idx, (win_type, label, color) in enumerate(win_types):
        ax = axes[idx]
        r = results[win_type]
        C = r['C']
        t_cep = r['t_cep']

        T_mesh, Q_mesh = np.meshgrid(t_cep, Q_depth_all)
        im = ax.pcolormesh(T_mesh, Q_mesh, -C, shading='auto', cmap='jet',
                           vmin=vmin, vmax=vmax)

        # Fracture + toe markers as left-side arrows (no in-plot lines)
        t_span = (t_cep[0], t_cep[-1])
        add_side_arrow(ax, x_f_aligned, f'Frac {x_f_aligned:.0f}m',
                       t_span, side='left', color='#FF2222')
        add_side_arrow(ax, L, f'Toe {L:.0f}m',
                       t_span, side='left', color='#FF8800')

        ax.set_title(label, fontsize=13, fontweight='bold', color=color)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel('Depth [m]', fontsize=10)
        ax.invert_yaxis()
        ax.set_ylim([0, depth_max_display])

        # Colorbar per panel (compact)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('-C', fontsize=8)

    # 6th panel: window shape overlay
    ax = axes[5]
    t_win = np.linspace(0, wlen_sec_used, wlen)
    from scipy import signal as scipy_signal
    windows = {
        'Kaiser': np.kaiser(wlen, 4),
        'Hamming': scipy_signal.windows.hamming(wlen, sym=False),
        'Hanning': scipy_signal.windows.hann(wlen, sym=False),
        'Rect': np.ones(wlen),
        'Gauss': scipy_signal.windows.gaussian(wlen, std=wlen / 6),
    }
    colors_win = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']
    for (name, win), c in zip(windows.items(), colors_win):
        ax.plot(t_win, win, '-', color=c, lw=1.5, label=name, alpha=0.85)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title('Window Functions (wlen={:.2f}s, N={})'.format(wlen_sec_used, wlen),
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower center', ncol=3)
    ax.set_ylim([-0.02, 1.08])
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_xlim([0, wlen_sec_used])

    fig.suptitle(
        f'2D Cepstrogram: Window Function Comparison\n'
        f'L={L}m, a={v:.1f}m/s, x_f={x_f_aligned:.0f}m, Cf={Cf:.0e}m$^2$, '
        f'k_leak={kleak}, wlen={wlen_sec_used:.2f}s, hop=0.1s',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = output_path(SERIES_ANALYSIS_WINDOW, None, 'compare_2d.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved: {out_path}')

    # --- Also export individual high-res panels ---
    for win_type, label, color in win_types:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        r = results[win_type]
        C = r['C']; t_cep = r['t_cep']
        T_mesh, Q_mesh = np.meshgrid(t_cep, Q_depth_all)
        im2 = ax2.pcolormesh(T_mesh, Q_mesh, -C, shading='auto', cmap='jet',
                             vmin=vmin, vmax=vmax)
        t_span2 = (t_cep[0], t_cep[-1]) if len(t_cep) > 1 else (0, 1)
        add_side_arrow(ax2, x_f_aligned, f'Frac {x_f_aligned:.0f}m',
                       t_span2, side='left', color='#FF2222')
        add_side_arrow(ax2, L, f'Toe {L:.0f}m',
                       t_span2, side='left', color='#FF8800')
        ax2.set_title(f'2D Cepstrogram — {label}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time [s]', fontsize=11)
        ax2.set_ylabel('Depth [m]', fontsize=11)
        ax2.invert_yaxis()
        ax2.set_ylim([0, depth_max_display])
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('-C', fontsize=10)

        single_path = output_path(SERIES_ANALYSIS_WINDOW, None, f'cepstrum_{win_type}.png')
        plt.tight_layout()
        plt.savefig(single_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'  Individual: {single_path}')

    plt.close('all')
    print('Done.')


if __name__ == '__main__':
    run()
