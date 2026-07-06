# -*- coding: utf-8 -*-
"""Kaiser-Bessel / AR 2D 倒谱核心算法（供 kaiser_bessel_single / multi 共用）。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal

from cepstrum_mocdata import cepstrogram


def _preemphasis(frame: np.ndarray, mu: float = 0.97) -> np.ndarray:
    y = np.empty_like(frame)
    y[0] = frame[0]
    y[1:] = frame[1:] - mu * frame[:-1]
    return y


def dynamic_kaiser_beta(
    target_depth_m: Optional[float],
    fs: float,
    wavespeed: float,
) -> float:
    if target_depth_m is None:
        return 8.6
    tau_expected = 2.0 * target_depth_m / wavespeed
    return float(np.clip(tau_expected * fs / 100.0, 4.0, 14.0))


def _raised_sine_lifter(
    n_bins: int,
    fs: float,
    tau_max: float,
    K: float = 22.0,
    zero_cutoff_sec: float = 0.01,
) -> np.ndarray:
    q = np.arange(n_bins) / fs
    lifter = np.ones(n_bins, dtype=float)
    mask = (q > 0) & (q < tau_max)
    lifter[mask] = 1.0 + (K / 2.0) * np.sin(np.pi * q[mask] / tau_max)
    lifter[q < zero_cutoff_sec] = 0.0
    return lifter


def max_quefrency_bins(
    fs: float,
    n_win: int,
    wavespeed: float,
    wellbore_length: float,
) -> int:
    n_by_window = int(np.ceil((1 + n_win) / 2))
    q_max_phys = 2.0 * wellbore_length / wavespeed
    n_by_depth = int(np.floor(q_max_phys * fs)) + 1
    return min(n_by_window, n_by_depth)


def optimized_2d_cepstrogram(
    x: np.ndarray,
    fs: float,
    window_length_sec: float,
    wavespeed: float = 1450.0,
    wellbore_length: float = 5000.0,
    hop_ratio: float = 0.2,
    target_depth_m: Optional[float] = None,
    *,
    use_dynamic_kaiser: bool = True,
    use_epsilon_log: bool = True,
    use_preemphasis: bool = True,
    use_lifter: bool = True,
    epsilon: float = 1e-12,
    preemph_mu: float = 0.97,
    lifter_K: float = 22.0,
    lifter_tau_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).flatten()
    n_win = max(4, int(window_length_sec * fs))
    n_hop = max(1, int(n_win * hop_ratio))
    n_fft = int(2 ** np.ceil(np.log2(n_win) + 2))
    n_cep = max_quefrency_bins(fs, n_win, wavespeed, wellbore_length)

    beta = dynamic_kaiser_beta(target_depth_m, fs, wavespeed) if use_dynamic_kaiser else 8.6
    window = signal.windows.kaiser(n_win, beta)

    if lifter_tau_max is None:
        lifter_tau_max = 2.0 * wellbore_length / wavespeed
    lifter = _raised_sine_lifter(n_cep, fs, lifter_tau_max, K=lifter_K)

    num_frames = 1 + (len(x) - n_win) // n_hop
    cgram = np.zeros((n_cep, num_frames))
    time_axis = np.zeros(num_frames)
    quefrency_axis = np.arange(n_cep) / fs

    for i in range(num_frames):
        start = i * n_hop
        frame = x[start:start + n_win]
        if len(frame) < n_win:
            break

        if use_preemphasis:
            frame = _preemphasis(frame, mu=preemph_mu)

        frame = frame - np.mean(frame)
        windowed = frame * window
        spectrum = np.fft.rfft(windowed, n=n_fft)

        if use_epsilon_log:
            log_spec = np.log(np.abs(spectrum) ** 2 + epsilon)
        else:
            log_spec = np.log(np.abs(spectrum) + np.finfo(float).eps)

        ceps_frame = np.fft.irfft(log_spec, n=n_fft)[:n_cep]
        if use_lifter:
            ceps_frame = ceps_frame * lifter

        cgram[:, i] = ceps_frame
        time_axis[i] = (start + n_win / 2.0) / fs

    return time_axis, quefrency_axis, cgram


def ar_cepstrum_frame(
    frame: np.ndarray,
    p_order: int = 30,
    num_coeffs: Optional[int] = None,
) -> np.ndarray:
    frame = np.asarray(frame, dtype=float).flatten()
    n = len(frame)
    if num_coeffs is None:
        num_coeffs = p_order * 2
    frame = frame - np.mean(frame)

    if n <= p_order + 2:
        return np.zeros(num_coeffs)

    r = np.correlate(frame, frame, mode='full')[n - 1:n + p_order + 1]
    R = linalg.toeplitz(r[:p_order])
    try:
        a_pos = linalg.solve(R, r[1:p_order + 1])
    except linalg.LinAlgError:
        return np.zeros(num_coeffs)

    a = a_pos
    c = np.zeros(num_coeffs + 1)
    c[1] = -a[0]
    for n_idx in range(2, p_order + 1):
        c[n_idx] = -a[n_idx - 1]
        for k in range(1, n_idx):
            c[n_idx] -= (1.0 - k / n_idx) * a[k - 1] * c[n_idx - k]
    for n_idx in range(p_order + 1, num_coeffs + 1):
        for k in range(1, p_order + 1):
            c[n_idx] -= (1.0 - k / n_idx) * a[k - 1] * c[n_idx - k]

    return c[1:num_coeffs + 1]


def ar_2d_cepstrogram(
    x: np.ndarray,
    fs: float,
    window_length_sec: float,
    hop_ratio: float = 0.2,
    p_order: int = 60,
    wavespeed: float = 1450.0,
    wellbore_length: float = 5000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).flatten()
    n_win = max(p_order + 4, int(window_length_sec * fs))
    n_hop = max(1, int(n_win * hop_ratio))
    num_coeffs = int(np.ceil(2.0 * wellbore_length / wavespeed * fs)) + 64
    p_order = min(p_order, n_win // 3)

    num_frames = 1 + (len(x) - n_win) // n_hop
    cgram = np.zeros((num_coeffs, num_frames))
    time_axis = np.zeros(num_frames)
    quefrency_axis = np.arange(1, num_coeffs + 1) / fs

    for i in range(num_frames):
        start = i * n_hop
        frame = x[start:start + n_win]
        if len(frame) < n_win:
            break
        cgram[:, i] = ar_cepstrum_frame(frame, p_order=p_order, num_coeffs=num_coeffs)
        time_axis[i] = (start + n_win / 2.0) / fs

    return time_axis, quefrency_axis, cgram


def depth_axis(q: np.ndarray, v: float) -> np.ndarray:
    return q * v / 2.0


def baseline_2d_cepstrogram(
    x: np.ndarray,
    fs: float,
    window_length_sec: float,
    hop_ratio: float = 0.2,
    win_type: str = 'hamming',
    kaiser_beta: float = 4.0,
    wavespeed: float = 1450.0,
    wellbore_length: float = 5000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_win = max(4, int(window_length_sec * fs))
    n_hop = max(1, int(n_win * hop_ratio))
    C, q, t = cepstrogram(x, wlen=n_win, hop=n_hop, fs=fs, win_type=win_type)
    if win_type == 'kaiser':
        _ = kaiser_beta
    n_cep = max_quefrency_bins(fs, n_win, wavespeed, wellbore_length)
    return t, q[:n_cep], C[:n_cep, :]


def compute_time_avg_depth_profile(
    C: np.ndarray,
    q: np.ndarray,
    v: float,
    depth_min: float = 100.0,
    depth_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = depth_axis(q, v)
    if depth_max is None:
        depth_max = float(depth[-1]) if len(depth) else 5000.0
    mask = (depth >= depth_min) & (depth <= depth_max)
    if not np.any(mask):
        return depth, np.zeros_like(depth)
    profile = -np.mean(C[mask, :], axis=1)
    return depth[mask], profile


def evaluate_fracture_peak(
    C: np.ndarray,
    q: np.ndarray,
    v: float,
    true_depth_m: float,
    depth_min: float = 100.0,
    depth_max: Optional[float] = None,
) -> Dict:
    depth_kept, profile = compute_time_avg_depth_profile(
        C, q, v, depth_min=depth_min, depth_max=depth_max,
    )
    if len(profile) == 0:
        return {
            'peak_depth_m': np.nan,
            'error_m': np.nan,
            'peak_val': np.nan,
            'snr': np.nan,
            'profile_depth': depth_kept,
            'profile': profile,
        }

    i_peak = int(np.argmax(profile))
    peak_depth = float(depth_kept[i_peak])
    bg = np.percentile(profile, 50)
    peak_val = float(profile[i_peak])
    snr = peak_val / max(abs(bg), 1e-12)

    return {
        'peak_depth_m': peak_depth,
        'error_m': abs(peak_depth - true_depth_m),
        'peak_val': peak_val,
        'snr': float(snr),
        'profile_depth': depth_kept,
        'profile': profile,
    }


def build_all_methods(
    x_work: np.ndarray,
    fs: float,
    wlen_sec: float,
    hop_ratio: float,
    v: float,
    L: float,
    target_depth_m: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    methods: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}

    t0, q0, C0 = baseline_2d_cepstrogram(
        x_work, fs, wlen_sec, hop_ratio, win_type='hamming',
        wavespeed=v, wellbore_length=L,
    )
    methods['基线 Hamming'] = (t0, q0, C0, 'Hamming, log|X|')

    t1, q1, C1 = baseline_2d_cepstrogram(
        x_work, fs, wlen_sec, hop_ratio, win_type='kaiser', kaiser_beta=4.0,
        wavespeed=v, wellbore_length=L,
    )
    methods['基线 Kaiser beta=4'] = (t1, q1, C1, 'Kaiser beta=4')

    beta = dynamic_kaiser_beta(target_depth_m, fs, v)
    t2, q2, C2 = optimized_2d_cepstrogram(
        x_work, fs, wlen_sec, wavespeed=v, wellbore_length=L, hop_ratio=hop_ratio,
        target_depth_m=target_depth_m,
        use_dynamic_kaiser=True, use_epsilon_log=False,
        use_preemphasis=False, use_lifter=False,
    )
    methods['方案1 动态Kaiser'] = (t2, q2, C2, f'动态 beta={beta:.1f}')

    t3, q3, C3 = optimized_2d_cepstrogram(
        x_work, fs, wlen_sec, wavespeed=v, wellbore_length=L, hop_ratio=hop_ratio,
        target_depth_m=target_depth_m,
        use_dynamic_kaiser=True, use_epsilon_log=True,
        use_preemphasis=True, use_lifter=False,
    )
    methods['方案1+2 eps+预加重'] = (t3, q3, C3, 'Kaiser + eps + 预加重')

    t4, q4, C4 = optimized_2d_cepstrogram(
        x_work, fs, wlen_sec, wavespeed=v, wellbore_length=L, hop_ratio=hop_ratio,
        target_depth_m=target_depth_m,
        use_dynamic_kaiser=True, use_epsilon_log=True,
        use_preemphasis=True, use_lifter=True,
    )
    methods['方案1+2+3 全复合'] = (t4, q4, C4, 'Kaiser + eps + 预加重 + Lifter')

    t5, q5, C5 = ar_2d_cepstrogram(
        x_work, fs, wlen_sec, hop_ratio=hop_ratio, p_order=60,
        wavespeed=v, wellbore_length=L,
    )
    methods['方案4 AR倒谱'] = (t5, q5, C5, f'AR p=60, n={len(q5)}')
    return methods


def plot_1d_depth_profile(
    ax,
    depth: np.ndarray,
    profile: np.ndarray,
    title: str,
    fracture_depth: float,
    peak_depth: Optional[float] = None,
    peak_val: Optional[float] = None,
) -> None:
    ax.plot(depth, profile, 'b-', lw=1.0)
    ax.axvline(fracture_depth, color='k', ls='--', lw=1.2, alpha=0.8,
               label=f'真实缝 {fracture_depth:.0f}m')
    if peak_depth is not None and peak_val is not None and np.isfinite(peak_depth):
        ax.scatter([peak_depth], [peak_val], c='r', s=60, zorder=5, marker='v',
                   label=f'检测峰 {peak_depth:.0f}m')
        ax.annotate(
            f'{peak_depth:.0f}m',
            (peak_depth, peak_val),
            textcoords='offset points', xytext=(0, 8),
            fontsize=8, color='r', ha='center',
        )
    ax.set_xlabel('深度 [m]')
    ax.set_ylabel('时间平均倒谱响应 (-C)')
    ax.set_title(title, fontsize=10)
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(fontsize=7, loc='upper right')


def plot_2d_panel_single(
    ax,
    t: np.ndarray,
    depth: np.ndarray,
    C: np.ndarray,
    title: str,
    fracture_depth: float,
    depth_min: float = 0.0,
    depth_max: float = 5000.0,
) -> None:
    mask = (depth >= depth_min) & (depth <= depth_max)
    depth_plot = depth[mask]
    C_plot = C[mask, :]
    T, D = np.meshgrid(t, depth_plot)
    data = -C_plot
    vmin = float(np.percentile(data, 2))
    vmax = float(np.percentile(data, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax.pcolormesh(T, D, data, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    ax.axhline(fracture_depth, color='yellow', ls='--', lw=0.8, alpha=0.7)
    ax.invert_yaxis()
    ax.set_ylim([depth_max, depth_min])
    ax.set_xlabel('时间 [s]')
    ax.set_ylabel('深度 [m]')
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, fraction=0.046, aspect=30)


def plot_2d_panel_multi(
    ax,
    t: np.ndarray,
    depth: np.ndarray,
    C: np.ndarray,
    title: str,
    fracture_depths: list,
    depth_max: float,
    line_colors: list,
) -> None:
    mask = (depth >= 0) & (depth <= depth_max)
    depth_plot = depth[mask]
    C_plot = C[mask, :]
    T, D = np.meshgrid(t, depth_plot)
    data = -C_plot
    vmin = float(np.percentile(data, 2))
    vmax = float(np.percentile(data, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax.pcolormesh(T, D, data, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    for i, fd in enumerate(fracture_depths):
        ax.axhline(fd, color=line_colors[i % len(line_colors)], ls='--', lw=0.8, alpha=0.85)
    ax.invert_yaxis()
    ax.set_ylim([depth_max, 0])
    ax.set_xlabel('时间 [s]')
    ax.set_ylabel('深度 [m]')
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, fraction=0.046, aspect=30)


def plot_1d_profile_multi(
    ax,
    depth: np.ndarray,
    profile: np.ndarray,
    title: str,
    fracture_depths: list,
    matches: list,
) -> None:
    ax.plot(depth, profile, 'b-', lw=1.0)
    for i, fd in enumerate(fracture_depths):
        ax.axvline(fd, color='gray', ls='--', lw=1.0, alpha=0.7,
                   label=f'真实F{i+1} {fd:.0f}m' if i < 3 else None)
    for m in matches:
        if m['matched'] and np.isfinite(m['peak_depth_m']):
            ax.scatter(m['peak_depth_m'], m['peak_val'], c='r', s=45, marker='v', zorder=5)
            ax.annotate(
                f"F{m['frac_id']}:{m['peak_depth_m']:.0f}",
                (m['peak_depth_m'], m['peak_val']),
                textcoords='offset points', xytext=(0, 6),
                fontsize=6, color='r', ha='center',
            )
    ax.set_xlabel('深度 [m]')
    ax.set_ylabel('时间平均 (-C)')
    ax.set_title(title, fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)
