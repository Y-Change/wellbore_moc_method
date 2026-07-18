# -*- coding: utf-8 -*-
"""Kaiser-Bessel / AR 2D 倒谱核心算法（供 kaiser_bessel_multi / wlen_sweep 共用）。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    profile = -np.sum(C[mask, :], axis=1)
    return depth[mask], profile


def peak_find_params(
    true_depths_m: List[float],
    v: float,
    fs: float,
) -> Tuple[float, int, int]:
    depth_step = v / (2.0 * fs)
    sorted_d = sorted(true_depths_m)
    min_spacing = float(min(np.diff(sorted_d))) if len(sorted_d) > 1 else 300.0
    match_tol_m = float(np.clip(min_spacing * 0.45, 80.0, 250.0))
    peak_distance = max(5, int(min_spacing / depth_step * 0.35))
    top_n = len(true_depths_m) + 4
    return match_tol_m, peak_distance, top_n


def _empty_fracture_peak_metrics(
    depth_kept: np.ndarray,
    profile: np.ndarray,
    true_depths_m: List[float],
) -> Dict:
    matches = [{
        'frac_id': i + 1,
        'true_depth_m': float(d),
        'peak_depth_m': np.nan,
        'peak_val': np.nan,
        'error_m': np.nan,
        'matched': False,
    } for i, d in enumerate(true_depths_m)]
    return {
        'matches': matches,
        'n_matched': 0,
        'n_fracs': len(true_depths_m),
        'mean_error_m': np.nan,
        'max_error_m': np.nan,
        'snr': np.nan,
        'match_tol_m': np.nan,
        'profile_depth': depth_kept,
        'profile': profile,
        'all_peak_depths': [],
    }


def evaluate_profile_fracture_peaks(
    depth: np.ndarray,
    profile: np.ndarray,
    true_depths_m: List[float],
    fs: float,
    v: float,
    depth_min: float = 100.0,
    depth_max: Optional[float] = None,
) -> Dict:
    """在 1D 深度剖面（已为 -C 响应）上检测峰并与真实缝深匹配。"""
    depth = np.asarray(depth, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if depth_max is None:
        depth_max = float(depth[-1]) if len(depth) else 5000.0
    mask = (depth >= depth_min) & (depth <= depth_max)
    depth_kept = depth[mask]
    profile_kept = profile[mask]

    n_fracs = len(true_depths_m)
    if len(profile_kept) < 10:
        return _empty_fracture_peak_metrics(depth_kept, profile_kept, true_depths_m)

    match_tol_m, peak_distance, top_n = peak_find_params(true_depths_m, v, fs)
    height_thresh = max(float(np.percentile(profile_kept, 88)), 1e-6)
    peaks, props = signal.find_peaks(
        profile_kept, height=height_thresh, distance=peak_distance,
    )

    if len(peaks) == 0:
        i_max = int(np.argmax(profile_kept))
        peaks = np.array([i_max])
        props = {'peak_heights': np.array([profile_kept[i_max]])}

    order = np.argsort(props['peak_heights'])[::-1]
    top_peaks = peaks[order[:top_n]]
    peak_depths = depth_kept[top_peaks]
    peak_heights = profile_kept[top_peaks]

    matches = []
    used_peak_idx = set()
    for i, true_d in enumerate(true_depths_m):
        best_j, best_err = None, np.inf
        for j, pd in enumerate(peak_depths):
            if j in used_peak_idx:
                continue
            err = abs(pd - true_d)
            if err < best_err:
                best_err = err
                best_j = j
        if best_j is not None and best_err <= match_tol_m:
            used_peak_idx.add(best_j)
            matches.append({
                'frac_id': i + 1,
                'true_depth_m': float(true_d),
                'peak_depth_m': float(peak_depths[best_j]),
                'peak_val': float(peak_heights[best_j]),
                'error_m': float(best_err),
                'matched': True,
            })
        else:
            matches.append({
                'frac_id': i + 1,
                'true_depth_m': float(true_d),
                'peak_depth_m': np.nan,
                'peak_val': np.nan,
                'error_m': np.nan,
                'matched': False,
            })

    matched_errs = [m['error_m'] for m in matches if m['matched']]
    mean_err = float(np.mean(matched_errs)) if matched_errs else np.nan
    max_err = float(np.max(matched_errs)) if matched_errs else np.nan
    n_matched = sum(1 for m in matches if m['matched'])
    bg = np.percentile(profile_kept, 50)
    snr = float(np.max(profile_kept) / max(abs(bg), 1e-12))

    return {
        'matches': matches,
        'n_matched': n_matched,
        'n_fracs': n_fracs,
        'mean_error_m': mean_err,
        'max_error_m': max_err,
        'snr': snr,
        'match_tol_m': match_tol_m,
        'profile_depth': depth_kept,
        'profile': profile_kept,
        'all_peak_depths': peak_depths.tolist(),
    }


def evaluate_multi_fracture_peaks(
    C: np.ndarray,
    q: np.ndarray,
    v: float,
    true_depths_m: List[float],
    fs: float,
    depth_min: float = 100.0,
    depth_max: Optional[float] = None,
) -> Dict:
    """2D 倒谱沿时间平均后，在 1D 深度剖面上匹配裂缝峰。"""
    depth_kept, profile = compute_time_avg_depth_profile(
        C, q, v, depth_min=depth_min, depth_max=depth_max,
    )
    return evaluate_profile_fracture_peaks(
        depth_kept, profile, true_depths_m, fs, v,
        depth_min=depth_min, depth_max=depth_max,
    )


def cepstrum_metrics_for_json(metrics: Dict) -> Dict:
    """提取可 JSON 序列化的倒谱缝深匹配摘要。"""

    def _json_float(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x) or np.isinf(x):
                return None
            return float(x)
        return x

    matches = []
    for mt in metrics['matches']:
        row = {}
        for k, v in mt.items():
            if k in ('peak_depth_m', 'peak_val', 'error_m', 'true_depth_m'):
                row[k] = _json_float(v)
            else:
                row[k] = v
        matches.append(row)

    return {
        'n_matched': int(metrics['n_matched']),
        'n_fracs': int(metrics['n_fracs']),
        'mean_error_m': _json_float(metrics['mean_error_m']),
        'max_error_m': _json_float(metrics['max_error_m']),
        'snr': _json_float(metrics.get('snr')),
        'match_tol_m': _json_float(metrics.get('match_tol_m')),
        'matches': matches,
    }


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


AR_CEPSTRUM_METHOD_NAME = '方案4 AR倒谱'


def is_ar_cepstrum_method(name: str) -> bool:
    return name == AR_CEPSTRUM_METHOD_NAME or 'AR' in name


def cepstrum_display_data(
    C: np.ndarray,
    depth: np.ndarray,
    depth_min: float = 0.0,
    depth_max: float = 5000.0,
) -> np.ndarray:
    """倒谱热力图显示矩阵：深度裁剪后的 -C。"""
    mask = (depth >= depth_min) & (depth <= depth_max)
    if not np.any(mask):
        return -C
    return -C[mask, :]


def compute_panel_vrange(
    C: np.ndarray,
    depth: np.ndarray,
    depth_max: float,
    depth_min: float = 0.0,
    vmin_pct: float = 2.0,
    vmax_pct: float = 98.0,
) -> Tuple[float, float]:
    data = cepstrum_display_data(C, depth, depth_min, depth_max)
    vmin = float(np.percentile(data, vmin_pct))
    vmax = float(np.percentile(data, vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def compute_shared_vrange(
    panels: List[Tuple[np.ndarray, np.ndarray]],
    depth_max: float,
    depth_min: float = 0.0,
    vmin_pct: float = 2.0,
    vmax_pct: float = 98.0,
) -> Tuple[float, float]:
    """多 panel 共用色标：在全部 panel 的显示数据上取联合分位数。"""
    chunks = [
        cepstrum_display_data(C, depth, depth_min, depth_max).ravel()
        for depth, C in panels
    ]
    if not chunks:
        return 0.0, 1.0
    pooled = np.concatenate(chunks)
    vmin = float(np.percentile(pooled, vmin_pct))
    vmax = float(np.percentile(pooled, vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _fracture_mark_t_span(mark_t_max: float = 10.0) -> Tuple[float, float]:
    """裂缝标注使用绝对仿真时间 [0, mark_t_max]（与停泵后反射到达时段对齐）。"""
    return 0.0, mark_t_max


def _plot_fracture_depth_marks(
    ax,
    fracture_depths,
    mark_t_max: float = 10.0,
) -> None:
    """在 2D 倒谱热力图上用 0~mark_t_max 秒的黑色实线标注裂缝深度。"""
    if fracture_depths is None:
        return
    if isinstance(fracture_depths, (int, float, np.floating)):
        depths = [float(fracture_depths)]
    else:
        depths = [float(d) for d in fracture_depths]
    t0, t1 = _fracture_mark_t_span(mark_t_max)
    for fd in depths:
        ax.plot(
            [t0, t1], [fd, fd],
            color='k', ls='-', lw=1.8, solid_capstyle='butt',
            alpha=1.0, zorder=10, clip_on=True,
        )


def plot_2d_panel_single(
    ax,
    t: np.ndarray,
    depth: np.ndarray,
    C: np.ndarray,
    title: str,
    fracture_depth: float,
    depth_min: float = 0.0,
    depth_max: float = 5000.0,
    mark_t_max: float = 10.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    mask = (depth >= depth_min) & (depth <= depth_max)
    depth_plot = depth[mask]
    C_plot = C[mask, :]
    T, D = np.meshgrid(t, depth_plot)
    data = -C_plot
    if vmin is None or vmax is None:
        vmin, vmax = compute_panel_vrange(C, depth, depth_max, depth_min)
    im = ax.pcolormesh(T, D, data, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    _plot_fracture_depth_marks(ax, fracture_depth, mark_t_max=mark_t_max)
    ax.invert_yaxis()
    ax.set_ylim([depth_max, depth_min])
    t_right = float(t[-1]) if len(t) else mark_t_max
    ax.set_xlim(0.0, max(t_right, mark_t_max))
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
    fracture_depths: Optional[list],
    depth_max: float,
    mark_t_max: float = 10.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    mask = (depth >= 0) & (depth <= depth_max)
    depth_plot = depth[mask]
    C_plot = C[mask, :]
    T, D = np.meshgrid(t, depth_plot)
    data = -C_plot
    if vmin is None or vmax is None:
        vmin, vmax = compute_panel_vrange(C, depth, depth_max, depth_min=0.0)
    im = ax.pcolormesh(T, D, data, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    _plot_fracture_depth_marks(ax, fracture_depths, mark_t_max=mark_t_max)
    ax.invert_yaxis()
    ax.set_ylim([depth_max, 0])
    t_right = float(t[-1]) if len(t) else mark_t_max
    ax.set_xlim(0.0, max(t_right, mark_t_max))
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
