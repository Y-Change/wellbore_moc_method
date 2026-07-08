# -*- coding: utf-8 -*-
"""
MOC 仿真井口水头倒谱分析 — 供 validate_moc_*.py 可视化调用

提供两种倒谱分析（合并输出同一张大图）：
  · 1D 实倒谱 (real cepstrum) — 全序列 FFT，无滑窗
  · 2D 倒谱图 (cepstrogram)   — 短时滑窗，参考 cepstrum_2d.py
  · plot_moc_cepstrum_analysis — 时域 / FFT / 1D + 2D 四联图
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def cepstrogram(x, wlen, hop, fs, win_type='rect'):
    """短时倒谱图（内联，无外部依赖）。"""
    x = np.asarray(x).flatten()
    xlen = len(x)
    if wlen > xlen:
        raise ValueError('窗口长度大于信号长度')

    if win_type == 'kaiser':
        win = np.kaiser(wlen, 4)
    elif win_type == 'hamming':
        win = scipy_signal.windows.hamming(wlen, sym=False)
    elif win_type == 'hanning':
        win = scipy_signal.windows.hann(wlen, sym=False)
    elif win_type == 'rect':
        win = np.ones(wlen)
    elif win_type == 'gauss':
        win = scipy_signal.windows.gaussian(wlen, std=wlen / 6)
    else:
        win = np.kaiser(wlen, 4)

    noverlap = wlen - hop
    nFrames = (xlen - noverlap) // hop
    shape = (wlen, nFrames)
    strides = (x.strides[0], hop * x.strides[0])
    X_frames = as_strided(x, shape=shape, strides=strides)
    X_frames = X_frames - np.mean(X_frames, axis=0, keepdims=True)
    X_windowed = X_frames * win[:, np.newaxis]

    spec = fft(X_windowed, axis=0)
    log_spec = np.log(np.abs(spec) + np.finfo(float).eps)
    raw_ceps = np.real(ifft(log_spec, axis=0))
    rown = int(np.ceil((1 + wlen) / 2))
    ceps = raw_ceps[:rown, :]
    q = np.arange(rown) / fs
    t = (wlen / 2 + np.arange(nFrames) * hop) / fs
    return ceps, q, t


# ── 参数解析（与 benchmark/methods/common.py 一致）────────────────

def _quefrency_to_depth(q: np.ndarray, v: float) -> np.ndarray:
    return q * v / 2.0


def _resolve_cepstrum_params(
    depth_max: float,
    v: float,
    wlen_sec: Optional[float] = None,
    lim2: Optional[float] = None,
    lim1: float = 0.0,
    fs: float = 1000.0,
    signal_len: Optional[int] = None,
) -> Dict:
    if depth_max <= 0 or v <= 0:
        raise ValueError(f"depth_max 与 v 须为正数，收到 depth_max={depth_max}, v={v}")

    wlen_min = 4.0 * depth_max / v
    lim2_min = 2.0 * depth_max / v

    if wlen_sec is not None:
        wlen_used = float(wlen_sec)
        wlen_requested = float(wlen_sec)
    else:
        wlen_used = wlen_min
        wlen_requested = wlen_min

    wlen_auto = wlen_sec is None or wlen_used < wlen_min - 1e-12
    if wlen_used < wlen_min:
        wlen_used = wlen_min

    if lim2 is not None:
        lim2_used = float(lim2)
    else:
        lim2_used = lim2_min + 0.1
    lim2_auto = lim2 is None
    if lim2_used < lim2_min:
        lim2_used = lim2_min + 0.1

    capped_by_signal = False
    if signal_len is not None and signal_len > 0:
        wlen_max_sec = signal_len / fs
        if wlen_used > wlen_max_sec:
            wlen_used = wlen_max_sec
            capped_by_signal = True

    rown = int(np.ceil((1 + max(2, int(wlen_used * fs))) / 2))
    q_max = (rown - 1) / fs
    depth_coverage_max = float(q_max * v / 2.0)

    return {
        'wlen_sec': wlen_used,
        'wlen_sec_requested': wlen_requested,
        'wlen_sec_min': wlen_min,
        'wlen_auto_adjusted': wlen_auto,
        'lim1': float(lim1),
        'lim2': lim2_used,
        'lim2_min': lim2_min,
        'lim2_auto': lim2_auto,
        'capped_by_signal': capped_by_signal,
        'depth_coverage_max': depth_coverage_max,
    }


def preprocess_moc_head(
    time: np.ndarray,
    head: np.ndarray,
    fs: float = 1000.0,
    ts: float = 1.0,
) -> Dict:
    """MOC 井口水头预处理：重采样 → 停泵后截取 → 去趋势。"""
    time = np.asarray(time, dtype=float).flatten()
    head = np.asarray(head, dtype=float).flatten()

    t_new = np.arange(0.0, float(time[-1]), 1.0 / fs)
    interp = interp1d(time, head, kind='cubic',
                      bounds_error=False, fill_value='extrapolate')
    h_resampled = interp(t_new)

    ts_idx = int(np.searchsorted(t_new, ts))
    if ts_idx >= len(t_new):
        ts_idx = len(t_new) // 2

    t_after = t_new[ts_idx:]
    h_after = h_resampled[ts_idx:]
    h_detrended = h_after - np.mean(h_after)

    return {
        't_after': t_after,
        'h_after': h_after,
        'h_detrended': h_detrended,
        'fs': fs,
        'ts': ts,
    }


def apply_derivative_preprocess(
    signal: np.ndarray,
    fs: float,
    order: int = 1,
) -> np.ndarray:
    """倒谱前可选求导预处理：突出反射边沿，抑制缓变成分。

    使用 np.gradient 保持与重采样序列等长；求导后再去均值。
    """
    if order <= 0:
        return np.asarray(signal, dtype=float).flatten()
    y = np.asarray(signal, dtype=float).flatten()
    dt = 1.0 / fs
    for _ in range(order):
        y = np.gradient(y, dt)
    return y - np.mean(y)


def prepare_cepstrum_signal(
    pre: Dict,
    derivative: bool = False,
    derivative_order: int = 1,
) -> np.ndarray:
    """从预处理结果得到送入倒谱的工作信号。"""
    h = pre['h_detrended']
    if not derivative:
        return h
    return apply_derivative_preprocess(h, pre['fs'], order=derivative_order)


def _resolve_quefrency_limits(
    depth_max: float,
    v: float,
    lim2: Optional[float] = None,
    lim1: float = 0.0,
) -> Dict:
    """解析 1D 倒谱的倒频率裁剪区间 (lim1, lim2)。"""
    lim2_min = 2.0 * depth_max / v
    if lim2 is not None:
        lim2_used = float(lim2)
    else:
        lim2_used = lim2_min + 0.1
    if lim2_used < lim2_min:
        lim2_used = lim2_min + 0.1
    return {
        'lim1': float(lim1),
        'lim2': lim2_used,
        'lim2_min': lim2_min,
    }


def real_cepstrum_1d(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """全序列实倒谱（无滑窗）。

    c(q) = Re{ IFFT( log|FFT(x)| ) }，仅保留正倒频率半轴。
    与 cepstrogram 单帧（wlen=信号全长）等价，但不分窗。
    """
    x = np.asarray(x, dtype=float).flatten()
    x = x - np.mean(x)
    n = len(x)
    spec = fft(x)
    log_spec = np.log(np.abs(spec) + np.finfo(float).eps)
    raw_ceps = np.real(ifft(log_spec))
    rown = n // 2 + 1
    q = np.arange(rown) / fs
    return raw_ceps[:rown], q


def compute_moc_cepstrum_1d(
    time: np.ndarray,
    head: np.ndarray,
    v: float,
    fs: Optional[float] = None,
    dt: Optional[float] = None,
    ts: float = 1.0,
    wellbore_length: float = 5000.0,
    lim1: float = 0.0,
    lim2: Optional[float] = None,
    depth_min: float = 0.0,
    lifter: bool = False,
    lifter_cutoff: Optional[float] = None,
    derivative: bool = False,
    derivative_order: int = 1,
) -> Dict:
    """计算 MOC 井口水头 1D 实倒谱（全序列，无滑窗）。

    参数
    ----
    derivative : 倒谱前是否对去趋势信号求导
    derivative_order : 求导阶数（1 或 2）

    返回
    ----
    dict: pre, C, q, depth, response, lim1, lim2, v, fs, wellbore_length
    """
    if fs is None:
        fs = 1.0 / dt if dt is not None else 1000.0

    pre = preprocess_moc_head(time, head, fs=fs, ts=ts)
    p_work = prepare_cepstrum_signal(
        pre, derivative=derivative, derivative_order=derivative_order,
    )

    lim_params = _resolve_quefrency_limits(wellbore_length, v, lim2=lim2, lim1=lim1)
    lim1_used = lim_params['lim1']
    lim2_used = lim_params['lim2']

    C, q = real_cepstrum_1d(p_work, fs)

    if lifter:
        if lifter_cutoff is None:
            lifter_cutoff = wellbore_length * 2.0 / v * 0.5
        lifter_win = np.ones(len(q))
        lifter_win[q < lifter_cutoff] = 0.0
        transition = (q >= lifter_cutoff) & (q < lifter_cutoff * 1.5)
        if np.any(transition):
            t_norm = (q[transition] - lifter_cutoff) / (lifter_cutoff * 0.5)
            lifter_win[transition] = 0.5 * (1 - np.cos(np.pi * t_norm))
        C = C * lifter_win

    valid = (q > lim1_used) & (q < lim2_used)
    q_kept = q[valid]
    C_kept = C[valid]
    depth = _quefrency_to_depth(q_kept, v)
    response = -C_kept

    q_max = float(q[-1]) if len(q) else 0.0
    depth_coverage_max = _quefrency_to_depth(np.array([q_max]), v)[0]

    return {
        'pre': pre,
        'C': C_kept,
        'q': q_kept,
        'depth': depth,
        'response': response,
        'lim_params': lim_params,
        'v': v,
        'fs': fs,
        'lim1': lim1_used,
        'lim2': lim2_used,
        'depth_min': depth_min,
        'wellbore_length': wellbore_length,
        'depth_coverage_max': depth_coverage_max,
        'signal_duration': len(p_work) / fs,
        'derivative': derivative,
        'derivative_order': derivative_order if derivative else 0,
        'p_work': p_work,
    }


def compute_moc_cepstrum(
    time: np.ndarray,
    head: np.ndarray,
    v: float,
    fs: Optional[float] = None,
    dt: Optional[float] = None,
    ts: float = 1.0,
    wellbore_length: float = 5000.0,
    wlen_sec: Optional[float] = None,
    hop_sec: float = 0.1,
    lim1: float = 0.0,
    lim2: Optional[float] = None,
    depth_min: float = 0.0,
    lifter: bool = False,
    lifter_cutoff: Optional[float] = None,
    derivative: bool = False,
    derivative_order: int = 1,
) -> Dict:
    """计算 MOC 井口水头的 2D 倒谱图（参考 cepstrum_2d.compute_cepstrum_1d）。

    参数
    ----
    derivative : 倒谱前是否对去趋势信号求导
    derivative_order : 求导阶数（1 或 2）

    返回
    ----
    dict: pre, C, q, t_cep, depth, response_2d, cep_params, v, fs
    """
    if fs is None:
        fs = 1.0 / dt if dt is not None else 1000.0

    pre = preprocess_moc_head(time, head, fs=fs, ts=ts)
    p_work = prepare_cepstrum_signal(
        pre, derivative=derivative, derivative_order=derivative_order,
    )

    cep_params = _resolve_cepstrum_params(
        wellbore_length, v,
        wlen_sec=wlen_sec, lim2=lim2, lim1=lim1,
        fs=fs, signal_len=len(p_work),
    )
    wlen_sec_used = cep_params['wlen_sec']
    lim1_used = cep_params['lim1']
    lim2_used = cep_params['lim2']

    wlen = int(wlen_sec_used * fs)
    hop = max(1, int(hop_sec * fs))
    if wlen > len(p_work):
        raise ValueError(
            f"窗长 {wlen} > 信号长度 {len(p_work)} "
            f"(需要 wlen_sec≥{cep_params['wlen_sec_min']:.3f}s 以覆盖 {wellbore_length}m)"
        )

    C, q, t_cep = cepstrogram(p_work, wlen=wlen, hop=hop, fs=fs)

    if lifter:
        if lifter_cutoff is None:
            lifter_cutoff = wellbore_length * 2.0 / v * 0.5
        lifter_win = np.ones(len(q))
        lifter_win[q < lifter_cutoff] = 0.0
        transition = (q >= lifter_cutoff) & (q < lifter_cutoff * 1.5)
        if np.any(transition):
            t_norm = (q[transition] - lifter_cutoff) / (lifter_cutoff * 0.5)
            lifter_win[transition] = 0.5 * (1 - np.cos(np.pi * t_norm))
        C = C * lifter_win[:, np.newaxis]

    valid = (q > lim1_used) & (q < lim2_used)
    q_kept = q[valid]
    C_kept = C[valid, :]
    depth = _quefrency_to_depth(q_kept, v)
    response_2d = -C_kept

    return {
        'pre': pre,
        'C': C_kept,
        'q': q_kept,
        't_cep': t_cep,
        'depth': depth,
        'response_2d': response_2d,
        'cep_params': cep_params,
        'v': v,
        'fs': fs,
        'wlen_sec': wlen_sec_used,
        'hop_sec': hop_sec,
        'lim1': lim1_used,
        'lim2': lim2_used,
        'depth_min': depth_min,
        'wellbore_length': wellbore_length,
    }


def effective_fft_fmax(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    wavespeed: float,
    wellbore_length: float,
    energy_fraction: float = 0.995,
    peak_threshold: float = 0.02,
    harmonic_margin: int = 12,
    fmax_cap: Optional[float] = None,
) -> float:
    """根据频谱能量与物理往返周期，确定 FFT 有效显示上限 (Hz)。"""
    nyquist = float(frequencies[-1]) if len(frequencies) else 500.0
    if len(frequencies) < 2 or wellbore_length <= 0 or wavespeed <= 0:
        return min(10.0, nyquist)

    freqs = frequencies[1:]
    mag = magnitude[1:]
    if len(mag) == 0 or np.max(mag) <= 0:
        return min(10.0, nyquist)

    power = mag ** 2
    total = power.sum()
    if total > 0:
        idx_e = int(np.searchsorted(np.cumsum(power), energy_fraction * total))
        f_energy = float(freqs[min(idx_e, len(freqs) - 1)])
    else:
        f_energy = float(freqs[-1])

    peak = float(np.max(mag))
    above = freqs[mag >= peak * peak_threshold]
    f_peak = float(above[-1]) if len(above) else f_energy

    # 往返周期基频 f0 = a/(2L)，保留若干谐波作为物理下限
    f0 = wavespeed / (2.0 * wellbore_length)
    f_phys = harmonic_margin * f0

    f_max = max(f_energy, f_peak, f_phys)
    if fmax_cap is not None:
        f_max = min(f_max, fmax_cap)

    return float(min(f_max * 1.05, nyquist))


def _plot_fft_panel(
    ax,
    h_detrended: np.ndarray,
    fs: float,
    wavespeed: float,
    wellbore_length: float,
    fft_fmax: Optional[float],
) -> float:
    """绘制 FFT 频域子图，返回实际显示上限。"""
    n = len(h_detrended)
    frequencies = fftfreq(n, d=1.0 / fs)[:n // 2]
    magnitude = np.abs(fft(h_detrended)[:n // 2]) / n
    if fft_fmax is None:
        f_display_max = effective_fft_fmax(
            frequencies, magnitude, wavespeed, wellbore_length,
        )
    else:
        f_display_max = min(fft_fmax, fs / 2.0)
    mask_f = frequencies <= f_display_max
    ax.plot(frequencies[mask_f], magnitude[mask_f], 'b-', lw=1.2, label='FFT 幅值')
    f0 = wavespeed / (2.0 * wellbore_length)
    ax.axvline(f0, color='orange', ls=':', lw=1.0, alpha=0.8,
               label=f'往返基频 $f_0$={f0:.3f} Hz')
    ax.set_xlabel('频率 [Hz]')
    ax.set_ylabel('幅值 [m]')
    ax.set_title(f'井口水头频域曲线 (FFT) | 有效范围 0–{f_display_max:.2f} Hz')
    ax.set_xlim([0, f_display_max])
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(fontsize=9)
    return f_display_max


def _fracture_match_tol_m(
    fracture_depths_m: List[float],
    v: float,
    fs: float = 1000.0,
) -> float:
    """缝深匹配容差 [m]（与 validation/cepstrum/_kb_core.peak_find_params 一致）。"""
    sorted_d = sorted(fracture_depths_m)
    min_spacing = float(min(np.diff(sorted_d))) if len(sorted_d) > 1 else 300.0
    return float(np.clip(min_spacing * 0.45, 80.0, 250.0))


def detect_1d_cepstrum_peaks(
    depth: np.ndarray,
    response: np.ndarray,
) -> List[Dict]:
    """1D 实倒谱峰检测（与 plot_moc_cepstrum_analysis 第 3 子图同一套参数）。"""
    depth_arr = np.asarray(depth, dtype=float)
    resp_arr = np.asarray(response, dtype=float)
    if len(resp_arr) <= 20:
        return []

    peak_height_thresh = max(float(np.percentile(resp_arr, 95)), 0.01)
    distance = max(3, len(resp_arr) // 200)
    peaks, props = scipy_signal.find_peaks(
        resp_arr,
        height=peak_height_thresh,
        distance=distance,
    )
    if len(peaks) == 0:
        return []

    top_n = min(15, len(peaks))
    top_peaks = peaks[np.argsort(props['peak_heights'])[-top_n:]]
    order = top_peaks[np.argsort(resp_arr[top_peaks])[::-1]]
    return [
        {
            'depth_m': float(depth_arr[pi]),
            'response': float(resp_arr[pi]),
            'rank': rank,
        }
        for rank, pi in enumerate(order, start=1)
    ]


def evaluate_1d_cepstrum_fracture_match(
    depth: np.ndarray,
    response: np.ndarray,
    fracture_nominal_m: List[float],
    v: float,
    fs: float = 1000.0,
) -> Dict:
    """将 1D 实倒谱检测峰与名义缝深匹配，输出 JSON 友好摘要。"""
    detected = detect_1d_cepstrum_peaks(depth, response)
    n_fracs = len(fracture_nominal_m)
    match_tol_m = _fracture_match_tol_m(fracture_nominal_m, v, fs)

    if not detected:
        matches = [{
            'frac_id': i + 1,
            'true_depth_m': float(d),
            'peak_depth_m': None,
            'peak_val': None,
            'error_m': None,
            'matched': False,
        } for i, d in enumerate(fracture_nominal_m)]
        return {
            'detected_peaks': [],
            'matches': matches,
            'n_matched': 0,
            'n_fracs': n_fracs,
            'mean_error_m': None,
            'max_error_m': None,
            'snr': None,
            'match_tol_m': match_tol_m,
        }

    peak_depths = np.array([p['depth_m'] for p in detected], dtype=float)
    peak_heights = np.array([p['response'] for p in detected], dtype=float)
    resp_arr = np.asarray(response, dtype=float)
    bg = float(np.percentile(resp_arr, 50))
    snr = float(np.max(resp_arr) / max(abs(bg), 1e-12))

    matches = []
    used = set()
    for i, true_d in enumerate(fracture_nominal_m):
        best_j, best_err = None, np.inf
        for j, pd in enumerate(peak_depths):
            if j in used:
                continue
            err = abs(float(pd) - float(true_d))
            if err < best_err:
                best_err = err
                best_j = j
        if best_j is not None and best_err <= match_tol_m:
            used.add(best_j)
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
                'peak_depth_m': None,
                'peak_val': None,
                'error_m': None,
                'matched': False,
            })

    matched_errs = [m['error_m'] for m in matches if m['matched']]
    mean_err = float(np.mean(matched_errs)) if matched_errs else None
    max_err = float(np.max(matched_errs)) if matched_errs else None
    n_matched = sum(1 for m in matches if m['matched'])

    return {
        'detected_peaks': detected,
        'matches': matches,
        'n_matched': n_matched,
        'n_fracs': n_fracs,
        'mean_error_m': mean_err,
        'max_error_m': max_err,
        'snr': snr,
        'match_tol_m': match_tol_m,
    }


def cepstrum_match_summary_for_json(metrics: Dict) -> Dict:
    """提取 1d_real 倒谱匹配 JSON 字段。"""

    def _json_float(x):
        if x is None:
            return None
        if isinstance(x, (float, np.floating)):
            if np.isnan(x) or np.isinf(x):
                return None
            return float(x)
        return x

    return {
        'n_matched': int(metrics['n_matched']),
        'n_fracs': int(metrics['n_fracs']),
        'mean_error_m': _json_float(metrics['mean_error_m']),
        'max_error_m': _json_float(metrics['max_error_m']),
        'snr': _json_float(metrics.get('snr')),
        'match_tol_m': _json_float(metrics.get('match_tol_m')),
        'detected_peaks': metrics.get('detected_peaks', []),
        'matches': [
            {k: _json_float(v) if k in (
                'true_depth_m', 'peak_depth_m', 'peak_val', 'error_m',
            ) else v for k, v in mt.items()}
            for mt in metrics['matches']
        ],
    }


def _robust_response_ylim(
    response: np.ndarray,
    pct: tuple[float, float] = (1.0, 99.0),
    margin: float = 0.12,
) -> tuple[float, float]:
    """1D 倒谱 y 轴稳健范围：基于峰值高度，剔除孤立离群尖峰。"""
    from scipy.signal import find_peaks

    r = np.asarray(response, dtype=float)
    if len(r) == 0:
        return 0.0, 1.0

    height_thresh = max(float(np.percentile(r, 85)), 1e-12)
    distance = max(3, len(r) // 200)
    peaks, props = find_peaks(r, height=height_thresh, distance=distance)

    if len(peaks) > 0:
        ph = np.sort(props['peak_heights'])
        lo = float(min(np.percentile(r, pct[0]), np.min(ph)))
        hi = float(max(ph[-1], np.max(r)))
    else:
        lo = float(np.percentile(r, pct[0]))
        hi = float(np.percentile(r, pct[1]))
        hi = max(hi, float(np.max(r)))

    if hi <= lo:
        span = max(abs(hi), abs(lo), 1e-9)
        lo, hi = lo - 0.5 * span, hi + 0.5 * span
    pad = margin * (hi - lo)
    return lo - pad, hi + pad


def _mark_fractures_on_depth_axis(
    ax,
    fracture_positions: List[float],
    orientation: str = 'horizontal',
) -> None:
    """在深度轴上标注裂缝参考位置。"""
    for i, depth in enumerate(fracture_positions):
        if orientation == 'horizontal':
            ax.axvline(depth, color='r', ls='--', lw=1.2, alpha=0.85)
            ax.text(depth, ax.get_ylim()[1], f' Frac{i + 1}: {depth:.0f}m',
                    color='r', fontsize=9, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.6, pad=1))
        else:
            ax.axhline(depth, color='white', ls='--', lw=1.5, alpha=0.85)
            ax.text(ax.get_xlim()[0], depth, f' Frac{i + 1}: {depth:.0f}m',
                    color='white', fontsize=9, va='center',
                    bbox=dict(facecolor='black', alpha=0.4, pad=1))


def _assign_fracture_label_levels(
    depths: List[float],
    x_span: float,
    min_gap_frac: float = 0.055,
    min_gap_abs: float = 80.0,
) -> List[int]:
    """为横轴下方标签分配纵向错开层级，避免相邻裂缝标注互相遮挡。"""
    if not depths:
        return []
    min_gap = max(min_gap_frac * x_span, min_gap_abs)
    order = sorted(range(len(depths)), key=lambda i: depths[i])
    level_slots: List[List[float]] = []
    levels = [0] * len(depths)
    for i in order:
        d = depths[i]
        for lev, slots in enumerate(level_slots):
            if all(abs(d - sd) >= min_gap for sd in slots):
                slots.append(d)
                levels[i] = lev
                break
        else:
            level_slots.append([d])
            levels[i] = len(level_slots) - 1
    return levels


def _mark_fractures_below_1d_axis(
    ax,
    fracture_positions: List[float],
) -> None:
    """在 1D 实倒谱横轴下方用箭头标注裂缝深度（无图内竖线）。"""
    if not fracture_positions:
        return

    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    xlim = ax.get_xlim()
    x_span = xlim[1] - xlim[0]
    depths = list(fracture_positions)
    levels = _assign_fracture_label_levels(depths, x_span)

    base_y = -0.09
    level_dy = 0.065
    y_tip = 0.012

    for i, depth in enumerate(depths):
        y_label = base_y - levels[i] * level_dy
        ax.annotate(
            f'{depth:.0f}m',
            xy=(depth, y_tip),
            xycoords=trans,
            xytext=(depth, y_label),
            textcoords=trans,
            fontsize=8,
            color='darkred',
            ha='center',
            va='top',
            arrowprops=dict(
                arrowstyle='->',
                color='darkred',
                lw=1.2,
                shrinkA=0,
                shrinkB=1,
            ),
            clip_on=False,
            zorder=10,
        )


def _mark_fractures_side_arrows(
    ax,
    fracture_positions: List[float],
    side: str = 'right',
) -> None:
    """在 2D 倒谱图侧边用箭头标注裂缝深度。"""
    if not fracture_positions:
        return

    from matplotlib.transforms import blended_transform_factory

    if side == 'right':
        x_point, x_text, ha = 1.0, 1.04, 'left'
    else:
        x_point, x_text, ha = 0.0, -0.04, 'right'

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, depth in enumerate(fracture_positions):
        ax.annotate(
            f'Frac{i + 1}: {depth:.0f}m',
            xy=(x_point, depth),
            xycoords=trans,
            xytext=(x_text, depth),
            textcoords=trans,
            fontsize=9,
            color='yellow',
            va='center',
            ha=ha,
            arrowprops=dict(
                arrowstyle='->',
                color='yellow',
                lw=1.5,
                shrinkA=2,
                shrinkB=2,
            ),
            bbox=dict(facecolor='black', alpha=0.55, pad=2, edgecolor='none'),
            clip_on=False,
            zorder=10,
        )


def plot_moc_cepstrum_1d_analysis(
    time: np.ndarray,
    head: np.ndarray,
    wavespeed: float,
    ts: float = 1.0,
    dt: Optional[float] = None,
    fs: Optional[float] = None,
    wellbore_length: float = 5000.0,
    fracture_positions: Optional[List[float]] = None,
    save_path: Optional[Union[str, os.PathLike]] = None,
    title_prefix: str = 'MOC 井口水头 1D 倒谱分析',
    lim1: float = 0.0,
    lim2: Optional[float] = None,
    depth_min: float = 0.0,
    lifter: bool = False,
    fft_fmax: Optional[float] = None,
    show: bool = False,
    **kwargs,
) -> Dict:
    """兼容入口：与 plot_moc_cepstrum_analysis 输出同一张合并图。"""
    return plot_moc_cepstrum_analysis(
        time, head,
        wavespeed=wavespeed, ts=ts, dt=dt, fs=fs,
        wellbore_length=wellbore_length,
        fracture_positions=fracture_positions,
        save_path=save_path, title_prefix=title_prefix,
        lim1=lim1, lim2=lim2, depth_min=depth_min,
        lifter=lifter, fft_fmax=fft_fmax, show=show,
        **kwargs,
    )


def plot_moc_cepstrum_analysis(
    time: np.ndarray,
    head: np.ndarray,
    wavespeed: float,
    ts: float = 1.0,
    dt: Optional[float] = None,
    fs: Optional[float] = None,
    wellbore_length: float = 5000.0,
    fracture_positions: Optional[List[float]] = None,
    save_path: Optional[Union[str, os.PathLike]] = None,
    title_prefix: str = 'MOC 井口水头倒谱分析',
    wlen_sec: Optional[float] = None,
    hop_sec: float = 0.1,
    lim1: float = 0.0,
    lim2: Optional[float] = None,
    depth_min: float = 0.0,
    lifter: bool = True,
    lifter_cutoff: Optional[float] = None,
    derivative: bool = False,
    derivative_order: int = 1,
    fft_fmax: Optional[float] = None,
    show: bool = False,
    **kwargs,
) -> Dict:
    """绘制 MOC 井口水头倒谱合并图：时域 / FFT / 1D 实倒谱 + 2D 倒谱图。

    参数
    ----
    time, head : MOC 仿真时间序列与井口水头 [m]
    wavespeed : 波速 [m/s]
    fracture_positions : 真实裂缝深度 [m]，在倒谱图上叠加参考线
    save_path : 保存路径；None 则不保存
    fft_fmax : FFT 显示上限 (Hz)；None 时按频谱能量自动裁剪有效范围
    """
    if fs is None:
        fs = 1.0 / dt if dt is not None else 1000.0

    # Default lifter cutoff: suppress depth < 100m (remove DC without affecting fracture detection)
    if lifter and lifter_cutoff is None:
        lifter_cutoff = 2.0 * 100.0 / wavespeed   # quefrency for 100m depth

    result_1d = compute_moc_cepstrum_1d(
        time, head,
        v=wavespeed, fs=fs, ts=ts,
        wellbore_length=wellbore_length,
        lim1=lim1, lim2=lim2, depth_min=depth_min,
        lifter=lifter, lifter_cutoff=lifter_cutoff,
        derivative=derivative, derivative_order=derivative_order,
    )
    result_2d = compute_moc_cepstrum(
        time, head,
        v=wavespeed, fs=fs, ts=ts,
        wellbore_length=wellbore_length,
        wlen_sec=wlen_sec, hop_sec=hop_sec,
        lim1=lim1, lim2=lim2, depth_min=depth_min,
        lifter=lifter, lifter_cutoff=lifter_cutoff,
        derivative=derivative, derivative_order=derivative_order,
    )

    pre = result_1d['pre']
    h_detrended = pre['h_detrended']
    v = result_1d['v']
    depth_1d = result_1d['depth']
    response_1d = result_1d['response']

    C = result_2d['C']
    q = result_2d['q']
    t_cep = result_2d['t_cep']
    cep_params = result_2d['cep_params']
    wlen = int(result_2d['wlen_sec'] * fs)

    depth_max_1d = min(wellbore_length, result_1d['depth_coverage_max'])
    depth_max_2d = min(wellbore_length, cep_params['depth_coverage_max'])
    depth_max_display = max(depth_max_1d, depth_max_2d)
    lim3 = depth_min * 2.0 / v
    lim4 = wellbore_length * 2.0 / v
    fracture_positions = fracture_positions or []

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        f'{title_prefix}\n'
        f'v={v:.1f} m/s, ts={ts}s, T={result_1d["signal_duration"]:.1f}s, '
        f'2D wlen={result_2d["wlen_sec"]:.3f}s ({wlen}点), hop={hop_sec}s, '
        f'深度 {depth_min:.0f}-{depth_max_display:.0f} m',
        fontsize=12, fontweight='bold',
    )

    # Row 1: 时域
    ax1 = plt.subplot(4, 1, 1)
    t_after = pre['t_after']
    h_after = pre['h_after']
    ax1.plot(t_after, h_after, 'r-', lw=0.8, alpha=0.6, label='停泵后井口水头')
    ax1.plot(t_after, h_detrended + np.mean(h_after), 'b-', lw=1.0,
             label='去趋势后 (上移对齐)')
    ax1.axvline(t_after[0], color='g', ls='--', lw=1.2, label=f'停泵 ts={ts}s')
    ax1.set_xlabel('时间 [s]')
    ax1.set_ylabel('水头 [m]')
    ax1.set_title(f'停泵后井口水头 (t ≥ {ts}s)')
    ax1.legend(fontsize=9)
    ax1.grid(True, ls='--', alpha=0.6)

    # Row 2: FFT
    ax2 = plt.subplot(4, 1, 2)
    fft_fmax_display = _plot_fft_panel(
        ax2, h_detrended, fs, wavespeed, wellbore_length, fft_fmax,
    )

    # Row 3: 1D 实倒谱
    ax3 = plt.subplot(4, 1, 3)
    depth_arr = np.asarray(depth_1d)
    resp_arr = np.asarray(response_1d)
    y_lo, y_hi = _robust_response_ylim(resp_arr)

    ax3.plot(depth_1d, response_1d, 'b-', lw=1.0, label='1D 倒谱响应 (-C)')
    ax3.set_ylim(y_lo, y_hi)

    detected_peaks = detect_1d_cepstrum_peaks(depth_arr, resp_arr)
    if detected_peaks:
        peak_depths = np.array([p['depth_m'] for p in detected_peaks])
        peak_heights = np.array([p['response'] for p in detected_peaks])
        ax3.scatter(peak_depths, peak_heights, c='r', s=50, zorder=5,
                    marker='v', label=f'检测峰 ({len(peak_depths)}个)')
        for pd, ph in zip(peak_depths, peak_heights):
            ax3.annotate(f'{pd:.0f}m', (pd, ph),
                         textcoords="offset points", xytext=(0, 8),
                         fontsize=7, color='r', ha='center',
                         clip_on=True)
    ax3.set_xlabel('深度 [m]')
    ax3.set_ylabel('倒谱能量 (-C)')
    deriv_label = f', deriv={derivative_order}' if derivative else ''
    ax3.set_title(
        f'1D 实倒谱（全序列，无滑窗）| lim=[{result_1d["lim1"]:.3f}, {result_1d["lim2"]:.3f}] s'
        f'{deriv_label} | lifter={lifter}',
    )
    ax3.set_xlim([depth_min, depth_max_1d])
    ax3.grid(True, ls='--', alpha=0.6)
    ax3.legend(fontsize=9)
    _mark_fractures_below_1d_axis(ax3, fracture_positions)

    # Row 4: 2D 倒谱图
    ax4 = plt.subplot(4, 1, 4)
    Q_depth = q * v / 2.0
    T_mesh, Q_mesh = np.meshgrid(t_cep, Q_depth)
    # Percentile-based color scaling to avoid extreme values washing out features
    C_data = -C
    vmin = float(np.percentile(C_data, 2))
    vmax = float(np.percentile(C_data, 98))
    if vmax <= vmin:
        vmax = vmin + 0.01
    im = ax4.pcolormesh(T_mesh, Q_mesh, C_data, shading='auto', cmap='jet',
                        vmin=vmin, vmax=vmax)
    ax4.set_xlabel('时间 [s]')
    ax4.set_ylabel('深度 [m]')
    ax4.set_title(f'2D 倒谱图 (Cepstrogram) | hop={hop_sec}s')
    ax4.invert_yaxis()
    y_lo = max(depth_min, lim3 * v / 2.0)
    y_hi = min(depth_max_2d, lim4 * v / 2.0)
    if y_hi > y_lo:
        ax4.set_ylim([y_lo, y_hi])
    _mark_fractures_side_arrows(ax4, fracture_positions, side='right')
    cbar = fig.colorbar(
        im, ax=ax4, orientation='horizontal',
        pad=0.12, fraction=0.05, aspect=40,
    )
    cbar.set_label('倒谱能量 (-C)', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.90, 0.96])
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  倒谱合并图已保存: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Report detected peaks（与图上标注 / JSON 一致）
    if detected_peaks:
        print(f"  1D 倒谱检测到 {len(detected_peaks)} 个峰 (top 10):")
        for p in detected_peaks[:10]:
            near_frac = ''
            for fp in (fracture_positions or []):
                if abs(p['depth_m'] - fp) < 200:
                    near_frac = f'  ← 裂缝({fp:.0f}m)'
            print(f"    #{p['rank']}: depth={p['depth_m']:8.1f}m  "
                  f"response={p['response']:.4f}{near_frac}")

    return {
        'pre': pre,
        'fft_fmax_display': fft_fmax_display,
        'depth_1d': depth_1d,
        'response_1d': response_1d,
        'C_2d': C,
        'q_2d': q,
        't_cep': t_cep,
        'result_1d': result_1d,
        'result_2d': result_2d,
        'v': v,
        'fs': fs,
        'figure_path': str(save_path) if save_path else None,
        'detected_peaks': detected_peaks,
    }
