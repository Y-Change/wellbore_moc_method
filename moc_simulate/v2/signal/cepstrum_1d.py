# -*- coding: utf-8 -*-
"""
moc_simulate.v2.signal.cepstrum_1d

1D 实数同态倒谱计算与倒频率-空间距离物理换算工具
c(tau) = Re{ IFFT( log( |FFT(w * x)| + eps ) ) }
空间距离换算: x = a * tau / 2
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.signal import get_window


def quefrency_to_distance(quefrency: np.ndarray, wavespeed: float) -> np.ndarray:
    """倒频率 tau [s] 转换为井筒单程空间反射距离 x [m]: x = a * tau / 2"""
    return (np.asarray(quefrency, dtype=np.float64) * wavespeed) / 2.0


def distance_to_quefrency(distance: np.ndarray, wavespeed: float) -> np.ndarray:
    """空间反射距离 x [m] 转换为双程水击走时倒频率 tau [s]: tau = 2 * x / a"""
    return (2.0 * np.asarray(distance, dtype=np.float64)) / wavespeed


def real_cepstrum(
    x: np.ndarray,
    fs: float,
    window: Optional[Union[str, Tuple]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算实数序列的单边实倒谱：
        1. 可选时域加窗抑制端点吉布斯泄漏
        2. 快速傅里叶变换: X(f) = FFT(x)
        3. 对数模谱: S(f) = ln(|X(f)| + eps)
        4. 傅里叶逆变换: c(tau) = Re{ IFFT(S(f)) }
        5. 截取正倒频率半轴 tau >= 0
    """
    signal = np.asarray(x, dtype=np.float64).flatten()
    signal = signal - np.mean(signal)
    n = len(signal)
    if n < 4:
        raise ValueError(f"输入信号长度过短 (len={n} < 4)")

    if window is not None:
        if isinstance(window, str):
            if window.lower() in ("none", "rect", "boxcar"):
                win_spec = None
            elif window.lower() == "kaiser":
                win_spec = ("kaiser", 14.0)
            else:
                win_spec = window
        else:
            win_spec = window

        if win_spec is not None:
            win = get_window(win_spec, n)
            signal = signal * win

    spec = fft(signal)
    log_spec = np.log(np.abs(spec) + np.finfo(np.float64).eps)
    raw_ceps = np.real(ifft(log_spec))

    half_n = n // 2 + 1
    quefrency = np.arange(half_n, dtype=np.float64) / fs
    cepstrum = raw_ceps[:half_n]
    return cepstrum, quefrency


def compute_cepstrum_1d(
    time: np.ndarray,
    head: np.ndarray,
    wavespeed: float,
    fs: Optional[float] = None,
    ts: float = 1.0,
    window: Optional[Union[str, Tuple]] = "hamming",
    derivative: bool = False,
    derivative_order: int = 1,
    max_distance: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    计算井口水头瞬变信号的 1D 空间深度实倒谱：
    自动执行等间隔重采样、关泵后波形提取、去趋势求导滤波与距离截断。

    返回字典:
        'distance': 空间反射深度坐标向量 [m]
        'cepstrum': 倒谱幅值向量
        'quefrency': 倒频率时间向量 [s]
        'fs': 采样频率 [Hz]
    """
    t_raw = np.asarray(time, dtype=np.float64).flatten()
    h_raw = np.asarray(head, dtype=np.float64).flatten()

    if fs is None:
        dt_est = float(np.median(np.diff(t_raw)))
        fs = 1.0 / dt_est if dt_est > 0 else 1000.0

    # 1. 规则网格插值与关泵后截取
    t_uniform = np.arange(t_raw[0], t_raw[-1], 1.0 / fs)
    interp_fn = interp1d(t_raw, h_raw, kind="cubic", fill_value="extrapolate")
    h_uniform = interp_fn(t_uniform)

    mask_post = t_uniform >= ts
    if not np.any(mask_post):
        mask_post = np.ones_like(t_uniform, dtype=bool)

    h_work = h_uniform[mask_post]
    h_work = h_work - np.mean(h_work)

    # 2. 可选波前差分求导
    if derivative and derivative_order > 0:
        dt = 1.0 / fs
        for _ in range(derivative_order):
            h_work = np.gradient(h_work, dt)
        h_work = h_work - np.mean(h_work)

    # 3. 核心实倒谱
    ceps, q = real_cepstrum(h_work, fs=fs, window=window)
    dist = quefrency_to_distance(q, wavespeed)

    # 4. 最大距离裁剪
    if max_distance is not None and max_distance > 0:
        mask_dist = dist <= max_distance
        dist = dist[mask_dist]
        ceps = ceps[mask_dist]
        q = q[mask_dist]

    return {
        "distance": dist,
        "cepstrum": ceps,
        "quefrency": q,
        "fs": float(fs),
    }
