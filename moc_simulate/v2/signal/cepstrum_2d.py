# -*- coding: utf-8 -*-
"""
moc_simulate.v2.signal.cepstrum_2d

2D 连续时空倒谱谱图 (Cepstrogram) 与多簇裂缝同步照亮时空分析
采用滑窗短时倒谱变换 (Short-Time Cepstrum Transform, STCT)，
解析声学透射脉冲随时间穿透各簇裂缝的动态时空演化。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import interp1d

from moc_simulate.v2.signal.cepstrum_1d import real_cepstrum, quefrency_to_distance


def compute_cepstrogram_2d(
    time: np.ndarray,
    head: np.ndarray,
    wavespeed: float,
    fs: Optional[float] = None,
    ts: float = 1.0,
    window_len_s: float = 30.0,
    hop_len_s: float = 1.0,
    window: Union[str, Tuple] = "kaiser",
    max_distance: Optional[float] = 5000.0,
) -> Dict[str, np.ndarray]:
    """
    计算井口瞬态信号的 2D 时空倒谱能量分布矩阵 (Cepstrogram)。

    参数:
        time: 时间序列 [s]
        head: 井口水头 [m]
        wavespeed: 声学波速 a [m/s]
        fs: 采样频率 [Hz]
        ts: 关泵时刻 [s]
        window_len_s: 滑动窗长 [s]
        hop_len_s: 步长 [s]
        window: 窗函数类型 ('kaiser', 'hamming', 'hanning' 等)
        max_distance: 最大观测井深 [m]

    返回字典:
        'time_centers': 各时间窗中心时刻向量 [s]
        'distances': 空间反射深度坐标向量 [m]
        'cepstrogram': 2D 倒谱能量矩阵，形状为 (n_time_frames, n_distances)
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
    t_post = t_uniform[mask_post]
    h_post = h_uniform[mask_post]

    total_samples = len(h_post)
    win_samples = int(round(window_len_s * fs))
    hop_samples = int(round(hop_len_s * fs))

    if win_samples > total_samples:
        win_samples = total_samples
        hop_samples = total_samples

    time_centers = []
    frames = []

    # 2. 滑窗短时倒谱
    start = 0
    ref_distances = None

    while start + win_samples <= total_samples:
        segment = h_post[start : start + win_samples]
        center_t = float(t_post[start + win_samples // 2])

        ceps, q = real_cepstrum(segment, fs=fs, window=window)
        dist = quefrency_to_distance(q, wavespeed)

        if max_distance is not None and max_distance > 0:
            mask_dist = dist <= max_distance
            dist = dist[mask_dist]
            ceps = ceps[mask_dist]

        if ref_distances is None:
            ref_distances = dist

        time_centers.append(center_t)
        frames.append(ceps[: len(ref_distances)])

        if start + win_samples == total_samples:
            break
        start += max(1, hop_samples)

    if not frames:
        # 兜底单帧
        ceps, q = real_cepstrum(h_post, fs=fs, window=window)
        dist = quefrency_to_distance(q, wavespeed)
        if max_distance is not None and max_distance > 0:
            mask_dist = dist <= max_distance
            dist = dist[mask_dist]
            ceps = ceps[mask_dist]
        ref_distances = dist
        time_centers = [float(np.mean(t_post))]
        frames = [ceps]

    cepstrogram_matrix = np.array(frames, dtype=np.float64)

    return {
        "time_centers": np.array(time_centers, dtype=np.float64),
        "distances": ref_distances,
        "cepstrogram": cepstrogram_matrix,
        "fs": float(fs),
    }
