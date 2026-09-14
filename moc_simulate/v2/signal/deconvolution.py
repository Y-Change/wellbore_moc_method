# -*- coding: utf-8 -*-
"""
moc_simulate.v2.signal.deconvolution

一阶时间导数波前锐化与极密簇间距 (如 20m) 混响谐波抑制滤波器
消除低频宏观大反弹的基线漂移，突出高频微弱裂缝反射脉冲并抑制互调谐波假阳性。
"""
from __future__ import annotations

from typing import Tuple, Union
import numpy as np


def apply_wavefront_derivative_filter(
    head: np.ndarray,
    dt: float,
    order: int = 1,
) -> np.ndarray:
    """
    对时域水头波形施加一阶或高阶数值微分：
        H_sharp(t) = dH(t) / dt
    在保留波前反射奇异阶跃的同时，彻底滤除缓慢膨胀回吐所带来的百米级反弹低频背景直流基线。
    """
    y = np.asarray(head, dtype=np.float64).flatten()
    if order <= 0:
        return y
    for _ in range(order):
        y = np.gradient(y, dt)
    return y - np.mean(y)


def suppress_cluster_harmonics(
    cepstrum_amp: np.ndarray,
    distances: np.ndarray,
    cluster_spacing: float,
    notch_width_m: float = 3.0,
    harmonics_count: int = 5,
) -> np.ndarray:
    """
    抑制极密簇间距多重水击来回混响产生的互调假阳性谐波峰 (Harmonic Reverberation Suppression)：
    在 d = k * Delta x (k >= 2) 处施加自适应陷波衰减窗，防止倒谱将混响误判为深部真实裂缝。
    """
    filtered_amp = np.copy(cepstrum_amp)
    if cluster_spacing <= 0.0 or harmonics_count < 1:
        return filtered_amp

    for k in range(2, harmonics_count + 1):
        target_dist = k * cluster_spacing
        mask_notch = np.abs(distances - target_dist) <= (notch_width_m / 2.0)
        if np.any(mask_notch):
            filtered_amp[mask_notch] *= 0.2  # 衰减谐波峰

    return filtered_amp
