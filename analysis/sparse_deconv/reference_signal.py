# -*- coding: utf-8 -*-
"""
无裂缝参考波形生成、差信号计算与子波提取。
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
from scipy.interpolate import interp1d

# 添加项目根目录到 sys.path
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

from moc_simulate.wellbore_moc import simulate_wellbore, MocConfig
from moc_simulate.leakoff_multi import _build_moc_config


def generate_reference_signal(friction: str = 'steady') -> Dict:
    """
    运行无裂缝 MOC 仿真，生成参考波形。
    
    Parameters
    ----------
    friction : str
        摩阻模型，如 'steady', 'brunone' 等，会自动解析配置。

    Returns
    -------
    dict:
        t       : 时间轴 [s], shape (n_steps+1,)
        H_wh    : 井口水头 [m], shape (n_steps+1,)
        cfg     : MocConfig 对象
        a_adj   : 调整后波速 [m/s]
        fs      : 采样率 [Hz]
    """
    # 解析摩擦配置获取 MocConfig
    cfg = _build_moc_config(friction)
    
    # 无裂缝仿真
    sim_result = simulate_wellbore(cfg, fracture_positions=None, store_full_field=False)
    
    t = sim_result['timestamps']
    H_wh = sim_result['wellhead_head']
    
    return {
        't': t,
        'H_wh': H_wh,
        'cfg': cfg,
        'a_adj': cfg.a_adj,
        'fs': 1.0 / cfg.dt_adj
    }


def extract_wavelet(y_ref: np.ndarray, fs: float, a_adj: float, L: float, ts: float, t_arr: np.ndarray) -> np.ndarray:
    """
    从无裂缝参考信号截取首个水击脉冲作为基本子波 h(t)。
    
    截取范围：停泵时刻 → 首个趾端全反射返回 (2L/a)
    
    Parameters
    ----------
    y_ref : np.ndarray
        无裂缝参考井口水头
    fs : float
        采样率 [Hz]
    a_adj : float
        波速 [m/s]
    L : float
        井深 [m]
    ts : float
        停泵时刻 [s]
    t_arr : np.ndarray
        时间轴
        
    Returns
    -------
    h : np.ndarray
        首个反射周期子波波形
    """
    ts_idx = int(np.searchsorted(t_arr, ts))
    T_pulse = 2.0 * L / a_adj
    pulse_len = int(T_pulse * fs)
    
    h_end = min(ts_idx + pulse_len, len(y_ref))
    h = y_ref[ts_idx:h_end]
    
    # 减去子波截取前的稳态均值，使子波从0附近开始扰动，或者去趋势
    # 停泵前的稳态均值
    steady_mean = np.mean(y_ref[max(0, ts_idx - int(fs)):ts_idx]) if ts_idx > 0 else y_ref[0]
    h = h - steady_mean
    
    return h


def compute_difference_signal(t_frac: np.ndarray, H_frac: np.ndarray, 
                              t_ref: np.ndarray, H_ref: np.ndarray,
                              fs: float, ts: float) -> Dict:
    """
    计算差信号: Δy = H_frac - H_ref (停泵后、重采样对齐)
    
    Parameters
    ----------
    t_frac, H_frac : 含缝仿真数据
    t_ref, H_ref   : 无裂缝仿真数据
    fs : float     : 目标采样率
    ts : float     : 停泵时刻
    
    Returns
    -------
    dict:
        t_after      : 停泵后时间轴
        delta_y      : 差信号 (H_frac - H_ref)
        H_frac_after : 含缝信号（停泵后去趋势前，减去了 steady_mean）
        H_ref_after  : 参考信号（停泵后去趋势前，减去了 steady_mean）
    """
    # 统一重采样到等间距 dt = 1/fs
    t_end = min(t_frac[-1], t_ref[-1])
    t_new = np.arange(0.0, float(t_end), 1.0 / fs)
    
    interp_frac = interp1d(t_frac, H_frac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    h_frac_resampled = interp_frac(t_new)
    
    interp_ref = interp1d(t_ref, H_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    h_ref_resampled = interp_ref(t_new)
    
    ts_idx = int(np.searchsorted(t_new, ts))
    if ts_idx >= len(t_new):
        ts_idx = len(t_new) // 2
        
    t_after = t_new[ts_idx:]
    h_f_after = h_frac_resampled[ts_idx:]
    h_r_after = h_ref_resampled[ts_idx:]
    
    # 分别减去各自的停泵前稳态均值，消除稳态压差
    mean_f = np.mean(h_frac_resampled[max(0, ts_idx - int(fs)):ts_idx]) if ts_idx > 0 else h_f_after[0]
    mean_r = np.mean(h_ref_resampled[max(0, ts_idx - int(fs)):ts_idx]) if ts_idx > 0 else h_r_after[0]
    
    h_f_norm = h_f_after - mean_f
    h_r_norm = h_r_after - mean_r
    
    delta_y = h_f_norm - h_r_norm
    
    return {
        't_after': t_after,
        'delta_y': delta_y,
        'H_frac_after': h_f_norm,
        'H_ref_after': h_r_norm,
    }
