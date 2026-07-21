# -*- coding: utf-8 -*-
"""
稀疏反卷积结果评估指标。
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
import scipy.signal as scipy_signal

def get_match_tol_m(fracture_nominal_m: List[float]) -> float:
    """计算匹配容差 [m]（与 cepstrum_mocdata 逻辑一致）"""
    sorted_d = sorted(fracture_nominal_m)
    min_spacing = float(min(np.diff(sorted_d))) if len(sorted_d) > 1 else 300.0
    return float(np.clip(min_spacing * 0.45, 80.0, 250.0))

def extract_peaks(r: np.ndarray, depth_axis: np.ndarray, 
                  height_thr: float = 1e-4) -> List[Dict]:
    """
    从反卷积结果 r 中提取峰值（裂缝位置）。
    因为 r 是稀疏的，非零值通常就是离散的峰，但也可以用 find_peaks。
    为了容忍轻微的抖动，我们取绝对值。
    """
    r_abs = np.abs(r)
    # 对于严格稀疏的序列，非零点就是峰
    peaks, props = scipy_signal.find_peaks(r_abs, height=height_thr)
    
    # 如果没有找到，可能是单点非零没有形成"山峰"，直接找所有显著非零点
    if len(peaks) == 0:
        peaks = np.where(r_abs > height_thr)[0]
    
    # 去除相距太近的杂点，如果存在的话
    # 对于L1反卷积，通常不需要
        
    detected = []
    # 按照幅度降序排列
    if len(peaks) > 0:
        order = np.argsort(r_abs[peaks])[::-1]
        for rank, pi in enumerate(peaks[order], start=1):
            detected.append({
                'depth_m': float(depth_axis[pi]),
                'response': float(r[pi]),  # 保留符号
                'rank': rank,
            })
            
    return detected


def evaluate_sparse_deconv(r: np.ndarray, depth_axis: np.ndarray, 
                           x_f_true: List[float], 
                           match_tol_m: Optional[float] = None) -> Dict:
    """
    评估稀疏反卷积结果。
    
    Parameters
    ----------
    r : np.ndarray
        反演得到的稀疏反射系数序列
    depth_axis : np.ndarray
        深度轴 [m]
    x_f_true : list
        真实的裂缝位置 [m]
    match_tol_m : float, optional
        匹配容差
        
    Returns
    -------
    dict : 包含匹配结果和指标的字典
    """
    n_fracs = len(x_f_true)
    if match_tol_m is None:
        match_tol_m = get_match_tol_m(x_f_true)
        
    # 提取峰
    # 阈值设为最大峰值的 1% 左右，过滤可能的极小数值噪声
    r_max = np.max(np.abs(r)) if len(r) > 0 else 0.0
    detected = extract_peaks(r, depth_axis, height_thr=max(1e-6, r_max * 1e-2))
    
    if not detected:
        matches = [{
            'frac_id': i + 1,
            'true_depth_m': float(d),
            'peak_depth_m': None,
            'peak_val': None,
            'error_m': None,
            'matched': False,
        } for i, d in enumerate(x_f_true)]
        return {
            'detected_peaks': [],
            'matches': matches,
            'n_matched': 0,
            'n_fracs': n_fracs,
            'n_detected': 0,
            'recall': 0.0,
            'precision': 0.0,
            'mean_error_m': None,
            'max_error_m': None,
            'match_tol_m': match_tol_m,
        }

    peak_depths = np.array([p['depth_m'] for p in detected], dtype=float)
    peak_vals = np.array([p['response'] for p in detected], dtype=float)

    matches = []
    used = set()
    errors = []
    for i, true_d in enumerate(x_f_true):
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
            errors.append(best_err)
            matches.append({
                'frac_id': i + 1,
                'true_depth_m': float(true_d),
                'peak_depth_m': float(peak_depths[best_j]),
                'peak_val': float(peak_vals[best_j]),
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

    n_matched = len(used)
    n_detected = len(detected)
    
    recall = n_matched / n_fracs if n_fracs > 0 else 0.0
    precision = n_matched / n_detected if n_detected > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'detected_peaks': detected,
        'matches': matches,
        'n_matched': n_matched,
        'n_fracs': n_fracs,
        'n_detected': n_detected,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'mean_error_m': float(np.mean(errors)) if errors else None,
        'max_error_m': float(np.max(errors)) if errors else None,
        'match_tol_m': match_tol_m,
    }
