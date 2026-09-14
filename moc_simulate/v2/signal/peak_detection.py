# -*- coding: utf-8 -*-
"""
moc_simulate.v2.signal.peak_detection

倒谱特征峰智能检出、亚米级定位误差评估与峰值信噪比 (PSNR) 诊断
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.signal import find_peaks


def detect_fracture_peaks(
    distances: np.ndarray,
    cepstrum_amp: np.ndarray,
    min_depth: float = 1000.0,
    max_depth: Optional[float] = None,
    min_separation_m: float = 5.0,
    height_quantile: float = 0.85,
    top_n: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    智能检出倒谱有效特征峰（严格盲检测协议，不泄露真值信息）：
    返回:
        peak_distances: 检出的空间深度坐标 [m]
        peak_heights: 检出的峰值高度
    """
    dist_arr = np.asarray(distances, dtype=np.float64)
    amp_arr = np.asarray(cepstrum_amp, dtype=np.float64)

    mask_range = dist_arr >= min_depth
    if max_depth is not None:
        mask_range &= (dist_arr <= max_depth)

    if not np.any(mask_range):
        return np.array([]), np.array([])

    sub_dist = dist_arr[mask_range]
    sub_amp = amp_arr[mask_range]

    dd = float(np.median(np.diff(sub_dist))) if len(sub_dist) > 1 else 1.0
    distance_pts = max(1, int(round(min_separation_m / dd)))

    # 高度门限
    height_threshold = float(np.quantile(sub_amp, height_quantile))

    peaks_idx, props = find_peaks(
        sub_amp,
        height=height_threshold,
        distance=distance_pts,
    )

    if len(peaks_idx) == 0:
        return np.array([]), np.array([])

    pk_dists = sub_dist[peaks_idx]
    pk_heights = sub_amp[peaks_idx]

    # 按峰高降序并截取 top_n
    sort_desc = np.argsort(pk_heights)[::-1]
    pk_dists = pk_dists[sort_desc][:top_n]
    pk_heights = pk_heights[sort_desc][:top_n]

    # 恢复深度排序
    sort_dist = np.argsort(pk_dists)
    return pk_dists[sort_dist], pk_heights[sort_dist]


def evaluate_peak_matching(
    detected_peaks: Sequence[float],
    true_positions: Sequence[float],
    tolerance_m: float = 5.0,
) -> Dict[str, Union[float, int, List[Tuple[float, float, float]]]]:
    """
    评估倒谱检出峰与真实裂缝位置的配对精度与亚米级误差统计。

    返回:
        'n_true': 真实裂缝数
        'n_detected': 检出峰数
        'n_matched': 成功匹配裂缝数
        'recall': 召回率 (匹配数 / 真实数)
        'precision': 精确率 (匹配数 / 检出数)
        'mean_error_m': 平均定位绝对误差 [m]
        'max_error_m': 最大定位误差 [m]
        'submeter_accuracy_rate': 亚米级定位 (误差 <= 1.0m) 占比
        'matched_pairs': 匹配配对列表 [(true_pos, detected_pos, error_m), ...]
    """
    true_arr = sorted([float(x) for x in true_positions])
    det_arr = sorted([float(x) for x in detected_peaks])
    n_true = len(true_arr)
    n_det = len(det_arr)

    if n_true == 0 or n_det == 0:
        return {
            "n_true": n_true,
            "n_detected": n_det,
            "n_matched": 0,
            "recall": 0.0,
            "precision": 0.0,
            "mean_error_m": float("nan"),
            "max_error_m": float("nan"),
            "submeter_accuracy_rate": 0.0,
            "matched_pairs": [],
        }

    matched_pairs = []
    used_det = set()
    errors = []

    for xt in true_arr:
        best_j = -1
        best_err = float("inf")
        for j, xd in enumerate(det_arr):
            if j in used_det:
                continue
            err = abs(xt - xd)
            if err <= tolerance_m and err < best_err:
                best_err = err
                best_j = j
        if best_j != -1:
            used_det.add(best_j)
            matched_pairs.append((xt, det_arr[best_j], best_err))
            errors.append(best_err)

    n_matched = len(matched_pairs)
    recall = float(n_matched / n_true) if n_true > 0 else 0.0
    precision = float(n_matched / n_det) if n_det > 0 else 0.0
    mean_err = float(np.mean(errors)) if errors else float("nan")
    max_err = float(np.max(errors)) if errors else float("nan")
    submeter_rate = float(np.mean([e <= 1.0 for e in errors])) if errors else 0.0

    return {
        "n_true": n_true,
        "n_detected": n_det,
        "n_matched": n_matched,
        "recall": recall,
        "precision": precision,
        "mean_error_m": mean_err,
        "max_error_m": max_err,
        "submeter_accuracy_rate": submeter_rate,
        "matched_pairs": matched_pairs,
    }


def compute_psnr(signal: np.ndarray, peak_indices: Sequence[int]) -> float:
    """计算峰值信噪比 (PSNR) [dB]"""
    sig = np.asarray(signal, dtype=np.float64)
    if len(sig) == 0 or len(peak_indices) == 0:
        return 0.0

    valid_idx = [int(i) for i in peak_indices if 0 <= int(i) < len(sig)]
    if not valid_idx:
        return 0.0

    pk_power = float(np.mean(sig[valid_idx] ** 2))
    if pk_power <= 0.0:
        return 0.0

    # 噪声估算 (剔除峰区域后的方差)
    mask = np.ones(len(sig), dtype=bool)
    r = max(1, len(sig) // 200)
    for idx in valid_idx:
        mask[max(0, idx - r) : min(len(sig), idx + r + 1)] = False

    noise_power = float(np.var(sig[mask])) if np.any(mask) else 1e-12
    if noise_power <= 0.0:
        return 100.0
    return float(10.0 * np.log10(pk_power / noise_power))
