# -*- coding: utf-8 -*-
"""
detection_protocol.py — 裂缝定位的**盲检测 / 独立评分**统一协议。

背景（EXP-20260730-018 / -025 审计）
------------------------------------
历史倒谱检测路径把真值泄漏进了检测器本身，导致所有历史匹配结果都**不是盲结果**：

1. 寻峰最小峰距 = ``0.35 × 真实最小缝距 / depth_step``
2. 保留峰数 top_n = ``真实缝数 + 4``
3. 匹配容差 = ``clip(0.45 × 真实最小缝距, 80, 250)`` m

其中第 3 条的 80 m 下限使近距（5–20 m）评估对「相邻峰合并」这一目标失效模式几乎全盲：
把两条相距 8 m 的缝检成一个居中峰，也会被判为匹配成功。

本模块的契约
------------
* **检测器** :func:`detect_peaks` 的签名中**不出现任何真值**。它只能看到深度轴、
  响应曲线和一组与真值无关的超参数。
* **评分器** :func:`score_detections` 只接受「检测器已经输出的峰」与真值，
  不能反过来影响检测。
* 匹配容差是**固定值**，不随真实缝距变化；并且必须同时报告容差扫描曲线
  （:func:`tolerance_sweep`），任何单点数值在引用时都要带上其容差。

新增指标
--------
* ``separation_success``：预测峰数 = 真值缝数 **且** 每条缝各自匹配到独立峰。
  这是「相邻峰是否真的被分开」的直接度量，F1 无法反映。
* ``n_likely_merged``：未匹配上的真缝中，其容差内存在**已被邻居占用**的预测峰的条数，
  即疑似被合并的缝数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.optimize import linear_sum_assignment

__all__ = [
    "DetectorConfig",
    "ScoringConfig",
    "detect_peaks",
    "score_detections",
    "tolerance_sweep",
    "evaluate_blind",
]

# 默认容差扫描点 [m]。任何单点 F1 在引用时都必须注明其容差。
DEFAULT_SWEEP_TOLERANCES_M: Tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 40.0)


@dataclass(frozen=True)
class DetectorConfig:
    """盲检测超参数。

    **这里的每一个字段都必须与真值无关。** 新增字段前请确认它不能由真实缝深/缝数导出。

    Attributes
    ----------
    min_separation_m
        寻峰的最小峰间距 [m]。历史实现用 ``0.35 × 真实最小缝距``（真值泄漏），
        现改为固定物理量。默认 5.0 m 是一个**待扫描的约定值**，不是标定结果；
        EXP-021 盲分辨率基准会把它作为扫描轴之一。
    height_pct, height_rel, height_abs
        高度门限 ``max(P{pct}, rel × max(resp), abs)``，沿用 ``CEPSTRUM_CONFIG`` 语义。
    prominence_rel
        相对显著度门限 ``prominence_rel × (max(resp) - min(resp))``；
        历史 1D 实现完全没有用 prominence，0.0 表示关闭。
    max_peaks
        最多保留的峰数。历史 2D 实现用 ``真实缝数 + 4``（真值泄漏），现改为固定上限。
    search_min_m, search_max_m
        深度搜索窗 [m]。这是**先验的井段范围**（例如射孔段 3500–4800 m），
        属于实验设计已知量，不是逐样本真值，允许使用。
    """

    min_separation_m: float = 5.0
    height_pct: float = 85.0
    height_rel: float = 0.03
    height_abs: float = 0.0
    prominence_rel: float = 0.0
    max_peaks: int = 10
    search_min_m: Optional[float] = None
    search_max_m: Optional[float] = None


@dataclass(frozen=True)
class ScoringConfig:
    """评分超参数（与检测器完全解耦）。"""

    tolerance_m: float = 10.0
    sweep_tolerances_m: Tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_SWEEP_TOLERANCES_M
    )
    # Phase 1 新增: 自适应容差，基于波速不确定度
    # 物理推导：波速相对误差 e_a = Δa/a 导致的往返时间误差为 Δt ≈ (2x/a) * e_a。
    # 转换回深度域时，引入的空间定位偏差为 Δx = (a/2)*Δt = x * e_a。
    # 因此，总容差应为：基础容差(采样率限制) + 深度 × 波速相对误差。
    adaptive_wavespeed_err: Optional[float] = None
    adaptive_base_tolerance_m: float = 10.0
    # Phase 1 新增: 失配分组标识
    mismatch_group: str = "none" # "none", "matched", "mismatched"


def detect_peaks(
    depth_m: Sequence[float],
    response: Sequence[float],
    config: Optional[DetectorConfig] = None,
) -> List[Dict[str, float]]:
    """在深度域响应曲线上盲检测峰。

    该函数**不接受任何真值参数**。若将来需要新增参数，请先确认其无法由
    真实缝深或缝数导出，否则会重新引入 EXP-025 记录的泄漏。

    Returns
    -------
    list of dict
        每项含 ``depth_m`` / ``response`` / ``prominence`` / ``rank``，按响应降序。
    """
    cfg = config or DetectorConfig()
    depth_arr = np.asarray(depth_m, dtype=float)
    resp_arr = np.asarray(response, dtype=float)

    if depth_arr.size != resp_arr.size:
        raise ValueError(
            f"depth 与 response 长度不一致：{depth_arr.size} vs {resp_arr.size}"
        )
    if resp_arr.size <= 20:
        return []

    # 深度搜索窗（先验井段，非逐样本真值）
    mask = np.ones(depth_arr.size, dtype=bool)
    if cfg.search_min_m is not None:
        mask &= depth_arr >= cfg.search_min_m
    if cfg.search_max_m is not None:
        mask &= depth_arr <= cfg.search_max_m
    if not np.any(mask):
        return []
    depth_win = depth_arr[mask]
    resp_win = resp_arr[mask]
    if resp_win.size <= 20:
        return []

    depth_step = _median_step(depth_win)
    distance_bins = max(1, int(round(cfg.min_separation_m / depth_step)))

    rmax = float(np.max(resp_win))
    height_thresh = max(
        float(np.percentile(resp_win, cfg.height_pct)),
        cfg.height_rel * rmax,
        cfg.height_abs,
    )

    find_kwargs: Dict[str, object] = {
        "height": height_thresh,
        "distance": distance_bins,
    }
    if cfg.prominence_rel > 0.0:
        span = rmax - float(np.min(resp_win))
        find_kwargs["prominence"] = cfg.prominence_rel * max(span, 0.0)

    peaks, props = scipy_signal.find_peaks(resp_win, **find_kwargs)
    if peaks.size == 0:
        return []

    prominences = props.get("prominences")
    if prominences is None:
        prominences = np.full(peaks.shape, np.nan)

    order = np.argsort(resp_win[peaks])[::-1][: cfg.max_peaks]
    return [
        {
            "depth_m": float(depth_win[peaks[i]]),
            "response": float(resp_win[peaks[i]]),
            "prominence": float(prominences[i]),
            "rank": rank,
        }
        for rank, i in enumerate(order, start=1)
    ]


def score_detections(
    predicted_depths_m: Sequence[float],
    true_depths_m: Sequence[float],
    tolerance_m: float,
) -> Dict[str, object]:
    """用固定容差对检测结果打分（Hungarian 最优一对一匹配）。

    容差是**固定输入**，不由 ``true_depths_m`` 导出——这是与历史
    ``_fracture_match_tol_m`` 的关键区别。
    """
    pred = np.asarray(predicted_depths_m, dtype=float)
    true = np.asarray(true_depths_m, dtype=float)
    n_pred, n_true = pred.size, true.size

    matched_pairs: List[Tuple[int, int, float]] = []
    if n_pred and n_true:
        cost = np.abs(true[:, None] - pred[None, :])
        row_idx, col_idx = linear_sum_assignment(cost)
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] <= tolerance_m:
                matched_pairs.append((int(r), int(c), float(cost[r, c])))

    tp = len(matched_pairs)
    fp = n_pred - tp
    fn = n_true - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    errors = np.array([e for _, _, e in matched_pairs], dtype=float)
    used_pred = {c for _, c, _ in matched_pairs}
    matched_true = {r for r, _, _ in matched_pairs}

    # 疑似合并：未匹配上的真缝，其容差内存在已被邻居占用的预测峰
    n_likely_merged = 0
    for i in range(n_true):
        if i in matched_true:
            continue
        if any(abs(true[i] - pred[j]) <= tolerance_m for j in used_pred):
            n_likely_merged += 1

    return {
        "tolerance_m": float(tolerance_m),
        "n_true": int(n_true),
        "n_pred": int(n_pred),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "count_error": int(n_pred - n_true),
        "exact_count": bool(n_pred == n_true),
        # 计数正确 且 每条缝各自匹配到独立峰
        "separation_success": bool(n_pred == n_true and tp == n_true),
        "n_likely_merged": int(n_likely_merged),
        "mean_error_m": float(errors.mean()) if errors.size else None,
        "median_error_m": float(np.median(errors)) if errors.size else None,
        "max_error_m": float(errors.max()) if errors.size else None,
        "p95_error_m": float(np.percentile(errors, 95)) if errors.size else None,
        "matches": [
            {
                "true_depth_m": float(true[r]),
                "pred_depth_m": float(pred[c]),
                "error_m": float(e),
            }
            for r, c, e in sorted(matched_pairs, key=lambda t: t[0])
        ],
    }


def tolerance_sweep(
    predicted_depths_m: Sequence[float],
    true_depths_m: Sequence[float],
    tolerances_m: Optional[Sequence[float]] = None,
) -> List[Dict[str, object]]:
    """在多个容差下重复打分，用于替代"单点容差 + 隐式真值下限"的旧做法。"""
    tols = tolerances_m if tolerances_m is not None else DEFAULT_SWEEP_TOLERANCES_M
    return [score_detections(predicted_depths_m, true_depths_m, t) for t in tols]


def evaluate_blind(
    depth_m: Sequence[float],
    response: Sequence[float],
    true_depths_m: Sequence[float],
    detector_config: Optional[DetectorConfig] = None,
    scoring_config: Optional[ScoringConfig] = None,
) -> Dict[str, object]:
    """先盲检测、后独立打分的标准组合入口。

    真值只在 :func:`score_detections` 一侧出现，检测阶段看不到它。
    """
    det_cfg = detector_config or DetectorConfig()
    sc_cfg = scoring_config or ScoringConfig()

    detected = detect_peaks(depth_m, response, det_cfg)
    pred_depths = [p["depth_m"] for p in detected]

    res = {
        "detected_peaks": detected,
        "primary": score_detections(pred_depths, true_depths_m, sc_cfg.tolerance_m),
        "tolerance_sweep": tolerance_sweep(
            pred_depths, true_depths_m, sc_cfg.sweep_tolerances_m
        ),
        "detector_config": det_cfg.__dict__.copy(),
        "mismatch_group": sc_cfg.mismatch_group,
    }
    
    if sc_cfg.adaptive_wavespeed_err is not None:
        mean_depth = float(np.mean(true_depths_m)) if len(true_depths_m) > 0 else 4000.0
        tau_adaptive = sc_cfg.adaptive_base_tolerance_m + mean_depth * sc_cfg.adaptive_wavespeed_err
        res["adaptive_primary"] = score_detections(pred_depths, true_depths_m, tau_adaptive)
        res["tau_adaptive_m"] = tau_adaptive
        
    return res


def _median_step(axis: np.ndarray) -> float:
    if axis.size >= 2:
        step = float(np.median(np.abs(np.diff(axis))))
        if step > 0:
            return step
    return 1.0
