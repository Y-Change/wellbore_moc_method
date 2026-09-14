# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.metrics

水击波物理反演阶段验收标准指标库 (对标技术文档 12.3 节与实施方案第 5 节):
1. alpha_mae: 流量份额平均绝对误差 (门槛: < 0.080 全集, < 0.030 单簇/稀疏)；
2. alpha_r2: 流量份额决定系数 R^2 (门槛: > 0.750 密集多簇)；
3. cf_mre_pct: 水力顺应性中位相对误差 Median Relative Error [%] (门槛: < 25.0%)；
4. cf_log10_mae: 对数尺度绝对误差 mean(|log10(Cf_hat / Cf)|) (门槛: < 0.200)；
5. wasserstein_1d_m: 1D 空间等效 Wasserstein 深度误差 [m] (门槛: < 5.0 m)；
6. simplex_max_dev: 物理守恒单纯形偏差 max |sum(alpha_hat) - 1.0| (门槛: < 1e-6)；
7. detection_f1_score: 裂缝起裂位置检出与分类 F1-score (容差 +/- 10m，门槛: > 0.880)；
8. nc_breakdown: 按不同簇数 (Nc in [1..6]) 分层统计指标。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class DetectionF1Result(dict):
    """
    裂缝起裂检出 F1 评估结果包装类:
    支持字典索引 (res['f1_score'])、属性访问 (res.f1_score) 以及三元组解包 (precision, recall, f1)。
    """

    @property
    def precision(self) -> float:
        return self["precision"]

    @property
    def recall(self) -> float:
        return self["recall"]

    @property
    def f1_score(self) -> float:
        return self["f1_score"]

    @property
    def f1(self) -> float:
        return self["f1_score"]

    @property
    def tp(self) -> int:
        return self["tp"]

    @property
    def fp(self) -> int:
        return self["fp"]

    @property
    def fn(self) -> int:
        return self["fn"]

    @property
    def tolerance_m(self) -> float:
        return self.get("tolerance_m", 10.0)

    def __iter__(self):
        yield self["precision"]
        yield self["recall"]
        yield self["f1_score"]


def compute_detection_f1_score(
    pred_positions: Any = None,
    pred_exist: Any = None,
    true_positions: Any = None,
    true_masks: Any = None,
    tolerance_m: float = 10.0,
    prob_threshold: float = 0.5,
    *,
    pred_probs: Any = None,
    true_mask: Any = None,
    pred_pos: Any = None,
    true_pos: Any = None,
) -> DetectionF1Result:
    """
    计算容差 +/- tolerance_m (默认 10.0m) 下的裂缝起裂位置检出 Precision, Recall 与 F1-Score。
    基于二分图贪心距离匹配 (Bipartite Greedy Nearest-Match)。

    参数:
        pred_positions: (N, M) 预测裂缝深度坐标 [m] (若未微调则可为设计射孔深度)
        pred_exist: (N, M) 预测起裂存在性概率 in [0, 1]
        true_positions: (N, M) 真实裂缝深度坐标 [m]
        true_masks: (N, M) 真实起裂布尔掩码
        tolerance_m: 空间位置容差 (m)，默认 10.0m
        prob_threshold: 起裂分类概率判定阈值，默认 0.5
    返回:
        DetectionF1Result 包含 precision, recall, f1_score, tp, fp, fn
    """
    def to_np(x: Any) -> np.ndarray:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # 别名解析支持
    if pred_positions is None and pred_pos is not None:
        pred_positions = pred_pos
    if pred_exist is None and pred_probs is not None:
        pred_exist = pred_probs
    if true_positions is None and true_pos is not None:
        true_positions = true_pos
    if true_masks is None and true_mask is not None:
        true_masks = true_mask

    if pred_positions is None or pred_exist is None or true_positions is None or true_masks is None:
        raise ValueError("必须传入 pred_positions, pred_exist, true_positions, true_masks")

    p_pos = to_np(pred_positions)
    p_ext = to_np(pred_exist)
    t_pos = to_np(true_positions)
    t_msk = to_np(true_masks).astype(bool)

    # 鲁棒参数位置自适应 (若调用者互换了前两参位置)
    if np.max(p_pos) <= 1.0 < np.max(p_ext):
        p_pos, p_ext = p_ext, p_pos
    if np.max(t_pos) <= 1.0 < np.max(t_msk.astype(float)):
        t_pos, t_msk = t_msk.astype(float), t_pos > 0.5

    p_pos = np.atleast_2d(p_pos)
    p_ext = np.atleast_2d(p_ext)
    t_pos = np.atleast_2d(t_pos)
    t_msk = np.atleast_2d(t_msk)

    N = len(p_pos)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(N):
        # 1. 提取当前样本预测为起裂的簇位置
        pred_active_idx = np.where(p_ext[i] >= prob_threshold)[0]
        p_active = p_pos[i][pred_active_idx]

        # 2. 提取当前样本真实起裂的簇位置
        t_active = t_pos[i][t_msk[i]]

        n_pred = len(p_active)
        n_true = len(t_active)

        if n_pred == 0 and n_true == 0:
            continue
        elif n_pred == 0:
            total_fn += n_true
            continue
        elif n_true == 0:
            total_fp += n_pred
            continue

        # 3. 构造二分图候选对 (dist, pred_idx, true_idx)
        candidates = []
        for p_idx in range(n_pred):
            for t_idx in range(n_true):
                dist = abs(float(p_active[p_idx]) - float(t_active[t_idx]))
                if dist <= tolerance_m:
                    candidates.append((dist, p_idx, t_idx))

        # 4. 贪心最近邻单射匹配 (Greedy Nearest First)
        candidates.sort(key=lambda item: item[0])
        matched_pred = set()
        matched_true = set()

        for dist, p_idx, t_idx in candidates:
            if p_idx not in matched_pred and t_idx not in matched_true:
                matched_pred.add(p_idx)
                matched_true.add(t_idx)

        tp = len(matched_pred)
        fp = n_pred - tp
        fn = n_true - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp + total_fp > 0:
        precision = float(total_tp / (total_tp + total_fp))
    else:
        precision = 1.0 if (total_tp + total_fn) == 0 else 0.0

    if total_tp + total_fn > 0:
        recall = float(total_tp / (total_tp + total_fn))
    else:
        recall = 1.0 if (total_tp + total_fp) == 0 else 0.0

    if precision + recall > 0.0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    return DetectionF1Result({
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f1": f1,
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "tolerance_m": float(tolerance_m),
    })


def compute_inversion_metrics(
    pred_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    target_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    L: float = 5000.0,
    tolerance_m: float = 10.0,
) -> Dict[str, Any]:
    """
    计算全套物理反演验收评估指标。
    """
    def to_np(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    pred_alpha = np.atleast_2d(to_np(pred_dict["alpha"]))        # (N, M)
    target_alpha = np.atleast_2d(to_np(target_dict["alpha"]))    # (N, M)
    mask = np.atleast_2d(to_np(target_dict["mask"])).astype(bool)# (N, M)
    positions = np.atleast_2d(to_np(target_dict["positions"]))   # (N, M)

    if "cf" in pred_dict:
        pred_cf = np.atleast_2d(to_np(pred_dict["cf"]))
    elif "log_cf" in pred_dict:
        pred_cf = np.atleast_2d(0.01 * np.exp(to_np(pred_dict["log_cf"])))
    else:
        raise KeyError("pred_dict 必须包含 'cf' 或 'log_cf'")

    if "cf" in target_dict:
        target_cf = np.atleast_2d(to_np(target_dict["cf"]))
    elif "log_cf" in target_dict:
        target_cf = np.atleast_2d(0.01 * np.exp(to_np(target_dict["log_cf"])))
    else:
        raise KeyError("target_dict 必须包含 'cf' 或 'log_cf'")

    n_samples = len(pred_alpha)

    # 1. 扁平化有效掩码簇
    valid_pred_alpha = pred_alpha[mask]
    valid_true_alpha = target_alpha[mask]
    valid_pred_cf = np.maximum(pred_cf[mask], 1e-8)
    valid_true_cf = np.maximum(target_cf[mask], 1e-8)

    # A. 流量份额指标
    alpha_diff = np.abs(valid_pred_alpha - valid_true_alpha)
    alpha_mae = float(np.mean(alpha_diff))

    # R^2 决定系数 (全集)
    ss_res = np.sum((valid_true_alpha - valid_pred_alpha) ** 2)
    ss_tot = np.sum((valid_true_alpha - np.mean(valid_true_alpha)) ** 2)
    alpha_r2 = float(1.0 - (ss_res / (ss_tot + 1e-10)))

    # B. 水力顺应性指标
    cf_rel_err = np.abs(valid_pred_cf - valid_true_cf) / valid_true_cf
    cf_mre_pct = float(np.median(cf_rel_err) * 100.0)
    cf_mean_re_pct = float(np.mean(cf_rel_err) * 100.0)

    log10_err = np.abs(np.log10(valid_pred_cf) - np.log10(valid_true_cf))
    cf_log10_mae = float(np.mean(log10_err))
    cf_log10_median = float(np.median(log10_err))

    # C. 物理守恒单纯形偏差
    simplex_sums = np.sum(np.where(mask, pred_alpha, 0.0), axis=1)
    simplex_deviations = np.abs(simplex_sums - 1.0)
    simplex_max_dev = float(np.max(simplex_deviations))

    # D. 1D Wasserstein-1 距离 (米)
    w1_meters_list = []
    for i in range(n_samples):
        m_i = mask[i]
        nc = int(np.sum(m_i))
        if nc <= 1:
            w1_meters_list.append(0.0)
            continue
        p_i = positions[i, :nc]
        pa_i = pred_alpha[i, :nc]
        ta_i = target_alpha[i, :nc]

        # 归一化保障
        pa_sum = np.sum(pa_i)
        ta_sum = np.sum(ta_i)
        if pa_sum > 0:
            pa_i = pa_i / pa_sum
        if ta_sum > 0:
            ta_i = ta_i / ta_sum

        cdf_p = np.cumsum(pa_i)
        cdf_t = np.cumsum(ta_i)
        dx_i = p_i[1:] - p_i[:-1]

        diff_cdf = np.abs(cdf_p[:-1] - cdf_t[:-1])
        w1_val = float(np.sum(diff_cdf * dx_i))
        w1_meters_list.append(w1_val)

    w1_mean_m = float(np.mean(w1_meters_list))
    w1_median_m = float(np.median(w1_meters_list))

    # E. 分簇数分层统计 (Nc in [1..6])
    nc_arr = np.sum(mask, axis=1)
    breakdown_by_nc = {}
    for nc_val in range(1, 7):
        subset_mask = (nc_arr == nc_val)
        n_sub = int(np.sum(subset_mask))
        if n_sub == 0:
            continue
        sub_mask = mask[subset_mask]
        sub_p_alpha = pred_alpha[subset_mask][sub_mask]
        sub_t_alpha = target_alpha[subset_mask][sub_mask]
        sub_p_cf = np.maximum(pred_cf[subset_mask][sub_mask], 1e-8)
        sub_t_cf = np.maximum(target_cf[subset_mask][sub_mask], 1e-8)

        sub_alpha_mae = float(np.mean(np.abs(sub_p_alpha - sub_t_alpha)))
        sub_cf_rel = np.abs(sub_p_cf - sub_t_cf) / sub_t_cf
        sub_cf_mre = float(np.median(sub_cf_rel) * 100.0)
        sub_log10_ae = float(np.mean(np.abs(np.log10(sub_p_cf) - np.log10(sub_t_cf))))
        sub_w1 = float(np.mean(np.array(w1_meters_list)[subset_mask]))

        # 计算分层 R^2
        sub_ss_res = np.sum((sub_t_alpha - sub_p_alpha) ** 2)
        sub_ss_tot = np.sum((sub_t_alpha - np.mean(sub_t_alpha)) ** 2)
        sub_r2 = float(1.0 - (sub_ss_res / (sub_ss_tot + 1e-10))) if sub_ss_tot > 1e-8 else 1.0

        breakdown_by_nc[f"Nc={nc_val}"] = {
            "samples": n_sub,
            "alpha_mae": sub_alpha_mae,
            "alpha_r2": sub_r2,
            "cf_mre_pct": sub_cf_mre,
            "cf_log10_mae": sub_log10_ae,
            "w1_mean_m": sub_w1,
        }

    # 密集工况 (Nc >= 4) 与稀疏工况 (Nc <= 3) 聚合统计
    dense_mask = (nc_arr >= 4)
    sparse_mask = (nc_arr <= 3)
    if np.sum(dense_mask) > 0:
        dense_p_a = pred_alpha[dense_mask][mask[dense_mask]]
        dense_t_a = target_alpha[dense_mask][mask[dense_mask]]
        dense_res = np.sum((dense_t_a - dense_p_a) ** 2)
        dense_tot = np.sum((dense_t_a - np.mean(dense_t_a)) ** 2)
        alpha_r2_dense = float(1.0 - (dense_res / (dense_tot + 1e-10)))
        alpha_mae_dense = float(np.mean(np.abs(dense_p_a - dense_t_a)))
    else:
        alpha_r2_dense = alpha_r2
        alpha_mae_dense = alpha_mae

    if np.sum(sparse_mask) > 0:
        sparse_p_a = pred_alpha[sparse_mask][mask[sparse_mask]]
        sparse_t_a = target_alpha[sparse_mask][mask[sparse_mask]]
        alpha_mae_sparse = float(np.mean(np.abs(sparse_p_a - sparse_t_a)))
    else:
        alpha_mae_sparse = alpha_mae

    # F. 若包含连续网格场，评估全域连续 1D Wasserstein 距离 (米)
    w1_field_mean_m = None
    if "m_alpha_grid" in pred_dict and "m_alpha_grid" in target_dict:
        pred_field = np.atleast_2d(to_np(pred_dict["m_alpha_grid"]))
        targ_field = np.atleast_2d(to_np(target_dict["m_alpha_grid"]))
        dx_f = L / (pred_field.shape[-1] - 1)
        cdf_p_f = np.cumsum(pred_field, axis=-1) * dx_f
        cdf_t_f = np.cumsum(targ_field, axis=-1) * dx_f
        cdf_p_f = cdf_p_f / np.maximum(cdf_p_f[:, -1:], 1e-6)
        cdf_t_f = cdf_t_f / np.maximum(cdf_t_f[:, -1:], 1e-6)
        w1_field_mean_m = float(np.mean(np.sum(np.abs(cdf_p_f - cdf_t_f), axis=-1) * dx_f))

    # G. 裂缝检出 F1-Score (若包含 p_exist 或分类预测)
    f1_res = None
    if "p_exist" in pred_dict:
        pred_exist = to_np(pred_dict["p_exist"])
        if "delta_x" in pred_dict:
            pred_pos = positions + to_np(pred_dict["delta_x"])
        else:
            pred_pos = positions
        f1_res = compute_detection_f1_score(
            pred_positions=pred_pos,
            pred_exist=pred_exist,
            true_positions=positions,
            true_masks=mask,
            tolerance_m=tolerance_m,
        )

    out_metrics = {
        "alpha_mae": alpha_mae,
        "alpha_r2": alpha_r2,
        "alpha_r2_dense": alpha_r2_dense,
        "alpha_mae_dense": alpha_mae_dense,
        "alpha_mae_sparse": alpha_mae_sparse,
        "cf_mre_pct": cf_mre_pct,
        "cf_mean_re_pct": cf_mean_re_pct,
        "cf_log10_mae": cf_log10_mae,
        "cf_log10_median": cf_log10_median,
        "w1_mean_m": w1_mean_m,
        "w1_median_m": w1_median_m,
        "w1_field_mean_m": w1_field_mean_m if w1_field_mean_m is not None else w1_mean_m,
        "simplex_max_dev": simplex_max_dev,
        "breakdown_by_nc": breakdown_by_nc,
    }

    if f1_res is not None:
        out_metrics["f1_score"] = f1_res["f1_score"]
        out_metrics["detection_precision"] = f1_res["precision"]
        out_metrics["detection_recall"] = f1_res["recall"]
        out_metrics["detection_tp"] = f1_res["tp"]
        out_metrics["detection_fp"] = f1_res["fp"]
        out_metrics["detection_fn"] = f1_res["fn"]

    return out_metrics
