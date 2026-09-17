# -*- coding: utf-8 -*-
"""CJ-AlphaNet 验收指标：密度 → 活动簇 F1 → 活动簇 logY MAE → 活动簇 α MAE。"""
from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import torch


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _binary_prf(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool).reshape(-1)
    true = true.astype(bool).reshape(-1)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if (tp + fn) == 0 else 0.0)
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if (tp + fp) == 0 else 0.0)
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def field_w1_and_corr(
    pred_m: np.ndarray,
    true_m: np.ndarray,
    L: float = 5000.0,
) -> Dict[str, float]:
    p = np.atleast_2d(_to_np(pred_m).astype(np.float64))
    t = np.atleast_2d(_to_np(true_m).astype(np.float64))
    n_grid = p.shape[-1]
    dx = float(L) / max(n_grid - 1, 1)
    cdf_p = np.cumsum(p, axis=-1) * dx
    cdf_t = np.cumsum(t, axis=-1) * dx
    cdf_p = cdf_p / np.maximum(cdf_p[:, -1:], 1e-12)
    cdf_t = cdf_t / np.maximum(cdf_t[:, -1:], 1e-12)
    w1 = np.sum(np.abs(cdf_p - cdf_t), axis=-1) * dx
    corrs = []
    for i in range(p.shape[0]):
        if np.std(p[i]) < 1e-12 or np.std(t[i]) < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(p[i], t[i])[0, 1]))
    return {
        "m_alpha_w1_m": float(np.mean(w1)),
        "m_alpha_w1_median_m": float(np.median(w1)),
        "m_alpha_corr": float(np.nanmean(corrs)),
    }


def compute_alpha_metrics(
    pred_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    target_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    L: float = 5000.0,
    act_thr: float = 0.5,
) -> Dict[str, Any]:
    if "mask_design" in target_dict:
        mask_raw = target_dict["mask_design"]
    else:
        mask_raw = target_dict["mask"]
    mask = np.atleast_2d(_to_np(mask_raw)).astype(bool)
    y_act = np.atleast_2d(_to_np(target_dict["y_act"])).astype(bool)
    alpha_t = np.atleast_2d(_to_np(target_dict["alpha"]))
    logY_t = np.atleast_2d(_to_np(target_dict["logY"]))

    if "pred_alpha" in pred_dict:
        alpha_p = np.atleast_2d(_to_np(pred_dict["pred_alpha"]))
    else:
        alpha_p = np.atleast_2d(_to_np(pred_dict["alpha"]))
    if "pred_m_alpha" in pred_dict:
        m_p = np.atleast_2d(_to_np(pred_dict["pred_m_alpha"]))
    else:
        m_p = np.atleast_2d(_to_np(pred_dict["m_alpha_grid"]))
    m_t = np.atleast_2d(_to_np(target_dict["m_alpha_grid"]))

    if "pred_active" in pred_dict:
        p_act = np.atleast_2d(_to_np(pred_dict["pred_active"]))
    elif "pred_active_logits" in pred_dict:
        logits = np.atleast_2d(_to_np(pred_dict["pred_active_logits"]))
        p_act = 1.0 / (1.0 + np.exp(-logits))
    else:
        p_act = np.ones_like(alpha_p)

    if "pred_logY" in pred_dict:
        logY_p = np.atleast_2d(_to_np(pred_dict["pred_logY"]))
    else:
        logY_p = np.zeros_like(alpha_p)

    dens = field_w1_and_corr(m_p, m_t, L=L)

    pred_bin = (p_act >= float(act_thr)) & mask
    true_bin = y_act & mask
    prf = _binary_prf(pred_bin, true_bin)

    act = mask & y_act
    if np.any(act):
        logY_mae = float(np.mean(np.abs(logY_p[act] - logY_t[act])))
        alpha_mae_act = float(np.mean(np.abs(alpha_p[act] - alpha_t[act])))
    else:
        logY_mae = 0.0
        alpha_mae_act = 0.0
    if np.any(mask):
        alpha_mae_design = float(np.mean(np.abs(alpha_p[mask] - alpha_t[mask])))
    else:
        alpha_mae_design = 0.0

    simplex = np.abs(np.sum(np.where(mask, alpha_p, 0.0), axis=1) - 1.0)
    simplex_max = float(np.max(simplex)) if simplex.size else 0.0
    simplex_mean = float(np.mean(simplex)) if simplex.size else 0.0

    ss_res = np.sum((alpha_t[mask] - alpha_p[mask]) ** 2) if np.any(mask) else 0.0
    ss_tot = np.sum((alpha_t[mask] - np.mean(alpha_t[mask])) ** 2) if np.any(mask) else 1.0
    alpha_r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    return {
        **dens,
        "active_precision": prf["precision"],
        "active_recall": prf["recall"],
        "active_f1": prf["f1"],
        "active_tp": prf["tp"],
        "active_fp": prf["fp"],
        "active_fn": prf["fn"],
        "active_logY_mae": logY_mae,
        "active_alpha_mae": alpha_mae_act,
        "design_alpha_mae": alpha_mae_design,
        "alpha_r2_design": alpha_r2,
        "simplex_max_dev": simplex_max,
        "simplex_mean_dev": simplex_mean,
        "n_design": int(np.sum(mask)),
        "n_active": int(np.sum(act)),
    }


def selection_score(metrics: Dict[str, Any]) -> float:
    """验证选点：密度 W1 + 活动簇漏检。越小越好。"""
    w1 = float(metrics.get("m_alpha_w1_m", 1e3))
    f1 = float(metrics.get("active_f1", 0.0))
    return (w1 / 50.0) + (1.0 - f1)
