# -*- coding: utf-8 -*-
"""
misspecification.py — 模型误差引起的估计偏差，及其与噪声下界的比较。

动机
----
CRB 只回答「噪声允许多好」。但如果正演模型本身是错的（波速取错、Brunone 系数
取错、摩阻模型形式错），估计量会产生**系统偏差**，而偏差不随 SNR 下降。因此
真正决定实用精度的是两者中较大者：

    total_error ~ max( CRB_std(SNR) ,  bias(model_error) )

本模块量化偏差，并给出「偏差 = 噪声下界」的临界模型误差，即：
    要让反演处于噪声受限（而非偏差受限）区间，波速/摩阻系数必须标定到多准。

一阶偏差公式
------------
设拟合模型的自由参数为 theta_F（Jacobian J_F），被固定在错误值上的讨厌参数为
eta（Jacobian J_eta），真实值与拟合模型在真参数处的残差为 ds。最小二乘解满足

    b = (J_F^T J_F)^{-1} J_F^T ds

当 eta 偏差为 d_eta 时 ds ≈ J_eta * d_eta；当模型形式整体错误时 ds 直接取两
模型输出之差（此时一阶式仅为量级指示）。

另一个诊断量：残差在参数子空间上的投影占比

    absorbed = ||P_J ds|| / ||ds||

absorbed 接近 1 表示模型误差几乎完全被参数「吸收」成偏差且不留痕迹（危险，
拟合看起来很好但参数是错的）；接近 0 表示模型误差主要表现为无法解释的残差
（可被检测出来）。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

__all__ = ["bias_from_residual", "nuisance_bias", "critical_nuisance_error"]


def _free_indices(names: Sequence[str], free: Sequence[str]) -> List[int]:
    return [i for i, n in enumerate(names) if n in set(free)]


def bias_from_residual(
    J: np.ndarray,
    names: Sequence[str],
    residual: np.ndarray,
    free: Sequence[str],
) -> Dict[str, object]:
    """给定模型残差，求最小二乘估计的一阶偏差与残差吸收比。"""
    idx = _free_indices(names, free)
    JF = J[:, idx]
    # 列尺度归一化后再解，避免 m / log10 单位混用造成的病态
    scale = np.linalg.norm(JF, axis=0)
    scale[scale == 0] = 1.0
    JFs = JF / scale
    sol, *_ = np.linalg.lstsq(JFs, residual, rcond=None)
    b = sol / scale
    proj = JFs @ sol
    rn = float(np.linalg.norm(residual))
    return {
        "bias": {names[i]: float(v) for i, v in zip(idx, b)},
        "absorbed_frac": float(np.linalg.norm(proj) / rn) if rn > 0 else 0.0,
        "residual_norm": rn,
        "unexplained_norm": float(np.linalg.norm(residual - proj)),
    }


def nuisance_bias(
    jac: Dict[str, object],
    *,
    nuisance: str,
    delta: float,
    free: Sequence[str],
) -> Dict[str, object]:
    """讨厌参数被固定在错误值（偏差 delta）时，自由参数的一阶偏差。

    delta 的单位与该参数一致：``a`` 为 m/s；``log_*`` 为 log10 单位
    （即 delta=log10(1+rel) 对应相对误差 rel）。
    """
    names = list(jac["names"])
    J = np.asarray(jac["J"])
    if nuisance not in names:
        raise KeyError(f"{nuisance} 不在参数列表 {names} 中")
    residual = J[:, names.index(nuisance)] * float(delta)
    out = bias_from_residual(J, names, residual, free)
    out["nuisance"] = nuisance
    out["delta"] = float(delta)
    return out


def critical_nuisance_error(
    jac: Dict[str, object],
    report: Dict[str, object],
    *,
    nuisance: str,
    free: Sequence[str],
    targets: Sequence[str],
) -> Dict[str, float]:
    """求使「偏差 = CRB 标准差」的临界讨厌参数误差。

    返回 {target: 临界 delta}。由于一阶偏差对 delta 线性，直接按比例外推。
    """
    names = list(jac["names"])
    probe = 1.0
    b = nuisance_bias(jac, nuisance=nuisance, delta=probe, free=free)["bias"]
    std = np.asarray(report["std_no_nuisance"])
    out: Dict[str, float] = {}
    for t in targets:
        j = names.index(t)
        s = float(std[j])
        slope = abs(b.get(t, 0.0))
        out[t] = float(s / slope) if slope > 0 and np.isfinite(s) else float("inf")
    return out
