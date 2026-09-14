# -*- coding: utf-8 -*-
"""
moc_simulate.v2.core.friction

井筒达西拟稳态摩阻与 Brunone 瞬态非恒定摩阻计算模型
包含迎风差分离散、近零流速 tanh 平滑与裂缝邻域数值隔离
"""
from __future__ import annotations

from typing import Union
import numpy as np


def reynolds(V: Union[float, np.ndarray], D: float, nu: float) -> Union[float, np.ndarray]:
    """计算雷诺数 Re = |V| * D / nu"""
    return np.abs(V) * D / nu


def darcy_friction_factor(
    Re: Union[float, np.ndarray],
    K_D: float,
    model: str = "steady",
) -> Union[float, np.ndarray]:
    """
    计算达西-魏斯巴赫 (Darcy-Weisbach) 沿程阻力系数 f：
    层流 (Re < 2000): f = 64 / Re
    紊流 (Re >= 2000): Haaland 显式拟合公式 1/sqrt(f) = -1.8 * log10(6.9/Re + K_D)
    """
    Re_scalar = np.isscalar(Re)
    Re_arr = np.atleast_1d(np.asarray(Re, dtype=np.float64))
    f_arr = np.zeros_like(Re_arr)

    # 层流区
    mask_lam = (Re_arr >= 1.0e-3) & (Re_arr < 2000.0)
    f_arr[mask_lam] = 64.0 / Re_arr[mask_lam]

    # 紊流区 (Haaland 公式)
    mask_turb = Re_arr >= 2000.0
    if np.any(mask_turb):
        a_h = -1.8 * np.log10(6.9 / Re_arr[mask_turb] + K_D)
        f_arr[mask_turb] = (1.0 / a_h) ** 2

    return float(f_arr[0]) if Re_scalar else f_arr


def friction_term_J(
    f: Union[float, np.ndarray],
    D: float,
    V: Union[float, np.ndarray],
    dt: float,
) -> Union[float, np.ndarray]:
    """
    达西水头摩阻离散代数项 J_s:
        J_s = f * dt * V * |V| / (2 * D)
    """
    return f * dt * V * np.abs(V) / (2.0 * D)


def brunone_k_vec(Re_arr: np.ndarray) -> np.ndarray:
    """
    根据雷诺数计算 Brunone 非定常摩阻系数 k (Vardy-Brown 理论解析拟合)：
    层流: C = 0.00476
    紊流: C = 7.41 / Re^(log10(14.3 / Re^0.05))
    k = sqrt(C) / 2
    """
    Re_arr = np.asarray(Re_arr, dtype=np.float64)
    C = np.full_like(Re_arr, 0.0)
    valid = Re_arr >= 1.0

    laminar = valid & (Re_arr < 2000.0)
    C[laminar] = 4.76e-3

    turbulent = valid & (Re_arr >= 2000.0)
    Re_t = Re_arr[turbulent]
    C[turbulent] = 7.41 / Re_t ** (np.log10(14.3 / Re_t ** 0.05))

    return np.sqrt(C) / 2.0


def brunone_friction_Ju(
    k: float,
    dt: float,
    dVdt: float,
    dVdx: float,
    V: float,
    a: float,
) -> float:
    """
    单点 Brunone 非恒定摩阻项 J_u 计算：
        J_u = (k / 2) * dt * (dV/dt + a * sign(V) * |dV/dx|)
    """
    return float((k / 2.0) * dt * (dVdt + a * np.sign(V) * np.abs(dVdx)))
