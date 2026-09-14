# -*- coding: utf-8 -*-
"""
moc_simulate.v2.core.boundary_condition

井口动力学斜坡关泵流速模型与趾端全反射闭端边界求解器
支持线性线性斜坡与 S 型余弦平滑过渡，真实模拟现场单流阀落座历程。
"""
from __future__ import annotations

from typing import Tuple, Union
import numpy as np


def compute_ramp_velocity(
    t: Union[float, np.ndarray],
    V0: float,
    ts: float,
    tc: float,
    ramp_type: str = "linear",
) -> Union[float, np.ndarray]:
    """
    计算关泵斜坡流速 V(0, t)：
    支持标量与 NumPy 数组输入。

    参数:
      t: 当前仿真时刻 [s]
      V0: 初始井口流速 [m/s]
      ts: 关泵起始时刻 [s]
      tc: 关泵历时 (斜坡时间) [s]。若 tc <= 0 则严格退化为瞬时阶跃关泵
      ramp_type: 'linear' (线性斜坡) 或 'cosine' / 'smooth' / 's_curve' (S型余弦平滑过渡)

    返回:
      V(0, t) 流速值 [m/s]
    """
    is_scalar = np.isscalar(t)
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    v_arr = np.empty_like(t_arr)

    # 1. 关泵前: V = V0
    mask_before = t_arr < ts
    v_arr[mask_before] = V0

    if tc <= 0.0:
        # 阶跃关泵: t >= ts 立即归零
        v_arr[~mask_before] = 0.0
    else:
        # 2. 关泵斜坡过渡期: ts <= t < ts + tc
        mask_ramp = (t_arr >= ts) & (t_arr < ts + tc)
        if np.any(mask_ramp):
            xi = (t_arr[mask_ramp] - ts) / tc  # 归一化进程 [0, 1)
            if ramp_type == "linear":
                tau = 1.0 - xi
            elif ramp_type in ("cosine", "smooth", "s_curve"):
                tau = 0.5 * (1.0 + np.cos(np.pi * xi))
            else:
                raise ValueError(
                    f"未知的 ramp_type: {ramp_type}，支持 'linear' 或 'cosine'/'smooth'/'s_curve'"
                )
            v_arr[mask_ramp] = V0 * tau

        # 3. 关泵完成后: t >= ts + tc, V = 0
        mask_after = t_arr >= ts + tc
        v_arr[mask_after] = 0.0

    return float(v_arr[0]) if is_scalar else v_arr


def compute_ramp_acceleration(
    t: Union[float, np.ndarray],
    V0: float,
    ts: float,
    tc: float,
    ramp_type: str = "linear",
) -> Union[float, np.ndarray]:
    """
    计算关泵斜坡加速度 dV/dt (供动力学诊断与陡度分析)：
    返回 [m/s^2]。
    """
    is_scalar = np.isscalar(t)
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    a_arr = np.zeros_like(t_arr)

    if tc > 0.0:
        mask_ramp = (t_arr >= ts) & (t_arr < ts + tc)
        if np.any(mask_ramp):
            xi = (t_arr[mask_ramp] - ts) / tc
            if ramp_type == "linear":
                a_arr[mask_ramp] = -V0 / tc
            elif ramp_type in ("cosine", "smooth", "s_curve"):
                a_arr[mask_ramp] = -0.5 * V0 * (np.pi / tc) * np.sin(np.pi * xi)
            else:
                raise ValueError(
                    f"未知的 ramp_type: {ramp_type}，支持 'linear' 或 'cosine'/'smooth'/'s_curve'"
                )

    return float(a_arr[0]) if is_scalar else a_arr


def apply_wellhead_bc(
    t: float,
    V0: float,
    ts: float,
    tc: float,
    wellhead_bc: str,
    ramp_type: str,
    Cm_0: float,
    ga: float,
) -> Tuple[float, float]:
    """
    求解井口边界点 (i=0) 水头与流速：
        已知 C^- 来自下游节点 i=1:
        V_0 - ga * H_0 = -Cm_0  =>  H_0 = (V_0 + Cm_0) / ga
    """
    if wellhead_bc == "velocity_step":
        V_wh = 0.0 if t >= ts else V0
    elif wellhead_bc == "ramp":
        V_wh = float(compute_ramp_velocity(t, V0, ts, tc, ramp_type))
    else:
        raise ValueError(f"未知 wellhead_bc: {wellhead_bc}")

    H_wh = (V_wh + Cm_0) / ga
    return V_wh, H_wh


def apply_toe_bc(
    toe_bc: str,
    toe_head: float,
    Cp_N: float,
    ga: float,
) -> Tuple[float, float]:
    """
    求解趾端边界点 (i=N) 水头与流速：
        已知 C^+ 来自上游节点 i=N-1:
        V_N + ga * H_N = Cp_N
    """
    if toe_bc == "dead_end":
        V_toe = 0.0
        H_toe = Cp_N / ga
    elif toe_bc == "reservoir":
        H_toe = toe_head
        V_toe = Cp_N - ga * H_toe
    else:
        raise ValueError(f"未知 toe_bc: {toe_bc}")

    return V_toe, H_toe
