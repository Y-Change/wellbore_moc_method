# -*- coding: utf-8 -*-
"""
moc_simulate.v2.core.fracture_node

裂缝-射孔耦合节点非线性求解器：
采用 Newton-Raphson 严格二阶局部收敛算法，
联立求解李曼特征线、射孔非线性二次节流压降 (Delta H_perf = sign(q_p) * K_p * q_p^2)、
地质尺度顺应性储能 (C_f * dH/dt) 与达西地层滤失。
支持 K_p <= 0 时的平滑无缝物理退化。
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


def solve_fracture_node_v2(
    Cp_f: float,
    Cm_f: float,
    H_prev_f: float,
    area: float,
    ga: float,
    Cf: float,
    kleak: float,
    H_ext: float,
    dt: float,
    Kp: float,
    q_init: float = 0.0,
    newton_tol: float = 1.0e-10,
    newton_max_iter: int = 40,
) -> Tuple[float, float, float, float, float]:
    """
    求解 MOC 裂缝分支节点的非线性耦合方程组。

    参数:
        Cp_f: 上游特征线因果李曼不变量 C^+
        Cm_f: 下游特征线因果李曼不变量 C^-
        H_prev_f: 上一时刻裂缝腔体内水头 [m]
        area: 井筒横截面积 [m^2]
        ga: 比值 g / a [s^-1]
        Cf: 裂缝宏观水头柔度 [m^2]
        kleak: 地层滤失系数 [m^{5/2}/s]
        H_ext: 地层孔隙水头 [m]
        dt: 时间步长 [s]
        Kp: 射孔节流阻抗系数 [s^2/m^5]
        q_init: 牛顿迭代初始猜测流量 [m^3/s]
        newton_tol: 收敛残差阈值
        newton_max_iter: 最大迭代步数

    返回:
        (H_well, H_frac, V_left, V_right, q_p)
        其中:
        H_well: 井筒与射孔交点处水头 [m]
        H_frac: 裂缝腔体内真实流体水头 [m]
        V_left: 节点左侧流速 [m/s]
        V_right: 节点右侧流速 [m/s]
        q_p: 侧向进入裂缝的净流量 [m^3/s]
    """
    H_moc0 = (Cp_f + Cm_f) / (2.0 * ga)
    B_head = 1.0 / (2.0 * area * ga)

    # 边界退化：无射孔压降 (Kp <= 0)
    if Kp <= 0.0:
        alpha = 2.0 * area * ga + Cf / dt
        beta = kleak
        gamma = -area * (Cp_f + Cm_f) - (Cf / dt) * H_prev_f

        H_P = H_prev_f
        if kleak == 0.0:
            H_P = -gamma / alpha
        else:
            for _ in range(newton_max_iter):
                arg = H_P - H_ext
                if arg <= 0.0:
                    f_val = alpha * H_P + gamma
                    f_prime = alpha
                else:
                    sqrt_arg = np.sqrt(arg)
                    f_val = alpha * H_P + beta * sqrt_arg + gamma
                    f_prime = alpha + beta / (2.0 * sqrt_arg)
                dH = -f_val / f_prime
                H_P += dH
                if abs(dH) < newton_tol:
                    break

        V_left = Cp_f - ga * H_P
        V_right = -Cm_f + ga * H_P
        Q_p = area * (V_left - V_right)
        return H_P, H_P, V_left, V_right, Q_p

    # 牛顿迭代求解非线性残差方程 F(q_p) = 0
    q = q_init
    cf_dt = Cf / dt

    for _ in range(newton_max_iter):
        abs_q = abs(q)
        H_w = H_moc0 - B_head * q
        H_f = H_w - np.sign(q) * Kp * (abs_q ** 2)

        dH_f_dq = -B_head - 2.0 * Kp * abs_q

        arg = H_f - H_ext
        if arg > 1.0e-8:
            sqrt_arg = np.sqrt(arg)
            q_leak = kleak * sqrt_arg
            dq_leak_dH = kleak / (2.0 * sqrt_arg)
        elif arg > 0.0:
            sqrt_arg = np.sqrt(arg)
            q_leak = kleak * sqrt_arg
            dq_leak_dH = kleak / (2.0 * 1.0e-4)
        else:
            q_leak = 0.0
            dq_leak_dH = 0.0

        F_val = q - (cf_dt * (H_f - H_prev_f) + q_leak)
        F_prime = 1.0 - (cf_dt + dq_leak_dH) * dH_f_dq

        dq = -F_val / F_prime
        q += dq

        if abs(F_val) < newton_tol or abs(dq) < newton_tol:
            break
    else:
        if abs(F_val) > 1.0e-4:
            raise RuntimeError(
                f"Newton solver failed to converge in {newton_max_iter} iterations: "
                f"residual={abs(F_val):.2e}, q={q:.6e}"
            )

    abs_q = abs(q)
    H_well = H_moc0 - B_head * q
    H_frac = H_well - np.sign(q) * Kp * (abs_q ** 2)
    V_left = Cp_f - ga * H_well
    V_right = -Cm_f + ga * H_well

    return H_well, H_frac, V_left, V_right, q
