# -*- coding: utf-8 -*-
"""
moc_simulate.v2.core.initial_field

稳态自洽流场空间解析积分器：
严格保证多簇进液质量守恒、达西沿程水头坡降与盲端死水区边界条件，
从物理底层彻底消除 t=0 前驱数值虚假激波与能量漂移。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

from moc_simulate.common.constants import G
from moc_simulate.v2.core.friction import reynolds, darcy_friction_factor


def compute_steady_state_field(
    L: float,
    N: int,
    dx: float,
    D: float,
    area: float,
    nu: float,
    K_D: float,
    V0: float,
    H0: float,
    g: float,
    toe_bc: str,
    frac_indices: Sequence[int],
    w_arr: np.ndarray,
    frac_Kp_arr: np.ndarray,
    sorted_pos: Sequence[float],
    H_ext: float = 100.0,
    raw_kleak_arr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算井筒初始自洽稳态水力空间分布剖面。

    返回:
        H_init: 节点初始水头向量 (N+1,)
        V_init: 节点初始流速向量 (N+1,)
        H_frac_ss: 各裂缝腔内初始稳态水头 (n_frac,)
        dH_perf_ss: 各簇初始射孔节流压降 (n_frac,)
        q_frac_ss: 各簇初始分配流量 (n_frac,)
        calibrated_kleak_arr: 保证稳态自洽的校准后滤失系数向量 (n_frac,)
    """
    n_frac = len(frac_indices)
    has_fractures = n_frac > 0

    V_init = np.full(N + 1, V0, dtype=np.float64)
    H_init = np.zeros(N + 1, dtype=np.float64)
    H_init[0] = H0

    q_frac_ss = np.zeros(n_frac, dtype=np.float64)
    H_frac_ss = np.zeros(n_frac, dtype=np.float64)
    dH_perf_ss = np.zeros(n_frac, dtype=np.float64)
    calibrated_kleak_arr = raw_kleak_arr.copy() if raw_kleak_arr is not None else np.zeros(n_frac, dtype=np.float64)

    if has_fractures:
        # 1. 质量守恒多簇分流
        Q_total = V0 * area
        q_frac_ss = w_arr * Q_total
        cum_q = 0.0
        cur_v = V0
        last_idx = 0
        for k, idx in enumerate(frac_indices):
            V_init[last_idx:idx] = cur_v
            cum_q += q_frac_ss[k]
            cur_v = max(0.0, V0 - cum_q / area)
            last_idx = idx
        # 最末簇至死趾端盲端死水区：流速严格归零
        V_init[last_idx:] = 0.0
    else:
        if toe_bc == "dead_end":
            V_init[:] = 0.0

    # 2. 沿井筒达西沿程阻力积分: dH/dx = -f*V*|V| / (2*g*D)
    for i in range(1, N + 1):
        v_local = V_init[i - 1]
        if abs(v_local) > 1.0e-4:
            Re_loc = reynolds(v_local, D, nu)
            f_loc = darcy_friction_factor(Re_loc, K_D, "steady")
            dH_dx = - (f_loc * v_local * abs(v_local)) / (2.0 * g * D)
        else:
            dH_dx = 0.0
        H_init[i] = H_init[i - 1] + dH_dx * dx

    # 3. 稳态射孔非线性节流压降与地层滤失自洽校准
    if has_fractures:
        for k, idx in enumerate(frac_indices):
            H_well_ss = H_init[idx]
            dH_perf_ss[k] = frac_Kp_arr[k] * (q_frac_ss[k] ** 2)
            H_frac_ss[k] = H_well_ss - dH_perf_ss[k]
            if H_frac_ss[k] <= H_ext:
                raise ValueError(
                    f"第 {k + 1} 簇裂缝 (x={sorted_pos[k]:.1f}m) 稳态水头 H_frac_ss={H_frac_ss[k]:.2f}m "
                    f"小于或等于地层水头 H_ext={H_ext:.2f}m，无法维持正向滤失。"
                )
            dp = H_frac_ss[k] - H_ext
            calibrated_kleak_arr[k] = q_frac_ss[k] / np.sqrt(dp)

    return H_init, V_init, H_frac_ss, dH_perf_ss, q_frac_ss, calibrated_kleak_arr
