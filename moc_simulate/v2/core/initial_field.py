# -*- coding: utf-8 -*-
"""
moc_simulate.v2.core.initial_field

稳态自洽流场空间解析积分器：
严格保证多簇进液质量守恒、达西沿程水头坡降与盲端死水区边界条件，
从物理底层彻底消除 t=0 前驱数值虚假激波与能量漂移。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.optimize import brentq

from moc_simulate.common.constants import G, H0_MAX_WORKING
from moc_simulate.v2.core.friction import reynolds, darcy_friction_factor


class InfeasibleSteadyStateError(ValueError):
    """物理稳态不存在，或实现水头超出工程工作包络。"""


def compute_physical_steady_residual(
    H0: float,
    dx: float,
    D: float,
    area: float,
    nu: float,
    K_D: float,
    V0: float,
    g: float,
    frac_indices: Sequence[int],
    frac_kleak_arr: np.ndarray,
    frac_Kp_arr: np.ndarray,
    H_ext_arr: np.ndarray,
) -> float:
    """
    计算给定井口水头 H0 下的死趾端闭端流量残差函数:
        R(H0) = Q0 - sum_{j=1}^{N_c} q_j(H_{w,j}(H0))
    当 H0 处于有效泄流范围时，R(H0) 严格单调递减 (dR/dH0 < 0)。
    """
    cur_h = float(H0)
    Q_total = float(V0 * area)
    cur_q = Q_total
    cur_v = float(V0)
    last_idx = 0
    for k, idx in enumerate(frac_indices):
        dx_seg = (idx - last_idx) * dx
        if abs(cur_v) > 1.0e-4:
            Re_loc = reynolds(cur_v, D, nu)
            f_loc = darcy_friction_factor(Re_loc, K_D, "steady")
            dH_dx = - (f_loc * cur_v * abs(cur_v)) / (2.0 * g * D)
        else:
            dH_dx = 0.0
        cur_h += dH_dx * dx_seg
        dp = cur_h - H_ext_arr[k]
        kj = frac_kleak_arr[k]
        kpj = frac_Kp_arr[k]
        if dp > 0.0 and kj > 0.0:
            denom = 1.0 + (kj ** 2) * kpj
            qk = kj * np.sqrt(dp / denom)
        else:
            qk = 0.0
        cur_q -= qk
        cur_v = cur_q / area
        last_idx = idx
    return cur_q


def solve_physical_steady_state(
    L: float,
    N: int,
    dx: float,
    D: float,
    area: float,
    nu: float,
    K_D: float,
    V0: float,
    g: float,
    toe_bc: str,
    frac_indices: Sequence[int],
    frac_kleak_arr: np.ndarray,
    frac_Kp_arr: np.ndarray,
    sorted_pos: Sequence[float],
    H_ext: Union[float, Sequence[float], np.ndarray] = 100.0,
    H0_guess: Optional[float] = None,
    H0_max: Optional[float] = None,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> Tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    前向物理稳态求解器 (方案 B: 给定总注入排量 Q0 前向反解 H0* 与真实物理分流 alpha^ss)。

    数学机理:
    1. 单簇渗流-节流耦合具有严格闭式解析实根:
       q_j(H_{w,j}) = k_j * sqrt(max(0, H_{w,j} - H_{ext,j}) / (1 + k_j^2 * K_{p,j}))
       dH_{perf,j} = K_{p,j} * q_j^2,  H_{frac,j} = H_{w,j} - dH_{perf,j}
    2. 沿井筒达西沿程摩阻积分: dH/dx = -f*V*|V| / (2*g*D)
    3. 死趾端闭端流量残差函数 R(H0) = Q0 - sum(q_j(H0))
       严格单调递减 (dR/dH0 < 0)，利用 scipy.optimize.brentq 高速收敛机器级精度根 H0*。
    4. 盲端死水区 (末簇下游至井底) 流速严格归零 (V=0)，水头严格平直。
    5. 当前仅实现封闭趾端 (toe_bc='dead_end')；reservoir 需单独的恒压稳态方程。
    6. 实现水头不得超过工程上限 H0_max (默认 H0_MAX_WORKING=30000 m)。传入 inf 可关闭上限。

    返回:
        H0_realized: 满足质量守恒与物理边界反解得到的真实井口水头 H0* [m]
        H_init: 节点初始自洽水头空间剖面 (N+1,) [m]
        V_init: 节点初始自洽流速空间剖面 (N+1,) [m/s]
        H_frac_ss: 各裂缝腔体内初始稳态水头 (n_frac,) [m]
        dH_perf_ss: 各簇初始射孔节流压降 (n_frac,) [m]
        q_frac_ss: 各簇物理正向实现的真实稳态流量 (n_frac,) [m^3/s]
        alpha_realized: 各簇真实分流比向量 alpha^ss = q_frac_ss / Q0 (n_frac,)
        mass_residual: 稳态全局相对质量守恒残差 |Q0 - sum(q_j)| / Q0
    """
    n_frac = len(frac_indices)
    has_fractures = n_frac > 0
    toe_bc_norm = str(toe_bc).strip().lower()
    H0_limit = H0_MAX_WORKING if H0_max is None else float(H0_max)

    if toe_bc_norm != "dead_end":
        raise ValueError(
            f"solve_physical_steady_state 当前仅实现封闭趾端 toe_bc='dead_end'，收到 '{toe_bc}'。"
            "reservoir 边界需要单独的趾端恒压稳态方程；"
            "禁止用死趾端零流量场初始化后再由瞬时边界把趾端拉到另一水头，否则会立刻产生假瞬态。"
        )
    if not (H0_limit > 0.0):
        raise ValueError(f"H0_max 必须为正数或 +inf，当前 H0_max={H0_limit}")

    if np.isscalar(H_ext):
        H_ext_arr = np.full(max(n_frac, 1), float(H_ext), dtype=np.float64)[:n_frac]
    else:
        H_ext_arr = np.asarray(H_ext, dtype=np.float64)

    Q_total = float(V0 * area)

    if V0 < 0.0:
        raise ValueError(f"初始注入流速 V0 必须 >= 0 (当前 V0={V0})，当前不支持负排量反抽稳态求解。")

    # 无裂缝 + 封闭趾端：V0>0 时注入无处泄流，不存在稳态；V0=0 才是静水休止态
    if not has_fractures:
        if V0 > 0.0:
            raise InfeasibleSteadyStateError(
                f"无裂缝、封闭趾端且 V0={V0} > 0 时不存在物理稳态解："
                "井口注入无法在趾端封闭条件下泄流。禁止返回全零速度场，"
                "否则第一时步井口速度会从 0 跳变到 V0 并产生假瞬态。"
            )
        h0_val = float(H0_guess) if H0_guess is not None else 100.0
        H_init = np.full(N + 1, h0_val, dtype=np.float64)
        V_init = np.zeros(N + 1, dtype=np.float64)
        return (
            h0_val,
            H_init,
            V_init,
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            0.0,
        )

    if Q_total <= 0.0:
        h0_val = float(np.max(H_ext_arr))
        H_init = np.full(N + 1, h0_val, dtype=np.float64)
        V_init = np.zeros(N + 1, dtype=np.float64)
        return (
            h0_val,
            H_init,
            V_init,
            H_ext_arr.copy(),
            np.zeros(n_frac, dtype=np.float64),
            np.zeros(n_frac, dtype=np.float64),
            np.zeros(n_frac, dtype=np.float64),
            0.0,
        )

    if float(np.sum(frac_kleak_arr)) <= 0.0:
        raise InfeasibleSteadyStateError(
            f"所有裂缝滤失系数均为零 (sum(k_leak)=0)，在死趾端 (dead_end) 边界下总排量 "
            f"Q0={Q_total:.4e} m^3/s 无法泄流，不存在物理稳态解。"
        )

    # 构造外层残差函数 R(H0) = Q0 - sum(q_j)
    def _eval_residual(h0_val: float) -> float:
        return compute_physical_steady_residual(
            H0=h0_val,
            dx=dx,
            D=D,
            area=area,
            nu=nu,
            K_D=K_D,
            V0=V0,
            g=g,
            frac_indices=frac_indices,
            frac_kleak_arr=frac_kleak_arr,
            frac_Kp_arr=frac_Kp_arr,
            H_ext_arr=H_ext_arr,
        )

    # 确定单调求根括号区间 [H_min, H_max]
    # 在 H_min 时，井筒水头低于地层水头，侧向泄流量为 0，残差 R(H_min) = Q0 > 0
    H_min = float(np.min(H_ext_arr))
    r_min = _eval_residual(H_min)
    if r_min <= 0.0:
        H_min -= 100.0
        while _eval_residual(H_min) <= 0.0:
            H_min -= 200.0

    # 粗估高点
    k_eff_sum = np.sum(frac_kleak_arr / np.sqrt(1.0 + (frac_kleak_arr ** 2) * frac_Kp_arr))
    dh_est = (Q_total / k_eff_sum) ** 2 if k_eff_sum > 0.0 else 100.0
    H_max = max(float(np.max(H_ext_arr)) + dh_est + 50.0, H_min + 10.0)
    if H0_guess is not None and H0_guess > H_max:
        H_max = float(H0_guess) + 50.0
    H_max = min(H_max, H0_limit)

    if np.isfinite(H0_limit) and H_min >= H0_limit:
        raise InfeasibleSteadyStateError(
            f"求根下界 H_min={H_min:.1f} m 已不低于工程水头上限 H0_max={H0_limit:.1f} m，"
            "不存在可行工作点。"
        )

    # 在工程上限内几何扩张直至 R(H_max) < 0；越限则判定工况不可行
    step = max(50.0, dh_est)
    for _ in range(100):
        r_max = _eval_residual(H_max)
        if r_max < 0.0:
            break
        if np.isfinite(H0_limit) and H_max >= H0_limit - 1.0e-12:
            raise InfeasibleSteadyStateError(
                f"井口稳态水头将超过工程上限 H0_max={H0_limit:.1f} m "
                f"(R(H0_max)={r_max:.4e} >= 0，滤失不足以在工作包络内吞掉 "
                f"Q0={Q_total:.4e} m^3/s)。应拒绝该样本并重新采样。"
            )
        next_h = H_max + step
        H_max = min(H0_limit, next_h) if np.isfinite(H0_limit) else next_h
        step *= 2.0
    else:
        raise RuntimeError(f"无法在 [{H_min}, {H_max}] 内包围物理稳态水头残差实根。")

    # Brent 法收敛求解真实井口水头 H0*
    H0_realized = float(brentq(_eval_residual, H_min, H_max, xtol=tol, rtol=tol, maxiter=max_iter))
    if np.isfinite(H0_limit) and H0_realized > H0_limit:
        raise InfeasibleSteadyStateError(
            f"反解井口水头 H0*={H0_realized:.1f} m 超过工程上限 H0_max={H0_limit:.1f} m。"
        )

    # 根据 H0_realized 重建自洽空间场分布
    V_init = np.zeros(N + 1, dtype=np.float64)
    H_init = np.zeros(N + 1, dtype=np.float64)
    H_init[0] = H0_realized

    q_frac_ss = np.zeros(n_frac, dtype=np.float64)
    H_frac_ss = np.zeros(n_frac, dtype=np.float64)
    dH_perf_ss = np.zeros(n_frac, dtype=np.float64)

    cur_v = float(V0)
    cur_q = Q_total
    last_idx = 0

    for k, idx in enumerate(frac_indices):
        V_init[last_idx:idx] = cur_v
        for i in range(last_idx + 1, idx + 1):
            v_local = cur_v
            if abs(v_local) > 1.0e-4:
                Re_loc = reynolds(v_local, D, nu)
                f_loc = darcy_friction_factor(Re_loc, K_D, "steady")
                dH_dx = - (f_loc * v_local * abs(v_local)) / (2.0 * g * D)
            else:
                dH_dx = 0.0
            H_init[i] = H_init[i - 1] + dH_dx * dx

        H_well_k = H_init[idx]
        dp = H_well_k - H_ext_arr[k]
        kj = frac_kleak_arr[k]
        kpj = frac_Kp_arr[k]
        if dp > 0.0 and kj > 0.0:
            denom = 1.0 + (kj ** 2) * kpj
            qk = kj * np.sqrt(dp / denom)
        else:
            qk = 0.0

        q_frac_ss[k] = qk
        dH_perf_ss[k] = kpj * (qk ** 2)
        H_frac_ss[k] = H_well_k - dH_perf_ss[k]

        cur_q -= qk
        cur_v = cur_q / area
        last_idx = idx

    # 最末簇至死趾端盲端死水区：流速严格归零，水头绝对平直
    V_init[last_idx:] = 0.0
    H_init[last_idx:] = H_init[last_idx]

    alpha_realized = q_frac_ss / Q_total if Q_total > 0.0 else np.zeros(n_frac, dtype=np.float64)
    mass_residual = float(abs(Q_total - np.sum(q_frac_ss)) / Q_total) if Q_total > 0.0 else 0.0

    return (
        H0_realized,
        H_init,
        V_init,
        H_frac_ss,
        dH_perf_ss,
        q_frac_ss,
        alpha_realized,
        mass_residual,
    )


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
