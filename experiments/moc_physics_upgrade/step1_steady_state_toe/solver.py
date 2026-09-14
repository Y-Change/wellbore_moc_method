# -*- coding: utf-8 -*-
"""
experiments/moc_physics_upgrade/step1_steady_state_toe/solver.py

Step 1 升级求解器：
1. 物理相容的稳态初值计算 (Physics-Consistent Steady-State Initialization):
   - 沿井筒各簇流量连续分流: sum(q_{j,0}) = Q_pump
   - 越过最末压裂簇后，至封闭趾端桥塞处流速严格为零: V(x >= x_Nc) = 0
   - 初始水头场严格满足沿程达西摩阻水头降: dH/dx = -f*V*|V|/(2*g*D)
   - 彻底消除原版在 t=0 处因趾端流速强行截断引起的虚假瞬态波
2. 封闭趾端 (dead_end) 天然满足无速度间断: V[-1] = 0
3. 裂缝节点在稳态阶段与瞬态阶段完全守恒自洽。
"""
from __future__ import annotations

import json
import time as time_module
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 物理常数
G = 9.81


@dataclass
class UpgradedMocConfig:
    """升级版 MOC 仿真参数"""
    wellbore_length: float = 5000.0          # [m]
    wellbore_diameter: float = 0.1397        # [m]
    fluid_density: float = 1000.0            # [kg/m^3]
    fluid_viscosity: float = 1.0e-6          # [m^2/s]
    wavespeed: float = 1450.0                # [m/s]
    roughness_height: float = 4.5e-5         # [m]
    friction_model: str = "brunone"          # steady / quasi-steady / brunone
    brunone_k_scale: float = 1.0
    dt: float = 1.0e-3                       # [s]
    tf: float = 100.0                        # [s]
    wellhead_bc: str = "velocity_step"       # velocity_step / ramp
    pump_shut_time: float = 1.0              # [s]
    pump_closure_duration: float = 1.0e-3    # [s] (用于 ramp)
    initial_velocity: float = 1.0            # [m/s]
    initial_head: float = 300.0              # [m]
    theta: float = 0.0
    toe_bc: str = "dead_end"                 # dead_end / reservoir
    toe_head: float = 300.0

    # 派生量
    area: float = field(init=False)
    N: int = field(init=False)
    dx: float = field(init=False)
    dt_adj: float = field(init=False)
    a_adj: float = field(init=False)
    n_steps: int = field(init=False)

    def __post_init__(self):
        self.area = np.pi * self.wellbore_diameter**2 / 4.0
        N_raw = round(self.wellbore_length / (self.wavespeed * self.dt))
        if N_raw < 4:
            raise ValueError(f"分段数过小 N={N_raw}")
        self.N = N_raw
        self.dx = self.wellbore_length / self.N
        self.a_adj = self.dx / self.dt
        self.dt_adj = self.dt
        self.n_steps = round(self.tf / self.dt)


def reynolds(V: float | np.ndarray, D: float, nu: float) -> float | np.ndarray:
    return np.abs(V) * D / nu


def darcy_friction_factor(Re: float | np.ndarray, K_D: float, model: str = "steady") -> float | np.ndarray:
    """达西摩阻系数 f (Zigrand-Swamee)"""
    Re_scalar = np.isscalar(Re)
    Re_arr = np.atleast_1d(np.asarray(Re, dtype=np.float64))
    f_arr = np.zeros_like(Re_arr)

    # 层流
    mask_lam = (Re_arr >= 1e-3) & (Re_arr < 2000.0)
    f_arr[mask_lam] = 64.0 / Re_arr[mask_lam]

    # 紊流
    mask_turb = Re_arr >= 2000.0
    if np.any(mask_turb):
        a = -1.8 * np.log10(6.9 / Re_arr[mask_turb] + K_D)
        f_arr[mask_turb] = (1.0 / a) ** 2

    return float(f_arr[0]) if Re_scalar else f_arr


def friction_term_J(f: float | np.ndarray, D: float, V: float | np.ndarray, dt: float) -> float | np.ndarray:
    return f * dt * V * np.abs(V) / (2.0 * D)


def brunone_k_vec(Re_arr: np.ndarray) -> np.ndarray:
    """Vardy-Brunone 衰减系数 k(Re)"""
    Re_arr = np.asarray(Re_arr, dtype=np.float64)
    C = np.full_like(Re_arr, 0.0)
    valid = Re_arr >= 1.0
    laminar = valid & (Re_arr < 2000.0)
    C[laminar] = 4.76e-3
    turbulent = valid & (Re_arr >= 2000.0)
    Re_t = Re_arr[turbulent]
    C[turbulent] = 7.41 / Re_t ** (np.log10(14.3 / Re_t ** 0.05))
    return np.sqrt(C) / 2.0


def brunone_friction_Ju(k: float, dt: float, dVdt: float, dVdx: float, V: float, a: float) -> float:
    return (k / 2.0) * dt * (dVdt + a * np.sign(V) * np.abs(dVdx))


def solve_fracture_node(
    Cp_f: float, Cm_f: float,
    H_prev_f: float, A: float, ga: float,
    Cf: float, kleak: float, H_ext: float, dt: float,
    newton_tol: float = 1.0e-10, newton_max_iter: int = 25,
) -> Tuple[float, float, float, float]:
    """
    求解裂缝节点连续性方程:
    A*(V_left - V_right) = q_f
    alpha*H_P + beta*sqrt(max(H_P - H_ext, 0)) + gamma = 0
    """
    alpha = 2.0 * A * ga + Cf / dt
    beta = kleak
    gamma = -A * (Cp_f + Cm_f) - (Cf / dt) * H_prev_f

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
    Q_f = A * (V_left - V_right)
    return H_P, V_left, V_right, Q_f


def simulate_wellbore_upgraded(
    cfg: UpgradedMocConfig,
    fracture_positions: Optional[List[float]] = None,
    fracture_Cf: Optional[List[float]] = None,
    fracture_kleak: Optional[List[float]] = None,
    fracture_inflow_weights: Optional[List[float]] = None,
    H_ext: float = 100.0,
    store_full_field: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """
    升级版 MOC 井筒仿真函数：
    包含：
    1. 物理自洽的稳态流场初始化 (各簇流量分流 + 沿程达西摩阻梯度 + 封闭趾端死水区自然满足)；
    2. Brunone 非定常摩阻平滑修正；
    3. 裂缝节点动量与质量自洽推进。
    """
    L = cfg.wellbore_length
    D = cfg.wellbore_diameter
    area = cfg.area
    wavespeed = cfg.wavespeed
    a = cfg.a_adj
    dt = cfg.dt_adj
    dx = cfg.dx
    N = cfg.N
    n_steps = cfg.n_steps
    ga = G / a
    K_D = cfg.roughness_height / D
    nu = cfg.fluid_viscosity
    t_s = cfg.pump_shut_time
    t_c = cfg.pump_closure_duration
    V0 = cfg.initial_velocity
    H0 = cfg.initial_head
    theta = cfg.theta
    k_scale = cfg.brunone_k_scale
    use_brunone = (cfg.friction_model == "brunone")
    use_quasi = (cfg.friction_model == "quasi-steady")

    # 达西稳态摩阻系数基准
    Re_base = reynolds(V0, D, nu)
    f_steady = darcy_friction_factor(Re_base, K_D, "steady")

    has_fractures = fracture_positions is not None and len(fracture_positions) > 0
    if has_fractures:
        n_frac = len(fracture_positions)
        frac_indices = []
        for xf in fracture_positions:
            idx = round(xf / dx)
            idx = max(1, min(N - 1, idx))
            frac_indices.append(idx)
        frac_Cf_arr = np.array(fracture_Cf if fracture_Cf is not None else [1e-5] * n_frac, dtype=np.float64)
        frac_kleak_arr = np.array(fracture_kleak if fracture_kleak is not None else [1e-4] * n_frac, dtype=np.float64)

        if fracture_inflow_weights is not None:
            w_arr = np.array(fracture_inflow_weights, dtype=np.float64)
            w_arr = w_arr / np.sum(w_arr)
        else:
            w_arr = np.full(n_frac, 1.0 / n_frac, dtype=np.float64)
    else:
        n_frac = 0
        frac_indices = []
        frac_Cf_arr = np.array([])
        frac_kleak_arr = np.array([])
        w_arr = np.array([])

    frac_index_set = set(frac_indices)

    # =========================================================================
    # 核心升级 1: 物理相容的初始稳态场初始化
    # =========================================================================
    x_grid = np.linspace(0.0, L, N + 1)
    V_init = np.full(N + 1, V0, dtype=np.float64)
    H_init = np.zeros(N + 1, dtype=np.float64)
    H_init[0] = H0

    if has_fractures:
        # 各簇在稳态时的分流流量 [m^3/s]
        Q_total = V0 * area
        q_frac_ss = w_arr * Q_total

        # 沿井筒计算阶梯流速 V(x)
        # 井口到第1缝: V0
        # 经过第k缝: 减去 q_k / area
        # 经过最后一缝: V 变为 0! (封闭趾端前无多余流速)
        cum_q = 0.0
        cur_v = V0
        last_idx = 0
        for k, idx in enumerate(frac_indices):
            V_init[last_idx:idx] = cur_v
            cum_q += q_frac_ss[k]
            cur_v = max(0.0, V0 - cum_q / area)
            last_idx = idx
        # 最末簇至趾端: 流速严格为 0 (死水区)
        V_init[last_idx:] = 0.0
    else:
        # 无缝工况: 若 toe_bc == "dead_end"，初始流速必须为 0 才能物理自洽；
        # 若为检验激波，维持传统设定
        if cfg.toe_bc == "dead_end":
            V_init[:] = 0.0

    # 沿井筒积分稳态达西水头降: dH/dx = -f*V*|V|/(2*g*D)
    for i in range(1, N + 1):
        v_local = V_init[i - 1]
        if abs(v_local) > 1e-4:
            Re_loc = reynolds(v_local, D, nu)
            f_loc = darcy_friction_factor(Re_loc, K_D, "steady")
            dH_dx = - (f_loc * v_local * abs(v_local)) / (2.0 * G * D)
        else:
            dH_dx = 0.0
        H_init[i] = H_init[i - 1] + dH_dx * dx

    # 稳态下裂缝等效滤失系数自洽校准
    # 使稳态下 kleak_eff * sqrt(H_f_ss - H_ext) == q_frac_ss[k]
    # 保证 t < ts 期间 dH/dt = 0, 零虚假扰动
    if has_fractures:
        for k, idx in enumerate(frac_indices):
            H_f_ss = H_init[idx]
            dp = max(1.0, H_f_ss - H_ext)
            # 校准稳态滤失使连续性完全自洽
            frac_kleak_arr[k] = q_frac_ss[k] / np.sqrt(dp)

    # 赋初值
    H = H_init.copy()
    V = V_init.copy()

    # 双流速数组
    V_prev_left = V.copy()
    V_prev_right = V.copy()
    if has_fractures:
        for k, idx in enumerate(frac_indices):
            # 裂缝上游流速与下游流速不同: V_left = V[idx-1], V_right = V[idx+1]
            V_prev_left[idx] = V_init[idx - 1]
            V_prev_right[idx] = V_init[idx]

    V_prev2_left = V_prev_left.copy()
    V_prev2_right = V_prev_right.copy()

    # 时间序列存储容器
    timestamps = np.zeros(n_steps + 1)
    wh_head_hist = np.zeros(n_steps + 1)
    wh_vel_hist = np.zeros(n_steps + 1)
    toe_head_hist = np.zeros(n_steps + 1)
    toe_vel_hist = np.zeros(n_steps + 1)

    wh_head_hist[0] = H[0]
    wh_vel_hist[0] = V_prev_left[0]
    toe_head_hist[0] = H[-1]
    toe_vel_hist[0] = V_prev_right[-1]

    frac_head_hist = np.zeros((n_steps + 1, n_frac)) if has_fractures else np.zeros((0, 0))
    frac_Q_hist = np.zeros((n_steps + 1, n_frac)) if has_fractures else np.zeros((0, 0))
    if has_fractures:
        for k, idx in enumerate(frac_indices):
            frac_head_hist[0, k] = H[idx]
            frac_Q_hist[0, k] = q_frac_ss[k]

    H_prev = H.copy()

    # =========================================================================
    # 时间推进循环
    # =========================================================================
    for n in range(1, n_steps + 1):
        t = n * dt
        timestamps[n] = t

        # ── 1. 内节点 1..N-1 ──────────────────────────────────
        V1 = V_prev_right[:-2]   # C⁺ 来源节点 0..N-2 下游侧流速
        H1 = H_prev[:-2]
        V2 = V_prev_left[2:]     # C⁻ 来源节点 2..N 上游侧流速
        H2 = H_prev[2:]

        if use_quasi:
            Re1 = reynolds(V1, D, nu)
            Re2 = reynolds(V2, D, nu)
            f1 = darcy_friction_factor(Re1, K_D, "quasi-steady")
            f2 = darcy_friction_factor(Re2, K_D, "quasi-steady")
            J1 = friction_term_J(f1, D, V1, dt)
            J2 = friction_term_J(f2, D, V2, dt)
        else:
            J1 = friction_term_J(f_steady, D, V1, dt)
            J2 = friction_term_J(f_steady, D, V2, dt)

        # Brunone 非定常摩阻
        if use_brunone and n >= 2:
            V_smooth = 0.05
            # 前向差分 C⁺
            dVdt_1 = (V1 - V_prev2_right[:-2]) / dt
            dVdx_1 = (V_prev_right[1:-1] - V1) / dx
            Re_1b = reynolds(V1, D, nu)
            k_1b = brunone_k_vec(Re_1b) * k_scale
            sign_V1 = np.tanh(V1 / V_smooth)
            Ju_1 = (k_1b / 2.0) * dt * (dVdt_1 + a * sign_V1 * np.abs(dVdx_1))

            # 后向差分 C⁻
            dVdt_2 = (V2 - V_prev2_left[2:]) / dt
            dVdx_2 = (V2 - V_prev_left[1:-1]) / dx
            Re_2b = reynolds(V2, D, nu)
            k_2b = brunone_k_vec(Re_2b) * k_scale
            sign_V2 = np.tanh(V2 / V_smooth)
            Ju_2 = (k_2b / 2.0) * dt * (dVdt_2 - a * sign_V2 * np.abs(dVdx_2))

            # 裂缝邻域避免越界数值差分
            if has_fractures:
                for idx in frac_indices:
                    for offset in [-1, 0, 1]:
                        target = idx + offset - 1
                        if 0 <= target < len(Ju_1):
                            Ju_1[target] = 0.0
                        if 0 <= target < len(Ju_2):
                            Ju_2[target] = 0.0

            J1 += Ju_1
            J2 += Ju_2

        # 相容方程系数
        Cp = V1 + ga * H1 - J1 + ga * dt * V1 * theta
        Cm = -V2 + ga * H2 + J2 + ga * dt * V2 * theta

        H_new = np.empty(N + 1, dtype=np.float64)
        V_new = np.empty(N + 1, dtype=np.float64)

        # 常规内节点
        H_new[1:N] = (Cp + Cm) / (2.0 * ga)
        V_new[1:N] = (Cp - Cm) / 2.0

        # 裂缝内节点单独求解
        if has_fractures:
            for k, i_f in enumerate(frac_indices):
                Cp_f = Cp[i_f - 1]
                Cm_f = Cm[i_f - 1]
                H_prev_f = frac_head_hist[n - 1, k]

                H_p, v_l, v_r, q_f = solve_fracture_node(
                    Cp_f, Cm_f, H_prev_f, area, ga,
                    frac_Cf_arr[k], frac_kleak_arr[k], H_ext, dt,
                )
                H_new[i_f] = H_p
                V_new[i_f] = v_l
                frac_head_hist[n, k] = H_p
                frac_Q_hist[n, k] = q_f

        # ── 2. 井口左边界 (i=0) ────────────────────────────────
        V2_0 = V_prev_left[1]
        H2_0 = H_prev[1]
        f_0 = darcy_friction_factor(reynolds(V2_0, D, nu), K_D, "steady")
        J_0 = friction_term_J(f_0, D, V2_0, dt)
        if use_brunone and n >= 2:
            dVdt_0 = (V_prev_left[1] - V_prev2_left[1]) / dt
            dVdx_0 = (V_prev_left[1] - V_prev_left[0]) / dx
            k_0 = brunone_k_vec(np.array([reynolds(V2_0, D, nu)]))[0] * k_scale
            J_0 += brunone_friction_Ju(k_0, dt, dVdt_0, dVdx_0, V2_0, a)
        Cm_0 = -V2_0 + ga * H2_0 + J_0 + ga * dt * V2_0 * theta

        if cfg.wellhead_bc == "velocity_step":
            V_wh = 0.0 if t >= t_s else V0
        elif cfg.wellhead_bc == "ramp":
            if t < t_s:
                V_wh = V0
            elif t < t_s + t_c:
                V_wh = V0 * (1.0 - (t - t_s) / t_c)
            else:
                V_wh = 0.0
        else:
            raise ValueError(f"未知 wellhead_bc: {cfg.wellhead_bc}")

        V_new[0] = V_wh
        H_new[0] = (V_wh + Cm_0) / ga

        # ── 3. 趾端右边界 (i=N) ────────────────────────────────
        V1_N = V_prev_right[N - 1]
        H1_N = H_prev[N - 1]
        f_N = darcy_friction_factor(reynolds(V1_N, D, nu), K_D, "steady")
        J_N = friction_term_J(f_N, D, V1_N, dt)
        if use_brunone and n >= 2:
            dVdt_N = (V_prev_right[N - 1] - V_prev2_right[N - 1]) / dt
            dVdx_N = (V_prev_right[N] - V_prev_right[N - 1]) / dx
            k_N = brunone_k_vec(np.array([reynolds(V1_N, D, nu)]))[0] * k_scale
            J_N += brunone_friction_Ju(k_N, dt, dVdt_N, dVdx_N, V1_N, a)
        Cp_N = V1_N + ga * H1_N - J_N + ga * dt * V1_N * theta

        if cfg.toe_bc == "dead_end":
            V_new[N] = 0.0
            H_new[N] = Cp_N / ga
        elif cfg.toe_bc == "reservoir":
            H_new[N] = cfg.toe_head
            V_new[N] = Cp_N - ga * H_new[N]

        # 记录
        wh_head_hist[n] = H_new[0]
        wh_vel_hist[n] = V_new[0]
        toe_head_hist[n] = H_new[-1]
        toe_vel_hist[n] = V_new[-1]

        # 状态回传
        V_prev2_left = V_prev_left.copy()
        V_prev2_right = V_prev_right.copy()
        V_prev_left = V_new.copy()
        V_prev_right = V_new.copy()

        if has_fractures:
            for k, i_f in enumerate(frac_indices):
                V_prev_left[i_f] = V_new[i_f]
                # V_right = -Cm + ga*H
                Cm_f = Cm[i_f - 1]
                V_prev_right[i_f] = -Cm_f + ga * H_new[i_f]

        H_prev = H_new.copy()

    return {
        "timestamps": timestamps,
        "wellhead_head": wh_head_hist,
        "wellhead_velocity": wh_vel_hist,
        "toe_head": toe_head_hist,
        "toe_velocity": toe_vel_hist,
        "fracture_heads": frac_head_hist,
        "fracture_Qs": frac_Q_hist,
        "fracture_indices": frac_indices,
        "x_grid": x_grid,
        "H_init": H_init,
        "V_init": V_init,
    }
