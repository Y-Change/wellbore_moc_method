# -*- coding: utf-8 -*-
"""
experiments/moc_physics_upgrade/step2_fracture_compliance/solver.py

Step 2 升级求解器：
1. 继承 Step 1 全部稳态自洽特性：
   - 沿井筒多簇流量分流: sum(q_{j,0}) = Q_pump
   - 最末簇至封闭趾端死水段流速严格归零: V(x >= x_Nc) = 0, V[-1] = 0 (无 t=0 虚假激波)
   - 初始水头场严格满足达西摩阻坡降积分: dH0/dx = -f*V*|V|/(2*g*D)
2. 核心物理升级 (Step 2: 地质尺度流体顺应性储量):
   - 支持地质尺度真实水力裂缝顺应性 C_f (C_p = 10^-6 ~ 10^-5 m^3/Pa, C_H = 10^-2 ~ 10^-1 m^2)
   - 裂缝充当真实流体弹性储能器（Capacitive Buffer），在关泵减压后产生强劲的正向流体回吐反弹 (Gamma_shunt -> -1)
   - 自发激发出与实测一致的 4L/a 周期平顶平底方波水击震荡，彻底摆脱对趾端假激波的依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 物理常数
G = 9.81


@dataclass
class Step2MocConfig:
    """Step 2 MOC 仿真参数"""
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
    pump_closure_duration: float = 1.0e-3    # [s]
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
    Re_scalar = np.isscalar(Re)
    Re_arr = np.atleast_1d(np.asarray(Re, dtype=np.float64))
    f_arr = np.zeros_like(Re_arr)

    mask_lam = (Re_arr >= 1e-3) & (Re_arr < 2000.0)
    f_arr[mask_lam] = 64.0 / Re_arr[mask_lam]

    mask_turb = Re_arr >= 2000.0
    if np.any(mask_turb):
        a = -1.8 * np.log10(6.9 / Re_arr[mask_turb] + K_D)
        f_arr[mask_turb] = (1.0 / a) ** 2

    return float(f_arr[0]) if Re_scalar else f_arr


def friction_term_J(f: float | np.ndarray, D: float, V: float | np.ndarray, dt: float) -> float | np.ndarray:
    return f * dt * V * np.abs(V) / (2.0 * D)


def brunone_k_vec(Re_arr: np.ndarray) -> np.ndarray:
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


def solve_fracture_node_step2(
    Cp_f: float, Cm_f: float,
    H_prev_f: float, A: float, ga: float,
    Cf: float, kleak: float, H_ext: float, dt: float,
    newton_tol: float = 1.0e-10, newton_max_iter: int = 30,
) -> Tuple[float, float, float, float]:
    """
    求解地质尺度大顺应性裂缝节点方程：
    A*(V_left - V_right) = q_f
    q_f = Cf*(H_P - H_prev_f)/dt + kleak*sqrt(max(H_P - H_ext, 0))
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


def simulate_wellbore_step2(
    cfg: Step2MocConfig,
    fracture_positions: Optional[List[float]] = None,
    fracture_Cf: Optional[List[float]] = None,
    fracture_kleak: Optional[List[float]] = None,
    fracture_inflow_weights: Optional[List[float]] = None,
    H_ext: float = 100.0,
) -> Dict:
    """
    Step 2 升级版仿真内核：
    包含地质尺度流体顺应性储量 + Step 1 稳态流场自洽 + 封闭趾端死水区。
    """
    L = cfg.wellbore_length
    D = cfg.wellbore_diameter
    area = cfg.area
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
        # 默认使用地质尺度顺应性 Cf = 0.01 m^2 (Cp ~ 10^-6 m^3/Pa)
        frac_Cf_arr = np.array(fracture_Cf if fracture_Cf is not None else [0.01] * n_frac, dtype=np.float64)
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

    # 1. 物理相容初始稳态场
    x_grid = np.linspace(0.0, L, N + 1)
    V_init = np.full(N + 1, V0, dtype=np.float64)
    H_init = np.zeros(N + 1, dtype=np.float64)
    H_init[0] = H0

    if has_fractures:
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
        V_init[last_idx:] = 0.0
    else:
        if cfg.toe_bc == "dead_end":
            V_init[:] = 0.0

    # 沿程摩阻水头降
    for i in range(1, N + 1):
        v_local = V_init[i - 1]
        if abs(v_local) > 1e-4:
            Re_loc = reynolds(v_local, D, nu)
            f_loc = darcy_friction_factor(Re_loc, K_D, "steady")
            dH_dx = - (f_loc * v_local * abs(v_local)) / (2.0 * G * D)
        else:
            dH_dx = 0.0
        H_init[i] = H_init[i - 1] + dH_dx * dx

    # 稳态滤失自洽校准
    if has_fractures:
        for k, idx in enumerate(frac_indices):
            H_f_ss = H_init[idx]
            dp = max(1.0, H_f_ss - H_ext)
            frac_kleak_arr[k] = q_frac_ss[k] / np.sqrt(dp)

    H = H_init.copy()
    V = V_init.copy()

    V_prev_left = V.copy()
    V_prev_right = V.copy()
    if has_fractures:
        for k, idx in enumerate(frac_indices):
            V_prev_left[idx] = V_init[idx - 1]
            V_prev_right[idx] = V_init[idx]

    V_prev2_left = V_prev_left.copy()
    V_prev2_right = V_prev_right.copy()

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

    # 时间推进
    for n in range(1, n_steps + 1):
        t = n * dt
        timestamps[n] = t

        V1 = V_prev_right[:-2]
        H1 = H_prev[:-2]
        V2 = V_prev_left[2:]
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

        if use_brunone and n >= 2:
            V_smooth = 0.05
            dVdt_1 = (V1 - V_prev2_right[:-2]) / dt
            dVdx_1 = (V_prev_right[1:-1] - V1) / dx
            Re_1b = reynolds(V1, D, nu)
            k_1b = brunone_k_vec(Re_1b) * k_scale
            sign_V1 = np.tanh(V1 / V_smooth)
            Ju_1 = (k_1b / 2.0) * dt * (dVdt_1 + a * sign_V1 * np.abs(dVdx_1))

            dVdt_2 = (V2 - V_prev2_left[2:]) / dt
            dVdx_2 = (V2 - V_prev_left[1:-1]) / dx
            Re_2b = reynolds(V2, D, nu)
            k_2b = brunone_k_vec(Re_2b) * k_scale
            sign_V2 = np.tanh(V2 / V_smooth)
            Ju_2 = (k_2b / 2.0) * dt * (dVdt_2 - a * sign_V2 * np.abs(dVdx_2))

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

        Cp = V1 + ga * H1 - J1 + ga * dt * V1 * theta
        Cm = -V2 + ga * H2 + J2 + ga * dt * V2 * theta

        H_new = np.empty(N + 1, dtype=np.float64)
        V_new = np.empty(N + 1, dtype=np.float64)

        H_new[1:N] = (Cp + Cm) / (2.0 * ga)
        V_new[1:N] = (Cp - Cm) / 2.0

        if has_fractures:
            for k, i_f in enumerate(frac_indices):
                Cp_f = Cp[i_f - 1]
                Cm_f = Cm[i_f - 1]
                H_prev_f = frac_head_hist[n - 1, k]

                H_p, v_l, v_r, q_f = solve_fracture_node_step2(
                    Cp_f, Cm_f, H_prev_f, area, ga,
                    frac_Cf_arr[k], frac_kleak_arr[k], H_ext, dt,
                )
                H_new[i_f] = H_p
                V_new[i_f] = v_l
                frac_head_hist[n, k] = H_p
                frac_Q_hist[n, k] = q_f

        # 井口 (i=0)
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

        # 趾端 (i=N)
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

        wh_head_hist[n] = H_new[0]
        wh_vel_hist[n] = V_new[0]
        toe_head_hist[n] = H_new[-1]
        toe_vel_hist[n] = V_new[-1]

        V_prev2_left = V_prev_left.copy()
        V_prev2_right = V_prev_right.copy()
        V_prev_left = V_new.copy()
        V_prev_right = V_new.copy()

        if has_fractures:
            for k, i_f in enumerate(frac_indices):
                V_prev_left[i_f] = V_new[i_f]
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
