# -*- coding: utf-8 -*-
"""
experiments/moc_physics_upgrade/step4_ramp_closure/solver.py

Step 4 升级求解器：关泵斜坡动力学（Ramp Pump Closure）实测对齐与水动力学求解器：
1. 继承 Step 1 全部物理自洽特性：
   - 沿井筒多簇流量分流: sum(q_{j,0}) = Q_pump
   - 最末簇至封闭趾端死水段流速严格归零: V(x >= x_Nc) = 0, V[-1] = 0 (消灭 t=0 虚假激波)
   - 初始水头场严格满足达西摩阻坡降积分: dH0/dx = -f*V*|V|/(2*g*D)
2. 继承 Step 2 地质尺度流体顺应性储量 (C_f = 0.01 m^2):
   - 裂缝充当真实流体弹性储能器，关泵后产生强劲的正向流体回吐反弹，彻底消除负压抽空
3. 继承 Step 3 射孔节流压降非线性耦合 (Delta H_perf = sign(q_p) * Kp * q_p^2):
   - 射孔阻抗充当声学阻尼/扼流器 (Acoustic Choke)，消除了首缝声学短路完全屏蔽
   - 声学高频脉冲穿透首缝照亮下游深部裂缝 (簇2、3、4)
4. 核心物理升级 (Step 4: 关泵斜坡动力学与阀门就位实测对齐):
   - 井口关泵流速边界模型：
     V(0, t) = V0,                                             t < t_s
     V(0, t) = V0 * tau(t),                                    t_s <= t < t_s + t_c
     V(0, t) = 0,                                              t >= t_s + t_c
     其中：
     tau_linear(t) = 1 - (t - t_s) / t_c                       (线性斜坡)
     tau_cosine(t) = 0.5 * [1 + cos(pi * (t - t_s) / t_c)]     (S 型平滑余弦过渡)
   - 消除 t_c = 0 瞬时阶跃截断所致的无限大加速度激波与高频数值吉布斯振荡
   - 真实再现油田现场单流阀弹簧起跳、阀板位移以及阻尼关断历时 (t_c = 0.5 ~ 2.0 s)
   - 探讨平滑波前对高频混响互调假阳性谐波的滤波抑制效应与亚米级定位保持性。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# 物理常数
G = 9.81


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
      t: 当前时间 [s]
      V0: 初始流速 [m/s]
      ts: 关泵起始时刻 [s]
      tc: 关泵历时 (斜坡时间) [s]。若 tc <= 0 则退化为瞬时阶跃关泵
      ramp_type: 'linear' (线性斜坡) 或 'cosine' / 'smooth' / 's_curve' (余弦平滑过渡)

    返回:
      V(0, t) 流速值
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


@dataclass
class Step4MocConfig:
    """Step 4 MOC 仿真参数（含斜坡关泵与射孔节流压降）"""
    wellbore_length: float = 5000.0          # 井深 L [m]
    wellbore_diameter: float = 0.1397        # 井径 D [m] (5.5 in 套管内径)
    fluid_density: float = 1000.0            # 压裂液密度 rho [kg/m^3]
    fluid_viscosity: float = 1.0e-6          # 运动粘度 nu [m^2/s]
    wavespeed: float = 1450.0                # 声学波速 a [m/s]
    roughness_height: float = 4.5e-5         # 套管粗糙度 eps [m]
    friction_model: str = "brunone"          # steady / quasi-steady / brunone
    brunone_k_scale: float = 1.0             # Brunone 非恒定摩阻缩放系数
    dt: float = 1.0e-3                       # 时间步长 dt [s] (1.0 ms 保证网格亚米级精度)
    tf: float = 100.0                        # 仿真截止时间 [s]

    # Step 4 核心：关泵流速边界模型
    wellhead_bc: str = "ramp"                # ramp / velocity_step
    ramp_type: str = "linear"                # linear / cosine / smooth
    pump_shut_time: float = 1.0              # 关泵起始时刻 t_s [s]
    pump_closure_duration: float = 1.0       # 关泵历时 t_c [s] (主基准 1.0s, 扫描 0.0, 0.5, 1.0, 2.0s)
    initial_velocity: float = 1.0            # 初始井口稳态流速 V0 [m/s]
    initial_head: float = 300.0              # 初始井口稳态水头 H0 [m]
    theta: float = 0.0                       # 井斜角 (水平井为 0)
    toe_bc: str = "dead_end"                 # 封闭死端趾端 dead_end / reservoir
    toe_head: float = 300.0

    # Step 3 继承：射孔限流节流压降参数 (基准 Np=6, dp=10mm, Cd=0.65)
    perf_num_holes: int = 6                  # 每簇孔数 N_p
    perf_diameter: float = 0.01              # 孔径 d_p [m] (10 mm)
    perf_cd: float = 0.65                    # 孔流系数 C_d

    # 派生计算量
    area: float = field(init=False)
    N: int = field(init=False)
    dx: float = field(init=False)
    dt_adj: float = field(init=False)
    a_adj: float = field(init=False)
    n_steps: int = field(init=False)
    perf_area: float = field(init=False)     # 单簇射孔总面积 A_p [m^2]
    perf_Kp: float = field(init=False)       # 射孔阻抗系数 K_p [s^2/m^5]

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

        # 射孔几何与压降系数
        single_hole_area = np.pi * (self.perf_diameter ** 2) / 4.0
        self.perf_area = float(self.perf_num_holes * single_hole_area)
        if self.perf_area > 0 and self.perf_cd > 0:
            self.perf_Kp = float(1.0 / (2.0 * G * (self.perf_cd ** 2) * (self.perf_area ** 2)))
        else:
            self.perf_Kp = 0.0


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


def solve_fracture_node_step4(
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
    求解 Step 4 耦合非线性射孔节流压降与裂缝顺应性的节点方程组：
    返回: (H_well, H_frac, V_left, V_right, q_p)
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

    # 牛顿迭代求解非线性残差 F(q_p) = 0
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


def simulate_wellbore_step4(
    cfg: Step4MocConfig,
    fracture_positions: Optional[List[float]] = None,
    fracture_Cf: Optional[List[float]] = None,
    fracture_kleak: Optional[List[float]] = None,
    fracture_inflow_weights: Optional[List[float]] = None,
    fracture_Kp: Optional[List[float]] = None,
    fracture_num_holes: Optional[List[int]] = None,
    fracture_perf_diameter: Optional[List[float]] = None,
    fracture_perf_cd: Optional[List[float]] = None,
    H_ext: float = 100.0,
) -> Dict:
    """
    Step 4 升级版全井 MOC 动力学仿真内核：
    包含：
    1. 关泵斜坡流速模型 (线性 / S型余弦过渡，支持任意 t_c >= 0)
    2. 地质尺度裂缝顺应性储量 (Cf = 0.01 m^2) 宏观大反弹
    3. 严格非线性射孔节流压降耦合 (Delta H_perf = sign(q_p) * Kp * q_p^2)
    4. 稳态动量自洽场 (消灭趾端假激波，无数值漂移)
    5. 封闭死趾端死水区边界 (V[-1] == 0)
    6. Brunone 瞬态非恒定摩阻
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

        # 单调递增排序与网格映射
        raw_pos = [float(xf) for xf in fracture_positions]
        sort_order = list(np.argsort(raw_pos))
        sorted_pos = [raw_pos[i] for i in sort_order]

        frac_indices = []
        for xf in sorted_pos:
            idx = round(xf / dx)
            idx = max(1, min(N - 1, idx))
            frac_indices.append(idx)

        # 检查网格冲突
        for k in range(1, n_frac):
            if frac_indices[k] <= frac_indices[k - 1]:
                raise ValueError(
                    f"裂缝簇间距小于 MOC 网格步长 dx={dx:.3f}m: 裂缝 {sort_order[k-1]+1}({sorted_pos[k-1]:.2f}m) "
                    f"与裂缝 {sort_order[k]+1}({sorted_pos[k]:.2f}m) 冲突映射到节点 {frac_indices[k]}。"
                )

        raw_Cf = list(fracture_Cf) if fracture_Cf is not None else [0.01] * n_frac
        frac_Cf_arr = np.array([raw_Cf[i] for i in sort_order], dtype=np.float64)

        raw_kleak = list(fracture_kleak) if fracture_kleak is not None else [1e-4] * n_frac
        frac_kleak_arr = np.array([raw_kleak[i] for i in sort_order], dtype=np.float64)

        if fracture_inflow_weights is not None:
            raw_w = [fracture_inflow_weights[i] for i in sort_order]
            w_arr = np.array(raw_w, dtype=np.float64)
            w_arr = w_arr / np.sum(w_arr)
        else:
            w_arr = np.full(n_frac, 1.0 / n_frac, dtype=np.float64)

        if fracture_Kp is not None:
            raw_Kp = [fracture_Kp[i] for i in sort_order]
            frac_Kp_arr = np.array(raw_Kp, dtype=np.float64)
        else:
            frac_Kp_arr = np.zeros(n_frac, dtype=np.float64)
            for k, orig_idx in enumerate(sort_order):
                n_h = fracture_num_holes[orig_idx] if fracture_num_holes is not None else cfg.perf_num_holes
                d_h = fracture_perf_diameter[orig_idx] if fracture_perf_diameter is not None else cfg.perf_diameter
                cd_h = fracture_perf_cd[orig_idx] if fracture_perf_cd is not None else cfg.perf_cd
                ap_h = n_h * np.pi * (d_h ** 2) / 4.0
                if ap_h > 0 and cd_h > 0:
                    frac_Kp_arr[k] = 1.0 / (2.0 * G * (cd_h ** 2) * (ap_h ** 2))
                else:
                    frac_Kp_arr[k] = 0.0
    else:
        n_frac = 0
        sorted_pos = []
        frac_indices = []
        frac_Cf_arr = np.array([])
        frac_kleak_arr = np.array([])
        frac_Kp_arr = np.array([])
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

    for i in range(1, N + 1):
        v_local = V_init[i - 1]
        if abs(v_local) > 1e-4:
            Re_loc = reynolds(v_local, D, nu)
            f_loc = darcy_friction_factor(Re_loc, K_D, "steady")
            dH_dx = - (f_loc * v_local * abs(v_local)) / (2.0 * G * D)
        else:
            dH_dx = 0.0
        H_init[i] = H_init[i - 1] + dH_dx * dx

    # 稳态射孔压降与滤失自洽校准
    H_frac_ss = np.zeros(n_frac, dtype=np.float64)
    dH_perf_ss = np.zeros(n_frac, dtype=np.float64)
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

    # 记录历史
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
    frac_well_head_hist = np.zeros((n_steps + 1, n_frac)) if has_fractures else np.zeros((0, 0))
    frac_perf_dH_hist = np.zeros((n_steps + 1, n_frac)) if has_fractures else np.zeros((0, 0))
    frac_Q_hist = np.zeros((n_steps + 1, n_frac)) if has_fractures else np.zeros((0, 0))

    if has_fractures:
        for k, idx in enumerate(frac_indices):
            frac_head_hist[0, k] = H_frac_ss[k]
            frac_well_head_hist[0, k] = H[idx]
            frac_perf_dH_hist[0, k] = dH_perf_ss[k]
            frac_Q_hist[0, k] = q_frac_ss[k]

    H_prev = H.copy()

    # 时间推进主循环
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
            v_r_arr = np.empty(n_frac, dtype=np.float64)
            for k, i_f in enumerate(frac_indices):
                Cp_f = Cp[i_f - 1]
                Cm_f = Cm[i_f - 1]
                H_prev_f = frac_head_hist[n - 1, k]
                q_prev = frac_Q_hist[n - 1, k]

                H_well_k, H_frac_k, v_l, v_r, q_p = solve_fracture_node_step4(
                    Cp_f=Cp_f,
                    Cm_f=Cm_f,
                    H_prev_f=H_prev_f,
                    area=area,
                    ga=ga,
                    Cf=frac_Cf_arr[k],
                    kleak=frac_kleak_arr[k],
                    H_ext=H_ext,
                    dt=dt,
                    Kp=frac_Kp_arr[k],
                    q_init=q_prev,
                )
                H_new[i_f] = H_well_k
                V_new[i_f] = v_l
                v_r_arr[k] = v_r
                frac_well_head_hist[n, k] = H_well_k
                frac_head_hist[n, k] = H_frac_k
                frac_perf_dH_hist[n, k] = H_well_k - H_frac_k
                frac_Q_hist[n, k] = q_p

        # 井口边界条件 (i=0)
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

        # Step 4: 井口关泵流速边界计算
        if cfg.wellhead_bc == "velocity_step":
            V_wh = 0.0 if t >= t_s else V0
        elif cfg.wellhead_bc == "ramp":
            V_wh = compute_ramp_velocity(t, V0, t_s, t_c, cfg.ramp_type)
        else:
            raise ValueError(f"未知 wellhead_bc: {cfg.wellhead_bc}")

        V_new[0] = V_wh
        H_new[0] = (V_wh + Cm_0) / ga

        # 趾端边界条件 (i=N)
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
                V_prev_right[i_f] = v_r_arr[k]

        H_prev = H_new.copy()

    return {
        "timestamps": timestamps,
        "wellhead_head": wh_head_hist,
        "wellhead_velocity": wh_vel_hist,
        "toe_head": toe_head_hist,
        "toe_velocity": toe_vel_hist,
        "fracture_heads": frac_head_hist,
        "fracture_well_heads": frac_well_head_hist,
        "fracture_perf_dHs": frac_perf_dH_hist,
        "fracture_Qs": frac_Q_hist,
        "fracture_indices": frac_indices,
        "x_grid": x_grid,
        "H_init": H_init,
        "V_init": V_init,
        "H_frac_ss": H_frac_ss,
        "dH_perf_ss": dH_perf_ss,
        "frac_Kp_arr": frac_Kp_arr,
        "cfg": cfg,
    }
