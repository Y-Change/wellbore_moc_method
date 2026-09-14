# -*- coding: utf-8 -*-
"""
wellbore_moc_v2.py — 自研轻量 MOC 核心 (沙箱 v2 求解器 / 阶段一 Stage 1 修复)

阶段一 (Stage 1) 核心物理修复：
-----------------------------
1. 严格锁定 toe_bc = 'dead_end' (反射系数 Gamma = +1.0)；
2. 初始场初始化时，显式将末簇裂缝至趾端死水段区间的初始流速置零：
       V(x > x_{f,last}, 0) = 0
   消除整管赋流速导致的开局撞墙假水击冲击波；
3. 稳态水头场平衡：流动段 (x <= x_last) 存在与稳态流速匹配的达西摩阻水力坡降，
   死水段 (x > x_last) 流速恒零、水力坡降恒零、水头严格水平静止为 H0。
   由此保证停泵前 (t < t_s) 井口水头严格水平恒定，无假水击波提前激发。
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# =====================================================================
# 物理常数
# =====================================================================
G = 9.81  # 重力加速度 [m/s^2]


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class MocConfig:
    """自研 MOC 仿真参数 (v2 沙箱版本，Stage 1 严格死端边界)"""
    # 几何
    wellbore_length: float = 1000.0          # [m]
    wellbore_diameter: float = 0.1397        # [m]  (5.5" 套管)
    # 流体（滑溜水）
    fluid_density: float = 1000.0            # [kg/m^3]
    fluid_viscosity: float = 1.0e-6          # [m^2/s] 运动黏度
    wavespeed: float = 1450.0                # [m/s]
    # 摩阻
    roughness_height: float = 4.5e-5         # [m] 商用钢绝对粗糙度
    friction_model: str = "steady"           # steady / quasi-steady / brunone
    brunone_k_scale: float = 1.0             # Brunone 系数 k 的标定倍率（1.0=纯 Vardy）
    # 仿真时间
    dt: float = 1.0e-3                       # [s]
    tf: float = 3.0                          # [s]
    # 井口边界
    wellhead_bc: str = "velocity_step"       # velocity_step / ramp
    pump_shut_time: float = 1.0              # [s] 停泵时刻
    pump_closure_duration: float = 1.0e-3    # [s] 仅 ramp 用
    initial_velocity: float = 1.0            # [m/s] 稳态流速（停泵前）
    # 倾角
    theta: float = 0.0                       # 井筒倾角正弦（水平井=0）
    # 初始水头
    initial_head: float = 300.0              # [m] 稳态水头基准
    # 趾端边界 (Stage 1 强制锁定 dead_end)
    toe_bc: str = "dead_end"                 # dead_end (V=0, Gamma=+1.0)
    toe_head: float = 300.0                  # [m]

    # 派生量（自动计算）
    area: float = field(init=False)
    N: int = field(init=False)
    dx: float = field(init=False)
    dt_adj: float = field(init=False)
    a_adj: float = field(init=False)
    n_steps: int = field(init=False)

    def __post_init__(self):
        self.area = np.pi * self.wellbore_diameter**2 / 4.0
        # Courant 精确：N = round(L/(a dt))，再调 a 使 a dt / dx = 1
        N_raw = round(self.wellbore_length / (self.wavespeed * self.dt))
        if N_raw < 4:
            raise ValueError(
                f"分段数过小 N={N_raw}，请减小 dt 或加长井筒。"
                f"当前 L={self.wellbore_length}, a={self.wavespeed}, dt={self.dt}"
            )
        self.N = N_raw
        self.dx = self.wellbore_length / self.N
        # 调整波速使 a dt = dx
        self.a_adj = self.dx / self.dt
        self.dt_adj = self.dt
        self.n_steps = round(self.tf / self.dt)
        # Stage 1 强制锁定 toe_bc = dead_end
        self.toe_bc = "dead_end"


# =====================================================================
# 摩阻计算
# =====================================================================
def reynolds(V: float, D: float, nu: float) -> float:
    return abs(V) * D / nu


def darcy_friction_factor(Re: float, K_D: float, model: str = "steady") -> float:
    """
    达西摩阻系数 f
    model='steady'      : 使用层流/紊流显式近似
    model='quasi-steady': 按 Re 实时更新（Zigrand-Swamee 显式近似）
    """
    if Re < 1e-3:
        return 0.0
    if Re < 2000:
        return 64.0 / Re
    a = -1.8 * np.log10(6.9 / Re + K_D)
    f = (1.0 / a) ** 2
    return f


def friction_term_J(f: float, D: float, V: float | np.ndarray, dt: float) -> float | np.ndarray:
    """稳态达西摩阻项 J = f dt V|V| / (2 D)"""
    return f * dt * V * abs(V) / (2.0 * D)


# =====================================================================
# Brunone 非定常摩阻
# =====================================================================
def brunone_k(Re: float) -> float:
    """
    Brunone 系数 k = sqrt(C)/2（标量版）
    C 为 Vardy 剪切衰减系数：
        层流 (Re<2000): C = 4.76e-3
        紊流 (Re≥2000): C = 7.41 / Re^(log10(14.3/Re^0.05))
    """
    if Re < 1.0:
        return 0.0
    if Re < 2000.0:
        C = 4.76e-3
    else:
        C = 7.41 / Re ** (np.log10(14.3 / Re ** 0.05))
    return np.sqrt(C) / 2.0


def brunone_k_vec(Re_arr: np.ndarray) -> np.ndarray:
    """brunone_k 的向量化版本"""
    Re_arr = np.asarray(Re_arr, dtype=np.float64)
    C = np.full_like(Re_arr, 0.0)
    valid = Re_arr >= 1.0
    laminar = valid & (Re_arr < 2000.0)
    C[laminar] = 4.76e-3
    turbulent = valid & (Re_arr >= 2000.0)
    Re_t = Re_arr[turbulent]
    C[turbulent] = 7.41 / Re_t ** (np.log10(14.3 / Re_t ** 0.05))
    return np.sqrt(C) / 2.0


def brunone_friction_Ju(k: float, dt: float,
                         dVdt: float, dVdx: float,
                         V: float, a: float) -> float:
    """Brunone 非定常摩阻项 J_u"""
    return (k / 2.0) * dt * (dVdt + a * np.sign(V) * np.abs(dVdx))


# =====================================================================
# 裂缝节点求解器
# =====================================================================
def solve_fracture_node(
    Cp_f: float, Cm_f: float,
    H_prev_f: float, A: float, ga: float,
    Cf: float, kleak: float, H_ext: float, dt: float,
    newton_tol: float = 1.0e-10, newton_max_iter: int = 20,
) -> Tuple[float, float, float, float]:
    """
    求解裂缝节点：alpha*H_P + beta*sqrt(H_P - H_ext) + gamma = 0
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
            dH = f_val / f_prime if f_prime != 0.0 else 0.0
            H_P -= dH
            if abs(dH) < newton_tol:
                break

    V_left = Cp_f - ga * H_P
    V_right = -Cm_f + ga * H_P
    Q_f = A * (V_left - V_right)
    return H_P, V_left, V_right, Q_f


# =====================================================================
# 主仿真器 (Stage 1 修复版)
# =====================================================================
def simulate_wellbore(
    cfg: MocConfig,
    fracture_positions: Optional[List[float]] = None,
    fracture_Cf: Optional[List[float]] = None,
    fracture_kleak: Optional[List[float]] = None,
    H_ext: float = 0.0,
    store_full_field: bool = True,
    snapshot_times: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[float, np.ndarray, np.ndarray, int, int], None]] = None,
    progress_interval_steps: Optional[int] = None,
) -> Dict:
    """
    运行 1D 井筒 MOC 仿真 (Stage 1 严格死端与死水段静止化)。
    """
    # 强制锁定 Stage 1 边界
    cfg.toe_bc = "dead_end"

    N = cfg.N
    dt = cfg.dt_adj
    dx = cfg.dx
    a = cfg.a_adj
    D = cfg.wellbore_diameter
    nu = cfg.fluid_viscosity
    K_D = cfg.roughness_height / D
    theta = cfg.theta
    ga = G / a
    V0 = cfg.initial_velocity
    H0 = cfg.initial_head
    t_s = cfg.pump_shut_time
    t_c = cfg.pump_closure_duration
    n_steps = cfg.n_steps
    area = cfg.area

    # 稳态达西 f
    Re0 = reynolds(V0, D, nu)
    f_steady = darcy_friction_factor(Re0, K_D, model=cfg.friction_model)
    use_quasi = (cfg.friction_model == "quasi-steady")
    use_brunone = (cfg.friction_model == "brunone")
    k_scale = float(cfg.brunone_k_scale)

    # ── 裂缝设置 ─────────────────────────────────────────────
    has_fractures = fracture_positions is not None and len(fracture_positions) > 0
    if has_fractures:
        n_frac = len(fracture_positions)
        if fracture_Cf is None:
            fracture_Cf = [0.0] * n_frac
        if fracture_kleak is None:
            fracture_kleak = [0.0] * n_frac
        frac_indices = []
        for xf in fracture_positions:
            idx = round(xf / dx)
            idx = max(1, min(N - 1, idx))
            frac_indices.append(idx)
        frac_Cf_arr = np.array(fracture_Cf, dtype=np.float64)
        frac_kleak_arr = np.array(fracture_kleak, dtype=np.float64)
        if len(set(frac_indices)) != len(frac_indices):
            raise ValueError(
                f"裂缝对齐后网格索引重合: {frac_indices}，请增大缝间距或减小 dt"
            )
    else:
        n_frac = 0
        frac_indices = []
        frac_Cf_arr = np.array([])
        frac_kleak_arr = np.array([])

    # ── 初始条件 (Stage 1 核心修复) ─────────────────────────
    x_grid = np.linspace(0.0, cfg.wellbore_length, N + 1)
    H = np.full(N + 1, H0, dtype=np.float64)
    V = np.full(N + 1, V0, dtype=np.float64)

    # 阶段一（Stage 1）核心物理修复：
    # 1. 严格死端边界锁定 toe_bc = 'dead_end' (Gamma = +1.0)
    # 2. 初始场初始化时，显式将末簇裂缝至趾端死水段区间的初始流速置零：
    #    V(x > x_{f,last}, 0) = 0
    #    彻底消除整管盲目赋初流速导致的开局撞墙假水击冲击波。
    # 3. 稳态水头场平衡：流动段 (x <= x_last) 存在与稳态流速匹配的达西摩阻水力坡降，
    #    死水段 (x > x_last) 流速恒零、水力坡降恒零、水头严格水平静止为 H0。
    #    由此保证停泵前 (t < t_s) 井口水头严格水平恒定，波动残差达到机器浮点精度。
    if has_fractures:
        last_frac_idx = frac_indices[-1]
        x_last = x_grid[last_frac_idx]
    else:
        last_frac_idx = N
        x_last = cfg.wellbore_length

    # 死水段流速显式置零
    V[last_frac_idx + 1:] = 0.0
    V[-1] = 0.0

    # 计算流动段稳态达西水力坡降
    Re0_check = reynolds(V0, D, nu)
    f0_check = darcy_friction_factor(Re0_check, K_D, model=cfg.friction_model)
    friction_slope = f0_check * V0 * abs(V0) / (2.0 * G * D)

    # 死水段水头为基准 H0，流动段逆流向上按摩阻梯度提升至井口
    H[last_frac_idx:] = H0
    H[:last_frac_idx] = H0 + friction_slope * (x_last - x_grid[:last_frac_idx])

    # 双流速数组：非缝节点 V_left = V_right = V
    V_prev_left = V.copy()
    V_prev_right = V.copy()
    if has_fractures:
        # 末簇裂缝右侧即为死水段，初值流速严格为 0
        V_prev_right[last_frac_idx] = 0.0

    # Brunone 需要上上步速度（用于 ∂V/∂t）
    V_prev2_left = V_prev_left.copy()
    V_prev2_right = V_prev_right.copy()

    # 时间序列容器
    timestamps = np.zeros(n_steps + 1)
    wh_head_hist = np.zeros(n_steps + 1)
    wh_vel_hist = np.zeros(n_steps + 1)
    toe_head_hist = np.zeros(n_steps + 1)
    toe_vel_hist = np.zeros(n_steps + 1)
    wh_head_hist[0] = H[0]
    wh_vel_hist[0] = V[0]
    toe_head_hist[0] = H[-1]
    toe_vel_hist[0] = V[-1]

    if store_full_field:
        head_hist = np.zeros((n_steps + 1, N + 1))
        vel_hist = np.zeros((n_steps + 1, N + 1))
        head_hist[0] = H
        vel_hist[0] = V
    else:
        head_hist = None
        vel_hist = None

    snapshots = {}
    snapshot_steps = set()
    if snapshot_times:
        for st in snapshot_times:
            si = round(st / dt)
            if 0 <= si <= n_steps:
                snapshot_steps.add(si)

    if progress_callback is not None:
        if progress_interval_steps is None:
            progress_interval_steps = max(1, n_steps // 100)
        progress_callback(0.0, H.copy(), V.copy(), 0, n_steps)

    frac_head_hist = np.zeros((0, 0))
    frac_Q_hist = np.zeros((0, 0))
    if has_fractures:
        frac_head_hist = np.zeros((n_steps + 1, n_frac))
        frac_Q_hist = np.zeros((n_steps + 1, n_frac))
        for k, idx in enumerate(frac_indices):
            frac_head_hist[0, k] = H[idx]

    H_prev = H.copy()

    # ── 时间推进 ─────────────────────────────────────────────
    for n in range(1, n_steps + 1):
        t = n * dt
        timestamps[n] = t

        # ── 1. 内节点 1..N-1（向量化，使用双流速数组）────────
        V1 = V_prev_right[:-2]   # nodes 0..N-2 的下游侧 → C⁺ for nodes 1..N-1
        H1 = H_prev[:-2]
        V2 = V_prev_left[2:]     # nodes 2..N 的上游侧 → C⁻ for nodes 1..N-1
        H2 = H_prev[2:]

        if use_quasi:
            Re1 = np.abs(V1) * D / nu
            Re2 = np.abs(V2) * D / nu
            f1 = np.array([darcy_friction_factor(r, K_D, "quasi-steady") for r in Re1])
            f2 = np.array([darcy_friction_factor(r, K_D, "quasi-steady") for r in Re2])
            J1 = f1 * dt * V1 * np.abs(V1) / (2.0 * D)
            J2 = f2 * dt * V2 * np.abs(V2) / (2.0 * D)
        else:
            J1 = friction_term_J(f_steady, D, V1, dt)
            J2 = friction_term_J(f_steady, D, V2, dt)

        # Brunone 非定常摩阻项（n≥2 时有 V_prev2 可算 dV/dt）
        if use_brunone and n >= 2:
            V_smooth = 0.05

            dVdt1 = (V_prev_right[:-2] - V_prev2_right[:-2]) / dt
            dVdx1 = (V_prev_right[1:-1] - V_prev_right[:-2]) / dx
            Re1b = np.abs(V1) * D / nu
            k1 = brunone_k_vec(Re1b) * k_scale
            sign_V1 = np.tanh(V1 / V_smooth)
            Ju1 = (k1 / 2.0) * dt * (dVdt1 + a * sign_V1 * np.abs(dVdx1))

            dVdt2 = (V_prev_left[2:] - V_prev2_left[2:]) / dt
            dVdx2 = (V_prev_left[2:] - V_prev_left[1:-1]) / dx
            Re2b = np.abs(V2) * D / nu
            k2 = brunone_k_vec(Re2b) * k_scale
            sign_V2 = np.tanh(V2 / V_smooth)
            Ju2 = (k2 / 2.0) * dt * (dVdt2 + a * sign_V2 * np.abs(dVdx2))

            # 裂缝邻域 Ju 置零（隔离 V 跃变，防 dV/dx 爆炸）
            if has_fractures:
                for i_f in frac_indices:
                    Ju1[i_f - 1] = 0.0
                    Ju2[i_f - 1] = 0.0

            J1 = J1 + Ju1
            J2 = J2 + Ju2

        Cp = V1 + ga * H1 - J1 + ga * dt * V1 * theta      # C⁺ 系数, shape (N-1,)
        Cm = -V2 + ga * H2 + J2 + ga * dt * V2 * theta     # C⁻ 系数, shape (N-1,)

        H_new = np.empty(N + 1)
        V_new = np.empty(N + 1)

        H_inner = (Cp + Cm) / (2.0 * ga)
        V_inner = Cp - ga * H_inner
        H_new[1:-1] = H_inner
        V_new[1:-1] = V_inner

        # ── 1b. 裂缝节点覆盖 ─────────────────────────────────
        if has_fractures:
            for k, i_f in enumerate(frac_indices):
                Cp_f = Cp[i_f - 1]
                Cm_f = Cm[i_f - 1]
                H_old_avg = 0.5 * (H_prev[i_f - 1] + H_prev[i_f + 1])

                H_f, V_left, V_right, Q_f = solve_fracture_node(
                    Cp_f, Cm_f, H_old_avg, area, ga,
                    frac_Cf_arr[k], frac_kleak_arr[k], H_ext, dt,
                )

                H_new[i_f] = H_f
                V_new[i_f] = V_left
                frac_head_hist[n, k] = H_f
                frac_Q_hist[n, k] = Q_f

        # ── 2. 井口（左边界，i=0），仅有 C⁻来自下游 i=1 ────
        V2_0 = V_prev_left[1]
        H2_0 = H_prev[1]
        if use_quasi:
            Re_0 = abs(V2_0) * D / nu
            f_0 = darcy_friction_factor(Re_0, K_D, "quasi-steady")
        else:
            f_0 = f_steady
        J_0 = friction_term_J(f_0, D, V2_0, dt)
        if use_brunone and n >= 2:
            dVdt_0 = (V_prev_left[1] - V_prev2_left[1]) / dt
            dVdx_0 = (V_prev_left[1] - V_prev_left[0]) / dx
            Re_0b = abs(V2_0) * D / nu
            k_0 = brunone_k(Re_0b) * k_scale
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

        # ── 3. 趾端（右边界，i=N），仅有 C⁺来自上游 i=N-1 ──
        # Stage 1: 严格死端边界 (V_toe = 0, Gamma = +1.0)
        V1_N = V_prev_right[N - 1]
        H1_N = H_prev[N - 1]
        if use_quasi:
            Re_N = abs(V1_N) * D / nu
            f_N = darcy_friction_factor(Re_N, K_D, "quasi-steady")
        else:
            f_N = f_steady
        J_N = friction_term_J(f_N, D, V1_N, dt)
        if use_brunone and n >= 2:
            dVdt_N = (V_prev_right[N - 1] - V_prev2_right[N - 1]) / dt
            dVdx_N = (V_prev_right[N] - V_prev_right[N - 1]) / dx
            Re_Nb = abs(V1_N) * D / nu
            k_N = brunone_k(Re_Nb) * k_scale
            J_N += brunone_friction_Ju(k_N, dt, dVdt_N, dVdx_N, V1_N, a)
        Cp_N = V1_N + ga * H1_N - J_N + ga * dt * V1_N * theta

        V_new[N] = 0.0
        H_new[N] = Cp_N / ga

        # ── 4. 记录 + 更新双流速数组 ─────────────────────────
        wh_head_hist[n] = H_new[0]
        wh_vel_hist[n] = V_new[0]
        toe_head_hist[n] = H_new[-1]
        toe_vel_hist[n] = V_new[-1]
        if store_full_field and head_hist is not None and vel_hist is not None:
            head_hist[n] = H_new
            vel_hist[n] = V_new
        if n in snapshot_steps:
            snapshots[n] = {'H': H_new.copy(), 'V': V_new.copy(), 't': t}

        if progress_callback is not None and (
            n % progress_interval_steps == 0 or n == n_steps
        ):
            progress_callback(t, H_new.copy(), V_new.copy(), n, n_steps)

        if use_brunone:
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

    result = {
        "timestamps": timestamps,
        "head": head_hist,
        "velocity": vel_hist,
        "wellhead_head": wh_head_hist,
        "wellhead_velocity": wh_vel_hist,
        "toe_head": toe_head_hist,
        "toe_velocity": toe_vel_hist,
        "x_grid": x_grid,
        "cfg": cfg,
        "fracture_indices": frac_indices,
        "snapshots": snapshots,
    }
    if has_fractures:
        result["fracture_heads"] = frac_head_hist
        result["fracture_Qs"] = frac_Q_hist
    return result


def simulate_case(config: Dict, friction_model: str = "brunone") -> Dict:
    """供外界调用的统一入口"""
    cfg = MocConfig(
        wellbore_length=float(config.get("wellbore_length", 1000.0)),
        wellbore_diameter=float(config.get("main_diameter_mm", 139.7)) / 1000.0 if "main_diameter_mm" in config else float(config.get("wellbore_diameter", 0.1397)),
        fluid_density=float(config.get("fluid_density", 1000.0)),
        fluid_viscosity=float(config.get("fluid_viscosity", 1.0e-6)),
        wavespeed=float(config.get("main_wavespeed", 1450.0)) if "main_wavespeed" in config else float(config.get("wavespeed", 1450.0)),
        roughness_height=float(config.get("roughness_height", 4.5e-5)),
        friction_model=str(config.get("friction_model", friction_model)),
        dt=float(config.get("time_step", 1.0e-3)) if "time_step" in config else float(config.get("dt", 1.0e-3)),
        tf=float(config.get("simulation_time", 3.0)) if "simulation_time" in config else float(config.get("tf", 3.0)),
        wellhead_bc=str(config.get("wellhead_bc", "velocity_step")),
        pump_shut_time=float(config.get("pump_shut_time", 1.0)),
        pump_closure_duration=float(config.get("pump_closure_duration", 1.0e-3)),
        initial_velocity=float(config.get("initial_velocity", 1.0)),
        theta=float(config.get("theta", 0.0)),
        initial_head=float(config.get("initial_head", 300.0)),
        toe_bc="dead_end",
        toe_head=float(config.get("toe_head", 300.0)),
    )

    fracture_positions = config.get("fracture_positions", [])
    fracture_Cf = config.get("fracture_Cf", None)
    if fracture_Cf is None and "Cf" in config:
        fracture_Cf = [float(config["Cf"])] * len(fracture_positions)
    fracture_kleak = config.get("fracture_kleak", None)
    if fracture_kleak is None and "kleak" in config:
        fracture_kleak = [float(config["kleak"])] * len(fracture_positions)
    H_ext = float(config.get("H_ext", 0.0))

    n_steps_est = round(cfg.tf / cfg.dt)
    store_full = (n_steps_est * cfg.N) < 5_000_000

    result = simulate_wellbore(
        cfg,
        fracture_positions=fracture_positions if fracture_positions else None,
        fracture_Cf=fracture_Cf,
        fracture_kleak=fracture_kleak,
        H_ext=H_ext,
        store_full_field=store_full,
    )

    junction_heads = []
    if fracture_positions:
        frac_heads = result.get("fracture_heads", np.zeros((len(result["timestamps"]), 0)))
        for k in range(frac_heads.shape[1]):
            junction_heads.append(frac_heads[:, k])

    return {
        "timestamps": result["timestamps"],
        "wellhead_head": result["wellhead_head"],
        "toe_head": result["toe_head"],
        "junction_heads": junction_heads,
        "x_grid": result["x_grid"],
        "head_field": result["head"],
        "velocity_field": result["velocity"],
        "cfg": cfg,
    }
