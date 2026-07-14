# -*- coding: utf-8 -*-
"""
自研轻量 MOC 核心 — 井筒-裂缝系统水击波仿真器（路线 B / Step 1+3）

物理模型
--------
1. 1D 井筒，单一波速 a，单管集中参数 MOC（无 EPANET 耦合）
2. 流体：滑溜水（牛顿近似，ρ、ν 显式参数）
3. 井口边界：停泵瞬时流速阶跃 V=0（柱塞泵近似）
4. 趾端：死端 V=0（全反射，反射系数 +1）或水库 H=常数（反射系数 -1）
5. 内节点：标准 C⁺/C⁻ 联立，稳态达西摩阻
6. 缝节点（Step 3）：集总柔度 + 分布滤失边界
       Q_f = C_f · dH/dt + k_leak · √(H - H_ext)
   连续性: A·V_left - A·V_right = Q_f
   代入 C⁺/C⁻ + 半隐式离散 → 关于 H_P 的非线性方程，Newton 迭代求解

控制方程
--------
∂H/∂t + (a²/g) ∂V/∂x = 0
∂V/∂t + g ∂H/∂x + (f/2D) V|V| + g θ = 0

特征线
------
C⁺: V_P = V_1 + (g/a) H_1 - J_1 + (g/a) Δt V_1 θ          (来自上游 i-1)
C⁻: V_P = -V_2 + (g/a) H_2 + J_2 + (g/a) Δt V_2 θ         (来自下游 i+1)
内节点: H_P = (C⁺ + C⁻) / (2 g/a),  V_P = C⁺ - (g/a) H_P

井口（左边界，仅有 C⁻来自下游）:
    t < t_s : V_P = V0            （稳态流速）
    t ≥ t_s : V_P = 0             （柱塞泵停泵，瞬时截流）
    H_P = (V_P + C⁻) / (g/a)

趾端（右边界，仅有 C⁺来自上游）:
    dead_end : V_P = 0, H_P = C⁺ / (g/a)           （反射系数 +1）
    reservoir: H_P = H_toe, V_P = C⁺ - (g/a) H_P   （反射系数 -1）

缝节点（内部，C⁺来自上游 + C⁻来自下游 + 侧向 Q_f）:
    α·H_P + β·√(H_P - H_ext) + γ = 0
    α = 2·A·g/a + C_f/Δt,  β = k_leak,  γ = -A·(C⁺+C⁻) - (C_f/Δt)·H_P^{n-1}
    V_left  = C⁺ - (g/a)·H_P
    V_right = -C⁻ + (g/a)·H_P
    Q_f = A·(V_left - V_right)

双流速数组: V_prev_left[i] 供上游邻居 C⁻ 用；V_prev_right[i] 供下游邻居 C⁺ 用。
非缝节点 V_left = V_right = V。

Courant 准则: a Δt / Δx = 1（精确，无数值耗散）
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =====================================================================
# 物理常数
# =====================================================================
G = 9.81  # 重力加速度 [m/s^2]


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class MocConfig:
    """自研 MOC 仿真参数（从 config.json / config.yaml 读取或直接构造）"""
    # 几何
    wellbore_length: float = 1000.0          # [m]
    wellbore_diameter: float = 0.1397        # [m]  (5.5" 套管)
    # 流体（滑溜水）
    fluid_density: float = 1000.0            # [kg/m^3]
    fluid_viscosity: float = 1.0e-6          # [m^2/s] 运动黏度
    wavespeed: float = 1450.0                # [m/s]
    # 摩阻
    roughness_height: float = 4.5e-5         # [m] 商用钢绝对粗糙度
    friction_model: str = "steady"           # steady / quasi-steady
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
    initial_head: float = 300.0              # [m] 稳态井口水头（停泵前）
    # 趾端边界
    toe_bc: str = "dead_end"                 # dead_end (V=0) / reservoir (H=const)
    toe_head: float = 300.0                  # [m] 仅 toe_bc='reservoir' 用

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


# =====================================================================
# 摩阻
# =====================================================================
def reynolds(V: float, D: float, nu: float) -> float:
    return abs(V) * D / nu


def darcy_friction_factor(Re: float, K_D: float, model: str = "steady") -> float:
    """
    达西摩阻系数 f
    model='steady'      : 使用层流/紊流显式近似（不随 Re 更新，由外部稳态 f 提供）
    model='quasi-steady': 按 Re 实时更新（Zigrand-Swamee 显式近似）
    """
    if Re < 1e-3:
        return 0.0
    if Re < 2000:
        # 层流
        return 64.0 / Re
    # 紊流 — Zigrand-Swami 显式近似（与 TSNet 一致，便于对照）
    a = -1.8 * np.log10(6.9 / Re + K_D)
    f = (1.0 / a) ** 2
    return f


def friction_term_J(f: float, D: float, V: float | np.ndarray, dt: float) -> float | np.ndarray:
    """稳态达西摩阻项 J = f dt V|V| / (2 D)"""
    return f * dt * V * abs(V) / (2.0 * D)


# =====================================================================
# Brunone 非定常摩阻（Step 3 扩展）
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
    """
    brunone_k 的向量化版本（支持 numpy 数组输入，消除 Python 循环瓶颈）。
    """
    Re_arr = np.asarray(Re_arr, dtype=np.float64)
    C = np.full_like(Re_arr, 0.0)
    # 有效区域 Re ≥ 1
    valid = Re_arr >= 1.0
    # 层流 1 ≤ Re < 2000
    laminar = valid & (Re_arr < 2000.0)
    C[laminar] = 4.76e-3
    # 紊流 Re ≥ 2000
    turbulent = valid & (Re_arr >= 2000.0)
    Re_t = Re_arr[turbulent]
    C[turbulent] = 7.41 / Re_t ** (np.log10(14.3 / Re_t ** 0.05))
    return np.sqrt(C) / 2.0


def brunone_friction_Ju(k: float, dt: float,
                         dVdt: float, dVdx: float,
                         V: float, a: float) -> float:
    """
    Brunone 非定常摩阻项 J_u（量纲 m/s，与稳态 J_s 一致）

    J_u = (k/2) · dt · (∂V/∂t + a·sign(V)·|∂V/∂x|)

    参数
    ----
    k     : Brunone 系数（无量纲）
    dt    : 时间步长 [s]
    dVdt  : 局部瞬时加速度 ∂V/∂t [m/s²]
    dVdx  : 对流瞬时加速度 ∂V/∂x [1/s]
    V     : 流速（用于 sign）[m/s]
    a     : 波速 [m/s]
    """
    return (k / 2.0) * dt * (dVdt + a * np.sign(V) * np.abs(dVdx))


# =====================================================================
# 裂缝节点求解器（Step 3）
# =====================================================================
def solve_fracture_node(
    Cp_f: float, Cm_f: float,
    H_prev_f: float, A: float, ga: float,
    Cf: float, kleak: float, H_ext: float, dt: float,
    newton_tol: float = 1.0e-10, newton_max_iter: int = 20,
) -> Tuple[float, float, float, float]:
    """
    求解裂缝节点：α·H_P + β·√(H_P - H_ext) + γ = 0

    参数
    ----
    Cp_f      : C⁺ 系数（来自上游 i_f-1）
    Cm_f      : C⁻ 系数（来自下游 i_f+1）
    H_prev_f  : 上一时步该缝节点的水头（用于 dH/dt 半隐式）
    A         : 井筒截面积 [m²]
    ga        : g/a
    Cf        : 裂缝柔度 [m²]
    kleak     : 滤失系数 [m²/s/√m]
    H_ext     : 地层孔隙压力水头 [m]
    dt        : 时间步长 [s]

    返回
    ----
    H_P    : 裂缝节点水头 [m]
    V_left : 上游侧流速 [m/s]
    V_right: 下游侧流速 [m/s]
    Q_f    : 裂缝侧向流量 [m³/s]（正=流入裂缝）
    """
    alpha = 2.0 * A * ga + Cf / dt
    beta = kleak
    gamma = -A * (Cp_f + Cm_f) - (Cf / dt) * H_prev_f

    # Newton 迭代： f(H) = α·H + β·√(H - H_ext) + γ
    #               f'(H) = α + β/(2·√(H - H_ext))
    # 初值：上一时步水头（稳态附近收敛快）
    H_P = H_prev_f

    if kleak == 0.0:
        # 纯柔度（无线性滤波失）：退化为线性方程 α·H_P + γ = 0
        H_P = -gamma / alpha
    else:
        for _ in range(newton_max_iter):
            arg = H_P - H_ext
            if arg <= 0.0:
                # 水头低于地层压力：滤失反向（或为零），简化为无滤失
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
# 主仿真器
# =====================================================================
def simulate_wellbore(
    cfg: MocConfig,
    fracture_positions: Optional[List[float]] = None,
    fracture_Cf: Optional[List[float]] = None,
    fracture_kleak: Optional[List[float]] = None,
    H_ext: float = 0.0,
    store_full_field: bool = True,
    snapshot_times: Optional[List[float]] = None,
) -> Dict:
    """
    运行 1D 井筒 MOC 仿真。

    Step 1 阶段：忽略 fracture_*（fracture_positions=None 或空）。
    Step 3 起：fracture_positions 给定后启用集总柔度裂缝边界。

    参数
    ----
    cfg                : MocConfig
    fracture_positions : 裂缝位置列表 [m]（相对井口），None=无缝
    fracture_Cf        : 各缝柔度 [m²]，None 时默认 0（无效缝）
    fracture_kleak     : 各缝滤失系数 [m²/s/√m]，None 时默认 0
    H_ext              : 地层孔隙压力水头 [m]
    store_full_field   : True=存储完整 head(n+1,N+1)/velocity(n+1,N+1)；
                         False=仅存 1D 时程（大仿真时省内存，~5GB→~10MB）
    snapshot_times     : store_full_field=False 时，指定要保存空间快照的时刻 [s]

    返回
    ----
    dict :
        timestamps, head, velocity, wellhead_head, wellhead_velocity,
        toe_head, toe_velocity, x_grid, cfg,
        fracture_indices   : 各缝对齐到的网格索引
        fracture_heads     : (n_steps+1, n_frac) 各缝节点水头时程
        fracture_Qs        : (n_steps+1, n_frac) 各缝侧向流量时程
    """
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

    # ── 裂缝设置 ─────────────────────────────────────────────
    has_fractures = fracture_positions is not None and len(fracture_positions) > 0
    if has_fractures:
        n_frac = len(fracture_positions)
        if fracture_Cf is None:
            fracture_Cf = [0.0] * n_frac
        if fracture_kleak is None:
            fracture_kleak = [0.0] * n_frac
        # 缝位置对齐到最近网格点
        frac_indices = []
        for xf in fracture_positions:
            idx = round(xf / dx)
            idx = max(1, min(N - 1, idx))   # 不能在边界上
            frac_indices.append(idx)
        frac_Cf_arr = np.array(fracture_Cf, dtype=np.float64)
        frac_kleak_arr = np.array(fracture_kleak, dtype=np.float64)
        # 检查缝不重合
        if len(set(frac_indices)) != len(frac_indices):
            raise ValueError(
                f"裂缝对齐后网格索引重合: {frac_indices}，请增大缝间距或减小 dt"
            )
    else:
        n_frac = 0
        frac_indices = []
        frac_Cf_arr = np.array([])
        frac_kleak_arr = np.array([])

    frac_index_set = set(frac_indices)

    # ── 初始条件 ─────────────────────────────────────────────
    x_grid = np.linspace(0.0, cfg.wellbore_length, N + 1)
    H = np.full(N + 1, H0, dtype=np.float64)
    V = np.full(N + 1, V0, dtype=np.float64)

    if cfg.toe_bc == "dead_end":
        V[-1] = 0.0
        H[-1] = H[-2] + (a / G) * V[-2]
    elif cfg.toe_bc == "reservoir":
        H[-1] = cfg.toe_head
        V[-1] = V0
        Re0_check = reynolds(V0, D, nu)
        f0_check = darcy_friction_factor(Re0_check, K_D, model=cfg.friction_model)
        friction_slope = f0_check * V0 * abs(V0) / (2.0 * G * D)
        H[:-1] = cfg.toe_head + friction_slope * (cfg.wellbore_length - x_grid[:-1])
    else:
        raise ValueError(f"未知 toe_bc: {cfg.toe_bc}")

    # 含滤失裂缝的稳态：沿井筒 V 递减（滤失消耗流量）
    # 简化处理：稳态时 dH/dt=0 → Q_f = k_leak·√(H-H_ext)
    # 严格稳态需迭代求解 V(x) 分布；这里近似处理：
    # 若 k_leak 很小（验证场景），V≈V0 即可
    # 若 k_leak 显著，在 simulate_case 中由调用方提供稳态 V 分布

    # 双流速数组：非缝节点 V_left = V_right = V
    V_prev_left = V.copy()
    V_prev_right = V.copy()
    # Brunone 需要上上步速度（用于 ∂V/∂t）
    V_prev2_left = V.copy()
    V_prev2_right = V.copy()

    # 时间序列容器
    timestamps = np.zeros(n_steps + 1)
    # 1D 时程（始终存储，内存小）
    wh_head_hist = np.zeros(n_steps + 1)
    wh_vel_hist = np.zeros(n_steps + 1)
    toe_head_hist = np.zeros(n_steps + 1)
    toe_vel_hist = np.zeros(n_steps + 1)
    wh_head_hist[0] = H[0]
    wh_vel_hist[0] = V[0]
    toe_head_hist[0] = H[-1]
    toe_vel_hist[0] = V[-1]

    # 完整时空场（可选，大仿真时关闭省内存）
    if store_full_field:
        head_hist = np.zeros((n_steps + 1, N + 1))
        vel_hist = np.zeros((n_steps + 1, N + 1))
        head_hist[0] = H
        vel_hist[0] = V
    else:
        head_hist = None
        vel_hist = None

    # 空间快照（store_full_field=False 时用）
    snapshots = {}  # {step_index: {'H': array, 'V': array, 't': float}}
    snapshot_steps = set()
    if snapshot_times:
        for st in snapshot_times:
            si = round(st / dt)
            if 0 <= si <= n_steps:
                snapshot_steps.add(si)

    # 裂缝记录
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
        # C⁺ from i-1: V1 = V_prev_right[i-1]（i-1 的下游侧流速）
        # C⁻ from i+1: V2 = V_prev_left[i+1]（i+1 的上游侧流速）
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
        # ★ 用 tanh 平滑 sign(V) 过渡，消除 V≈0 处符号跳变冲击（锯齿根因）
        #   前向/后向差分保留 dV/dx 幅值 → 保留 Brunone 衰减强度
        # ★ 动作1：裂缝邻域 Ju 强制置零（V 在缝处有跃变，dV/dx 跨缝不可靠）
        if use_brunone and n >= 2:
            V_smooth = 0.05   # sign(V) 平滑阈值 [m/s]，tanh(V/V_smooth)

            # C⁺ 来源点 j=0..N-2（V_prev_right）：前向差分 dV/dx
            dVdt1 = (V_prev_right[:-2] - V_prev2_right[:-2]) / dt
            dVdx1 = (V_prev_right[1:-1] - V_prev_right[:-2]) / dx
            Re1b = np.abs(V1) * D / nu
            k1 = brunone_k_vec(Re1b)
            sign_V1 = np.tanh(V1 / V_smooth)   # 平滑符号函数
            Ju1 = (k1 / 2.0) * dt * (dVdt1 + a * sign_V1 * np.abs(dVdx1))

            # C⁻ 来源点 j=2..N（V_prev_left）：后向差分 dV/dx
            dVdt2 = (V_prev_left[2:] - V_prev2_left[2:]) / dt
            dVdx2 = (V_prev_left[2:] - V_prev_left[1:-1]) / dx
            Re2b = np.abs(V2) * D / nu
            k2 = brunone_k_vec(Re2b)
            sign_V2 = np.tanh(V2 / V_smooth)
            Ju2 = (k2 / 2.0) * dt * (dVdt2 + a * sign_V2 * np.abs(dVdx2))

            # ★ 动作1：裂缝邻域 Ju 置零（隔离 V 跃变，防 dV/dx 爆炸）
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

        # 标准内节点解（含缝位置，后续覆盖）
        H_inner = (Cp + Cm) / (2.0 * ga)
        V_inner = Cp - ga * H_inner
        H_new[1:-1] = H_inner
        V_new[1:-1] = V_inner

        # ── 1b. 裂缝节点覆盖 ─────────────────────────────────
        if has_fractures:
            for k, i_f in enumerate(frac_indices):
                Cp_f = Cp[i_f - 1]
                Cm_f = Cm[i_f - 1]

                # ★ 终极修复：使用同色网格（邻居节点）的空间平均代替本节点历史值
                # 这样 H_old 就与 Cp_f, Cm_f 处于同一个特征线网格体系中，彻底消除奇偶解耦
                H_old_avg = 0.5 * (H_prev[i_f - 1] + H_prev[i_f + 1])

                # 恢复最标准、最稳定、无跨步的 dt 后向欧拉求解
                H_f, V_left, V_right, Q_f = solve_fracture_node(
                    Cp_f, Cm_f, H_old_avg, area, ga,
                    frac_Cf_arr[k], frac_kleak_arr[k], H_ext, dt,
                )

                H_new[i_f] = H_f
                V_new[i_f] = V_left   # 主数组存上游侧（供记录与显示）
                frac_head_hist[n, k] = H_f
                frac_Q_hist[n, k] = Q_f

        # ── 2. 井口（左边界，i=0），仅有 C⁻来自下游 i=1 ────
        V2_0 = V_prev_left[1]   # node 1 的上游侧流速（=V[1] 非缝）
        H2_0 = H_prev[1]
        if use_quasi:
            Re_0 = abs(V2_0) * D / nu
            f_0 = darcy_friction_factor(Re_0, K_D, "quasi-steady")
        else:
            f_0 = f_steady
        J_0 = friction_term_J(f_0, D, V2_0, dt)
        # Brunone（井口 C⁻ 来源 node 1：后向差分 dV/dx）
        if use_brunone and n >= 2:
            dVdt_0 = (V_prev_left[1] - V_prev2_left[1]) / dt
            dVdx_0 = (V_prev_left[1] - V_prev_left[0]) / dx
            Re_0b = abs(V2_0) * D / nu
            k_0 = brunone_k(Re_0b)
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
        V1_N = V_prev_right[N - 1]   # node N-1 的下游侧流速
        H1_N = H_prev[N - 1]
        if use_quasi:
            Re_N = abs(V1_N) * D / nu
            f_N = darcy_friction_factor(Re_N, K_D, "quasi-steady")
        else:
            f_N = f_steady
        J_N = friction_term_J(f_N, D, V1_N, dt)
        # Brunone（趾端 C⁺ 来源 node N-1：前向差分 dV/dx）
        if use_brunone and n >= 2:
            dVdt_N = (V_prev_right[N - 1] - V_prev2_right[N - 1]) / dt
            dVdx_N = (V_prev_right[N] - V_prev_right[N - 1]) / dx
            Re_Nb = abs(V1_N) * D / nu
            k_N = brunone_k(Re_Nb)
            J_N += brunone_friction_Ju(k_N, dt, dVdt_N, dVdx_N, V1_N, a)
        Cp_N = V1_N + ga * H1_N - J_N + ga * dt * V1_N * theta

        if cfg.toe_bc == "dead_end":
            V_new[N] = 0.0
            H_new[N] = Cp_N / ga
        elif cfg.toe_bc == "reservoir":
            H_new[N] = cfg.toe_head
            V_new[N] = Cp_N - ga * H_new[N]
        else:
            raise ValueError(f"未知 toe_bc: {cfg.toe_bc}")

        # ── 4. 记录 + 更新双流速数组 ─────────────────────────
        wh_head_hist[n] = H_new[0]
        wh_vel_hist[n] = V_new[0]
        toe_head_hist[n] = H_new[-1]
        toe_vel_hist[n] = V_new[-1]
        if store_full_field and head_hist is not None and vel_hist is not None:
            head_hist[n] = H_new
            vel_hist[n] = V_new
        # 空间快照
        if n in snapshot_steps:
            snapshots[n] = {'H': H_new.copy(), 'V': V_new.copy(), 't': t}

        # Brunone 需要上上步速度
        if use_brunone:
            V_prev2_left = V_prev_left.copy()
            V_prev2_right = V_prev_right.copy()

        # 更新双流速：非缝节点 left=right=V_new；缝节点 left/right 分离
        V_prev_left = V_new.copy()
        V_prev_right = V_new.copy()
        if has_fractures:
            for k, i_f in enumerate(frac_indices):
                # 重新计算该步 V_left/V_right（V_new[i_f] 存的是 V_left）
                V_prev_left[i_f] = V_new[i_f]
                # V_right = -Cm_f + ga·H_f，重新算
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


# =====================================================================
# dataset_builder 调用入口（与 simulator.py 的 _moc_simulate_case 对接）
# =====================================================================
def simulate_case(config: Dict, friction_model: str = "steady") -> Dict:
    """
    供 dataset_builder/simulator.py 调用的统一入口。

    参数
    ----
    config : dict
        case_dir/config.json 内容
    friction_model : str
        'steady' / 'quasi-steady' / 'brunone'（brunone 暂按 steady 处理，
        Step 3+ 加非定常项）

    返回
    ----
    dict :
        timestamps       : (n+1,) 时间 [s]
        wellhead_head    : (n+1,) 井口水头 [m]
        toe_head         : (n+1,) 趾端水头 [m]
        junction_heads   : list of (n+1,) 各裂缝分支点水头
        frac_end_heads   : list of (n+1,) 各裂缝末端水头（Step 3+）
    """
    cfg = MocConfig(
        wellbore_length=float(config.get("wellbore_length", 1000.0)),
        wellbore_diameter=float(config.get("main_diameter_mm", 139.7)) / 1000.0,
        fluid_density=float(config.get("fluid_density", 1000.0)),
        fluid_viscosity=float(config.get("fluid_viscosity", 1.0e-6)),
        wavespeed=float(config.get("main_wavespeed", 1450.0)),
        roughness_height=float(config.get("roughness_height", 4.5e-5)),
        friction_model=str(config.get("friction_model", "brunone")),
        dt=float(config.get("time_step", 1.0e-3)),
        tf=float(config.get("simulation_time", 3.0)),
        wellhead_bc=str(config.get("wellhead_bc", "velocity_step")),
        pump_shut_time=float(config.get("pump_shut_time", 1.0)),
        pump_closure_duration=float(config.get("pump_closure_duration", 1.0e-3)),
        initial_velocity=float(config.get("initial_velocity", 1.0)),
        theta=float(config.get("theta", 0.0)),
        initial_head=float(config.get("initial_head", 300.0)),
    )

    fracture_positions = config.get("fracture_positions", [])
    fracture_Cf = config.get("fracture_Cf", None)
    fracture_kleak = config.get("fracture_kleak", None)
    H_ext = float(config.get("H_ext", 0.0))

    # 大仿真时自动关闭完整场存储（>10万步 × >1000节点 → >1GB）
    n_steps_est = round(cfg.tf / cfg.dt)
    store_full = (n_steps_est * cfg.N) < 5_000_000   # 5M 点 ≈ 40MB

    # Step 3: 接缝（若 fracture_positions 非空）
    result = simulate_wellbore(
        cfg,
        fracture_positions=fracture_positions if fracture_positions else None,
        fracture_Cf=fracture_Cf,
        fracture_kleak=fracture_kleak,
        H_ext=H_ext,
        store_full_field=store_full,
    )

    # 提取各缝节点水头时程
    junction_heads = []
    frac_end_heads = []
    if fracture_positions:
        frac_heads = result.get("fracture_heads", np.zeros((len(result["timestamps"]), 0)))
        for k in range(frac_heads.shape[1]):
            junction_heads.append(frac_heads[:, k])

    return {
        "timestamps": result["timestamps"],
        "wellhead_head": result["wellhead_head"],
        "toe_head": result["toe_head"],
        "junction_heads": junction_heads,
        "frac_end_heads": frac_end_heads,
        "x_grid": result["x_grid"],
        "head_field": result["head"],
        "velocity_field": result["velocity"],
        "cfg": cfg,
    }


if __name__ == "__main__":
    # 简易 smoke test
    cfg = MocConfig(
        wellbore_length=1000.0,
        wellbore_diameter=0.1397,
        wavespeed=1450.0,
        dt=1.0e-3,
        tf=3.0,
        pump_shut_time=1.0,
        initial_velocity=1.0,
        initial_head=300.0,
        toe_bc="reservoir",
        toe_head=300.0,
    )
    print(f"N={cfg.N}, dx={cfg.dx:.4f} m, a_adj={cfg.a_adj:.4f} m/s, dt={cfg.dt_adj}")
    print(f"n_steps={cfg.n_steps}, area={cfg.area:.6f} m^2")
    res = simulate_wellbore(cfg)
    wh = res["wellhead_head"]
    ts_idx = int(cfg.pump_shut_time / cfg.dt)
    H_pre = wh[ts_idx - 1]
    H_post = wh[ts_idx]
    dH_sim = H_post - H_pre
    dH_ana = cfg.a_adj * cfg.initial_velocity / G
    print(f"井口 H[ts-1]={H_pre:.3f}, H[ts]={H_post:.3f}")
    print(f"井口 Joukowsky 跳变 dH_sim = {dH_sim:.3f} m")
    print(f"Joukowsky 解析 dH_ana = a V0 / g = {dH_ana:.3f} m")
    print(f"误差 = {abs(dH_sim - dH_ana)/dH_ana*100:.4f} %")
