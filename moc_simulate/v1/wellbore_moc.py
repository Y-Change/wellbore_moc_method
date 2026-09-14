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
        critical_vals = {
            "wellbore_length": self.wellbore_length,
            "wellbore_diameter": self.wellbore_diameter,
            "fluid_density": self.fluid_density,
            "fluid_viscosity": self.fluid_viscosity,
            "wavespeed": self.wavespeed,
            "roughness_height": self.roughness_height,
            "dt": self.dt,
            "tf": self.tf,
            "pump_shut_time": self.pump_shut_time,
            "pump_closure_duration": self.pump_closure_duration,
            "initial_velocity": self.initial_velocity,
            "initial_head": self.initial_head,
            "theta": self.theta,
            "toe_head": self.toe_head,
            "brunone_k_scale": self.brunone_k_scale,
        }
        for name, val in critical_vals.items():
            if not np.isfinite(val):
                raise ValueError(f"配置参数 {name}={val} 必须为有限数值 (np.isfinite)")

        if self.wellbore_length <= 0.0:
            raise ValueError(f"wellbore_length 必须大于 0: {self.wellbore_length}")
        if self.wellbore_diameter <= 0.0:
            raise ValueError(f"wellbore_diameter 必须大于 0: {self.wellbore_diameter}")
        if self.fluid_density <= 0.0:
            raise ValueError(f"fluid_density 必须大于 0: {self.fluid_density}")
        if self.fluid_viscosity <= 0.0:
            raise ValueError(f"fluid_viscosity 必须大于 0: {self.fluid_viscosity}")
        if self.wavespeed <= 0.0:
            raise ValueError(f"wavespeed 必须大于 0: {self.wavespeed}")
        if self.dt <= 0.0:
            raise ValueError(f"dt 必须大于 0: {self.dt}")
        if self.tf <= 0.0:
            raise ValueError(f"tf 必须大于 0: {self.tf}")

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
    Rp: float = 0.0,
    newton_tol: float = 1.0e-10, newton_max_iter: int = 20,
) -> Tuple[float, float, float, float, float]:
    """
    求解裂缝节点（支持水头柔度、等效滤失与可选射孔/近井阻力 Rp）。

    物理方程
    --------
    井筒节点水头 H_w 与裂缝内水头 H_f 满足射孔压降关系：
        H_w - H_f = Rp · Q_f · |Q_f|
    裂缝侧向流量（储液 + 滤失）：
        Q_f = Cf · (H_f - H_prev_f) / dt + kleak · √(max(H_f - H_ext, 0))
    井筒特征线方程：
        V_left = Cp_f - ga · H_w
        V_right = -Cm_f + ga · H_w
        Q_f = A · (V_left - V_right) = A · (Cp_f + Cm_f) - 2·A·ga · H_w
    ==> H_w = (Cp_f + Cm_f)/(2·ga) - Q_f / (2·A·ga)

    当 Rp == 0.0 时（无孔眼阻力基准）：
        H_w = H_f

    返回
    ----
    H_w    : 井筒网格节点水头 [m]
    H_f    : 裂缝内部水头 [m]
    V_left : 上游侧流速 [m/s]
    V_right: 下游侧流速 [m/s]
    Q_f    : 裂缝侧向流量 [m³/s]（正=流入裂缝）
    """
    if not (np.isfinite(Cp_f) and np.isfinite(Cm_f) and np.isfinite(H_prev_f)):
        raise ValueError(f"solve_fracture_node 收到非有限特征线或历史水头参数: Cp={Cp_f}, Cm={Cm_f}, H_prev={H_prev_f}")
    if not (np.isfinite(Cf) and np.isfinite(kleak) and np.isfinite(Rp) and np.isfinite(H_ext) and np.isfinite(dt) and np.isfinite(A) and np.isfinite(ga)):
        raise ValueError(f"solve_fracture_node 收到非有限物理参数: Cf={Cf}, kleak={kleak}, Rp={Rp}, H_ext={H_ext}, dt={dt}")
    if Cf < 0.0 or kleak < 0.0 or Rp < 0.0 or dt <= 0.0:
        raise ValueError(f"solve_fracture_node 物理参数必须非负: Cf={Cf}, kleak={kleak}, Rp={Rp}, dt={dt}")

    if Rp <= 0.0:
        alpha = 2.0 * A * ga + Cf / dt
        beta = kleak
        gamma = -A * (Cp_f + Cm_f) - (Cf / dt) * H_prev_f

        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(f"裂缝节点 alpha 非法: alpha={alpha}")

        # 解析精确解：在 y = sqrt(max(H - H_ext, 0)) 下为严格二次方程，彻底消除近 H_ext 处的导数奇异与震荡
        C = alpha * H_ext + gamma
        if C >= 0.0:
            H_P = -gamma / alpha
        else:
            disc = beta * beta - 4.0 * alpha * C
            if disc < 0.0 or not np.isfinite(disc):
                raise ValueError(f"裂缝节点二次判别式非法: disc={disc}")
            y = (-beta + np.sqrt(disc)) / (2.0 * alpha)
            H_P = H_ext + y * y

        if not np.isfinite(H_P):
            raise ValueError(f"裂缝节点求解产生非有限水头: H_P={H_P}")

        H_f = H_P
        H_w = H_f
        V_left = Cp_f - ga * H_w
        V_right = -Cm_f + ga * H_w
        Q_f = A * (V_left - V_right)
    else:
        # Rp > 0: 牛顿迭代求解裂缝内水头 H_f
        H_w0 = (Cp_f + Cm_f) / (2.0 * ga)
        C_dt = Cf / dt
        inv_2Aga = 1.0 / (2.0 * A * ga)
        H_f = H_prev_f

        converged = False
        for _ in range(newton_max_iter):
            arg = H_f - H_ext
            if arg <= 0.0:
                Q_leak = 0.0
                dQ_leak = 0.0
            else:
                sqrt_arg = np.sqrt(arg)
                Q_leak = kleak * sqrt_arg
                dQ_leak = kleak / (2.0 * sqrt_arg)

            Q_f = C_dt * (H_f - H_prev_f) + Q_leak
            dQ_f = C_dt + dQ_leak

            H_w = H_w0 - Q_f * inv_2Aga
            dH_w = -dQ_f * inv_2Aga

            loss = Rp * Q_f * abs(Q_f)
            dloss = 2.0 * Rp * abs(Q_f) * dQ_f

            f_val = H_w - H_f - loss
            f_prime = dH_w - 1.0 - dloss

            if f_prime == 0.0 or not np.isfinite(f_prime):
                raise ValueError(f"Newton (Rp>0) 导数奇异或非有限值: f_prime={f_prime}")
            dH = f_val / f_prime
            if not np.isfinite(dH):
                raise ValueError(f"Newton (Rp>0) 步长非有限值: dH={dH}")
            H_f -= dH
            if abs(dH) < newton_tol:
                converged = True
                break

        if not converged:
            raise RuntimeError(
                f"裂缝节点 (Rp={Rp}) Newton 迭代未在 {newton_max_iter} 步内收敛 (最后 |dH|={abs(dH):.2e} > tol={newton_tol:.2e})"
            )

        arg = max(H_f - H_ext, 0.0)
        Q_leak = kleak * np.sqrt(arg)
        Q_f = C_dt * (H_f - H_prev_f) + Q_leak
        H_w = H_w0 - Q_f * inv_2Aga
        V_left = Cp_f - ga * H_w
        V_right = -Cm_f + ga * H_w

    # 严格残差校验（验收阈值）
    r_mass = Q_f - Cf * (H_f - H_prev_f) / dt - kleak * np.sqrt(max(H_f - H_ext, 0.0))
    r_Rp = H_w - H_f - Rp * Q_f * abs(Q_f)
    if abs(r_mass) > 1.0e-9:
        raise RuntimeError(f"裂缝质量守恒残差超标: |r_mass| = {abs(r_mass):.4e} > 1e-9 m³/s")
    if abs(r_Rp) > 1.0e-8:
        raise RuntimeError(f"射孔压降残差超标: |r_Rp| = {abs(r_Rp):.4e} > 1e-8 m")

    return H_w, H_f, V_left, V_right, Q_f


# =====================================================================
# 严格稳态流场求解器
# =====================================================================
def compute_steady_state_profile(
    cfg: MocConfig,
    N: int,
    dx: float,
    dt: float,
    a: float,
    ga: float,
    area: float,
    D: float,
    nu: float,
    K_D: float,
    frac_indices: List[int],
    frac_kleak_arr: np.ndarray,
    frac_inflow_weights: Optional[List[float]],
    H_ext: float,
    frac_Rp_arr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算全井筒离散特征线精确稳态解（彻底消除 t=0 初始伪水击激扰）。

    物理机制
    --------
    对于封闭趾端 (toe_bc == 'dead_end')：
    1. 质量守恒要求井口注入量全部分配给各压裂簇：Qin = ∑ Q_leak,i；
    2. 最后一缝至封闭趾端区间为死水区：流速严格为 0；
    3. 各段流速恒定，水头沿井深按离散特征线关系精确递推：
       H[j] = H[j-1] - (1/ga)*J(V_right[j-1]) + dt*V_right[j-1]*sin(theta)
    4. 稳态孔眼压降：Hf,ss = Hw,ss - Rp * Qf,ss * |Qf,ss|
    5. 各缝等效滤失系数由稳态水头反求：k_equiv,i = Q_leak,i / √(H_f,i - H_ext)
    在 t < ts 停泵前，全井水头与流速的时间偏导数严格为零（达到机器精度）。

    返回
    ----
    H, V_left, V_right, w, kleak_equiv, Hw_ss, Hf_ss, Qf_ss
    """
    n_frac = len(frac_indices)
    if not np.isfinite(H_ext):
        raise ValueError(f"地层孔隙压力水头 H_ext 必须为有限数值: H_ext={H_ext}")
    H0 = cfg.initial_head
    V0 = cfg.initial_velocity
    Qin = V0 * area

    V_left = np.zeros(N + 1, dtype=np.float64)
    V_right = np.zeros(N + 1, dtype=np.float64)
    H = np.zeros(N + 1, dtype=np.float64)

    frac_Rp = (
        np.zeros(n_frac, dtype=np.float64)
        if frac_Rp_arr is None
        else np.asarray(frac_Rp_arr, dtype=np.float64)
    )

    if cfg.toe_bc == "dead_end":
        if n_frac == 0:
            V_left[:] = 0.0
            V_right[:] = 0.0
            H[:] = H0
            empty = np.array([], dtype=np.float64)
            return H, V_left, V_right, empty, empty, empty, empty, empty

        if cfg.initial_velocity > 0:
            if frac_inflow_weights is None:
                raise ValueError("封闭趾端 (dead_end) 且 initial_velocity > 0 时必须提供 fracture_inflow_weights")
            w = np.asarray(frac_inflow_weights, dtype=np.float64)
            if len(w) != n_frac:
                raise ValueError(f"fracture_inflow_weights 长度 ({len(w)}) 与裂缝数 ({n_frac}) 不匹配。")
            if not np.all(np.isfinite(w)):
                raise ValueError("fracture_inflow_weights 包含非有限数值 (NaN 或 Inf)")
            if not np.all(w >= 0.0):
                raise ValueError("fracture_inflow_weights 必须全部非负 (>= 0.0)")
            sum_w = float(np.sum(w))
            if abs(sum_w - 1.0) > 1e-10:
                raise ValueError(
                    f"fracture_inflow_weights 之和偏离 1.0 (sum={sum_w}, abs(sum-1)={abs(sum_w - 1.0):.4e} > 1e-10)，禁止静默归一化"
                )
            w = w / sum_w
        else:
            if frac_inflow_weights is not None:
                w = np.asarray(frac_inflow_weights, dtype=np.float64)
                if len(w) != n_frac:
                    raise ValueError(f"fracture_inflow_weights 长度 ({len(w)}) 与裂缝数 ({n_frac}) 不匹配。")
                if not np.all(np.isfinite(w)):
                    raise ValueError("fracture_inflow_weights 包含非有限数值 (NaN 或 Inf)")
                if not np.all(w >= 0.0):
                    raise ValueError("fracture_inflow_weights 必须全部非负 (>= 0.0)")
                sum_w = float(np.sum(w))
                if abs(sum_w - 1.0) > 1e-10:
                    raise ValueError(
                        f"fracture_inflow_weights 之和偏离 1.0 (sum={sum_w}, abs(sum-1)={abs(sum_w - 1.0):.4e} > 1e-10)，禁止静默归一化"
                    )
                w = w / sum_w
            else:
                w = np.zeros(n_frac, dtype=np.float64)

        Q_curr = Qin
        cur_idx = 0
        Q_leak_arr = w * Qin
        for k, ifr in enumerate(frac_indices):
            v_seg = Q_curr / area
            V_left[cur_idx:ifr] = v_seg
            V_right[cur_idx:ifr] = v_seg
            V_left[ifr] = v_seg
            Q_curr -= Q_leak_arr[k]
            v_next = Q_curr / area
            V_right[ifr] = v_next
            cur_idx = ifr + 1

        V_left[cur_idx:] = 0.0
        V_right[cur_idx:] = 0.0

        use_quasi = (cfg.friction_model == "quasi-steady")
        Re0 = reynolds(V0, D, nu)
        f_steady = darcy_friction_factor(Re0, K_D, model=cfg.friction_model)

        H[0] = H0
        for j in range(1, N + 1):
            v_seg = V_right[j - 1]
            if v_seg != 0.0:
                if use_quasi:
                    Re = abs(v_seg) * D / nu
                    f_d = darcy_friction_factor(Re, K_D, model="quasi-steady")
                else:
                    f_d = f_steady
                J_val = friction_term_J(f_d, D, v_seg, dt)
            else:
                J_val = 0.0
            H[j] = H[j - 1] - (1.0 / ga) * J_val + dt * v_seg * cfg.theta

        Hw_ss = np.zeros(n_frac, dtype=np.float64)
        Hf_ss = np.zeros(n_frac, dtype=np.float64)
        Qf_ss = np.zeros(n_frac, dtype=np.float64)
        kleak_equiv = np.zeros(n_frac, dtype=np.float64)

        for k, ifr in enumerate(frac_indices):
            Hw_ss[k] = H[ifr]
            Qf_ss[k] = Q_leak_arr[k]
            Hf_ss[k] = Hw_ss[k] - frac_Rp[k] * Qf_ss[k] * abs(Qf_ss[k])
            dH = Hf_ss[k] - H_ext
            if not np.isfinite(dH) or dH <= 1.0e-3:
                raise ValueError(
                    f"裂缝 #{k} (网格索引 {ifr}) 稳态内部水头不足: "
                    f"Hf_ss={Hf_ss[k]:.4f} m, H_ext={H_ext:.4f} m, dH={dH:.4e} m <= 1e-3 m。"
                    f"物理不可行：井底水头过低、射孔阻力过大或孔隙水头过高。"
                )
            kleak_equiv[k] = Qf_ss[k] / np.sqrt(dH)

        return H, V_left, V_right, w, kleak_equiv, Hw_ss, Hf_ss, Qf_ss

    elif cfg.toe_bc == "reservoir":
        x_grid = np.linspace(0.0, cfg.wellbore_length, N + 1)
        H[-1] = cfg.toe_head
        V_left[:] = V0
        V_right[:] = V0
        Re0_check = reynolds(V0, D, nu)
        f0_check = darcy_friction_factor(Re0_check, K_D, model=cfg.friction_model)
        friction_slope = f0_check * V0 * abs(V0) / (2.0 * G * D)
        H[:-1] = cfg.toe_head + friction_slope * (cfg.wellbore_length - x_grid[:-1])
        w = np.full(n_frac, 1.0 / n_frac, dtype=np.float64) if n_frac > 0 else np.array([], dtype=np.float64)
        kleak_equiv = frac_kleak_arr.copy()
        Hw_ss = np.array([H[ifr] for ifr in frac_indices], dtype=np.float64) if n_frac > 0 else np.array([], dtype=np.float64)
        Hf_ss = Hw_ss.copy()
        Qf_ss = np.zeros(n_frac, dtype=np.float64)
        return H, V_left, V_right, w, kleak_equiv, Hw_ss, Hf_ss, Qf_ss
    else:
        raise ValueError(f"未知 toe_bc: {cfg.toe_bc}")


# =====================================================================
# 主仿真器
# =====================================================================
def simulate_wellbore(
    cfg: MocConfig,
    fracture_positions: Optional[List[float]] = None,
    fracture_compliance_m2: Optional[List[float]] = None,
    fracture_kleak: Optional[List[float]] = None,
    fracture_inflow_weights: Optional[List[float]] = None,
    fracture_Rp: Optional[List[float]] = None,
    H_ext: float = 0.0,
    store_full_field: bool = True,
    snapshot_times: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[float, np.ndarray, np.ndarray, int, int], None]] = None,
    progress_interval_steps: Optional[int] = None,
    fracture_Cf: Optional[List[float]] = None,
) -> Dict:
    """
    运行 1D 井筒 MOC 仿真。

    参数
    ----
    cfg                    : MocConfig
    fracture_positions     : 裂缝位置列表 [m]（相对井口），None=无缝
    fracture_compliance_m2 : 各缝水头柔度 [m²] (C_H = ρ·g·C_p)
    fracture_kleak         : 各缝滤失系数 [m^{5/2}/s]
    fracture_inflow_weights: 各簇稳态进液权重 w_i (∑w_i = 1)，用于封闭趾端流量分配
    fracture_Rp            : 各缝射孔/近井阻力系数 [s²/m⁵]，None 时默认 0（无孔眼压降）
    H_ext                  : 地层孔隙压力水头 [m]
    store_full_field       : True=存储完整时空场；False=仅存 1D 时程
    snapshot_times         : store_full_field=False 时保存空间快照的时刻 [s]
    progress_callback      : 可选实时回调 (t, H, V, step, n_steps)
    progress_interval_steps: 实时回调步长
    fracture_Cf            : fracture_compliance_m2 的向后兼容别名
    """
    if fracture_compliance_m2 is None:
        fracture_compliance_m2 = fracture_Cf

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

    # ── 严格输入校验与裂缝设置 ─────────────────────────────────
    if not np.isfinite(H_ext):
        raise ValueError(f"地层孔隙压力水头 H_ext 必须为有限数值: H_ext={H_ext}")

    # 1. 严格检查裂缝参数的有限性、数值范围与权重归一化（无论有无裂缝）
    if fracture_compliance_m2 is not None:
        Cf_chk = np.asarray(fracture_compliance_m2, dtype=float)
        if not np.all(np.isfinite(Cf_chk)):
            raise ValueError("fracture_compliance_m2 包含非有限数值 (NaN 或 Inf)")
        if not np.all(Cf_chk >= 0.0):
            raise ValueError("fracture_compliance_m2 必须非负 (>= 0.0)")

    if fracture_kleak is not None:
        kleak_chk = np.asarray(fracture_kleak, dtype=float)
        if not np.all(np.isfinite(kleak_chk)):
            raise ValueError("fracture_kleak 包含非有限数值 (NaN 或 Inf)")
        if not np.all(kleak_chk >= 0.0):
            raise ValueError("fracture_kleak 必须非负 (>= 0.0)")

    if fracture_Rp is not None:
        Rp_chk = np.asarray(fracture_Rp, dtype=float)
        if not np.all(np.isfinite(Rp_chk)):
            raise ValueError("fracture_Rp 包含非有限数值 (NaN 或 Inf)")
        if not np.all(Rp_chk >= 0.0):
            raise ValueError("fracture_Rp 必须非负 (>= 0.0)")

    if fracture_inflow_weights is not None:
        w_chk = np.asarray(fracture_inflow_weights, dtype=float)
        if not np.all(np.isfinite(w_chk)):
            raise ValueError("fracture_inflow_weights 包含非有限数值 (NaN 或 Inf)")
        if not np.all(w_chk >= 0.0):
            raise ValueError("fracture_inflow_weights 必须全部非负 (>= 0.0)")
        sum_w = float(np.sum(w_chk))
        if abs(sum_w - 1.0) > 1e-10:
            raise ValueError(
                f"fracture_inflow_weights 之和偏离 1.0 (sum={sum_w}, abs(sum-1)={abs(sum_w - 1.0):.4e} > 1e-10)，禁止静默归一化"
            )

    has_fractures = fracture_positions is not None and len(fracture_positions) > 0
    if not has_fractures:
        if fracture_compliance_m2 is not None and len(fracture_compliance_m2) > 0:
            raise ValueError("未指定 fracture_positions 时禁止传入 fracture_compliance_m2")
        if fracture_kleak is not None and len(fracture_kleak) > 0:
            raise ValueError("未指定 fracture_positions 时禁止传入 fracture_kleak")
        if fracture_inflow_weights is not None and len(fracture_inflow_weights) > 0:
            raise ValueError("未指定 fracture_positions 时禁止传入 fracture_inflow_weights")
        if fracture_Rp is not None and len(fracture_Rp) > 0:
            raise ValueError("未指定 fracture_positions 时禁止传入 fracture_Rp")

    if cfg.toe_bc == "dead_end" and cfg.initial_velocity > 0:
        if not has_fractures:
            raise ValueError("封闭趾端 (dead_end) 且 initial_velocity > 0 时必须包含至少一条裂缝以泄流，否则物理不可行。")
        if fracture_inflow_weights is not None and fracture_kleak is not None:
            raise ValueError("封闭趾端 (dead_end) 且 initial_velocity > 0 时禁止同时显式传入 fracture_inflow_weights 与 fracture_kleak（生产模式必须且仅提供 fracture_inflow_weights）。")
        if fracture_inflow_weights is None and fracture_kleak is not None:
            raise ValueError("封闭趾端 (dead_end) 且 initial_velocity > 0 时禁止仅传入 fracture_kleak，必须提供 fracture_inflow_weights 且 fracture_kleak 必须为 None。")
        if fracture_inflow_weights is None and fracture_kleak is None:
            raise ValueError("封闭趾端 (dead_end) 且 initial_velocity > 0 时必须提供 fracture_inflow_weights（权重与 kleak 均未提供）。")

    if has_fractures:
        n_frac = len(fracture_positions)
        pos_arr = np.asarray(fracture_positions, dtype=float)
        if not np.all(np.isfinite(pos_arr)):
            raise ValueError("fracture_positions 包含非有限数值 (NaN 或 Inf)")
        for k, xf in enumerate(pos_arr):
            if not (0.0 < xf < cfg.wellbore_length):
                raise ValueError(
                    f"裂缝 #{k} 位置 x_f={xf} m 超出有效范围 (0, {cfg.wellbore_length}) m。严禁位于边界或越界。"
                )

        if fracture_compliance_m2 is None:
            Cf = np.zeros(n_frac, dtype=float)
        else:
            Cf = np.asarray(fracture_compliance_m2, dtype=float)
            if len(Cf) != n_frac:
                raise ValueError(f"fracture_compliance_m2 长度 ({len(Cf)}) 与裂缝数 ({n_frac}) 不匹配。")

        if fracture_Rp is None:
            Rp = np.zeros(n_frac, dtype=float)
        else:
            Rp = np.asarray(fracture_Rp, dtype=float)
            if len(Rp) != n_frac:
                raise ValueError(f"fracture_Rp 长度 ({len(Rp)}) 与裂缝数 ({n_frac}) 不匹配。")

        if fracture_kleak is not None:
            kleak = np.asarray(fracture_kleak, dtype=float)
            if len(kleak) != n_frac:
                raise ValueError(f"fracture_kleak 长度 ({len(kleak)}) 与裂缝数 ({n_frac}) 不匹配。")
        else:
            kleak = np.zeros(n_frac, dtype=float)

        if fracture_inflow_weights is not None:
            w = np.asarray(fracture_inflow_weights, dtype=float)
            if len(w) != n_frac:
                raise ValueError(f"fracture_inflow_weights 长度 ({len(w)}) 与裂缝数 ({n_frac}) 不匹配。")
            w = w / float(np.sum(w))
            fracture_inflow_weights = w.tolist()

        frac_Cf_arr = Cf
        frac_kleak_arr = kleak
        frac_Rp_arr = Rp

        # 缝位置对齐到最近网格点
        frac_indices = []
        for xf in pos_arr:
            idx = round(xf / dx)
            if idx <= 0 or idx >= N:
                raise ValueError(
                    f"裂缝对齐后网格索引越界 ({idx} ∉ [1, {N-1}]), xf={xf}, dx={dx}, L={cfg.wellbore_length}"
                )
            frac_indices.append(idx)

        # 检查缝不重合
        if len(set(frac_indices)) != len(frac_indices):
            raise ValueError(
                f"裂缝对齐后网格索引重合: {frac_indices}，请增大缝间距或减小 dt"
            )
    else:
        n_frac = 0
        frac_indices = []
        frac_Cf_arr = np.array([], dtype=np.float64)
        frac_kleak_arr = np.array([], dtype=np.float64)
        frac_Rp_arr = np.array([], dtype=np.float64)

    frac_index_set = set(frac_indices)

    # ── 初始条件（严格稳态流场计算，消除 t=0 伪水击波）──────────────
    x_grid = np.linspace(0.0, cfg.wellbore_length, N + 1)
    H, V_prev_left, V_prev_right, steady_w, kleak_equiv, Hw_ss, Hf_ss, Qf_ss = compute_steady_state_profile(
        cfg, N, dx, dt, a, ga, area, D, nu, K_D,
        frac_indices, frac_kleak_arr, fracture_inflow_weights, H_ext, frac_Rp_arr
    )

    if cfg.toe_bc == "dead_end" and has_fractures:
        # 在封闭趾端且未单独指定 kleak 时，采用守恒反算的等效 kleak
        frac_kleak_arr = kleak_equiv

    V = V_prev_left.copy()
    # Brunone 需要上上步速度（用于 ∂V/∂t）
    V_prev2_left = V_prev_left.copy()
    V_prev2_right = V_prev_right.copy()

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

    if progress_callback is not None:
        if progress_interval_steps is None:
            progress_interval_steps = max(1, n_steps // 100)
        if progress_interval_steps < 1:
            raise ValueError("progress_interval_steps 必须大于 0。")
        # 先发送 t=0 的全井筒初始场，使前端在第一步计算前即有可视反馈。
        progress_callback(0.0, H.copy(), V.copy(), 0, n_steps)

    # 裂缝记录与独立状态初始化
    fracture_wellbore_heads = np.zeros((n_steps + 1, n_frac), dtype=np.float64)
    fracture_internal_heads = np.zeros((n_steps + 1, n_frac), dtype=np.float64)
    fracture_Qs = np.zeros((n_steps + 1, n_frac), dtype=np.float64)

    if has_fractures:
        fracture_wellbore_heads[0, :] = Hw_ss
        fracture_internal_heads[0, :] = Hf_ss
        fracture_Qs[0, :] = Qf_ss
        Hf_prev = Hf_ss.copy()
    else:
        Hf_prev = np.array([], dtype=np.float64)

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
        if use_brunone and n >= 2:
            V_smooth = 0.05   # sign(V) 平滑阈值 [m/s]，tanh(V/V_smooth)

            # C⁺ 来源点 j=0..N-2（V_prev_right）：前向差分 dV/dx
            dVdt1 = (V_prev_right[:-2] - V_prev2_right[:-2]) / dt
            dVdx1 = (V_prev_right[1:-1] - V_prev_right[:-2]) / dx
            Re1b = np.abs(V1) * D / nu
            k1 = brunone_k_vec(Re1b) * k_scale
            sign_V1 = np.tanh(V1 / V_smooth)   # 平滑符号函数
            Ju1 = (k1 / 2.0) * dt * (dVdt1 + a * sign_V1 * np.abs(dVdx1))

            # C⁻ 来源点 j=2..N（V_prev_left）：后向差分 dV/dx
            dVdt2 = (V_prev_left[2:] - V_prev2_left[2:]) / dt
            dVdx2 = (V_prev_left[2:] - V_prev_left[1:-1]) / dx
            Re2b = np.abs(V2) * D / nu
            k2 = brunone_k_vec(Re2b) * k_scale
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

                # 裂缝内部水头历史状态 Hf_prev (严禁使用井筒水头 H_prev[i_f])
                H_old_f = Hf_prev[k]

                # 求解裂缝节点（返回井筒水头 H_w 与内部水头 H_f）
                H_w, H_f, V_left, V_right, Q_f = solve_fracture_node(
                    Cp_f, Cm_f, H_old_f, area, ga,
                    frac_Cf_arr[k], frac_kleak_arr[k], H_ext, dt,
                    Rp=frac_Rp_arr[k],
                )

                H_new[i_f] = H_w
                V_new[i_f] = V_left   # 主数组存上游侧（供记录与显示）
                fracture_wellbore_heads[n, k] = H_w
                fracture_internal_heads[n, k] = H_f
                fracture_Qs[n, k] = Q_f

        # ── 2. 井口边界节点 j=0（左边界，由下游 C⁻ 决定）──────
        # C⁻ from j=1: V2_0 = V_prev_left[1], H2_0 = H_prev[1]
        V2_0 = V_prev_left[1]
        H2_0 = H_prev[1]
        if use_quasi:
            Re2_0 = abs(V2_0) * D / nu
            f2_0 = darcy_friction_factor(Re2_0, K_D, "quasi-steady")
        else:
            f2_0 = f_steady
        J_0 = friction_term_J(f2_0, D, V2_0, dt)
        if use_brunone and n >= 2:
            dVdt_0 = (V_prev_left[1] - V_prev2_left[1]) / dt
            dVdx_0 = (V_prev_left[1] - V_prev_left[0]) / dx
            Re_0b = abs(V2_0) * D / nu
            k_0 = brunone_k(Re_0b) * k_scale
            J_0 += brunone_friction_Ju(k_0, dt, dVdt_0, dVdx_0, V2_0, a)
        Cm_0 = -V2_0 + ga * H2_0 + J_0 + ga * dt * V2_0 * theta

        if cfg.wellhead_bc == "velocity_step":
            if t < t_s:
                V_new[0] = V0
            else:
                V_new[0] = 0.0
            H_new[0] = (V_new[0] + Cm_0) / ga
        elif cfg.wellhead_bc == "ramp":
            if t < t_s:
                V_new[0] = V0
            elif t < t_s + t_c:
                tau = (t - t_s) / t_c
                V_new[0] = V0 * (1.0 - tau)
            else:
                V_new[0] = 0.0
            H_new[0] = (V_new[0] + Cm_0) / ga
        else:
            raise ValueError(f"未知 wellhead_bc: {cfg.wellhead_bc}")

        # ── 3. 趾端边界节点 j=N（右边界，由上游 C⁺ 决定）──────
        # C⁺ from j=N-1: V1_N = V_prev_right[N-1], H1_N = H_prev[N-1]
        V1_N = V_prev_right[N - 1]
        H1_N = H_prev[N - 1]
        if use_quasi:
            Re1_N = abs(V1_N) * D / nu
            f_N = darcy_friction_factor(Re1_N, K_D, "quasi-steady")
        else:
            f_N = f_steady
        J_N = friction_term_J(f_N, D, V1_N, dt)
        # Brunone（趾端 C⁺ 来源 node N-1：前向差分 dV/dx）
        if use_brunone and n >= 2:
            dVdt_N = (V_prev_right[N - 1] - V_prev2_right[N - 1]) / dt
            dVdx_N = (V_prev_right[N] - V_prev_right[N - 1]) / dx
            Re_Nb = abs(V1_N) * D / nu
            k_N = brunone_k(Re_Nb) * k_scale
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

        # 实时可视化仅按指定步长取样，不影响 MOC 的原始时间推进与数据记录。
        if progress_callback is not None and (
            n % progress_interval_steps == 0 or n == n_steps
        ):
            progress_callback(t, H_new.copy(), V_new.copy(), n, n_steps)

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
                # V_right = -Cm_f + ga·H_w，重新算
                Cm_f = Cm[i_f - 1]
                V_prev_right[i_f] = -Cm_f + ga * H_new[i_f]
            Hf_prev = fracture_internal_heads[n].copy()

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
        result["fracture_heads"] = fracture_internal_heads
        result["fracture_wellbore_heads"] = fracture_wellbore_heads
        result["fracture_internal_heads"] = fracture_internal_heads
        result["fracture_Qs"] = fracture_Qs
        result["steady_state"] = {
            "inflow_weights": steady_w,
            "equivalent_kleak": frac_kleak_arr,
            "Hw_ss": Hw_ss,
            "Hf_ss": Hf_ss,
            "Qf_ss": Qf_ss,
            "Qin_ss": V0 * area,
        }
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
    dict 包含:
        timestamps       : (n+1,) 时间轴 [s]
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
        friction_model=str(config.get("friction_model", friction_model)),
        dt=float(config.get("time_step", 1.0e-3)),
        tf=float(config.get("simulation_time", 3.0)),
        wellhead_bc=str(config.get("wellhead_bc", "velocity_step")),
        pump_shut_time=float(config.get("pump_shut_time", 1.0)),
        pump_closure_duration=float(config.get("pump_closure_duration", 1.0e-3)),
        initial_velocity=float(config.get("initial_velocity", 1.0)),
        theta=float(config.get("theta", 0.0)),
        initial_head=float(config.get("initial_head", 300.0)),
        toe_bc=str(config.get("toe_bc", "dead_end")),
        toe_head=float(config.get("toe_head", 300.0)),
    )

    fracture_positions = config.get("fracture_positions", [])
    fracture_Cf = config.get("fracture_Cf", None)
    fracture_kleak = config.get("fracture_kleak", None)
    fracture_inflow_weights = config.get("fracture_inflow_weights", config.get("inflow_weights", None))
    fracture_Rp = config.get("fracture_Rp", None)
    H_ext = float(config.get("H_ext", 0.0))

    # 大仿真时自动关闭完整场存储（>10万步 × >1000节点 → >1GB）
    n_steps_est = round(cfg.tf / cfg.dt)
    store_full = (n_steps_est * cfg.N) < 5_000_000   # 5M 点 ≈ 40MB

    # Step 3: 接缝（若 fracture_positions 非空）
    result = simulate_wellbore(
        cfg,
        fracture_positions=fracture_positions if fracture_positions else None,
        fracture_compliance_m2=fracture_Cf,
        fracture_kleak=fracture_kleak,
        fracture_inflow_weights=fracture_inflow_weights,
        fracture_Rp=fracture_Rp,
        H_ext=H_ext,
        store_full_field=store_full,
    )

    # 提取各缝节点水头时程（井筒交点水头与裂缝内部末端水头）
    junction_heads = []
    frac_end_heads = []
    if fracture_positions:
        frac_w_heads = result.get("fracture_wellbore_heads", np.zeros((len(result["timestamps"]), 0)))
        frac_i_heads = result.get("fracture_internal_heads", np.zeros((len(result["timestamps"]), 0)))
        for k in range(frac_w_heads.shape[1]):
            junction_heads.append(frac_w_heads[:, k])
        for k in range(frac_i_heads.shape[1]):
            frac_end_heads.append(frac_i_heads[:, k])

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
