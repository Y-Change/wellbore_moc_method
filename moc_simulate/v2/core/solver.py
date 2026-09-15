# -*- coding: utf-8 -*-
"""
moc_simulate.v2.core.solver

MOC_V2 生产级水锤动力学主求解器 (WellboreMocV2Solver 与便捷函数 simulate_v2)
集成：
1. 现场斜坡关泵动力学 (消除瞬态截断吉布斯假波)
2. 地质尺度宏观流体顺应性储能 (Cf 宏观大反弹)
3. 限流射孔非线性节流压降 (牛顿-拉夫逊二阶收敛求解)
4. 自洽稳态动量-质量空间解析场 (零假激波)
5. 闭端趾端全反射与 Brunone 瞬态非恒定摩阻
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from moc_simulate.common.constants import G
from moc_simulate.v2.configs import MocV2Config, FractureConfig, PerforationConfig
from moc_simulate.v2.core.moc_mesh import MocGrid, compute_riemann_invariants
from moc_simulate.v2.core.friction import (
    reynolds,
    darcy_friction_factor,
    friction_term_J,
    brunone_k_vec,
    brunone_friction_Ju,
)
from moc_simulate.v2.core.initial_field import (
    compute_steady_state_field,
    solve_physical_steady_state,
)
from moc_simulate.v2.core.fracture_node import solve_fracture_node_v2
from moc_simulate.v2.core.boundary_condition import (
    compute_ramp_velocity,
    apply_wellhead_bc,
    apply_toe_bc,
)

STEADY_MODE_PHYSICAL = "physical_flow_control"
STEADY_MODE_LEGACY = "prescribed_flow_split_legacy"


def ensure_v2_config(cfg: Any) -> MocV2Config:
    """确保配置对象为 MocV2Config，如果是 V1 MocConfig 或类似对象则自动转换适配"""
    if isinstance(cfg, MocV2Config):
        return cfg
    kwargs = {}
    for attr in [
        "wellbore_length", "wellbore_diameter", "fluid_density", "fluid_viscosity",
        "wavespeed", "roughness_height", "friction_model", "brunone_k_scale",
        "theta", "dt", "tf", "wellhead_bc", "pump_shut_time", "pump_closure_duration",
        "initial_velocity", "initial_head", "toe_bc", "toe_head",
    ]:
        if hasattr(cfg, attr):
            kwargs[attr] = getattr(cfg, attr)
    if hasattr(cfg, "g"):
        kwargs["g"] = getattr(cfg, "g")
    if hasattr(cfg, "store_full_field"):
        kwargs["store_full_field"] = getattr(cfg, "store_full_field")
    if hasattr(cfg, "snapshot_times"):
        kwargs["snapshot_times"] = getattr(cfg, "snapshot_times")
    if hasattr(cfg, "ramp_type"):
        kwargs["ramp_type"] = getattr(cfg, "ramp_type")
    if hasattr(cfg, "perf_num_holes"):
        kwargs["perf_num_holes"] = getattr(cfg, "perf_num_holes")
    if hasattr(cfg, "perf_diameter"):
        kwargs["perf_diameter"] = getattr(cfg, "perf_diameter")
    if hasattr(cfg, "perf_cd"):
        kwargs["perf_cd"] = getattr(cfg, "perf_cd")
    if hasattr(cfg, "perf_Kp_override"):
        kwargs["perf_Kp_override"] = getattr(cfg, "perf_Kp_override")
    return MocV2Config(**kwargs)


def resolve_steady_mode(
    requested: Optional[str],
    user_specified_weights: bool,
    user_specified_kleak: bool,
) -> str:
    """
    解析稳态求解模式。

    迁移策略:
    - 未指定 mode 且只给了人为权重 → 自动走历史分流模式 (兼容旧 V2 调用)
    - 未指定 mode 且只给了 kleak / 两者都未给 → 正向物理模式
    - 未指定 mode 却同时给了 kleak 与权重 → 拒绝，要求显式选择
    - 显式 physical 时仍禁止混用权重
    """
    if requested is None:
        if user_specified_weights and user_specified_kleak:
            raise ValueError(
                "同时传入 fracture_inflow_weights 与 fracture_kleak 时必须显式指定 "
                "steady_mode='physical_flow_control'（忽略人为权重，仅用滤失正向求解）"
                " 或 steady_mode='prescribed_flow_split_legacy'（按人为分流校准滤失）。"
                " 禁止依赖默认值静默选择，以免历史脚本与新物理模式混用。"
            )
        if user_specified_weights:
            return STEADY_MODE_LEGACY
        return STEADY_MODE_PHYSICAL

    mode = str(requested).strip()
    if mode not in (STEADY_MODE_PHYSICAL, STEADY_MODE_LEGACY):
        raise ValueError(
            f"未知的稳态求解模式: '{requested}'。支持 '{STEADY_MODE_PHYSICAL}' (默认正向物理模式) "
            f"与 '{STEADY_MODE_LEGACY}' (历史强制分流模式)。"
        )
    return mode


class WellboreMocV2Solver:
    """MOC_V2 面向对象瞬变流求解器"""

    def __init__(
        self,
        cfg: Union[MocV2Config, Any],
        fractures: Optional[Sequence[Union[FractureConfig, dict]]] = None,
        fracture_positions: Optional[Sequence[float]] = None,
        fracture_Cf: Optional[Sequence[float]] = None,
        fracture_kleak: Optional[Sequence[float]] = None,
        fracture_inflow_weights: Optional[Sequence[float]] = None,
        fracture_Kp: Optional[Sequence[float]] = None,
        fracture_num_holes: Optional[Sequence[int]] = None,
        fracture_perf_diameter: Optional[Sequence[float]] = None,
        fracture_perf_cd: Optional[Sequence[float]] = None,
        H_ext: float = 100.0,
        fracture_compliance_m2: Optional[Sequence[float]] = None,
        store_full_field: Optional[bool] = None,
        snapshot_times: Optional[Sequence[float]] = None,
        fracture_Rp: Optional[Sequence[float]] = None,
        steady_mode: Optional[str] = None,
        **kwargs,
    ):
        self.cfg = ensure_v2_config(cfg)
        if store_full_field is not None:
            self.cfg.store_full_field = bool(store_full_field)
        if snapshot_times is not None:
            self.cfg.snapshot_times = list(snapshot_times)

        self.g = self.cfg.g
        self.H_ext = float(H_ext)

        # 兼容历史别名 fracture_compliance_m2
        if fracture_compliance_m2 is not None and fracture_Cf is None:
            fracture_Cf = fracture_compliance_m2

        # 1. 建立时空因果网格
        self.grid = MocGrid.create(
            L=self.cfg.wellbore_length,
            wavespeed=self.cfg.wavespeed,
            dt=self.cfg.dt,
            tf=self.cfg.tf,
        )
        self.N = self.grid.N
        self.dx = self.grid.dx
        self.a = self.grid.a_adj
        self.dt = self.grid.dt
        self.ga = self.g / self.a
        self.D = cfg.wellbore_diameter
        self.area = cfg.area
        self.nu = cfg.fluid_viscosity
        self.K_D = cfg.roughness_height / self.D
        self.theta = cfg.theta
        self.k_scale = cfg.brunone_k_scale
        self.use_brunone = (cfg.friction_model == "brunone")
        self.use_quasi = (cfg.friction_model == "quasi-steady")

        # 2. 解析裂缝输入 (支持结构化 FractureConfig 序列或散装列表)
        self._setup_fractures(
            fractures=fractures,
            fracture_positions=fracture_positions,
            fracture_Cf=fracture_Cf,
            fracture_kleak=fracture_kleak,
            fracture_inflow_weights=fracture_inflow_weights,
            fracture_Kp=fracture_Kp,
            fracture_num_holes=fracture_num_holes,
            fracture_perf_diameter=fracture_perf_diameter,
            fracture_perf_cd=fracture_perf_cd,
        )

        self.steady_mode = resolve_steady_mode(
            requested=steady_mode,
            user_specified_weights=self.user_specified_weights,
            user_specified_kleak=self.user_specified_kleak,
        )

        if self.steady_mode == STEADY_MODE_PHYSICAL:
            if self.user_specified_weights and self.user_specified_kleak:
                raise ValueError(
                    "不可同时指定人为分流权重 (fracture_inflow_weights) 与物理滤失系数 (fracture_kleak)。"
                    "人为指定流量分配与独立采样地层滤失属于互斥的稳态控制边界，严防非物理矛盾混用。"
                )

            if self.user_specified_weights:
                raise ValueError(
                    "在正向物理生产模式 'physical_flow_control' 下，各簇流量分流由地质工程物性正向确定，"
                    "禁止显式传入人为分流权重 'fracture_inflow_weights'。如需复现历史人为分流工况，"
                    "请显式指定 steady_mode='prescribed_flow_split_legacy'。"
                )

        # 3. 计算稳态自洽场
        if self.steady_mode == STEADY_MODE_PHYSICAL:
            self.frac_kleak_arr = self.raw_kleak_arr.copy()
            (
                self.H0_realized,
                self.H_init,
                self.V_init,
                self.H_frac_ss,
                self.dH_perf_ss,
                self.q_frac_ss,
                self.frac_alpha_ss,
                self.steady_mass_residual,
            ) = solve_physical_steady_state(
                L=self.cfg.wellbore_length,
                N=self.N,
                dx=self.dx,
                D=self.D,
                area=self.area,
                nu=self.nu,
                K_D=self.K_D,
                V0=self.cfg.initial_velocity,
                g=self.g,
                toe_bc=self.cfg.toe_bc,
                frac_indices=self.frac_indices,
                frac_kleak_arr=self.frac_kleak_arr,
                frac_Kp_arr=self.frac_Kp_arr,
                sorted_pos=self.sorted_pos,
                H_ext=self.H_ext,
                H0_guess=self.cfg.initial_head,
            )
            self.cfg.initial_head = self.H0_realized
        else:
            (
                self.H_init,
                self.V_init,
                self.H_frac_ss,
                self.dH_perf_ss,
                self.q_frac_ss,
                calibrated_kleak_arr,
            ) = compute_steady_state_field(
                L=self.cfg.wellbore_length,
                N=self.N,
                dx=self.dx,
                D=self.D,
                area=self.area,
                nu=self.nu,
                K_D=self.K_D,
                V0=self.cfg.initial_velocity,
                H0=self.cfg.initial_head,
                g=self.g,
                toe_bc=self.cfg.toe_bc,
                frac_indices=self.frac_indices,
                w_arr=self.w_arr,
                frac_Kp_arr=self.frac_Kp_arr,
                sorted_pos=self.sorted_pos,
                H_ext=self.H_ext,
                raw_kleak_arr=self.raw_kleak_arr,
            )
            self.H0_realized = float(self.cfg.initial_head)
            self.frac_kleak_arr = calibrated_kleak_arr
            Q_in = float(self.cfg.initial_velocity * self.area)
            self.frac_alpha_ss = self.q_frac_ss / Q_in if Q_in > 0 else self.w_arr.copy()
            self.steady_mass_residual = (
                float(abs(Q_in - np.sum(self.q_frac_ss)) / Q_in) if Q_in > 0 else 0.0
            )

    def _setup_fractures(
        self,
        fractures: Optional[Sequence[Union[FractureConfig, dict]]],
        fracture_positions: Optional[Sequence[float]],
        fracture_Cf: Optional[Sequence[float]],
        fracture_kleak: Optional[Sequence[float]],
        fracture_inflow_weights: Optional[Sequence[float]],
        fracture_Kp: Optional[Sequence[float]],
        fracture_num_holes: Optional[Sequence[int]],
        fracture_perf_diameter: Optional[Sequence[float]],
        fracture_perf_cd: Optional[Sequence[float]],
    ):
        raw_pos: List[float] = []
        raw_Cf: List[float] = []
        raw_kleak: List[float] = []
        raw_w: List[float] = []
        raw_Kp: Optional[List[float]] = None
        raw_nh: Optional[List[int]] = None
        raw_dp: Optional[List[float]] = None
        raw_cd: Optional[List[float]] = None

        self.user_specified_kleak = False
        self.user_specified_weights = False
        if fracture_inflow_weights is not None:
            self.user_specified_weights = True

        if fractures is not None and len(fractures) > 0:
            for f in fractures:
                if isinstance(f, FractureConfig):
                    raw_pos.append(f.position)
                    raw_Cf.append(f.compliance)
                    raw_kleak.append(f.leakoff_coef)
                    if f.leakoff_coef != 1.0e-4:
                        self.user_specified_kleak = True
                    raw_w.append(f.inflow_weight)
                    if f.inflow_weight != 1.0:
                        self.user_specified_weights = True
                    if f.perforation is not None:
                        kp = f.perforation.compute_Kp(self.g)
                    else:
                        kp = self.cfg.perf_Kp
                    if raw_Kp is None:
                        raw_Kp = []
                    raw_Kp.append(kp)
                elif isinstance(f, dict):
                    raw_pos.append(float(f.get("position", f.get("x_f", 0.0))))
                    raw_Cf.append(float(f.get("compliance", f.get("Cf", 0.01))))
                    if "leakoff_coef" in f or "kleak" in f:
                        self.user_specified_kleak = True
                    raw_kleak.append(float(f.get("leakoff_coef", f.get("kleak", 1.0e-4))))
                    if "inflow_weight" in f or "weight" in f:
                        self.user_specified_weights = True
                    raw_w.append(float(f.get("inflow_weight", f.get("weight", 1.0))))
                    if raw_Kp is None:
                        raw_Kp = []
                    if "Kp" in f and f["Kp"] is not None:
                        raw_Kp.append(float(f["Kp"]))
                    elif "perforation" in f and f["perforation"] is not None:
                        perf_obj = f["perforation"]
                        if isinstance(perf_obj, PerforationConfig):
                            raw_Kp.append(perf_obj.compute_Kp(self.g))
                        elif isinstance(perf_obj, dict):
                            nh = int(perf_obj.get("num_holes", 6))
                            dp = float(perf_obj.get("diameter", 0.01))
                            cd = float(perf_obj.get("cd", 0.65))
                            ap = nh * np.pi * (dp ** 2) / 4.0
                            kp = 1.0 / (2.0 * self.g * (cd ** 2) * (ap ** 2)) if (ap > 0 and cd > 0) else 0.0
                            raw_Kp.append(float(perf_obj.get("Kp_override", kp)))
                        else:
                            raw_Kp.append(self.cfg.perf_Kp)
                    elif any(k in f for k in ("perf_num_holes", "num_holes", "perf_diameter", "diameter")):
                        nh = int(f.get("perf_num_holes", f.get("num_holes", self.cfg.perf_num_holes)))
                        dp = float(f.get("perf_diameter", f.get("diameter", self.cfg.perf_diameter)))
                        cd = float(f.get("perf_cd", f.get("cd", self.cfg.perf_cd)))
                        ap = nh * np.pi * (dp ** 2) / 4.0
                        kp = 1.0 / (2.0 * self.g * (cd ** 2) * (ap ** 2)) if (ap > 0 and cd > 0) else 0.0
                        raw_Kp.append(float(f.get("perf_Kp", kp)))
                    else:
                        raw_Kp.append(self.cfg.perf_Kp)
        elif fracture_positions is not None and len(fracture_positions) > 0:
            raw_pos = [float(x) for x in fracture_positions]
            n_f = len(raw_pos)
            for name, arr in [
                ("fracture_Cf", fracture_Cf),
                ("fracture_kleak", fracture_kleak),
                ("fracture_inflow_weights", fracture_inflow_weights),
                ("fracture_Kp", fracture_Kp),
                ("fracture_num_holes", fracture_num_holes),
                ("fracture_perf_diameter", fracture_perf_diameter),
                ("fracture_perf_cd", fracture_perf_cd),
            ]:
                if arr is not None and len(arr) != n_f:
                    raise ValueError(f"{name} 长度 ({len(arr)}) 与裂缝数 ({n_f}) 不匹配。")

            if fracture_kleak is not None:
                self.user_specified_kleak = True

            raw_Cf = list(fracture_Cf) if fracture_Cf is not None else [0.01] * n_f
            raw_kleak = list(fracture_kleak) if fracture_kleak is not None else [1.0e-4] * n_f
            raw_w = list(fracture_inflow_weights) if fracture_inflow_weights is not None else [1.0] * n_f
            raw_Kp = list(fracture_Kp) if fracture_Kp is not None else None
            raw_nh = list(fracture_num_holes) if fracture_num_holes is not None else None
            raw_dp = list(fracture_perf_diameter) if fracture_perf_diameter is not None else None
            raw_cd = list(fracture_perf_cd) if fracture_perf_cd is not None else None

        if len(raw_pos) > 0:
            self.has_fractures = True
            self.n_frac = len(raw_pos)
            # 网格节点映射与单调排序
            self.frac_indices, self.sorted_pos, sort_order = self.grid.map_fracture_positions(raw_pos)

            self.frac_Cf_arr = np.array([raw_Cf[i] for i in sort_order], dtype=np.float64)
            self.raw_kleak_arr = np.array([raw_kleak[i] for i in sort_order], dtype=np.float64)

            w_raw = np.array([raw_w[i] for i in sort_order], dtype=np.float64)
            sum_w = np.sum(w_raw)
            self.w_arr = w_raw / sum_w if sum_w > 0 else np.full(self.n_frac, 1.0 / self.n_frac)

            if raw_Kp is not None:
                self.frac_Kp_arr = np.array([raw_Kp[i] for i in sort_order], dtype=np.float64)
            else:
                self.frac_Kp_arr = np.zeros(self.n_frac, dtype=np.float64)
                for k, orig_idx in enumerate(sort_order):
                    nh = raw_nh[orig_idx] if raw_nh is not None else self.cfg.perf_num_holes
                    dp = raw_dp[orig_idx] if raw_dp is not None else self.cfg.perf_diameter
                    cd = raw_cd[orig_idx] if raw_cd is not None else self.cfg.perf_cd
                    ap = nh * np.pi * (dp ** 2) / 4.0
                    if ap > 0.0 and cd > 0.0:
                        self.frac_Kp_arr[k] = 1.0 / (2.0 * self.g * (cd ** 2) * (ap ** 2))
                    else:
                        self.frac_Kp_arr[k] = 0.0
        else:
            self.has_fractures = False
            self.n_frac = 0
            self.frac_indices = []
            self.sorted_pos = []
            self.frac_Cf_arr = np.array([], dtype=np.float64)
            self.raw_kleak_arr = np.array([], dtype=np.float64)
            self.w_arr = np.array([], dtype=np.float64)
            self.frac_Kp_arr = np.array([], dtype=np.float64)

    def solve(self, progress_callback: Optional[Callable[[float, int, int], None]] = None) -> Dict[str, Any]:
        """执行完整时间推进求解"""
        cfg = self.cfg
        N = self.N
        dt = self.dt
        dx = self.dx
        a = self.a
        D = self.D
        nu = self.nu
        K_D = self.K_D
        ga = self.ga
        theta = self.theta
        V0 = cfg.initial_velocity
        t_s = cfg.pump_shut_time
        t_c = cfg.pump_closure_duration
        n_steps = self.grid.n_steps
        has_fractures = self.has_fractures
        n_frac = self.n_frac
        frac_indices = self.frac_indices
        k_scale = self.k_scale
        use_brunone = self.use_brunone
        use_quasi = self.use_quasi

        Re_base = reynolds(V0, D, nu)
        f_steady = darcy_friction_factor(Re_base, K_D, "steady")

        H = self.H_init.copy()
        V = self.V_init.copy()

        V_prev_left = V.copy()
        V_prev_right = V.copy()
        if has_fractures:
            for k, idx in enumerate(frac_indices):
                V_prev_left[idx] = self.V_init[idx - 1]
                V_prev_right[idx] = self.V_init[idx]

        V_prev2_left = V_prev_left.copy()
        V_prev2_right = V_prev_right.copy()

        # 结果容器
        timestamps = np.zeros(n_steps + 1, dtype=np.float64)
        wh_head_hist = np.zeros(n_steps + 1, dtype=np.float64)
        wh_vel_hist = np.zeros(n_steps + 1, dtype=np.float64)
        toe_head_hist = np.zeros(n_steps + 1, dtype=np.float64)
        toe_vel_hist = np.zeros(n_steps + 1, dtype=np.float64)

        wh_head_hist[0] = H[0]
        wh_vel_hist[0] = V_prev_left[0]
        toe_head_hist[0] = H[-1]
        toe_vel_hist[0] = V_prev_right[-1]

        frac_head_hist = np.zeros((n_steps + 1, n_frac), dtype=np.float64) if has_fractures else np.zeros((0, 0))
        frac_well_head_hist = np.zeros((n_steps + 1, n_frac), dtype=np.float64) if has_fractures else np.zeros((0, 0))
        frac_perf_dH_hist = np.zeros((n_steps + 1, n_frac), dtype=np.float64) if has_fractures else np.zeros((0, 0))
        frac_Q_hist = np.zeros((n_steps + 1, n_frac), dtype=np.float64) if has_fractures else np.zeros((0, 0))

        if has_fractures:
            for k, idx in enumerate(frac_indices):
                frac_head_hist[0, k] = self.H_frac_ss[k]
                frac_well_head_hist[0, k] = H[idx]
                frac_perf_dH_hist[0, k] = self.dH_perf_ss[k]
                frac_Q_hist[0, k] = self.q_frac_ss[k]

        full_head_field = None
        full_vel_field = None
        if cfg.store_full_field:
            full_head_field = np.zeros((n_steps + 1, N + 1), dtype=np.float64)
            full_vel_field = np.zeros((n_steps + 1, N + 1), dtype=np.float64)
            full_head_field[0, :] = H
            full_vel_field[0, :] = V

        snapshots: Dict[float, Dict[str, np.ndarray]] = {}
        snapshot_steps = set()
        if cfg.snapshot_times:
            for st in cfg.snapshot_times:
                si = int(round(st / dt))
                if 0 <= si <= n_steps:
                    snapshot_steps.add(si)

        H_prev = H.copy()

        # 主时间循环
        for n in range(1, n_steps + 1):
            t = n * dt
            timestamps[n] = t

            V1 = V_prev_right[:-2]
            H1 = H_prev[:-2]
            V2 = V_prev_left[2:]
            H2 = H_prev[2:]

            if self.steady_mode != "prescribed_flow_split_legacy" or use_quasi:
                Re1 = reynolds(V1, D, nu)
                Re2 = reynolds(V2, D, nu)
                f_mode = "quasi-steady" if use_quasi else "steady"
                f1 = darcy_friction_factor(Re1, K_D, f_mode)
                f2 = darcy_friction_factor(Re2, K_D, f_mode)
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

                    H_well_k, H_frac_k, v_l, v_r, q_p = solve_fracture_node_v2(
                        Cp_f=Cp_f,
                        Cm_f=Cm_f,
                        H_prev_f=H_prev_f,
                        area=self.area,
                        ga=ga,
                        Cf=self.frac_Cf_arr[k],
                        kleak=self.frac_kleak_arr[k],
                        H_ext=self.H_ext,
                        dt=dt,
                        Kp=self.frac_Kp_arr[k],
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

            V_wh, H_wh = apply_wellhead_bc(
                t=t,
                V0=V0,
                ts=t_s,
                tc=t_c,
                wellhead_bc=cfg.wellhead_bc,
                ramp_type=cfg.ramp_type,
                Cm_0=Cm_0,
                ga=ga,
            )
            V_new[0] = V_wh
            H_new[0] = H_wh

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

            V_toe, H_toe = apply_toe_bc(
                toe_bc=cfg.toe_bc,
                toe_head=cfg.toe_head,
                Cp_N=Cp_N,
                ga=ga,
            )
            V_new[N] = V_toe
            H_new[N] = H_toe

            wh_head_hist[n] = H_new[0]
            wh_vel_hist[n] = V_new[0]
            toe_head_hist[n] = H_new[-1]
            toe_vel_hist[n] = V_new[-1]

            if full_head_field is not None:
                full_head_field[n, :] = H_new
                full_vel_field[n, :] = V_new

            if n in snapshot_steps:
                snapshots[round(t, 4)] = {
                    "H": H_new.copy(),
                    "V": V_new.copy(),
                    "t": t,
                }

            V_prev2_left = V_prev_left.copy()
            V_prev2_right = V_prev_right.copy()
            V_prev_left = V_new.copy()
            V_prev_right = V_new.copy()

            if has_fractures:
                for k, i_f in enumerate(frac_indices):
                    V_prev_left[i_f] = V_new[i_f]
                    V_prev_right[i_f] = v_r_arr[k]

            H_prev = H_new.copy()

            if progress_callback and n % max(1, n_steps // 100) == 0:
                progress_callback(t, n, n_steps)

        res: Dict[str, Any] = {
            "timestamps": timestamps,
            "wellhead_head": wh_head_hist,
            "wellhead_velocity": wh_vel_hist,
            "toe_head": toe_head_hist,
            "toe_velocity": toe_vel_hist,
            "fracture_heads": frac_head_hist,
            "fracture_internal_heads": frac_head_hist,
            "fracture_well_heads": frac_well_head_hist,
            "fracture_wellbore_heads": frac_well_head_hist,
            "fracture_perf_dHs": frac_perf_dH_hist,
            "fracture_Qs": frac_Q_hist,
            "fracture_indices": frac_indices,
            "x_grid": self.grid.x_grid,
            "H_init": self.H_init,
            "V_init": self.V_init,
            "H_frac_ss": self.H_frac_ss,
            "dH_perf_ss": self.dH_perf_ss,
            "frac_Kp_arr": self.frac_Kp_arr,
            "cfg": cfg,
            "head": full_head_field,
            "velocity": full_vel_field,
            "head_field": full_head_field,
            "velocity_field": full_vel_field,
            "H0_realized": self.H0_realized,
            "q_frac_ss": self.q_frac_ss,
            "fracture_alpha_ss": self.frac_alpha_ss,
            "steady_mass_residual": self.steady_mass_residual,
            "fracture_inflow_weights": self.frac_alpha_ss,
            "fracture_weights": self.frac_alpha_ss,
            "steady_mode": self.steady_mode,
        }
        if snapshots:
            res["snapshots"] = snapshots

        return res


def simulate_v2(
    cfg: Union[MocV2Config, Any],
    fractures: Optional[Sequence[Union[FractureConfig, dict]]] = None,
    fracture_positions: Optional[Sequence[float]] = None,
    fracture_Cf: Optional[Sequence[float]] = None,
    fracture_kleak: Optional[Sequence[float]] = None,
    fracture_inflow_weights: Optional[Sequence[float]] = None,
    fracture_Kp: Optional[Sequence[float]] = None,
    fracture_num_holes: Optional[Sequence[int]] = None,
    fracture_perf_diameter: Optional[Sequence[float]] = None,
    fracture_perf_cd: Optional[Sequence[float]] = None,
    H_ext: float = 100.0,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
    fracture_compliance_m2: Optional[Sequence[float]] = None,
    store_full_field: Optional[bool] = None,
    snapshot_times: Optional[Sequence[float]] = None,
    fracture_Rp: Optional[Sequence[float]] = None,
    steady_mode: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    MOC_V2 顶级正向瞬变流仿真便捷函数
    完全向下兼容 Step 4 与 V1 的调用风格。
    """
    solver = WellboreMocV2Solver(
        cfg=cfg,
        fractures=fractures,
        fracture_positions=fracture_positions,
        fracture_Cf=fracture_Cf,
        fracture_kleak=fracture_kleak,
        fracture_inflow_weights=fracture_inflow_weights,
        fracture_Kp=fracture_Kp,
        fracture_num_holes=fracture_num_holes,
        fracture_perf_diameter=fracture_perf_diameter,
        fracture_perf_cd=fracture_perf_cd,
        H_ext=H_ext,
        fracture_compliance_m2=fracture_compliance_m2,
        store_full_field=store_full_field,
        snapshot_times=snapshot_times,
        fracture_Rp=fracture_Rp,
        steady_mode=steady_mode,
        **kwargs,
    )
    return solver.solve(progress_callback=progress_callback)


# 别名供全库调用
simulate_wellbore_v2 = simulate_v2
