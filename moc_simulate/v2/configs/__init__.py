# -*- coding: utf-8 -*-
"""
moc_simulate.v2.configs

MOC_V2 模块化配置层统一入口
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np

from moc_simulate.common.constants import (
    G,
    RHO_WATER,
    NU_WATER,
    CASING_ROUGHNESS,
    CASING_OD_5_5,
    WAVESPEED_DEFAULT,
)
from moc_simulate.v2.configs.wellbore import WellboreConfig
from moc_simulate.v2.configs.fracture import FractureConfig
from moc_simulate.v2.configs.perforation import PerforationConfig
from moc_simulate.v2.configs.boundary import BoundaryConfig
from moc_simulate.v2.configs.simulation import SimulationConfig


@dataclass
class MocV2Config:
    """
    MOC_V2 统一全量配置容器：
    完全向下兼容 Step4MocConfig 的传参形式，同时支持模块化子配置组装与自动校验。
    """
    # 井筒几何与流体物性
    wellbore_length: float = 5000.0          # 井深 L [m]
    wellbore_diameter: float = CASING_OD_5_5 # 井径 D [m]
    fluid_density: float = RHO_WATER         # 压裂液密度 rho [kg/m^3]
    fluid_viscosity: float = NU_WATER        # 运动粘度 nu [m^2/s]
    wavespeed: float = WAVESPEED_DEFAULT     # 声学波速 a [m/s]
    roughness_height: float = CASING_ROUGHNESS # 套管粗糙度 eps [m]
    friction_model: str = "brunone"          # 摩阻模型: 'steady' / 'quasi-steady' / 'brunone'
    brunone_k_scale: float = 1.0             # Brunone 摩阻缩放系数
    theta: float = 0.0                       # 井筒倾角正弦

    # 时空离散与仿真设定
    dt: float = 1.0e-3                       # 时间步长 dt [s]
    tf: float = 100.0                        # 仿真截止时间 [s]
    g: float = G                             # 重力加速度 [m/s^2]
    store_full_field: bool = False           # 是否保存完整 2D 时空场矩阵
    snapshot_times: Optional[List[float]] = None # 保存空间快照的特定时刻 [s]

    # 井口与趾端边界条件
    wellhead_bc: str = "ramp"                # 'ramp' / 'velocity_step'
    ramp_type: str = "linear"                # 'linear' / 'cosine' / 'smooth'
    pump_shut_time: float = 1.0              # 关泵起始时刻 t_s [s]
    pump_closure_duration: float = 1.0       # 关泵历时 t_c [s]
    initial_velocity: float = 1.0            # 初始井口流速 V0 [m/s]
    initial_head: float = 300.0              # 初始井口水头 H0 [m]
    toe_bc: str = "dead_end"                 # 'dead_end' / 'reservoir'
    toe_head: float = 300.0                  # 趾端水库水头 [m]

    # 全局默认限流射孔配置 (基准 Np=6, dp=10mm, Cd=0.65)
    perf_num_holes: int = 6                  # 单簇射孔数 N_p
    perf_diameter: float = 0.01              # 孔径 d_p [m] (10 mm)
    perf_cd: float = 0.65                    # 孔流系数 C_d
    perf_Kp_override: Optional[float] = None # 显式指定的 Kp [s^2/m^5]

    # 派生计算量 (在 __post_init__ 中自动完成)
    area: float = field(init=False)
    N: int = field(init=False)
    dx: float = field(init=False)
    dt_adj: float = field(init=False)
    a_adj: float = field(init=False)
    n_steps: int = field(init=False)
    perf_area: float = field(init=False)     # 单簇射孔总面积 A_p [m^2]
    perf_Kp: float = field(init=False)       # 射孔阻抗系数 K_p [s^2/m^5]

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
            "g": self.g,
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
        if self.g <= 0.0:
            raise ValueError(f"g 必须大于 0: {self.g}")

        # 1. 井筒截面积
        self.area = float(np.pi * (self.wellbore_diameter ** 2) / 4.0)

        # 2. 网格分段与 Courant 自洽参数
        N_raw = round(self.wellbore_length / (self.wavespeed * self.dt))
        if N_raw < 4:
            raise ValueError(f"分段数过小 N={N_raw} < 4")
        self.N = int(N_raw)
        self.dx = float(self.wellbore_length / self.N)
        self.a_adj = float(self.dx / self.dt)
        self.dt_adj = float(self.dt)
        self.n_steps = int(round(self.tf / self.dt))

        # 3. 射孔几何与流阻系数
        single_hole_area = np.pi * (self.perf_diameter ** 2) / 4.0
        self.perf_area = float(self.perf_num_holes * single_hole_area)
        if self.perf_Kp_override is not None:
            self.perf_Kp = float(self.perf_Kp_override)
        elif self.perf_area > 0.0 and self.perf_cd > 0.0:
            self.perf_Kp = float(1.0 / (2.0 * self.g * (self.perf_cd ** 2) * (self.perf_area ** 2)))
        else:
            self.perf_Kp = 0.0

    @classmethod
    def from_subconfigs(
        cls,
        wellbore: WellboreConfig,
        boundary: BoundaryConfig,
        simulation: SimulationConfig,
        perforation: PerforationConfig,
    ) -> MocV2Config:
        """从解耦的子配置对象合成统一的 MocV2Config"""
        return cls(
            wellbore_length=wellbore.wellbore_length,
            wellbore_diameter=wellbore.wellbore_diameter,
            fluid_density=wellbore.fluid_density,
            fluid_viscosity=wellbore.fluid_viscosity,
            wavespeed=wellbore.wavespeed,
            roughness_height=wellbore.roughness_height,
            friction_model=wellbore.friction_model,
            brunone_k_scale=wellbore.brunone_k_scale,
            theta=wellbore.theta,
            dt=simulation.dt,
            tf=simulation.tf,
            g=simulation.g,
            store_full_field=simulation.store_full_field,
            snapshot_times=simulation.snapshot_times,
            wellhead_bc=boundary.wellhead_bc,
            ramp_type=boundary.ramp_type,
            pump_shut_time=boundary.pump_shut_time,
            pump_closure_duration=boundary.pump_closure_duration,
            initial_velocity=boundary.initial_velocity,
            initial_head=boundary.initial_head,
            toe_bc=boundary.toe_bc,
            toe_head=boundary.toe_head,
            perf_num_holes=perforation.num_holes,
            perf_diameter=perforation.diameter,
            perf_cd=perforation.cd,
            perf_Kp_override=perforation.Kp_override,
        )


__all__ = [
    "WellboreConfig",
    "FractureConfig",
    "PerforationConfig",
    "BoundaryConfig",
    "SimulationConfig",
    "MocV2Config",
]
