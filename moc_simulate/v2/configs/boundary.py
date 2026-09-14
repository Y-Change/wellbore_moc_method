# -*- coding: utf-8 -*-
"""
moc_simulate.v2.configs.boundary

水击动力学两端边界条件配置 (井口斜坡流速模型与封闭死端趾端约束)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundaryConfig:
    """边界条件动力学配置"""
    wellhead_bc: str = "ramp"                # 井口模型: 'ramp' (斜坡关泵) 或 'velocity_step' (瞬时阶跃)
    ramp_type: str = "linear"                # 斜坡曲线类型: 'linear' (线性) 或 'cosine' / 'smooth' (S型余弦平滑)
    pump_shut_time: float = 1.0              # 关泵起始时刻 t_s [s]
    pump_closure_duration: float = 1.0       # 关泵过渡历时 t_c [s] (<= 0 退化为阶跃关泵)
    initial_velocity: float = 1.0            # 初始稳态井口流速 V0 [m/s]
    initial_head: float = 300.0              # 初始稳态井口水头 H0 [m]
    toe_bc: str = "dead_end"                 # 趾端边界类型: 'dead_end' (封闭死端全反射) 或 'reservoir' (恒压水库)
    toe_head: float = 300.0                  # 水库边界水头 (仅 toe_bc='reservoir' 时有效) [m]
