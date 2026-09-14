# -*- coding: utf-8 -*-
"""
moc_simulate.v2.configs.fracture

裂缝空间几何、顺应性储能、地质滤失与局部射孔耦合配置
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from moc_simulate.v2.configs.perforation import PerforationConfig


@dataclass
class FractureConfig:
    """单簇裂缝力学与几何配置"""
    position: float                     # 裂缝井深位置 x_f [m]
    compliance: float = 0.01            # 地质尺度水头柔度 C_f [m^2] (典型 0.005 ~ 0.03 m^2)
    leakoff_coef: float = 1.0e-4        # 等效滤失系数 k_leak [m^{5/2}/s]
    inflow_weight: float = 1.0          # 稳态进液流量权重 (多簇分流时内部归一化)
    H_ext: float = 100.0                # 地层孔隙水头 [m]
    perforation: Optional[PerforationConfig] = None # 本簇专属射孔配置 (None 则使用全局射孔配置)

    # 快捷兼容别名
    @property
    def x_f(self) -> float:
        return self.position

    @property
    def Cf(self) -> float:
        return self.compliance

    @property
    def kleak(self) -> float:
        return self.leakoff_coef
