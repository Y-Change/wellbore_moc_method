# -*- coding: utf-8 -*-
"""
moc_simulate.v2.configs.wellbore

井筒几何、物性、摩阻与倾角结构化配置
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from moc_simulate.common.constants import (
    RHO_WATER,
    NU_WATER,
    CASING_ROUGHNESS,
    CASING_OD_5_5,
    WAVESPEED_DEFAULT,
)


@dataclass
class WellboreConfig:
    """井筒几何与流体物性配置"""
    wellbore_length: float = 5000.0          # 井深 L [m]
    wellbore_diameter: float = CASING_OD_5_5 # 井径 D [m] (默认 5.5 in 套管 0.1397m)
    fluid_density: float = RHO_WATER         # 压裂液密度 rho [kg/m^3]
    fluid_viscosity: float = NU_WATER        # 运动粘度 nu [m^2/s]
    wavespeed: float = WAVESPEED_DEFAULT     # 声学波速 a [m/s]
    roughness_height: float = CASING_ROUGHNESS # 绝对粗糙度 eps [m]
    friction_model: str = "brunone"          # 摩阻模型: 'steady' / 'quasi-steady' / 'brunone'
    brunone_k_scale: float = 1.0             # Brunone 非恒定摩阻缩放系数
    theta: float = 0.0                       # 井筒倾角正弦 (水平井为 0.0)

    @property
    def area(self) -> float:
        """井筒横截面积 [m^2]"""
        return float(np.pi * (self.wellbore_diameter ** 2) / 4.0)

    @property
    def K_D(self) -> float:
        """相对粗糙度 eps / D"""
        return float(self.roughness_height / self.wellbore_diameter)
