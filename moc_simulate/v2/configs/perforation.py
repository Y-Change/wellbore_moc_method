# -*- coding: utf-8 -*-
"""
moc_simulate.v2.configs.perforation

射孔限流节流压降与孔眼几何配置
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from moc_simulate.common.constants import G


@dataclass
class PerforationConfig:
    """射孔节流压降配置"""
    num_holes: int = 6                  # 单簇射孔孔数 N_p
    diameter: float = 0.01              # 孔径 d_p [m] (10 mm)
    cd: float = 0.65                    # 孔流流量系数 C_d
    Kp_override: Optional[float] = None # 显式指定的 Kp 流阻系数 [s^2/m^5] (非 None 时覆盖几何计算)

    @property
    def single_hole_area(self) -> float:
        """单孔截面积 [m^2]"""
        return float(np.pi * (self.diameter ** 2) / 4.0)

    @property
    def total_area(self) -> float:
        """单簇总射孔有效流通面积 A_p [m^2]"""
        return float(self.num_holes * self.single_hole_area)

    def compute_Kp(self, g: float = G) -> float:
        """
        计算射孔节流流阻系数 K_p [s^2/m^5]：
            Delta H_perf = sign(q_p) * K_p * q_p^2
            K_p = 1 / (2 * g * C_d^2 * A_p^2)
        """
        if self.Kp_override is not None:
            return float(self.Kp_override)
        ap = self.total_area
        if ap > 0.0 and self.cd > 0.0:
            return float(1.0 / (2.0 * g * (self.cd ** 2) * (ap ** 2)))
        return 0.0
