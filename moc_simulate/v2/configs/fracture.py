# -*- coding: utf-8 -*-
"""
moc_simulate.v2.configs.fracture

裂缝空间几何、顺应性储能、地质滤失与局部射孔耦合配置
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from moc_simulate.v2.configs.perforation import PerforationConfig


@dataclass
class FractureConfig:
    """单簇裂缝力学与几何配置"""
    position: float                     # 裂缝井深位置 x_f [m]
    compliance: float = 0.01            # 地质尺度水头柔度 C_f [m^2] (典型 0.005 ~ 0.03 m^2)
    leakoff_coef: Optional[float] = None  # 等效滤失系数 k_leak [m^{5/2}/s]；None 表示未指定，求解器用默认 1e-4
    inflow_weight: Optional[float] = None  # 仅 legacy 分流模式使用；None 表示未指定人为权重
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
        return 1.0e-4 if self.leakoff_coef is None else float(self.leakoff_coef)
