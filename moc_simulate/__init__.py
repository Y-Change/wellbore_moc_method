# -*- coding: utf-8 -*-
"""
moc_simulate

自研井筒-裂缝系统水击瞬变流特征线法 (MOC) 仿真体系：
- 顶层默认绑定生产级 V2 物理完善版本 (从第一性原理解决稳态自洽、地质顺应性、射孔节流与关泵斜坡)；
- 原始 V1 基准版本保留在 moc_simulate.v1 中供历史对标；
- 根目录 wellbore_moc.py 等透明门面保持对旧代码与测试 100% 零修改兼容。
"""
from __future__ import annotations

# 基础共享常数与路径
from moc_simulate import common

# 版本命名空间
from moc_simulate import v1
from moc_simulate import v2

# 顶层默认导出生产级 V2 接口 (决策偏好选项 A)
from moc_simulate.v2 import (
    MocV2Config,
    WellboreMocV2Solver,
    simulate_v2,
    simulate_wellbore_v2,
    WellboreConfig,
    FractureConfig,
    PerforationConfig,
    BoundaryConfig,
    SimulationConfig,
)

# 顶层便捷别名 (指向 V2)
simulate_wellbore = simulate_v2
MocConfig = MocV2Config
solve_moc = simulate_v2

__all__ = [
    "common",
    "v1",
    "v2",
    "MocV2Config",
    "WellboreMocV2Solver",
    "simulate_v2",
    "simulate_wellbore_v2",
    "simulate_wellbore",
    "MocConfig",
    "solve_moc",
    "WellboreConfig",
    "FractureConfig",
    "PerforationConfig",
    "BoundaryConfig",
    "SimulationConfig",
]
