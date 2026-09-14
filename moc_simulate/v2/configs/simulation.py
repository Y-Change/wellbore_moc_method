# -*- coding: utf-8 -*-
"""
moc_simulate.v2.configs.simulation

时空离散时间步长、求解时长与 Courant 条件自洽校验配置
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from moc_simulate.common.constants import G


@dataclass
class SimulationConfig:
    """时间与求解器离散配置"""
    dt: float = 1.0e-3                       # 基础时间步长 dt [s] (1.0 ms)
    tf: float = 100.0                        # 仿真截止时间 [s]
    g: float = G                             # 重力加速度 [m/s^2] (默认 9.81，保持与基准完全一致)
    store_full_field: bool = False           # 是否保存全时空 2D 网格矩阵 (大仿真关闭节省内存)
    snapshot_times: Optional[List[float]] = None # 保存特定空间剖面快照的时刻列表 [s]

    def compute_grid(self, wellbore_length: float, wavespeed: float) -> Tuple[int, float, float, float, int]:
        """
        基于 Courant 条件 Cr = a * dt / dx = 1.0 自动计算网格离散参数：
        返回: (N, dx, a_adj, dt_adj, n_steps)
        """
        N_raw = round(wellbore_length / (wavespeed * self.dt))
        if N_raw < 4:
            raise ValueError(f"井筒网格分段数过小 N={N_raw} < 4，请增大井深或减小 dt")
        N = int(N_raw)
        dx = float(wellbore_length / N)
        a_adj = float(dx / self.dt)
        dt_adj = float(self.dt)
        n_steps = int(round(self.tf / self.dt))
        return N, dx, a_adj, dt_adj, n_steps
