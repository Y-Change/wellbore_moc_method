# -*- coding: utf-8 -*-
"""
moc_simulate.v2.core.moc_mesh

特征线法 (MOC) 时空菱形网格拓扑、Cr=1 稳定性自洽校验与李曼不变量更新
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np


@dataclass
class MocGrid:
    """MOC 时空网格与几何拓扑管理器"""
    L: float                    # 井深 [m]
    wavespeed: float            # 目标波速 [m/s]
    dt: float                   # 时间步长 [s]
    N: int                      # 空间网格段数
    dx: float                   # 空间网格步长 [m]
    a_adj: float                # Courant 校验自洽波速 a_adj = dx / dt [m/s]
    n_steps: int                # 总时间推进步数
    x_grid: np.ndarray          # 空间离散坐标向量 [m] (长度 N+1)

    @classmethod
    def create(cls, L: float, wavespeed: float, dt: float, tf: float) -> MocGrid:
        """根据 L, a, dt, tf 自洽构建 Cr=1 的 MOC 网格"""
        N_raw = round(L / (wavespeed * dt))
        if N_raw < 4:
            raise ValueError(f"网格分段数过小 N={N_raw} < 4，请增大井深或减小 dt")
        N = int(N_raw)
        dx = float(L / N)
        a_adj = float(dx / dt)
        n_steps = int(round(tf / dt))
        x_grid = np.linspace(0.0, L, N + 1)
        return cls(
            L=L,
            wavespeed=wavespeed,
            dt=dt,
            N=N,
            dx=dx,
            a_adj=a_adj,
            n_steps=n_steps,
            x_grid=x_grid,
        )

    def map_fracture_positions(
        self,
        raw_positions: Sequence[float],
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        将任意连续裂缝位置严格映射到 MOC 整数网格节点上，并进行严格递增排序与冲突碰撞检测。

        返回:
            frac_indices: 排序后的整数网格索引列表 [1, N-1]
            sorted_positions: 排序后的物理连续位置列表 [m]
            sort_order: 原始输入的排序置换索引用以对齐属性
        """
        if not raw_positions:
            return [], [], []

        for xf in raw_positions:
            if not np.isfinite(xf):
                raise ValueError(f"裂缝位置必须为有限数值，发现: {xf}")
            if xf <= 0.0 or xf >= self.L:
                raise ValueError(f"裂缝位置 x_f={xf:.2f}m 超出井筒有效内部区间 (0, {self.L:.2f}m)。")

        n_frac = len(raw_positions)
        float_pos = [float(xf) for xf in raw_positions]
        sort_order = list(np.argsort(float_pos))
        sorted_pos = [float_pos[i] for i in sort_order]

        frac_indices: List[int] = []
        for xf in sorted_pos:
            idx = int(round(xf / self.dx))
            idx = max(1, min(self.N - 1, idx))
            frac_indices.append(idx)

        # 检查网格节点重叠冲突
        for k in range(1, n_frac):
            if frac_indices[k] <= frac_indices[k - 1]:
                raise ValueError(
                    f"裂缝簇间距小于 MOC 网格步长 dx={self.dx:.3f}m: 裂缝 {sort_order[k-1]+1}({sorted_pos[k-1]:.2f}m) "
                    f"与裂缝 {sort_order[k]+1}({sorted_pos[k]:.2f}m) 冲突映射到同一节点 {frac_indices[k]}。"
                )

        return frac_indices, sorted_pos, sort_order


def compute_riemann_invariants(
    V1: np.ndarray,
    H1: np.ndarray,
    J1: np.ndarray,
    V2: np.ndarray,
    H2: np.ndarray,
    J2: np.ndarray,
    ga: float,
    dt: float,
    theta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算正特征线 C^+ (来自上游 i-1) 与负特征线 C^- (来自下游 i+1) 的因果李曼不变量：
        C^+ = V_1 + (g/a)*H_1 - J_1 + (g/a)*dt*V_1*sin(theta)
        C^- = -V_2 + (g/a)*H_2 + J_2 + (g/a)*dt*V_2*sin(theta)
    """
    gravity_comp1 = ga * dt * V1 * theta
    gravity_comp2 = ga * dt * V2 * theta

    Cp = V1 + ga * H1 - J1 + gravity_comp1
    Cm = -V2 + ga * H2 + J2 + gravity_comp2
    return Cp, Cm


def update_internal_nodes(Cp: np.ndarray, Cm: np.ndarray, ga: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解标准内部流动节点 (无侧向支管) 的联立特征方程：
        H = (C^+ + C^-) / (2 * g/a)
        V = (C^+ - C^-) / 2
    """
    H_internal = (Cp + Cm) / (2.0 * ga)
    V_internal = (Cp - Cm) / 2.0
    return H_internal, V_internal
