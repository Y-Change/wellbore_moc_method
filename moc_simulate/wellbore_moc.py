# -*- coding: utf-8 -*-
"""
moc_simulate.wellbore_moc (向后兼容透明门面 / Backward Compatibility Facade)

透明转发至 moc_simulate.v1.wellbore_moc，确保历史脚本与已有全部测试 100% 零修改兼容。
"""
from __future__ import annotations

from moc_simulate.v1.wellbore_moc import *
from moc_simulate.v1.wellbore_moc import (
    MocConfig,
    simulate_wellbore,
    simulate_case,
    solve_fracture_node,
    compute_steady_state_profile,
    darcy_friction_factor,
    reynolds,
    G,
)

# 兼容别名
solve_moc = simulate_wellbore

if __name__ == "__main__":
    # 保留原始 CLI smoke test
    cfg = MocConfig(
        wellbore_length=1000.0,
        wellbore_diameter=0.1397,
        wavespeed=1450.0,
        dt=1.0e-3,
        tf=3.0,
        pump_shut_time=1.0,
        initial_velocity=1.0,
        initial_head=300.0,
        toe_bc="reservoir",
        toe_head=300.0,
    )
    print(f"N={cfg.N}, dx={cfg.dx:.4f} m, a_adj={cfg.a_adj:.4f} m/s, dt={cfg.dt_adj}")
    print(f"n_steps={cfg.n_steps}, area={cfg.area:.6f} m^2")
    res = simulate_wellbore(cfg)
    wh = res["wellhead_head"]
    ts_idx = int(cfg.pump_shut_time / cfg.dt)
    H_pre = wh[ts_idx - 1]
    H_post = wh[ts_idx]
    dH_sim = H_post - H_pre
    dH_ana = cfg.a_adj * cfg.initial_velocity / G
    print(f"井口 H[ts-1]={H_pre:.3f}, H[ts]={H_post:.3f}")
    print(f"井口 Joukowsky 跳变 dH_sim = {dH_sim:.3f} m")
    print(f"Joukowsky 解析 dH_ana = a V0 / g = {dH_ana:.3f} m")
    print(f"误差 = {abs(dH_sim - dH_ana)/dH_ana*100:.4f} %")
