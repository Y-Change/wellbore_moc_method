# -*- coding: utf-8 -*-
"""
tests/test_steady_state_and_toe.py
验证封闭趾端 (dead_end) 严格稳态分流初始化与无扰动停泵物理特性。

验证目标：
1. 稳态流场在 t < ts 期间无任何伪水击激扰 (波动 < 1e-6 m)；
2. 封闭趾端质量绝对守恒：井口注入流量严格等于各簇稳态侧向泄流之和 ∑Q_leak = Qin；
3. 最后一缝至趾端死水区流速严格为零；
4. 射孔阻力接口 (Rp > 0) 正确产生孔眼压降 H_w - H_f = Rp * Qf * |Qf|。
"""

import numpy as np
import pytest

from moc_simulate.config import WELL_CONFIG
from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore, solve_fracture_node


def test_steady_state_zero_disturbance():
    """
    测试 1 缝、2 缝、4 缝在 t < ts 期间井口水头绝对平稳（消除 t=0 初始伪扰动）。
    """
    w = WELL_CONFIG
    for n_frac, positions, weights in [
        (1, [4100.0], [1.0]),
        (2, [4100.0, 4200.0], [0.6, 0.4]),
        (4, [4000.0, 4100.0, 4200.0, 4300.0], [0.4, 0.3, 0.2, 0.1]),
    ]:
        cfg = MocConfig(
            wellbore_length=w['L'],
            wellbore_diameter=w['wellbore_diameter'],
            fluid_density=w['fluid_density'],
            fluid_viscosity=w['fluid_viscosity'],
            wavespeed=w['wavespeed'],
            roughness_height=w['roughness_height'],
            friction_model='steady',
            wellhead_bc='velocity_step',
            pump_shut_time=5.0,   # 延迟 5s 停泵，充分检验稳态
            initial_velocity=1.0,
            initial_head=300.0,
            toe_bc='dead_end',
            dt=1e-3,
            tf=4.0,              # 仿真到 4s (< ts=5s)
        )
        res = simulate_wellbore(
            cfg,
            fracture_positions=positions,
            fracture_compliance_m2=[1e-5] * n_frac,
            fracture_inflow_weights=weights,
            H_ext=100.0,
            store_full_field=False,
        )
        H_wh = res['wellhead_head']
        max_fluc = float(np.max(H_wh) - np.min(H_wh))
        print(f"\n{n_frac} 缝稳态测试 (t < ts): 井口水头最大波动 = {max_fluc:.4e} m")
        assert max_fluc < 1.0e-8, f"{n_frac} 缝在停泵前存在非物理初始扰动: {max_fluc} m"


def test_dead_end_mass_balance_and_stagnant_zone():
    """
    检验稳态流场质量平衡与死水区零流速。
    """
    w = WELL_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wellbore_diameter=w['wellbore_diameter'],
        fluid_density=w['fluid_density'],
        fluid_viscosity=w['fluid_viscosity'],
        wavespeed=w['wavespeed'],
        roughness_height=w['roughness_height'],
        friction_model='steady',
        wellhead_bc='velocity_step',
        pump_shut_time=2.0,
        initial_velocity=1.2,
        initial_head=320.0,
        toe_bc='dead_end',
        dt=1e-3,
        tf=1.0,
    )
    positions = [3800.0, 4100.0, 4400.0]
    weights = [0.5, 0.3, 0.2]
    Qin = cfg.initial_velocity * cfg.area

    res = simulate_wellbore(
        cfg,
        fracture_positions=positions,
        fracture_compliance_m2=[1.5e-5, 1.0e-5, 0.8e-5],
        fracture_inflow_weights=weights,
        H_ext=120.0,
        store_full_field=True,
    )

    ss_info = res["steady_state"]
    kleak_equiv = ss_info["equivalent_kleak"]
    
    # 验证稳态下各缝泄流量之和等于注水流量
    head_at_fracs = [res["head"][0, idx] for idx in res["fracture_indices"]]
    Q_leak_sum = sum(
        k * np.sqrt(h - 120.0) for k, h in zip(kleak_equiv, head_at_fracs)
    )
    rel_mass_err = abs(Q_leak_sum - Qin) / Qin
    print(f"\n封闭趾端流量平衡: Qin = {Qin:.6f} m^3/s, ∑Q_leak = {Q_leak_sum:.6f} m^3/s, 相对误差 = {rel_mass_err:.4e}")
    assert rel_mass_err < 1.0e-10, f"稳态泄流量与注入量不守恒: {rel_mass_err}"

    # 验证最后一缝至趾端区间流速严格为 0
    last_idx = res["fracture_indices"][-1]
    dead_zone_velocities = res["velocity"][0, last_idx + 1:]
    max_dead_v = float(np.max(np.abs(dead_zone_velocities)))
    print(f"死水区最大流速: {max_dead_v:.4e} m/s")
    assert max_dead_v == 0.0, f"死水区流速必须严格为 0，当前为 {max_dead_v}"


def test_perforation_friction_feature_flag():
    """
    验证射孔阻力接口 (Rp > 0) 正确计算孔眼压降。
    """
    area = 0.015328
    ga = 9.81 / 1450.0
    dt = 1e-3
    Cf = 1e-5
    kleak = 1e-4
    H_ext = 100.0
    H_prev = 250.0
    Cp_f = 1.0 + ga * 260.0
    Cm_f = -0.5 + ga * 260.0

    # 1. Rp = 0 (基准无孔眼阻力)
    H_w0, H_f0, vl0, vr0, Qf0 = solve_fracture_node(
        Cp_f, Cm_f, H_prev, area, ga, Cf, kleak, H_ext, dt, Rp=0.0
    )
    assert abs(H_w0 - H_f0) < 1e-12, "Rp=0 时井筒水头与裂缝内部水头必须严格相等"

    # 2. Rp > 0 (开启射孔阻力)
    Rp = 100.0
    H_w_rp, H_f_rp, vl_rp, vr_rp, Qf_rp = solve_fracture_node(
        Cp_f, Cm_f, H_prev, area, ga, Cf, kleak, H_ext, dt, Rp=Rp
    )

    # 孔眼阻力限制了进入裂缝的流量：Qf_rp 应小于 Qf0
    print(f"\n射孔阻力测试: Qf(Rp=0) = {Qf0:.6f} m^3/s, Qf(Rp=100) = {Qf_rp:.6f} m^3/s")
    assert Qf_rp < Qf0, "孔眼阻力应阻碍裂缝进液"
    assert Qf_rp > 0.0, "正压差下裂缝流量应为正"
    assert H_w_rp > H_f_rp, "正向进液时井筒水头必须高于裂缝内部水头"
    assert abs(H_w_rp - H_f_rp - Rp * Qf_rp * abs(Qf_rp)) <= 1e-8, "必须严格满足孔眼压降关系"
