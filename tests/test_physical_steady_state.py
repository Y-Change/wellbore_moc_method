# -*- coding: utf-8 -*-
"""
tests/test_physical_steady_state.py

P0 级正向物理稳态求解体系硬性验收测试套件：
1. test_physical_steady_state_closed_form_and_monotonicity:
   单簇渗流-节流耦合闭式解析实根精度对标与外层残差函数严格单调性证明
2. test_physical_steady_state_mass_conservation:
   全工况机器精度质量守恒 (|sum(q_j) - Q0| / Q0 < 1e-10)
3. test_zero_disturbance_pre_shut_drift:
   停泵前零假波验证 (延迟停泵至 5s，t in [0, 4] 井口水头漂移量严格 < 1e-6 m)
4. test_realized_alpha_matches_fracture_flow:
   物理实现的真实分流比与各簇稳态流量逐元素自洽 (alpha_j = q_j / Q0, sum alpha = 1.0)
5. test_dead_end_toe_flow_is_zero:
   死趾端盲端死水区流速严格归零 (V=0) 与水头平直性
6. test_sampler_forward_causality:
   采样器正向物理因果、拒绝采样对比度保证与无反向人工干预
7. test_legacy_and_physical_modes_cannot_be_mixed:
   生产模式与历史归档模式互斥隔离与非法参数防御
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from moc_simulate.common.constants import G
from moc_simulate.v2.configs import MocV2Config, FractureConfig
from moc_simulate.v2.core.moc_mesh import MocGrid
from moc_simulate.v2.core.initial_field import (
    solve_physical_steady_state,
    compute_steady_state_field,
    compute_physical_steady_residual,
    InfeasibleSteadyStateError,
)
from moc_simulate.v2.core.solver import WellboreMocV2Solver, simulate_v2
from moc_simulate.v2.batch.sampler import (
    LatinHypercubeSampler,
    LhsSamplingBounds,
    sample_preset_scenario,
)


def test_physical_steady_state_closed_form_and_monotonicity():
    """
    测试 1: 验证单簇闭式解与残差函数严格单调递减性。
    """
    # 1. 验证单簇闭式解析解与牛顿/数值求根逐位一致 (< 1e-12)
    H_w = 350.0
    H_ext = 100.0
    kleak = 2.5e-4
    Kp = 5.0e5

    # 闭式解
    denom = 1.0 + (kleak ** 2) * Kp
    q_closed = kleak * np.sqrt((H_w - H_ext) / denom)
    dH_perf_closed = Kp * (q_closed ** 2)
    H_frac_closed = H_w - dH_perf_closed

    # 数值验证: q = kleak * sqrt(H_frac - H_ext)
    q_from_hfrac = kleak * np.sqrt(H_frac_closed - H_ext)
    assert np.isclose(q_closed, q_from_hfrac, atol=1e-12), "闭式解流量与根据裂缝腔内水头反算的滤失量不一致"

    # 数值求根验证: F(q) = H_w - Kp*q^2 - H_ext - (q/kleak)^2 = 0
    def obj_q(q_val):
        return (H_w - Kp * (q_val ** 2)) - H_ext - (q_val / kleak) ** 2

    q_numerical = brentq(obj_q, 0.0, 1.0)
    assert np.isclose(q_closed, q_numerical, atol=1e-12), f"闭式解与数值求根存在偏差: {abs(q_closed - q_numerical)}"

    # 2. 验证多簇残差函数 R(H0) = Q0 - sum(q_j(H0)) 的严格单调递减性 (dR/dH0 < 0)
    L = 5000.0
    N = 3448
    dx = L / N
    D = 0.1397
    area = np.pi * (D ** 2) / 4.0
    nu = 1.0e-6
    K_D = 0.045e-3 / D
    V0 = 1.2
    Q0 = V0 * area
    positions = [4000.0, 4030.0, 4060.0, 4090.0]
    frac_indices = [int(round(p / dx)) for p in positions]
    frac_kleak = np.array([1.0e-4, 2.0e-4, 3.0e-4, 1.5e-4], dtype=np.float64)
    frac_Kp = np.array([3.0e5, 5.0e5, 8.0e5, 4.0e5], dtype=np.float64)

    H_ext_arr = np.full(len(positions), H_ext, dtype=np.float64)
    # 选取高于地层水头+摩阻损失的有效生产区间 [200.0, 1000.0]
    H0_samples = np.linspace(200.0, 1000.0, 50)
    residuals = []
    for h0 in H0_samples:
        r_val = compute_physical_steady_residual(
            H0=h0,
            dx=dx,
            D=D,
            area=area,
            nu=nu,
            K_D=K_D,
            V0=V0,
            g=G,
            frac_indices=frac_indices,
            frac_kleak_arr=frac_kleak,
            frac_Kp_arr=frac_Kp,
            H_ext_arr=H_ext_arr,
        )
        residuals.append(r_val)

    residuals = np.array(residuals)
    # 单调递减检验: 任意后项必须严格小于前项 (差值 < 0)
    d_res = np.diff(residuals)
    assert np.all(d_res < 0.0), f"残差函数未满足严格单调递减: 最大正增量={np.max(d_res)}"
    assert residuals[0] > 0.0, f"下限残差应大于0: {residuals[0]}"
    assert residuals[-1] < 0.0, f"上限残差应小于0: {residuals[-1]}"


def test_physical_steady_state_mass_conservation():
    """
    测试 2: 验证各种簇数与非对称物性下的全局质量守恒 (|sum(q_j) - Q0| / Q0 < 1e-10)。
    """
    L = 5000.0
    a = 1450.0
    dt = 0.001
    N = int(round(L / (a * dt)))
    dx = L / N
    D = 0.1397
    area = np.pi * (D ** 2) / 4.0

    test_cases = [
        # (n_frac, positions, kleaks, Kps, V0)
        (1, [4100.0], [2.0e-4], [4.0e5], 1.0),
        (2, [4100.0, 4150.0], [1.0e-4, 3.0e-4], [3.0e5, 6.0e5], 1.2),
        (4, [4000.0, 4030.0, 4060.0, 4100.0], [0.8e-4, 1.5e-4, 2.5e-4, 0.5e-4], [2e5, 5e5, 8e5, 3e5], 1.5),
        (6, [3800.0, 3840.0, 3880.0, 3920.0, 3960.0, 4000.0], [1e-4, 2e-4, 8e-4, 0.4e-4, 1.2e-4, 2e-4], [5e5]*6, 0.9),
    ]

    for n_frac, pos, kleak, kp, v0 in test_cases:
        frac_indices = [int(round(p / dx)) for p in pos]
        (
            H0_realized,
            H_init,
            V_init,
            H_frac_ss,
            dH_perf_ss,
            q_frac_ss,
            alpha_realized,
            mass_res,
        ) = solve_physical_steady_state(
            L=L,
            N=N,
            dx=dx,
            D=D,
            area=area,
            nu=1.0e-6,
            K_D=0.045e-3 / D,
            V0=v0,
            g=G,
            toe_bc="dead_end",
            frac_indices=frac_indices,
            frac_kleak_arr=np.array(kleak, dtype=np.float64),
            frac_Kp_arr=np.array(kp, dtype=np.float64),
            sorted_pos=pos,
            H_ext=100.0,
        )

        total_q = v0 * area
        actual_sum_q = np.sum(q_frac_ss)
        rel_err = abs(actual_sum_q - total_q) / total_q

        assert rel_err < 1.0e-10, f"{n_frac} 簇质量不守恒: rel_err={rel_err:.2e}, sum_q={actual_sum_q}, Q0={total_q}"
        assert mass_res < 1.0e-10, f"{n_frac} 簇 mass_residual 诊断值超标: {mass_res:.2e}"
        assert H0_realized > 100.0, f"{n_frac} 簇反解水头低于地层水头: {H0_realized}"


def test_zero_disturbance_pre_shut_drift():
    """
    测试 3: 推迟关泵至 5.0s，验证 t in [0, 4]s 井口水头绝对平稳，漂移严格 < 1e-6 m。
    彻底证明消除了 t=0 虚假激波与非自洽松弛波，并验证在所有摩阻模型下的一致零漂移。
    """
    for fm in ["brunone", "quasi-steady", "steady"]:
        for n_frac, pos, kleak in [
            (1, [4100.0], [1.5e-4]),
            (2, [4100.0, 4130.0], [1.0e-4, 2.0e-4]),
            (4, [4100.0, 4120.0, 4140.0, 4160.0], [0.8e-4, 1.2e-4, 2.5e-4, 0.5e-4]),
        ]:
            cfg = MocV2Config(
                wellbore_length=5000.0,
                wavespeed=1450.0,
                friction_model=fm,
                dt=0.001,
                tf=4.0,              # 仿真到 4s (< ts=5s)
                pump_shut_time=5.0,   # 延迟 5s 关泵，纯稳态保持
                pump_closure_duration=1.0,
                initial_velocity=1.2,
                toe_bc="dead_end",
            )

            res = simulate_v2(
                cfg=cfg,
                fracture_positions=pos,
                fracture_Cf=[0.01] * n_frac,
                fracture_kleak=kleak,
                steady_mode="physical_flow_control",
                H_ext=100.0,
            )

            t = res["timestamps"]
            wh_head = res["wellhead_head"]
            mask_pre = t <= 4.0
            drift = float(np.max(np.abs(wh_head[mask_pre] - wh_head[0])))

            print(f"\n[{fm}] {n_frac} 簇零扰动稳态漂移: max_drift = {drift:.4e} m")
            assert drift < 1.0e-6, f"{fm} 模式下 {n_frac} 簇关泵前存在虚假前驱波: drift={drift:.2e} m >= 1e-6 m"


def test_realized_alpha_matches_fracture_flow():
    """
    测试 4: 验证 alpha_j^ss = q_j^ss / Q0 逐元素一致，且 sum(alpha) == 1.0。
    """
    cfg = MocV2Config(wellbore_length=5000.0, initial_velocity=1.0, dt=0.002, tf=2.0)
    pos = [4000.0, 4030.0, 4060.0, 4090.0]
    kleak = [1.0e-4, 2.0e-4, 3.0e-4, 1.5e-4]
    Q0 = cfg.initial_velocity * cfg.area

    solver = WellboreMocV2Solver(
        cfg=cfg,
        fracture_positions=pos,
        fracture_kleak=kleak,
        steady_mode="physical_flow_control",
    )

    alpha = solver.frac_alpha_ss
    q_ss = solver.q_frac_ss

    assert len(alpha) == 4
    assert len(q_ss) == 4
    np.testing.assert_allclose(alpha, q_ss / Q0, rtol=1e-12, atol=1e-14)
    assert np.isclose(np.sum(alpha), 1.0, rtol=1e-10)

    # 验证强滤失簇对应更大的真实分流比
    assert alpha[2] > alpha[0], "滤失系数更大者真实实现的稳态分流比应更大"


def test_dead_end_toe_flow_is_zero():
    """
    测试 5: 验证盲端死水区流速严格归零 (V=0) 且水头绝对平直。
    """
    cfg = MocV2Config(wellbore_length=5000.0, initial_velocity=1.0, dt=0.002, tf=2.0)
    pos = [3500.0, 3600.0]  # 最末簇在 3600m，距井底 1400m
    solver = WellboreMocV2Solver(
        cfg=cfg,
        fracture_positions=pos,
        fracture_kleak=[1e-4, 2e-4],
        steady_mode="physical_flow_control",
    )

    last_idx = solver.frac_indices[-1]
    # 1. 盲端死水区初始流速严格归零
    assert np.all(solver.V_init[last_idx:] == 0.0), "最末簇下游死水区初始流速不全为零"
    # 2. 盲端死水区初始水头绝对平直 (等同最末簇水头)
    assert np.all(solver.H_init[last_idx:] == solver.H_init[last_idx]), "最末簇下游死水区初始水头不平直"


def test_sampler_forward_causality():
    """
    测试 6: 验证采样器先采样物性后求解流量的正向因果链与拒绝采样机制。
    """
    sampler = LatinHypercubeSampler(seed=2026)
    samples = sampler.sample(n_samples=5)

    for s in samples:
        # 1. 人为权重 fracture_inflow_weights 必须被移除
        assert "fracture_inflow_weights" not in s, "采样器不应包含人为指定的 fracture_inflow_weights"

        # 2. 必须包含物理前向求解生成的诊断量
        assert "fracture_alpha_ss" in s
        assert "fracture_Q_ss" in s
        assert "H0_realized" in s
        assert "steady_mass_residual" in s

        # 3. 真实物理分流比必须归一化为 1.0
        alpha = s["fracture_alpha_ss"]
        assert np.isclose(sum(alpha), 1.0, atol=1e-8)

        # 4. 拒绝采样生效：多簇阶段不应退化为完全平直无对比度
        if s["n_frac"] >= 2:
            spread = max(alpha) - min(alpha)
            assert spread >= 0.02, f"拒绝采样未过滤平直无对比度工况: spread={spread}"


def test_legacy_and_physical_modes_cannot_be_mixed():
    """
    测试 7: 验证模式解耦、参数冲突拦截与向后兼容性。
    """
    cfg = MocV2Config(wellbore_length=5000.0, initial_velocity=1.0)
    pos = [4100.0, 4200.0]

    # 1. 生产模式下显式传入 fracture_inflow_weights 必须拦截
    with pytest.raises(ValueError, match="在正向物理生产模式 'physical_flow_control' 下"):
        WellboreMocV2Solver(
            cfg=cfg,
            fracture_positions=pos,
            fracture_inflow_weights=[0.6, 0.4],
            steady_mode="physical_flow_control",
        )

    # 2. 同时传入 fracture_inflow_weights 与 fracture_kleak 必须拦截
    with pytest.raises(ValueError, match="不可同时指定人为分流权重"):
        WellboreMocV2Solver(
            cfg=cfg,
            fracture_positions=pos,
            fracture_inflow_weights=[0.6, 0.4],
            fracture_kleak=[1e-4, 2e-4],
            steady_mode="physical_flow_control",
        )

    # 3. 未知模式拦截
    with pytest.raises(ValueError, match="未知的稳态求解模式"):
        WellboreMocV2Solver(
            cfg=cfg,
            fracture_positions=pos,
            steady_mode="quantum_flow_mode",
        )

    # 4. 显式指定 legacy 模式正常运行并使用校准滤失
    solver_legacy = WellboreMocV2Solver(
        cfg=cfg,
        fracture_positions=pos,
        fracture_inflow_weights=[0.7, 0.3],
        steady_mode="prescribed_flow_split_legacy",
    )
    assert solver_legacy.steady_mode == "prescribed_flow_split_legacy"
    assert np.isclose(solver_legacy.frac_alpha_ss[0], 0.7, atol=1e-8)
    assert solver_legacy.frac_kleak_arr is not None

    # 5. FractureConfig 序列在 legacy 模式下支持显式指定权重
    fractures_legacy = [
        FractureConfig(position=4100.0, inflow_weight=0.6),
        FractureConfig(position=4200.0, inflow_weight=0.4),
    ]
    solver_fc_legacy = WellboreMocV2Solver(
        cfg=cfg,
        fractures=fractures_legacy,
        steady_mode="prescribed_flow_split_legacy",
    )
    assert np.allclose(solver_fc_legacy.frac_alpha_ss, [0.6, 0.4], atol=1e-8)

    # 6. FractureConfig 序列在 physical 模式下若指定 inflow_weight != 1.0 必须拦截
    with pytest.raises(ValueError, match="在正向物理生产模式 'physical_flow_control' 下"):
        WellboreMocV2Solver(
            cfg=cfg,
            fractures=fractures_legacy,
            steady_mode="physical_flow_control",
        )

    # 7. FractureConfig 序列在 physical 模式下正常指定物理滤失系数必须通过
    fractures_phys = [
        FractureConfig(position=4100.0, leakoff_coef=2.0e-4),
        FractureConfig(position=4200.0, leakoff_coef=1.0e-4),
    ]
    solver_fc_phys = WellboreMocV2Solver(
        cfg=cfg,
        fractures=fractures_phys,
        steady_mode="physical_flow_control",
    )
    assert np.isclose(np.sum(solver_fc_phys.frac_alpha_ss), 1.0, atol=1e-10)
    assert solver_fc_phys.frac_alpha_ss[0] > solver_fc_phys.frac_alpha_ss[1]

    # 8. 未指定 mode 且只传入人为权重：自动回退历史分流模式
    solver_auto_legacy = WellboreMocV2Solver(
        cfg=cfg,
        fracture_positions=pos,
        fracture_inflow_weights=[0.7, 0.3],
    )
    assert solver_auto_legacy.steady_mode == "prescribed_flow_split_legacy"
    assert np.isclose(solver_auto_legacy.frac_alpha_ss[0], 0.7, atol=1e-8)

    # 9. 未指定 mode 却同时传入 kleak 与权重：必须显式选择，禁止静默混用
    with pytest.raises(ValueError, match="必须显式指定"):
        WellboreMocV2Solver(
            cfg=cfg,
            fracture_positions=pos,
            fracture_inflow_weights=[0.6, 0.4],
            fracture_kleak=[1e-4, 2e-4],
        )


def test_zero_leakoff_and_negative_flow_boundary_checks():
    """
    测试 8: 物理边界防御测试：全零滤失在死趾端下的质量不守恒拒绝与负流速反抽非法输入拦截。
    """
    L = 5000.0
    N = 3448
    dx = L / N
    D = 0.1397
    area = np.pi * (D ** 2) / 4.0

    # 1. 死趾端边界下全零滤失拒绝
    with pytest.raises(ValueError, match="所有裂缝滤失系数均为零"):
        solve_physical_steady_state(
            L=L,
            N=N,
            dx=dx,
            D=D,
            area=area,
            nu=1.0e-6,
            K_D=0.045e-3 / D,
            V0=1.0,
            g=G,
            toe_bc="dead_end",
            frac_indices=[2500, 2700],
            frac_kleak_arr=np.array([0.0, 0.0]),
            frac_Kp_arr=np.array([1.0e5, 1.0e5]),
            sorted_pos=[3600.0, 3900.0],
        )

    # 2. 负流速注入拦截
    with pytest.raises(ValueError, match="初始注入流速 V0 必须 >= 0"):
        solve_physical_steady_state(
            L=L,
            N=N,
            dx=dx,
            D=D,
            area=area,
            nu=1.0e-6,
            K_D=0.045e-3 / D,
            V0=-1.0,
            g=G,
            toe_bc="dead_end",
            frac_indices=[2500],
            frac_kleak_arr=np.array([1.0e-4]),
            frac_Kp_arr=np.array([1.0e5]),
            sorted_pos=[3600.0],
        )


def test_preset_scenario_direct_simulation_compatibility():
    """
    测试 9: 验证 sample_preset_scenario 产出字典不含人为 fracture_inflow_weights，
    且可直接解包传入 simulate_v2 无缝推进瞬态求解。
    """
    sc = sample_preset_scenario("perfect_uniform", tf=1.0, dt=0.002)
    assert "fracture_inflow_weights" not in sc, "预设工况字典不应包含人为 fracture_inflow_weights"
    assert "fracture_alpha_ss" in sc

    cfg = MocV2Config(
        wellbore_length=sc["wellbore_length"],
        wavespeed=sc["wavespeed"],
        initial_velocity=sc["initial_velocity"],
        pump_closure_duration=sc["pump_closure_duration"],
        ramp_type=sc["ramp_type"],
        tf=1.0,
        dt=0.002,
    )
    res = simulate_v2(
        cfg=cfg,
        fracture_positions=sc["fracture_positions"],
        fracture_Cf=sc["fracture_Cf"],
        fracture_kleak=sc["fracture_kleak"],
        fracture_Kp=sc["fracture_Kp"],
        steady_mode="physical_flow_control",
    )
    assert res["status"] == "success" if "status" in res else True
    assert len(res["wellhead_head"]) > 0


def _dead_end_ss_kwargs(**overrides):
    L = 5000.0
    N = 3448
    dx = L / N
    D = 0.1397
    params = dict(
        L=L,
        N=N,
        dx=dx,
        D=D,
        area=np.pi * (D ** 2) / 4.0,
        nu=1.0e-6,
        K_D=0.045e-3 / D,
        V0=1.0,
        g=G,
        toe_bc="dead_end",
        frac_indices=[2500],
        frac_kleak_arr=np.array([1.0e-4], dtype=np.float64),
        frac_Kp_arr=np.array([1.0e5], dtype=np.float64),
        sorted_pos=[3600.0],
        H_ext=100.0,
    )
    params.update(overrides)
    return params


def test_physical_steady_state_rejects_reservoir_toe_bc():
    """P1: physical 稳态求解器不得用死趾端场去冒充 reservoir 边界。"""
    with pytest.raises(ValueError, match="仅实现封闭趾端"):
        solve_physical_steady_state(**_dead_end_ss_kwargs(toe_bc="reservoir"))


def test_zero_fracture_dead_end_with_injection_is_infeasible():
    """P2: V0>0、零裂缝、封闭趾端没有稳态解，禁止返回全零速度场。"""
    with pytest.raises(InfeasibleSteadyStateError, match="无裂缝"):
        solve_physical_steady_state(
            **_dead_end_ss_kwargs(
                V0=1.0,
                frac_indices=[],
                frac_kleak_arr=np.zeros(0, dtype=np.float64),
                frac_Kp_arr=np.zeros(0, dtype=np.float64),
                sorted_pos=[],
            )
        )

    H0, H_init, V_init, *_rest, mass_res = solve_physical_steady_state(
        **_dead_end_ss_kwargs(
            V0=0.0,
            H0_guess=300.0,
            frac_indices=[],
            frac_kleak_arr=np.zeros(0, dtype=np.float64),
            frac_Kp_arr=np.zeros(0, dtype=np.float64),
            sorted_pos=[],
        )
    )
    assert np.allclose(V_init, 0.0)
    assert np.allclose(H_init, 300.0)
    assert mass_res == 0.0
    assert H0 == 300.0


def test_engineering_head_cap_rejects_infeasible_working_pressure():
    """P1: 滤失过小导致实现水头超出工程上限时必须拒绝。"""
    with pytest.raises(InfeasibleSteadyStateError, match="工程上限"):
        solve_physical_steady_state(
            **_dead_end_ss_kwargs(
                V0=1.5,
                frac_kleak_arr=np.array([1.0e-8], dtype=np.float64),
                H0_max=5000.0,
            )
        )


def test_sampler_respects_head_cap_and_exhausts_instead_of_keeping_reject():
    """P2: 拒绝采样耗尽后抛错，不得沿用未通过约束的最后一次候选。"""
    import moc_simulate.v2.batch.sampler as sampler_mod

    def _always_infeasible(**kwargs):
        raise InfeasibleSteadyStateError("forced infeasible")

    original = sampler_mod.solve_physical_steady_state
    sampler_mod.solve_physical_steady_state = _always_infeasible
    try:
        with pytest.raises(RuntimeError, match="无法在工程水头上限"):
            LatinHypercubeSampler(
                bounds=LhsSamplingBounds(reject_attempts=3),
                seed=1,
            ).sample(n_samples=1)
    finally:
        sampler_mod.solve_physical_steady_state = original

    samples = LatinHypercubeSampler(seed=2026).sample(n_samples=8)
    for s in samples:
        assert s["H0_realized"] <= LhsSamplingBounds().h0_max
        assert s["H0_realized"] > s["H_ext"] + 10.0
