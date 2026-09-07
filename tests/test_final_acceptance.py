# -*- coding: utf-8 -*-
"""
tests/test_final_acceptance.py — MOC 求解器最终修正与生产级验收测试集

覆盖验收要求：
1. solve_fracture_node 返回 5 元组 (H_w, H_f, V_left, V_right, Q_f) 与严格残差；
2. 牛顿迭代未收敛/奇异输入的硬报错测试；
3. 储液项使用 H_f (内部水头) 状态分离测试；
4. 稳态 head_margin (1e-3 m) 硬校验测试（严禁静默截断）；
5. 裂缝位置边界硬校验 (0, L) 测试；
6. 封闭趾端死端无缝与冲突参数校验测试；
7. Dirichlet gammaincinv 采样物理特性与保和精度测试；
8. NPZ 23+ 项元数据与稳态双射关系测试。
"""
import os
import shutil
import tempfile
import numpy as np
import pytest
from scipy.special import gammaincinv

from moc_simulate.wellbore_moc import (
    MocConfig,
    simulate_wellbore,
    solve_fracture_node,
    G,
)
from moc_simulate.config import WELL_CONFIG, FRACTURE_CONFIG
from moc_simulate.lhs_config import LHS_PARAM_RANGES
from moc_simulate.run_lhs_batch_simulate import (
    generate_lhs_params,
    _worker_simulate,
)


def test_solve_fracture_node_returns_5_tuple():
    """1. 验证 solve_fracture_node 返回 5 元组并满足严格残差"""
    area = 0.015328
    ga = G / 1450.0
    dt = 1e-3
    Cf = 1.0e-5
    kleak = 2.0e-4
    H_ext = 100.0
    H_prev = 250.0
    Cp_f = 1.5 + ga * 280.0
    Cm_f = -0.8 + ga * 280.0

    # Rp = 0.0 (基准)
    res_0 = solve_fracture_node(
        Cp_f, Cm_f, H_prev, area, ga, Cf, kleak, H_ext, dt, Rp=0.0
    )
    assert len(res_0) == 5, f"Expected 5-tuple, got {len(res_0)}"
    H_w0, H_f0, vl0, vr0, Qf0 = res_0
    assert abs(H_w0 - H_f0) <= 1e-12, "Rp=0 时 H_w 必须严格等于 H_f"
    assert Qf0 > 0.0, "正压差下裂缝流量应为正"

    # Rp = 80.0 (带孔眼压降)
    Rp = 80.0
    res_rp = solve_fracture_node(
        Cp_f, Cm_f, H_prev, area, ga, Cf, kleak, H_ext, dt, Rp=Rp
    )
    assert len(res_rp) == 5
    H_w_rp, H_f_rp, vl_rp, vr_rp, Qf_rp = res_rp

    # 压降关系 H_w - H_f = Rp * Qf * |Qf|
    dp_calc = H_w_rp - H_f_rp
    dp_theory = Rp * Qf_rp * abs(Qf_rp)
    assert abs(dp_calc - dp_theory) <= 1.0e-8, f"射孔压降残差超标: {abs(dp_calc - dp_theory)}"

    # 质量守恒残差
    r_mass = Qf_rp - Cf * (H_f_rp - H_prev) / dt - kleak * np.sqrt(max(H_f_rp - H_ext, 0.0))
    assert abs(r_mass) <= 1.0e-9, f"裂缝质量守恒残差超标: {abs(r_mass)}"

    # 孔眼阻力阻碍流量 Qf(Rp>0) < Qf(Rp=0)
    assert Qf_rp < Qf0


def test_solve_fracture_node_divergence_raises():
    """2. 验证无法收敛或奇异输入时硬报错"""
    area = 0.015328
    ga = G / 1450.0
    dt = 1e-3

    # 非有限输入
    with pytest.raises(ValueError, match="非有限特征线或历史水头"):
        solve_fracture_node(np.nan, 1.0, 100.0, area, ga, 1e-5, 1e-4, 0.0, dt)

    # 极端无法在 1 步内收敛的情况
    with pytest.raises(RuntimeError, match="未在 1 步内收敛"):
        solve_fracture_node(
            1e5, 1e5, 100.0, area, ga, 1e-5, 1e-4, 0.0, dt,
            Rp=100.0, newton_tol=1e-14, newton_max_iter=1
        )


def test_storage_term_uses_hf_internal():
    """3. 验证储液项独立跟踪 Hf_prev 且 H_w 与 H_f 分离"""
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
        pump_shut_time=1.0,
        initial_velocity=1.0,
        initial_head=350.0,
        toe_bc='dead_end',
        dt=1e-3,
        tf=2.0,
    )
    positions = [4000.0]
    Rp_val = 200.0

    res = simulate_wellbore(
        cfg,
        fracture_positions=positions,
        fracture_compliance_m2=[2.0e-5],
        fracture_inflow_weights=[1.0],
        fracture_Rp=[Rp_val],
        H_ext=100.0,
        store_full_field=False,
    )

    assert "fracture_wellbore_heads" in res
    assert "fracture_internal_heads" in res
    assert "fracture_Qs" in res

    H_w_hist = res["fracture_wellbore_heads"][:, 0]
    H_f_hist = res["fracture_internal_heads"][:, 0]
    Q_f_hist = res["fracture_Qs"][:, 0]

    # 在稳态时 (t < ts=1.0s)，进液量 Q_f > 0，H_w 必须严格高于 H_f
    ss_diff = H_w_hist[0] - H_f_hist[0]
    expected_ss_diff = Rp_val * Q_f_hist[0] * abs(Q_f_hist[0])
    assert abs(ss_diff - expected_ss_diff) <= 1e-8
    assert ss_diff > 0.0, "稳态进液时孔眼压降必须大于 0"


def test_steady_state_head_margin_check():
    """4. 验证稳态内部水头不足 (dH <= 1e-3 m) 硬报错，禁止静默截断"""
    w = WELL_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wellbore_diameter=w['wellbore_diameter'],
        wavespeed=w['wavespeed'],
        friction_model='steady',
        wellhead_bc='velocity_step',
        pump_shut_time=1.0,
        initial_velocity=1.0,
        initial_head=200.0,   # 较低初始水头
        toe_bc='dead_end',
        dt=1e-3,
        tf=1.5,
    )

    # 地层水头过高 (250m > 200m)，必然产生 dH <= 1e-3
    with pytest.raises(ValueError, match="稳态内部水头不足"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_inflow_weights=[1.0],
            H_ext=250.0,
        )

    # 射孔阻力过大导致 H_f = H_w - Rp*Qf^2 下降到 H_ext 以下
    with pytest.raises(ValueError, match="稳态内部水头不足"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_inflow_weights=[1.0],
            fracture_Rp=[1e8], # 极大阻力
            H_ext=100.0,
        )


def test_fracture_position_boundary_validation():
    """5. 验证裂缝位置严格在 (0, L) 内，越界直接报错"""
    w = WELL_CONFIG
    cfg = MocConfig(
        wellbore_length=5000.0,
        wavespeed=1450.0,
        friction_model='steady',
        initial_velocity=1.0,
        initial_head=300.0,
        toe_bc='dead_end',
    )

    # 负数位置
    with pytest.raises(ValueError, match="超出有效范围"):
        simulate_wellbore(cfg, fracture_positions=[-50.0], fracture_inflow_weights=[1.0])

    # 位于井口 x=0
    with pytest.raises(ValueError, match="超出有效范围"):
        simulate_wellbore(cfg, fracture_positions=[0.0], fracture_inflow_weights=[1.0])

    # 位于井底 x=L
    with pytest.raises(ValueError, match="超出有效范围"):
        simulate_wellbore(cfg, fracture_positions=[5000.0], fracture_inflow_weights=[1.0])

    # 超出井底 x > L
    with pytest.raises(ValueError, match="超出有效范围"):
        simulate_wellbore(cfg, fracture_positions=[5100.0], fracture_inflow_weights=[1.0])


def test_dead_end_zero_fracture_validation():
    """6. 验证封闭趾端且 Qin>0 时无缝与参数冲突硬报错"""
    w = WELL_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wavespeed=w['wavespeed'],
        friction_model='steady',
        initial_velocity=1.0,  # Qin > 0
        initial_head=300.0,
        toe_bc='dead_end',
    )

    # 无裂缝
    with pytest.raises(ValueError, match="必须包含至少一条裂缝以泄流"):
        simulate_wellbore(cfg, fracture_positions=None)

    with pytest.raises(ValueError, match="必须包含至少一条裂缝以泄流"):
        simulate_wellbore(cfg, fracture_positions=[])

    # 同时显式指定 inflow_weights 与 kleak 发生冲突
    with pytest.raises(ValueError, match="禁止同时显式传入"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_inflow_weights=[1.0],
            fracture_kleak=[1e-4],
        )


def test_dirichlet_gammaincinv_sampling():
    """7. 验证基于 gammaincinv 的 Dirichlet 采样保和与分布性质"""
    alpha_choices = [0.3, 1.0, 3.0, 10.0]
    n_cl = 4
    rng = np.random.default_rng(12345)

    for alpha in alpha_choices:
        # 生成 100 组均匀 LHS 变量
        u = rng.uniform(1e-10, 1.0 - 1e-10, size=(100, n_cl))
        gamma_vars = gammaincinv(alpha, u)
        sum_g = np.sum(gamma_vars, axis=1, keepdims=True)
        w = gamma_vars / sum_g

        # 验证和为 1.0 与正性
        for row in w:
            assert abs(np.sum(row) - 1.0) <= 1e-12
            assert np.all(row > 0.0)

    # 验证 alpha=0.3 强偏流 vs alpha=10.0 均匀
    u_skew = np.array([[0.01, 0.02, 0.03, 0.99]])
    g_03 = gammaincinv(0.3, u_skew)
    w_03 = g_03 / np.sum(g_03)
    max_w_03 = np.max(w_03)

    g_10 = gammaincinv(10.0, u_skew)
    w_10 = g_10 / np.sum(g_10)
    max_w_10 = np.max(w_10)

    # alpha=0.3 时最大权重明显更具支配性
    assert max_w_03 > max_w_10


def test_npz_metadata_and_bijection():
    """8. 验证批量运行生成 NPZ 的 23+ 项元数据及稳态双射关系"""
    temp_dir = tempfile.mkdtemp(prefix="moc_test_acceptance_")
    out_data_dir = os.path.join(temp_dir, "data")
    os.makedirs(out_data_dir, exist_ok=True)

    try:
        samples = generate_lhs_params(n_samples=4, seed=42, domain_randomization=False)
        assert len(samples) == 4

        # 运行 Worker 仿真单条样本
        sample = samples[0]
        args = (
            sample,
            "steady",
            2.0,   # tf
            1e-3,  # dt
            out_data_dir,
            1450.0,
            1.0,
            False,
        )
        res_info = _worker_simulate(args)
        assert res_info["status"] == "PASS", f"Worker 仿真失败: {res_info.get('error')}"

        npz_file = os.path.join(out_data_dir, res_info["npz_file"])
        assert os.path.exists(npz_file)

        data = np.load(npz_file)
        expected_keys = [
            "schema_version", "seed", "N", "wellbore_length",
            "dt_requested", "dt_adj", "wavespeed_requested", "wavespeed_adj",
            "x_f_requested", "x_f_aligned", "fracture_indices",
            "t", "H_wh", "Q_wh",
            "x_f", "x_f_raw", "grid_index",
            "dx", "dt", "wavespeed", "wavespeed_nominal",
            "friction", "brunone_k_scale",
            "toe_bc", "initial_head", "initial_velocity", "H_ext",
            "Rp",
            "compliance_head_m2", "Cf",
            "inflow_weight",
            "kleak_equiv", "kleak",
            "Hw_ss", "Hf_ss", "Qf_ss", "Qin_ss",
            "alpha_dirichlet",
            "n_frac", "tf", "case_id",
        ]
        for key in expected_keys:
            assert key in data, f"NPZ 缺少元数据字段: {key}"

        # 验证 x_f 严格对齐到网格坐标
        dx = float(data["dx"])
        grid_idx = data["grid_index"]
        x_f_aligned = grid_idx * dx
        np.testing.assert_allclose(data["x_f"], x_f_aligned, rtol=1e-5, atol=1e-5)

        # 验证稳态双射关系
        Qf_ss = data["Qf_ss"]
        kleak_equiv = data["kleak_equiv"]
        Hf_ss = data["Hf_ss"]
        H_ext = float(data["H_ext"])
        Qin_ss = float(data["Qin_ss"])

        Q_theory = kleak_equiv * np.sqrt(Hf_ss - H_ext)
        np.testing.assert_allclose(Qf_ss, Q_theory, atol=1e-10)
        assert abs(np.sum(Qf_ss) - Qin_ss) <= 1e-10

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



def test_rejects_negative_inflow_weight():
    """验证负权重输入被严格拒绝 (如 [1.2, -0.2])"""
    w = WELL_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wavespeed=w['wavespeed'],
        initial_velocity=1.0,
        toe_bc='dead_end',
    )
    # 反例 1: [1.2, -0.2] (和为 1 但含负值)
    with pytest.raises(ValueError, match="必须全部非负"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0, 4200.0],
            fracture_inflow_weights=[1.2, -0.2],
            H_ext=100.0,
        )

    # 反例 2: [-0.5, 1.5]
    with pytest.raises(ValueError, match="必须全部非负"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0, 4200.0],
            fracture_inflow_weights=[-0.5, 1.5],
            H_ext=100.0,
        )


def test_rejects_nan_and_inf_parameters():
    """验证 NaN 与 Inf 参数被严格拒绝：inflow_weights, Cf, Rp, H_ext, kleak 等，以及负值校验"""
    w = WELL_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wavespeed=w['wavespeed'],
        initial_velocity=1.0,
        toe_bc='dead_end',
    )

    # 1. inflow_weights=[np.nan, 1.0]
    with pytest.raises(ValueError, match="非有限数值"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0, 4200.0],
            fracture_inflow_weights=[np.nan, 1.0],
            H_ext=100.0,
        )

    # 2. Cf=[np.inf] 与 Cf=[np.nan]
    with pytest.raises(ValueError, match="非有限数值"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[np.inf],
            fracture_inflow_weights=[1.0],
            H_ext=100.0,
        )

    with pytest.raises(ValueError, match="非有限数值"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[np.nan],
            fracture_inflow_weights=[1.0],
            H_ext=100.0,
        )

    # 3. Cf=[-1e-5] (负柔度)
    with pytest.raises(ValueError, match="必须非负"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[-1e-5],
            fracture_inflow_weights=[1.0],
            H_ext=100.0,
        )

    # 4. Rp=[np.nan] 与 Rp=[-10.0] (负孔眼阻力)
    with pytest.raises(ValueError, match="非有限数值"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_inflow_weights=[1.0],
            fracture_Rp=[np.nan],
            H_ext=100.0,
        )

    with pytest.raises(ValueError, match="必须非负"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_inflow_weights=[1.0],
            fracture_Rp=[-10.0],
            H_ext=100.0,
        )

    # 5. kleak=[np.nan] 与 kleak=[-1e-4] (在 reservoir 模式下)
    cfg_res = MocConfig(
        wellbore_length=w['L'],
        wavespeed=w['wavespeed'],
        initial_velocity=1.0,
        toe_bc='reservoir',
        toe_head=300.0,
    )
    with pytest.raises(ValueError, match="非有限数值"):
        simulate_wellbore(
            cfg_res,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_kleak=[np.nan],
            H_ext=100.0,
        )

    with pytest.raises(ValueError, match="必须非负"):
        simulate_wellbore(
            cfg_res,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_kleak=[-1e-4],
            H_ext=100.0,
        )

    # 6. H_ext=np.nan
    with pytest.raises(ValueError, match="有限数值"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_inflow_weights=[1.0],
            H_ext=np.nan,
        )

    # 7. fracture_positions=[np.nan]
    with pytest.raises(ValueError, match="非有限数值"):
        simulate_wellbore(
            cfg,
            fracture_positions=[np.nan],
            fracture_inflow_weights=[1.0],
            H_ext=100.0,
        )

    # 8. 无裂缝位置时传入裂缝参数
    with pytest.raises(ValueError, match="未指定 fracture_positions"):
        simulate_wellbore(
            cfg_res,
            fracture_positions=None,
            fracture_compliance_m2=[1e-5],
        )

    # 9. 配置关键数值为 NaN / Inf
    with pytest.raises(ValueError, match="有限数值"):
        MocConfig(wavespeed=np.nan)

    with pytest.raises(ValueError, match="有限数值"):
        MocConfig(dt=np.inf)


def test_rejects_dead_end_kleak_only_mode():
    """验证封闭趾端 (dead_end) 且 V0>0 时仅输入 kleak 被严格拒绝"""
    w = WELL_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wavespeed=w['wavespeed'],
        initial_velocity=1.0,
        toe_bc='dead_end',
    )
    with pytest.raises(ValueError, match="禁止仅传入 fracture_kleak"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_kleak=[1e-4],
            fracture_inflow_weights=None,
            H_ext=100.0,
        )


def test_rejects_missing_dead_end_weights():
    """验证封闭趾端 (dead_end) 且 V0>0 时未提供 weights 或同时提供 weights 与 kleak 被拒绝"""
    w = WELL_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wavespeed=w['wavespeed'],
        initial_velocity=1.0,
        toe_bc='dead_end',
    )

    # 权重和 kleak 均未提供
    with pytest.raises(ValueError, match="必须提供 fracture_inflow_weights"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_kleak=None,
            fracture_inflow_weights=None,
            H_ext=100.0,
        )

    # 同时提供权重与 kleak
    with pytest.raises(ValueError, match="禁止同时显式传入"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0],
            fracture_compliance_m2=[1e-5],
            fracture_kleak=[1e-4],
            fracture_inflow_weights=[1.0],
            H_ext=100.0,
        )

    # 权重和明显偏离 1.0 (禁止静默归一化)
    with pytest.raises(ValueError, match="禁止静默归一化"):
        simulate_wellbore(
            cfg,
            fracture_positions=[4000.0, 4200.0],
            fracture_compliance_m2=[1e-5, 1e-5],
            fracture_inflow_weights=[0.5, 0.4],
            H_ext=100.0,
        )


def test_npz_contains_version_and_grid_metadata():
    """验证生成的 NPZ 包含新版 schema_version、seed、网格与波速元数据字段"""
    temp_dir = tempfile.mkdtemp(prefix="moc_test_v21_")
    out_data_dir = os.path.join(temp_dir, "data")
    os.makedirs(out_data_dir, exist_ok=True)

    try:
        seed = 20260907
        samples = generate_lhs_params(n_samples=2, seed=seed, domain_randomization=False)
        sample = samples[0]

        args = (
            sample,
            "steady",
            2.0,   # tf
            1e-3,  # dt
            out_data_dir,
            1450.0,
            1.0,
            False,
            seed,
        )
        res_info = _worker_simulate(args)
        assert res_info["status"] == "PASS"

        npz_file = os.path.join(out_data_dir, res_info["npz_file"])
        assert os.path.exists(npz_file)

        data = np.load(npz_file)

        # 检查要求的新版字段
        assert str(data["schema_version"]) == "moc_lhs_v2.1"
        assert int(data["seed"]) == seed
        assert int(data["N"]) > 0
        assert float(data["wellbore_length"]) > 0.0

        assert float(data["dt_requested"]) == 1e-3
        assert float(data["dt_adj"]) == float(data["dt_requested"])

        assert float(data["wavespeed_requested"]) == 1450.0
        assert np.isfinite(float(data["wavespeed_adj"]))

        # 裂缝位置与对齐
        x_f_req = np.asarray(data["x_f_requested"], dtype=float)
        x_f_ali = np.asarray(data["x_f_aligned"], dtype=float)
        frac_idx = np.asarray(data["fracture_indices"], dtype=int)
        assert len(x_f_req) == sample["n_frac"]
        assert len(x_f_ali) == sample["n_frac"]
        assert len(frac_idx) == sample["n_frac"]

        # 别名兼容性
        np.testing.assert_array_equal(data["x_f"], x_f_ali)
        np.testing.assert_array_equal(data["x_f_raw"], x_f_req)
        np.testing.assert_array_equal(data["grid_index"], frac_idx)
        assert float(data["dt"]) == float(data["dt_adj"])
        assert float(data["wavespeed"]) == float(data["wavespeed_adj"])

        # 物理有效性
        Cf_arr = np.asarray(data["Cf"], dtype=float)
        w_arr = np.asarray(data["inflow_weight"], dtype=float)
        kleak_arr = np.asarray(data["kleak_equiv"], dtype=float)
        assert np.all(np.isfinite(Cf_arr)) and np.all(Cf_arr >= 0.0)
        assert np.all(np.isfinite(w_arr)) and np.all(w_arr >= 0.0)
        assert abs(np.sum(w_arr) - 1.0) <= 1e-10
        assert np.all(np.isfinite(kleak_arr)) and np.all(kleak_arr >= 0.0)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_same_seed_reproduces_parameters_and_labels():
    """验证相同种子重复生成时，参数、标签与波形逐项完全一致"""
    seed = 42
    samples1 = generate_lhs_params(n_samples=3, seed=seed, domain_randomization=True)
    samples2 = generate_lhs_params(n_samples=3, seed=seed, domain_randomization=True)

    # 1. 验证参数字典逐项一致
    assert len(samples1) == len(samples2)
    for s1, s2 in zip(samples1, samples2):
        assert s1["case_id"] == s2["case_id"]
        assert s1["seed"] == s2["seed"]
        assert s1["n_frac"] == s2["n_frac"]
        np.testing.assert_allclose(s1["positions"], s2["positions"], atol=1e-12)
        np.testing.assert_allclose(s1["Cf_list"], s2["Cf_list"], atol=1e-12)
        np.testing.assert_allclose(s1["inflow_weights"], s2["inflow_weights"], atol=1e-12)
        assert s1["wavespeed"] == s2["wavespeed"]
        assert s1["friction_model"] == s2["friction_model"]

    # 2. 独立运行仿真验证 NPZ 输出逐项一致
    temp_dir1 = tempfile.mkdtemp(prefix="moc_repro_1_")
    temp_dir2 = tempfile.mkdtemp(prefix="moc_repro_2_")
    try:
        out1 = os.path.join(temp_dir1, "data")
        out2 = os.path.join(temp_dir2, "data")
        os.makedirs(out1, exist_ok=True)
        os.makedirs(out2, exist_ok=True)

        res1 = _worker_simulate((samples1[0], "steady", 2.0, 1e-3, out1, 1450.0, 1.0, False, seed))
        res2 = _worker_simulate((samples2[0], "steady", 2.0, 1e-3, out2, 1450.0, 1.0, False, seed))

        assert res1["status"] == "PASS"
        assert res2["status"] == "PASS"

        data1 = np.load(os.path.join(out1, res1["npz_file"]))
        data2 = np.load(os.path.join(out2, res2["npz_file"]))

        for k in ["schema_version", "seed", "N", "wellbore_length", "dt_requested", "dt_adj",
                  "wavespeed_requested", "wavespeed_adj"]:
            assert data1[k] == data2[k], f"标量元数据不一致: {k}"

        for arr_key in ["t", "H_wh", "Q_wh", "x_f_requested", "x_f_aligned", "fracture_indices",
                        "inflow_weight", "kleak_equiv", "Hw_ss", "Hf_ss", "Qf_ss", "Qin_ss"]:
            np.testing.assert_array_equal(data1[arr_key], data2[arr_key], err_msg=f"数组不一致: {arr_key}")

    finally:
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)
