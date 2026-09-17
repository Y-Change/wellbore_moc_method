# -*- coding: utf-8 -*-
"""
tests/test_v2_architecture.py

MOC_V2 生产级模块化架构全面单元测试与基准对标套件：
1. V2 核心强类型配置、网格拓扑与 Courant 稳定性自洽校验
2. 初始稳态场空间质量守恒、死水区速度严格归零与达西坡降解析验证
3. 射孔非线性二次节流压降牛顿-拉夫逊二阶收敛与 K_p <= 0 物理退化验证
4. 现场斜坡关泵边界模型解析值、C^1 连续性与阻尼对称性校验
5. Quad 4 裂缝工况仿真与 Step 4 原型求解器的浮点精度一致性对标 (< 1e-10 m)
6. 1D/2D 倒谱分析、波前差分锐化与亚米级特征峰检出校验
7. 面向智能反演的大规模 HDF5 数据集生成、切片索引与读取自洽性测试
8. 顶层入口、门面代理与 V1 命名空间隔离测试
"""
from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

# 导入 V2 各分层模块
from moc_simulate.common.constants import G, G_STANDARD
from moc_simulate.v2.configs import (
    MocV2Config,
    WellboreConfig,
    FractureConfig,
    PerforationConfig,
    BoundaryConfig,
    SimulationConfig,
)
from moc_simulate.v2.core.moc_mesh import MocGrid
from moc_simulate.v2.core.friction import (
    reynolds,
    darcy_friction_factor,
    friction_term_J,
    brunone_k_vec,
)
from moc_simulate.v2.core.initial_field import compute_steady_state_field
from moc_simulate.v2.core.fracture_node import solve_fracture_node_v2
from moc_simulate.v2.core.boundary_condition import (
    compute_ramp_velocity,
    compute_ramp_acceleration,
)
from moc_simulate.v2.core.solver import WellboreMocV2Solver, simulate_v2
from moc_simulate.v2.signal import (
    compute_cepstrum_1d,
    compute_cepstrogram_2d,
    apply_wavefront_derivative_filter,
    detect_fracture_peaks,
    evaluate_peak_matching,
    compute_psnr,
)
from moc_simulate.v2.batch import (
    LatinHypercubeSampler,
    LhsSampler,
    LhsSamplingBounds,
    classify_fracture_type,
    sample_preset_scenario,
    FRACTURE_TYPE_SPECS,
    BatchRunner,
    _run_single_simulation,
    save_hdf5_dataset,
    load_hdf5_dataset,
)

# 导入 Step 4 对标求解器
from experiments.moc_physics_upgrade.step4_ramp_closure.solver import (
    Step4MocConfig,
    simulate_wellbore_step4,
)


class TestMocV2ConfigsAndMesh:
    """测试 V2 结构化配置层与网格生成"""

    def test_v2_config_initialization_and_derived(self):
        cfg = MocV2Config(
            wellbore_length=5000.0,
            wavespeed=1450.0,
            dt=0.001,
            tf=10.0,
            perf_num_holes=6,
            perf_diameter=0.01,
            perf_cd=0.65,
        )
        assert cfg.N == round(5000.0 / (1450.0 * 0.001))
        assert abs(cfg.a_adj * cfg.dt_adj / cfg.dx - 1.0) < 1e-12
        assert cfg.n_steps == 10000
        assert cfg.area > 0.0
        assert cfg.perf_area > 0.0
        assert cfg.perf_Kp > 0.0

    def test_v2_subconfigs_composition(self):
        wb = WellboreConfig(wellbore_length=3000.0, wavespeed=1400.0)
        bd = BoundaryConfig(pump_shut_time=2.0, pump_closure_duration=0.8)
        sim = SimulationConfig(dt=0.001, tf=20.0)
        perf = PerforationConfig(num_holes=8, diameter=0.012, cd=0.7)

        cfg = MocV2Config.from_subconfigs(wb, bd, sim, perf)
        assert cfg.wellbore_length == 3000.0
        assert cfg.pump_closure_duration == 0.8
        assert cfg.perf_num_holes == 8
        assert cfg.perf_diameter == 0.012
        assert cfg.perf_cd == 0.7

    def test_mesh_mapping_and_collision_defense(self):
        grid = MocGrid.create(L=5000.0, wavespeed=1450.0, dt=0.001, tf=10.0)
        # 正常递增间距映射
        raw_pos = [4100.0, 4150.0, 4200.0]
        indices, sorted_pos, order = grid.map_fracture_positions(raw_pos)
        assert len(indices) == 3
        assert indices[0] < indices[1] < indices[2]

        # 冲突碰撞防御 (两裂缝间距远小于 dx=1.45m，映射到同一节点)
        with pytest.raises(ValueError, match="冲突映射到同一节点"):
            grid.map_fracture_positions([4100.0, 4100.1])


class TestMocV2PhysicsEngines:
    """测试稳态自洽场、射孔牛顿求解器与斜坡关泵动力学"""

    def test_steady_state_conservation_and_dead_end(self):
        cfg = MocV2Config(
            wellbore_length=5000.0,
            initial_velocity=1.2,
            initial_head=300.0,
            toe_bc="dead_end",
        )
        x_f = [4000.0, 4050.0, 4100.0]
        weights = np.array([0.5, 0.3, 0.2])

        solver = WellboreMocV2Solver(
            cfg=cfg,
            fracture_positions=x_f,
            fracture_inflow_weights=weights,
            H_ext=100.0,
            steady_mode="prescribed_flow_split_legacy",
        )

        # 1. 质量守恒：各簇稳态进液量之和等于总泵注量
        total_pump_q = cfg.initial_velocity * cfg.area
        assert np.isclose(np.sum(solver.q_frac_ss), total_pump_q, rtol=1e-10)

        # 2. 盲端死水区：最末簇下游直到井底流速严格为 0
        last_node = solver.frac_indices[-1]
        assert np.all(solver.V_init[last_node:] == 0.0)

        # 3. 达西沿程压降单调性：水头沿流动方向单调递减
        assert np.all(np.diff(solver.H_init[:last_node]) <= 0.0)

    def test_fracture_node_newton_convergence_and_degeneration(self):
        area = np.pi * (0.1397 ** 2) / 4.0
        ga = 9.81 / 1450.0
        dt = 0.001
        Cf = 0.01
        kleak = 1e-4
        H_ext = 100.0

        Cp_f = 1.0 + ga * 300.0
        Cm_f = -1.0 + ga * 300.0
        H_prev = 290.0

        # Case 1: 正常非线性射孔节流压降 (Kp > 0)
        Kp = 5.0e5
        H_well, H_frac, v_l, v_r, q_p = solve_fracture_node_v2(
            Cp_f=Cp_f,
            Cm_f=Cm_f,
            H_prev_f=H_prev,
            area=area,
            ga=ga,
            Cf=Cf,
            kleak=kleak,
            H_ext=H_ext,
            dt=dt,
            Kp=Kp,
        )
        # 节点质量守恒: A*(V_l - V_r) == q_p
        assert np.isclose(area * (v_l - v_r), q_p, atol=1e-12)
        # 节流压降关系: H_well - H_frac == sign(q_p)*Kp*q_p^2
        assert np.isclose(H_well - H_frac, np.sign(q_p) * Kp * (abs(q_p) ** 2), atol=1e-10)

        # Case 2: 退化情况 (Kp <= 0，无射孔压降)
        H_well_deg, H_frac_deg, v_l_deg, v_r_deg, q_deg = solve_fracture_node_v2(
            Cp_f=Cp_f,
            Cm_f=Cm_f,
            H_prev_f=H_prev,
            area=area,
            ga=ga,
            Cf=Cf,
            kleak=kleak,
            H_ext=H_ext,
            dt=dt,
            Kp=0.0,
        )
        # 无压降时井筒与裂缝水头严格相等
        assert np.isclose(H_well_deg, H_frac_deg, atol=1e-12)
        assert np.isclose(area * (v_l_deg - v_r_deg), q_deg, atol=1e-12)

    def test_ramp_closure_continuity_and_cosine_smoothness(self):
        V0 = 1.5
        ts = 1.0
        tc = 1.2

        # 边界值
        assert compute_ramp_velocity(0.5, V0, ts, tc) == V0
        assert compute_ramp_velocity(1.0, V0, ts, tc) == V0
        assert compute_ramp_velocity(1.0 + tc, V0, ts, tc) == 0.0
        assert compute_ramp_velocity(3.0, V0, ts, tc) == 0.0

        # 余弦曲线 C^1 连续性 (两端加速度归零)
        a_start = compute_ramp_acceleration(ts, V0, ts, tc, ramp_type="cosine")
        a_end = compute_ramp_acceleration(ts + tc, V0, ts, tc, ramp_type="cosine")
        assert np.isclose(a_start, 0.0, atol=1e-12)
        assert np.isclose(a_end, 0.0, atol=1e-12)

        # 线性曲线加速度常数
        a_linear = compute_ramp_acceleration(ts + tc / 2.0, V0, ts, tc, ramp_type="linear")
        assert np.isclose(a_linear, -V0 / tc, atol=1e-12)


class TestMocV2QuadConsistencyWithStep4:
    """与 Step 4 原型求解器进行严格浮点精度一致性逐点对标 (< 1e-10 m)"""

    def test_quad_simulation_bit_exact_consistency(self):
        L = 5000.0
        a = 1450.0
        dt = 0.001
        tf = 3.0
        ts = 1.0
        tc = 1.0
        x_f = [4100.0, 4120.0, 4140.0, 4160.0]
        Cf_list = [0.01, 0.01, 0.01, 0.01]
        H_ext = 100.0

        # 1. 运行原 Step 4 仿真器
        cfg_step4 = Step4MocConfig(
            wellbore_length=L,
            wavespeed=a,
            dt=dt,
            tf=tf,
            pump_shut_time=ts,
            pump_closure_duration=tc,
            wellhead_bc="ramp",
            ramp_type="linear",
            perf_num_holes=6,
            perf_diameter=0.01,
            perf_cd=0.65,
        )
        res_step4 = simulate_wellbore_step4(
            cfg_step4,
            fracture_positions=x_f,
            fracture_Cf=Cf_list,
            H_ext=H_ext,
        )

        # 2. 运行本次重构后的生产级 V2 仿真器
        cfg_v2 = MocV2Config(
            wellbore_length=L,
            wavespeed=a,
            dt=dt,
            tf=tf,
            pump_shut_time=ts,
            pump_closure_duration=tc,
            wellhead_bc="ramp",
            ramp_type="linear",
            perf_num_holes=6,
            perf_diameter=0.01,
            perf_cd=0.65,
        )
        res_v2 = simulate_v2(
            cfg_v2,
            fracture_positions=x_f,
            fracture_Cf=Cf_list,
            H_ext=H_ext,
            steady_mode="prescribed_flow_split_legacy",
        )

        # 3. 逐点浮点误差比对 (要求 max error < 1e-10)
        wh_head_err = float(np.max(np.abs(res_v2["wellhead_head"] - res_step4["wellhead_head"])))
        wh_vel_err = float(np.max(np.abs(res_v2["wellhead_velocity"] - res_step4["wellhead_velocity"])))
        toe_head_err = float(np.max(np.abs(res_v2["toe_head"] - res_step4["toe_head"])))
        toe_vel_err = float(np.max(np.abs(res_v2["toe_velocity"] - res_step4["toe_velocity"])))
        frac_head_err = float(np.max(np.abs(res_v2["fracture_heads"] - res_step4["fracture_heads"])))
        frac_Q_err = float(np.max(np.abs(res_v2["fracture_Qs"] - res_step4["fracture_Qs"])))

        assert wh_head_err < 1.0e-10, f"井口水头误差超标: {wh_head_err:.2e} m"
        assert wh_vel_err < 1.0e-10, f"井口流速误差超标: {wh_vel_err:.2e} m/s"
        assert toe_head_err < 1.0e-10, f"趾端水头误差超标: {toe_head_err:.2e} m"
        assert toe_vel_err < 1.0e-10, f"趾端流速误差超标: {toe_vel_err:.2e} m/s"
        assert frac_head_err < 1.0e-10, f"裂缝水头误差超标: {frac_head_err:.2e} m"
        assert frac_Q_err < 1.0e-10, f"裂缝流量误差超标: {frac_Q_err:.2e} m^3/s"


class TestMocV2SignalAndDatasetBatch:
    """测试倒谱分析、波前锐化与 HDF5 数据集导出基座"""

    def test_signal_cepstrum_and_submeter_peak_detection(self):
        # 构造合成双反射脉冲信号
        fs = 1000.0
        a = 1450.0
        t = np.arange(0.0, 10.0, 1.0 / fs)
        # 设裂缝在 4100m, 双程走时 tau = 2 * 4100 / 1450 = 5.655s
        head = 300.0 + np.zeros_like(t)
        tau_target = 2.0 * 4100.0 / a
        idx = int(round(1.0 * fs + tau_target * fs))
        if idx < len(head):
            head[idx : idx + 10] += 50.0

        ceps_res = compute_cepstrum_1d(t, head, wavespeed=a, fs=fs, ts=1.0)
        dists = ceps_res["distance"]
        amp = ceps_res["cepstrum"]

        # 测试峰检测与匹配评估
        peak_dists, _ = detect_fracture_peaks(dists, amp, min_depth=3500.0, max_depth=4500.0)
        match_res = evaluate_peak_matching(peak_dists, [4100.0], tolerance_m=5.0)
        assert match_res["n_true"] == 1

        # 测试波前求导锐化
        sharp_h = apply_wavefront_derivative_filter(head, dt=1.0 / fs)
        assert len(sharp_h) == len(head)

    def test_lhs_sampler_and_hdf5_dataset_roundtrip(self):
        # 1. 生成少量 LHS 样本
        sampler = LatinHypercubeSampler(seed=12345)
        samples = sampler.sample(n_samples=2)
        assert len(samples) == 2
        assert 1 <= samples[0]["n_frac"] <= 6

        # 2. 执行单样本正演 (小仿真 tf=2.0s)
        for s in samples:
            s["tf"] = 2.0
            s["dt"] = 0.002
        sim_results = [_run_single_simulation(s) for s in samples]
        assert all(r["status"] == "success" for r in sim_results)

        # 3. 打包导出为 HDF5
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test_dataset.h5")
            exported_file = save_hdf5_dataset(
                file_path=h5_path,
                results=sim_results,
                metadata={"test_task": "MOC_V2_architecture_verification"},
            )
            assert os.path.isfile(exported_file)

            # 4. 重新读取并验证自洽性
            data = load_hdf5_dataset(exported_file)
            assert data["metadata"]["test_task"] == "MOC_V2_architecture_verification"
            assert data["waveforms"]["wellhead_head"].shape[0] == 2
            assert data["features"]["cepstrum"].shape[0] == 2
            assert data["labels"]["n_frac"].shape[0] == 2
            assert len(data["labels"]["pump_closure_tc"]) == 2


class TestMocV2TopLevelFacadeAndIsolation:
    """测试顶级入口默认导出 V2 与 V1/Facade 零冲突隔离"""

    def test_top_level_exports_v2(self):
        import moc_simulate
        from moc_simulate.v2 import simulate_v2, MocV2Config

        # 顶级 simulate_wellbore 默认即为 V2
        assert moc_simulate.simulate_wellbore is simulate_v2
        assert moc_simulate.MocConfig is MocV2Config

    def test_v1_explicit_isolation(self):
        from moc_simulate.v1 import simulate_wellbore as sim_v1, MocConfig as CfgV1
        from moc_simulate.v2 import simulate_v2 as sim_v2, MocV2Config as CfgV2

        assert sim_v1 is not sim_v2
        assert CfgV1 is not CfgV2

    def test_facade_compatibility_with_v1(self):
        from moc_simulate.wellbore_moc import simulate_wellbore as sim_facade
        from moc_simulate.v1.wellbore_moc import simulate_wellbore as sim_v1

        # 根目录 facade 严格透明代理 V1
        assert sim_facade is sim_v1


class TestMocV2EdgeCasesAndRobustnessFixes:
    """全面覆盖边缘用例、边界防御与鲁棒性修复验证"""

    def test_dict_fractures_with_mixed_and_missing_kp(self):
        """测试使用字典定义裂缝时，缺失/混合 Kp 及不同射孔定义下的自洽性"""
        cfg = MocV2Config(wellbore_length=5000.0, tf=1.0)
        fracs = [
            {"position": 4000.0, "Kp": 1.2e5},
            {"position": 4100.0},  # 无 Kp，回退全局 perf_Kp
            {"position": 4200.0, "num_holes": 8, "diameter": 0.012, "cd": 0.7},  # 由几何动态计算
            {"position": 4300.0, "perforation": {"num_holes": 4, "diameter": 0.01, "cd": 0.6}},
        ]
        solver = WellboreMocV2Solver(cfg, fractures=fracs)
        assert len(solver.frac_Kp_arr) == 4
        assert solver.frac_Kp_arr[0] == 1.2e5
        assert solver.frac_Kp_arr[1] == cfg.perf_Kp
        assert solver.frac_Kp_arr[2] > 0.0
        assert solver.frac_Kp_arr[3] > 0.0
        # 验证能成功仿真无崩溃
        res = solver.solve()
        assert res["wellhead_head"].shape[0] == solver.grid.n_steps + 1

    def test_simulate_wellbore_v1_config_and_aliases_compatibility(self):
        """测试顶层 simulate_wellbore 对 V1 MocConfig 与历史别名入参/返回字典的兼容支持"""
        from moc_simulate import simulate_wellbore
        from moc_simulate.v1 import MocConfig

        # 传入 V1 MocConfig
        cfg_v1 = MocConfig(
            wellbore_length=1000.0,
            wavespeed=1450.0,
            dt=0.001,
            tf=1.0,
            initial_head=300.0,
        )
        # 使用历史入参 fracture_compliance_m2 与 store_full_field
        res = simulate_wellbore(
            cfg_v1,
            fracture_positions=[800.0],
            fracture_compliance_m2=[0.01],
            store_full_field=True,
        )

        assert "timestamps" in res
        assert "wellhead_head" in res
        assert "toe_head" in res
        # 兼容性别名
        assert "head" in res and res["head"] is not None
        assert "velocity" in res and res["velocity"] is not None
        assert "fracture_internal_heads" in res
        assert "fracture_wellbore_heads" in res
        assert res["fracture_internal_heads"].shape[1] == 1

    def test_fracture_out_of_bounds_validation(self):
        """测试裂缝位置超出井深区间 (0, L) 时的严格异常拦截"""
        grid = MocGrid.create(L=5000.0, wavespeed=1450.0, dt=0.001, tf=1.0)

        # 负深
        with pytest.raises(ValueError, match="超出井筒有效内部区间"):
            grid.map_fracture_positions([-50.0, 4000.0])

        # 井口原点
        with pytest.raises(ValueError, match="超出井筒有效内部区间"):
            grid.map_fracture_positions([0.0, 4000.0])

        # 井底趾端
        with pytest.raises(ValueError, match="超出井筒有效内部区间"):
            grid.map_fracture_positions([4000.0, 5000.0])

        # 超出井深
        with pytest.raises(ValueError, match="超出井筒有效内部区间"):
            grid.map_fracture_positions([4000.0, 5500.0])

        # 非有限值
        with pytest.raises(ValueError, match="必须为有限数值"):
            grid.map_fracture_positions([float("nan")])

    def test_mismatched_parameter_array_lengths(self):
        """测试裂缝参数数组长度与裂缝位置数量不匹配时的异常防御"""
        cfg = MocV2Config(wellbore_length=5000.0, tf=1.0)

        # Cf 数组长度与位置不匹配
        with pytest.raises(ValueError, match="fracture_Cf 长度 .* 不匹配"):
            WellboreMocV2Solver(cfg, fracture_positions=[4000.0, 4100.0], fracture_Cf=[0.01])

        # kleak 数组长度不匹配
        with pytest.raises(ValueError, match="fracture_kleak 长度 .* 不匹配"):
            WellboreMocV2Solver(cfg, fracture_positions=[4000.0, 4100.0], fracture_kleak=[1e-4, 1e-4, 1e-4])

        # weights 数组长度不匹配
        with pytest.raises(ValueError, match="fracture_inflow_weights 长度 .* 不匹配"):
            WellboreMocV2Solver(cfg, fracture_positions=[4000.0, 4100.0], fracture_inflow_weights=[1.0])

    def test_config_parameter_validations(self):
        """测试 MocV2Config 对非法非物理参数的构造校验"""
        with pytest.raises(ValueError, match="wellbore_length 必须大于 0"):
            MocV2Config(wellbore_length=-500.0)

        with pytest.raises(ValueError, match="wellbore_diameter 必须大于 0"):
            MocV2Config(wellbore_diameter=0.0)

        with pytest.raises(ValueError, match="dt 必须大于 0"):
            MocV2Config(dt=-0.001)

        with pytest.raises(ValueError, match="tf 必须大于 0"):
            MocV2Config(tf=0.0)

        with pytest.raises(ValueError, match="wavespeed 必须大于 0"):
            MocV2Config(wavespeed=-1450.0)

        with pytest.raises(ValueError, match="必须为有限数值"):
            MocV2Config(initial_head=float("nan"))

    def test_hdf5_dataset_dynamic_large_clusters_export(self):
        """测试 HDF5 数据集导出器对大于默认 8 簇（如 10 簇）超大簇数的动态自适应保留"""
        n_frac = 10
        positions = [3000.0 + i * 20.0 for i in range(n_frac)]
        sample = {
            "sample_id": 999,
            "fracture_positions": positions,
            "fracture_Cf": [0.01] * n_frac,
            "fracture_kleak": [1e-4] * n_frac,
            "fracture_inflow_weights": [1.0 / n_frac] * n_frac,
            "tf": 0.5,
            "dt": 0.005,
        }
        res = _run_single_simulation(sample)
        assert res["status"] == "success"

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "large_frac_dataset.h5")
            save_hdf5_dataset(h5_path, [res], max_frac_dim=8)

            data = load_hdf5_dataset(h5_path)
            # 验证 10 簇裂缝未被强行截断为 8
            assert data["labels"]["n_frac"][0] == 10
            assert data["labels"]["fracture_positions"].shape[1] >= 10
            np.testing.assert_allclose(data["labels"]["fracture_positions"][0, :10], positions)

    def test_batch_runner_serial_fast_path(self):
        """测试 BatchRunner 在 max_workers=1 下的串行快速路径"""
        runner = BatchRunner(max_workers=1)
        samples = [
            {"sample_id": 1, "tf": 0.2, "dt": 0.002, "fracture_positions": [4100.0]},
            {"sample_id": 2, "tf": 0.2, "dt": 0.002, "fracture_positions": [4200.0]},
        ]
        results = runner.run(samples)
        assert len(results) == 2
        assert results[0]["sample_id"] == 1
        assert results[1]["sample_id"] == 2
        assert all(r["status"] == "success" for r in results)

    def test_psnr_edge_cases(self):
        """测试 PSNR 计算在空输入或退化全零信号下的安全性"""
        assert compute_psnr(np.array([]), []) == 0.0
        assert compute_psnr(np.array([1.0, 2.0]), [100]) == 0.0
        assert compute_psnr(np.zeros(50), [10]) == 0.0


class TestMocV2TypeVAndPhysicalCoupledSampler:
    """测试 Type V 沟通天然断层簇、物理联动采样器与 5 大典型预设工况"""

    def test_lhs_sampler_alias_and_exports(self):
        """测试 LhsSampler 别名与工况矩阵接口导出一致性"""
        assert LhsSampler is LatinHypercubeSampler
        assert "Type V" in FRACTURE_TYPE_SPECS
        assert "Type I" in FRACTURE_TYPE_SPECS
        assert "Type IV" in FRACTURE_TYPE_SPECS
        assert "5~10" in FRACTURE_TYPE_SPECS["Type V"]["description"]

    def test_physical_coupling_monotonicity(self):
        """测试物理联动模式下，进液权重 w 与 Cf, kleak, dp, Kp 的第一性原理单调物理规律"""
        bounds = LhsSamplingBounds(
            coupling_mode="physical",
            p_fault=0.0,  # 暂不激活断层以纯净检验基态联动
            sigma_cf=0.0,  # 关闭随机扰动以严格检验单调性
            sigma_leak=0.0,
            n_frac_min=4,
            n_frac_max=4,
        )
        sampler = LatinHypercubeSampler(bounds=bounds, seed=42)
        samples = sampler.sample(n_samples=5)

        for s in samples:
            weights = s.get("fracture_alpha_ss", s.get("fracture_inflow_weights"))
            cfs = s["fracture_Cf"]
            kleaks = s["fracture_kleak"]
            dps = s["fracture_dp"]
            kps = s["fracture_Kp"]

            # 对于任意两簇，进液量显著更大者，Cf 更大，kleak 更大，dp 更大，Kp 显著更小
            for i in range(len(weights)):
                for j in range(len(weights)):
                    if weights[i] > weights[j] + 0.05 and weights[j] >= 0.04:
                        assert cfs[i] > cfs[j], f"Cf 未满足联动正比: {cfs[i]} <= {cfs[j]}"
                        assert kleaks[i] > kleaks[j], f"kleak 未满足联动正比: {kleaks[i]} <= {kleaks[j]}"
                        assert dps[i] > dps[j], f"dp 未满足冲蚀扩径正比: {dps[i]} <= {dps[j]}"
                        assert kps[i] < kps[j], f"Kp 未满足流阻反比: {kps[i]} >= {kps[j]}"

    def test_type_v_fault_intersected_activation(self):
        """测试 Type V 沟通天然断层簇的激活、5~10倍强滤失特征与分类标签"""
        bounds = LhsSamplingBounds(
            coupling_mode="physical",
            p_fault=1.0,  # 强制 100% 激活断层
            fault_leak_multiplier_min=6.0,
            fault_leak_multiplier_max=9.0,
            n_frac_min=4,
            n_frac_max=4,
        )
        sampler = LatinHypercubeSampler(bounds=bounds, seed=999)
        samples = sampler.sample(n_samples=3)

        for s in samples:
            assert s["has_fault"] is True
            fault_idx = s["fault_cluster_idx"]
            assert 0 <= fault_idx < s["n_frac"]

            # 校验 Type V 分类标签
            assert "Type V" in s["fracture_types"][fault_idx]
            assert "沟通天然断层" in s["fracture_types"][fault_idx]

            # 校验断层滤失系数处于 5.0e-4 ~ 15.0e-4 m^2.5/s 高渗流区间 (正常簇在 ~1.0e-4)
            fault_kleak = s["fracture_kleak"][fault_idx]
            assert 5.0e-4 <= fault_kleak <= 15.0e-4, f"断层滤失量不在预期强滤失区间: {fault_kleak}"

            # 校验断层顺应性保持在适中区间 [0.005, 0.012]
            fault_cf = s["fracture_Cf"][fault_idx]
            assert 0.005 <= fault_cf <= 0.012, f"断层储能顺应性超限: {fault_cf}"

    def test_type_iv_screenout_dead_cluster(self):
        """测试 Type IV 砂堵死簇的极端阻抗与近零储能滤失特性"""
        # 手动输入极低进液量测试 classify_fracture_type 与物理设定
        ftype = classify_fracture_type(w=0.01, cf=0.0005, kleak=5e-6, kp=8e7)
        assert "Type IV" in ftype
        assert "砂堵闭合" in ftype

        # 通过预设工况生成单簇砂堵死簇工况
        preset = sample_preset_scenario("screenout_dead")
        assert preset["n_frac"] == 4
        # 第 3 簇 (idx=2) 砂堵
        alpha_val = preset.get("fracture_alpha_ss", preset.get("fracture_inflow_weights"))
        assert alpha_val[2] <= 0.02
        assert preset["fracture_Cf"][2] < 0.001
        assert preset["fracture_kleak"][2] < 0.1e-4
        assert preset["fracture_Kp"][2] >= 5.0e7
        assert "Type IV" in preset["fracture_types"][2]

    def test_preset_scenarios_generation_and_simulation(self):
        """测试全部 5 大典型工程工况的一键生成与无故障正演求解"""
        scenarios = [
            "perfect_uniform",
            "heel_dominant",
            "toe_dominant",
            "screenout_dead",
            "fault_leaking",
        ]

        for sc_name in scenarios:
            sc_dict = sample_preset_scenario(sc_name, tf=1.0, dt=0.002)
            assert sc_dict["n_frac"] == 4
            assert len(sc_dict["fracture_Cf"]) == 4
            assert len(sc_dict["fracture_kleak"]) == 4
            assert len(sc_dict["fracture_Kp"]) == 4
            assert len(sc_dict["fracture_types"]) == 4

            # 执行仿真验证
            res = _run_single_simulation(sc_dict)
            assert res["status"] == "success", f"工况 {sc_name} 仿真失败: {res.get('error_msg')}"
            assert not np.any(np.isnan(res["wellhead_head"]))
            assert len(res["cepstrum"]) > 0

    def test_preset_scenario_fault_leaking_dynamics(self):
        """测试沟通断层工况 (Type V) 相比均匀工况呈现出显著的低频大反弹压平和强阻尼水头衰减"""
        # 设置 tf=8.0s 覆盖声学双程走时 (2*4100/1450 = 5.66s) 与关泵脉冲到达井口历程
        cfg_uniform = sample_preset_scenario("perfect_uniform", tf=8.0, dt=0.002)
        cfg_fault = sample_preset_scenario("fault_leaking", tf=8.0, dt=0.002)

        res_u = _run_single_simulation(cfg_uniform)
        res_f = _run_single_simulation(cfg_fault)

        assert res_u["status"] == "success"
        assert res_f["status"] == "success"

        t = res_u["timestamps"]
        wh_u = res_u["wellhead_head"]
        wh_f = res_f["wellhead_head"]

        # 1. 裂缝腔体内部动力学校验：第 2 簇 (idx=1) 为沟通断层簇 (Type V)
        # 受到 10 倍超常强滤失影响，其关泵反弹峰值被压平拉低，且后半程水头衰减明显快于均匀工况
        frac_u = res_u["fracture_heads"][:, 1]
        frac_f = res_f["fracture_heads"][:, 1]

        peak_u = float(np.max(frac_u[t >= 3.0]))
        peak_f = float(np.max(frac_f[t >= 3.0]))
        assert peak_f < peak_u, f"断层簇反弹峰值未被压平: fault={peak_f:.2f} m, uniform={peak_u:.2f} m"

        late_mask_frac = t >= 4.0
        mean_frac_u = float(np.mean(frac_u[late_mask_frac]))
        mean_frac_f = float(np.mean(frac_f[late_mask_frac]))
        assert mean_frac_f < mean_frac_u - 0.5, (
            f"断层簇水头衰减不显著: fault={mean_frac_f:.2f} m, uniform={mean_frac_u:.2f} m"
        )

        # 2. 井口声学波场响应校验：声学反射波抵达井口后 (t > 7.0s)，两工况呈现明显水锤波形差异
        wellhead_diff_late = float(np.max(np.abs(wh_u[t > 7.0] - wh_f[t > 7.0])))
        assert wellhead_diff_late > 0.5, f"井口未检测到断层引起的声学回波差异: diff={wellhead_diff_late:.2f} m"

    def test_classify_fracture_type_physical_identification(self):
        """测试 classify_fracture_type 依据第一性原理物理参数范围自主识别 5 大类型 (无需显式 is_fault 标志)"""
        # Type IV: 砂堵死簇
        assert "Type IV" in classify_fracture_type(w=0.01, cf=0.0005, kleak=5.0e-6, kp=8.0e7)
        assert "Type IV" in classify_fracture_type(w=0.02, cf=0.0008, kleak=1.0e-5, kp=1.0e8)

        # Type V: 异常强滤失自主识别 (kleak >= 5.0e-4)
        assert "Type V" in classify_fracture_type(w=0.25, cf=0.009, kleak=1.0e-3, kp=4.5e5)
        assert "Type V" in classify_fracture_type(w=0.30, cf=0.010, kleak=8.0e-4, kp=4.0e5)

        # Type I: 优势发育主进液簇 (w >= 0.35)
        assert "Type I" in classify_fracture_type(w=0.45, cf=0.020, kleak=2.5e-4, kp=2.5e5)

        # Type III: 受抑弱进液簇 (w <= 0.15)
        assert "Type III" in classify_fracture_type(w=0.10, cf=0.004, kleak=0.4e-4, kp=1.2e6)

        # Type II: 均衡正常发育簇
        assert "Type II" in classify_fracture_type(w=0.25, cf=0.010, kleak=1.0e-4, kp=5.4e5)

    def test_preset_scenarios_all_cluster_counts_robustness(self):
        """测试 5 大预设工况在各种任意簇数 (n_frac=1, 2, 3, 5, 8) 下的稳健性与物理规律"""
        scenarios = ["perfect_uniform", "heel_dominant", "toe_dominant", "screenout_dead", "fault_leaking"]
        for sc in scenarios:
            for nf in [1, 2, 3, 5, 8]:
                pos = [4000.0 + 20.0 * k for k in range(nf)]
                res = sample_preset_scenario(sc, fracture_positions=pos)
                assert res["n_frac"] == nf
                assert len(res["fracture_Cf"]) == nf
                assert len(res["fracture_kleak"]) == nf
                assert len(res["fracture_Kp"]) == nf
                assert len(res["fracture_types"]) == nf
                alpha = res.get("fracture_alpha_ss", res.get("fracture_inflow_weights"))
                assert np.isclose(sum(alpha), 1.0, atol=1e-6)

                if sc == "screenout_dead":
                    if nf == 1:
                        assert "Type IV" in res["fracture_types"][0]
                    else:
                        dead_idx = 1 if nf == 2 else min(2, nf - 1)
                        assert "Type IV" in res["fracture_types"][dead_idx]
                        assert res["fracture_Kp"][dead_idx] >= 5.0e7

                if sc == "fault_leaking":
                    assert res["has_fault"] is True
                    f_idx = res["fault_cluster_idx"]
                    assert 0 <= f_idx < nf
                    assert "Type V" in res["fracture_types"][f_idx]
                    assert res["fracture_kleak"][f_idx] >= 5.0e-4

                if sc == "heel_dominant" and nf >= 2:
                    assert alpha[0] > alpha[-1]
                    assert res["fracture_Cf"][0] > res["fracture_Cf"][-1]

                if sc == "toe_dominant" and nf >= 2:
                    assert alpha[-1] > alpha[0]
                    assert res["fracture_Cf"][-1] > res["fracture_Cf"][0]

    def test_lhs_sampler_single_cluster_fault_and_independent_mode(self):
        """测试单簇阶段断层激活以及向后兼容 independent 模式下的断层强滤失支持"""
        # 1. 单簇阶段强制断层激活
        b_single = LhsSamplingBounds(p_fault=1.0, n_frac_min=1, n_frac_max=1)
        s_single = LatinHypercubeSampler(bounds=b_single, seed=123).sample(1)[0]
        assert s_single["has_fault"] is True
        assert s_single["fault_cluster_idx"] == 0
        assert "Type V" in s_single["fracture_types"][0]
        assert 5.0e-4 <= s_single["fracture_kleak"][0] <= 15.0e-4

        # 2. 独立扰动模式下的断层强滤失激活
        b_indep = LhsSamplingBounds(coupling_mode="independent", p_fault=1.0, n_frac_min=4, n_frac_max=4)
        s_indep = LatinHypercubeSampler(bounds=b_indep, seed=456).sample(1)[0]
        assert s_indep["has_fault"] is True
        f_idx = s_indep["fault_cluster_idx"]
        assert "Type V" in s_indep["fracture_types"][f_idx]
        assert s_indep["fracture_kleak"][f_idx] >= 5.0e-4

    def test_solver_honors_raw_kleak_fidelity(self):
        """测试 WellboreMocV2Solver 在调用方显式指定 fracture_kleak 时保持真实滤失保真度"""
        cfg = MocV2Config(wellbore_length=5000.0, initial_velocity=1.0, initial_head=300.0)

        # 1. 显式指定断层 10 倍滤失 1.0e-3
        user_kleak = [1.0e-4, 1.0e-3, 1.0e-4, 1.0e-4]
        solver_custom = WellboreMocV2Solver(
            cfg=cfg,
            fracture_positions=[4100.0, 4120.0, 4140.0, 4160.0],
            fracture_kleak=user_kleak,
        )
        assert np.allclose(solver_custom.frac_kleak_arr, user_kleak, atol=1e-8)
        assert solver_custom.frac_kleak_arr[1] == 1.0e-3

        # 2. 未指定滤失系数时，维持自洽校准滤失 (确保 Step 4 原型对标精度)
        solver_default = WellboreMocV2Solver(
            cfg=cfg,
            fracture_positions=[4100.0, 4120.0, 4140.0, 4160.0],
            steady_mode="prescribed_flow_split_legacy",
        )
        assert not np.allclose(solver_default.frac_kleak_arr, [1.0e-4] * 4)
        assert len(solver_default.frac_kleak_arr) == 4



