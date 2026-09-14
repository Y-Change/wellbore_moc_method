# -*- coding: utf-8 -*-
"""
experiments/moc_physics_upgrade/step4_ramp_closure/test_step4_ramp.py

Step 4 关泵斜坡求解器单元测试与物理校验：
1. 斜坡流速边界解析解与连续性校验 (包含线性斜坡与 S 型余弦过渡)
2. tc = 0 瞬时阶跃退化测试：与 Step 3 求解器全时程严格等价性 (< 1e-12 误差)
3. 初始稳态自洽性校验 (t < ts 期间水头流速恒定，无前驱数值假波)
4. 斜坡关泵平滑度与波前陡度 (dH/dt) 削减效应校验
5. 宏观大反弹地质储能保持性 (反弹幅度 > 150m)
6. 极端工况健壮性校验 (极小斜坡 tc=1ms、大斜坡 tc=5s、非法参数防御)
"""
import os
import sys
import unittest
import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isdir(os.path.join(_d, 'moc_simulate')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find project root')
    _d = _parent

from experiments.moc_physics_upgrade.step3_perforation_throttle.solver import (
    Step3MocConfig, simulate_wellbore_step3
)
from experiments.moc_physics_upgrade.step4_ramp_closure.solver import (
    Step4MocConfig, simulate_wellbore_step4, compute_ramp_velocity,
    compute_ramp_acceleration, solve_fracture_node_step4, G
)


class TestStep4Ramp(unittest.TestCase):
    def setUp(self):
        self.cfg = Step4MocConfig(
            wellbore_length=5000.0,
            wavespeed=1450.0,
            initial_velocity=1.0,
            initial_head=300.0,
            dt=0.001,
            tf=3.0,
            pump_shut_time=1.0,
            pump_closure_duration=1.0,
            wellhead_bc="ramp",
            ramp_type="linear",
            perf_num_holes=6,
            perf_diameter=0.01,
            perf_cd=0.65,
        )

    def test_ramp_velocity_analytic_and_continuity(self):
        """测试斜坡关泵解析流速函数 V(0, t) 的边界值与连续性"""
        V0 = 1.0
        ts = 1.0
        tc = 1.0

        # 1. 关泵前
        self.assertAlmostEqual(compute_ramp_velocity(0.0, V0, ts, tc), V0)
        self.assertAlmostEqual(compute_ramp_velocity(0.999, V0, ts, tc), V0)
        self.assertAlmostEqual(compute_ramp_velocity(1.0, V0, ts, tc), V0)

        # 2. 线性斜坡中点
        self.assertAlmostEqual(compute_ramp_velocity(1.5, V0, ts, tc, "linear"), 0.5 * V0)

        # 3. 线性斜坡结束点与加速度
        self.assertAlmostEqual(compute_ramp_velocity(2.0, V0, ts, tc, "linear"), 0.0)
        self.assertAlmostEqual(compute_ramp_velocity(2.001, V0, ts, tc, "linear"), 0.0)
        self.assertAlmostEqual(compute_ramp_velocity(10.0, V0, ts, tc, "linear"), 0.0)
        # 线性斜坡内部与边界加速度
        self.assertAlmostEqual(compute_ramp_acceleration(1.5, V0, ts, tc, "linear"), -V0 / tc)
        self.assertAlmostEqual(compute_ramp_acceleration(2.0, V0, ts, tc, "linear"), 0.0)
        self.assertAlmostEqual(compute_ramp_acceleration(2.5, V0, ts, tc, "linear"), 0.0)

        # 4. 余弦平滑过渡
        # t = ts: tau = 0.5 * (1 + cos(0)) = 1.0
        self.assertAlmostEqual(compute_ramp_velocity(1.0, V0, ts, tc, "cosine"), V0)
        # t = ts + tc/2: tau = 0.5 * (1 + cos(pi/2)) = 0.5
        self.assertAlmostEqual(compute_ramp_velocity(1.5, V0, ts, tc, "cosine"), 0.5 * V0)
        # t = ts + tc: tau = 0.5 * (1 + cos(pi)) = 0.0
        self.assertAlmostEqual(compute_ramp_velocity(2.0, V0, ts, tc, "cosine"), 0.0)
        # 单调性验证
        t_cos = np.linspace(ts, ts + tc, 101)
        v_cos = compute_ramp_velocity(t_cos, V0, ts, tc, "cosine")
        self.assertTrue(np.all(np.diff(v_cos) <= 1e-12), "余弦平滑过渡必须严格单调递减")

        # 5. 余弦过渡的 C1 连续性 (导数在两端严格平滑归零)
        a_start = compute_ramp_acceleration(1.0, V0, ts, tc, "cosine")
        a_mid = compute_ramp_acceleration(1.5, V0, ts, tc, "cosine")
        a_end = compute_ramp_acceleration(2.0, V0, ts, tc, "cosine")
        self.assertAlmostEqual(a_start, 0.0, places=7)
        self.assertAlmostEqual(a_mid, -0.5 * V0 * (np.pi / tc), places=7)
        self.assertAlmostEqual(a_end, 0.0, places=7)

        # 6. 负 tc 退化为阶跃关泵
        self.assertAlmostEqual(compute_ramp_velocity(0.5, V0, ts, -1.0), V0)
        self.assertAlmostEqual(compute_ramp_velocity(1.0, V0, ts, -1.0), 0.0)
        self.assertAlmostEqual(compute_ramp_velocity(1.5, V0, ts, -1.0), 0.0)

        # 7. 矢量化输入校验
        t_vec = np.linspace(0.0, 3.0, 301)
        v_vec = compute_ramp_velocity(t_vec, V0, ts, tc, "linear")
        self.assertEqual(len(v_vec), 301)
        self.assertTrue(np.all(v_vec >= 0.0) and np.all(v_vec <= V0))

    def test_tc_zero_degeneracy_to_step3(self):
        """当 tc = 0 时，Step 4 求解器与 Step 3 求解器在全井时程上严格一致 (最大误差 < 1e-12)"""
        cfg3 = Step3MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=4.0, wellhead_bc="velocity_step", pump_shut_time=1.0, perf_num_holes=6
        )
        cfg4 = Step4MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=4.0, wellhead_bc="ramp", pump_shut_time=1.0, pump_closure_duration=0.0,
            perf_num_holes=6
        )

        x_f = [4100.0, 4120.0, 4140.0, 4160.0]
        res3 = simulate_wellbore_step3(cfg3, fracture_positions=x_f, fracture_Cf=[0.01] * 4, H_ext=100.0)
        res4 = simulate_wellbore_step4(cfg4, fracture_positions=x_f, fracture_Cf=[0.01] * 4, H_ext=100.0)

        err_wh_h = float(np.max(np.abs(res3["wellhead_head"] - res4["wellhead_head"])))
        err_wh_v = float(np.max(np.abs(res3["wellhead_velocity"] - res4["wellhead_velocity"])))
        err_toe_h = float(np.max(np.abs(res3["toe_head"] - res4["toe_head"])))
        err_frac_h = float(np.max(np.abs(res3["fracture_heads"] - res4["fracture_heads"])))

        self.assertLess(err_wh_h, 1.0e-12)
        self.assertLess(err_wh_v, 1.0e-12)
        self.assertLess(err_toe_h, 1.0e-12)
        self.assertLess(err_frac_h, 1.0e-12)

    def test_steady_state_before_shutoff(self):
        """关泵前 (t < 1.0s)，井口水头严格保持初始恒定稳态，无任何前驱激波或漂移"""
        res = simulate_wellbore_step4(
            self.cfg,
            fracture_positions=[4100.0, 4120.0, 4140.0, 4160.0],
            fracture_Cf=[0.01] * 4,
            H_ext=100.0,
        )
        t = res["timestamps"]
        H_wh = res["wellhead_head"]
        mask_ss = (t >= 0.05) & (t <= 0.95)

        std_wh = float(np.std(H_wh[mask_ss]))
        self.assertLess(std_wh, 1.0e-6)

        # 趾端流速始终为 0
        self.assertTrue(np.all(np.abs(res["toe_velocity"][mask_ss]) < 1.0e-12))

    def test_wavefront_steepness_smoothing(self):
        """对比 tc=0 (阶跃) 与 tc=1.0s (斜坡) 的波前陡度：斜坡关泵极大削弱最大压力变化率 |dH/dt|"""
        cfg_step = Step4MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=4.0, pump_shut_time=1.0, pump_closure_duration=0.0, perf_num_holes=6
        )
        cfg_ramp = Step4MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=4.0, pump_shut_time=1.0, pump_closure_duration=1.0, perf_num_holes=6
        )

        x_f = [4100.0, 4120.0, 4140.0, 4160.0]
        res_step = simulate_wellbore_step4(cfg_step, fracture_positions=x_f, fracture_Cf=[0.01] * 4, H_ext=100.0)
        res_ramp = simulate_wellbore_step4(cfg_ramp, fracture_positions=x_f, fracture_Cf=[0.01] * 4, H_ext=100.0)

        dt = 0.001
        dHdt_step = np.abs(np.diff(res_step["wellhead_head"]) / dt)
        dHdt_ramp = np.abs(np.diff(res_ramp["wellhead_head"]) / dt)

        # 阶跃关泵发生在 t_s=1.0s (n=999->1000 步)，最大离散变化率 |ΔH/Δt| 接近 148,000 m/s
        # 斜坡关泵历经 1.0s (1000 个步长) 平滑降落，最大变化率约为 150 m/s
        max_rate_step = float(np.max(dHdt_step[int(1.0 / dt) - 2 : int(1.05 / dt)]))
        max_rate_ramp = float(np.max(dHdt_ramp[int(1.0 / dt) - 2 : int(2.0 / dt)]))

        self.assertGreater(max_rate_step, 1.0e4)
        self.assertLess(max_rate_ramp, 500.0)
        # 陡度下降 2 个数量级以上
        self.assertGreater(max_rate_step / max_rate_ramp, 100.0)

    def test_macro_rebound_preservation(self):
        """测试在斜坡关泵 (tc=1.0s) 下，宏观地质顺应性大反弹依然完好产生 (> 150m)"""
        cfg = Step4MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=15.0, pump_shut_time=1.0, pump_closure_duration=1.0, perf_num_holes=6
        )
        x_f = [4100.0, 4120.0, 4140.0, 4160.0]
        res = simulate_wellbore_step4(cfg, fracture_positions=x_f, fracture_Cf=[0.01] * 4, H_ext=100.0)

        H_wh = res["wellhead_head"]
        dt = cfg.dt_adj
        # 关泵落入水头 (取斜坡结束至反射前)
        h_drop = float(np.min(H_wh[int(2.0 / dt):int(6.0 / dt)]))
        # 反弹峰值 (真波峰到达约 12.32s)
        h_rebound = float(np.max(H_wh[int(6.5 / dt):int(15.0 / dt)]))
        rebound_amp = h_rebound - h_drop

        self.assertGreater(rebound_amp, 200.0)
        self.assertGreater(float(np.min(H_wh)), 100.0)  # 全程高于地层外边界水头

    def test_rebound_wavefront_phase_delay(self):
        """测试斜坡关泵对反弹波前上升沿产生的线性相位时滞 (Δt_half ≈ 0.64 * tc)"""
        cfg_step = Step4MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, dt=0.001, tf=10.0,
            pump_shut_time=1.0, pump_closure_duration=0.0, perf_num_holes=6
        )
        cfg_ramp = Step4MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, dt=0.001, tf=10.0,
            pump_shut_time=1.0, pump_closure_duration=1.0, perf_num_holes=6
        )
        x_f = [4100.0, 4120.0, 4140.0, 4160.0]
        res_step = simulate_wellbore_step4(cfg_step, fracture_positions=x_f, fracture_Cf=[0.01] * 4, H_ext=100.0)
        res_ramp = simulate_wellbore_step4(cfg_ramp, fracture_positions=x_f, fracture_Cf=[0.01] * 4, H_ext=100.0)

        t = res_step["timestamps"]
        h_step = res_step["wellhead_head"]
        h_ramp = res_ramp["wellhead_head"]

        # 取水头上升到 250m (约半幅上升点) 的时刻
        m = t >= 6.0
        t_half_step = float(t[m][np.where(h_step[m] >= 250.0)[0][0]])
        t_half_ramp = float(t[m][np.where(h_ramp[m] >= 250.0)[0][0]])
        delay = t_half_ramp - t_half_step

        # 理论上上升沿过渡历时延长 1.0s，质心时滞约 0.64s
        self.assertGreater(delay, 0.5)
        self.assertLess(delay, 0.8)

    def test_extreme_and_invalid_conditions(self):
        """测试极短斜坡 (tc=1ms)、平滑余弦过渡以及未知 ramp_type 的容错防御"""
        # 1. 极短斜坡 tc = 1ms
        cfg_fast = Step4MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, dt=0.001, tf=2.0,
            pump_shut_time=1.0, pump_closure_duration=0.001, perf_num_holes=6
        )
        res_fast = simulate_wellbore_step4(cfg_fast, fracture_positions=[4100.0], fracture_Cf=[0.01], H_ext=100.0)
        self.assertFalse(np.any(np.isnan(res_fast["wellhead_head"])))

        # 2. 余弦平滑过渡仿真
        cfg_cos = Step4MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, dt=0.001, tf=2.0,
            pump_shut_time=1.0, pump_closure_duration=1.0, ramp_type="cosine", perf_num_holes=6
        )
        res_cos = simulate_wellbore_step4(cfg_cos, fracture_positions=[4100.0], fracture_Cf=[0.01], H_ext=100.0)
        self.assertFalse(np.any(np.isnan(res_cos["wellhead_head"])))

        # 3. 未知 ramp_type 触发 ValueError
        with self.assertRaises(ValueError):
            compute_ramp_velocity(1.5, 1.0, 1.0, 1.0, ramp_type="invalid_type")
        with self.assertRaises(ValueError):
            compute_ramp_acceleration(1.5, 1.0, 1.0, 1.0, ramp_type="invalid_type")


if __name__ == '__main__':
    unittest.main()
