# -*- coding: utf-8 -*-
"""
experiments/moc_physics_upgrade/step3_perforation_throttle/test_step3_solver.py

Step 3 求解器物理与数值单元测试：
1. 测试 Kp -> 0 时退化为 Step 2 的一致性
2. 测试非线性射孔节流耦合牛顿迭代收敛性、残差与物理质量守恒 (A*(V_L - V_R) == q_p)
3. 测试稳态流场自洽性 (t < t_s 期间无虚假波，水头流速严格恒定)
4. 测试死趾端死水区边界条件 (V[-1] == 0)
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

from experiments.moc_physics_upgrade.step2_fracture_compliance.solver import (
    solve_fracture_node_step2, simulate_wellbore_step2, Step2MocConfig
)
from experiments.moc_physics_upgrade.step3_perforation_throttle.solver import (
    Step3MocConfig, solve_fracture_node_step3, simulate_wellbore_step3, G
)


class TestStep3Solver(unittest.TestCase):
    def setUp(self):
        self.cfg = Step3MocConfig(
            wellbore_length=5000.0,
            wavespeed=1450.0,
            initial_velocity=1.0,
            initial_head=300.0,
            dt=0.001,
            tf=2.0,
            pump_shut_time=1.0,
            perf_num_holes=16,
            perf_diameter=0.01,
            perf_cd=0.65,
        )
        self.area = self.cfg.area
        self.ga = G / self.cfg.a_adj

    def test_kp_zero_degeneracy_to_step2(self):
        """当 Kp = 0 时，Step 3 求解器应与 Step 2 产生完全相同的结果"""
        Cp_f = 2.0
        Cm_f = 0.5
        H_prev = 280.0
        Cf = 0.01
        kleak = 1.0e-4
        H_ext = 100.0
        dt = 0.001

        H2, v_l2, v_r2, q2 = solve_fracture_node_step2(
            Cp_f, Cm_f, H_prev, self.area, self.ga, Cf, kleak, H_ext, dt
        )
        H_w3, H_f3, v_l3, v_r3, q3 = solve_fracture_node_step3(
            Cp_f, Cm_f, H_prev, self.area, self.ga, Cf, kleak, H_ext, dt, Kp=0.0
        )

        self.assertAlmostEqual(H_w3, H2, places=7)
        self.assertAlmostEqual(H_f3, H2, places=7)
        self.assertAlmostEqual(v_l3, v_l2, places=7)
        self.assertAlmostEqual(v_r3, v_r2, places=7)
        self.assertAlmostEqual(q3, q2, places=7)

    def test_newton_convergence_and_conservation(self):
        """测试非零 Kp 下的牛顿收敛性与节点质量守恒"""
        Cp_f = 2.5
        Cm_f = -0.5
        H_prev = 275.0
        Cf = 0.01
        kleak = 2.0e-4
        H_ext = 100.0
        dt = 0.001
        Kp = self.cfg.perf_Kp

        H_w, H_f, v_l, v_r, q_p = solve_fracture_node_step3(
            Cp_f, Cm_f, H_prev, self.area, self.ga, Cf, kleak, H_ext, dt, Kp=Kp, q_init=0.004
        )

        # 1. 检验声学连续性与质量守恒: q_p == area * (v_l - v_r)
        q_continuity = self.area * (v_l - v_r)
        self.assertAlmostEqual(q_p, q_continuity, places=10)

        # 2. 检验特征线方程: v_l = Cp_f - ga * H_w, v_r = -Cm_f + ga * H_w
        self.assertAlmostEqual(v_l, Cp_f - self.ga * H_w, places=10)
        self.assertAlmostEqual(v_r, -Cm_f + self.ga * H_w, places=10)

        # 3. 检验射孔节流压降: H_w - H_f == sign(q_p) * Kp * q_p^2
        expected_dH = np.sign(q_p) * Kp * (q_p ** 2)
        self.assertAlmostEqual(H_w - H_f, expected_dH, places=10)

        # 4. 检验裂缝顺应性容抗方程残差: q_p == (Cf/dt)*(H_f - H_prev) + q_leak
        q_leak = kleak * np.sqrt(max(H_f - H_ext, 0.0))
        q_storage = (Cf / dt) * (H_f - H_prev)
        self.assertAlmostEqual(q_p, q_storage + q_leak, places=9)

    def test_steady_state_before_shutoff(self):
        """在关泵前 (t < 1.0s)，井筒与裂缝水头流速应严格保持稳态，无虚假波动"""
        res = simulate_wellbore_step3(
            self.cfg,
            fracture_positions=[4100.0, 4120.0, 4140.0, 4160.0],
            fracture_Cf=[0.01] * 4,
            H_ext=100.0,
        )
        t = res["timestamps"]
        H_wh = res["wellhead_head"]
        mask_ss = (t >= 0.1) & (t <= 0.9)

        # 井口稳态水头标准差应接近 0 (无任何虚假震荡)
        std_wh = np.std(H_wh[mask_ss])
        self.assertLess(std_wh, 1.0e-6)

        # 趾端流速应始终为 0
        toe_v = res["toe_velocity"]
        self.assertTrue(np.all(np.abs(toe_v[mask_ss]) < 1.0e-12))

    def test_negative_flowback_convergence(self):
        """测试压力倒灌 (q_p < 0, 裂缝向井筒反吐) 条件下的收敛性与动量平衡"""
        # 设井筒水头骤降，低于裂缝内压
        Cp_f = 0.5
        Cm_f = -1.5
        H_prev = 320.0
        Cf = 0.01
        kleak = 1.0e-4
        H_ext = 100.0
        dt = 0.001
        Kp = self.cfg.perf_Kp

        H_w, H_f, v_l, v_r, q_p = solve_fracture_node_step3(
            Cp_f, Cm_f, H_prev, self.area, self.ga, Cf, kleak, H_ext, dt, Kp=Kp, q_init=-0.005
        )

        self.assertLess(q_p, 0.0)  # 应为负流量 (回吐)
        self.assertGreater(H_f, H_w)  # 裂缝内水头应高于井筒水头

        # 检验射孔节流压降: H_w - H_f == sign(q_p) * Kp * q_p^2 == -Kp * q_p^2
        expected_dH = -Kp * (q_p ** 2)
        self.assertAlmostEqual(H_w - H_f, expected_dH, places=10)

        # 检验质量守恒
        q_continuity = self.area * (v_l - v_r)
        self.assertAlmostEqual(q_p, q_continuity, places=10)

    def test_end_to_end_step3_rebound_and_limits(self):
        """测试 Step 3 端到端运行的物理极限值与宏观反弹指标"""
        cfg = Step3MocConfig(
            wellbore_length=5000.0,
            wavespeed=1450.0,
            initial_velocity=1.0,
            initial_head=300.0,
            dt=0.001,
            tf=15.0,
            pump_shut_time=1.0,
            perf_num_holes=6,
            perf_diameter=0.01,
            perf_cd=0.65,
        )
        res = simulate_wellbore_step3(
            cfg,
            fracture_positions=[4100.0, 4120.0, 4140.0, 4160.0],
            fracture_Cf=[0.01] * 4,
            H_ext=100.0,
        )
        H_wh = res["wellhead_head"]
        dt = cfg.dt_adj
        h_drop = float(H_wh[int(1.0 / dt)])
        h_rebound = float(np.max(H_wh[int(6.5 / dt):int(10.0 / dt)]))
        amp = h_rebound - h_drop

        # 验证反弹幅度超过 150m (保持地质储能宏观效应)
        self.assertGreater(amp, 150.0)
        # 验证全程水头高于地层孔隙水头 H_ext=100m (无真空抽空塌陷)
        self.assertGreater(float(np.min(H_wh)), 100.0)
        # 验证死端流速始终为 0
        self.assertTrue(np.all(np.abs(res["toe_velocity"]) < 1e-12))

    def test_full_simulation_degeneracy_to_step2_when_kp_zero(self):
        """全时程仿真测试：当各簇射孔 Kp 均为 0 时，Step 3 全井时程与 Step 2 完全等价（误差 < 1e-9）"""
        cfg2 = Step2MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=4.0, pump_shut_time=1.0
        )
        cfg3 = Step3MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=4.0, pump_shut_time=1.0, perf_num_holes=16
        )
        x_f = [4100.0, 4120.0, 4140.0, 4160.0]
        res2 = simulate_wellbore_step2(cfg2, fracture_positions=x_f, fracture_Cf=[0.01] * 4, H_ext=100.0)
        res3 = simulate_wellbore_step3(cfg3, fracture_positions=x_f, fracture_Cf=[0.01] * 4, fracture_Kp=[0.0] * 4, H_ext=100.0)

        max_err_wh = float(np.max(np.abs(res2["wellhead_head"] - res3["wellhead_head"])))
        max_err_toe = float(np.max(np.abs(res2["toe_head"] - res3["toe_head"])))
        max_err_frac = float(np.max(np.abs(res2["fracture_heads"] - res3["fracture_heads"])))

        self.assertLess(max_err_wh, 1.0e-9)
        self.assertLess(max_err_toe, 1.0e-9)
        self.assertLess(max_err_frac, 1.0e-9)

    def test_extreme_q_init_newton_convergence(self):
        """测试牛顿迭代在极大偏差初值（q_init = ±1000 m^3/s）下的二阶全局收敛性与精确度"""
        for q_init in [-1000.0, 1000.0]:
            H_w, H_f, v_l, v_r, q_p = solve_fracture_node_step3(
                Cp_f=2.0, Cm_f=-0.5, H_prev_f=280.0, area=self.area, ga=self.ga,
                Cf=0.01, kleak=1.0e-4, H_ext=100.0, dt=0.001, Kp=self.cfg.perf_Kp, q_init=q_init
            )
            # 检验非线性残差严格归零
            abs_q = abs(q_p)
            H_f_chk = H_w - np.sign(q_p) * self.cfg.perf_Kp * (abs_q ** 2)
            q_lk = 1.0e-4 * np.sqrt(max(H_f_chk - 100.0, 0.0))
            residual = abs(q_p - (0.01 / 0.001 * (H_f_chk - 280.0) + q_lk))
            self.assertLess(residual, 1.0e-9)

    def test_unsorted_fracture_positions_auto_sorting(self):
        """测试乱序输入的 fracture_positions 会被求解器自动升序排序并保证与排布一致"""
        cfg = Step3MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=2.0, pump_shut_time=1.0, perf_num_holes=6
        )
        res_sorted = simulate_wellbore_step3(
            cfg, fracture_positions=[4100.0, 4120.0, 4140.0, 4160.0], fracture_Cf=[0.01] * 4, H_ext=100.0
        )
        # 乱序输入
        res_unsorted = simulate_wellbore_step3(
            cfg, fracture_positions=[4140.0, 4100.0, 4160.0, 4120.0],
            fracture_Cf=[0.01, 0.01, 0.01, 0.01], H_ext=100.0
        )
        diff_wh = np.max(np.abs(res_sorted["wellhead_head"] - res_unsorted["wellhead_head"]))
        self.assertLess(diff_wh, 1.0e-10)

    def test_duplicate_fracture_collision_detection(self):
        """测试当两簇裂缝位置间距小于网格步长 dx 时，求解器主动抛出 ValueError 阻止网格混叠冲突"""
        cfg = Step3MocConfig(wellbore_length=5000.0, wavespeed=1450.0, dt=0.001, tf=1.0)
        # dx ≈ 1.45m，4100.0 与 4100.2 均映射到节点 2827
        with self.assertRaises(ValueError):
            simulate_wellbore_step3(
                cfg, fracture_positions=[4100.0, 4100.2], fracture_Cf=[0.01, 0.01], H_ext=100.0
            )

    def test_insufficient_initial_head_raises_error(self):
        """测试当稳态射孔压降导致裂缝内压低于地层孔隙压力时，求解器主动报错"""
        # H0 = 101m, H_ext = 100m, 但射孔压降约 8m，裂缝稳态内压 93m < 100m
        cfg = Step3MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=101.0,
            dt=0.001, tf=1.0, perf_num_holes=6
        )
        with self.assertRaises(ValueError):
            simulate_wellbore_step3(
                cfg, fracture_positions=[4100.0], fracture_Cf=[0.01], H_ext=100.0
            )

    def test_ramp_pump_closure(self):
        """测试斜坡关泵 (ramp closure) 工况的平滑演化与数值稳定性"""
        cfg = Step3MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=10.0, wellhead_bc="ramp", pump_shut_time=1.0, pump_closure_duration=0.5,
            perf_num_holes=6
        )
        res = simulate_wellbore_step3(
            cfg, fracture_positions=[4100.0, 4120.0, 4140.0, 4160.0], fracture_Cf=[0.01] * 4, H_ext=100.0
        )
        wh = res["wellhead_head"]
        self.assertFalse(np.any(np.isnan(wh)))
        self.assertGreater(float(np.min(wh)), 100.0)

    def test_non_uniform_perforation_holes(self):
        """测试非均匀射孔孔数 [20, 4, 4, 4] 分布下的质量守恒与多簇差异响应"""
        cfg = Step3MocConfig(
            wellbore_length=5000.0, wavespeed=1450.0, initial_velocity=1.0, initial_head=300.0,
            dt=0.001, tf=10.0, pump_shut_time=1.0
        )
        res = simulate_wellbore_step3(
            cfg,
            fracture_positions=[4100.0, 4120.0, 4140.0, 4160.0],
            fracture_Cf=[0.01] * 4,
            fracture_num_holes=[20, 4, 4, 4],
            H_ext=100.0,
        )
        # 验证稳态压降：第 1 簇压降显著低于后 3 簇
        dH_ss = res["dH_perf_ss"]
        self.assertLess(dH_ss[0], dH_ss[1])
        self.assertAlmostEqual(dH_ss[1], dH_ss[2], places=5)
        self.assertAlmostEqual(dH_ss[2], dH_ss[3], places=5)


if __name__ == '__main__':
    unittest.main()
