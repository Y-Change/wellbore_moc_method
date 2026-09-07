# -*- coding: utf-8 -*-
"""
tests/test_fracture_storage_physics.py
单缝裂缝节点储液项数值格式与物理相容性验证套件。

验证目标：
1. 局部质量守恒检验：∑(Q_f - Q_leak)Δt 是否严格等于 C_H * (H_f_end - H_f_start)
2. 无摩阻、无滤失条件下的解析阻抗与瞬态反射/透射解对照：
   理论松弛时间常数 tau = a * C_H / (2 * g * A)
   反射波解析解: h_r(t) = -ΔH * exp(-t / tau)
   透射波解析解: h_t(t) = ΔH * (1 - exp(-t / tau))
3. 比较两种离散格式：
   - scheme 'local': H_old = H_prev[i_f] (局部时间后向 Euler，严格望远镜可积)
   - scheme 'spatial_avg': H_old = 0.5 * (H_prev[i_f-1] + H_prev[i_f+1]) (空间平均)
4. 网格步长收敛性：dt = 2.0ms, 1.0ms, 0.5ms 下与解析解的 L2 误差单调递减。
"""

import numpy as np
import pytest
from typing import Dict, Tuple

from moc_simulate.wellbore_moc import solve_fracture_node


def run_single_fracture_step_wave(
    scheme: str = "local",
    dt: float = 1.0e-3,
    Cf: float = 1.0e-5,
    kleak: float = 0.0,
    tf: float = 2.5,
    L: float = 2000.0,
    xf: float = 1000.0,
    a: float = 1000.0,
    D: float = 0.1,
    V0: float = 1.0,
    H0: float = 200.0,
) -> Dict[str, np.ndarray]:
    """
    轻量单缝 MOC 求解器（无摩阻，支持自定义裂缝储液项格式）。
    用于严格检验局部反射与质量守恒。
    """
    g = 9.80665
    ga = g / a
    area = np.pi * 0.25 * D * D
    dx = a * dt
    N = int(round(L / dx))
    dx = L / N
    a_eff = dx / dt
    ga = g / a_eff

    n_steps = int(round(tf / dt))
    i_f = int(round(xf / dx))

    H_grid = np.full(N + 1, H0, dtype=np.float64)
    V_left = np.full(N + 1, V0, dtype=np.float64)
    V_right = np.full(N + 1, V0, dtype=np.float64)

    t_hist = np.zeros(n_steps + 1)
    H_wh_hist = np.zeros(n_steps + 1)
    H_frac_hist = np.zeros(n_steps + 1)
    Q_frac_hist = np.zeros(n_steps + 1)
    V_frac_l_hist = np.zeros(n_steps + 1)
    V_frac_r_hist = np.zeros(n_steps + 1)

    t_hist[0] = 0.0
    H_wh_hist[0] = H0
    H_frac_hist[0] = H0

    ts = 0.2
    H_prev = H_grid.copy()
    V_prev_l = V_left.copy()
    V_prev_r = V_right.copy()

    H_new = np.empty(N + 1, dtype=np.float64)
    V_new_l = np.empty(N + 1, dtype=np.float64)
    V_new_r = np.empty(N + 1, dtype=np.float64)

    for n in range(1, n_steps + 1):
        t = n * dt
        t_hist[n] = t

        Cp = V_prev_r[:-2] + ga * H_prev[:-2]
        Cm = -V_prev_l[2:] + ga * H_prev[2:]

        H_new[1:-1] = (Cp + Cm) / (2.0 * ga)
        V_new_l[1:-1] = Cp - ga * H_new[1:-1]
        V_new_r[1:-1] = V_new_l[1:-1]

        Cp_f = Cp[i_f - 1]
        Cm_f = Cm[i_f - 1]

        if scheme == "local":
            H_old = H_prev[i_f]
        elif scheme == "spatial_avg":
            H_old = 0.5 * (H_prev[i_f - 1] + H_prev[i_f + 1])
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        H_w, H_f, vl, vr, Qf = solve_fracture_node(
            Cp_f, Cm_f, H_old, area, ga,
            Cf, kleak, H_ext=0.0, dt=dt
        )
        H_new[i_f] = H_w
        V_new_l[i_f] = vl
        V_new_r[i_f] = vr

        V_wh = 0.0 if t >= ts else V0
        Cm_0 = -V_prev_l[1] + ga * H_prev[1]
        H_new[0] = (V_wh + Cm_0) / ga
        V_new_l[0] = V_wh
        V_new_r[0] = V_wh

        H_new[N] = H_prev[N - 1]
        V_new_l[N] = V_prev_r[N - 1]
        V_new_r[N] = V_new_l[N]

        H_wh_hist[n] = H_new[0]
        H_frac_hist[n] = H_f
        Q_frac_hist[n] = Qf
        V_frac_l_hist[n] = vl
        V_frac_r_hist[n] = vr

        H_prev[:] = H_new
        V_prev_l[:] = V_new_l
        V_prev_r[:] = V_new_r

    return {
        "t": t_hist,
        "H_wh": H_wh_hist,
        "H_frac": H_frac_hist,
        "Q_frac": Q_frac_hist,
        "V_frac_l": V_frac_l_hist,
        "V_frac_r": V_frac_r_hist,
        "tau": (a_eff * Cf) / (2.0 * g * area),
        "dt": dt,
        "ts": ts,
        "xf": xf,
        "a": a_eff,
        "g": g,
        "area": area,
        "delta_H_jouk": a_eff * V0 / g,
    }


def test_mass_conservation_comparison():
    """
    量化对比 local (H_prev[i]) 与 spatial_avg (0.5*(H[i-1]+H[i+1])) 的质量守恒性。
    """
    dt = 1.0e-3
    Cf = 1.0e-5

    res_local = run_single_fracture_step_wave(scheme="local", dt=dt, Cf=Cf)
    res_avg = run_single_fracture_step_wave(scheme="spatial_avg", dt=dt, Cf=Cf)

    # 1. Local 格式
    vol_in_local = np.sum(res_local["Q_frac"][1:]) * dt
    vol_theory_local = Cf * (res_local["H_frac"][-1] - res_local["H_frac"][0])
    err_local = abs(vol_in_local - vol_theory_local)
    rel_err_local = err_local / abs(vol_theory_local)

    # 2. Spatial Average 格式
    vol_in_avg = np.sum(res_avg["Q_frac"][1:]) * dt
    vol_theory_avg = Cf * (res_avg["H_frac"][-1] - res_avg["H_frac"][0])
    err_avg = abs(vol_in_avg - vol_theory_avg)
    rel_err_avg = err_avg / abs(vol_theory_avg)

    print("\n--- 裂缝储液质量守恒性量化对比 ---")
    print(f"Local scheme (H_f^n):       ΔV_in = {vol_in_local:.8e}, ΔV_storage = {vol_theory_local:.8e}, 相对误差 = {rel_err_local:.4e}")
    print(f"Spatial avg scheme (H_avg): ΔV_in = {vol_in_avg:.8e}, ΔV_storage = {vol_theory_avg:.8e}, 相对误差 = {rel_err_avg:.4e}")

    # Local 格式具有严格的望远镜伸缩性，误差达到机器精度级别 (< 1e-12)
    assert rel_err_local < 1.0e-12, f"Local scheme 必须严格保质量，当前误差 {rel_err_local}"
    # Spatial average 格式由于非望远镜伸缩，误差显著高于机器精度多个数量级
    assert rel_err_avg > 1.0e-6, f"Spatial avg scheme 存在理论上的质量误差"


def test_analytical_step_response():
    """
    检验单缝局部水头演化与解析解 h_t(t) = H0 - ΔH * (1 - exp(-t / tau)) 的吻合度。
    """
    dt = 1.0e-3
    Cf = 1.0e-5
    res = run_single_fracture_step_wave(scheme="local", dt=dt, Cf=Cf)

    t = res["t"]
    H_frac = res["H_frac"]
    tau = res["tau"]
    a = res["a"]
    xf = res["xf"]
    ts = res["ts"]
    delta_H = res["delta_H_jouk"]
    H0 = H_frac[0]

    t_arr = ts + xf / a
    idx_arr = int(round(t_arr / dt))
    k_eval = int(round(min(3.0 * tau, 0.5 * (xf / a)) / dt))

    t_sub = t[idx_arr:idx_arr + k_eval] - t_arr
    H_num = H_frac[idx_arr:idx_arr + k_eval]
    # 解析透射减量（停泵减速引发负压波）
    H_exact = H0 - delta_H * (1.0 - np.exp(-t_sub / tau))

    # 计算相对 RMSE
    rmse = np.sqrt(np.mean((H_num - H_exact) ** 2)) / delta_H
    print(f"\nLocal scheme 解析透射吻合度: 相对 RMSE = {rmse:.4e} (tau = {tau*1e3:.2f} ms)")
    assert rmse < 0.02, f"透射波响应应与理论解析解高度吻合 (RMSE < 2%), 当前 {rmse}"


def test_dt_convergence():
    """
    步长减半收敛性检验：随着 dt 从 2ms 缩减到 1ms 到 0.5ms，数值解单调一阶收敛。
    """
    dt_list = [2.0e-3, 1.0e-3, 0.5e-3, 0.25e-3]
    errors = []

    for dt in dt_list:
        res = run_single_fracture_step_wave(scheme="local", dt=dt, Cf=1.0e-5)
        t = res["t"]
        H_frac = res["H_frac"]
        tau = res["tau"]
        a = res["a"]
        xf = res["xf"]
        ts = res["ts"]
        delta_H = res["delta_H_jouk"]
        H0 = H_frac[0]

        t_arr = ts + xf / a
        idx_arr = int(round(t_arr / dt))
        k_eval = int(round(0.05 / dt))

        t_sub = t[idx_arr:idx_arr + k_eval] - t_arr
        H_num = H_frac[idx_arr:idx_arr + k_eval]
        H_exact = H0 - delta_H * (1.0 - np.exp(-t_sub / tau))

        l2_err = np.sqrt(np.mean((H_num - H_exact) ** 2))
        errors.append(l2_err)

    print("\n步长收敛阶测试:")
    for dt, err in zip(dt_list, errors):
        print(f"dt = {dt*1e3:.2f} ms: L2 误差 = {err:.6f} m")

    for i in range(len(errors) - 1):
        assert errors[i] > errors[i+1], "误差随时间步长减小必须单调下降！"
    rate = np.log(errors[0] / errors[-1]) / np.log(dt_list[0] / dt_list[-1])
    print(f"实测收敛阶: {rate:.2f}")
    assert rate > 0.9, f"后向 Euler 理论一阶收敛，实测收敛阶应 > 0.9，当前 {rate:.2f}"


def test_odd_even_stability():
    """
    检查 local 格式是否存在严重的奇偶解耦高频振荡。
    """
    res = run_single_fracture_step_wave(scheme="local", dt=1.0e-3, Cf=1.0e-5)
    H_frac = res["H_frac"]
    diff2 = np.abs(H_frac[2:] - 2 * H_frac[1:-1] + H_frac[:-2])
    max_oscillation = np.max(diff2)
    print(f"\nLocal scheme 最大连续锯齿阶跃测度: {max_oscillation:.6f} m")
    assert max_oscillation < 5.0, f"波形存在数值发散或异常振荡，max_oscillation={max_oscillation}"
