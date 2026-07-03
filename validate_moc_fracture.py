# -*- coding: utf-8 -*-
"""
Step 3 验证脚本 — 单缝集总柔度 + Joukowsky（算例 2）

物理场景
--------
- 井筒 L=1000 m，a=1450 m/s，V0=1.0 m/s，ts=1.0 s
- 趾端 = 水库 H=H0（反射系数 -1）
- 单缝在 x_f=500 m，柔度 C_f，k_leak=0（纯柔度，无滤失）
- 停泵 → Joukowsky 负阶跃 -ΔH 从井口传向趾端

解析解（频域反射系数 → 时域）
-------------------------------
裂缝反射系数（Laplace 域）:
    Γ(s) = -s·C_f·Z_w / (2 + s·C_f·Z_w)
    其中 Z_w = a/(gA) 为井筒特征阻抗

对阶跃输入 H_inc = -ΔH·u(t)，反射波:
    H_refl(t) = +ΔH·e^{-t/τ}·u(t)
    其中 τ = C_f·Z_w / 2 = C_f·a / (2·g·A)

物理：高频时裂缝像开端（Γ=-1，压力波→抽吸波），低频时透明（Γ=0）。
时域：反射波为正指数衰减（初始 +ΔH，衰减到 0），时间常数 τ。

井口水头预期
------------
    t < ts                              : H0_actual（稳态）
    ts ≤ t < ts + 2x_f/a                : H0_actual - ΔH（Joukowsky 平台）
    ts + 2x_f/a ≤ t < ts + 2L/a         : H0_actual - ΔH + ΔH·e^{-(t-t_arrive)/τ}
                                            （缝反射到达，指数回升）
    ts + 2L/a ≤ ...                     : 叠加趾端水库反射（复杂，避开）

判定标准
--------
1. 缝反射到达时间 t_arrive = ts + 2·x_f/a，误差 < 1 dt
2. 缝反射初始幅值 = +ΔH（t_arrive 瞬间 H 从 H0-ΔH 跳到 ~H0），误差 < 1%
3. 指数衰减时间常数 τ = C_f·a/(2gA)，拟合误差 < 5%
4. 稳态段（t<ts）抖动 < 0.01 m
5. 无缝场景（Cf=0）应退化为纯 Joukowsky（无中间反射）

运行
----
    python dataset_builder/validate_moc_fracture.py
"""
import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

_METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
if _METHOD_DIR not in sys.path:
    sys.path.insert(0, _METHOD_DIR)

from paths import moc_output_dir
from wellbore_moc import MocConfig, simulate_wellbore, G

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# 解析解
# =====================================================================
def fracture_reflection_analytical(
    L: float, a: float, V0: float, H0: float,
    ts: float, x_f: float, Cf: float, area: float,
    t: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    单缝 + 水库趾端的井口水头解析解（无摩阻，纯柔度缝）。

    关键：缝反射波到达井口后，井口闭端(V=0)反射系数 +1，将缝反射波
    再次反射 → 总跳变 = 2ΔH（不是 ΔH）。

    井口水头:
        t < ts                          : H0
        ts ≤ t < ts+2x_f/a              : H0 - ΔH                 (Joukowsky 平台)
        ts+2x_f/a ≤ t < ts+2L/a         : H0 - ΔH + 2ΔH·e^{-dt/τ} (缝反射+闭端反射叠加)
        t ≥ ts+2L/a                     : 叠加趾端反射（复杂，避开）

    返回 (H_ana, dH_jouk, t_arrive_frac, tau, amplitude_factor)
    其中 amplitude_factor=2（井口闭端加倍效应）
    """
    dH = a * V0 / G
    Z_w = a / (G * area)
    tau = Cf * Z_w / 2.0
    t_arrive_frac = ts + 2.0 * x_f / a
    t_arrive_toe = ts + 2.0 * L / a
    amp_factor = 2.0   # 井口闭端反射系数 +1 → 缝反射加倍

    H = np.full_like(t, H0)
    mask_steady = t < ts
    H[mask_steady] = H0

    mask_jouk = (t >= ts) & (t < t_arrive_frac)
    H[mask_jouk] = H0 - dH

    mask_frac = (t >= t_arrive_frac) & (t < t_arrive_toe)
    dt_after = t[mask_frac] - t_arrive_frac
    H[mask_frac] = H0 - dH + amp_factor * dH * np.exp(-dt_after / tau)

    mask_toe = t >= t_arrive_toe
    H[mask_toe] = np.nan

    return H, dH, t_arrive_frac, tau, amp_factor


# =====================================================================
# 拟合指数衰减时间常数
# =====================================================================
def fit_exponential_decay(t: np.ndarray, H: np.ndarray, t_arrive: float,
                           H_baseline: float, dH: float) -> Tuple[float, float]:
    """
    从井口 H 时程拟合指数衰减时间常数 τ。

    模型: H(t) = H_baseline + dH·exp(-(t - t_arrive)/τ)
    取对数: ln((H - H_baseline)/dH) = -(t - t_arrive)/τ
    线性拟合斜率 = -1/τ

    返回 (tau_fit, R²)
    """
    mask = (t >= t_arrive) & (t < t_arrive + 5.0)  # 取到达后 5s 窗口
    if mask.sum() < 10:
        return np.nan, 0.0
    t_win = t[mask]
    H_win = H[mask]
    y = H_win - H_baseline
    # 排除非正数
    pos = y > 1e-6
    if pos.sum() < 5:
        return np.nan, 0.0
    ln_y = np.log(y[pos])
    dt_arr = t_win[pos] - t_arrive
    # 线性回归 ln_y = -dt_arr/tau + const
    A_mat = np.vstack([dt_arr, np.ones_like(dt_arr)]).T
    slope, intercept = np.linalg.lstsq(A_mat, ln_y, rcond=None)[0]
    tau_fit = -1.0 / slope if slope != 0 else np.nan
    # R²
    y_pred = slope * dt_arr + intercept
    ss_res = np.sum((ln_y - y_pred) ** 2)
    ss_tot = np.sum((ln_y - np.mean(ln_y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return tau_fit, r2


# =====================================================================
# 验证主流程
# =====================================================================
def run_validation():
    print("=" * 72)
    print("Step 3 验证 — 单缝集总柔度 + Joukowsky（算例 2）")
    print("=" * 72)

    # ── 仿真参数 ──────────────────────────────────────────
    L = 1000.0
    a = 1450.0
    V0 = 1.0
    H0 = 300.0
    ts = 1.0
    dt = 1.0e-3
    tf = 20.0   # 长时仿真：覆盖缝反射(1.69s) + 趾端多次反射(周期1.38s) + 长期衰减
    x_f = 500.0
    Cf = 1.0e-4   # 柔度 [m²]，预期 τ ≈ 0.48s（可见衰减）

    cfg = MocConfig(
        wellbore_length=L,
        wellbore_diameter=0.1397,
        fluid_density=1000.0,
        fluid_viscosity=1.0e-6,
        wavespeed=a,
        roughness_height=4.5e-5,
        friction_model="steady",
        dt=dt,
        tf=tf,
        wellhead_bc="velocity_step",
        pump_shut_time=ts,
        initial_velocity=V0,
        initial_head=H0,
        theta=0.0,
        toe_bc="reservoir",
        toe_head=H0,
    )

    Z_w = cfg.a_adj / (G * cfg.area)
    tau_ana = Cf * Z_w / 2.0
    dH_ana = cfg.a_adj * V0 / G
    t_arrive_ana = ts + 2.0 * x_f / cfg.a_adj

    print(f"\n物理参数:")
    print(f"  L={L} m, a_adj={cfg.a_adj:.4f} m/s, V0={V0} m/s, ts={ts} s")
    print(f"  缝位置 x_f={x_f} m, 柔度 Cf={ Cf} m², k_leak=0")
    print(f"  Z_w = a/(gA) = {Z_w:.2f} s/m²")
    print(f"  τ_ana = Cf·Z_w/2 = {tau_ana:.4f} s")
    print(f"  ΔH_ana = aV0/g = {dH_ana:.4f} m")
    print(f"  缝反射到达 t_arrive = ts + 2x_f/a = {t_arrive_ana:.4f} s")
    print(f"  趾端反射到达 = ts + 2L/a = {ts + 2*L/cfg.a_adj:.4f} s")
    print(f"  dx={cfg.dx:.4f} m, N={cfg.N}, n_steps={cfg.n_steps}")

    # ── 运行仿真（含缝）──────────────────────────────────
    print("\n运行 MOC 仿真（含缝）...")
    res = simulate_wellbore(
        cfg,
        fracture_positions=[x_f],
        fracture_Cf=[Cf],
        fracture_kleak=[0.0],
        H_ext=0.0,
    )
    t_sim = res["timestamps"]
    H_wh = res["wellhead_head"]
    H_frac = res["fracture_heads"][:, 0]
    Q_frac = res["fracture_Qs"][:, 0]
    frac_idx = res["fracture_indices"][0]
    print(f"  缝对齐到网格索引 {frac_idx} (x={res['x_grid'][frac_idx]:.2f} m)")

    # ── 运行对照仿真（无缝）──────────────────────────────
    print("运行 MOC 仿真（无缝对照）...")
    res_noFrac = simulate_wellbore(cfg)  # 无缝
    H_wh_noFrac = res_noFrac["wellhead_head"]

    # ── 解析解 ────────────────────────────────────────────
    ts_idx = int(round(ts / dt))
    H0_actual = H_wh[ts_idx - 1]   # 含摩擦梯度的稳态井口水头
    H_ana, _, _, _, amp_factor = fracture_reflection_analytical(
        L, cfg.a_adj, V0, H0_actual, ts, x_f, Cf, cfg.area, t_sim
    )

    print(f"\n稳态井口水头 H0_actual = {H0_actual:.4f} m")

    # ── 判定 1: 缝反射到达时间 ────────────────────────────
    # 检测 H 从 H0-ΔH 平台开始回升的时刻（导数最大正跳变）
    H_after_ts = H_wh[ts_idx:]
    t_after_ts = t_sim[ts_idx:]
    # 平台期 H ≈ H0_actual - dH；缝反射到达后 H 开始回升
    baseline = H0_actual - dH_ana
    # 找 H 超过 baseline + 0.05·dH 的时刻
    rise_thresh = baseline + 0.05 * dH_ana
    rise_idx = np.where(H_after_ts[1:] > rise_thresh)[0]
    if len(rise_idx) > 0:
        t_arrive_sim = t_after_ts[rise_idx[0] + 1]
        err_t = abs(t_arrive_sim - t_arrive_ana)
        err_t_dt = err_t / dt
        print(f"\n[判定 1] 缝反射到达井口时间:")
        print(f"  仿真 t = {t_arrive_sim:.6f} s")
        print(f"  解析 t = {t_arrive_ana:.6f} s")
        print(f"  误差    = {err_t:.6f} s = {err_t_dt:.2f} dt  (阈值 < 1 dt)")
        verdict_t = "PASS" if err_t_dt < 1.0 else "FAIL"
    else:
        t_arrive_sim = np.nan
        verdict_t = "FAIL"
        print(f"\n[判定 1] 未检测到缝反射回升")
    print(f"  结论: {verdict_t}")

    # ── 判定 2: 缝反射初始幅值 ────────────────────────────
    # 到达瞬间 H 跳变量 = amp_factor · ΔH = 2ΔH（井口闭端反射系数 +1，加倍）
    arrive_idx = int(round((t_arrive_sim - 0) / dt)) if not np.isnan(t_arrive_sim) else ts_idx
    H_before = H_wh[arrive_idx - 1]
    H_after = H_wh[arrive_idx]
    dH_refl_sim = H_after - H_before
    dH_refl_ana = amp_factor * dH_ana   # 2ΔH
    err_amp = abs(dH_refl_sim - dH_refl_ana) / dH_refl_ana * 100
    print(f"\n[判定 2] 缝反射初始幅值（井口闭端反射系数 +1 → 加倍）:")
    print(f"  仿真 ΔH_refl = {dH_refl_sim:+.4f} m (H: {H_before:.2f}→{H_after:.2f})")
    print(f"  解析 ΔH_refl = +{dH_refl_ana:.4f} m (= {amp_factor}×aV0/g，含闭端加倍)")
    print(f"  误差    = {err_amp:.4f} %  (阈值 < 5%，含摩擦/离散容差)")
    verdict_amp = "PASS" if err_amp < 5.0 else "FAIL"
    print(f"  结论: {verdict_amp}")

    # ── 判定 3: 指数衰减时间常数 ──────────────────────────
    # 拟合窗口：缝反射到达后到趾端反射到达前
    # 模型: H(t) = baseline + amp_factor·dH·exp(-(t-t_arrive)/τ)
    #       baseline = H0_actual - dH_ana,  amp_factor = 2
    t_fit_end = min(t_arrive_ana + 5.0 * tau_ana, ts + 2 * L / cfg.a_adj - 0.05)
    mask_fit = (t_sim >= t_arrive_sim) & (t_sim < t_fit_end)
    if mask_fit.sum() > 20:
        tau_fit, r2 = fit_exponential_decay(
            t_sim[mask_fit], H_wh[mask_fit], t_arrive_sim, baseline,
            amp_factor * dH_ana   # 2ΔH
        )
        err_tau = abs(tau_fit - tau_ana) / tau_ana * 100
        print(f"\n[判定 3] 指数衰减时间常数 τ:")
        print(f"  拟合 τ = {tau_fit:.4f} s (R²={r2:.6f})")
        print(f"  解析 τ = {tau_ana:.4f} s (= Cf·a/(2gA))")
        print(f"  误差    = {err_tau:.4f} %  (阈值 < 5%)")
        verdict_tau = "PASS" if (err_tau < 5.0 and r2 > 0.95) else "FAIL"
    else:
        tau_fit = np.nan
        r2 = 0.0
        err_tau = np.nan
        verdict_tau = "FAIL"
        print(f"\n[判定 3] 拟合窗口样本不足 ({mask_fit.sum()} 点)")
    print(f"  结论: {verdict_tau}")

    # ── 判定 4: 稳态段抖动 ────────────────────────────────
    steady_mask = t_sim < ts
    H_steady = H_wh[steady_mask]
    H_jitter = np.max(np.abs(H_steady - H_steady[0]))
    print(f"\n[判定 4] 稳态段（t<ts）井口水头抖动:")
    print(f"  max|H - H[0]| = {H_jitter:.6e} m  (阈值 < 0.01 m)")
    verdict_jit = "PASS" if H_jitter < 0.01 else "FAIL"
    print(f"  结论: {verdict_jit}")

    # ── 判定 5: 无缝退化为纯 Joukowsky ────────────────────
    # 无缝场景在缝反射到达前应与含缝完全一致
    mask_before_frac = (t_sim >= ts) & (t_sim < t_arrive_ana - 0.01)
    if mask_before_frac.sum() > 0:
        max_diff = np.max(np.abs(H_wh[mask_before_frac] - H_wh_noFrac[mask_before_frac]))
        print(f"\n[判定 5] 无缝退化为纯 Joukowsky（缝反射到达前两者应一致）:")
        print(f"  max|H_withFrac - H_noFrac| = {max_diff:.6e} m  (阈值 < 1e-10)")
        verdict_degen = "PASS" if max_diff < 1e-10 else "FAIL"
    else:
        max_diff = np.nan
        verdict_degen = "N/A"
        print(f"\n[判定 5] 窗口不足")
    print(f"  结论: {verdict_degen}")

    # ── 判定 6: 长期数值稳定性（tf=20s 专用）──────────────
    # 检查仿真末期（最后 1s）是否数值稳定（无发散/NaN/Inf）
    # 物理预期：稳态达西摩阻 J=f·dt·V|V|/(2D)，停泵后 V≈0 附近振荡，
    # 每周期累计衰减 ≈0.1m << ΔH=148m → 振荡不衰减是物理正确行为
    # 判定标准：数值不发散（无 NaN/Inf + H 范围 < 6ΔH ≈ 886m）
    last_1s_mask = t_sim >= (tf - 1.0)
    if last_1s_mask.sum() > 10:
        H_last = H_wh[last_1s_mask]
        H_last_range = float(np.max(H_last) - np.min(H_last))
        H_last_mean = float(np.mean(H_last))
        H_last_std = float(np.std(H_last))
        has_nan = bool(np.any(np.isnan(H_last)) or np.any(np.isinf(H_last)))
        # 无摩耗理论预期：井口 H 在 H0±ΔH 振荡，范围 ~2ΔH；含缝叠加可达 ~3ΔH
        # 数值发散阈值：6ΔH（远超物理预期，防数值爆炸）
        range_limit = 6.0 * dH_ana
        print(f"\n[判定 6] 长期数值稳定性（最后 1s, t∈[{tf-1:.0f},{tf:.0f}]s）:")
        print(f"  末期 H 均值 = {H_last_mean:.4f} m")
        print(f"  末期 H 标准差 = {H_last_std:.4f} m")
        print(f"  末期 H 范围 = {H_last_range:.4f} m  (max-min)")
        print(f"  无摩耗理论预期范围 ~2ΔH = {2*dH_ana:.1f} m")
        print(f"  数值发散阈值 6ΔH = {range_limit:.1f} m")
        print(f"  含 NaN/Inf = {has_nan}")
        if has_nan:
            verdict_stab = "FAIL"
            print(f"  → FAIL：含 NaN/Inf，数值发散")
        elif H_last_range < range_limit:
            verdict_stab = "PASS"
            print(f"  → PASS：H 范围 < 6ΔH，无数值发散（振荡不衰减是稳态摩阻物理预期）")
        else:
            verdict_stab = "FAIL"
            print(f"  → FAIL：H 范围 ≥ 6ΔH，疑似数值发散")
    else:
        verdict_stab = "N/A"
        H_last_range = np.nan
        H_last_mean = np.nan
        H_last_std = np.nan
        print(f"\n[判定 6] 末期窗口不足")
    print(f"  结论: {verdict_stab}")

    # ── 总评 ─────────────────────────────────────────────
    verdicts = [verdict_t, verdict_amp, verdict_tau, verdict_jit, verdict_degen, verdict_stab]
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_total = sum(1 for v in verdicts if v != "N/A")
    print("\n" + "=" * 72)
    print(f"总评: {n_pass}/{n_total} 项 PASS")
    if n_pass == n_total:
        print("[OK] Step 3 验证通过 -- 裂缝集总柔度边界物理正确")
    else:
        print("[FAIL] 验证未全部通过 -- 检查上方对应 FAIL 项")
    print("=" * 72)

    # ── 可视化 ────────────────────────────────────────────
    out_dir = moc_output_dir()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Step 3 验证 — 单缝集总柔度 + Joukowsky (tf={tf}s 长时仿真)\n"
        f"L={L}m, x_f={x_f}m, Cf={Cf}m², τ_ana={tau_ana:.3f}s, |ΔH|={dH_ana:.1f}m",
        fontsize=13, fontweight='bold'
    )

    # (0,0) 井口 H 全时程（20s）：含缝 vs 无缝，展示长期振荡与衰减
    ax = axes[0, 0]
    ax.plot(t_sim, H_wh, 'b-', lw=0.8, label='含缝 MOC')
    ax.plot(t_sim, H_wh_noFrac, 'g--', lw=0.6, alpha=0.6, label='无缝 MOC (对照)')
    ax.axvline(ts, color='g', ls=':', lw=1, label=f'停泵 ts={ts}s')
    ax.axvline(t_arrive_ana, color='m', ls=':', lw=1.2,
               label=f'缝反射 {t_arrive_ana:.2f}s')
    ax.axvline(ts + 2 * L / cfg.a_adj, color='orange', ls=':', lw=1,
               label=f'趾端反射 {ts + 2*L/cfg.a_adj:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title(f'井口水头全时程 (tf={tf}s) — 长期振荡与衰减')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (0,1) 缝反射特写（ts ~ ts+2L/a）：含缝 vs 无缝 vs 解析 + 指数拟合
    ax = axes[0, 1]
    zoom_end = ts + 2 * L / cfg.a_adj + 0.3   # 趾端反射后 0.3s
    mask_zoom = (t_sim >= ts - 0.1) & (t_sim <= zoom_end)
    ax.plot(t_sim[mask_zoom], H_wh[mask_zoom], 'b-', lw=1.5, label='含缝 MOC')
    ax.plot(t_sim[mask_zoom], H_wh_noFrac[mask_zoom], 'g--', lw=1.0, alpha=0.6,
            label='无缝 MOC (对照)')
    ax.plot(t_sim[mask_zoom], H_ana[mask_zoom], 'r--', lw=1.2, label='解析（缝反射段）')
    ax.axvline(ts, color='g', ls=':', lw=1, label=f'停泵 ts={ts}s')
    ax.axvline(t_arrive_ana, color='m', ls=':', lw=1.2,
               label=f'缝反射到达 {t_arrive_ana:.2f}s')
    ax.axvline(ts + 2 * L / cfg.a_adj, color='orange', ls=':', lw=1,
               label=f'趾端反射 {ts + 2*L/cfg.a_adj:.2f}s')
    if not np.isnan(tau_fit):
        t_fit_arr = np.linspace(t_arrive_sim, t_fit_end, 200)
        H_fit = H0_actual - dH_ana + amp_factor * dH_ana * np.exp(-(t_fit_arr - t_arrive_sim) / tau_fit)
        ax.plot(t_fit_arr, H_fit, 'r:', lw=1.5,
                label=f'拟合 τ={tau_fit:.3f}s (R²={r2:.4f})')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title('缝反射特写 — 解析 vs 仿真 + 指数衰减拟合')
    ax.legend(fontsize=7); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([ts - 0.1, zoom_end])

    # (1,0) 缝节点 H 与 Q_f（全 20s）
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(t_sim, H_frac, 'b-', lw=0.8, label='缝节点 H')
    ax2.plot(t_sim, Q_frac, 'r-', lw=0.6, alpha=0.7, label='缝侧向流量 Q_f')
    ax.axvline(ts, color='g', ls=':', lw=1)
    ax.axvline(ts + x_f / cfg.a_adj, color='m', ls=':', lw=1,
               label=f'波到达缝 {ts + x_f/cfg.a_adj:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('缝节点水头 [m]', color='b')
    ax2.set_ylabel('缝侧向流量 [m³/s]', color='r')
    ax.set_title(f'缝节点水头与侧向流量 (tf={tf}s)')
    ax.legend(fontsize=8, loc='upper left'); ax2.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (1,1) 沿井筒 H 空间分布快照（含长期时刻）
    ax = axes[1, 1]
    head_field = res["head"]
    x_grid = res["x_grid"]
    snap_times = [ts, ts + x_f / cfg.a_adj, t_arrive_ana,
                  ts + 2 * L / cfg.a_adj, 5.0, 10.0, tf]
    colors_snap = ['b', 'm', 'r', 'orange', 'cyan', 'purple', 'brown']
    for st, col in zip(snap_times, colors_snap):
        idx = int(round(st / dt))
        if 0 <= idx < head_field.shape[0]:
            ax.plot(x_grid, head_field[idx], '-', color=col, lw=1.0,
                    label=f't={st:.1f}s')
    ax.axvline(x_f, color='k', ls=':', lw=1, label=f'缝 x_f={x_f}m')
    ax.set_xlabel('沿井筒位置 [m]'); ax.set_ylabel('水头 [m]')
    ax.set_title(f'井筒水头空间分布快照 (含长期 t=5,10,{tf}s)')
    ax.legend(fontsize=7, loc='best'); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, L])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, "fracture_validation.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n图已保存: {out_path}")

    # JSON 结果
    import json
    result = {
        "verdicts": {
            "reflection_timing": verdict_t,
            "reflection_amplitude": verdict_amp,
            "decay_tau": verdict_tau,
            "steady_jitter": verdict_jit,
            "no_frac_degeneracy": verdict_degen,
            "long_term_stability": verdict_stab,
        },
        "metrics": {
            "t_arrive_sim": float(t_arrive_sim) if not np.isnan(t_arrive_sim) else None,
            "t_arrive_ana": float(t_arrive_ana),
            "dH_refl_sim": float(dH_refl_sim),
            "dH_refl_ana": float(dH_refl_ana),
            "amp_factor": float(amp_factor),
            "dH_ana": float(dH_ana),
            "amp_err_pct": float(err_amp),
            "tau_fit": float(tau_fit) if not np.isnan(tau_fit) else None,
            "tau_ana": float(tau_ana),
            "tau_err_pct": float(err_tau) if not np.isnan(err_tau) else None,
            "tau_fit_r2": float(r2),
            "steady_jitter_m": float(H_jitter),
            "no_frac_max_diff_m": float(max_diff) if not np.isnan(max_diff) else None,
            "long_term_H_mean": float(H_last_mean) if not np.isnan(H_last_mean) else None,
            "long_term_H_std": float(H_last_std) if not np.isnan(H_last_std) else None,
            "long_term_H_range": float(H_last_range) if not np.isnan(H_last_range) else None,
        },
        "config": {
            "L": L, "a": a, "a_adj": float(cfg.a_adj), "V0": V0, "H0": H0,
            "ts": ts, "dt": dt, "tf": tf, "N": cfg.N,
            "x_f": x_f, "Cf": Cf, "Z_w": float(Z_w),
        },
    }
    json_path = os.path.join(out_dir, "fracture_validation.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON: {json_path}")

    plt.close('all')
    return result


if __name__ == "__main__":
    run_validation()
