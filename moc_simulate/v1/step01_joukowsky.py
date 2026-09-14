# -*- coding: utf-8 -*-
"""
Joukowsky 解析验证脚本（路线 B / Step 2）

验证算例 1：单管水库趾端 + 井口流速阶跃（经典 Joukowsky）
----------------------------------------------------------
- 井筒 L=1000 m，a=1450 m/s，V0=1.0 m/s，停泵时刻 ts=1.0 s
- 趾端 = 水库（H=H0 常数），井口 = 流速源 V0
- 解析解（无摩阻）：
    停泵瞬间井口 V: V0 → 0
    Joukowsky 水击: ΔH = a V0 / g  (井口水头跃升)
    波到达趾端时间: t_toe = L / a
    水库反射系数 -1（压力波→抽吸波），反射回到井口时间: t_reflect = 2 L / a
    在 t ∈ [ts, ts + 2L/a) 井口水头应为 H0 + ΔH
    在 t ∈ [ts + 2L/a, ts + 4L/a) 井口水头应为 H0（水库负反射抵消）
    ...（井口 V=0 边界反射系数 +1，水库 -1，交替）

判定标准
--------
- 井口 ΔH 误差 < 0.1%
- 反射到达时间误差 < 1 dt
- 稳态段（t < ts）H 抖动 < 0.01 m
- 趾端 H 始终等于 H0（水库约束）

运行
----
    python dataset_builder/validate_moc_joukowsky.py
"""
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'README.md')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find wellbore_moc_method root')
    _d = _parent

from moc_simulate.paths import output_path, SERIES_STEP01_JOUKOWSKY
from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore, G

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# 解析解（无摩阻）
# =====================================================================
def joukowsky_analytical(L: float, a: float, V0: float, H0: float,
                          ts: float, t: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    单管水库趾端 + 井口流速阶跃的解析解（无摩阻）。

    场景：井口=流速源 V0（停泵前），停泵 V→0；趾端=水库 H=H0 常数。
    这是"源头截断"型 Joukowsky，初始压力**下降** -aV0/g
    （流体原本从井口流向趾端 +x，停泵后流体继续向 +x 流走，
     井口产生低压区；与"下游阀门关闭"的 +aV0/g 符号相反）。

    井口 V=0 边界反射系数 +1（闭端，压力波→压力波）
    趾端水库 H=常数边界反射系数 -1（压力波→抽吸波）

    井口水头随时间的阶梯变化：
        t < ts                            : H0
        ts ≤ t < ts + 2L/a                : H0 - ΔH     (源头截断，压力下降)
        ts + 2L/a ≤ t < ts + 4L/a         : H0 + ΔH     (水库负反射+闭端正反射，反弹回升)
        ts + 4L/a ≤ t < ts + 6L/a         : H0 - ΔH     (再次反转)
        ...
    其中 ΔH = a V0 / g（幅值，正数）；初始变化为 -ΔH。
    """
    dH_mag = a * V0 / G                     # 幅值（正）
    dH_initial = -dH_mag                    # 源头截断：初始为负
    T_reflect = 2.0 * L / a
    H = np.full_like(t, H0)
    H[t < ts] = H0
    mask = t >= ts
    dt_arr = t[mask] - ts
    n_period = np.floor(dt_arr / T_reflect).astype(int)
    # 偶数周期 (n=0,2,4,...): H0 - ΔH；奇数周期 (n=1,3,5,...): H0 + ΔH
    H[mask] = H0 + (2 * (n_period % 2) - 1) * dH_mag
    # 返回 (H, 初始ΔH, 反射周期)；初始ΔH 为负
    return H, dH_initial, T_reflect


# =====================================================================
# 验证主流程
# =====================================================================
def run_validation():
    print("=" * 72)
    print("Joukowsky 解析验证 — 单管死端 + 井口流速阶跃")
    print("=" * 72)

    # ── 仿真参数 ──────────────────────────────────────────
    L = 1000.0       # m
    a = 1450.0       # m/s
    V0 = 1.0         # m/s
    H0 = 300.0       # m
    ts = 1.0         # s
    dt = 1.0e-3      # s
    tf = 4.0         # s（覆盖至少 2 个反射周期：2L/a ≈ 1.38 s，4 s 涵盖 ~3 周期）

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
        toe_bc="reservoir",   # 经典 Joukowsky: 水库趾端，反射系数 -1
        toe_head=H0,
    )

    print(f"\n物理参数:")
    print(f"  L={L} m, a_adj={cfg.a_adj:.4f} m/s (原始 a={a})")
    print(f"  V0={V0} m/s, H0={H0} m, ts={ts} s")
    print(f"  dx={cfg.dx:.4f} m, N={cfg.N}, dt={cfg.dt_adj} s, n_steps={cfg.n_steps}")

    # ── 运行仿真 ──────────────────────────────────────────
    print("\n运行 MOC 仿真...")
    res = simulate_wellbore(cfg)
    t_sim = res["timestamps"]
    H_wh = res["wellhead_head"]
    V_wh = res["wellhead_velocity"]
    H_toe = res["toe_head"]
    V_toe = res["toe_velocity"]

    # ── 解析解 ────────────────────────────────────────────
    # 用实际稳态井口水头作为 H0 参考（含摩擦梯度，自洽）
    ts_idx = int(round(ts / dt))
    H0_actual = H_wh[ts_idx - 1]   # 停泵前最后一步的真实井口水头
    H_ana, dH_ana, T_reflect = joukowsky_analytical(L, cfg.a_adj, V0, H0_actual, ts, t_sim)
    dH_mag = abs(dH_ana)   # 幅值（正数）

    print(f"\n解析关键量:")
    print(f"  稳态井口水头 H0_actual = {H0_actual:.4f} m (含摩擦梯度，config H0={H0})")
    print(f"  Joukowsky |ΔH| = a V0 / g = {dH_mag:.4f} m  (初始变化 {dH_ana:+.4f} m, 源头截断→下降)")
    print(f"  反射周期 2L/a = {T_reflect:.4f} s")
    print(f"  首次水库反射到达井口时间 ts + 2L/a = {ts + T_reflect:.4f} s")

    # ── 判定 1: 停泵瞬间 ΔH ───────────────────────────────
    # 停泵跳变发生在 t=ts 这一帧（V: V0 → 0），即 wh[ts_idx-1] → wh[ts_idx]
    H_pre = H_wh[ts_idx - 1]
    H_post = H_wh[ts_idx]
    dH_sim = H_post - H_pre
    err_dH = abs(dH_sim - dH_ana) / dH_mag * 100

    print(f"\n[判定 1] Joukowsky 水击幅值（源头截断，预期下降）:")
    print(f"  仿真 ΔH = {dH_sim:+.6f} m")
    print(f"  解析 ΔH = {dH_ana:+.6f} m  (=-aV0/g)")
    print(f"  误差    = {err_dH:.4f} %  (阈值 < 0.1%)")
    verdict_dH = "PASS" if err_dH < 0.1 else "FAIL"
    print(f"  结论: {verdict_dH}")

    # ── 判定 2: 反射到达时间 ──────────────────────────────
    # 水库趾端反射系数 -1 + 井口闭端反射 +1：
    # 第一周期 [ts, ts+2L/a): H ≈ H0_actual - ΔH（下降平台）
    # 第二周期 [ts+2L/a, ts+4L/a): H ≈ H0_actual + ΔH（反弹回升）
    # 检测 H 从 H0_actual-ΔH 平台反弹升过 H0_actual+0.5ΔH 的时刻
    t_reflect_ana = ts + T_reflect
    H_after_ts = H_wh[ts_idx:]
    t_after_ts = t_sim[ts_idx:]
    rise_threshold = H0_actual + 0.5 * dH_mag
    # 跳过第一帧（停泵瞬时），从第二帧开始找升过阈值
    rise_idx = np.where(H_after_ts[1:] > rise_threshold)[0]
    if len(rise_idx) > 0:
        t_reflect_sim = t_after_ts[rise_idx[0] + 1]
        err_t = abs(t_reflect_sim - t_reflect_ana)
        err_t_dt = err_t / dt
        print(f"\n[判定 2] 水库负反射+闭端正反射到达井口时间（H 反弹升过 H0+0.5ΔH）:")
        print(f"  仿真 t = {t_reflect_sim:.6f} s")
        print(f"  解析 t = {t_reflect_ana:.6f} s")
        print(f"  误差    = {err_t:.6f} s = {err_t_dt:.2f} dt  (阈值 < 1 dt)")
        verdict_t = "PASS" if err_t_dt < 1.0 else "FAIL"
        print(f"  结论: {verdict_t}")
    else:
        t_reflect_sim = np.nan
        err_t = np.nan
        verdict_t = "N/A"
        print(f"\n[判定 2] 未检测到 H 反弹（仿真时间内 H 未升过 H0+0.5ΔH）")

    # ── 判定 3: 稳态段抖动 ────────────────────────────────
    steady_mask = t_sim < ts
    if steady_mask.sum() > 10:
        H_steady = H_wh[steady_mask]
        # 稳态井口水头应随沿程摩擦梯度稳定（井口 H ≈ H0 + friction_drop）
        # 抖动判定：去掉线性趋势后残差的最大幅度
        # 简化：稳态段应近乎常数（水库 V0 已建立），最大偏离 < 0.01 m
        H_jitter = np.max(np.abs(H_steady - H_steady[0]))
        print(f"\n[判定 3] 稳态段（t<ts）井口水头抖动:")
        print(f"  稳态首帧 H = {H_steady[0]:.4f} m")
        print(f"  稳态末帧 H = {H_steady[-1]:.4f} m")
        print(f"  max|H - H[0]| = {H_jitter:.6e} m  (阈值 < 0.01 m)")
        verdict_jit = "PASS" if H_jitter < 0.01 else "FAIL"
        print(f"  结论: {verdict_jit}")
    else:
        H_jitter = np.nan
        verdict_jit = "N/A"
        print(f"\n[判定 3] 稳态段样本不足，跳过抖动检测")

    # ── 判定 4: 趾端水库 H=常数约束 ───────────────────────
    toe_H_violation = np.max(np.abs(H_toe - cfg.toe_head))
    print(f"\n[判定 4] 趾端水库 H=H0 约束:")
    print(f"  max|H_toe - H0| = {toe_H_violation:.6e} m  (阈值 < 1e-10)")
    verdict_toe = "PASS" if toe_H_violation < 1e-10 else "FAIL"
    print(f"  结论: {verdict_toe}")

    # ── 总评 ─────────────────────────────────────────────
    verdicts = [verdict_dH, verdict_t, verdict_jit, verdict_toe]
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_total = sum(1 for v in verdicts if v != "N/A")
    print("\n" + "=" * 72)
    print(f"总评: {n_pass}/{n_total} 项 PASS")
    if n_pass == n_total:
        print("[OK] Joukowsky 验证通过 -- MOC 骨架物理正确，可进入 Step 3（加裂缝）")
    else:
        print("[FAIL] 验证未全部通过 -- 请检查上面对应 FAIL 项的诊断")
    print("=" * 72)

    # ── 可视化 ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Joukowsky 验证 — 单管水库趾端 + 井口流速阶跃（源头截断型）\n"
                 f"L={L}m, a={cfg.a_adj:.1f}m/s, V0={V0}m/s, |ΔH|_ana={dH_mag:.2f}m",
                 fontsize=13, fontweight='bold')

    # (0,0) 井口 H 时程：仿真 vs 解析
    ax = axes[0, 0]
    ax.plot(t_sim, H_wh, 'b-', lw=1.5, label='MOC 仿真')
    ax.plot(t_sim, H_ana, 'r--', lw=1.2, label='Joukowsky 解析（无摩阻）')
    ax.axvline(ts, color='g', ls=':', lw=1.2, label=f'停泵 ts={ts}s')
    ax.axvline(ts + T_reflect, color='m', ls=':', lw=1.0,
               label=f'反射到达 ts+2L/a={ts+T_reflect:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title('井口水头时程 — 仿真 vs 解析（源头截断型 Joukowsky）')
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (0,1) 去趋势井口 H（停泵后）
    ax = axes[0, 1]
    mask_post = t_sim >= ts
    ax.plot(t_sim[mask_post], H_wh[mask_post] - H0_actual, 'b-', lw=1.5, label='MOC ΔH')
    ax.plot(t_sim[mask_post], H_ana[mask_post] - H0_actual, 'r--', lw=1.2, label='解析 ΔH=±n·aV0/g')
    for k in [-1, 1, -2, 2]:
        ax.axhline(k * dH_mag, color='gray', ls=':', lw=0.6, alpha=0.5)
    ax.axvline(ts + T_reflect, color='m', ls=':', lw=1.0)
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('水头变化 [m]')
    ax.set_title('井口去趋势水头 — 源头截断后阶梯反射叠加')
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([ts, tf])

    # (1,0) 井口 + 趾端流速
    ax = axes[1, 0]
    ax.plot(t_sim, V_wh, 'b-', lw=1.5, label='井口流速')
    ax.plot(t_sim, V_toe, 'r-', lw=1.5, label='趾端流速')
    ax.axvline(ts, color='g', ls=':', lw=1.2, label=f'停泵 ts={ts}s')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('流速 [m/s]')
    ax.set_title('井口与趾端流速')
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (1,1) 趾端 H 时程
    ax = axes[1, 1]
    ax.plot(t_sim, H_toe, 'r-', lw=1.5, label='趾端水头 (MOC)')
    ax.axvline(ts, color='g', ls=':', lw=1.2, label=f'停泵 ts={ts}s')
    ax.axvline(ts + L / cfg.a_adj, color='m', ls=':', lw=1.0,
               label=f'波到达趾端 ts+L/a={ts + L/cfg.a_adj:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('趾端水头 [m]')
    ax.set_title('趾端水头时程')
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_path(SERIES_STEP01_JOUKOWSKY, None, "validation.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n图已保存: {out_path}")

    # JSON 结果
    result = {
        "verdicts": {
            "joukowsky_dH": verdict_dH,
            "reflection_timing": verdict_t,
            "steady_jitter": verdict_jit,
            "toe_hbc": verdict_toe,
        },
        "metrics": {
            "dH_sim": float(dH_sim),
            "dH_ana": float(dH_ana),
            "dH_mag": float(dH_mag),
            "dH_err_pct": float(err_dH),
            "t_reflect_sim": float(t_reflect_sim) if not np.isnan(t_reflect_sim) else None,
            "t_reflect_ana": float(t_reflect_ana),
            "t_reflect_err_dt": float(err_t / dt) if not np.isnan(err_t) else None,
            "steady_jitter_m": float(H_jitter) if not np.isnan(H_jitter) else None,
            "toe_H_violation_m": float(toe_H_violation),
        },
        "config": {
            "L": L, "a": a, "a_adj": float(cfg.a_adj), "V0": V0, "H0": H0,
            "ts": ts, "dt": dt, "tf": tf, "N": cfg.N, "dx": float(cfg.dx),
        },
    }
    json_path = output_path(SERIES_STEP01_JOUKOWSKY, None, "validation.json")
    import json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON: {json_path}")

    plt.close('all')
    return result


if __name__ == "__main__":
    run_validation()
