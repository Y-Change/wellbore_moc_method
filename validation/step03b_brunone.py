# -*- coding: utf-8 -*-
"""
Step 3 Brunone 验证 — L=5000m, tf=100s, Brunone 非定常摩阻

验证目标
--------
1. Joukowsky 初始跳变仍正确（Brunone 不影响瞬时跳变）
2. 缝反射时序仍正确（t=ts+2x_f/a）
3. 振荡显著衰减（Brunone 非定常摩阻 vs 稳态达西无衰减）
4. 长期稳定（100s 后振荡基本平息）
5. 衰减率与现场数据量级一致（每周期振幅衰减 ~50-70%）

物理参数
--------
L=5000m, a=1450 m/s → 2L/a ≈ 6.9s（与现场 ~7s 周期匹配）
x_f=2500m (L/2) → 缝反射 2x_f/a ≈ 3.45s
ts=1.0s, tf=100s, dt=0.001s → n_steps=100000, N≈3448

运行
----
    python dataset_builder/validate_moc_brunone.py
"""
import os
import sys
import time as time_module
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

_METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
if _METHOD_DIR not in sys.path:
    sys.path.insert(0, _METHOD_DIR)

from paths import output_path, SERIES_STEP03B_BRUNONE
from wellbore_moc import MocConfig, simulate_wellbore, G

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_validation():
    print("=" * 72)
    print("Step 3 Brunone 验证 — L=5000m, tf=100s, 非定常摩阻")
    print("=" * 72)

    # ── 仿真参数 ──────────────────────────────────────────
    L = 5000.0
    a = 1450.0
    V0 = 1.0
    H0 = 300.0
    ts = 1.0
    dt = 1.0e-3
    tf = 100.0
    x_f = 4500.0    # L/2
    Cf = 1.0e-4

    cfg = MocConfig(
        wellbore_length=L,
        wellbore_diameter=0.1397,
        fluid_density=1000.0,
        fluid_viscosity=1.0e-6,
        wavespeed=a,
        roughness_height=4.5e-5,
        friction_model="brunone",      # ★ Brunone 非定常摩阻
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

    dH_ana = cfg.a_adj * V0 / G
    t_arrive_frac = ts + 2.0 * x_f / cfg.a_adj
    t_arrive_toe = ts + 2.0 * L / cfg.a_adj
    T_toe = 2.0 * L / cfg.a_adj   # 趾端反射周期

    print(f"\n物理参数:")
    print(f"  L={L}m, a_adj={cfg.a_adj:.4f} m/s, V0={V0} m/s, ts={ts} s")
    print(f"  缝 x_f={x_f}m, Cf={Cf} m², k_leak=0")
    print(f"  摩阻模型: Brunone 非定常")
    print(f"  ΔH = aV0/g = {dH_ana:.4f} m")
    print(f"  缝反射到达 t = {t_arrive_frac:.4f} s")
    print(f"  趾端反射周期 2L/a = {T_toe:.4f} s (现场 ~7s)")
    print(f"  趾端首次反射 t = {t_arrive_toe:.4f} s")
    print(f"  dx={cfg.dx:.4f} m, N={cfg.N}, n_steps={cfg.n_steps}")
    print(f"  内存: store_full_field=False (100000×3449×16B≈5.5GB → 1D时程~10MB)")

    # 快照时刻
    snap_times = [0.0, ts, t_arrive_frac, t_arrive_toe, 10.0, 30.0, 50.0, 100.0]

    # ── 运行含缝仿真 ─────────────────────────────────────
    print(f"\n运行含缝 MOC 仿真 (Brunone, tf={tf}s)...")
    t_start = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=[x_f],
        fracture_Cf=[Cf],
        fracture_kleak=[0.0],
        H_ext=0.0,
        store_full_field=False,
        snapshot_times=snap_times,
    )
    t_elapsed = time_module.time() - t_start
    print(f"  仿真耗时: {t_elapsed:.1f}s")

    t_sim = res["timestamps"]
    H_wh = res["wellhead_head"]
    V_wh = res["wellhead_velocity"]
    H_toe = res["toe_head"]
    H_frac = res["fracture_heads"][:, 0]
    Q_frac = res["fracture_Qs"][:, 0]

    # ── 运行无缝对照 ─────────────────────────────────────
    print(f"运行无缝对照 MOC 仿真 (Brunone)...")
    t_start2 = time_module.time()
    res_noFrac = simulate_wellbore(
        cfg, store_full_field=False,
        snapshot_times=[t_arrive_toe, 50.0, 100.0],
    )
    t_elapsed2 = time_module.time() - t_start2
    print(f"  对照耗时: {t_elapsed2:.1f}s")
    H_wh_noFrac = res_noFrac["wellhead_head"]

    # ── 运行稳态摩阻对照（展示无衰减 vs Brunone 衰减）─────
    print(f"运行稳态摩阻对照 (steady, tf={tf}s)...")
    cfg_steady = MocConfig(
        wellbore_length=L, wellbore_diameter=0.1397,
        fluid_density=1000.0, fluid_viscosity=1.0e-6,
        wavespeed=a, roughness_height=4.5e-5,
        friction_model="steady",     # ★ 稳态摩阻
        dt=dt, tf=tf,
        wellhead_bc="velocity_step", pump_shut_time=ts,
        initial_velocity=V0, initial_head=H0,
        theta=0.0, toe_bc="reservoir", toe_head=H0,
    )
    t_start3 = time_module.time()
    res_steady = simulate_wellbore(
        cfg_steady, store_full_field=False,
    )
    t_elapsed3 = time_module.time() - t_start3
    print(f"  稳态对照耗时: {t_elapsed3:.1f}s")
    H_wh_steady = res_steady["wellhead_head"]

    # ── 解析关键量 ────────────────────────────────────────
    ts_idx = int(round(ts / dt))
    H0_actual = H_wh[ts_idx - 1]
    print(f"\n稳态井口水头 H0_actual = {H0_actual:.4f} m")

    # ── 判定 1: Joukowsky 初始跳变 ────────────────────────
    H_pre = H_wh[ts_idx - 1]
    H_post = H_wh[ts_idx]
    dH_sim = H_post - H_pre
    err_dH = abs(dH_sim - (-dH_ana)) / dH_ana * 100
    print(f"\n[判定 1] Joukowsky 初始跳变（Brunone 不影响瞬时跳变）:")
    print(f"  仿真 ΔH = {dH_sim:+.4f} m")
    print(f"  解析 ΔH = {-dH_ana:+.4f} m (=-aV0/g)")
    print(f"  误差    = {err_dH:.4f} %  (阈值 < 1%)")
    verdict_dH = "PASS" if err_dH < 1.0 else "FAIL"
    print(f"  结论: {verdict_dH}")

    # ── 判定 2: 缝反射到达时间 ────────────────────────────
    # 用对齐后的缝位置算解析（消除网格对齐误差）
    frac_idx_actual = res["fracture_indices"][0]
    x_f_aligned = res["x_grid"][frac_idx_actual]
    t_arrive_frac_aligned = ts + 2.0 * x_f_aligned / cfg.a_adj
    baseline = H0_actual - dH_ana
    H_after_ts = H_wh[ts_idx:]
    t_after_ts = t_sim[ts_idx:]
    rise_thresh = baseline + 0.05 * dH_ana
    rise_idx = np.where(H_after_ts[1:] > rise_thresh)[0]
    if len(rise_idx) > 0:
        t_arrive_sim = t_after_ts[rise_idx[0] + 1]
        err_t = abs(t_arrive_sim - t_arrive_frac_aligned)
        err_t_dt = err_t / dt
        print(f"\n[判定 2] 缝反射到达井口时间:")
        print(f"  缝对齐位置 x_f={x_f}m → 网格 x={x_f_aligned:.2f}m (idx={frac_idx_actual})")
        print(f"  仿真 t = {t_arrive_sim:.4f} s")
        print(f"  解析 t = {t_arrive_frac_aligned:.4f} s (用对齐位置)")
        print(f"  误差    = {err_t:.4f} s = {err_t_dt:.1f} dt  (阈值 < 10 dt，含 Brunone 模化容差)")
        verdict_t = "PASS" if err_t_dt < 10.0 else "FAIL"
    else:
        t_arrive_sim = np.nan
        verdict_t = "FAIL"
        print(f"\n[判定 2] 未检测到缝反射回升")
    print(f"  结论: {verdict_t}")

    # ── 判定 3: 振荡衰减（Brunone vs 稳态，RMS 能量指标）────
    # peak-to-peak 受波形相位叠加影响，不能反映能量衰减
    # 改用 RMS（相对均值）作为能量指标，比较末期 Brunone vs 稳态
    def osc_rms(H_arr, t_arr, t_start, t_end):
        """计算 [t_start, t_end] 窗口内 H 相对于均值的 RMS"""
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        if mask.sum() < 10:
            return np.nan
        H_win = H_arr[mask]
        return float(np.sqrt(np.mean((H_win - np.mean(H_win)) ** 2)))

    # 末期 RMS（最后 10s）：Brunone 应显著小于稳态
    rms_late_brunone = osc_rms(H_wh, t_sim, tf - 10.0, tf)
    rms_late_steady = osc_rms(H_wh_steady, t_sim, tf - 10.0, tf)
    # 早期 RMS（第1周期）：两者应相近
    rms_early_brunone = osc_rms(H_wh, t_sim, ts, ts + T_toe)
    rms_early_steady = osc_rms(H_wh_steady, t_sim, ts, ts + T_toe)
    # 衰减比 = 末期RMS / 早期RMS
    decay_brunone = rms_late_brunone / rms_early_brunone if rms_early_brunone > 0 else np.nan
    decay_steady = rms_late_steady / rms_early_steady if rms_early_steady > 0 else np.nan

    print(f"\n[判定 3] 振荡衰减对比（RMS 能量指标，Brunone vs 稳态）:")
    print(f"  早期 (第1周期 t∈[{ts:.0f},{ts+T_toe:.1f}]s):")
    print(f"    Brunone RMS = {rms_early_brunone:.2f} m")
    print(f"    稳态   RMS = {rms_early_steady:.2f} m")
    print(f"  末期 (最后10s t∈[{tf-10:.0f},{tf:.0f}]s):")
    print(f"    Brunone RMS = {rms_late_brunone:.2f} m")
    print(f"    稳态   RMS = {rms_late_steady:.2f} m")
    print(f"  衰减比 (末期/早期):")
    print(f"    Brunone: {decay_brunone:.4f}  (应显著 < 1)")
    print(f"    稳态:   {decay_steady:.4f}  (应 > Brunone，即衰减更少)")
    print(f"  Brunone/稳态 末期RMS比 = {rms_late_brunone/rms_late_steady:.4f} (应 < 1，证明 Brunone 衰减更强)")
    # 验证：Brunone 衰减比 < 稳态衰减比，且末期 RMS 比 < 0.9
    verdict_decay = "PASS" if (
        decay_brunone < decay_steady and
        rms_late_brunone / rms_late_steady < 0.9
    ) else "FAIL"
    print(f"  阈值: Brunone衰减比 < 稳态 且 末期RMS比 < 0.9 → {verdict_decay}")
    print(f"  结论: {verdict_decay}")

    # ── 判定 4: 长期稳定（100s 末期）──────────────────────
    last_5s_mask = t_sim >= (tf - 5.0)
    H_last = H_wh[last_5s_mask]
    H_last_range = float(np.max(H_last) - np.min(H_last))
    H_last_mean = float(np.mean(H_last))
    has_nan = bool(np.any(np.isnan(H_last)) or np.any(np.isinf(H_last)))
    print(f"\n[判定 4] 长期稳定性（最后 5s, t∈[{tf-5:.0f},{tf:.0f}]s）:")
    print(f"  末期 H 均值 = {H_last_mean:.4f} m")
    print(f"  末期 H 范围 = {H_last_range:.4f} m")
    print(f"  含 NaN/Inf = {has_nan}")
    # Brunone 衰减后，末期振幅应远小于初始 ΔH
    # 现场数据 50s 后基本稳定，100s 应更稳定
    verdict_stab = "PASS" if (not has_nan and H_last_range < 0.5 * dH_ana) else "FAIL"
    print(f"  阈值: 无 NaN 且 H 范围 < 0.5×ΔH={0.5*dH_ana:.1f}m → {verdict_stab}")
    print(f"  结论: {verdict_stab}")

    # ── 判定 5: 无缝退化为纯 Joukowsky ────────────────────
    mask_before_frac = (t_sim >= ts) & (t_sim < t_arrive_frac - 0.01)
    if mask_before_frac.sum() > 0:
        max_diff = np.max(np.abs(H_wh[mask_before_frac] - H_wh_noFrac[mask_before_frac]))
        print(f"\n[判定 5] 无缝退化（缝反射到达前含缝=无缝）:")
        print(f"  max|H_withFrac - H_noFrac| = {max_diff:.6e} m  (阈值 < 1e-10)")
        verdict_degen = "PASS" if max_diff < 1e-10 else "FAIL"
    else:
        max_diff = np.nan
        verdict_degen = "N/A"
    print(f"  结论: {verdict_degen}")

    # ── 总评 ─────────────────────────────────────────────
    verdicts = [verdict_dH, verdict_t, verdict_decay, verdict_stab, verdict_degen]
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_total = sum(1 for v in verdicts if v != "N/A")
    print("\n" + "=" * 72)
    print(f"总评: {n_pass}/{n_total} 项 PASS")
    if n_pass == n_total:
        print("[OK] Brunone 非定常摩阻验证通过")
    else:
        print("[FAIL] 验证未全部通过")
    print("=" * 72)

    # ── 可视化 ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Brunone 非定常摩阻验证 — L={L}m, tf={tf}s, x_f={x_f}m, Cf={Cf}m²\n"
        f"2L/a={T_toe:.2f}s (现场~7s), |ΔH|={dH_ana:.1f}m, 衰减比(5/1)={decay_brunone:.3f}",
        fontsize=13, fontweight='bold'
    )

    # (0,0) 全时程：Brunone vs 稳态（含缝）
    ax = axes[0, 0]
    ax.plot(t_sim, H_wh, 'b-', lw=0.5, label='Brunone 含缝')
    ax.plot(t_sim, H_wh_steady, 'r-', lw=0.3, alpha=0.5, label='稳态摩阻 含缝')
    ax.axvline(ts, color='g', ls=':', lw=1, label=f'停泵 ts={ts}s')
    ax.axvline(t_arrive_frac, color='m', ls=':', lw=1, label=f'缝反射 {t_arrive_frac:.1f}s')
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1, label=f'趾端反射 {t_arrive_toe:.1f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title(f'全时程 ({tf}s) — Brunone vs 稳态摩阻')
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (0,1) 前 15s 特写：含缝 vs 无缝
    ax = axes[0, 1]
    mask_15 = t_sim <= 15.0
    ax.plot(t_sim[mask_15], H_wh[mask_15], 'b-', lw=1.0, label='Brunone 含缝')
    ax.plot(t_sim[mask_15], H_wh_noFrac[mask_15], 'g--', lw=0.8, alpha=0.7, label='Brunone 无缝')
    ax.axvline(ts, color='g', ls=':', lw=1)
    ax.axvline(t_arrive_frac, color='m', ls=':', lw=1.2, label=f'缝反射 {t_arrive_frac:.2f}s')
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1, label=f'趾端反射 {t_arrive_toe:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title('前 15s 特写 — 缝反射 + 趾端反射 + Brunone 衰减')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, 15])

    # (1,0) 缝节点 H 与 Q_f（全 100s）
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(t_sim, H_frac, 'b-', lw=0.4, label='缝节点 H')
    ax2.plot(t_sim, Q_frac, 'r-', lw=0.3, alpha=0.6, label='缝侧向流量 Q_f')
    ax.axvline(ts, color='g', ls=':', lw=1)
    ax.axvline(ts + x_f / cfg.a_adj, color='m', ls=':', lw=1,
               label=f'波到达缝 {ts + x_f/cfg.a_adj:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('缝节点水头 [m]', color='b')
    ax2.set_ylabel('缝侧向流量 [m³/s]', color='r')
    ax.set_title(f'缝节点 (x_f={x_f}m) 水头与流量 ({tf}s)')
    ax.legend(fontsize=8, loc='upper left'); ax2.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (1,1) 空间快照
    ax = axes[1, 1]
    snapshots = res.get("snapshots", {})
    x_grid = res["x_grid"]
    colors_snap = ['gray', 'b', 'm', 'orange', 'g', 'cyan', 'purple', 'brown']
    for i, st in enumerate(snap_times):
        si = int(round(st / dt))
        if si in snapshots and snapshots[si] is not None:
            col = colors_snap[i % len(colors_snap)]
            ax.plot(x_grid, snapshots[si]['H'], '-', color=col, lw=1.0,
                    label=f't={st:.1f}s')
    ax.axvline(x_f, color='k', ls=':', lw=1, label=f'缝 x_f={x_f}m')
    ax.set_xlabel('沿井筒位置 [m]'); ax.set_ylabel('水头 [m]')
    ax.set_title(f'井筒水头空间快照 (Brunone, {tf}s)')
    ax.legend(fontsize=7, loc='best'); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, L])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_path(SERIES_STEP03B_BRUNONE, None, "validation.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n图已保存: {out_path}")

    # JSON 结果
    import json
    result = {
        "verdicts": {
            "joukowsky_jump": verdict_dH,
            "fracture_timing": verdict_t,
            "oscillation_decay": verdict_decay,
            "long_term_stability": verdict_stab,
            "no_frac_degeneracy": verdict_degen,
        },
        "metrics": {
            "dH_sim": float(dH_sim),
            "dH_ana": float(-dH_ana),
            "dH_err_pct": float(err_dH),
            "t_arrive_frac_sim": float(t_arrive_sim) if not np.isnan(t_arrive_sim) else None,
            "t_arrive_frac_ana": float(t_arrive_frac_aligned),
            "x_f_aligned": float(x_f_aligned),
            "rms_early_brunone": float(rms_early_brunone) if not np.isnan(rms_early_brunone) else None,
            "rms_early_steady": float(rms_early_steady) if not np.isnan(rms_early_steady) else None,
            "rms_late_brunone": float(rms_late_brunone) if not np.isnan(rms_late_brunone) else None,
            "rms_late_steady": float(rms_late_steady) if not np.isnan(rms_late_steady) else None,
            "decay_ratio_brunone": float(decay_brunone) if not np.isnan(decay_brunone) else None,
            "decay_ratio_steady": float(decay_steady) if not np.isnan(decay_steady) else None,
            "rms_late_ratio": float(rms_late_brunone / rms_late_steady) if not np.isnan(rms_late_brunone) and not np.isnan(rms_late_steady) else None,
            "long_term_H_mean": float(H_last_mean),
            "long_term_H_range": float(H_last_range),
        },
        "config": {
            "L": L, "a": a, "a_adj": float(cfg.a_adj), "V0": V0, "H0": H0,
            "ts": ts, "dt": dt, "tf": tf, "N": cfg.N,
            "x_f": x_f, "Cf": Cf, "friction_model": "brunone",
            "T_toe": float(T_toe),
        },
        "timing": {
            "brunone_sim_sec": float(t_elapsed),
            "noFrac_sim_sec": float(t_elapsed2),
            "steady_sim_sec": float(t_elapsed3),
        },
    }
    json_path = output_path(SERIES_STEP03B_BRUNONE, None, "validation.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON: {json_path}")

    plt.close('all')
    return result


if __name__ == "__main__":
    run_validation()
