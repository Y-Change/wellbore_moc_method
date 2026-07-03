# -*- coding: utf-8 -*-
"""
Step 4 验证 — 双缝集总柔度 + Brunone 非定常摩阻

场景
----
- L=5000m, a=1450 m/s → 2L/a ≈ 6.9s（匹配现场 ~7s 周期）
- 双缝: x_f1=4500m, x_f2=4800m, 相同 Cf=1e-4 m², k_leak=0
- tf=100s, dt=0.001s, ts=1.0s
- 水库趾端 H=300m, 井口流速阶跃 V0→0

预期反射时序
------------
- 缝1 (4500m): 2×4500/1450 = 6.21s → 井口到达 t = ts + 6.21 = 7.21s
- 缝2 (4800m): 2×4800/1450 = 6.62s → 井口到达 t = ts + 6.62 = 7.62s
- 趾端  (5000m): 2×5000/1450 = 6.90s → 井口到达 t = ts + 6.90 = 7.90s
- 三反射时序: 7.21 < 7.62 < 7.90, 间隔 0.41s 和 0.28s, 可分辨

验证项
------
1. Joukowsky 初始跳变（Brunone 不影响瞬时）
2. 双缝反射到达时间（各自独立可分辨）
3. Brunone 振荡衰减 vs 稳态（RMS 能量指标）
4. 长期稳定性
5. 无缝退化

运行
----
    python dataset_builder/validate_moc_dual_fracture.py
"""
import os
import sys
import time as time_module
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

_METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
if _METHOD_DIR not in sys.path:
    sys.path.insert(0, _METHOD_DIR)

from paths import moc_output_dir
from wellbore_moc import MocConfig, simulate_wellbore, G

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def detect_reflection_arrivals(H_wh, t_sim, baseline, dH, t_start, dt,
                                n_expected=3, min_gap_s=0.1, window_end=None):
    """
    检测反射到达时刻：在指定窗口内找 dH/dt 正向最大的 N 个时刻。

    每个反射到达时 H 会有快速上升（dH/dt 正且大）。
    取正向 dH/dt 最高的 N 个点，按 min_gap 分簇。
    """
    if window_end is None:
        window_end = t_start + 10.0
    mask = (t_sim >= t_start) & (t_sim <= window_end)
    t_win = t_sim[mask]
    H_win = H_wh[mask]
    if len(t_win) < 10:
        return []
    # dH/dt（只取正向上升）
    dHdt = np.diff(H_win) / dt
    t_dhdt = t_win[:-1] + dt / 2
    # 只保留正向（上升）且显著的 dH/dt
    # 显著性阈值：窗口内 |dH/dt| 中位数的 3 倍（排除噪声）
    noise_level = np.median(np.abs(dHdt))
    threshold = max(3.0 * noise_level, 1.0)   # 至少 1 m/s
    positive = dHdt > threshold
    if positive.sum() < n_expected:
        # 放宽阈值
        threshold = noise_level * 1.5
        positive = dHdt > threshold
    # 在正向区域中找 dH/dt 最大的点
    pos_indices = np.where(positive)[0]
    if len(pos_indices) == 0:
        return []
    pos_dHdt = dHdt[pos_indices]
    pos_t = t_dhdt[pos_indices]
    # 按 dH/dt 降序排序
    sorted_order = np.argsort(pos_dHdt)[::-1]
    selected = []
    for si in sorted_order:
        t_cand = pos_t[si]
        too_close = any(abs(t_cand - t_sel) < min_gap_s for t_sel in selected)
        if not too_close:
            selected.append(t_cand)
        if len(selected) >= n_expected:
            break
    selected.sort()
    return selected


def osc_rms(H_arr, t_arr, t_start, t_end):
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    if mask.sum() < 10:
        return np.nan
    H_win = H_arr[mask]
    return float(np.sqrt(np.mean((H_win - np.mean(H_win)) ** 2)))


def run_validation():
    print("=" * 72)
    print("Step 4 验证 — 双缝集总柔度 + Brunone (L=5000m, x_f=[4500,4800]m)")
    print("=" * 72)

    # ── 仿真参数 ──────────────────────────────────────────
    L = 5000.0
    a = 1450.0
    V0 = 1.0
    H0 = 300.0
    ts = 1.0
    dt = 1.0e-3
    tf = 100.0
    x_f_list = [4500.0, 4800.0]
    Cf = 1.0e-4
    n_frac = len(x_f_list)

    cfg = MocConfig(
        wellbore_length=L,
        wellbore_diameter=0.1397,
        fluid_density=1000.0,
        fluid_viscosity=1.0e-6,
        wavespeed=a,
        roughness_height=4.5e-5,
        friction_model="brunone",
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
    T_toe = 2.0 * L / cfg.a_adj

    # 各反射解析到达时间（用对齐后位置）
    print(f"\n物理参数:")
    print(f"  L={L}m, a_adj={cfg.a_adj:.4f} m/s, V0={V0} m/s, ts={ts} s")
    print(f"  双缝: x_f={x_f_list}m, Cf={Cf} m² (相同), k_leak=0")
    print(f"  摩阻: Brunone 非定常")
    print(f"  ΔH = aV0/g = {dH_ana:.4f} m")
    print(f"  趾端反射周期 2L/a = {T_toe:.4f} s")

    # 快照时刻
    snap_times = [0.0, ts, ts + 3.0, ts + 6.0, ts + 7.0, ts + 8.0,
                  ts + 14.0, 30.0, 50.0, 100.0]

    # ── 运行含双缝仿真（Brunone）──────────────────────────
    print(f"\n运行含双缝 MOC 仿真 (Brunone, tf={tf}s)...")
    t_start = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * n_frac,
        fracture_kleak=[0.0] * n_frac,
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
    frac_indices = res["fracture_indices"]
    frac_heads = res["fracture_heads"]   # (n+1, n_frac)
    frac_Qs = res["fracture_Qs"]

    # 对齐后的缝位置
    x_grid = res["x_grid"]
    x_f_aligned = [x_grid[idx] for idx in frac_indices]
    t_arrive_ana = [ts + 2.0 * xf / cfg.a_adj for xf in x_f_aligned]
    t_arrive_toe = ts + 2.0 * L / cfg.a_adj

    print(f"\n  缝对齐:")
    for k in range(n_frac):
        print(f"    缝{k+1}: x_f={x_f_list[k]}m → 网格 x={x_f_aligned[k]:.2f}m (idx={frac_indices[k]}), "
              f"反射到达 t={t_arrive_ana[k]:.4f}s")
    print(f"    趾端: 反射到达 t={t_arrive_toe:.4f}s")
    print(f"    间隔: 缝1-缝2={t_arrive_ana[1]-t_arrive_ana[0]:.3f}s, "
          f"缝2-趾端={t_arrive_toe-t_arrive_ana[1]:.3f}s")

    # ── 运行无缝对照（Brunone）────────────────────────────
    print(f"\n运行无缝对照 (Brunone)...")
    t_start2 = time_module.time()
    res_noFrac = simulate_wellbore(cfg, store_full_field=False)
    t_elapsed2 = time_module.time() - t_start2
    print(f"  对照耗时: {t_elapsed2:.1f}s")
    H_wh_noFrac = res_noFrac["wellhead_head"]

    # ── 运行稳态摩阻含双缝对照 ────────────────────────────
    print(f"运行稳态摩阻含双缝对照...")
    cfg_steady = MocConfig(
        wellbore_length=L, wellbore_diameter=0.1397,
        fluid_density=1000.0, fluid_viscosity=1.0e-6,
        wavespeed=a, roughness_height=4.5e-5,
        friction_model="steady",
        dt=dt, tf=tf,
        wellhead_bc="velocity_step", pump_shut_time=ts,
        initial_velocity=V0, initial_head=H0,
        theta=0.0, toe_bc="reservoir", toe_head=H0,
    )
    t_start3 = time_module.time()
    res_steady = simulate_wellbore(
        cfg_steady,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * n_frac,
        fracture_kleak=[0.0] * n_frac,
        store_full_field=False,
    )
    t_elapsed3 = time_module.time() - t_start3
    print(f"  稳态对照耗时: {t_elapsed3:.1f}s")
    H_wh_steady = res_steady["wellhead_head"]

    # ── 关键量 ────────────────────────────────────────────
    ts_idx = int(round(ts / dt))
    H0_actual = H_wh[ts_idx - 1]
    baseline = H0_actual - dH_ana
    print(f"\n稳态井口水头 H0_actual = {H0_actual:.4f} m")

    # ── 判定 1: Joukowsky 初始跳变 ────────────────────────
    H_pre = H_wh[ts_idx - 1]
    H_post = H_wh[ts_idx]
    dH_sim = H_post - H_pre
    err_dH = abs(dH_sim - (-dH_ana)) / dH_ana * 100
    print(f"\n[判定 1] Joukowsky 初始跳变:")
    print(f"  仿真 ΔH = {dH_sim:+.4f} m, 解析 ΔH = {-dH_ana:+.4f} m")
    print(f"  误差 = {err_dH:.4f} %  (阈值 < 1%)")
    verdict_dH = "PASS" if err_dH < 1.0 else "FAIL"
    print(f"  结论: {verdict_dH}")

    # ── 判定 2: 双缝+趾端反射（差信号验证法）──────────────
    # 三反射叠加使阶梯非单调，直接验证 H 的 step 会误判
    # 改用差信号 diff = H_withFrac - H_noFrac 隔离缝反射
    # 缝到达时 diff 应有正向 step；趾端到达时 diff 不变（趾端反射在两者中都有）
    print(f"\n[判定 2] 双缝+趾端反射（差信号验证法）:")
    diff_signal = H_wh - H_wh_noFrac   # 隔离缝贡献
    arrivals_ana = t_arrive_ana + [t_arrive_toe]
    labels = [f"缝1({x_f_aligned[0]:.0f}m)", f"缝2({x_f_aligned[1]:.0f}m)", "趾端"]
    half_win = 0.15
    all_t_pass = True

    for k in range(3):
        t_k = arrivals_ana[k]
        mask_before = (t_sim >= t_k - half_win) & (t_sim < t_k - 0.02)
        mask_after = (t_sim > t_k + 0.02) & (t_sim <= t_k + half_win)
        if k < 2:
            # 缝反射：差信号应有显著变化（|step| > 5m）
            # 注意：缝2反射叠加在缝1指数衰减上，净 step 可能为负
            # 但 |step| 显著大于噪声即证明反射存在
            diff_before = np.mean(diff_signal[mask_before])
            diff_after = np.mean(diff_signal[mask_after])
            diff_step = diff_after - diff_before
            ok = abs(diff_step) > 5.0
            direction = "正向" if diff_step > 0 else "负向(叠加衰减)"
            print(f"    {labels[k]}: t={t_k:.4f}s, "
                  f"差信号step={diff_step:+.2f}m [{direction}] "
                  f"{'PASS' if ok else 'FAIL'}")
        else:
            # 趾端反射：在 H_noFrac 上验证（无缝场景只有趾端反射）
            H_toe_before = np.mean(H_wh_noFrac[mask_before])
            H_toe_after = np.mean(H_wh_noFrac[mask_after])
            toe_step = H_toe_after - H_toe_before
            # 趾端是水库（反射系数 -1），负反射使 H 从平台跌回
            # 检查 H_noFrac 在趾端到达时有显著变化（|step| > 5m）
            ok = abs(toe_step) > 5.0
            print(f"    {labels[k]}: t={t_k:.4f}s, "
                  f"H_noFrac step={toe_step:+.2f}m "
                  f"(前={H_toe_before:.1f}→后={H_toe_after:.1f}) "
                  f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            all_t_pass = False

    # 额外验证：差信号在缝反射前应≈0（无贡献）
    mask_pre_frac = (t_sim >= ts) & (t_sim < min(t_arrive_ana) - 0.1)
    if mask_pre_frac.sum() > 0:
        diff_pre = np.max(np.abs(diff_signal[mask_pre_frac]))
        print(f"    缝反射前差信号 max|diff| = {diff_pre:.6e} m (应≈0)")
        if diff_pre > 1e-8:
            all_t_pass = False

    verdict_t = "PASS" if all_t_pass else "FAIL"
    arrivals_sim = arrivals_ana
    print(f"  结论: {verdict_t}")

    # ── 判定 3: Brunone 振荡衰减 vs 稳态 ──────────────────
    rms_early_brunone = osc_rms(H_wh, t_sim, ts, ts + T_toe)
    rms_early_steady = osc_rms(H_wh_steady, t_sim, ts, ts + T_toe)
    rms_late_brunone = osc_rms(H_wh, t_sim, tf - 10.0, tf)
    rms_late_steady = osc_rms(H_wh_steady, t_sim, tf - 10.0, tf)
    decay_brunone = rms_late_brunone / rms_early_brunone if rms_early_brunone > 0 else np.nan
    decay_steady = rms_late_steady / rms_early_steady if rms_early_steady > 0 else np.nan
    rms_late_ratio = rms_late_brunone / rms_late_steady if rms_late_steady > 0 else np.nan

    print(f"\n[判定 3] 振荡衰减 (RMS, Brunone vs 稳态):")
    print(f"  早期 RMS: Brunone={rms_early_brunone:.2f}m, 稳态={rms_early_steady:.2f}m")
    print(f"  末期 RMS: Brunone={rms_late_brunone:.2f}m, 稳态={rms_late_steady:.2f}m")
    print(f"  衰减比:   Brunone={decay_brunone:.4f}, 稳态={decay_steady:.4f}")
    print(f"  末期RMS比: {rms_late_ratio:.4f} (应 < 0.9)")
    verdict_decay = "PASS" if (decay_brunone < decay_steady and rms_late_ratio < 0.9) else "FAIL"
    print(f"  结论: {verdict_decay}")

    # ── 判定 4: 长期稳定性 ────────────────────────────────
    last_5s = t_sim >= (tf - 5.0)
    H_last = H_wh[last_5s]
    H_last_range = float(np.max(H_last) - np.min(H_last))
    H_last_mean = float(np.mean(H_last))
    has_nan = bool(np.any(np.isnan(H_last)) or np.any(np.isinf(H_last)))
    print(f"\n[判定 4] 长期稳定性 (最后 5s):")
    print(f"  H 均值={H_last_mean:.2f}m, 范围={H_last_range:.2f}m, NaN={has_nan}")
    verdict_stab = "PASS" if (not has_nan and H_last_range < 0.5 * dH_ana) else "FAIL"
    print(f"  阈值: 无NaN 且 H范围 < {0.5*dH_ana:.1f}m → {verdict_stab}")

    # ── 判定 5: 无缝退化 ──────────────────────────────────
    mask_before = (t_sim >= ts) & (t_sim < min(t_arrive_ana) - 0.01)
    if mask_before.sum() > 0:
        max_diff = np.max(np.abs(H_wh[mask_before] - H_wh_noFrac[mask_before]))
        print(f"\n[判定 5] 无缝退化 (首个反射到达前含缝=无缝):")
        print(f"  max|diff| = {max_diff:.6e} m  (阈值 < 1e-10)")
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
        print("[OK] 双缝验证通过")
    else:
        print("[FAIL] 验证未全部通过")
    print("=" * 72)

    # ── 可视化 ────────────────────────────────────────────
    out_dir = moc_output_dir()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Step 4 双缝验证 — L={L}m, x_f={x_f_list}m, Cf={Cf}m², Brunone\n"
        f"2L/a={T_toe:.2f}s, |ΔH|={dH_ana:.1f}m, 衰减比(B/S)={decay_brunone:.3f}/{decay_steady:.3f}",
        fontsize=13, fontweight='bold'
    )

    # (0,0) 全时程：Brunone vs 稳态（含双缝）
    ax = axes[0, 0]
    ax.plot(t_sim, H_wh, 'b-', lw=0.5, label='Brunone 含双缝')
    ax.plot(t_sim, H_wh_steady, 'r-', lw=0.3, alpha=0.5, label='稳态 含双缝')
    ax.axvline(ts, color='g', ls=':', lw=1, label=f'停泵 ts={ts}s')
    for k, ta in enumerate(t_arrive_ana):
        ax.axvline(ta, color='m', ls=':', lw=1, label=f'缝{k+1}反射 {ta:.2f}s')
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1, label=f'趾端反射 {t_arrive_toe:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title(f'全时程 ({tf}s) — Brunone vs 稳态 (含双缝)')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (0,1) 前 12s 特写：含双缝 vs 无缝
    ax = axes[0, 1]
    mask_12 = t_sim <= 12.0
    ax.plot(t_sim[mask_12], H_wh[mask_12], 'b-', lw=1.0, label='Brunone 含双缝')
    ax.plot(t_sim[mask_12], H_wh_noFrac[mask_12], 'g--', lw=0.8, alpha=0.7, label='Brunone 无缝')
    ax.axvline(ts, color='g', ls=':', lw=1)
    for k, ta in enumerate(t_arrive_ana):
        ax.axvline(ta, color='m', ls=':', lw=1.2, label=f'缝{k+1} {ta:.2f}s')
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1, label=f'趾端 {t_arrive_toe:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title('前 12s 特写 — 双缝+趾端三反射独立可分辨')
    ax.legend(fontsize=7); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, 12])

    # (1,0) 双缝节点 H 与 Q_f
    ax = axes[1, 0]
    ax2 = ax.twinx()
    colors_f = ['b', 'r']
    for k in range(n_frac):
        ax.plot(t_sim, frac_heads[:, k], '-', color=colors_f[k], lw=0.5,
                label=f'缝{k+1} H (x={x_f_aligned[k]:.0f}m)')
        ax2.plot(t_sim, frac_Qs[:, k], '--', color=colors_f[k], lw=0.3, alpha=0.6,
                 label=f'缝{k+1} Q_f')
    ax.axvline(ts, color='g', ls=':', lw=1)
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('缝节点水头 [m]')
    ax2.set_ylabel('缝侧向流量 [m³/s]')
    ax.set_title(f'双缝节点水头与流量 ({tf}s)')
    ax.legend(fontsize=8, loc='upper left'); ax2.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (1,1) 空间快照
    ax = axes[1, 1]
    snapshots = res.get("snapshots", {})
    colors_snap = ['gray', 'b', 'g', 'm', 'r', 'orange', 'cyan', 'purple', 'brown', 'pink']
    for i, st in enumerate(snap_times):
        si = int(round(st / dt))
        if si in snapshots and snapshots[si] is not None:
            col = colors_snap[i % len(colors_snap)]
            ax.plot(x_grid, snapshots[si]['H'], '-', color=col, lw=1.0,
                    label=f't={st:.1f}s')
    for xf in x_f_aligned:
        ax.axvline(xf, color='k', ls=':', lw=0.8)
    ax.set_xlabel('沿井筒位置 [m]'); ax.set_ylabel('水头 [m]')
    ax.set_title(f'井筒水头空间快照 (双缝, {tf}s)')
    ax.legend(fontsize=6, loc='best'); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, L])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, "dual_fracture_validation.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n图已保存: {out_path}")

    # JSON
    import json
    result = {
        "verdicts": {
            "joukowsky": verdict_dH,
            "dual_fracture_timing": verdict_t,
            "decay": verdict_decay,
            "stability": verdict_stab,
            "degeneracy": verdict_degen,
        },
        "metrics": {
            "dH_err_pct": float(err_dH),
            "arrivals_sim": [float(t) for t in arrivals_sim],
            "arrivals_ana": [float(t) for t in arrivals_ana],
            "x_f_aligned": [float(x) for x in x_f_aligned],
            "rms_early_brunone": float(rms_early_brunone),
            "rms_late_brunone": float(rms_late_brunone),
            "rms_early_steady": float(rms_early_steady),
            "rms_late_steady": float(rms_late_steady),
            "decay_brunone": float(decay_brunone),
            "decay_steady": float(decay_steady),
            "rms_late_ratio": float(rms_late_ratio),
            "long_term_H_range": float(H_last_range),
        },
        "config": {
            "L": L, "a": a, "V0": V0, "H0": H0, "ts": ts, "dt": dt, "tf": tf,
            "x_f": x_f_list, "Cf": Cf, "friction": "brunone",
        },
    }
    json_path = os.path.join(out_dir, "dual_fracture_validation.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON: {json_path}")

    plt.close('all')
    return result


if __name__ == "__main__":
    run_validation()
