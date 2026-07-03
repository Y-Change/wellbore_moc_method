# -*- coding: utf-8 -*-
"""
测试 B — 含单缝 + 稳态摩阻（无 Brunone 干扰）

目的
----
验证裂缝 ODE 求解器（Newton 迭代 + 半隐式格式）在没有 Brunone 干扰时
是否绝对可靠：指数衰减、无超调、100s 稳定不发散。

配置
----
- L=5000m, a=1450 m/s, V0=1.0 m/s, ts=1.0s, tf=100s
- 单缝 x_f=4500m, Cf=1e-4 m², k_leak=2e-4 m²/s/√m, H_ext=200m
- friction_model='steady'（稳态达西，无非定常项）
- 趾端=水库 H=300m

预期
----
- t≈7.2s 处缝反射到达井口，呈衰减（滤失使衰减快于纯柔度）
- Q_f 含稳态正分量（滤失流量），瞬态叠加在其上
- 无超调（overshoot）、无数值振荡
- 100s 内稳定不发散

运行
----
    python dataset_builder/validate_moc_test_b.py
"""
import os, sys, time as time_module
import numpy as np
import matplotlib.pyplot as plt

_METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
if _METHOD_DIR not in sys.path:
    sys.path.insert(0, _METHOD_DIR)

from paths import moc_output_dir
from wellbore_moc import MocConfig, simulate_wellbore, G
from cepstrum_mocdata import plot_moc_cepstrum_analysis

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def osc_rms(H_arr, t_arr, t_start, t_end):
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    if mask.sum() < 10:
        return np.nan
    H_win = H_arr[mask]
    return float(np.sqrt(np.mean((H_win - np.mean(H_win)) ** 2)))


def run_validation():
    print("=" * 72)
    print("测试 B — 含单缝 + 稳态摩阻（无 Brunone 干扰）")
    print("=" * 72)

    L = 5000.0; a = 1450.0; V0 = 1.0; H0 = 300.0
    ts = 1.0; dt = 1.0e-3; tf = 100.0
    x_f = 4500.0; Cf = 1.0e-5
    kleak = 0.0001         # 滤失系数 [m²/s/√m]
    H_ext = 100.0           # 地层孔隙压力水头 [m]

    cfg = MocConfig(
        wellbore_length=L, wellbore_diameter=0.1397,
        fluid_density=1000.0, fluid_viscosity=1.0e-6,
        wavespeed=a, roughness_height=4.5e-5,
        friction_model="steady",       # ★ 稳态摩阻，无 Brunone
        dt=dt, tf=tf,
        wellhead_bc="velocity_step", pump_shut_time=ts,
        initial_velocity=V0, initial_head=H0,
        theta=0.0, toe_bc="reservoir", toe_head=H0,
    )

    dH_ana = cfg.a_adj * V0 / G
    T_toe = 2.0 * L / cfg.a_adj
    t_arrive_toe = ts + T_toe

    print(f"\n物理参数:")
    print(f"  L={L}m, a_adj={cfg.a_adj:.4f} m/s, V0={V0} m/s, ts={ts} s")
    print(f"  单缝 x_f={x_f}m, Cf={Cf} m², k_leak={kleak} m²/s/√m, H_ext={H_ext}m")
    print(f"  摩阻: steady（稳态达西，无 Brunone）")
    print(f"  ΔH = aV0/g = {dH_ana:.4f} m")
    print(f"  趾端反射周期 2L/a = {T_toe:.4f} s")
    print(f"  dx={cfg.dx:.4f} m, N={cfg.N}, n_steps={cfg.n_steps}")

    snap_times = [0.0, ts, 5.0, 7.0, 8.0, 10.0, 30.0, 50.0, 100.0]

    # ── 含缝仿真（含滤失）────────────────────────────────
    print(f"\n运行含缝仿真 (steady, kleak={kleak}, tf={tf}s)...")
    t0 = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=[x_f],
        fracture_Cf=[Cf],
        fracture_kleak=[kleak],
        H_ext=H_ext,
        store_full_field=False,
        snapshot_times=snap_times,
    )
    print(f"  耗时: {time_module.time()-t0:.1f}s")

    # ── 含缝对照（纯柔度，无滤失）────────────────────────
    print(f"运行纯柔度对照 (steady, kleak=0)...")
    t0 = time_module.time()
    res_pure = simulate_wellbore(
        cfg,
        fracture_positions=[x_f],
        fracture_Cf=[Cf],
        fracture_kleak=[0.0],
        H_ext=H_ext,
        store_full_field=False,
    )
    print(f"  耗时: {time_module.time()-t0:.1f}s")

    t_sim = res["timestamps"]
    H_wh = res["wellhead_head"]
    H_frac = res["fracture_heads"][:, 0]
    Q_frac = res["fracture_Qs"][:, 0]
    frac_idx = res["fracture_indices"][0]
    x_f_aligned = res["x_grid"][frac_idx]
    t_arrive_frac = ts + 2.0 * x_f_aligned / cfg.a_adj
    H_wh_pure = res_pure["wellhead_head"]          # 纯柔度对照
    Q_frac_pure = res_pure["fracture_Qs"][:, 0]

    # ── 无缝对照 ─────────────────────────────────────────
    print(f"运行无缝对照 (steady)...")
    t0 = time_module.time()
    res_noFrac = simulate_wellbore(cfg, store_full_field=False)
    print(f"  耗时: {time_module.time()-t0:.1f}s")
    H_wh_noFrac = res_noFrac["wellhead_head"]

    ts_idx = int(round(ts / dt))
    H0_actual = H_wh[ts_idx - 1]
    baseline = H0_actual - dH_ana
    diff_signal = H_wh - H_wh_noFrac

    print(f"\n  缝对齐: x_f={x_f}m → 网格 x={x_f_aligned:.2f}m (idx={frac_idx})")
    print(f"  缝反射到达 t = {t_arrive_frac:.4f} s")
    print(f"  趾端反射到达 t = {t_arrive_toe:.4f} s")
    print(f"  稳态井口水头 H0_actual = {H0_actual:.4f} m")

    # ── 判定 1: Joukowsky 跳变 ────────────────────────────
    dH_sim = H_wh[ts_idx] - H_wh[ts_idx - 1]
    err_dH = abs(dH_sim - (-dH_ana)) / dH_ana * 100
    print(f"\n[判定 1] Joukowsky 初始跳变:")
    print(f"  仿真 ΔH = {dH_sim:+.4f} m, 解析 = {-dH_ana:+.4f} m, 误差 = {err_dH:.4f}%")
    verdict_dH = "PASS" if err_dH < 0.1 else "FAIL"
    print(f"  结论: {verdict_dH}")

    # ── 判定 2: 缝反射到达（差信号 step）──────────────────
    half_win = 0.15
    mask_b = (t_sim >= t_arrive_frac - half_win) & (t_sim < t_arrive_frac - 0.02)
    mask_a = (t_sim > t_arrive_frac + 0.02) & (t_sim <= t_arrive_frac + half_win)
    diff_step = np.mean(diff_signal[mask_a]) - np.mean(diff_signal[mask_b])
    print(f"\n[判定 2] 缝反射（差信号 step）:")
    print(f"  t={t_arrive_frac:.4f}s, 差信号step = {diff_step:+.2f}m (|step|>5m → 存在)")
    verdict_frac = "PASS" if abs(diff_step) > 5.0 else "FAIL"
    print(f"  结论: {verdict_frac}")

    # ── 判定 3: 滤失阻尼（含滤失衰减快于纯柔度）──────────
    # 滤失项 k_leak·√(H-H_ext) 引入额外阻尼，井口振荡 RMS 应小于纯柔度对照
    Z_w = cfg.a_adj / (G * cfg.area)
    tau_ana_pure = Cf * Z_w / 2.0   # 纯柔度解析衰减时间（参考）
    win_lo, win_hi = t_arrive_frac + 0.5, t_arrive_toe - 0.1
    rms_leak = osc_rms(H_wh, t_sim, win_lo, win_hi)
    rms_pure = osc_rms(H_wh_pure, t_sim, win_lo, win_hi)
    damping_ratio = rms_pure / rms_leak if rms_leak > 0 else np.nan
    print(f"\n[判定 3] 滤失阻尼 (窗口 {win_lo:.2f}-{win_hi:.2f}s):")
    print(f"  RMS(含滤失) = {rms_leak:.4f} m, RMS(纯柔度) = {rms_pure:.4f} m")
    print(f"  阻尼比 RMS_pure/RMS_leak = {damping_ratio:.3f} (>1.05 → 滤失增阻尼)")
    verdict_damp = "PASS" if (not np.isnan(damping_ratio) and damping_ratio > 1.05) else "FAIL"
    print(f"  纯柔度参考 τ = {tau_ana_pure:.4f} s")
    print(f"  结论: {verdict_damp}")

    # ── 判定 3b: 滤失稳态分量（Q_f 含正稳态流量）─────────
    # 稳态时 dH/dt=0 → Q_f = k_leak·√(H - H_ext) > 0（只要 H > H_ext）
    Q_last_mask = (t_sim >= tf - 5.0)
    Q_last = Q_frac[Q_last_mask]
    Q_steady = float(np.mean(Q_last))
    Q_steady_ana = kleak * np.sqrt(max(H0 - H_ext, 0.0))   # 粗略稳态估计
    print(f"\n[判定 3b] 滤失稳态分量 (最后 5s):")
    print(f"  mean(Q_f) = {Q_steady:.6f} m³/s, 估计 ≈ k_leak·√(H0-H_ext) = {Q_steady_ana:.6f}")
    verdict_qss = "PASS" if (Q_steady > 0 and Q_steady > 0.3 * Q_steady_ana) else "FAIL"
    print(f"  阈值: mean(Q_f)>0 且 > 0.3·估计 → {verdict_qss}")
    # 占位变量，保持后续兼容
    tau_fit = np.nan; r2 = 0.0; err_tau = np.nan

    # ── 判定 4: 无超调/无振荡 ─────────────────────────────
    # 稳态摩阻不产生 Brunone 锯齿，波形应平滑
    # 检查：在缝反射到达后的 0.5s 窗口内，H 的逐时步变化 |dH/dt| 不应有高频抖动
    smooth_mask = (t_sim >= t_arrive_frac) & (t_sim < t_arrive_frac + 0.5)
    if smooth_mask.sum() > 10:
        dHdt_smooth = np.abs(np.diff(H_wh[smooth_mask]) / dt)
        # 高频抖动判定：dH/dt 的标准差 / 均值 > 5 表示有锯齿
        mean_dHdt = np.mean(dHdt_smooth)
        std_dHdt = np.std(dHdt_smooth)
        jitter_ratio = std_dHdt / mean_dHdt if mean_dHdt > 0 else 0
        print(f"\n[判定 4] 波形平滑性（缝反射后 0.5s 窗口）:")
        print(f"  |dH/dt| 均值={mean_dHdt:.2f}, 标准差={std_dHdt:.2f}, 抖动比={jitter_ratio:.4f}")
        print(f"  阈值: 抖动比 < 2.0（稳态摩阻应无锯齿）")
        verdict_smooth = "PASS" if jitter_ratio < 2.0 else "FAIL"
    else:
        verdict_smooth = "N/A"
        jitter_ratio = np.nan
    print(f"  结论: {verdict_smooth}")

    # ── 判定 5: 长期稳定性（100s 不发散）─────────────────
    last_5s = t_sim >= (tf - 5.0)
    H_last = H_wh[last_5s]
    H_last_range = float(np.max(H_last) - np.min(H_last))
    has_nan = bool(np.any(np.isnan(H_last)) or np.any(np.isinf(H_last)))
    print(f"\n[判定 5] 长期稳定性 (最后 5s):")
    print(f"  H范围={H_last_range:.2f}m, NaN={has_nan}")
    verdict_stab = "PASS" if (not has_nan and H_last_range < dH_ana) else "FAIL"
    print(f"  阈值: 无NaN 且 H范围 < ΔH={dH_ana:.1f}m → {verdict_stab}")

    # ── 判定 6: 停泵前滤失稳态（Q_f 早段接近估计值）─────
    # 滤失从 t=0 起作用，停泵前缝节点应已建立稳态滤失流量
    mask_pre = (t_sim >= 0.5 * ts) & (t_sim < ts - 0.01)
    Q_pre = float(np.mean(Q_frac[mask_pre]))
    ratio_Qpre = Q_pre / Q_steady_ana if Q_steady_ana > 0 else 0.0
    print(f"\n[判定 6] 停泵前滤失稳态 (t∈[0.5·ts, ts]):")
    print(f"  mean(Q_f) = {Q_pre:.6f} m³/s, 估计 = {Q_steady_ana:.6f}, 比值 = {ratio_Qpre:.3f}")
    verdict_degen = "PASS" if (Q_pre > 0 and 0.3 < ratio_Qpre < 2.0) else "FAIL"
    print(f"  阈值: Q_f>0 且比值∈(0.3, 2.0) → {verdict_degen}")

    # ── 总评 ─────────────────────────────────────────────
    verdicts = [verdict_dH, verdict_frac, verdict_damp, verdict_qss,
                verdict_smooth, verdict_stab, verdict_degen]
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_total = sum(1 for v in verdicts if v != "N/A")
    print("\n" + "=" * 72)
    print(f"总评: {n_pass}/{n_total} 项 PASS")
    if n_pass == n_total:
        print("[OK] 测试 B（含滤失）通过 — 含缝+稳态摩阻+滤失可靠")
    else:
        print("[FAIL] 验证未全部通过")
    print("=" * 72)

    # ── 可视化 ────────────────────────────────────────────
    out_dir = moc_output_dir()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"测试 B — 含单缝 + 稳态摩阻 + 滤失\n"
        f"L={L}m, x_f={x_f}m, Cf={Cf}m², k_leak={kleak}, H_ext={H_ext}m, tf={tf}s\n"
        f"阻尼比={damping_ratio:.3f}, Q_steady={Q_steady:.5f} m³/s",
        fontsize=13, fontweight='bold'
    )

    # (0,0) 全时程：含滤失 vs 纯柔度 vs 无缝
    ax = axes[0, 0]
    ax.plot(t_sim, H_wh, 'b-', label='含缝+滤失 steady')
    ax.plot(t_sim, H_wh_pure, 'r-', label='纯柔度(无滤失)')
    ax.plot(t_sim, H_wh_noFrac, 'k--', label='无缝 steady')
    ax.axvline(ts, color='g', ls=':', lw=1, label=f'停泵 ts={ts}s')
    ax.axvline(t_arrive_frac, color='m', ls=':', lw=1, label=f'缝反射 {t_arrive_frac:.2f}s')
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1, label=f'趾端 {t_arrive_toe:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title(f'全时程 ({tf}s) — 含缝 vs 无缝 (steady)')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (0,1) 前 12s 特写：含滤失 vs 纯柔度
    ax = axes[0, 1]
    mask_12 = t_sim <= 12.0
    ax.plot(t_sim[mask_12], H_wh[mask_12], 'b-',  label='含缝+滤失')
    ax.plot(t_sim[mask_12], H_wh_pure[mask_12], 'r-',  label='纯柔度(无滤失)')
    ax.plot(t_sim[mask_12], H_wh_noFrac[mask_12], 'k--', label='无缝')
    ax.axvline(ts, color='g', ls=':', lw=1)
    ax.axvline(t_arrive_frac, color='m', ls=':', lw=1.2, label=f'缝反射 {t_arrive_frac:.2f}s')
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1, label=f'趾端 {t_arrive_toe:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title('前 12s 特写 — 滤失增阻尼（衰减更快）')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, 12])

    # (1,0) 差信号：含滤失 vs 纯柔度
    ax = axes[1, 0]
    mask_diff = (t_sim >= t_arrive_frac - 0.5) & (t_sim <= t_arrive_toe + 0.5)
    diff_pure = H_wh_pure - H_wh_noFrac
    ax.plot(t_sim[mask_diff], diff_signal[mask_diff], 'b-', lw=1.0, label='差信号(含滤失)')
    ax.plot(t_sim[mask_diff], diff_pure[mask_diff], 'c--', lw=0.8, alpha=0.7, label='差信号(纯柔度)')
    ax.axvline(t_arrive_frac, color='m', ls=':', lw=1)
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1)
    ax.axhline(0, color='k', lw=0.3)
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('差信号 [m]')
    ax.set_title(f'差信号 — 滤失阻尼比={damping_ratio:.3f}')
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.6)

    # (1,1) 缝节点 H 与 Q_f（含滤失 vs 纯柔度）
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(t_sim, H_frac, 'b-', lw=0.5, label='缝节点 H')
    ax2.plot(t_sim, Q_frac, 'r-', lw=0.3, alpha=0.7, label='缝 Q_f (含滤失)')
    ax2.plot(t_sim, Q_frac_pure, 'm:', lw=0.3, alpha=0.5, label='缝 Q_f (纯柔度)')
    ax.axvline(ts, color='g', ls=':', lw=1)
    ax.axvline(ts + x_f_aligned / cfg.a_adj, color='m', ls=':', lw=1,
               label=f'波到达缝 {ts + x_f_aligned/cfg.a_adj:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('缝节点水头 [m]')
    ax2.set_ylabel('缝侧向流量 [m³/s]')
    ax.set_title(f'缝节点 (x={x_f_aligned:.0f}m) H 与 Q_f (Q_steady={Q_steady:.5f})')
    ax.legend(fontsize=8, loc='upper left'); ax2.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(out_dir, "test_b_fracture_leakoff.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n图已保存: {out_path}")
    plt.close(fig)

    # ── 井口水头倒谱合并图（时域 / FFT / 1D + 2D 倒谱）────────
    cep_path = os.path.join(out_dir, "test_b_cepstrum.png")
    plot_moc_cepstrum_analysis(
        t_sim, H_wh,
        wavespeed=cfg.a_adj,
        ts=ts,
        dt=dt,
        wellbore_length=L,
        fracture_positions=[x_f_aligned],
        save_path=cep_path,
        title_prefix=(
            f"测试 B — 井口水头倒谱分析\n"
            f"x_f={x_f_aligned:.0f}m, k_leak={kleak}, steady 摩阻"
        ),
        hop_sec=0.1,
    )

    # JSON
    import json
    result = {
        "verdicts": {
            "joukowsky": verdict_dH,
            "fracture_reflection": verdict_frac,
            "leakoff_damping": verdict_damp,
            "leakoff_steady_Q": verdict_qss,
            "smoothness": verdict_smooth,
            "stability": verdict_stab,
            "degeneracy": verdict_degen,
        },
        "metrics": {
            "dH_err_pct": float(err_dH),
            "diff_step": float(diff_step),
            "rms_leak": float(rms_leak),
            "rms_pure": float(rms_pure),
            "damping_ratio": float(damping_ratio) if not np.isnan(damping_ratio) else None,
            "Q_steady": float(Q_steady),
            "Q_steady_ana": float(Q_steady_ana),
            "jitter_ratio": float(jitter_ratio) if not np.isnan(jitter_ratio) else None,
            "long_term_range": float(H_last_range),
        },
        "config": {
            "L": L, "a": a, "V0": V0, "H0": H0, "ts": ts, "dt": dt, "tf": tf,
            "x_f": x_f, "Cf": Cf, "kleak": kleak, "H_ext": H_ext, "friction": "steady",
        },
    }
    json_path = os.path.join(out_dir, "test_b_fracture_leakoff.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON: {json_path}")

    return result


if __name__ == "__main__":
    run_validation()
