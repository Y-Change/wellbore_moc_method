# -*- coding: utf-8 -*-
"""
experiments/moc_physics_upgrade/step1_steady_state_toe/run_quad_validation.py

运行 Step 1 修复后的 MOC 模型在 PaperA brunone_D20/quad 上的仿真验证：
1. 采用 quad 裂缝位置: x_f = [4100, 4120, 4140, 4160] m
2. 运行 100s 长时仿真 (Brunone 摩阻, dt=1ms)
3. 与 PaperA 原始 quad 结果进行波形叠合比对，展示 t=0 虚假激波消除效果
4. 生成标准时程 CSV、2x2 验证图、5 联倒谱图、裂缝特写图与 JSON 指标
输出目录: output/moc_physics_upgrade/step1_steady_state_toe/quad/
"""
from __future__ import annotations

import json
import os
import sys
import time as time_module

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# 确保项目根目录在 sys.path
_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'README.md')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find project root')
    _d = _parent

from experiments.moc_physics_upgrade.step1_steady_state_toe.solver import (
    UpgradedMocConfig, simulate_wellbore_upgraded, G
)
from moc_simulate.cepstrum_mocdata import (
    plot_moc_cepstrum_analysis,
    plot_moc_cepstrum_fracture_zoom,
    evaluate_1d_cepstrum_fracture_match,
    cepstrum_match_summary_for_json,
)
from moc_simulate.config import CEPSTRUM_CONFIG

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
FRAC_COLORS = ['b', 'r', 'g', 'm', 'c', 'orange', 'brown', 'olive']


def main():
    print("=" * 76)
    print("【Step 1 升级验证】PaperA brunone_D20/quad 裂缝结构水击仿真")
    print("修复内容: 物理自洽的稳态流场初始化 + 多簇分流 + 沿程摩阻平衡 + 消除 t=0 趾端激波")
    print("=" * 76)

    # 1. 配置参数 (与 PaperA quad 1:1 对齐)
    L = 5000.0
    a = 1450.0
    V0 = 1.0
    H0 = 300.0
    ts = 1.0
    dt = 0.001
    tf = 100.0
    x_f_list = [4100.0, 4120.0, 4140.0, 4160.0]
    Cf = 1.0e-5
    kleak = 0.0001
    H_ext = 100.0
    n_frac = len(x_f_list)

    cfg = UpgradedMocConfig(
        wellbore_length=L,
        wellbore_diameter=0.1397,
        fluid_density=1000.0,
        fluid_viscosity=1.0e-6,
        wavespeed=a,
        roughness_height=4.5e-5,
        friction_model="brunone",
        brunone_k_scale=1.0,
        dt=dt,
        tf=tf,
        wellhead_bc="velocity_step",
        pump_shut_time=ts,
        initial_velocity=V0,
        initial_head=H0,
        toe_bc="dead_end",
    )

    out_dir = os.path.join(
        _d, "output", "moc_physics_upgrade", "step1_steady_state_toe", "quad"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 2. 运行升级版仿真
    print(f"\n[1/3] 运行含裂缝仿真 (N={cfg.N}, tf={tf}s, 10万步)...")
    t0 = time_module.time()
    res = simulate_wellbore_upgraded(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * n_frac,
        fracture_kleak=[kleak] * n_frac,
        H_ext=H_ext,
    )
    elapsed_main = time_module.time() - t0
    print(f"  主仿真完成，耗时: {elapsed_main:.1f}s")

    t_sim = res["timestamps"]
    H_wh = res["wellhead_head"]
    V_wh = res["wellhead_velocity"]
    Q_wh = V_wh * cfg.area
    frac_heads = res["fracture_heads"]
    frac_Qs = res["fracture_Qs"]
    frac_indices = res["fracture_indices"]
    x_grid = res["x_grid"]
    x_f_aligned = [float(x_grid[idx]) for idx in frac_indices]

    # 3. 运行纯柔度对照组 (kleak=0)
    print(f"[2/3] 运行纯柔度对照组 (kleak=0)...")
    t0 = time_module.time()
    res_pure = simulate_wellbore_upgraded(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * n_frac,
        fracture_kleak=[0.0] * n_frac,
        H_ext=H_ext,
    )
    H_wh_pure = res_pure["wellhead_head"]
    print(f"  纯柔度完成，耗时: {time_module.time() - t0:.1f}s")

    # 4. 运行无缝对照组
    print(f"[3/3] 运行无缝对照组...")
    t0 = time_module.time()
    res_noFrac = simulate_wellbore_upgraded(cfg, fracture_positions=[])
    H_wh_noFrac = res_noFrac["wellhead_head"]
    print(f"  无缝完成，耗时: {time_module.time() - t0:.1f}s")

    # 5. 加载原版 PaperA quad 结果以进行直接对比
    papera_csv = os.path.join(
        _d, "PaperA井口多裂缝水击响应", "03_leakoff验证", "brunone_D20", "quad", "moc_timeseries.csv"
    )
    has_papera = os.path.isfile(papera_csv)
    if has_papera:
        data_orig = np.loadtxt(papera_csv, delimiter=',', skiprows=1)
        t_orig = data_orig[:, 0]
        H_wh_orig = data_orig[:, 1]
    else:
        t_orig = None
        H_wh_orig = None

    # 6. 指标评估
    ts_idx = int(round(ts / dt))
    dH_ana = cfg.a_adj * V0 / G
    dH_sim = H_wh[ts_idx] - H_wh[ts_idx - 1]
    err_dH = abs(abs(dH_sim) - dH_ana) / dH_ana * 100

    # 检查稳态阶段（0 <= t < ts）的波动标准差
    pre_mask = t_sim < ts
    std_pre = float(np.std(H_wh[pre_mask]))
    h_pre_mean = float(np.mean(H_wh[pre_mask]))

    print(f"\n--- 物理指标评估 ---")
    print(f"关泵前稳态水头平均值: {h_pre_mean:.4f} m (理论基准: {H0:.1f} m)")
    print(f"关泵前稳态水头标准差: {std_pre:.8e} m (原版由于趾端激波产生剧烈漂移，现为严格稳态)")
    print(f"Joukowsky 降落: 仿真值={dH_sim:.4f} m, 理论值={-dH_ana:.4f} m, 误差={err_dH:.4e}%")

    # 7. 保存全时程 CSV
    csv_cols = [t_sim, H_wh, Q_wh]
    csv_header = ['t', 'H_wh', 'Q_wh']
    for k in range(n_frac):
        csv_cols.append(frac_heads[:, k])
        csv_cols.append(frac_Qs[:, k])
        csv_header.append(f'H_f{k + 1}')
        csv_header.append(f'Q_f{k + 1}')
    csv_path = os.path.join(out_dir, "moc_timeseries.csv")
    np.savetxt(csv_path, np.column_stack(csv_cols), delimiter=',', header=','.join(csv_header), comments='')
    print(f"\n[交付 1] 时程 CSV 已保存: {csv_path}")

    # 8. 绘制 2x2 验证图
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Step 1 升级仿真验证 — PaperA quad (Brunone D=20m, 4裂缝)\n"
        f"初始稳态流场自洽 + 沿程达西摩阻平衡 + 消除 t=0 虚假激波\n"
        f"xf={[int(x) for x in x_f_list]}m, Cf={Cf}, kleak={kleak}, tf={tf}s",
        fontsize=13, fontweight='bold'
    )

    t_arrive_frac = [ts + 2.0 * xf / cfg.a_adj for xf in x_f_aligned]
    t_arrive_toe = ts + 2.0 * L / cfg.a_adj

    # (0,0) 全时程波形
    ax = axes[0, 0]
    ax.plot(t_sim, H_wh, 'b-', label='Step 1 含缝+滤失', lw=0.9)
    ax.plot(t_sim, H_wh_pure, 'r--', label='Step 1 纯柔度(无滤失)', lw=0.8, alpha=0.8)
    if has_papera:
        ax.plot(t_orig, H_wh_orig, 'k:', label='PaperA 原版(含初始激波漂移)', lw=0.8, alpha=0.6)
    ax.axvline(ts, color='g', ls=':', lw=1, label=f'关泵 ts={ts}s')
    for k, ta in enumerate(t_arrive_frac):
        ax.axvline(ta, color=FRAC_COLORS[k % len(FRAC_COLORS)], ls=':', lw=0.8)
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title(f'全时程井口水头演化 ({tf}s)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])

    # (0,1) 前 15s 特写 (重点看关泵前与关泵瞬间)
    ax = axes[0, 1]
    mask_15 = t_sim <= 15.0
    ax.plot(t_sim[mask_15], H_wh[mask_15], 'b-', label='Step 1 含缝+滤失 (稳态完全平直)', lw=1.2)
    if has_papera:
        mask_orig_15 = t_orig <= 15.0
        ax.plot(t_orig[mask_orig_15], H_wh_orig[mask_orig_15], 'r--', label='PaperA 原版 (t<1s 即有趾端虚假回波)', lw=1.0)
    ax.axvline(ts, color='g', ls='-', lw=1.2, label='关泵时刻 ts=1.0s')
    for k, ta in enumerate(t_arrive_frac):
        ax.axvline(ta, color=FRAC_COLORS[k % len(FRAC_COLORS)], ls=':', lw=1, label=f'缝{k+1}反射 ta={ta:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title('前 15s 特写: 稳态平直度与关泵波前响应')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, 15])

    # (1,0) 差信号 (含缝 - 无缝)
    ax = axes[1, 0]
    diff_signal = H_wh - H_wh_noFrac
    mask_diff = (t_sim >= ts) & (t_sim <= 15.0)
    ax.plot(t_sim[mask_diff], diff_signal[mask_diff], 'b-', lw=1.0, label='差信号 (裂缝净诱导反射)')
    for k, ta in enumerate(t_arrive_frac):
        ax.axvline(ta, color=FRAC_COLORS[k % len(FRAC_COLORS)], ls=':', lw=1)
    ax.axhline(0, color='k', lw=0.3)
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('差信号水头 [m]')
    ax.set_title('关泵后裂缝净反射差信号 (前15s)')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([ts, 15])

    # (1,1) 各裂缝节点水头与滤失流量时程
    ax = axes[1, 1]
    ax2 = ax.twinx()
    for k in range(n_frac):
        c = FRAC_COLORS[k % len(FRAC_COLORS)]
        ax.plot(t_sim[mask_15], frac_heads[mask_15, k], '-', color=c, lw=0.9, label=f'缝{k+1} H')
        ax2.plot(t_sim[mask_15], frac_Qs[mask_15, k], '--', color=c, lw=0.6, alpha=0.7, label=f'缝{k+1} Q_f')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('缝节点水头 [m]')
    ax2.set_ylabel('缝滤失流量 [m$^3$/s]')
    ax.set_title('裂缝节点 H 与 Q_f 动态响应 (前15s)')
    ax.legend(fontsize=6, loc='upper left'); ax2.legend(fontsize=6, loc='upper right')
    ax.grid(True, ls='--', alpha=0.6); ax.set_xlim([0, 15])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    png_2x2 = os.path.join(out_dir, "moc_leakoff.png")
    plt.savefig(png_2x2, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[交付 2] 2x2 验证图已保存: {png_2x2}")

    # 9. 倒谱分析与绘制
    print(f"\n[4/4] 计算 1D 实倒谱与 2D 滑窗倒谱云图...")
    cep_path = os.path.join(out_dir, "cepstrum_standard.png")
    cep_zoom_path = os.path.join(out_dir, "cepstrum_fracture_zoom.png")

    title_prefix = f"Step 1 quad — 井口水头倒谱分析\nx_f={[round(x) for x in x_f_aligned]}m, Brunone 摩阻"
    cep_result = plot_moc_cepstrum_analysis(
        t_sim, H_wh,
        wavespeed=cfg.a_adj, ts=ts, dt=dt, wellbore_length=L,
        fracture_positions=x_f_aligned, save_path=cep_path,
        title_prefix=title_prefix,
        wlen_sec=CEPSTRUM_CONFIG['wlen_sec'],
        hop_sec=CEPSTRUM_CONFIG['hop_sec'],
        win_type=CEPSTRUM_CONFIG['win_type'],
    )
    print(f"[交付 3] 标准 5 联倒谱图已保存: {cep_path}")

    plot_moc_cepstrum_fracture_zoom(
        cep_result,
        fracture_positions=x_f_aligned,
        save_path=cep_zoom_path,
        title_prefix=f"Step 1 quad — 裂缝区倒谱放大\nx_f={[round(x) for x in x_f_aligned]}m",
        wellbore_length=L,
    )
    print(f"[交付 4] 裂缝区放大特写图已保存: {cep_zoom_path}")

    fs_cep = cep_result['fs']
    v_cep = cep_result['v']
    cep_1d = evaluate_1d_cepstrum_fracture_match(
        cep_result['depth_1d'], cep_result['response_1d'],
        x_f_list, v=v_cep, fs=fs_cep,
    )
    cep_2d_avg = evaluate_1d_cepstrum_fracture_match(
        cep_result['depth_profile_2d'], cep_result['response_profile_2d'],
        x_f_list, v=v_cep, fs=fs_cep,
    )

    # 10. 额外输出 Step 1 vs 原版叠合对比大图
    if has_papera:
        fig_cmp, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
        fig_cmp.suptitle("Step 1 物理修复成果比对: 稳态基态流场自洽 vs PaperA 原版缺陷", fontsize=13, fontweight='bold')

        # 关泵前后特写 (0~10s)
        ax_c1.plot(t_sim[:10000], H_wh[:10000], 'b-', lw=1.5, label='Step 1: 物理相容稳态初值 (t<1s 平直无扰动)')
        ax_c1.plot(t_orig[:10000], H_wh_orig[:10000], 'r--', lw=1.2, label='PaperA 原版: 趾端强行切零激发假激波 (t<1s 剧烈漂移)')
        ax_c1.axvline(ts, color='g', ls='-', label='关泵时刻 ts=1.0s')
        ax_c1.set_ylabel('井口水头 [m]'); ax_c1.set_title('关泵前后过渡段 (0-10s): 假激波消除效果对比')
        ax_c1.grid(True, ls='--', alpha=0.6); ax_c1.legend(loc='upper right')
        ax_c1.set_xlim([0, 10])

        # 全时程长时衰减对比 (0~100s)
        ax_c2.plot(t_sim, H_wh, 'b-', lw=1.0, alpha=0.9, label='Step 1 升级版时程')
        ax_c2.plot(t_orig, H_wh_orig, 'k--', lw=0.8, alpha=0.6, label='PaperA 原版时程')
        ax_c2.set_xlabel('时间 [s]'); ax_c2.set_ylabel('井口水头 [m]')
        ax_c2.set_title('全时程 (100s) 对比')
        ax_c2.grid(True, ls='--', alpha=0.6); ax_c2.legend(loc='upper right')
        ax_c2.set_xlim([0, tf])

        plt.tight_layout()
        cmp_png = os.path.join(out_dir, "comparison_step1_vs_original.png")
        plt.savefig(cmp_png, dpi=130, bbox_inches='tight')
        plt.close(fig_cmp)
        print(f"[交付 5] 对比大图已保存: {cmp_png}")

    # 11. 输出 JSON 指标
    result_json = {
        "step": "step1_steady_state_toe",
        "description": "初始稳态流场自洽 + 多簇分流 + 沿程摩阻平衡 + 封闭趾端无激波",
        "metrics": {
            "h_pre_mean": h_pre_mean,
            "h_pre_std": std_pre,
            "dH_sim": float(dH_sim),
            "dH_ana": float(-dH_ana),
            "dH_err_pct": float(err_dH),
            "x_f_aligned": x_f_aligned,
        },
        "cepstrum": {
            "1d_real": cepstrum_match_summary_for_json(cep_1d),
            "2d_time_avg": cepstrum_match_summary_for_json(cep_2d_avg),
        },
        "config": {
            "L": L, "a": a, "V0": V0, "H0": H0, "ts": ts, "dt": dt, "tf": tf,
            "x_f": x_f_list, "Cf": Cf, "kleak": kleak, "H_ext": H_ext,
            "friction": "brunone_D20", "toe_bc": "dead_end"
        }
    }
    json_path = os.path.join(out_dir, "moc_leakoff.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"[交付 6] 结果 JSON 已更新: {json_path}")
    print("=" * 76)
    print("Step 1 quad 验证运行完成！")


if __name__ == '__main__':
    main()
