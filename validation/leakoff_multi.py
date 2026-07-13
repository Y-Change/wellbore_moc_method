# -*- coding: utf-8 -*-
"""
统一 leakoff_multi.py — steady / brunone 摩阻 + 裂缝滤失多缝验证。

通过 --friction steady|brunone|steady_D*|brunone_D*|steady_Dall|brunone_Dall
与 --case single|dual|...|all 切换。
参数集中管理在 validation/config.py，修改该文件即可全局调整。

输出路径: output/leakoff/{friction}/{case}/
  - moc_leakoff.png       2×2 MOC 验证图
  - cepstrum_standard.png 标准倒谱图（时域/FFT/1D/2D/时间平均剖面）
  - moc_timeseries.csv    井口/缝口水头与流量时程
  - moc_leakoff.json      PASS/FAIL 判定 + 倒谱缝深匹配
                          (cepstrum.1d_real / cepstrum.2d_time_avg)

运行
----
    python validation/leakoff_multi.py --friction steady --case all
    python validation/leakoff_multi.py --friction steady_D10 --case all
    python validation/leakoff_multi.py --friction steady_Dall --case all
    python validation/leakoff_multi.py --friction brunone_Dall --case dual
    python validation/leakoff_multi.py --friction brunone --case dual

仅重绘倒谱图并更新 JSON（不重跑 MOC，读已有 moc_timeseries.csv）
----
    python validation/leakoff_multi.py --replay --friction steady_D20 --case quad
    python validation/leakoff_multi.py --replay --friction steady_Dall --case all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from typing import Dict, List, Optional, Tuple

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'paths.py')) and os.path.isfile(
        os.path.join(_d, 'wellbore_moc.py')
    ):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find wellbore_moc_method root')
    _d = _parent

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from paths import output_path, SERIES_LEAKOFF
from wellbore_moc import MocConfig, simulate_wellbore, G
from cepstrum_mocdata import (
    plot_moc_cepstrum_analysis,
    evaluate_1d_cepstrum_fracture_match,
    cepstrum_match_summary_for_json,
)
from validation.config import (
    WELL_CONFIG, SIM_CONFIG, FRACTURE_CONFIG, CEPSTRUM_CONFIG,
    CASES, FRICTION_PARAMS, build_cases,
    expand_friction_keys, friction_cli_choices,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FRAC_COLORS = ['b', 'r', 'g', 'm', 'c', 'orange']
CEP_WLEN_SEC = CEPSTRUM_CONFIG['wlen_sec']
CEP_HOP_SEC = CEPSTRUM_CONFIG['hop_sec']
CEP_WIN_TYPE = CEPSTRUM_CONFIG['win_type']


# ── 辅助函数 ──────────────────────────────────────────────

def osc_rms(H_arr, t_arr, t_start, t_end):
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    if mask.sum() < 10:
        return np.nan
    H_win = H_arr[mask]
    return float(np.sqrt(np.mean((H_win - np.mean(H_win)) ** 2)))


def check_diff_step(diff_signal, t_sim, t_arrive, half_win=0.15, min_step=5.0):
    mask_b = (t_sim >= t_arrive - half_win) & (t_sim < t_arrive - 0.02)
    mask_a = (t_sim > t_arrive + 0.02) & (t_sim <= t_arrive + half_win)
    if mask_b.sum() == 0 or mask_a.sum() == 0:
        return np.nan, False
    diff_step = float(np.mean(diff_signal[mask_a]) - np.mean(diff_signal[mask_b]))
    return diff_step, abs(diff_step) > min_step


def _build_moc_config(friction: str) -> MocConfig:
    fr = FRICTION_PARAMS[friction]
    w = WELL_CONFIG
    s = SIM_CONFIG
    return MocConfig(
        wellbore_length=w['L'],
        wellbore_diameter=w['wellbore_diameter'],
        fluid_density=w['fluid_density'],
        fluid_viscosity=w['fluid_viscosity'],
        wavespeed=w['wavespeed'],
        roughness_height=w['roughness_height'],
        friction_model=fr['friction_model'],
        dt=s['dt'], tf=s['tf'],
        wellhead_bc='velocity_step', pump_shut_time=s['ts'],
        initial_velocity=w['V0'], initial_head=w['H0'],
        theta=w['theta'], toe_bc='reservoir', toe_head=w['H0'],
    )


def resolve_cases(friction: str) -> Dict:
    """按 FRICTION_PARAMS[friction].spacing_m 生成缝形态；无则用默认 CASES。"""
    fr = FRICTION_PARAMS[friction]
    if 'spacing_m' in fr:
        return build_cases(fr['spacing_m'])
    return CASES


def _case_paths(friction: str, case_key: str) -> Dict[str, str]:
    series = f"{SERIES_LEAKOFF}/{friction}"
    return {
        'series': series,
        'csv': output_path(series, case_key, 'moc_timeseries.csv'),
        'json': output_path(series, case_key, 'moc_leakoff.json'),
        'cepstrum_png': output_path(series, case_key, 'cepstrum_standard.png'),
        'moc_png': output_path(series, case_key, 'moc_leakoff.png'),
    }


def load_timeseries_csv(csv_path: str) -> Dict:
    """读取 moc_timeseries.csv → t, H_wh, Q_wh, frac_heads, frac_Qs。"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'找不到时程 CSV: {csv_path}')
    with open(csv_path, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    col = {name: i for i, name in enumerate(header)}
    for req in ('t', 'H_wh', 'Q_wh'):
        if req not in col:
            raise ValueError(f'CSV 缺少列 {req}: {csv_path}')
    n_frac = 0
    while f'H_f{n_frac + 1}' in col and f'Q_f{n_frac + 1}' in col:
        n_frac += 1
    frac_heads = np.column_stack(
        [data[:, col[f'H_f{k + 1}']] for k in range(n_frac)]
    ) if n_frac else np.zeros((len(data), 0))
    frac_Qs = np.column_stack(
        [data[:, col[f'Q_f{k + 1}']] for k in range(n_frac)]
    ) if n_frac else np.zeros((len(data), 0))
    return {
        't': data[:, col['t']],
        'H_wh': data[:, col['H_wh']],
        'Q_wh': data[:, col['Q_wh']],
        'frac_heads': frac_heads,
        'frac_Qs': frac_Qs,
        'n_frac': n_frac,
        'header': header,
    }


def _print_cep_match(tag: str, metrics: dict) -> None:
    detail = ' '.join(
        f"F{mt['frac_id']}:{mt['peak_depth_m']:.0f}m(Δ{mt['error_m']:.1f}m)"
        if mt['matched'] and mt['peak_depth_m'] is not None
        else f"F{mt['frac_id']}:×"
        for mt in metrics['matches']
    )
    mean_e = metrics['mean_error_m']
    max_e = metrics['max_error_m']
    mean_s = f"{mean_e:.1f}" if mean_e is not None else "nan"
    max_s = f"{max_e:.1f}" if max_e is not None else "nan"
    print(
        f"  [{tag}] {metrics['n_matched']}/{metrics['n_fracs']}匹配 "
        f"mean_err={mean_s}m max_err={max_s}m  {detail}"
    )


def run_cepstrum_analysis_and_match(
    t_sim: np.ndarray,
    H_wh: np.ndarray,
    *,
    a_adj: float,
    L: float,
    ts: float,
    dt: float,
    x_f_list: List[float],
    x_f_plot: List[float],
    friction: str,
    label: str,
    kleak: float,
    cep_path: str,
) -> Tuple[Dict, Dict, Dict]:
    """绘制倒谱五联图，并返回 (cep_result, cep_1d, cep_2d_avg)。"""
    cep_result = plot_moc_cepstrum_analysis(
        t_sim, H_wh,
        wavespeed=a_adj, ts=ts, dt=dt, wellbore_length=L,
        fracture_positions=x_f_plot, save_path=cep_path,
        title_prefix=(
            f"测试 {label} — 井口水头倒谱分析\n"
            f"x_f={[round(x) for x in x_f_plot]}m, k_leak={kleak}, {friction} 摩阻"
        ),
        wlen_sec=CEP_WLEN_SEC, hop_sec=CEP_HOP_SEC, win_type=CEP_WIN_TYPE,
    )
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
    print(f"\n[倒谱] 缝深匹配 (名义缝深 {x_f_list}):")
    _print_cep_match('1D 实倒谱', cep_1d)
    _print_cep_match('2D 时间平均剖面', cep_2d_avg)
    return cep_result, cep_1d, cep_2d_avg


def cepstrum_block_for_json(cep_result: Dict, cep_1d: Dict, cep_2d_avg: Dict) -> Dict:
    return {
        "1d_real": cepstrum_match_summary_for_json(cep_1d),
        "2d_time_avg": {
            **cepstrum_match_summary_for_json(cep_2d_avg),
            "wlen_sec": float(CEP_WLEN_SEC),
            "hop_sec": float(CEP_HOP_SEC),
            "win_type": CEP_WIN_TYPE,
            "n_frames": int(len(cep_result['t_cep'])),
        },
    }


def align_frac_to_grid(x_f_list: List[float], cfg: MocConfig) -> List[float]:
    """名义缝深对齐到 MOC 网格节点。"""
    x_grid = np.linspace(0.0, cfg.wellbore_length, cfg.N + 1)
    return [float(x_grid[int(np.argmin(np.abs(x_grid - xf)))]) for xf in x_f_list]


# ── 核心 ──────────────────────────────────────────────────

def run_case(case_key: str, friction: str = 'steady') -> Dict:
    cases = resolve_cases(friction)
    cfg_case = cases[case_key]
    label = cfg_case['label']
    x_f_list = cfg_case['x_f_list']
    fr_params = FRICTION_PARAMS[friction]
    n_frac = len(x_f_list)

    w = WELL_CONFIG
    s = SIM_CONFIG
    fc = FRACTURE_CONFIG
    L = w['L']
    ts = s['ts']
    dt = s['dt']
    tf = s['tf']
    Cf = fc['Cf']
    kleak = fc['kleak']
    H_ext = fc['H_ext']

    print("\n" + "=" * 72)
    print(f"测试 {label} — {fr_params['label']}")
    print(f"x_f={x_f_list}m, Cf={Cf}, k_leak={kleak}, H_ext={H_ext}m")
    print("=" * 72)

    cfg = _build_moc_config(friction)
    dH_ana = cfg.a_adj * w['V0'] / G
    T_toe = 2.0 * L / cfg.a_adj
    t_arrive_toe = ts + T_toe

    print(f"\n物理参数:")
    print(f"  L={L}m, a_adj={cfg.a_adj:.4f} m/s, V0={w['V0']} m/s, ts={ts} s")
    print(f"  摩阻: {fr_params['label']}")
    print(f"  ΔH = aV0/g = {dH_ana:.4f} m, 2L/a = {T_toe:.4f} s")
    print(f"  dx={cfg.dx:.4f} m, N={cfg.N}, n_steps={cfg.n_steps}")

    snap_times = [0.0, ts, 5.0, 7.0, 8.0, 10.0, 30.0, 50.0, tf]

    # ── 仿真 1: 含缝+滤失 ──────────────────────────────────
    print(f"\n运行含缝仿真 ({friction}, kleak={kleak}, tf={tf}s)...")
    t0 = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * n_frac,
        fracture_kleak=[kleak] * n_frac,
        H_ext=H_ext,
        store_full_field=False,
        snapshot_times=snap_times,
    )
    print(f"  耗时: {time_module.time() - t0:.1f}s")

    # ── 仿真 2: 纯柔度对照（无滤失）────────────────────────
    print(f"运行纯柔度对照 ({friction}, kleak=0)...")
    t0 = time_module.time()
    res_pure = simulate_wellbore(
        cfg,
        fracture_positions=x_f_list,
        fracture_Cf=[Cf] * n_frac,
        fracture_kleak=[0.0] * n_frac,
        H_ext=H_ext,
        store_full_field=False,
    )
    print(f"  耗时: {time_module.time() - t0:.1f}s")

    # ── 仿真 3: 无缝对照 ───────────────────────────────────
    print(f"运行无缝对照 ({friction})...")
    t0 = time_module.time()
    res_noFrac = simulate_wellbore(cfg, store_full_field=False)
    print(f"  耗时: {time_module.time() - t0:.1f}s")

    # ── 提取数据 ───────────────────────────────────────────
    t_sim = res["timestamps"]
    H_wh = res["wellhead_head"]
    V_wh = res["wellhead_velocity"]
    Q_wh = V_wh * cfg.area
    H_wh_pure = res_pure["wellhead_head"]
    H_wh_noFrac = res_noFrac["wellhead_head"]
    frac_indices = res["fracture_indices"]
    frac_heads = res["fracture_heads"]
    frac_Qs = res["fracture_Qs"]
    x_grid = res["x_grid"]
    x_f_aligned = [x_grid[idx] for idx in frac_indices]
    t_arrive_frac = [ts + 2.0 * xf / cfg.a_adj for xf in x_f_aligned]

    ts_idx = int(round(ts / dt))
    H0_actual = H_wh[ts_idx - 1]
    diff_signal = H_wh - H_wh_noFrac
    diff_pure = H_wh_pure - H_wh_noFrac

    print(f"\n  缝对齐:")
    for k in range(n_frac):
        print(f"    缝{k + 1}: x_f={x_f_list[k]}m → 网格 x={x_f_aligned[k]:.2f}m "
              f"(idx={frac_indices[k]}), 反射到达 t={t_arrive_frac[k]:.4f}s")
    print(f"    趾端反射到达 t = {t_arrive_toe:.4f}s")
    for k in range(n_frac - 1):
        print(f"    间隔: 缝{k + 1}-缝{k + 2}={t_arrive_frac[k + 1] - t_arrive_frac[k]:.3f}s")
    print(f"    间隔: 缝{n_frac}-趾端={t_arrive_toe - t_arrive_frac[-1]:.3f}s")
    print(f"  稳态井口水头 H0_actual = {H0_actual:.4f} m")

    # ── 判定 1: Joukowsky 跳变 ────────────────────────────
    dH_sim = H_wh[ts_idx] - H_wh[ts_idx - 1]
    err_dH = abs(dH_sim - (-dH_ana)) / dH_ana * 100
    print(f"\n[判定 1] Joukowsky 初始跳变: 误差 = {err_dH:.4f}%")
    verdict_dH = "PASS" if err_dH < 0.1 else "FAIL"
    print(f"  结论: {verdict_dH}")

    # ── 判定 2: 缝+趾端反射（差信号 step）──────────────────
    print(f"\n[判定 2] 缝+趾端反射（差信号 step）:")
    frac_steps = []
    all_frac_pass = True
    for k in range(n_frac):
        step_k, ok_k = check_diff_step(diff_signal, t_sim, t_arrive_frac[k])
        frac_steps.append(step_k)
        direction = "正向" if step_k > 0 else "负向(叠加衰减)"
        print(f"    缝{k + 1} ({x_f_aligned[k]:.0f}m): t={t_arrive_frac[k]:.4f}s, "
              f"step={step_k:+.2f}m [{direction}] {'PASS' if ok_k else 'FAIL'}")
        all_frac_pass = all_frac_pass and ok_k

    half_win = 0.15
    mask_b = (t_sim >= t_arrive_toe - half_win) & (t_sim < t_arrive_toe - 0.02)
    mask_a = (t_sim > t_arrive_toe + 0.02) & (t_sim <= t_arrive_toe + half_win)
    H_toe_before = float(np.mean(H_wh_noFrac[mask_b]))
    H_toe_after = float(np.mean(H_wh_noFrac[mask_a]))
    toe_step = H_toe_after - H_toe_before
    ok_toe = abs(toe_step) > 5.0
    print(f"    趾端: t={t_arrive_toe:.4f}s, step={toe_step:+.2f}m {'PASS' if ok_toe else 'FAIL'}")
    verdict_frac = "PASS" if (all_frac_pass and ok_toe) else "FAIL"
    print(f"  结论: {verdict_frac}")

    # ── 判定 3: 滤失阻尼 ──────────────────────────────────
    win_lo = t_arrive_frac[0] + 0.5
    win_hi = t_arrive_toe - 0.1
    rms_leak = osc_rms(H_wh, t_sim, win_lo, win_hi)
    rms_pure = osc_rms(H_wh_pure, t_sim, win_lo, win_hi)
    damping_ratio = rms_pure / rms_leak if rms_leak > 0 else np.nan
    print(f"\n[判定 3] 滤失阻尼: 阻尼比 = {damping_ratio:.3f} (>1.05 → 滤失增阻尼)")
    verdict_damp = "PASS" if (not np.isnan(damping_ratio) and damping_ratio > 1.05) else "FAIL"
    print(f"  结论: {verdict_damp}")

    # ── 判定 3b: 滤失稳态 Q ───────────────────────────────
    Q_last_mask = t_sim >= (tf - 5.0)
    Q_steady_ana = kleak * np.sqrt(max(w['H0'] - H_ext, 0.0))
    Q_steady_list = []
    verdict_qss_list = []
    print(f"\n[判定 3b] 滤失稳态分量 (最后 5s, 各缝):")
    for k in range(n_frac):
        Q_steady_k = float(np.mean(frac_Qs[Q_last_mask, k]))
        Q_steady_list.append(Q_steady_k)
        ok_qss = Q_steady_k > 0 and Q_steady_k > 0.3 * Q_steady_ana
        verdict_qss_list.append("PASS" if ok_qss else "FAIL")
        print(f"    缝{k + 1}: mean(Q_f) = {Q_steady_k:.6f} → {verdict_qss_list[-1]}")
    verdict_qss = "PASS" if all(v == "PASS" for v in verdict_qss_list) else "FAIL"
    print(f"  结论: {verdict_qss}")

    # ── 判定 4: friction 特异 ─────────────────────────────
    verdict_j4 = "N/A"
    jitter_ratio = np.nan
    decay_ratio = np.nan
    decay_ratio_pure = np.nan
    rms_early = rms_late = rms_early_pure = rms_late_pure = np.nan

    if fr_params['judgment4'] == 'smoothness':
        # steady: 波形平滑性（抖动比 < 2.0）
        smooth_mask = (t_sim >= t_arrive_frac[0]) & (t_sim < t_arrive_frac[0] + 0.5)
        if smooth_mask.sum() > 10:
            dHdt_smooth = np.abs(np.diff(H_wh[smooth_mask]) / dt)
            mean_dHdt = np.mean(dHdt_smooth)
            std_dHdt = np.std(dHdt_smooth)
            jitter_ratio = std_dHdt / mean_dHdt if mean_dHdt > 0 else 0
            print(f"\n[判定 4] 波形平滑性: 抖动比 = {jitter_ratio:.4f} (< 2.0)")
            verdict_j4 = "PASS" if jitter_ratio < 2.0 else "FAIL"
        print(f"  结论: {verdict_j4}")

    elif fr_params['judgment4'] == 'oscillation_decay':
        # brunone: 振荡衰减（RMS 衰减比 < 0.5）
        rms_early = osc_rms(H_wh, t_sim, ts, ts + T_toe)
        rms_late = osc_rms(H_wh, t_sim, tf - 10.0, tf)
        rms_early_pure = osc_rms(H_wh_pure, t_sim, ts, ts + T_toe)
        rms_late_pure = osc_rms(H_wh_pure, t_sim, tf - 10.0, tf)
        decay_ratio = rms_late / rms_early if rms_early > 0 else np.nan
        decay_ratio_pure = rms_late_pure / rms_early_pure if rms_early_pure > 0 else np.nan
        print(f"\n[判定 4] Brunone 振荡衰减: 衰减比 = {decay_ratio:.4f} (< 0.5), "
              f"纯柔度 = {decay_ratio_pure:.4f}")
        verdict_j4 = "PASS" if (
            not np.isnan(decay_ratio) and decay_ratio < 0.5 and rms_late < rms_late_pure
        ) else "FAIL"
        print(f"  结论: {verdict_j4}")

    # ── 判定 5: 长期稳定性 ────────────────────────────────
    last_5s = t_sim >= (tf - 5.0)
    H_last = H_wh[last_5s]
    H_last_range = float(np.max(H_last) - np.min(H_last))
    has_nan = bool(np.any(np.isnan(H_last)) or np.any(np.isinf(H_last)))
    stab_factor = fr_params['stab_factor']
    print(f"\n[判定 5] 长期稳定性: H范围={H_last_range:.2f}m, NaN={has_nan}")
    verdict_stab = "PASS" if (not has_nan and H_last_range < stab_factor * dH_ana) else "FAIL"
    print(f"  阈值: H范围 < {stab_factor}×ΔH={stab_factor * dH_ana:.1f}m → {verdict_stab}")

    # ── 判定 6: 停泵前滤失稳态 ────────────────────────────
    mask_pre = (t_sim >= 0.5 * ts) & (t_sim < ts - 0.01)
    verdict_degen_list = []
    print(f"\n[判定 6] 停泵前滤失稳态 (t∈[0.5·ts, ts]):")
    for k in range(n_frac):
        Q_pre_k = float(np.mean(frac_Qs[mask_pre, k]))
        ratio_Qpre = Q_pre_k / Q_steady_ana if Q_steady_ana > 0 else 0.0
        ok_degen = Q_pre_k > 0 and 0.3 < ratio_Qpre < 2.0
        verdict_degen_list.append("PASS" if ok_degen else "FAIL")
        print(f"    缝{k + 1}: mean(Q_f) = {Q_pre_k:.6f}, 比值 = {ratio_Qpre:.3f} → "
              f"{verdict_degen_list[-1]}")
    verdict_degen = "PASS" if all(v == "PASS" for v in verdict_degen_list) else "FAIL"
    print(f"  结论: {verdict_degen}")

    # ── 总评 ─────────────────────────────────────────────
    verdicts = [verdict_dH, verdict_frac, verdict_damp, verdict_qss,
                verdict_j4, verdict_stab, verdict_degen]
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_total = sum(1 for v in verdicts if v != "N/A")
    print("\n" + "=" * 72)
    print(f"总评: {n_pass}/{n_total} 项 PASS")
    if n_pass == n_total:
        print(f"[OK] 测试 {label}（{fr_params['label']}）通过")
    else:
        print("[FAIL] 验证未全部通过")
    print("=" * 72)

    # ── 可视化: 2×2 四联图 ────────────────────────────────
    series = f"{SERIES_LEAKOFF}/{friction}"
    x_f_label = "/".join(f"{int(x)}" for x in x_f_list)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    j4_label = "抖动比" if fr_params['judgment4'] == 'smoothness' else f"衰减比={decay_ratio:.4f}"
    fig.suptitle(
        f"测试 {label} — {fr_params['label']}\n"
        f"x_f=[{x_f_label}]m, Cf={Cf}, k_leak={kleak}, H_ext={H_ext}m, tf={tf}s\n"
        f"阻尼比={damping_ratio:.3f}, {j4_label}, Q_steady≈{Q_steady_ana:.5f} m$^3$/s/缝",
        fontsize=13, fontweight='bold'
    )

    # (0,0) 全时程
    ax = axes[0, 0]
    ax.plot(t_sim, H_wh, 'b-', label=f'含缝+滤失 {friction}')
    ax.plot(t_sim, H_wh_pure, 'r-', label='纯柔度(无滤失)')
    ax.plot(t_sim, H_wh_noFrac, 'k--', label=f'无缝 {friction}')
    ax.axvline(ts, color='g', ls=':', lw=1, label=f'停泵 ts={ts}s')
    for k, ta in enumerate(t_arrive_frac):
        ax.axvline(ta, color=FRAC_COLORS[k % len(FRAC_COLORS)], ls=':', lw=1,
                   label=f'缝{k + 1} {ta:.2f}s')
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1, label=f'趾端 {t_arrive_toe:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title(f'全时程 ({tf}s) — 含缝 vs 无缝 ({friction})')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, tf])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()

    # (0,1) 前 12s 特写
    ax = axes[0, 1]
    mask_12 = t_sim <= 12.0
    ax.plot(t_sim[mask_12], H_wh[mask_12], 'b-', label='含缝+滤失')
    ax.plot(t_sim[mask_12], H_wh_pure[mask_12], 'r-', label='纯柔度(无滤失)')
    ax.plot(t_sim[mask_12], H_wh_noFrac[mask_12], 'k--', label='无缝')
    ax.axvline(ts, color='g', ls=':', lw=1)
    for k, ta in enumerate(t_arrive_frac):
        ax.axvline(ta, color=FRAC_COLORS[k % len(FRAC_COLORS)], ls=':', lw=1.2,
                   label=f'缝{k + 1} {ta:.2f}s')
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1, label=f'趾端 {t_arrive_toe:.2f}s')
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('井口水头 [m]')
    ax.set_title(f'前 12s 特写 — {fr_params["label"]}')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim([0, 12])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()

    # (1,0) 差信号
    ax = axes[1, 0]
    mask_diff = (t_sim >= t_arrive_frac[0] - 0.5) & (t_sim <= t_arrive_toe + 0.5)
    ax.plot(t_sim[mask_diff], diff_signal[mask_diff], 'b-', lw=1.0, label='差信号(含滤失)')
    ax.plot(t_sim[mask_diff], diff_pure[mask_diff], 'c--', lw=0.8, alpha=0.7, label='差信号(纯柔度)')
    for k, ta in enumerate(t_arrive_frac):
        ax.axvline(ta, color=FRAC_COLORS[k % len(FRAC_COLORS)], ls=':', lw=1)
    ax.axvline(t_arrive_toe, color='orange', ls=':', lw=1)
    ax.axhline(0, color='k', lw=0.3)
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('差信号 [m]')
    ax.set_title(f'差信号 — 滤失阻尼比={damping_ratio:.3f}')
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()

    # (1,1) 缝节点 H 与 Q_f
    ax = axes[1, 1]
    ax2 = ax.twinx()
    for k in range(n_frac):
        c = FRAC_COLORS[k % len(FRAC_COLORS)]
        ax.plot(t_sim, frac_heads[:, k], '-', color=c, lw=0.5,
                label=f'缝{k + 1} H (x={x_f_aligned[k]:.0f}m)')
        ax2.plot(t_sim, frac_Qs[:, k], '--', color=c, lw=0.3, alpha=0.7,
                 label=f'缝{k + 1} Q_f')
    ax.axvline(ts, color='g', ls=':', lw=1)
    ax.set_xlabel('时间 [s]'); ax.set_ylabel('缝节点水头 [m]')
    ax2.set_ylabel('缝侧向流量 [m$^3$/s]')
    ax.set_title(f'{label}节点 H 与 Q_f (Q_steady≈{Q_steady_ana:.5f}/缝)')
    ax.legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right')
    ax.grid(True, ls='--', alpha=0.6); ax.set_xlim([0, tf])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = output_path(series, case_key, "moc_leakoff.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n图已保存: {out_path}")
    plt.close(fig)

    # ── 时程 CSV：井口 / 缝口水头与流量 ────────────────────
    csv_cols = [t_sim, H_wh, Q_wh]
    csv_header = ['t', 'H_wh', 'Q_wh']
    for k in range(n_frac):
        csv_cols.append(frac_heads[:, k])
        csv_cols.append(frac_Qs[:, k])
        csv_header.append(f'H_f{k + 1}')
        csv_header.append(f'Q_f{k + 1}')
    csv_path = output_path(series, case_key, "moc_timeseries.csv")
    np.savetxt(
        csv_path,
        np.column_stack(csv_cols),
        delimiter=',',
        header=','.join(csv_header),
        comments='',
    )
    print(f"时程 CSV: {csv_path}")

    # ── 倒谱分析图 ────────────────────────────────────────
    cep_path = output_path(series, case_key, "cepstrum_standard.png")
    cep_result, cep_1d, cep_2d_avg = run_cepstrum_analysis_and_match(
        t_sim, H_wh,
        a_adj=cfg.a_adj, L=L, ts=ts, dt=dt,
        x_f_list=x_f_list, x_f_plot=x_f_aligned,
        friction=friction, label=label, kleak=kleak,
        cep_path=cep_path,
    )

    # ── JSON ──────────────────────────────────────────────
    j4_key = fr_params['judgment4']
    result = {
        "verdicts": {
            "joukowsky": verdict_dH,
            "fracture_reflection": verdict_frac,
            "leakoff_damping": verdict_damp,
            "leakoff_steady_Q": verdict_qss,
            j4_key: verdict_j4,
            "stability": verdict_stab,
            "degeneracy": verdict_degen,
        },
        "metrics": {
            "dH_err_pct": float(err_dH),
            "frac_steps": [float(s) if not np.isnan(s) else None for s in frac_steps],
            "toe_step": float(toe_step),
            "rms_leak": float(rms_leak) if not np.isnan(rms_leak) else None,
            "rms_pure": float(rms_pure) if not np.isnan(rms_pure) else None,
            "damping_ratio": float(damping_ratio) if not np.isnan(damping_ratio) else None,
            "Q_steady_per_frac": Q_steady_list,
            "Q_steady_ana": float(Q_steady_ana),
            "jitter_ratio": float(jitter_ratio) if not np.isnan(jitter_ratio) else None,
            "decay_ratio": float(decay_ratio) if not np.isnan(decay_ratio) else None,
            "decay_ratio_pure": float(decay_ratio_pure) if not np.isnan(decay_ratio_pure) else None,
            "long_term_range": float(H_last_range),
            "t_arrive_frac": [float(t) for t in t_arrive_frac],
            "t_arrive_toe": float(t_arrive_toe),
            "x_f_aligned": [float(x) for x in x_f_aligned],
        },
        "cepstrum": cepstrum_block_for_json(cep_result, cep_1d, cep_2d_avg),
        "config": {
            "L": L, "a": w['wavespeed'], "V0": w['V0'], "H0": w['H0'],
            "ts": ts, "dt": dt, "tf": tf,
            "x_f": x_f_list, "Cf": Cf, "kleak": kleak, "H_ext": H_ext,
            "friction": friction,
        },
    }
    json_path = output_path(series, case_key, "moc_leakoff.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON: {json_path}")

    return result


def replay_case(case_key: str, friction: str = 'steady') -> Dict:
    """从已有 moc_timeseries.csv 重绘倒谱图并更新 JSON（不重跑 MOC）。

    - 必读：moc_timeseries.csv
    - 若存在 moc_leakoff.json：保留 verdicts/metrics，仅刷新 cepstrum
    - 若不存在 JSON：写入仅含 config + cepstrum 的精简结果
    """
    cases = resolve_cases(friction)
    if case_key not in cases:
        raise KeyError(f'未知 case={case_key}，可选 {list(cases.keys())}')
    cfg_case = cases[case_key]
    label = cfg_case['label']
    x_f_list = list(cfg_case['x_f_list'])
    fr_params = FRICTION_PARAMS[friction]
    fc = FRACTURE_CONFIG
    kleak = fc['kleak']

    paths = _case_paths(friction, case_key)
    series = paths['series']
    print("\n" + "=" * 72)
    print(f"[replay] {label} — {fr_params['label']}")
    print(f"  CSV : {paths['csv']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 72)

    ts_data = load_timeseries_csv(paths['csv'])
    t_sim = ts_data['t']
    H_wh = ts_data['H_wh']

    prev: Optional[Dict] = None
    if os.path.isfile(paths['json']):
        with open(paths['json'], encoding='utf-8') as f:
            prev = json.load(f)

    # 优先用旧 JSON 的 config / 对齐缝深；否则用当前 config + 网格对齐
    if prev and isinstance(prev.get('config'), dict):
        cfg_prev = prev['config']
        x_f_list = list(cfg_prev.get('x_f') or x_f_list)
        L = float(cfg_prev.get('L', WELL_CONFIG['L']))
        ts = float(cfg_prev.get('ts', SIM_CONFIG['ts']))
        dt = float(cfg_prev.get('dt', SIM_CONFIG['dt']))
        kleak = float(cfg_prev.get('kleak', kleak))
    else:
        L = float(WELL_CONFIG['L'])
        ts = float(SIM_CONFIG['ts'])
        dt = float(SIM_CONFIG['dt'])

    cfg = _build_moc_config(friction)
    # 若 JSON 里已有对齐缝深则直接用
    x_f_aligned = None
    if prev and isinstance(prev.get('metrics'), dict):
        xa = prev['metrics'].get('x_f_aligned')
        if xa and len(xa) == len(x_f_list):
            x_f_aligned = [float(x) for x in xa]
    if x_f_aligned is None:
        x_f_aligned = align_frac_to_grid(x_f_list, cfg)

    print(f"  L={L}m, a_adj={cfg.a_adj:.4f} m/s, ts={ts}s, dt={dt}s")
    print(f"  x_f={x_f_list} → aligned={[round(x, 2) for x in x_f_aligned]}")
    print(f"  时程点数={len(t_sim)}, t∈[{t_sim[0]:.3f}, {t_sim[-1]:.3f}]s")

    cep_result, cep_1d, cep_2d_avg = run_cepstrum_analysis_and_match(
        t_sim, H_wh,
        a_adj=cfg.a_adj, L=L, ts=ts, dt=dt,
        x_f_list=x_f_list, x_f_plot=x_f_aligned,
        friction=friction, label=label, kleak=kleak,
        cep_path=paths['cepstrum_png'],
    )

    cep_block = cepstrum_block_for_json(cep_result, cep_1d, cep_2d_avg)
    if prev is None:
        result = {
            "verdicts": {},
            "metrics": {"x_f_aligned": x_f_aligned},
            "cepstrum": cep_block,
            "config": {
                "L": L,
                "a": WELL_CONFIG['wavespeed'],
                "V0": WELL_CONFIG['V0'],
                "H0": WELL_CONFIG['H0'],
                "ts": ts, "dt": dt, "tf": SIM_CONFIG['tf'],
                "x_f": x_f_list,
                "Cf": FRACTURE_CONFIG['Cf'],
                "kleak": kleak,
                "H_ext": FRACTURE_CONFIG['H_ext'],
                "friction": friction,
            },
            "replay": True,
        }
    else:
        result = dict(prev)
        result['cepstrum'] = cep_block
        result['replay'] = True
        # 同步对齐缝深（若原先缺失）
        if 'metrics' not in result or not isinstance(result['metrics'], dict):
            result['metrics'] = {}
        result['metrics']['x_f_aligned'] = x_f_aligned

    with open(paths['json'], 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON 已更新: {paths['json']}")
    print(f"倒谱图已更新: {paths['cepstrum_png']}")
    return result


# ── CLI ───────────────────────────────────────────────────

def _summarize_friction(friction: str, case_keys: List[str], results: Dict) -> None:
    cases = resolve_cases(friction)
    print("\n" + "=" * 72)
    print(f"汇总 ({FRICTION_PARAMS[friction]['label']})")
    print("=" * 72)
    for key in case_keys:
        res = results[key]
        verdicts = res['verdicts']
        n_pass = sum(1 for v in verdicts.values() if v == "PASS")
        n_total = sum(1 for v in verdicts.values() if v != "N/A")
        status = "OK" if n_pass == n_total else "FAIL"
        print(f"  {cases[key]['label']}: {n_pass}/{n_total} PASS [{status}]")


def main():
    parser = argparse.ArgumentParser(
        description='统一 leakoff 多缝验证（steady/brunone + D* 间距 + 滤失）'
    )
    parser.add_argument(
        '--friction', choices=friction_cli_choices(), default='steady',
        help='摩阻/间距键；steady_Dall / brunone_Dall 一次跑完 SPACING_PRESETS_M',
    )
    parser.add_argument(
        '--case', choices=['single', 'dual', 'triple', 'quad', 'quint', 'all'],
        default='all', help='运行哪个 case（默认 all）',
    )
    parser.add_argument(
        '--replay', action='store_true',
        help='不重跑 MOC：从已有 moc_timeseries.csv 重绘倒谱图并更新 moc_leakoff.json',
    )
    args = parser.parse_args()

    friction_keys = expand_friction_keys(args.friction)
    all_results: Dict[str, Dict] = {}
    run_fn = replay_case if args.replay else run_case
    mode = 'replay' if args.replay else 'simulate'

    for friction in friction_keys:
        cases = resolve_cases(friction)
        case_keys = list(cases.keys()) if args.case == 'all' else [args.case]
        results = {}
        for key in case_keys:
            results[key] = run_fn(key, friction=friction)
        if not args.replay:
            _summarize_friction(friction, case_keys, results)
        else:
            print("\n" + "=" * 72)
            print(f"replay 完成 ({FRICTION_PARAMS[friction]['label']})")
            for key in case_keys:
                cep = results[key].get('cepstrum') or {}
                r1 = cep.get('1d_real') or {}
                r2 = cep.get('2d_time_avg') or {}
                print(
                    f"  {cases[key]['label']}: "
                    f"1D {r1.get('n_matched', '?')}/{r1.get('n_fracs', '?')} | "
                    f"2Davg {r2.get('n_matched', '?')}/{r2.get('n_fracs', '?')}"
                )
            print("=" * 72)
        all_results[friction] = results

    if len(friction_keys) > 1:
        print("\n" + "=" * 72)
        print(f"批量完成 ({mode}: {args.friction} → {friction_keys})")
        print("=" * 72)

    return all_results if len(friction_keys) > 1 else all_results[friction_keys[0]]


if __name__ == '__main__':
    main()
