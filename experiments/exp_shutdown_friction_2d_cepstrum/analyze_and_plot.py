# -*- coding: utf-8 -*-
"""
experiments/exp_shutdown_friction_2d_cepstrum/analyze_and_plot.py
----------------------------------------------------------------
对 8 组 MOC 仿真结果进行 2D 倒谱计算、特征提取、定量指标评估与专业科研出图。
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# 项目根目录
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_curr_dir))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from moc_simulate.cepstrum_mocdata import cepstrogram

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_BASE = os.path.join(_root_dir, 'output', 'exp_shutdown_friction_2d_cepstrum')
DATA_DIR = os.path.join(OUTPUT_BASE, 'data')
FIG_DIR = os.path.join(OUTPUT_BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

WAVESPEED = 1450.0
TRUE_FRACTURES = [3000.0, 3050.0, 3100.0]
TC_LIST = [0.01, 0.1, 0.5, 1.0]
FRICTION_LIST = ['steady', 'brunone']

WLEN_SEC = 16.0   # 窗长 [s]
HOP_SEC = 0.5     # 步长 [s]
FS = 1000.0       # 采样频率 [Hz]
WIN_TYPE = 'kaiser'

DEPTH_ZOOM_MIN = 2700.0
DEPTH_ZOOM_MAX = 3400.0


def load_data(tc: float, fric: str):
    tc_str = str(tc).replace('.', 'p')
    case_name = f"tc_{tc_str}s_{fric}"
    csv_path = os.path.join(DATA_DIR, case_name, 'moc_timeseries.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")
    df = pd.read_csv(csv_path)
    t = df['t'].to_numpy(dtype=np.float64)
    h = df['H_wh'].to_numpy(dtype=np.float64)
    return case_name, t, h


def compute_fft(t: np.ndarray, h: np.ndarray, ts: float = 1.0):
    mask = t >= ts
    t_work = t[mask]
    h_work = h[mask] - np.mean(h[mask])
    n = len(h_work)
    freqs = fftfreq(n, d=1.0 / FS)[:n // 2]
    mag = np.abs(fft(h_work)[:n // 2]) / n
    return freqs, mag


def compute_case_2d_cepstrum(t: np.ndarray, h: np.ndarray, ts: float = 1.0):
    mask = t >= ts
    t_work = t[mask]
    h_work = h[mask] - np.mean(h[mask])

    wlen = int(WLEN_SEC * FS)
    hop = int(HOP_SEC * FS)

    ceps, q, t_cep = cepstrogram(h_work, wlen=wlen, hop=hop, fs=FS, win_type=WIN_TYPE)
    depth = q * WAVESPEED / 2.0
    resp_2d = -ceps  # 裂缝反射取正向响应

    # 深度截取到 0 ~ 5000 m
    mask_d = (depth >= 0.0) & (depth <= 5000.0)
    depth_kept = depth[mask_d]
    resp_2d_kept = resp_2d[mask_d, :]

    # 时间平均剖面 (在深度切片上)
    # 取中间稳态演化时间段的均值，避免边缘效应
    profile = np.mean(resp_2d_kept, axis=1)

    return {
        'depth': depth_kept,
        't_cep': t_cep,
        'resp_2d': resp_2d_kept,
        'profile': profile,
    }


def extract_metrics(depth: np.ndarray, profile: np.ndarray):
    """提取三裂缝峰值、峰谷比 PVR 与定位偏差"""
    # 截取裂缝区
    mask_z = (depth >= DEPTH_ZOOM_MIN) & (depth <= DEPTH_ZOOM_MAX)
    d_z = depth[mask_z]
    p_z = profile[mask_z]

    metrics = {}
    peak_depths = []
    peak_heights = []

    # 搜索各真值附近的局部峰值
    for k, x_true in enumerate(TRUE_FRACTURES):
        window = 25.0  # ±25m
        m_k = (d_z >= x_true - window) & (d_z <= x_true + window)
        if np.any(m_k):
            sub_d = d_z[m_k]
            sub_p = p_z[m_k]
            idx_max = np.argmax(sub_p)
            p_val = float(sub_p[idx_max])
            d_val = float(sub_d[idx_max])
        else:
            p_val = np.nan
            d_val = np.nan
        peak_depths.append(d_val)
        peak_heights.append(p_val)
        metrics[f'peak_{k+1}_depth_m'] = d_val
        metrics[f'peak_{k+1}_height'] = p_val
        metrics[f'peak_{k+1}_error_m'] = d_val - x_true if not np.isnan(d_val) else np.nan

    # 提取谷值与 PVR
    # 缝 1-2 之间谷值 (3000 ~ 3050 m)
    m_v12 = (d_z >= 3005.0) & (d_z <= 3045.0)
    v12 = float(np.min(p_z[m_v12])) if np.any(m_v12) else 0.0
    # 缝 2-3 之间谷值 (3050 ~ 3100 m)
    m_v23 = (d_z >= 3055.0) & (d_z <= 3095.0)
    v23 = float(np.min(p_z[m_v23])) if np.any(m_v23) else 0.0

    p1, p2, p3 = peak_heights

    # 当谷值降至 0 或负值时，说明峰间完全跌落回零基线，解离度达到最大，封顶为 10.0
    def calc_pvr(p_min, v_val):
        if p_min <= 0:
            return 1.0
        if v_val <= 0.001 * p_min:
            return 10.0  # 完全解离（谷值接近或低于零）
        return float(np.clip(p_min / v_val, 1.0, 10.0))

    pvr12 = calc_pvr(min(p1, p2), v12)
    pvr23 = calc_pvr(min(p2, p3), v23)
    pvr_mean = 0.5 * (pvr12 + pvr23)

    metrics['valley_12'] = v12
    metrics['valley_23'] = v23
    metrics['PVR_12'] = pvr12
    metrics['PVR_23'] = pvr23
    metrics['PVR_mean'] = pvr_mean

    # 判定分辨能力
    if pvr_mean >= 2.0:
        verdict = "完全清晰分辨 (Resolved)"
    elif pvr_mean >= 1.2:
        verdict = "基本可分辨 (Moderately Resolved)"
    elif pvr_mean >= 1.03:
        verdict = "临界可辨 (Marginally Resolved)"
    else:
        verdict = "融合粘连/无法分辨 (Merged/Unresolved)"
    metrics['resolvability_verdict'] = verdict

    return metrics


def run_analysis():
    print("开始执行 2D 倒谱计算与多维度对比分析...")

    all_results = {}
    metrics_list = []

    for tc in TC_LIST:
        for fric in FRICTION_LIST:
            case_name, t, h = load_data(tc, fric)
            print(f"处理工况: {case_name} ...")
            freqs, mag = compute_fft(t, h)
            cep_res = compute_case_2d_cepstrum(t, h)
            mets = extract_metrics(cep_res['depth'], cep_res['profile'])
            mets['case_name'] = case_name
            mets['Tc'] = tc
            mets['friction'] = fric
            metrics_list.append(mets)

            all_results[case_name] = {
                't': t, 'h': h,
                'freqs': freqs, 'mag': mag,
                'cep': cep_res,
                'metrics': mets,
            }

    # 保存指标表格
    df_metrics = pd.DataFrame(metrics_list)
    csv_out = os.path.join(OUTPUT_BASE, 'metrics_summary.csv')
    json_out = os.path.join(OUTPUT_BASE, 'metrics_summary.json')
    df_metrics.to_csv(csv_out, index=False, encoding='utf-8-sig')
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)
    print(f"指标表已保存: {csv_out}")

    # =========================================================================
    # 图 1: 时域压力波形与 FFT 谱对比 (4 x 2 阵列)
    # =========================================================================
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex='col')
    for i, tc in enumerate(TC_LIST):
        tc_str = str(tc).replace('.', 'p')
        c_st = f"tc_{tc_str}s_steady"
        c_br = f"tc_{tc_str}s_brunone"

        # 左列: 时域波形 (截取前 15 秒展示波前特征)
        ax_t = axes[i, 0]
        ax_t.plot(all_results[c_st]['t'], all_results[c_st]['h'], 'b-', lw=1.2, label='Steady (稳态达西)')
        ax_t.plot(all_results[c_br]['t'], all_results[c_br]['h'], 'r--', lw=1.2, label='Brunone (非定常)')
        ax_t.set_xlim([0, 15.0])
        ax_t.set_ylabel('井口水头 $H_{wh}$ [m]', fontsize=10)
        ax_t.set_title(f'$T_c = {tc}\\,$s 时域水击响应对比', fontsize=11, fontweight='bold')
        ax_t.grid(True, ls=':', alpha=0.6)
        ax_t.legend(loc='upper right', fontsize=9)

        # 右列: 频域 FFT 谱对比 (0 ~ 30 Hz)
        ax_f = axes[i, 1]
        mask_f = all_results[c_st]['freqs'] <= 30.0
        f_axis = all_results[c_st]['freqs'][mask_f]
        mag_st = all_results[c_st]['mag'][mask_f]
        mag_br = all_results[c_br]['mag'][mask_f]

        ax_f.plot(f_axis, mag_st, 'b-', lw=1.2, label='Steady')
        ax_f.plot(f_axis, mag_br, 'r--', lw=1.2, label='Brunone')
        fc_theor = min(30.0, 1.0 / tc)
        ax_f.axvline(fc_theor, color='k', ls=':', lw=1.5, label=f'$f_c \\approx 1/T_c = {1.0/tc:.1f}\\,Hz$')
        ax_f.set_xlim([0, 30.0])
        ax_f.set_ylabel('FFT 幅值 [m]', fontsize=10)
        ax_f.set_title(f'$T_c = {tc}\\,$s 频域幅值谱对比', fontsize=11, fontweight='bold')
        ax_f.grid(True, ls=':', alpha=0.6)
        ax_f.legend(loc='upper right', fontsize=9)

    axes[3, 0].set_xlabel('时间 $t$ [s]', fontsize=11)
    axes[3, 1].set_xlabel('频率 $f$ [Hz]', fontsize=11)
    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, 'fig1_time_domain_and_fft.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"图1已保存: {fig1_path}")

    # =========================================================================
    # 图 2: 二维倒谱时频-深度云图矩阵 (2 x 4 阵列)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    for col, tc in enumerate(TC_LIST):
        tc_str = str(tc).replace('.', 'p')
        for row, fric in enumerate(['steady', 'brunone']):
            case_name = f"tc_{tc_str}s_{fric}"
            cep_data = all_results[case_name]['cep']
            d_arr = cep_data['depth']
            t_arr = cep_data['t_cep']
            resp = cep_data['resp_2d']

            # 截取裂缝邻域 2600 ~ 3500 m
            mask_d = (d_arr >= 2600.0) & (d_arr <= 3500.0)
            d_sub = d_arr[mask_d]
            resp_sub = resp[mask_d, :]

            # 归一化便于对比云图
            vmax = np.percentile(resp_sub, 99.5)
            vmin = np.percentile(resp_sub, 10.0)

            ax = axes[row, col]
            im = ax.pcolormesh(t_arr, d_sub, resp_sub, shading='auto', cmap='plasma',
                               vmin=vmin, vmax=vmax)
            # 标注 3 条裂缝真实深度
            for xf in TRUE_FRACTURES:
                ax.axhline(xf, color='cyan', ls='--', lw=1.0, alpha=0.85)

            fric_label = "稳态 (Steady)" if fric == 'steady' else "非稳态 (Brunone)"
            ax.set_title(f"$T_c={tc}\\,s$ | {fric_label}", fontsize=11, fontweight='bold')
            ax.grid(True, ls=':', alpha=0.3, color='white')
            if col == 0:
                ax.set_ylabel('当量深度 $d$ [m]', fontsize=11)
            ax.set_xlabel('倒谱分析窗中心时间 $t$ [s]', fontsize=10)

    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, 'fig2_2d_cepstrogram_matrix.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"图2已保存: {fig2_path}")

    # =========================================================================
    # 图 3: 裂缝区时间平均倒谱剖面对比 (4 个子图，每个子图叠合 Steady vs Brunone)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for i, tc in enumerate(TC_LIST):
        ax = axes[i]
        tc_str = str(tc).replace('.', 'p')
        c_st = f"tc_{tc_str}s_steady"
        c_br = f"tc_{tc_str}s_brunone"

        cep_st = all_results[c_st]['cep']
        cep_br = all_results[c_br]['cep']

        d_arr = cep_st['depth']
        mask_z = (d_arr >= DEPTH_ZOOM_MIN) & (d_arr <= DEPTH_ZOOM_MAX)
        d_z = d_arr[mask_z]
        prof_st = cep_st['profile'][mask_z]
        prof_br = cep_br['profile'][mask_z]

        # 归一化到各自主峰高度以便对比形态细节
        scale_st = np.max(prof_st) if np.max(prof_st) > 0 else 1.0
        scale_br = np.max(prof_br) if np.max(prof_br) > 0 else 1.0

        ax.plot(d_z, prof_st / scale_st, 'b-', lw=1.8, label=f'Steady (峰值={scale_st:.4f})')
        ax.plot(d_z, prof_br / scale_br, 'r--', lw=1.8, label=f'Brunone (峰值={scale_br:.4f})')

        # 真实裂缝位置标注
        for k, xf in enumerate(TRUE_FRACTURES):
            ax.axvline(xf, color='gray', ls=':', lw=1.5, label='真值缝位' if k == 0 else "")
            ax.text(xf, 1.06, f"{int(xf)}m", color='dimgray', ha='center', fontsize=9, fontweight='bold')

        # 标注 PVR 指标
        m_st = all_results[c_st]['metrics']
        m_br = all_results[c_br]['metrics']
        txt = (f"Steady: PVR={m_st['PVR_mean']:.2f} ({m_st['resolvability_verdict']})\n"
               f"Brunone: PVR={m_br['PVR_mean']:.2f} ({m_br['resolvability_verdict']})")
        ax.text(0.03, 0.80, txt, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85))

        ax.set_title(f"停泵历时 $T_c = {tc}\\,s$ 裂缝倒谱识别剖面", fontsize=11, fontweight='bold')
        ax.set_xlabel('深度 $d$ [m]', fontsize=10)
        ax.set_ylabel('归一化倒谱响应 (相对主峰)', fontsize=10)
        ax.set_xlim([DEPTH_ZOOM_MIN, DEPTH_ZOOM_MAX])
        ax.set_ylim([-0.08, 1.22])
        ax.grid(True, ls='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, 'fig3_fracture_zoom_profiles.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"图3已保存: {fig3_path}")

    # =========================================================================
    # 图 4: 量化指标对比柱状图 (PVR 峰谷比与主峰绝对响应)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x_indices = np.arange(len(TC_LIST))
    bar_width = 0.35

    # 提取 PVR 数据
    pvr_st = [all_results[f"tc_{str(tc).replace('.', 'p')}s_steady"]['metrics']['PVR_mean'] for tc in TC_LIST]
    pvr_br = [all_results[f"tc_{str(tc).replace('.', 'p')}s_brunone"]['metrics']['PVR_mean'] for tc in TC_LIST]

    # 提取峰 1 高度数据
    h1_st = [all_results[f"tc_{str(tc).replace('.', 'p')}s_steady"]['metrics']['peak_1_height'] for tc in TC_LIST]
    h1_br = [all_results[f"tc_{str(tc).replace('.', 'p')}s_brunone"]['metrics']['peak_1_height'] for tc in TC_LIST]

    # 子图 1: 平均 PVR 峰谷比 (分辨率)
    ax1.bar(x_indices - bar_width/2, pvr_st, bar_width, label='Steady (稳态)', color='royalblue', alpha=0.85)
    ax1.bar(x_indices + bar_width/2, pvr_br, bar_width, label='Brunone (非定常)', color='crimson', alpha=0.85)
    ax1.axhline(2.0, color='green', ls='--', lw=1.5, label='清晰分辨门限 (PVR=2.0)')
    ax1.axhline(1.2, color='orange', ls=':', lw=1.5, label='临界分辨门限 (PVR=1.2)')
    ax1.set_ylim([0, 11])
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"{tc} s" for tc in TC_LIST], fontsize=10)
    ax1.set_xlabel('停泵时间 $T_c$', fontsize=11)
    ax1.set_ylabel('平均峰谷比 $\\mathrm{PVR}_{mean}$ (上限 10 为完全解离)', fontsize=10)
    ax1.set_title('裂缝空间分辨能力对比 (PVR 指标)', fontsize=12, fontweight='bold')
    ax1.grid(True, ls=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9)

    # 子图 2: 首缝倒谱峰值强度对比
    ax2.bar(x_indices - bar_width/2, h1_st, bar_width, label='Steady (稳态)', color='royalblue', alpha=0.85)
    ax2.bar(x_indices + bar_width/2, h1_br, bar_width, label='Brunone (非定常)', color='crimson', alpha=0.85)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"{tc} s" for tc in TC_LIST], fontsize=10)
    ax2.set_xlabel('停泵时间 $T_c$', fontsize=11)
    ax2.set_ylabel('首缝倒谱峰高度 (绝对幅值)', fontsize=11)
    ax2.set_title('倒谱能量响应强度对比', fontsize=12, fontweight='bold')
    ax2.grid(True, ls=':', alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig4_path = os.path.join(FIG_DIR, 'fig4_resolvability_metrics.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"图4已保存: {fig4_path}")

    print("=" * 70)
    print("分析完成！所有高清图件与量化表格已生成在:")
    print(f"图件目录: {FIG_DIR}")
    print(f"数据汇总: {csv_out}")
    print("=" * 70)


if __name__ == '__main__':
    run_analysis()
