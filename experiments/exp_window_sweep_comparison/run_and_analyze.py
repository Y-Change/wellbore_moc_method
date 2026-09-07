# -*- coding: utf-8 -*-
"""
experiments/exp_window_sweep_comparison/run_and_analyze.py
----------------------------------------------------------
5000m MOC 三裂缝稳态 vs Brunone 非稳态摩阻在不同滑窗大小 (10s, 20s, 40s, 60s) 下的
二维倒谱对比实验主程序。包含：
1. 100s MOC 水击仿真 (steady 与 brunone)
2. 4 种滑窗长度下的 2D 倒谱计算与时间平均剖面提取
3. 关键指标提取：定位偏差 (Delta x, delta x)、峰谷比 (PVR)、半高宽 (FWHM)、信噪比 (SNR)
4. Nature Figure 规范绘图 (Fig1: 2x4 云图矩阵, Fig2: 剖面对照, Fig3: 指标演化趋势)
"""
from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec

# 引入项目根目录
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_curr_dir))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore
from moc_simulate.config import WELL_CONFIG
from moc_simulate.cepstrum_mocdata import cepstrogram

# Nature Figure 格式配置
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "SimHei"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10,
    "axes.linewidth": 0.75,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.dpi": 300,
})

# 目录配置
OUTPUT_BASE = os.path.join(_root_dir, 'output', 'exp_window_sweep_comparison')
DATA_DIR = os.path.join(OUTPUT_BASE, 'data')
FIG_DIR = os.path.join(OUTPUT_BASE, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 物理与网格参数
WELLBORE_LENGTH = 5000.0       # [m]
WAVESPEED = 1450.0             # [m/s]
DIAMETER = 0.1397              # [m]
V0 = 1.0                       # [m/s]
H0 = 300.0                     # [m]
TS = 1.0                       # [s] 停泵开始时间
TC = 0.1                       # [s] 快速线性关井
TF = 100.0                     # [s] 满足 60s 滑窗推进的仿真总长
DT = 1.0e-3                    # [s]
FS = 1.0 / DT                  # 1000 Hz

# 裂缝参数
TRUE_FRACTURES = [3000.0, 3050.0, 3100.0]
N_FRAC = len(TRUE_FRACTURES)
CF_LIST = [1.0e-5] * N_FRAC
KLEAK_LIST = [1.0e-4] * N_FRAC
H_EXT = 100.0

# 倒谱与滑窗参数
WINDOW_SIZES = [10.0, 20.0, 40.0, 60.0]
HOP_SEC = 0.5                  # [s]
WIN_TYPE = 'hamming'
DEPTH_ZOOM_MIN = 2700.0
DEPTH_ZOOM_MAX = 3400.0

# Nature 配色方案
PALETTE = {
    "steady": "#0F4D92",       # 经典深蓝
    "brunone": "#B64342",      # 学术深红
    "win_10": "#4575B4",       # 浅钢蓝
    "win_20": "#2C7BB6",       # 钴蓝
    "win_40": "#FDAE61",       # 暖杏黄
    "win_60": "#D7191C",       # 艳红
    "neutral_gray": "#7F8C8D",
    "bg_light": "#F8F9F9",
    "fracture_cyan": "#00E5FF",
}

WIN_COLORS = {
    10.0: "#4575B4",
    20.0: "#2C7BB6",
    40.0: "#D95F02",
    60.0: "#B64342",
}


# =====================================================================
# 1. 仿真运行模块
# =====================================================================
def run_simulation_case(friction_model: str) -> Tuple[np.ndarray, np.ndarray]:
    csv_file = os.path.join(DATA_DIR, f"moc_timeseries_{friction_model}_100s.csv")
    if os.path.exists(csv_file):
        print(f"[{friction_model}] 发现已存在仿真数据，直接加载: {csv_file}")
        df = pd.read_csv(csv_file)
        return df['t'].to_numpy(dtype=np.float64), df['H_wh'].to_numpy(dtype=np.float64)

    print(f"\n=======================================================")
    print(f"开始运行 {friction_model} 摩阻 100s MOC 仿真...")
    print(f"=======================================================")
    cfg = MocConfig(
        wellbore_length=WELLBORE_LENGTH,
        wellbore_diameter=DIAMETER,
        fluid_density=WELL_CONFIG['fluid_density'],
        fluid_viscosity=WELL_CONFIG['fluid_viscosity'],
        wavespeed=WAVESPEED,
        roughness_height=WELL_CONFIG['roughness_height'],
        friction_model=friction_model,
        dt=DT,
        tf=TF,
        wellhead_bc='ramp',
        pump_shut_time=TS,
        pump_closure_duration=TC,
        initial_velocity=V0,
        initial_head=H0,
        theta=0.0,
        toe_bc='reservoir',
        toe_head=H0,
    )

    t0 = time.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=TRUE_FRACTURES,
        fracture_Cf=CF_LIST,
        fracture_kleak=KLEAK_LIST,
        H_ext=H_EXT,
        store_full_field=False,
    )
    elapsed = time.time() - t0
    print(f"  -> {friction_model} 仿真完成，用时: {elapsed:.2f} s")

    t_arr = res["timestamps"]
    H_wh = res["wellhead_head"]
    V_wh = res["wellhead_velocity"]
    Q_wh = V_wh * cfg.area

    df = pd.DataFrame({
        't': t_arr,
        'H_wh': H_wh,
        'V_wh': V_wh,
        'Q_wh': Q_wh,
    })
    df.to_csv(csv_file, index=False)
    print(f"  -> 数据已归档至: {csv_file}")
    return t_arr, H_wh


# =====================================================================
# 2. 二维短时倒谱计算
# =====================================================================
def compute_2d_cepstrum(t: np.ndarray, h: np.ndarray, wlen_sec: float, hop_sec: float = HOP_SEC):
    mask = t >= TS
    t_work = t[mask]
    h_work = h[mask] - np.mean(h[mask])

    wlen = int(round(wlen_sec * FS))
    hop = int(round(hop_sec * FS))

    ceps, q, t_cep = cepstrogram(h_work, wlen=wlen, hop=hop, fs=FS, win_type=WIN_TYPE)
    depth = q * WAVESPEED / 2.0
    resp_2d = -ceps  # 反射峰取正向响应

    # 深度截取到 0 ~ 5000m
    mask_d = (depth >= 0.0) & (depth <= WELLBORE_LENGTH)
    depth_kept = depth[mask_d]
    resp_2d_kept = resp_2d[mask_d, :]

    # 时间累积平均深度剖面
    profile = np.mean(resp_2d_kept, axis=1)

    return {
        'depth': depth_kept,
        't_cep': t_cep,
        'resp_2d': resp_2d_kept,
        'profile': profile,
    }


# =====================================================================
# 3. 定量评估指标提取
# =====================================================================
def evaluate_metrics(depth: np.ndarray, profile: np.ndarray, fric_type: str, wlen: float) -> Dict:
    mask_z = (depth >= DEPTH_ZOOM_MIN) & (depth <= DEPTH_ZOOM_MAX)
    d_z = depth[mask_z]
    p_z = profile[mask_z]

    metrics = {
        'friction': fric_type,
        'wlen_s': wlen,
    }

    peak_depths = []
    peak_heights = []

    # 1. 独立区间精确搜索 3 条裂缝局部峰值（防止前缝宽尾部渗漏）
    # 缝间距 50m，各峰有效搜索区间设定为 [xtrue - 15m, xtrue + 25m]
    for k, x_true in enumerate(TRUE_FRACTURES):
        m_k = (d_z >= x_true - 15.0) & (d_z <= x_true + 25.0)
        if np.any(m_k):
            sub_d = d_z[m_k]
            sub_p = p_z[m_k]
            idx_max = np.argmax(sub_p)
            d_val = float(sub_d[idx_max])
            p_val = float(sub_p[idx_max])
        else:
            d_val = np.nan
            p_val = np.nan
        peak_depths.append(d_val)
        peak_heights.append(p_val)
        metrics[f'peak_{k+1}_depth_m'] = d_val
        metrics[f'peak_{k+1}_error_m'] = d_val - x_true if not np.isnan(d_val) else np.nan
        metrics[f'peak_{k+1}_height'] = p_val

    # 2. 计算峰谷比 PVR
    # 缝 1-2 谷值 (3020 ~ 3040 m)
    m_v12 = (d_z >= 3020.0) & (d_z <= 3040.0)
    v12 = float(np.min(p_z[m_v12])) if np.any(m_v12) else 0.0
    # 缝 2-3 谷值 (3070 ~ 3090 m)
    m_v23 = (d_z >= 3070.0) & (d_z <= 3090.0)
    v23 = float(np.min(p_z[m_v23])) if np.any(m_v23) else 0.0

    p1, p2, p3 = peak_heights

    def get_pvr(p_min, v_val):
        if p_min <= 0:
            return 1.0
        if v_val <= 0.001 * p_min:
            return 10.0  # 谷值降至基线以下，完全分离
        return float(np.clip(p_min / v_val, 1.0, 10.0))

    metrics['valley_12'] = v12
    metrics['valley_23'] = v23
    metrics['pvr_12'] = get_pvr(min(p1, p2), v12)
    metrics['pvr_23'] = get_pvr(min(p2, p3), v23)

    # 3. 计算首缝 FWHM (半高全宽)
    if not np.isnan(d_val) and p1 > 0:
        half_h = p1 / 2.0
        m_p1 = (d_z >= 2970.0) & (d_z <= 3030.0)
        dp1 = d_z[m_p1]
        pp1 = p_z[m_p1]
        above_half = dp1[pp1 >= half_h]
        if len(above_half) >= 2:
            metrics['fwhm_1_m'] = float(above_half[-1] - above_half[0])
        else:
            metrics['fwhm_1_m'] = np.nan
    else:
        metrics['fwhm_1_m'] = np.nan

    # 4. 计算倒谱信噪比 SNR
    # 选取背景区间 2400 ~ 2800 m
    m_bg = (depth >= 2400.0) & (depth <= 2800.0)
    if np.any(m_bg):
        bg_std = float(np.std(profile[m_bg]))
        metrics['snr'] = float(p1 / max(bg_std, 1e-12))
    else:
        metrics['snr'] = np.nan

    return metrics


# =====================================================================
# 4. Nature Figure 绘图模块
# =====================================================================
def plot_figure_1_matrix(results: Dict):
    """
    Figure 1: 2 x 4 二维倒谱云图矩阵 (Nature Figure 通栏 180mm 标准)
    上行: 稳态摩阻 (10s, 20s, 40s, 60s)
    下行: Brunone 非稳态摩阻 (10s, 20s, 40s, 60s)
    """
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), sharey=True, sharex='col')
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.90, wspace=0.15, hspace=0.22)

    panel_tags = [
        ['a', 'b', 'c', 'd'],
        ['e', 'f', 'g', 'h']
    ]

    fric_names = ['steady', 'brunone']
    fric_titles = ['Steady Darcy', 'Brunone Unsteady']

    for row, fric in enumerate(fric_names):
        for col, wlen in enumerate(WINDOW_SIZES):
            ax = axes[row, col]
            cdata = results[fric][wlen]
            d = cdata['depth']
            t_cep = cdata['t_cep']
            resp = cdata['resp_2d']

            mask_d = (d >= DEPTH_ZOOM_MIN) & (d <= DEPTH_ZOOM_MAX)
            d_sub = d[mask_d]
            resp_sub = resp[mask_d, :]

            vmin = np.percentile(resp_sub, 5.0)
            vmax = np.percentile(resp_sub, 99.2)

            im = ax.pcolormesh(t_cep, d_sub, resp_sub, shading='auto', cmap='magma',
                               vmin=vmin, vmax=vmax, rasterized=True)

            # 裂缝真值基准线
            for xf in TRUE_FRACTURES:
                ax.axhline(xf, color=PALETTE["fracture_cyan"], ls='--', lw=0.75, alpha=0.9)

            # Nature 子图编号 a, b, c... (常规文本 + 粗体)
            tag = panel_tags[row][col]
            ax.text(0.04, 0.93, tag, transform=ax.transAxes,
                    fontsize=9.5, fontweight='bold', va='top', ha='left', color='white')

            # 标题设置
            if row == 0:
                ax.set_title(f"$T_{{\\mathrm{{win}}}} = {int(wlen)}\\,\\mathrm{{s}}$", fontsize=8.5, pad=5)

            if col == 0:
                ax.set_ylabel(f"{fric_titles[row]}\nDepth $x$ (m)", fontsize=8)

            if row == 1:
                ax.set_xlabel("Time $t$ (s)", fontsize=8)

            ax.set_ylim(DEPTH_ZOOM_MIN, DEPTH_ZOOM_MAX)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    # 添加全局色条
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Cepstral Amplitude $C(\\tau, t)$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    fig_png = os.path.join(FIG_DIR, 'fig1_2d_cepstrogram_matrix.png')
    fig_pdf = os.path.join(FIG_DIR, 'fig1_2d_cepstrogram_matrix.pdf')
    fig_svg = os.path.join(FIG_DIR, 'fig1_2d_cepstrogram_matrix.svg')
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.savefig(fig_svg)
    plt.close()
    print(f"Figure 1 生成成功: {fig_png} / {fig_pdf}")


def plot_figure_2_profiles(results: Dict):
    """
    Figure 2: 裂缝区 (2800m ~ 3300m) 时间平均深度剖面对照 (Nature Figure 3 栏标准)
    Panel a: Steady 摩阻下 4 种窗长剖面
    Panel b: Brunone 摩阻下 4 种窗长剖面
    Panel c: 20s 黄金窗长下 Steady vs Brunone 直接同轴对比 (标注峰漂移 delta x 与谷底抬升)
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.28)

    # Panel a: Steady
    ax = axes[0]
    ax.text(-0.18, 1.05, "a", transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')
    for wlen in WINDOW_SIZES:
        cdata = results['steady'][wlen]
        mask_d = (cdata['depth'] >= 2850.0) & (cdata['depth'] <= 3250.0)
        ax.plot(cdata['depth'][mask_d], cdata['profile'][mask_d],
                label=f"$T_{{\\mathrm{{win}}}}={int(wlen)}\\,\\mathrm{{s}}$",
                color=WIN_COLORS[wlen], lw=1.1)
    for xf in TRUE_FRACTURES:
        ax.axvline(xf, color='gray', ls=':', lw=0.8, alpha=0.8)
    ax.set_title("Steady Darcy Friction", fontsize=8.5)
    ax.set_xlabel("Depth $x$ (m)", fontsize=8)
    ax.set_ylabel("Time-Avg Cepstrum $P(x)$", fontsize=8)
    ax.legend(frameon=False, fontsize=6.8, loc='upper right')
    ax.set_xlim(2880, 3220)

    # Panel b: Brunone
    ax = axes[1]
    ax.text(-0.18, 1.05, "b", transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')
    for wlen in WINDOW_SIZES:
        cdata = results['brunone'][wlen]
        mask_d = (cdata['depth'] >= 2850.0) & (cdata['depth'] <= 3250.0)
        ax.plot(cdata['depth'][mask_d], cdata['profile'][mask_d],
                label=f"$T_{{\\mathrm{{win}}}}={int(wlen)}\\,\\mathrm{{s}}$",
                color=WIN_COLORS[wlen], lw=1.1)
    for xf in TRUE_FRACTURES:
        ax.axvline(xf, color='gray', ls=':', lw=0.8, alpha=0.8)
    ax.set_title("Brunone Unsteady Friction", fontsize=8.5)
    ax.set_xlabel("Depth $x$ (m)", fontsize=8)
    ax.legend(frameon=False, fontsize=6.8, loc='upper right')
    ax.set_xlim(2880, 3220)

    # Panel c: Steady vs Brunone Direct Comparison at Twin = 20s
    ax = axes[2]
    ax.text(-0.18, 1.05, "c", transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')
    cs = results['steady'][20.0]
    cb = results['brunone'][20.0]
    mask_d = (cs['depth'] >= 2950.0) & (cs['depth'] <= 3180.0)
    d_eval = cs['depth'][mask_d]

    ax.plot(d_eval, cs['profile'][mask_d], color=PALETTE["steady"], lw=1.3, label="Steady (20s)")
    ax.plot(d_eval, cb['profile'][mask_d], color=PALETTE["brunone"], lw=1.3, ls='--', label="Brunone (20s)")
    for xf in TRUE_FRACTURES:
        ax.axvline(xf, color='black', ls=':', lw=0.8, alpha=0.6)

    # 标注裂缝序号与参考线
    y_max_c = max(np.max(cs['profile'][mask_d]), np.max(cb['profile'][mask_d]))
    ax.set_ylim(-0.0008, 0.0050)
    ax.set_xlim(2960, 3190)

    for i, xf in enumerate(TRUE_FRACTURES):
        ax.text(xf, 0.0044, f"Frac {i+1}", fontsize=7.0, ha='center', color='#111111', fontweight='bold')

    ax.set_title("Friction Comparison ($T_{{\\mathrm{{win}}}}=20\\,\\mathrm{{s}}$)", fontsize=8.5)
    ax.set_xlabel("Depth $x$ (m)", fontsize=8)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.9, fontsize=7.0, loc='lower left')

    for ax in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    fig_png = os.path.join(FIG_DIR, 'fig2_depth_profiles_zoom.png')
    fig_pdf = os.path.join(FIG_DIR, 'fig2_depth_profiles_zoom.pdf')
    fig_svg = os.path.join(FIG_DIR, 'fig2_depth_profiles_zoom.svg')
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.savefig(fig_svg)
    plt.close()
    print(f"Figure 2 生成成功: {fig_png} / {fig_pdf}")


def plot_figure_3_metrics_trends(df_metrics: pd.DataFrame):
    """
    Figure 3: 4 面板定量指标对比趋势 (Nature 2x2 规范)
    Panel a: 定位偏差 Delta x vs 窗长
    Panel b: 相对漂移 delta x (Brunone - Steady) vs 窗长
    Panel c: 峰谷比 PVR vs 窗长 (解离度)
    Panel d: 信噪比 SNR vs 窗长 (能量稀释展示)
    """
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.92, wspace=0.25, hspace=0.35)

    df_steady = df_metrics[df_metrics['friction'] == 'steady'].sort_values('wlen_s')
    df_brunone = df_metrics[df_metrics['friction'] == 'brunone'].sort_values('wlen_s')

    wlens = df_steady['wlen_s'].to_numpy()

    # Panel a: 绝对定位偏差 Delta x (首缝)
    ax = axes[0, 0]
    ax.text(-0.18, 1.06, "a", transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')
    ax.plot(wlens, df_steady['peak_1_error_m'], marker='o', color=PALETTE["steady"], lw=1.2, label='Steady $\\Delta x_1$')
    ax.plot(wlens, df_brunone['peak_1_error_m'], marker='s', color=PALETTE["brunone"], lw=1.2, label='Brunone $\\Delta x_1$')
    ax.axhline(0, color='gray', ls='--', lw=0.7)
    ax.set_ylabel("Localization Error $\\Delta x_1$ (m)", fontsize=8)
    ax.set_xlabel("Window Length $T_{\\mathrm{win}}$ (s)", fontsize=8)
    ax.legend(frameon=False, fontsize=7.0)
    ax.set_xticks(WINDOW_SIZES)

    # Panel b: 相对偏深漂移 delta x = Brunone - Steady
    ax = axes[0, 1]
    ax.text(-0.18, 1.06, "b", transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')
    delta_x1 = df_brunone['peak_1_depth_m'].to_numpy() - df_steady['peak_1_depth_m'].to_numpy()
    delta_x2 = df_brunone['peak_2_depth_m'].to_numpy() - df_steady['peak_2_depth_m'].to_numpy()
    delta_x3 = df_brunone['peak_3_depth_m'].to_numpy() - df_steady['peak_3_depth_m'].to_numpy()

    ax.plot(wlens, delta_x1, marker='^', color='#B64342', lw=1.2, label='Frac 1 ($x=3000\\,\\mathrm{m}$)')
    ax.plot(wlens, delta_x2, marker='v', color='#D95F02', lw=1.2, label='Frac 2 ($x=3050\\,\\mathrm{m}$)')
    ax.plot(wlens, delta_x3, marker='d', color='#7570B3', lw=1.2, label='Frac 3 ($x=3100\\,\\mathrm{m}$)')
    ax.set_ylabel("Systematic Bias $\\delta x$ (m)", fontsize=8)
    ax.set_xlabel("Window Length $T_{\\mathrm{win}}$ (s)", fontsize=8)
    ax.legend(frameon=False, fontsize=6.8)
    ax.set_xticks(WINDOW_SIZES)

    # Panel c: 峰谷比 PVR
    ax = axes[1, 0]
    ax.text(-0.18, 1.06, "c", transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')
    ax.plot(wlens, df_steady['pvr_12'], marker='o', color=PALETTE["steady"], lw=1.2, label='Steady PVR$_{12}$')
    ax.plot(wlens, df_brunone['pvr_12'], marker='s', color=PALETTE["brunone"], lw=1.2, label='Brunone PVR$_{12}$')
    ax.axhline(1.2, color='darkred', ls=':', lw=0.8, label='Resolution Threshold')
    ax.set_ylabel("Peak-to-Valley Ratio (PVR$_{12}$)", fontsize=8)
    ax.set_xlabel("Window Length $T_{\\mathrm{win}}$ (s)", fontsize=8)
    ax.legend(frameon=False, fontsize=6.8)
    ax.set_xticks(WINDOW_SIZES)

    # Panel d: 信噪比 SNR
    ax = axes[1, 1]
    ax.text(-0.18, 1.06, "d", transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')
    ax.plot(wlens, df_steady['snr'], marker='o', color=PALETTE["steady"], lw=1.2, label='Steady SNR')
    ax.plot(wlens, df_brunone['snr'], marker='s', color=PALETTE["brunone"], lw=1.2, label='Brunone SNR')
    ax.set_ylabel("Cepstral SNR", fontsize=8)
    ax.set_xlabel("Window Length $T_{\\mathrm{win}}$ (s)", fontsize=8)
    ax.legend(frameon=False, fontsize=7.0)
    ax.set_xticks(WINDOW_SIZES)

    for r in range(2):
        for c in range(2):
            axes[r, c].yaxis.set_minor_locator(AutoMinorLocator(2))

    fig_png = os.path.join(FIG_DIR, 'fig3_metrics_trends.png')
    fig_pdf = os.path.join(FIG_DIR, 'fig3_metrics_trends.pdf')
    fig_svg = os.path.join(FIG_DIR, 'fig3_metrics_trends.svg')
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.savefig(fig_svg)
    plt.close()
    print(f"Figure 3 生成成功: {fig_png} / {fig_pdf}")


# =====================================================================
# 主函数
# =====================================================================
def main():
    print("=" * 70)
    print("5000m MOC 三裂缝稳态 vs Brunone 摩阻不同滑窗倒谱响应实验启动")
    print("=" * 70)

    # 1. 运行仿真
    sim_data = {}
    for fric in ['steady', 'brunone']:
        t_arr, h_wh = run_simulation_case(fric)
        sim_data[fric] = (t_arr, h_wh)

    # 2. 计算各窗长下的 2D 倒谱并提取指标
    results = {'steady': {}, 'brunone': {}}
    metrics_records = []

    for fric in ['steady', 'brunone']:
        t_arr, h_wh = sim_data[fric]
        print(f"\n--- 计算 [{fric}] 摩阻下的 2D 倒谱 ---")
        for wlen in WINDOW_SIZES:
            t_start = time.time()
            cep_res = compute_2d_cepstrum(t_arr, h_wh, wlen_sec=wlen, hop_sec=HOP_SEC)
            results[fric][wlen] = cep_res
            met = evaluate_metrics(cep_res['depth'], cep_res['profile'], fric_type=fric, wlen=wlen)
            metrics_records.append(met)
            elap = time.time() - t_start
            print(f"  wlen={wlen:4.1f}s | 用时: {elap:.2f}s | "
                  f"Peak1={met['peak_1_depth_m']:.1f}m (Err={met['peak_1_error_m']:+.2f}m) | "
                  f"PVR12={met['pvr_12']:.2f} | SNR={met['snr']:.2f}")

    df_metrics = pd.DataFrame(metrics_records)
    csv_metrics_path = os.path.join(DATA_DIR, 'window_metrics_comparison.csv')
    df_metrics.to_csv(csv_metrics_path, index=False)
    print(f"\n指标汇总已保存至: {csv_metrics_path}")

    # 3. 绘制 Nature Figure 图件
    print("\n--- 正在生成 Nature Figure 标准插图 ---")
    plot_figure_1_matrix(results)
    plot_figure_2_profiles(results)
    plot_figure_3_metrics_trends(df_metrics)

    print("\n" + "=" * 70)
    print("所有实验计算与出图全部顺利完成！")
    print(f"数据目录: {DATA_DIR}")
    print(f"图件目录: {FIG_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
