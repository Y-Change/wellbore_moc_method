# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.fft import fft, fftfreq, ifft
import scipy.signal

# Bootstrap
_d = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_d)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
_root = os.path.dirname(_parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore
from moc_simulate.cepstrum_mocdata import preprocess_moc_head, cepstrogram, _quefrency_to_depth
from moc_simulate.config import CASES, FRICTION_PARAMS, CEPSTRUM_CONFIG

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_simulation(args):
    # 从已经仿真好的 CSV 读取数据
    folder_name = f"{args.friction}_{args.d_size}"
    csv_path = os.path.join(_root, 'output', 'leakoff', folder_name, args.case, 'moc_timeseries.csv')
    print(f"正在从 {csv_path} 读取仿真数据...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"错误: 找不到文件 {csv_path}")
        
    df = pd.read_csv(csv_path)
    timestamps = df['t'].values
    wellhead_head = df['H_wh'].values
    
    # 重新构造 config 对象
    config = MocConfig(
        wellbore_length=5000.0,
        wavespeed=1450.0,
        initial_velocity=1.0,
        initial_head=300.0,
        dt=1.0e-3,
        tf=100.0,
        friction_model=args.friction,
    )
    
    # 自动根据 case 设定裂缝位置
    if args.case in CASES:
        x_f_list = CASES[args.case].get('x_f_list', [])
    else:
        x_f_list = []
    
    res = {
        'timestamps': timestamps,
        'wellhead_head': wellhead_head
    }
    
    print(f"数据加载完成，共 {len(timestamps)} 个时间步。")
    return res, config, x_f_list

def compute_fracture_energy_and_weights(res, config):
    time = res['timestamps']
    head = res['wellhead_head']
    fs = 1.0 / config.dt
    v = config.wavespeed
    L = config.wellbore_length
    
    pre = preprocess_moc_head(time, head, fs=fs, ts=1.0)
    p_work = pre['h_detrended']
    
    wlen_sec = CEPSTRUM_CONFIG.get('wlen_sec', 30.0)
    hop_sec = CEPSTRUM_CONFIG.get('hop_sec', 3.0)
    win_type = CEPSTRUM_CONFIG.get('win_type', 'hamming')
    
    wlen = int(wlen_sec * fs)
    hop = int(hop_sec * fs)
    
    print("计算 2D 倒谱图...")
    C, q, t_cep = cepstrogram(p_work, wlen=wlen, hop=hop, fs=fs, win_type=win_type)
    depth = _quefrency_to_depth(q, v)
    
    # 1. 计算倒谱特征区能量（裂缝反射能量）
    # 关注可能存在裂缝的深度区间（例如 1000m - L）
    depth_mask = (depth >= 2000.0) & (depth <= L + 200.0)
    
    # 改为使用区间内的最大峰值（Peak），而不是 RMS。因为裂缝特征是稀疏的尖峰，
    # 全局 RMS 会被不衰减的背景计算噪声稀释。
    E_ceps = np.max(-C[depth_mask, :], axis=0)
    
    # 平滑能量曲线
    E_ceps_smoothed = np.convolve(E_ceps, np.ones(3)/3, mode='same')
    
    # 2. 计算映射关注权重 W(t)
    # 取前几个时间窗的最大能量作为参考基准
    E_ref = np.max(E_ceps_smoothed[:3]) 
    E_min = np.min(E_ceps_smoothed)
    
    # 设置一个绝对门限，低于基准能量的 20% 认为信号衰减严重
    threshold = E_min + 0.20 * (E_ref - E_min)
    
    # 用 Sigmoid 函数平滑映射，陡峭度 alpha
    alpha = 20.0 / (E_ref - E_min + 1e-12)
    weights = 1.0 / (1.0 + np.exp(-alpha * (E_ceps_smoothed - threshold)))
    
    return {
        'C': C, 'q': q, 'depth': depth, 't_cep': t_cep,
        'E_ceps': E_ceps_smoothed, 'weights': weights,
        'wlen': wlen, 'hop': hop, 'fs': fs, 'v': v, 'L': L,
        'p_work': p_work, 'threshold': threshold
    }

def plot_weight_evolution(data, out_dir):
    t_cep = data['t_cep']
    E_ceps = data['E_ceps']
    weights = data['weights']
    threshold = data['threshold']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t_cep, E_ceps, 'b-', lw=2, label='裂缝反射能量 $E_{frac}(t)$')
    ax1.axhline(threshold, color='r', linestyle='--', label='能量衰减门限')
    ax1.set_ylabel('倒谱区能量')
    ax1.set_title('Brunone 非定常摩阻下裂缝反射能量衰减')
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.6)
    
    ax2.plot(t_cep, weights, 'g-', lw=2, label='关注权重 $W(t)$')
    ax2.set_xlabel('时间 [s]')
    ax2.set_ylabel('权重')
    ax2.set_ylim([-0.1, 1.1])
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_weight_evolution.png'), dpi=150)
    plt.close()
    print("绘制能量演化曲线 -> 1_weight_evolution.png")

def plot_single_window_comparison(data, x_f_list, out_dir):
    t_cep = data['t_cep']
    weights = data['weights']
    p_work = data['p_work']
    wlen = data['wlen']
    hop = data['hop']
    fs = data['fs']
    v = data['v']
    
    # 自动挑选三个典型时间窗
    idx_strong = np.argmax(weights)                   # 权重约 1.0 (高信噪比)
    idx_transient = np.argmin(np.abs(weights - 0.5))  # 权重约 0.5 (过渡区)
    idx_weak = np.argmin(weights)                     # 权重约 0.0 (失效区)
    
    indices = [idx_strong, idx_transient, idx_weak]
    labels = ["强信号区 (W≈1)", "过渡区 (W≈0.5)", "失效区 (W≈0)"]
    
    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    fig.suptitle('不同关注权重下的单窗波形、频谱与倒谱特征对比', fontsize=16)
    
    for col, (idx, label) in enumerate(zip(indices, labels)):
        start_sample = idx * hop
        end_sample = start_sample + wlen
        window_signal = p_work[start_sample:end_sample]
        
        # 原始未处理信号 (去均值之前，只做简单的时间轴对应)
        t_win = np.arange(wlen) / fs
        ax_raw = axes[0, col]
        ax_raw.plot(t_win, window_signal, 'k-', lw=1)
        ax_raw.set_title(f'[{label}] t={t_cep[idx]:.1f}s 原始未处理信号\nWeight={weights[idx]:.2f}')
        ax_raw.set_xlabel('窗内时间 [s]')
        ax_raw.set_ylabel('水头波动 [m]')
        ax_raw.grid(True, ls='--', alpha=0.5)
        if col > 0:
            ax_raw.set_ylim(axes[0, 0].get_ylim())
            
        # 加窗去均值
        window_signal = window_signal - np.mean(window_signal)
        win = scipy.signal.windows.hamming(wlen, sym=False)
        windowed = window_signal * win
        
        # 波形 (高通滤波突出高频反射)
        b, a = scipy.signal.butter(2, 2.0 / (fs/2), 'high') # 2Hz 高通
        window_hf = scipy.signal.filtfilt(b, a, windowed)
        
        ax_t = axes[1, col]
        ax_t.plot(t_win, window_hf, 'b-', lw=1)
        ax_t.set_title(f'高频波形 (2Hz高通)')
        ax_t.set_xlabel('窗内时间 [s]')
        ax_t.set_ylabel('高频波动 [m]')
        ax_t.grid(True, ls='--', alpha=0.5)
        # 固定 Y 轴方便对比幅度衰减
        if col > 0:
            ax_t.set_ylim(axes[1, 0].get_ylim())
            
        # 频谱
        spec = fft(windowed)
        mag = np.abs(spec)[:wlen//2] / wlen
        freqs = fftfreq(wlen, 1/fs)[:wlen//2]
        
        ax_f = axes[2, col]
        mask_f = freqs <= 15.0 # 显示 0-15Hz
        ax_f.plot(freqs[mask_f], mag[mask_f], 'g-', lw=1)
        f0 = v / (4.0 * data['L'])
        ax_f.axvline(f0, color='orange', ls='--', lw=1.2, label='基频')
        ax_f.set_title(f'FFT 频谱 (0-15Hz)')
        ax_f.set_xlabel('频率 [Hz]')
        ax_f.set_ylabel('幅值')
        ax_f.legend(loc='upper right')
        ax_f.grid(True, ls='--', alpha=0.5)
        if col > 0:
            ax_f.set_ylim(axes[2, 0].get_ylim())
            
        # 倒谱
        log_spec = np.log(np.abs(spec) + np.finfo(float).eps)
        ceps = np.real(ifft(log_spec))
        ceps = ceps[:wlen//2]
        q = np.arange(wlen//2) / fs
        depth = _quefrency_to_depth(q, v)
        
        ax_c = axes[3, col]
        mask_d = (depth >= 1000) & (depth <= data['L'] + 200)
        ax_c.plot(depth[mask_d], -ceps[mask_d], 'k-', lw=1.2)
        for xf in x_f_list:
            ax_c.axvline(xf, color='r', ls='--', alpha=0.7)
        ax_c.set_title(f'1D 实倒谱响应 (-C)')
        ax_c.set_xlabel('深度 [m]')
        ax_c.set_ylabel('能量')
        ax_c.grid(True, ls='--', alpha=0.5)
        if col > 0:
            ax_c.set_ylim(axes[3, 0].get_ylim())
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_single_window_comparison.png'), dpi=150)
    plt.close()
    print("绘制单窗特征对比图 -> 2_single_window_comparison.png")

def plot_weighted_cepstrogram(data, x_f_list, out_dir):
    C = data['C']
    depth = data['depth']
    t_cep = data['t_cep']
    weights = data['weights']
    
    # 限制显示深度在有效范围内以提高清晰度
    d_mask = (depth >= 3000) & (depth <= data['L'] + 200)
    d_show = depth[d_mask]
    C_show = C[d_mask, :]
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.05], height_ratios=[0.2, 1, 0.8], hspace=0.3, wspace=0.1)
    
    # --- 顶栏：权重条 ---
    ax_w = fig.add_subplot(gs[0, 0])
    ax_w.plot(t_cep, weights, 'g-', lw=2)
    ax_w.fill_between(t_cep, 0, weights, color='g', alpha=0.3)
    ax_w.set_xlim([t_cep[0], t_cep[-1]])
    ax_w.set_ylim([0, 1.1])
    ax_w.set_ylabel('关注权重 $W(t)$')
    ax_w.set_title('2D 倒谱图时间轴关注权重分布', fontweight='bold')
    ax_w.tick_params(labelbottom=False)
    ax_w.grid(True, ls='--', alpha=0.5)
    
    # --- 中栏：2D倒谱图 ---
    ax_2d = fig.add_subplot(gs[1, 0], sharex=ax_w)
    T_mesh, D_mesh = np.meshgrid(t_cep, d_show)
    C_data = -C_show
    vmin, vmax = np.percentile(C_data, [2, 98])
    im = ax_2d.pcolormesh(T_mesh, D_mesh, C_data, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    ax_2d.set_ylabel('深度 [m]')
    ax_2d.set_xlabel('时间 [s]')
    
    for xf in x_f_list:
        ax_2d.axhline(xf, color='yellow', ls='--', lw=1, alpha=0.8)
        
    # 添加被权重屏蔽的阴影指示
    # 为低于权重的区域叠加半透明黑色矩形
    for i in range(len(t_cep)-1):
        if weights[i] < 0.1:
            ax_2d.axvspan(t_cep[i], t_cep[i+1], color='black', alpha=0.5, lw=0)
            
    cax = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=cax, label='倒谱能量 (-C)')
    
    # --- 底栏：直接平均 vs 加权平均对比 ---
    ax_prof = fig.add_subplot(gs[2, 0])
    
    # 1. 传统平均 (算术平均)
    prof_avg = np.mean(-C[d_mask, :], axis=1)
    
    # 2. 权重加权平均
    sum_w = np.sum(weights)
    if sum_w > 0:
        prof_weighted = np.sum(-C[d_mask, :] * weights, axis=1) / sum_w
    else:
        prof_weighted = prof_avg
        
    ax_prof.plot(d_show, prof_avg, 'gray', lw=1.5, ls='--', label='传统算术平均')
    ax_prof.plot(d_show, prof_weighted, 'b', lw=2, label='关注权重加权平均')
    
    for idx, xf in enumerate(x_f_list):
        ax_prof.axvline(xf, color='r', ls='--', alpha=0.6, label=f'裂缝位置 ({xf}m)' if idx==0 else "")
        
    ax_prof.set_xlabel('深度 [m]')
    ax_prof.set_ylabel('时间平均能量 (-C)')
    ax_prof.set_title('1D 深度剖面对比：传统平均 vs 加权平均 (抑制后期噪声有效提升信噪比)')
    ax_prof.legend()
    ax_prof.grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_weighted_cepstrogram.png'), dpi=150)
    plt.close()
    print("绘制加权倒谱图 -> 3_weighted_cepstrogram.png")

def plot_drift_verification(data, x_f_list, out_dir):
    if len(x_f_list) == 0:
        return
        
    cepstrogram_2d = data['C']
    d_show = data['depth']
    t_windows = data['t_cep']
    weights = data['weights']
    
    # 找第一个裂缝作为验证目标
    target_xf = x_f_list[0]
    
    # 划定搜索区间 +- 150m
    search_radius = 150.0
    idx_min = np.searchsorted(d_show, target_xf - search_radius)
    idx_max = np.searchsorted(d_show, target_xf + search_radius)
    
    if idx_min >= idx_max:
        return
        
    peak_depths = []
    peak_fwhms = []
    
    for i in range(len(t_windows)):
        c_slice = -cepstrogram_2d[idx_min:idx_max, i]
        
        # 找峰
        peaks, props = scipy.signal.find_peaks(c_slice, width=0)
        if len(peaks) > 0:
            # 找最高峰
            max_idx = np.argmax(c_slice[peaks])
            peak_loc = peaks[max_idx]
            
            # 记录深度
            actual_depth = d_show[idx_min + peak_loc]
            peak_depths.append(actual_depth)
            
            # 半高宽
            w = props['widths'][max_idx]
            # w是index跨度，转成实际深度跨度
            dd = d_show[1] - d_show[0]
            peak_fwhms.append(w * dd)
        else:
            peak_depths.append(np.nan)
            peak_fwhms.append(np.nan)
            
    # 画图
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # 图 1: 峰值深度随时间的漂移
    ax1 = axes[0]
    ax1.plot(t_windows, peak_depths, 'r.-', label='单时间窗识别出的峰值深度')
    ax1.axhline(target_xf, color='k', ls='--', label=f'真实裂缝深度 ({target_xf}m)')
    ax1.set_ylabel('倒谱峰深度 [m]')
    ax1.set_title('随时间推移，裂缝响应在倒谱域的"漂移轨迹"及 W(t) 的抑制作用')
    ax1.grid(True, ls='--', alpha=0.5)
    
    # 双Y轴画权重
    ax1_w = ax1.twinx()
    ax1_w.fill_between(t_windows, 0, weights, color='blue', alpha=0.1, label='关注权重 W(t)')
    ax1_w.set_ylabel('关注权重 W(t)', color='blue')
    ax1_w.tick_params(axis='y', labelcolor='blue')
    ax1_w.set_ylim(-0.1, 1.1)
    
    # 规避合并图例的复杂性，简单处理
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_w.get_legend_handles_labels()
    ax1_w.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # 图 2: 峰宽胖化
    ax2 = axes[1]
    ax2.plot(t_windows, peak_fwhms, 'g.-', label='峰半高宽 (FWHM)')
    ax2.set_xlabel('时间窗中心 [s]')
    ax2.set_ylabel('峰宽 [m]')
    ax2.set_title('随时间推移，裂缝响应在倒谱域的分辨率丧失 ("胖化")')
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    # 图 3: 最终定位误差对比
    prof_unweighted = np.mean(-cepstrogram_2d, axis=1)
    
    sum_w = np.sum(weights)
    if sum_w > 0:
        prof_weighted = np.sum(-cepstrogram_2d * weights, axis=1) / sum_w
    else:
        prof_weighted = prof_unweighted
    
    ax3 = axes[2]
    # 取局部
    d_local = d_show[idx_min:idx_max]
    c_unw = prof_unweighted[idx_min:idx_max]
    c_w = prof_weighted[idx_min:idx_max]
    
    ax3.plot(d_local, c_unw, 'k', lw=1.5, alpha=0.6, label='传统等权重平均')
    ax3.plot(d_local, c_w, 'b', lw=2, label='关注权重 W(t) 加权平均')
    ax3.axvline(target_xf, color='r', ls='--', label=f'真实深度 ({target_xf}m)')
    
    # 找最终峰位
    p_unw, _ = scipy.signal.find_peaks(c_unw)
    if len(p_unw)>0:
        loc = d_local[p_unw[np.argmax(c_unw[p_unw])]]
        err = abs(loc - target_xf)
        ax3.plot(loc, c_unw[p_unw[np.argmax(c_unw[p_unw])]], 'ko')
        ax3.text(loc, c_unw[p_unw[np.argmax(c_unw[p_unw])]], f" 传统误差 {err:.1f}m", color='k')
        
    p_w, _ = scipy.signal.find_peaks(c_w)
    if len(p_w)>0:
        loc = d_local[p_w[np.argmax(c_w[p_w])]]
        err = abs(loc - target_xf)
        ax3.plot(loc, c_w[p_w[np.argmax(c_w[p_w])]], 'bo')
        ax3.text(loc, c_w[p_w[np.argmax(c_w[p_w])]], f" 加权误差 {err:.1f}m", color='b')
        
    ax3.set_xlabel('深度 [m]')
    ax3.set_ylabel('1D 时间平均能量')
    ax3.set_title('局部寻峰实测验证：剔除后期漂移极大地提升了最终定位精度')
    ax3.grid(True, ls='--', alpha=0.5)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_drift_verification.png'), dpi=150)
    plt.close()
    print("绘制漂移验证图 -> 4_drift_verification.png")

def process_single_case(friction, d_size, case_name):
    class Args: pass
    args = Args()
    args.friction = friction
    args.d_size = d_size
    args.case = case_name
    
    folder_name = f"{args.friction}_{args.d_size}"
    out_dir = os.path.join(_root, 'output', 'analysis', 'brunone_decay', folder_name, args.case)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        res, config, x_f_list = load_simulation(args)
    except FileNotFoundError as e:
        print(e)
        print(f"跳过 {args.case} 算例分析。")
        return
        
    data = compute_fracture_energy_and_weights(res, config)
    
    plot_weight_evolution(data, out_dir)
    plot_single_window_comparison(data, x_f_list, out_dir)
    plot_weighted_cepstrogram(data, x_f_list, out_dir)
    plot_drift_verification(data, x_f_list, out_dir)
    
    print(f"分析完成: {args.case}，结果已保存至: {out_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="倒谱能量衰减权重分析")
    parser.add_argument('--friction', type=str, default='brunone', help='摩阻模型，例如 brunone, steady')
    parser.add_argument('--d_size', type=str, default='D50', help='直径或特征尺寸标识，例如 D10, D50')
    parser.add_argument('--case', type=str, default='dual', help='裂缝数量算例，例如 dual, single, quad, 或 all 遍历所有')
    args = parser.parse_args()
    
    if args.case.lower() == 'all':
        folder_name = f"{args.friction}_{args.d_size}"
        base_dir = os.path.join(_root, 'output', 'leakoff', folder_name)
        if not os.path.exists(base_dir):
            print(f"错误: 找不到数据目录 {base_dir}")
            sys.exit(1)
            
        cases = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"检测到 {len(cases)} 个算例: {cases}\n")
        for c in cases:
            process_single_case(args.friction, args.d_size, c)
    else:
        process_single_case(args.friction, args.d_size, args.case)

if __name__ == '__main__':
    main()
