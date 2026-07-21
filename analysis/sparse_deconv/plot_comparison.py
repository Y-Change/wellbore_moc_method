# -*- coding: utf-8 -*-
"""
可视化对比 1D 倒谱、L1 反卷积与 BSD 盲反卷积的结果。
"""
import os
import sys
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

from moc_simulate.paths import output_path, SERIES_LEAKOFF
from analysis.sparse_deconv.reference_signal import generate_reference_signal, extract_wavelet, compute_difference_signal
from analysis.sparse_deconv.l1_deconv import L1Deconvolver
from moc_simulate.leakoff_multi import resolve_cases, load_timeseries_csv
from moc_simulate.config import SIM_CONFIG
from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum_1d


def plot_comparison(friction: str, spacing: int, case_key: str):
    """
    针对单 case 绘制深度域响应对比图：倒谱 vs L1反卷积
    """
    print(f"Generating plot for {friction} D{spacing} {case_key}...")
    
    # 1. 准备数据
    friction_key = f"{friction}_D{spacing}"
    cases = resolve_cases(friction_key)
    if case_key not in cases:
        print(f"Case {case_key} not found.")
        return
        
    x_f_true = cases[case_key]['x_f_list']
    n_fracs = len(x_f_true)
    
    ref_data = generate_reference_signal(friction=friction)
    t_ref = ref_data['t']
    H_ref = ref_data['H_wh']
    fs = ref_data['fs']
    a_adj = ref_data['a_adj']
    L = ref_data['cfg'].wellbore_length
    ts = SIM_CONFIG['ts']
    
    csv_path = output_path(f"{SERIES_LEAKOFF}/{friction_key}", case_key, 'moc_timeseries.csv')
    frac_data = load_timeseries_csv(csv_path)
    t_frac = frac_data['t']
    H_frac = frac_data['H_wh']
    
    # 2. 计算差信号和子波
    diff_data = compute_difference_signal(t_frac, H_frac, t_ref, H_ref, fs, ts)
    delta_y = diff_data['delta_y']
    h = extract_wavelet(H_ref, fs, a_adj, L, ts, t_ref)
    
    # 3. L1 反卷积
    l1 = L1Deconvolver(h, fs, a_adj)
    l1_res = l1.solve(delta_y, n_fracs=n_fracs)
    
    # 4. 1D 倒谱 (作为 Baseline)
    cep_res = compute_moc_cepstrum_1d(t_frac, H_frac, v=a_adj, fs=fs, ts=ts, wellbore_length=L)
    
    # 5. 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 子图1：原始差信号 (时间域映射到空间)
    ax = axes[0]
    t_axis = diff_data['t_after'] - ts
    d_axis = t_axis * a_adj / 2.0
    
    mask = (d_axis >= 3500) & (d_axis <= 4500)
    ax.plot(d_axis[mask], delta_y[mask], color='gray', lw=1)
    ax.set_title(f"差信号 $\Delta y$ (映射到空间深度) - {friction} D{spacing} {case_key}")
    ax.set_ylabel("水头扰动 [m]")
    for x_f in x_f_true:
        ax.axvline(x_f, color='red', linestyle='--', alpha=0.5)
        
    # 子图2：1D 倒谱
    ax = axes[1]
    cep_d = cep_res['depth']
    cep_r = cep_res['response']
    cep_mask = (cep_d >= 3500) & (cep_d <= 4500)
    ax.plot(cep_d[cep_mask], cep_r[cep_mask], color='blue', lw=1.5, label='1D Cepstrum')
    ax.set_title(f"1D 倒谱响应 (受限于 38m 瑞利极限)")
    ax.set_ylabel("倒谱响应幅值")
    for x_f in x_f_true:
        ax.axvline(x_f, color='red', linestyle='--', alpha=0.5)
        
    # 子图3：L1 稀疏反卷积
    ax = axes[2]
    l1_d = l1_res['depth_axis']
    l1_r = l1_res['r']
    l1_mask = (l1_d >= 3500) & (l1_d <= 4500)
    ax.plot(l1_d[l1_mask], np.abs(l1_r[l1_mask]), color='green', lw=2, label='L1 Sparse Deconv')
    ax.set_title(f"已知先验缝数的 ℓ₁ 反卷积 (突破瑞利极限)")
    ax.set_xlabel("深度 [m]")
    ax.set_ylabel("反射系数幅值 |r|")
    for i, x_f in enumerate(x_f_true):
        ax.axvline(x_f, color='red', linestyle='--', alpha=0.5, label=f"True F{i+1}" if i==0 else "")
        
    for ax in axes:
        ax.set_xlim(3800, 4400)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    
    out_dir = output_path('analysis/sparse_deconv', friction, '')
    save_path = os.path.join(out_dir, f"plot_comparison_D{spacing}_{case_key}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == '__main__':
    plot_comparison('steady', 20, 'dual')
    plot_comparison('steady', 20, 'triple')
    plot_comparison('steady', 20, 'quad')
    plot_comparison('steady', 20, 'quint')
    plot_comparison('steady', 20, 'hex')
    plot_comparison('steady', 20, 'hept')
    plot_comparison('steady', 20, 'oct')
