# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

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

from moc_simulate.paths import output_path, SERIES_DECAY_REGRESSION

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, 'cf_kleak_study', 'cf_kleak_table.csv')
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Create output directory for plots
    plot_dir = output_path(SERIES_DECAY_REGRESSION, 'cf_kleak_study', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. P_1d vs Cf (fixing Kleak=0.0 and Kleak=1e-4), for frac_idx=1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df_f1 = df[df['frac_idx'] == 1]
    
    ax = axes[0]
    for k in [0.0, 1e-4, 1e-3]:
        df_sub = df_f1[(df_f1['Kleak'] == k) & (df_f1['friction_model'] == 'steady')]
        if not df_sub.empty:
            ax.plot(df_sub['Cf'], df_sub['P_1d'], marker='o', label=f'steady (Kleak={k:.0e})')
            
        df_sub_b = df_f1[(df_f1['Kleak'] == k) & (df_f1['friction_model'] == 'brunone')]
        if not df_sub_b.empty:
            ax.plot(df_sub_b['Cf'], df_sub_b['P_1d'], marker='s', linestyle='--', label=f'brunone (Kleak={k:.0e})')
            
    ax.set_xscale('log')
    ax.set_xlabel('裂缝柔度 Cf [m²]')
    ax.set_ylabel('第一缝倒谱峰值 P_1d')
    ax.set_title('倒谱峰值随柔度 Cf 的变化')
    ax.grid(True, which='both', ls='--', alpha=0.6)
    ax.legend()
    
    # 2. P_1d vs Kleak (fixing Cf=1e-5), for frac_idx=1
    ax = axes[1]
    for c in [1e-6, 1e-5, 1e-4]:
        df_sub = df_f1[(df_f1['Cf'] == c) & (df_f1['friction_model'] == 'steady')]
        if not df_sub.empty:
            # Drop Kleak=0 for log scale plotting, or replace with small value
            df_plot = df_sub.copy()
            df_plot.loc[df_plot['Kleak'] == 0, 'Kleak'] = 1e-6
            df_plot = df_plot.sort_values('Kleak')
            ax.plot(df_plot['Kleak'], df_plot['P_1d'], marker='o', label=f'steady (Cf={c:.0e})')
            
        df_sub_b = df_f1[(df_f1['Cf'] == c) & (df_f1['friction_model'] == 'brunone')]
        if not df_sub_b.empty:
            df_plot_b = df_sub_b.copy()
            df_plot_b.loc[df_plot_b['Kleak'] == 0, 'Kleak'] = 1e-6
            df_plot_b = df_plot_b.sort_values('Kleak')
            ax.plot(df_plot_b['Kleak'], df_plot_b['P_1d'], marker='s', linestyle='--', label=f'brunone (Cf={c:.0e})')
            
    ax.set_xscale('log')
    ax.set_xlabel('滤失系数 Kleak (1e-6 代表 0)')
    ax.set_ylabel('第一缝倒谱峰值 P_1d')
    ax.set_title('倒谱峰值随滤失系数 Kleak 的变化')
    ax.grid(True, which='both', ls='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    out_path1 = os.path.join(plot_dir, 'peak_vs_cf_kleak.png')
    plt.savefig(out_path1, dpi=150)
    print(f"Saved plot to {out_path1}")
    plt.close()
    
    # 3. alpha_2d vs frac_idx (Decay across fractures) for different Cf and fixed Kleak
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Panel: Varying Cf, fixing Kleak=0.0
    ax = axes[0]
    df_k0 = df[df['Kleak'] == 0.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, 4))
    for i, c in enumerate([1e-7, 1e-6, 1e-5, 1e-4]):
        # steady
        df_sub = df_k0[(df_k0['Cf'] == c) & (df_k0['friction_model'] == 'steady')].sort_values('frac_idx')
        if not df_sub.empty:
            ax.plot(df_sub['frac_idx'], df_sub['alpha_2d'], marker='o', color=colors[i], ls='-', label=f'steady (Cf={c:.0e})')
        # brunone
        df_sub_b = df_k0[(df_k0['Cf'] == c) & (df_k0['friction_model'] == 'brunone')].sort_values('frac_idx')
        if not df_sub_b.empty:
            ax.plot(df_sub_b['frac_idx'], df_sub_b['alpha_2d'], marker='s', color=colors[i], ls='--', alpha=0.7, label=f'brunone (Cf={c:.0e})')
            
    ax.set_xticks(range(1, 6))
    ax.set_xlabel('裂缝序号 (frac_idx)')
    ax.set_ylabel('相对峰值 alpha_2d (P_i / P_1)')
    ax.set_title('不同 Cf 下缝间能量衰减 (Kleak=0)')
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(fontsize=9, ncol=2)
    
    # Right Panel: Varying Kleak, fixing Cf=1e-5
    ax = axes[1]
    df_cf1e5 = df[df['Cf'] == 1e-5]
    colors = plt.cm.plasma(np.linspace(0, 0.9, 4))
    for i, k in enumerate([0.0, 1e-5, 1e-4, 1e-3]):
        # steady
        df_sub = df_cf1e5[(df_cf1e5['Kleak'] == k) & (df_cf1e5['friction_model'] == 'steady')].sort_values('frac_idx')
        if not df_sub.empty:
            ax.plot(df_sub['frac_idx'], df_sub['alpha_2d'], marker='o', color=colors[i], ls='-', label=f'steady (Kleak={k:.0e})')
        # brunone
        df_sub_b = df_cf1e5[(df_cf1e5['Kleak'] == k) & (df_cf1e5['friction_model'] == 'brunone')].sort_values('frac_idx')
        if not df_sub_b.empty:
            ax.plot(df_sub_b['frac_idx'], df_sub_b['alpha_2d'], marker='s', color=colors[i], ls='--', alpha=0.7, label=f'brunone (Kleak={k:.0e})')
            
    ax.set_xticks(range(1, 6))
    ax.set_xlabel('裂缝序号 (frac_idx)')
    ax.set_ylabel('相对峰值 alpha_2d (P_i / P_1)')
    ax.set_title('不同 Kleak 下缝间能量衰减 (Cf=1e-5)')
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(fontsize=9, ncol=2)
    
    plt.tight_layout()
    out_path2 = os.path.join(plot_dir, 'alpha_decay.png')
    plt.savefig(out_path2, dpi=150)
    print(f"Saved plot to {out_path2}")
    plt.close()

if __name__ == '__main__':
    main()
