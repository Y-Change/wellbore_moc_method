import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

_d = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_d))
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.config import WELL_CONFIG, SIM_CONFIG, FRACTURE_CONFIG
from moc_simulate.leakoff_multi import run_cepstrum_analysis_and_match

output_dir = os.path.join(_root, 'output', 'analysis', 'brunone_spacing_effect')
spacings = [5, 10, 20, 50, 100]
k_values = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
a = WELL_CONFIG['wavespeed']
ts = SIM_CONFIG['ts']
L = WELL_CONFIG['L']
dt = SIM_CONFIG['dt']
FRAC_FIRST_M = 4100.0

# 提取特征
metrics = []

for D in spacings:
    for k in k_values:
        case_dir = os.path.join(output_dir, f'D{D}_k{k}')
        csv_path = os.path.join(case_dir, 'moc_timeseries.csv')
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path)
        t = df['t'].values
        H = df['H_wh'].values
        
        # 产生倒谱图
        x_f_list = [FRAC_FIRST_M + i * D for i in range(4)]
        cep_path = os.path.join(case_dir, 'cepstrum_standard.png')
        zoom_path = os.path.join(case_dir, 'cepstrum_fracture_zoom.png')
        
        # 为了不消耗太多时间，仅绘制 D=20 或 k=0, 0.1 典型工况的倒谱，或者全画
        # 这里选择全部生成 (如果已经存在就跳过，极大节省二次运行时间)
        if not os.path.exists(cep_path):
            run_cepstrum_analysis_and_match(
                t_sim=t, H_wh=H, a_adj=a, L=L, ts=ts, dt=dt,
                x_f_list=x_f_list, x_f_plot=x_f_list,
                friction='steady' if k==0 else 'brunone',
                label=f'D={D}m_k={k}',
                kleak=FRACTURE_CONFIG['kleak'],
                cep_path=cep_path,
                cep_zoom_path=zoom_path
            )
        
        # 计算时域导数用于提取漂移、展宽
        dH = np.gradient(H, dt)
        
        # 补充绘制 moc_leakoff.png (2x2 subplots)
        leakoff_fig_path = os.path.join(case_dir, 'moc_leakoff.png')
        if not os.path.exists(leakoff_fig_path):
            fig_l, axes_l = plt.subplots(2, 2, figsize=(15, 10))
            
            # 全程 H
            axes_l[0, 0].plot(t, H, 'k-')
            axes_l[0, 0].set_title('Full Time History (H)')
            axes_l[0, 0].set_xlabel('Time (s)')
            axes_l[0, 0].set_ylabel('Head (m)')
            axes_l[0, 0].grid(True)
            
            # 前12s H
            mask_12s = t <= 12.0
            axes_l[0, 1].plot(t[mask_12s], H[mask_12s], 'b-')
            axes_l[0, 1].set_title('First 12s (H)')
            axes_l[0, 1].set_xlabel('Time (s)')
            axes_l[0, 1].set_ylabel('Head (m)')
            axes_l[0, 1].grid(True)
            
            # 全程 dH/dt
            axes_l[1, 0].plot(t, dH, 'k-')
            axes_l[1, 0].set_title('Full Time History (dH/dt)')
            axes_l[1, 0].set_xlabel('Time (s)')
            axes_l[1, 0].set_ylabel('dH/dt')
            axes_l[1, 0].set_ylim([-500, 500])
            axes_l[1, 0].grid(True)
            
            # 前12s dH/dt
            axes_l[1, 1].plot(t[mask_12s], dH[mask_12s], 'r-')
            axes_l[1, 1].set_title('First 12s (dH/dt)')
            axes_l[1, 1].set_xlabel('Time (s)')
            axes_l[1, 1].set_ylabel('dH/dt')
            axes_l[1, 1].set_ylim([-500, 500])
            axes_l[1, 1].grid(True)
            
            plt.tight_layout()
            fig_l.savefig(leakoff_fig_path, dpi=200, bbox_inches='tight')
            plt.close(fig_l)
        
        # 我们只看 4100m 裂缝的反射，时间窗口大概 [6.5, 7.5]
        mask = (t >= 6.5) & (t <= 7.5)
        t_win = t[mask]
        dH_win = dH[mask]
        
        # 因为 leakoff 产生的反射是正峰，我们可以用 find_peaks
        # 寻找首个峰 (对应 4100m)
        peaks, props = find_peaks(dH_win, height=np.max(dH_win)*0.1, distance=int(0.002/dt))
        
        if len(peaks) > 0:
            # 取最前面的峰作为首缝峰
            p_idx = peaks[0]
            t_peak = t_win[p_idx]

            # ── Rayleigh 可分辨性判据 ──
            # 两峰可分辨 <=> 谷底低于较低峰的 81% (瑞利限 8/π²)
            # rayleigh = (P_min - V) / P_min / 0.81, 截断到 [0, 1]
            # = 1: 谷底达到瑞利限以下, 完全可分辨
            # = 0: 谷底与较低峰齐平, 完全不可分辨
            rayleigh = 0.0
            if len(peaks) >= 2:
                p1, p2 = peaks[0], peaks[1]
                v_idx = p1 + np.argmin(dH_win[p1:p2])
                P_min = min(dH_win[p1], dH_win[p2])
                V = dH_win[v_idx]
                if P_min > 0:
                    raw = (P_min - V) / P_min
                    rayleigh = min(1.0, max(0.0, raw / 0.81))
        else:
            t_peak = np.nan
            rayleigh = 0.0
        
        # ── 能量弥散时间宽度 (Energy Spread Time, EST) ──
        # 用正部能量的累积分布，取 E=10% 到 E=90% 的时间跨度
        # 不依赖峰形假设，单调递增，对衰减变形鲁棒
        pos = np.maximum(dH_win, 0)
        E_cum = np.cumsum(pos**2) * dt
        E_total = E_cum[-1]
        if E_total > 0:
            t10 = t_win[np.searchsorted(E_cum, 0.1 * E_total)]
            t90 = t_win[np.searchsorted(E_cum, 0.9 * E_total)]
            est = t90 - t10
        else:
            est = np.nan

        # ── 三种等效波速测量 ──
        # 1. a_f0: 从 FFT 基频推算 (水锤全井筒振荡周期 T0=4L/a)
        mask_post = t > 1.0
        H_post = H[mask_post] - np.mean(H[mask_post])
        N_post = len(H_post)
        Y = np.fft.rfft(H_post)
        freqs_fft = np.fft.rfftfreq(N_post, dt)
        amp_fft = np.abs(Y) * 2.0 / N_post
        f0_theory = a / (4.0 * L)
        band_f0 = (freqs_fft >= f0_theory * 0.5) & (freqs_fft <= f0_theory * 1.5)
        bf = freqs_fft[band_f0]
        ba = amp_fft[band_f0]
        if len(ba) > 0:
            pk = np.argmax(ba)
            if 0 < pk < len(ba) - 1:
                y0_, y1_, y2_ = ba[pk-1], ba[pk], ba[pk+1]
                denom = y0_ - 2.0*y1_ + y2_
                delta = 0.5*(y0_-y2_)/denom if abs(denom) > 1e-30 else 0.0
                f0_meas = bf[pk] + delta*(bf[1]-bf[0])
            else:
                f0_meas = bf[pk]
            a_f0 = f0_meas * 4.0 * L
        else:
            a_f0 = np.nan

        # 2. a_onset: 从前沿到达时间推算 (首缝反射波 2*x_f/a)
        if len(peaks) > 0:
            peak_val = np.max(dH_win)
            ab = np.where(dH_win > peak_val * 0.01)[0]
            t_onset = t_win[ab[0]] if len(ab) > 0 else np.nan
            a_onset = 2.0 * FRAC_FIRST_M / (t_onset - ts) if not np.isnan(t_onset) else np.nan
        else:
            t_onset = np.nan
            a_onset = np.nan

        # 3. a_peak: 从峰值到达时间推算
        a_peak = 2.0 * FRAC_FIRST_M / (t_peak - ts) if not np.isnan(t_peak) else np.nan

        metrics.append({
            'D': D,
            'k': k,
            't_peak': t_peak,
            'est': est,
            'rayleigh': rayleigh,
            'a_f0': a_f0,
            'a_onset': a_onset,
            'a_peak': a_peak,
            'dH_win': dH_win,
            't_win': t_win
        })

df_metrics = pd.DataFrame(metrics)
# 计算 relative shift (vs k=0) and relative energy spread
for D in spacings:
    base_mask = (df_metrics['D'] == D) & (df_metrics['k'] == 0)
    if base_mask.sum() > 0:
        base_t = df_metrics.loc[base_mask, 't_peak'].values[0]
        base_est = df_metrics.loc[base_mask, 'est'].values[0]
        
        D_mask = df_metrics['D'] == D
        df_metrics.loc[D_mask, 'dt_shift'] = df_metrics.loc[D_mask, 't_peak'] - base_t
        df_metrics.loc[D_mask, 'd_est'] = df_metrics.loc[D_mask, 'est'] - base_est

# Save metrics
df_metrics.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)

# ================= 绘图 =================

# 图1：波形演化对比图 (瀑布图 改为分行子图), e.g., for D=20
fig1, axes1 = plt.subplots(len(k_values), 1, figsize=(10, 12), sharex=True)
D_plot = 20
colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
for i, k in enumerate(k_values):
    row = df_metrics[(df_metrics['D'] == D_plot) & (df_metrics['k'] == k)].iloc[0]
    ax = axes1[i]
    ax.plot(row['t_win'], row['dH_win'], label=f'k={k}', color=colors[i])
    ax.set_ylabel('dH/dt')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    if i == 0:
        ax.set_title(f'Waveform Evolution with Brunone Damping (D={D_plot}m)')
    
axes1[-1].set_xlim(6.6, 7.0)
axes1[-1].set_xlabel('Time (s)')
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'fig1_waveform_evolution.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)

# 图2：趋势曲线 (以 D=20 单缝看)
fig2, ax2 = plt.subplots(figsize=(8, 5))
sub_df = df_metrics[df_metrics['D'] == D_plot].sort_values('k')
ax2.plot(sub_df['k'], sub_df['dt_shift']*1000, 'o-', color='tab:red', label='Peak Shift (ms)')
ax2.set_xlabel('Brunone Parameter k')
ax2.set_ylabel('Peak Shift (ms)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax2_twin = ax2.twinx()
ax2_twin.plot(sub_df['k'], sub_df['d_est']*1000, 's--', color='tab:blue', label='Energy Spread Time (ms)')
ax2_twin.set_ylabel('EST (ms)', color='tab:blue')
ax2_twin.tick_params(axis='y', labelcolor='tab:blue')
plt.title(f'Peak Shift and Energy Spread vs k (D={D_plot}m, 1st Fracture)')
fig2.savefig(os.path.join(output_dir, 'fig2_trend.png'), dpi=300, bbox_inches='tight')

# 图3：双裂缝干涉演变 (这里是4裂缝)
fig3, axes = plt.subplots(len(spacings), 3, figsize=(15, 12), sharex=False, sharey=False)
selected_k = [0, 0.05, 0.2]
for i, D in enumerate(spacings):
    for j, k in enumerate(selected_k):
        ax = axes[i, j]
        row = df_metrics[(df_metrics['D'] == D) & (df_metrics['k'] == k)]
        if len(row) > 0:
            t = row.iloc[0]['t_win']
            dH = row.iloc[0]['dH_win']
            ax.plot(t, dH, 'k-')
            ax.set_title(f'D={D}m, k={k}')
            # highlight peaks
            peaks, _ = find_peaks(dH, height=np.max(dH)*0.1, distance=int(0.002/dt))
            ax.plot(t[peaks], dH[peaks], 'rx')
            ax.set_xlim(6.6, 6.6 + 4*D*2/1450 + 0.1) # zoom in properly
plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'fig3_interference.png'), dpi=300, bbox_inches='tight')

# 图4：Rayleigh 可分辨性热力图
ray_matrix = np.zeros((len(k_values), len(spacings)))
for i, k in enumerate(k_values):
    for j, D in enumerate(spacings):
        row = df_metrics[(df_metrics['D'] == D) & (df_metrics['k'] == k)]
        if len(row) > 0:
            ray_matrix[i, j] = row.iloc[0]['rayleigh']

fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.heatmap(ray_matrix, xticklabels=spacings, yticklabels=k_values,
            annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4, vmin=0, vmax=1)
ax4.set_xlabel('Fracture Spacing D (m)')
ax4.set_ylabel('Brunone Parameter k')
ax4.set_title('Rayleigh Criterion Resolvability Heatmap')
fig4.savefig(os.path.join(output_dir, 'fig4_rayleigh_heatmap.png'), dpi=300, bbox_inches='tight')

# 图5：三种等效波速测量对比 (D=20)
fig5, ax5 = plt.subplots(figsize=(8, 5))
sub5 = df_metrics[df_metrics['D'] == D_plot].sort_values('k')
ax5.plot(sub5['k'], sub5['a_f0'], 'o-', color='tab:red', lw=2, ms=7,
         label='$a_{f0}$ (FFT base frequency)')
ax5.plot(sub5['k'], sub5['a_onset'], 's--', color='tab:blue', lw=2, ms=7,
         label='$a_{onset}$ (wavefront arrival)')
ax5.plot(sub5['k'], sub5['a_peak'], '^-.', color='tab:green', lw=2, ms=7,
         label='$a_{peak}$ (peak arrival)')
ax5.axhline(a, color='gray', ls=':', lw=1.5, label=f'Nominal $a$={a:.0f} m/s')
ax5.set_xlabel('Brunone Parameter k')
ax5.set_ylabel('Effective Wave Speed (m/s)')
ax5.set_title(f'Three Wave Speed Measurements vs k (D={D_plot}m, 1st Fracture)')
ax5.legend(fontsize=9)
ax5.grid(True, ls='--', alpha=0.6)
fig5.savefig(os.path.join(output_dir, 'fig5_wave_speeds.png'), dpi=300, bbox_inches='tight')
plt.close(fig5)

# 图6：三种波速测量随 k 的变化率（相对 k=0 的百分比）
fig6, ax6 = plt.subplots(figsize=(8, 5))
base_a_f0 = sub5['a_f0'].iloc[0]
base_a_onset = sub5['a_onset'].iloc[0]
base_a_peak = sub5['a_peak'].iloc[0]
da_f0 = (base_a_f0 - sub5['a_f0']) / base_a_f0 * 100
da_onset = (base_a_onset - sub5['a_onset']) / base_a_onset * 100
da_peak = (base_a_peak - sub5['a_peak']) / base_a_peak * 100

ax6.plot(sub5['k'], da_f0, 'o-', color='tab:red', lw=2, ms=7,
         label='$\\Delta a_{f0}$ / $a_{f0}$ (%)')
ax6.plot(sub5['k'], da_onset, 's--', color='tab:blue', lw=2, ms=7,
         label='$\\Delta a_{onset}$ / $a_{onset}$ (%)')
ax6.plot(sub5['k'], da_peak, '^-.', color='tab:green', lw=2, ms=7,
         label='$\\Delta a_{peak}$ / $a_{peak}$ (%)')
ax6.set_xlabel('Brunone Parameter k')
ax6.set_ylabel('Wave Speed Reduction (%)')
ax6.set_title(f'Wave Speed Reduction vs k (D={D_plot}m, 1st Fracture)')
ax6.legend(fontsize=9)
ax6.grid(True, ls='--', alpha=0.6)
fig6.savefig(os.path.join(output_dir, 'fig6_wave_speed_reduction.png'), dpi=300, bbox_inches='tight')
plt.close(fig6)

print("All plots generated successfully.")
