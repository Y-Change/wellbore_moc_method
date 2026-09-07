import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

_d = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_d))
if _root not in sys.path:
    sys.path.insert(0, _root)

output_dir = os.path.join(_root, 'output', 'analysis', 'brunone_spacing_effect')

# 选取三个有代表性的 K 值对比频散效应
k_vals = [0, 0.1, 0.2]
D = 20

fig, axes = plt.subplots(len(k_vals), 1, figsize=(10, 10), sharex=True)

fs = 1000.0  # 采样率 1 / dt
# 重点关注 5Hz ~ 150Hz 的频带（涵盖主低频与裂缝反射高频）
frequencies = np.linspace(5, 150, 150) 
w = 6.0 # Morlet2 小波的 \omega_0 宽度参数
widths = w * fs / (2 * np.pi * frequencies)

for i, k in enumerate(k_vals):
    csv_path = os.path.join(output_dir, f'D{D}_k{k}', 'moc_timeseries.csv')
    df = pd.read_csv(csv_path)
    t = df['t'].values
    H = df['H_wh'].values
    dt = t[1] - t[0]
    
    # 截取裂缝反射区段附近 (6.5s ~ 7.2s)
    mask = (t >= 6.5) & (t <= 7.2)
    t_win = t[mask]
    H_win = H[mask]
    
    # 提取导数以放大高频突变信号（裂缝反射主要体现在极短时的导数尖峰中）
    dH = np.gradient(H_win, dt)
    
    # 执行连续小波变换 (CWT)
    cwtmatr = cwt(dH, morlet2, widths, w=w)
    cwt_amp = np.abs(cwtmatr)
    
    ax = axes[i]
    # 使用 pcolormesh 绘制高分辨率时频热力图
    im = ax.pcolormesh(t_win, frequencies, cwt_amp, shading='gouraud', cmap='jet')
    
    ax.set_ylabel('频率 (Hz)')
    title = f'CWT 时频能量图 - 稳态摩阻 (k={k})' if k == 0 else f'CWT 时频能量图 - 强非定常摩阻 (k={k})'
    ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.3)
    fig.colorbar(im, ax=ax, label='CWT 幅值')

axes[-1].set_xlabel('时间 (s)')
fig.suptitle(f'水锤传播的小波时频分析：非定常摩阻引发的“高频滞后”证明 (间距 D={D}m)', fontsize=14, fontweight='bold', y=0.97)
plt.tight_layout()

save_path = os.path.join(output_dir, 'fig5_cwt_dispersion.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"小波变换图已成功保存至: {save_path}")
