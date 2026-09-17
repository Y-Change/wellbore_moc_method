import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

_d = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_d))
if _root not in sys.path:
    sys.path.insert(0, _root)

output_dir = os.path.join(_root, 'output', 'analysis', 'brunone_spacing_effect')

# 选取所有 K 值对比频散效应
k_vals = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
D = 20

fig, axes = plt.subplots(len(k_vals), 1, figsize=(10, 16), sharex=True, sharey=True)

for i, k in enumerate(k_vals):
    csv_path = os.path.join(output_dir, f'D{D}_k{k}', 'moc_timeseries.csv')
    df = pd.read_csv(csv_path)
    t = df['t'].values
    H = df['H_wh'].values
    dt = t[1] - t[0]
    
    # 截取裂缝反射区段 (扩大一点窗口看全貌)
    mask = (t >= 6.4) & (t <= 7.2)
    t_win = t[mask]
    H_win = H[mask]
    
    # 提取导数放大高频突变信号
    dH = np.gradient(H_win, dt)
    
    # 执行短时傅里叶变换 (STFT) 时频分析
    fs = 1.0 / dt
    f, t_spec, Sxx = spectrogram(dH, fs=fs, nperseg=128, noverlap=120)
    
    ax = axes[i]
    # 使用 pcolormesh 绘制高分辨率时频热力图 (转换分贝对数刻度)
    im = ax.pcolormesh(t_win[0] + t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
    
    ax.set_ylabel('频率 [Hz]')
    ax.set_ylim([0, 150]) # 限制观察高频区 0-150Hz
    
    title = f'时频能量谱 (STFT) - 稳态摩阻 (k={k})' if k == 0 else f'时频能量谱 (STFT) - 强非定常摩阻 (k={k})'
    ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.3)
    fig.colorbar(im, ax=ax, label='能量 [dB]')

axes[-1].set_xlabel('时间 (s)')
fig.suptitle(f'水锤传播的时频分析：非定常摩阻引发的“高频衰减与滞后”频散证明 (D={D}m)', fontsize=14, fontweight='bold', y=0.97)
plt.tight_layout()

save_path = os.path.join(output_dir, 'fig5_tf_dispersion.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"时频分析图已成功保存至: {save_path}")
