import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

from paths import output_path, SERIES_DECAY_REGRESSION
from cepstrum_mocdata import compute_moc_cepstrum
from analysis.paper_plots import apply_paper_rc, save_figure

apply_paper_rc()

def main():
    # Load raw npz
    raw_dir = output_path(SERIES_DECAY_REGRESSION, 'raw_data', '')
    npz_path = os.path.join(raw_dir, 'steady_x1_1000_sp_100_n_8.npz')
    
    if not os.path.isfile(npz_path):
        print(f"File not found: {npz_path}")
        return
        
    data = np.load(npz_path)
    t_sim = data['t_sim']
    H_wh = data['H_wh']
    v = data['v'][0]
    fs = data['fs'][0]
    ts = data['ts'][0]
    L = data['L'][0]
    x_f_aligned = data['x_f_aligned']
    
    # Compute 2D Cepstrum
    out_2d = compute_moc_cepstrum(t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
                                  wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
    depth = out_2d['depth']
    profile = -np.mean(out_2d['C'], axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Limit to 950 - 1100
    mask = (depth >= 950) & (depth <= 1100)
    d_plot = depth[mask]
    p_plot = profile[mask]
    
    ax.plot(d_plot, p_plot, 'k-', lw=1.5, label='2D Cepstrum Profile')
    
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(x_f_aligned)))
    
    for i, xf in enumerate(x_f_aligned):
        idx = np.argmin(np.abs(depth - xf))
        ax.plot(depth[idx], profile[idx], 'o', color=colors[i], markersize=6, 
                label=f'Frac {i+1} ({xf:.1f}m)')
        ax.axvline(xf, color=colors[i], linestyle='--', alpha=0.4)
        
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Cepstrum Amplitude')
    ax.set_title('2D Cepstrum Local Zoom (steady, x1=1000, S=10, n=8)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save to output/analysis/decay_regression/case_zoom
    out_dir = output_path(SERIES_DECAY_REGRESSION, 'case_zoom', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'steady_cepstrum_x1_1000_s10_zoom.png')
    save_figure(fig, save_path)
    print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
