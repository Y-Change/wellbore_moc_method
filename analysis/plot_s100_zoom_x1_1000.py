import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

from paths import output_path, SERIES_DECAY_REGRESSION
from cepstrum_mocdata import compute_moc_cepstrum
from analysis.paper_plots import apply_paper_rc, save_figure

apply_paper_rc()

def main():
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
    
    out_2d = compute_moc_cepstrum(t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
                                  wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
    depth = out_2d['depth']
    profile = -np.mean(out_2d['C'], axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Range 950 - 1750 to cover all 8 fractures (1000m to 1700m)
    mask = (depth >= 950) & (depth <= 1750)
    d_plot = depth[mask]
    p_plot = profile[mask]
    
    ax.plot(d_plot, p_plot, 'k-', lw=1.5, label='2D Cepstrum Profile')
    
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(x_f_aligned)))
    
    search_radius = 15.0
    if len(x_f_aligned) > 1:
        min_spacing = min(abs(x_f_aligned[i] - x_f_aligned[i-1]) for i in range(1, len(x_f_aligned)))
        search_radius = min(search_radius, min_spacing * 0.49)
        
    for i, xf in enumerate(x_f_aligned):
        mask_search = (depth >= xf - search_radius) & (depth <= xf + search_radius)
        if np.any(mask_search):
            local_d = depth[mask_search]
            local_p = profile[mask_search]
            max_idx = np.argmax(local_p)
            peak_d = local_d[max_idx]
            peak_val = local_p[max_idx]
        else:
            idx = np.argmin(np.abs(depth - xf))
            peak_d = depth[idx]
            peak_val = profile[idx]
            
        ax.plot(peak_d, peak_val, 'o', color=colors[i], markersize=6, 
                label=f'Frac {i+1} ({xf:.1f}m -> {peak_d:.1f}m)')
        ax.axvline(xf, color=colors[i], linestyle='--', alpha=0.4)
        
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Cepstrum Amplitude')
    ax.set_title('2D Cepstrum Global Zoom (steady, x1=1000, S=100, n=8)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, 'case_zoom', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'steady_cepstrum_x1_1000_s100_zoom.png')
    save_figure(fig, save_path)
    
    art_dir = r"C:\Users\Change\.gemini\antigravity\brain\4850e2b5-0e72-4d7e-8d5e-d9a05614c45c"
    art_path = os.path.join(art_dir, 'steady_cepstrum_x1_1000_s100_zoom.png')
    shutil.copy2(save_path, art_path)
    
    print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
