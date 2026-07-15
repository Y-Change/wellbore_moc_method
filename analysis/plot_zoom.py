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

def main():
    raw_dir = output_path(SERIES_DECAY_REGRESSION, 'raw_data', '')
    npz_path = os.path.join(raw_dir, 'brunone_x1_4000_sp_10_n_8.npz')
    
    if not os.path.isfile(npz_path):
        print(f"File not found: {npz_path}")
        return
        
    data = np.load(npz_path)
    t_sim = data['t_sim']
    H_wh = data['H_wh']
    v = float(data['v'][0])
    L = float(data['L'][0])
    fs = float(data['fs'][0])
    ts = float(data['ts'][0])
    x_f_aligned = data['x_f_aligned']
    
    out_2d = compute_moc_cepstrum(t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
                                  wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
    depth = out_2d['depth']
    profile_2d = -np.mean(out_2d['C'], axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot cepstrum curve
    mask = (depth >= 3900) & (depth <= 4200)
    ax.plot(depth[mask], profile_2d[mask], 'k-', lw=1.5, label='2D Cepstrum Profile')
    
    # Mark theoretical fracture positions
    colors = plt.cm.jet(np.linspace(0, 1, len(x_f_aligned)))
    for i, (xf, c) in enumerate(zip(x_f_aligned, colors)):
        ax.axvline(xf, color=c, ls='--', lw=1, alpha=0.8, label=f'Frac {i+1} ({xf:.1f}m)')
        
    # Mark the picked peaks
    # Same logic as decay_regression.py
    search_radius = 15.0
    min_spacing = min(abs(x_f_aligned[i] - x_f_aligned[i-1]) for i in range(1, len(x_f_aligned)))
    search_radius = min(search_radius, min_spacing * 0.49)
    
    for i, xf in enumerate(x_f_aligned):
        win_mask = (depth >= xf - search_radius) & (depth <= xf + search_radius)
        if np.any(win_mask):
            peak_val = np.max(profile_2d[win_mask])
            peak_depth = depth[win_mask][np.argmax(profile_2d[win_mask])]
            ax.plot(peak_depth, peak_val, 'r*', ms=10, label='Detected Peak' if i==0 else "")
            ax.axvspan(xf - search_radius, xf + search_radius, color='gray', alpha=0.1, label='Search Window' if i==0 else "")

    ax.set_xlim(3950, 4120)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Cepstrum Amplitude')
    ax.set_title(f'Zoomed Cepstrum for Brunone, S=10.0m, x1=4000.0m, N=8\nMin Spacing: {min_spacing:.2f}m, Search Window: ±{search_radius:.2f}m')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.tight_layout()
    
    out_img = r'C:\Users\Change\.gemini\antigravity\brain\4850e2b5-0e72-4d7e-8d5e-d9a05614c45c\cepstrum_fracture_zoom.png'
    plt.savefig(out_img, dpi=300)
    print(f"Saved {out_img}")

if __name__ == '__main__':
    main()
