import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import argparse

import os
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'paths.py')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find wellbore_moc_method root')
    _d = _parent

from paths import output_path, SERIES_DECAY_REGRESSION
from cepstrum_mocdata import compute_moc_cepstrum, compute_moc_cepstrum_1d
from analysis.plotting.paper_plots import apply_paper_rc, save_figure

apply_paper_rc()

def process_file(npz_path, out_dir, method='2d'):
    try:
        fname = os.path.basename(npz_path).replace('.npz', '')
        # Name format: {friction_key}_x1_{x1}_sp_{sp}_n_{n_fracs}
        parts = fname.split('_')
        friction = parts[0]
        x1 = float(parts[2])
        sp = float(parts[4])
        n = int(parts[6])
        
        save_path = os.path.join(out_dir, friction, f"x1_{int(x1)}", f"{fname}_{method}.png")
        if os.path.exists(save_path):
            return  # skip if already generated
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = np.load(npz_path)
        t_sim = data['t_sim']
        H_wh = data['H_wh']
        v = data['v'][0]
        fs = data['fs'][0]
        ts = data['ts'][0]
        L = data['L'][0]
        x_f_aligned = data['x_f_aligned']
        
        if method == '1d':
            out = compute_moc_cepstrum_1d(t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L)
            depth = out['depth']
            profile = out['response']
            method_str = '1D'
        else:
            out_2d = compute_moc_cepstrum(t_sim, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
                                          wlen_sec=30.0, hop_sec=5.0, win_type='hamming')
            depth = out_2d['depth']
            profile = -np.sum(out_2d['C'], axis=1)
            method_str = '2D'
        
        # Plot range: from 50m before first frac, to 100m (or 1.5*spacing) after last frac
        min_d = max(0, x_f_aligned[0] - 50)
        max_d = min(L, x_f_aligned[-1] + max(50, sp * 1.5))
        mask = (depth >= min_d) & (depth <= max_d)
        d_plot = depth[mask]
        p_plot = profile[mask]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(d_plot, p_plot, 'k-', lw=1.2, label=f'{method_str} Cepstrum Profile')
        
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(x_f_aligned)))
        
        search_radius = 15.0
        if len(x_f_aligned) > 1:
            min_spacing = min(abs(x_f_aligned[i] - x_f_aligned[i-1]) for i in range(1, len(x_f_aligned)))
            search_radius = min(search_radius, min_spacing * 0.49)
            
        peak_val_0 = 1.0
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
                
            if i == 0:
                peak_val_0 = peak_val if peak_val != 0 else 1.0
                
            alpha = peak_val / peak_val_0
            
            ax.plot(peak_d, peak_val, 'o', color=colors[i], markersize=6, 
                    label=f'Frac {i+1} ({xf:.1f}m)')
            ax.axvline(xf, color=colors[i], linestyle='--', alpha=0.4)
            
            # Label the decay ratio alpha
            ax.text(peak_d, peak_val, f" $\\alpha$={alpha:.3f}", 
                    color=colors[i], fontsize=9, fontweight='bold',
                    verticalalignment='bottom', horizontalalignment='left')
            
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Cepstrum Amplitude')
        title_str = fname.replace('_', ' ').replace('sp', 'S=').replace('n', 'n=').capitalize()
        ax.set_title(f'Local Peak Extraction ({method_str}) | {title_str}')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        save_figure(fig, save_path)
        plt.close(fig)
        return f"Saved {save_path}"
    except Exception as e:
        return f"Error on {npz_path}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Plot zoom views for all simulation cases")
    parser.add_argument('--n_only', type=int, default=None, help='Only plot specific n_fracs (e.g. 8) to save time')
    parser.add_argument('--friction', type=str, default=None, help='Filter by friction model (e.g. steady or brunone)')
    parser.add_argument('--method', type=str, choices=['1d', '2d', 'both'], default='2d', help='Cepstrum method to plot')
    parser.add_argument('--workers', type=int, default=6, help='Number of multiprocessing workers')
    args = parser.parse_args()

    raw_dir = output_path(SERIES_DECAY_REGRESSION, '01_simulated_waves', '')
    out_dir = output_path(SERIES_DECAY_REGRESSION, '02_peak_tracking_zooms', '')
    
    npz_files = glob(os.path.join(raw_dir, "*.npz"))
    print(f"Found total {len(npz_files)} npz files.")
    
    if args.n_only is not None:
        npz_files = [f for f in npz_files if f.endswith(f"_n_{args.n_only}.npz")]
    if args.friction is not None:
        npz_files = [f for f in npz_files if os.path.basename(f).startswith(args.friction)]
        
    print(f"Filtered down to {len(npz_files)} files to plot.")
    os.makedirs(out_dir, exist_ok=True)
    
    if len(npz_files) == 0:
        return
        
    methods = ['1d', '2d'] if args.method == 'both' else [args.method]
        
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for f in npz_files:
            for m in methods:
                futures.append(executor.submit(process_file, f, out_dir, m))
                
        for i, fut in enumerate(futures):
            res = fut.result()
            if res:
                print(f"[{i+1}/{len(npz_files)}] {res}")

if __name__ == '__main__':
    main()
