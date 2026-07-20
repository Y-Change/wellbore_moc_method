import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from analysis.plotting.paper_plots import apply_paper_rc, save_figure

apply_paper_rc()

def load_params(csv_path: str):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ('x1', 'spacing_m', 'b', 'beta', 'r2'):
                r[k] = float(r[k])
            for k in ('n_total',):
                r[k] = int(r[k])
            rows.append(r)
    return rows

def plot_heatmaps(rows):
    x1_list = sorted(list(set(r['x1'] for r in rows)))
    sp_list = sorted(list(set(r['spacing_m'] for r in rows)))
    
    b_matrix = np.full((len(x1_list), len(sp_list)), np.nan)
    beta_matrix = np.full((len(x1_list), len(sp_list)), np.nan)
    
    for i, x1 in enumerate(x1_list):
        for j, sp in enumerate(sp_list):
            sub = [r for r in rows if r['x1'] == x1 and r['spacing_m'] == sp]
            if sub:
                # average over n_total
                b_matrix[i, j] = np.mean([r['b'] for r in sub])
                beta_matrix[i, j] = np.mean([r['beta'] for r in sub])
                
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = ax1.imshow(b_matrix, cmap='viridis', aspect='auto', origin='lower',
                     extent=[min(sp_list)-5, max(sp_list)+5, min(x1_list)-250, max(x1_list)+250])
    ax1.set_xlabel('Spacing S (m)')
    ax1.set_ylabel('Depth X1 (m)')
    ax1.set_title('Spatial Decay Rate $b$ Heatmap')
    fig.colorbar(im1, ax=ax1, label='Decay Rate $b$')
    
    im2 = ax2.imshow(beta_matrix, cmap='plasma', aspect='auto', origin='lower',
                     extent=[min(sp_list)-5, max(sp_list)+5, min(x1_list)-250, max(x1_list)+250])
    ax2.set_xlabel('Spacing S (m)')
    ax2.set_ylabel('Depth X1 (m)')
    ax2.set_title('Stretching Exponent $\\beta$ Heatmap')
    fig.colorbar(im2, ax=ax2, label='Shape Factor $\\beta$')
    
    fig.suptitle('Stretched Exponential Parameters (Steady Friction)', fontsize=16)
    fig.tight_layout()
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '04_collapse_and_scaling', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'stretched_exp_heatmaps.png')
    save_figure(fig, save_path)
    print(f"Saved {save_path}")

def plot_collapse(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x1_list = sorted(list(set(r['x1'] for r in rows)))
    cmap = plt.cm.jet
    
    for i, x1 in enumerate(x1_list):
        sub = [r for r in rows if r['x1'] == x1]
        
        sp_list = sorted(list(set(r['spacing_m'] for r in sub)))
        b_vals = []
        for sp in sp_list:
            b_vals.append(np.mean([r['b'] for r in sub if r['spacing_m'] == sp]))
            
        color = cmap(i / max(1, len(x1_list)-1))
        
        x_ratio = x1 / np.array(sp_list)
        ax1.plot(x_ratio, b_vals, marker='o', ls='-', color=color, alpha=0.7, label=f'X1={x1}m')
        
        ax2.plot(sp_list, b_vals, marker='s', ls='-', color=color, alpha=0.7, label=f'X1={x1}m')
        
    ax1.set_xlabel('Dimensionless Ratio $X_1 / S$')
    ax1.set_ylabel('Spatial Decay Rate $b$')
    ax1.set_title('Collapse Attempt: $b$ vs $X_1 / S$')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Spacing $S$ (m)')
    ax2.set_ylabel('Spatial Decay Rate $b$')
    ax2.set_title('$b$ vs Spacing $S$')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Scaling Laws of Decay Rate $b$', fontsize=16)
    fig.tight_layout()
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '04_collapse_and_scaling', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'stretched_exp_collapse.png')
    save_figure(fig, save_path)
    print(f"Saved {save_path}")

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_stretched_exp_params.csv')
    if not os.path.exists(csv_path):
        print("Params CSV not found. Please run extraction script first.")
        return
        
    rows = load_params(csv_path)
    
    plot_heatmaps(rows)
    plot_collapse(rows)

if __name__ == '__main__':
    main()
