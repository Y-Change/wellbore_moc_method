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

def load_decay_rows(csv_path: str):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ('x1', 'spacing_m', 'x_f', 'delta_x', 'P_1d', 'P_2d', 'alpha_1d', 'alpha_2d'):
                r[k] = float(r[k])
            for k in ('n_total', 'frac_idx'):
                r[k] = int(r[k])
            rows.append(r)
    return rows

def compute_fits(group_rows, alpha_key='alpha_2d'):
    alpha = np.array([r[alpha_key] for r in group_rows])
    delta_x = np.array([r['delta_x'] for r in group_rows])
    idx = np.array([r['frac_idx'] for r in group_rows])
    
    mask = (alpha > 0)
    alpha = alpha[mask]
    delta_x = delta_x[mask]
    idx = idx[mask]
    
    # Pow(dx)
    if np.any(delta_x > 0):
        dx_m = delta_x[delta_x > 0]
        a_m = alpha[delta_x > 0]
        k_dx, _, _, _ = np.linalg.lstsq(np.log(1 + dx_m)[:, None], -np.log(a_m), rcond=None)
        k_dx = k_dx[0]
        pred1 = (1 + delta_x)**(-k_dx)
        r2_kdx = 1 - np.sum((alpha - pred1)**2) / np.sum((alpha - np.mean(alpha))**2)
    else:
        k_dx, r2_kdx = np.nan, np.nan
        
    # Pow(idx)
    if np.any(idx > 1):
        idx_m = idx[idx > 1]
        a_m = alpha[idx > 1]
        k, _, _, _ = np.linalg.lstsq(np.log(idx_m)[:, None], -np.log(a_m), rcond=None)
        k = k[0]
        pred2 = idx**(-k)
        r2_k = 1 - np.sum((alpha - pred2)**2) / np.sum((alpha - np.mean(alpha))**2)
    else:
        k, r2_k = np.nan, np.nan
        
    return k_dx, r2_kdx, k, r2_k

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_table.csv')
    all_rows = load_decay_rows(csv_path)
    all_rows = [r for r in all_rows if 1000.0 < r['x1'] < 4500.0]
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '04_collapse_and_scaling', '')
    os.makedirs(out_dir, exist_ok=True)
    
    frictions = ['steady', 'brunone']
    
    for fr in frictions:
        # Filter rows
        rows = [r for r in all_rows if r['friction_model'] == fr]
        if not rows:
            continue
            
        x1_list = sorted(list(set(r['x1'] for r in rows)))
        spacings = sorted(list(set(r['spacing_m'] for r in rows)))
        
        # We will plot 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        cmap = plt.cm.Set1
        
        for i, x1 in enumerate(x1_list):
            k_idx_list = []
            r2_idx_list = []
            k_dx_list = []
            r2_dx_list = []
            valid_S = []
            
            for sp in spacings:
                sub = [r for r in rows if r['x1'] == x1 and r['spacing_m'] == sp]
                if not sub:
                    continue
                k_dx, r2_kdx, k, r2_k = compute_fits(sub, 'alpha_2d')
                k_idx_list.append(k)
                r2_idx_list.append(r2_k)
                k_dx_list.append(k_dx)
                r2_dx_list.append(r2_kdx)
                valid_S.append(sp)
                
            color = cmap(i / max(1, len(x1_list)-1))
            label = f'x1 = {int(x1)}m'
            
            axes[0, 0].plot(valid_S, k_idx_list, marker='o', lw=2, color=color, label=label)
            axes[0, 1].plot(valid_S, r2_idx_list, marker='o', lw=2, color=color, label=label)
            axes[1, 0].plot(valid_S, k_dx_list, marker='s', ls='--', lw=2, color=color, label=label)
            axes[1, 1].plot(valid_S, r2_dx_list, marker='s', ls='--', lw=2, color=color, label=label)
            
        axes[0, 0].set_ylabel('Decay Exponent k')
        axes[0, 0].set_title('Pow(idx) Model: Exponent k vs Spacing')
        axes[0, 0].set_xlabel('Spacing S [m]')
        axes[0, 0].legend()
        
        axes[0, 1].set_ylabel('Goodness of Fit R²')
        axes[0, 1].set_title('Pow(idx) Model: R² vs Spacing')
        axes[0, 1].set_xlabel('Spacing S [m]')
        axes[0, 1].set_ylim(0.9, 1.01)
        
        axes[1, 0].set_ylabel('Decay Exponent k_dx')
        axes[1, 0].set_title('Pow(dx) Model: Exponent k_dx vs Spacing')
        axes[1, 0].set_xlabel('Spacing S [m]')
        
        axes[1, 1].set_ylabel('Goodness of Fit R²')
        axes[1, 1].set_title('Pow(dx) Model: R² vs Spacing')
        axes[1, 1].set_xlabel('Spacing S [m]')
        
        fig.suptitle(f'Analysis of Decay Parameters ({fr} Friction, 2D Cepstrum)', fontsize=14)
        fig.tight_layout()
        
        save_path = os.path.join(out_dir, f'{fr}_k_parameter_analysis.png')
        save_figure(fig, save_path)
        plt.close(fig)
        print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
