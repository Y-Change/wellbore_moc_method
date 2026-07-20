# -*- coding: utf-8 -*-
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

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
from analysis.plotting.paper_plots import apply_paper_rc, save_figure

apply_paper_rc()

def model_pidx(idx, k):
    idx = np.asarray(idx, dtype=float)
    return idx ** (-k)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

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

def fit_models(group_rows, alpha_key):
    alpha = np.array([r[alpha_key] for r in group_rows])
    idx = np.array([r['frac_idx'] for r in group_rows])
    
    mask = (alpha > 0)
    alpha = alpha[mask]
    idx = idx[mask]
    
    res = {'pidx': {'k': np.nan, 'r2': np.nan}}
    
    if len(alpha) >= 3:
        try:
            popt, _ = curve_fit(model_pidx, idx, alpha, p0=[1.0])
            k = popt[0]
            r2 = r_squared(alpha, model_pidx(idx, k))
            res['pidx'] = {'k': k, 'r2': r2}
        except:
            pass
            
    return res

def plot_for_x1(rows, x1, friction):
    spacings = sorted(list(set(r['spacing_m'] for r in rows)))
    if not spacings:
        return
        
    n_cols = (len(spacings) + 1) // 2
    alphas = ['alpha_1d', 'alpha_2d']
    titles = ['1D Cepstrum', '2D Cepstrum']
    suffixes = ['1d', '2d']
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '05_friction_comparisons_pidx', '')
    os.makedirs(out_dir, exist_ok=True)
    
    for method_idx, alpha_key in enumerate(alphas):
        fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7), sharex=False, sharey=True)
        axes_flat = axes.flatten() if n_cols > 1 else axes
        
        for sp_idx, sp in enumerate(spacings):
            ax = axes_flat[sp_idx]
            sub = [r for r in rows if r['x1'] == x1 and r['spacing_m'] == sp]
            if not sub:
                continue
                
            n_totals = sorted(list(set(r['n_total'] for r in sub)))
            cmap = plt.cm.viridis
            
            for c_idx, n in enumerate(n_totals):
                pts = [r for r in sub if r['n_total'] == n]
                pts.sort(key=lambda x: x['frac_idx'])
                
                xs = [r['frac_idx'] for r in pts]
                ys = [r[alpha_key] for r in pts]
                
                color = cmap(c_idx / max(1, len(n_totals)-1))
                ax.plot(xs, ys, marker='o', ls='none', color=color, alpha=0.6, label=f'n={n}')
                
            fits = fit_models(sub, alpha_key)
            
            all_idx = np.linspace(1, max(n_totals), 100)
            
            title_str = f'S={sp}m'
            if not np.isnan(fits['pidx']['r2']):
                k = fits['pidx']['k']
                r2 = fits['pidx']['r2']
                ax.plot(all_idx, model_pidx(all_idx, k), 'r-', lw=2.5, label='Pidx Fit')
                title_str += f'\n$k={k:.4f}, R^2={r2:.3f}$'

            ax.set_yscale('log')
            ax.set_xlabel('Fracture Index (Level)')
            # Only set ticks to integers for x axis
            ax.set_xticks(range(1, max(n_totals) + 1))
            if sp_idx % n_cols == 0:
                ax.set_ylabel(f'Relative Peak ({alpha_key})')
            
            ax.set_title(title_str, fontsize=9)
            
        for i in range(len(spacings), len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        fig.suptitle(f'Pidx Decay Model Comparison ({friction}, {titles[method_idx]}) for x1 = {x1}m', fontsize=14, y=1.02)
        
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.10, 0.15), fontsize=8, frameon=False)
            
        fig.tight_layout()
        save_path = os.path.join(out_dir, f'{friction}_pidx_compare_x1_{int(x1)}_{suffixes[method_idx]}.png')
        save_figure(fig, save_path)
        plt.close(fig)
        print(f"Saved {save_path}")

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_table.csv')
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}")
        return
    all_rows = load_decay_rows(csv_path)
    all_rows = [r for r in all_rows if 1000.0 < r['x1'] < 4500.0]
    
    frictions = sorted(list(set(r['friction_model'] for r in all_rows)))
    for fr in frictions:
        fr_rows = [r for r in all_rows if r['friction_model'] == fr]
        x1s = sorted(list(set(r['x1'] for r in fr_rows)))
        for x1 in x1s:
            plot_for_x1(fr_rows, x1, fr)

if __name__ == '__main__':
    main()
