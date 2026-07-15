import csv
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
from analysis.paper_plots import apply_paper_rc, save_figure

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

def fit_models(group_rows, alpha_key):
    """
    Fit models to a group of points and return parameters and R2.
    Model 1: alpha = exp(-lam * delta_x)
    Model 2: alpha = idx^(-k)
    """
    alpha = np.array([r[alpha_key] for r in group_rows])
    delta_x = np.array([r['delta_x'] for r in group_rows])
    idx = np.array([r['frac_idx'] for r in group_rows])
    
    mask = (alpha > 0)
    alpha = alpha[mask]
    delta_x = delta_x[mask]
    idx = idx[mask]
    
    if len(alpha) < 2:
        return {'lam': np.nan, 'lam_r2': np.nan, 'k': np.nan, 'k_r2': np.nan}
        
    # Model 1
    if np.any(delta_x > 0):
        dx_m = delta_x[delta_x > 0]
        a_m = alpha[delta_x > 0]
        lam, _, _, _ = np.linalg.lstsq(dx_m[:, None], -np.log(a_m), rcond=None)
        lam = lam[0]
        pred1 = np.exp(-lam * delta_x)
        r2_lam = 1 - np.sum((alpha - pred1)**2) / np.sum((alpha - np.mean(alpha))**2)
    else:
        lam, r2_lam = np.nan, np.nan
        
    # Model 2
    if np.any(idx > 1):
        idx_m = idx[idx > 1]
        a_m = alpha[idx > 1]
        k, _, _, _ = np.linalg.lstsq(np.log(idx_m)[:, None], -np.log(a_m), rcond=None)
        k = k[0]
        pred2 = idx**(-k)
        r2_k = 1 - np.sum((alpha - pred2)**2) / np.sum((alpha - np.mean(alpha))**2)
    else:
        k, r2_k = np.nan, np.nan
        
    return {'lam': lam, 'lam_r2': r2_lam, 'k': k, 'k_r2': r2_k}

def plot_for_x1(rows, x1, friction):
    spacings = sorted(list(set(r['spacing_m'] for r in rows)))
    
    fig, axes = plt.subplots(2, len(spacings), figsize=(3.5 * len(spacings), 7), sharex=False, sharey=True)
    
    alphas = ['alpha_1d', 'alpha_2d']
    titles = ['1D Cepstrum', '2D Cepstrum']
    
    for row_idx, alpha_key in enumerate(alphas):
        for col_idx, sp in enumerate(spacings):
            ax = axes[row_idx, col_idx]
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
            all_idx = np.linspace(1, max(n_totals), 50)
            
            if not np.isnan(fits['k']):
                ax.plot(all_idx, all_idx**(-fits['k']), 'r-', lw=2, label=f"idx^{{-{fits['k']:.2f}}}")
                
            if not np.isnan(fits['lam']):
                ax.plot(all_idx, np.exp(-fits['lam'] * sp * (all_idx - 1)), 'b--', lw=2, label=f"exp(-{fits['lam']:.4f}*dx)")

            ax.set_yscale('log')
            ax.set_xlabel('Fracture Index')
            if col_idx == 0:
                ax.set_ylabel(f'Relative Peak ({alpha_key})')
            
            title_str = f"S={sp}m | {titles[row_idx]}\nExp R2:{fits['lam_r2']:.3f} | Pow R2:{fits['k_r2']:.3f}"
            ax.set_title(title_str, fontsize=10)
            if row_idx == 0 and col_idx == len(spacings) - 1:
                ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle(f'Decay Model Comparison ({friction}) for x1 = {x1}m', fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, 'model_compare', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{friction}_decay_compare_x1_{int(x1)}.png')
    save_figure(fig, save_path)
    return save_path

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, 'data', 'decay_table.csv')
    all_rows = load_decay_rows(csv_path)
    
    frictions = sorted(list(set(r['friction_model'] for r in all_rows)))
    for fr in frictions:
        rows = [r for r in all_rows if r['friction_model'] == fr]
        x1_list = sorted(list(set(r['x1'] for r in rows)))
        for x1 in x1_list:
            p = plot_for_x1(rows, x1, fr)
            if p:
                print(f"Saved {p}")

if __name__ == '__main__':
    main()
