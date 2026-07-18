import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

from paths import output_path, SERIES_DECAY_REGRESSION
from analysis.paper_plots import apply_paper_rc, save_figure

apply_paper_rc()

def model_pow_idx(x, k):
    return x**(-k)

def model_pow_dx(x, k):
    return (1 + x)**(-k)

def model_exp(x, a, b):
    return a * np.exp(-b * x)

def model_rational(x, a, b):
    return 1.0 / (1.0 + a * x + b * (x**2))

def model_stretched_exp(x, a, b, beta):
    return a * np.exp(-((b * x)**beta))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

def fit_all_models(group_rows, alpha_key):
    alpha = np.array([r[alpha_key] for r in group_rows])
    delta_x = np.array([r['delta_x'] for r in group_rows])
    idx = np.array([r['frac_idx'] for r in group_rows])
    
    mask = (alpha > 0)
    alpha = alpha[mask]
    delta_x = delta_x[mask]
    idx = idx[mask]
    
    res = {
        'pow_idx': {'k': np.nan, 'r2': np.nan},
        'pow_dx': {'k': np.nan, 'r2': np.nan},
        'exp': {'a': np.nan, 'b': np.nan, 'r2': np.nan},
        'rat': {'a': np.nan, 'b': np.nan, 'r2': np.nan},
        'str_exp': {'a': np.nan, 'b': np.nan, 'beta': np.nan, 'r2': np.nan}
    }
    
    if len(alpha) < 3:
        return res
        
    if np.any(idx > 1):
        idx_m = idx[idx > 1]
        a_m = alpha[idx > 1]
        try:
            k, _, _, _ = np.linalg.lstsq(np.log(idx_m)[:, None], -np.log(a_m), rcond=None)
            res['pow_idx']['k'] = k[0]
            pred = idx**(-k[0])
            res['pow_idx']['r2'] = r_squared(alpha, pred)
        except:
            pass
            
    if np.any(delta_x > 0):
        dx_m = delta_x[delta_x > 0]
        a_m = alpha[delta_x > 0]
        try:
            k_dx, _, _, _ = np.linalg.lstsq(np.log(1 + dx_m)[:, None], -np.log(a_m), rcond=None)
            res['pow_dx']['k'] = k_dx[0]
            pred = (1 + delta_x)**(-k_dx[0])
            res['pow_dx']['r2'] = r_squared(alpha, pred)
        except:
            pass
            
    try:
        popt, _ = curve_fit(model_exp, delta_x, alpha, p0=[1.0, 0.01])
        res['exp']['a'], res['exp']['b'] = popt
        res['exp']['r2'] = r_squared(alpha, model_exp(delta_x, *popt))
    except:
        pass
        
    try:
        popt, _ = curve_fit(model_rational, delta_x, alpha, p0=[0.01, 0.0001])
        res['rat']['a'], res['rat']['b'] = popt
        res['rat']['r2'] = r_squared(alpha, model_rational(delta_x, *popt))
    except:
        pass
        
    try:
        popt, _ = curve_fit(model_stretched_exp, delta_x, alpha, p0=[1.0, 0.01, 0.5], bounds=([0, 0, 0.1], [10, 1, 5]))
        res['str_exp']['a'], res['str_exp']['b'], res['str_exp']['beta'] = popt
        res['str_exp']['r2'] = r_squared(alpha, model_stretched_exp(delta_x, *popt))
    except:
        pass
        
    return res

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

def plot_for_x1(rows, x1, friction):
    spacings = sorted(list(set(r['spacing_m'] for r in rows)))
    n_cols = (len(spacings) + 1) // 2
    
    alphas = ['alpha_1d', 'alpha_2d']
    titles = ['1D Cepstrum', '2D Cepstrum']
    suffixes = ['1d', '2d']
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '07_new_fits_comparisons', '')
    os.makedirs(out_dir, exist_ok=True)
    
    for method_idx, alpha_key in enumerate(alphas):
        fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8.5), sharex=False, sharey=True)
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
                
                xs = [r['delta_x'] for r in pts]
                ys = [r[alpha_key] for r in pts]
                
                color = cmap(c_idx / max(1, len(n_totals)-1))
                ax.plot(xs, ys, marker='o', ls='none', color=color, alpha=0.6, label=f'n={n}')
                
            fits = fit_all_models(sub, alpha_key)
            
            all_idx = np.linspace(1, max(n_totals), 50)
            all_dx = sp * (all_idx - 1)
            
            if not np.isnan(fits['pow_idx']['r2']):
                ax.plot(all_dx, model_pow_idx(all_idx, fits['pow_idx']['k']), 'r-', lw=2, label=f"Pow(idx) R²={fits['pow_idx']['r2']:.3f}")
            if not np.isnan(fits['pow_dx']['r2']):
                ax.plot(all_dx, model_pow_dx(all_dx, fits['pow_dx']['k']), 'b--', lw=2, label=f"Pow(dx) R²={fits['pow_dx']['r2']:.3f}")
            if not np.isnan(fits['exp']['r2']):
                ax.plot(all_dx, model_exp(all_dx, fits['exp']['a'], fits['exp']['b']), 'g-.', lw=2, label=f"Exp R²={fits['exp']['r2']:.3f}")
            if not np.isnan(fits['rat']['r2']):
                ax.plot(all_dx, model_rational(all_dx, fits['rat']['a'], fits['rat']['b']), 'm:', lw=2, label=f"Rational R²={fits['rat']['r2']:.3f}")
            if not np.isnan(fits['str_exp']['r2']):
                ax.plot(all_dx, model_stretched_exp(all_dx, fits['str_exp']['a'], fits['str_exp']['b'], fits['str_exp']['beta']), 'c-', lw=2, alpha=0.5, label=f"Str.Exp R²={fits['str_exp']['r2']:.3f}")

            ax.set_yscale('log')
            ax.set_xlabel('Distance $\Delta x$ (m)')
            if sp_idx % n_cols == 0:
                ax.set_ylabel(f'Relative Peak ({alpha_key})')
            
            ax.set_title(f"S={sp}m", fontsize=11, pad=8)
            ax.legend(fontsize=7, loc='best')

        for sp_idx in range(len(spacings), len(axes_flat)):
            axes_flat[sp_idx].set_visible(False)

        fig.suptitle(f'New Decay Models ({friction}, {titles[method_idx]}) for x1 = {x1}m', fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = os.path.join(out_dir, f'{friction}_new_fits_compare_x1_{int(x1)}_{suffixes[method_idx]}.png')
        save_figure(fig, save_path)
        plt.close(fig)

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_table.csv')
    all_rows = load_decay_rows(csv_path)
    
    all_rows = [r for r in all_rows if abs(r['x1'] - 4000.0) < 1.0]
    
    frictions = sorted(list(set(r['friction_model'] for r in all_rows)))
    for fr in frictions:
        rows = [r for r in all_rows if r['friction_model'] == fr]
        plot_for_x1(rows, 4000.0, fr)

if __name__ == '__main__':
    main()
