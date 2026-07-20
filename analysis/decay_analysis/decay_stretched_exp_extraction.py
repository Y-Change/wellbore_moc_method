import csv
import os
import sys
import numpy as np
from scipy.optimize import curve_fit

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

def model_stretched_exp(x, b, beta):
    return np.exp(-((b * x)**beta))

def log_model_stretched_exp(x, b, beta):
    return -((b * x)**beta)

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

def extract_parameters():
    csv_path = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_table.csv')
    all_rows = load_decay_rows(csv_path)
    
    # Filter steady and x1 between 2000 and 4000
    all_rows = [r for r in all_rows if r['friction_model'] == 'steady' or r['friction_model'] == 'brunone' and 1000.0 < r['x1'] < 4500.0]
    
    x1_list = sorted(list(set(r['x1'] for r in all_rows)))
    sp_list = sorted(list(set(r['spacing_m'] for r in all_rows)))
    
    frictions = sorted(list(set(r['friction_model'] for r in all_rows)))
    
    results = []
    
    for fr in frictions:
        for x1 in x1_list:
            for sp in sp_list:
                sub1 = [r for r in all_rows if r['friction_model'] == fr and r['x1'] == x1 and r['spacing_m'] == sp]
                if not sub1:
                    continue
                
                n_totals = sorted(list(set(r['n_total'] for r in sub1)))
                for n in n_totals:
                    pts = [r for r in sub1 if r['n_total'] == n]
                    pts.sort(key=lambda x: x['frac_idx'])
                    
                    dx = np.array([r['delta_x'] for r in pts])
                    alpha_2d = np.array([r['alpha_2d'] for r in pts])
                    
                    # We need at least 3 points to fit 2 parameters reliably
                    if len(alpha_2d) < 3:
                        continue
                    
                    # Fit Stretched Exponential in log space
                    try:
                        popt, _ = curve_fit(
                            log_model_stretched_exp, 
                            dx, 
                            np.log(alpha_2d), 
                            p0=[0.01, 0.5], 
                            bounds=([1e-6, 0.1], [1.0, 5.0])
                        )
                        b, beta = popt
                        r2 = r_squared(alpha_2d, model_stretched_exp(dx, b, beta))
                    except Exception as e:
                        print(f"Fit failed for {fr} x1={x1}, sp={sp}, n={n}: {e}")
                        continue
                        
                    results.append({
                        'friction_model': fr,
                        'x1': x1,
                        'spacing_m': sp,
                        'n_total': n,
                        'b': b,
                        'beta': beta,
                        'r2': r2
                    })
    
    # Save results
    out_csv = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_stretched_exp_params.csv')
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['friction_model', 'x1', 'spacing_m', 'n_total', 'a', 'b', 'beta', 'r2'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Successfully extracted parameters for {len(results)} cases. Saved to {out_csv}")

if __name__ == '__main__':
    extract_parameters()
