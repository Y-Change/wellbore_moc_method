import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

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

# Models to test
def model_pow_idx(x, k):
    return x**(-k)

def model_pow_dx(x, k):
    return (1 + x)**(-k)

def model_exp(x, a, b):
    return a * np.exp(-b * x)

def model_exp_offset(x, a, b, c):
    return a * np.exp(-b * x) + c

def model_rational(x, a, b):
    return 1.0 / (1.0 + a * x + b * (x**2))

def model_stretched_exp(x, a, b, beta):
    # a * exp(-(b*x)^beta)
    return a * np.exp(-((b * x)**beta))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_table.csv')
    
    # Load specific data: steady, x1=4000, spacing=20, n_total=8
    target_friction = 'steady'
    target_x1 = 4000.0
    target_sp = 20.0
    target_n = 8
    
    xs_idx = []
    xs_dx = []
    ys_1d = []
    ys_2d = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r['friction_model'] == target_friction and 
                float(r['x1']) == target_x1 and 
                float(r['spacing_m']) == target_sp and 
                int(r['n_total']) == target_n):
                
                xs_idx.append(int(r['frac_idx']))
                xs_dx.append(float(r['delta_x']))
                ys_1d.append(float(r['alpha_1d']))
                ys_2d.append(float(r['alpha_2d']))
                
    # Sort by index
    data = sorted(zip(xs_idx, xs_dx, ys_1d, ys_2d), key=lambda x: x[0])
    idx = np.array([d[0] for d in data])
    dx = np.array([d[1] for d in data])
    alpha_1d = np.array([d[2] for d in data])
    alpha_2d = np.array([d[3] for d in data])
    
    if len(idx) == 0:
        print("Data not found for the specified conditions.")
        return
        
    print(f"Data points: {len(idx)}")
    print(f"delta_x: {dx}")
    print(f"alpha_2d: {alpha_2d}")
    
    # We will test on 2D cepstrum as it's usually cleaner, but we can plot both
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = [('1D Cepstrum', alpha_1d, axes[0]), ('2D Cepstrum', alpha_2d, axes[1])]
    
    dx_fine = np.linspace(0, max(dx), 100)
    idx_fine = np.linspace(1, max(idx), 100)
    
    for title, y_data, ax in methods:
        ax.plot(dx, y_data, 'ko', markersize=8, label='Simulation Data')
        
        # 1. Pow(dx) model: (1+dx)^-k
        try:
            popt_pow, _ = curve_fit(model_pow_dx, dx, y_data, p0=[0.05])
            y_pred = model_pow_dx(dx, *popt_pow)
            r2_pow = r_squared(y_data, y_pred)
            ax.plot(dx_fine, model_pow_dx(dx_fine, *popt_pow), 
                    label=f'Pow(dx): $(1+\Delta x)^{{-{popt_pow[0]:.3f}}}$, $R^2={r2_pow:.4f}$', lw=2)
        except:
            pass
            
        # 2. Exponential decay: a*exp(-b*dx)
        try:
            popt_exp, _ = curve_fit(model_exp, dx, y_data, p0=[1.0, 0.01])
            y_pred = model_exp(dx, *popt_exp)
            r2_exp = r_squared(y_data, y_pred)
            ax.plot(dx_fine, model_exp(dx_fine, *popt_exp), 
                    label=f'Exp: ${popt_exp[0]:.2f}e^{{-{popt_exp[1]:.4f}\Delta x}}$, $R^2={r2_exp:.4f}$', lw=2)
        except:
            pass
            
        # 3. Rational model: 1 / (1 + a*dx + b*dx^2)
        try:
            popt_rat, _ = curve_fit(model_rational, dx, y_data, p0=[0.01, 0.0001])
            y_pred = model_rational(dx, *popt_rat)
            r2_rat = r_squared(y_data, y_pred)
            ax.plot(dx_fine, model_rational(dx_fine, *popt_rat), 
                    label=f'Rational: $1/(1+{popt_rat[0]:.4f}\Delta x+{popt_rat[1]:.5f}\Delta x^2)$, $R^2={r2_rat:.4f}$', lw=2)
        except:
            pass
            
        # 4. Stretched Exp: a*exp(-(b*dx)^beta)
        try:
            # Add bounds to prevent beta from blowing up
            popt_str, _ = curve_fit(model_stretched_exp, dx, y_data, p0=[1.0, 0.01, 0.5], bounds=([0, 0, 0.1], [10, 1, 5]))
            y_pred = model_stretched_exp(dx, *popt_str)
            r2_str = r_squared(y_data, y_pred)
            ax.plot(dx_fine, model_stretched_exp(dx_fine, *popt_str), 
                    label=f'Str.Exp: ${popt_str[0]:.2f}e^{{-({popt_str[1]:.4f}\Delta x)^{{{popt_str[2]:.2f}}}}}$, $R^2={r2_str:.4f}$', lw=2)
        except:
            pass
            
        ax.set_title(f"{title} - Fitting Models")
        ax.set_xlabel('$\Delta x$ (m)')
        ax.set_ylabel('Relative Peak $\\alpha$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    out_path = os.path.join(output_path(SERIES_DECAY_REGRESSION, '', ''), 'test_new_fits.png')
    save_figure(fig, out_path)
    print(f"Saved figure to {out_path}")

if __name__ == '__main__':
    main()
