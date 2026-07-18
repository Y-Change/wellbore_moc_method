import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_stretched_exp(x, a, b, beta):
    return a * np.exp(-((b * x)**beta))

def model_stretched_exp_c(x, a, b, beta, c):
    return a * np.exp(-((b * x)**beta)) + c

def load_data(x1=4000.0, sp=90.0):
    rows = []
    with open('output/analysis/decay_regression/03_extracted_peaks_csv/decay_table.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['friction_model'] == 'steady' and float(r['x1']) == x1 and float(r['spacing_m']) == sp:
                rows.append({
                    'dx': float(r['delta_x']),
                    'a_1d': float(r['alpha_1d']),
                    'n': int(r['n_total'])
                })
    
    rows.sort(key=lambda x: (x['n'], x['dx']))
    dx = np.array([r['dx'] for r in rows])
    a_1d = np.array([r['a_1d'] for r in rows])
    mask = a_1d > 0
    return dx[mask], a_1d[mask]

def main():
    dx, alpha = load_data()
    
    # 1. Linear Fit (Current)
    popt_lin, _ = curve_fit(model_stretched_exp, dx, alpha, p0=[1.0, 0.01, 0.5], bounds=([0.1, 1e-6, 0.1], [2.0, 1.0, 5.0]))
    
    # 2. Log Fit
    # log(alpha) = log(a) - (b*x)**beta
    def log_model(x, a, b, beta):
        return np.log(model_stretched_exp(x, a, b, beta))
    
    popt_log, _ = curve_fit(log_model, dx, np.log(alpha), p0=[1.0, 0.01, 0.5], bounds=([0.1, 1e-6, 0.1], [2.0, 1.0, 5.0]))
    
    # 3. Linear Fit with C
    popt_c, _ = curve_fit(model_stretched_exp_c, dx, alpha, p0=[1.0, 0.01, 0.5, 0.001], bounds=([0.1, 1e-6, 0.1, 0.0], [2.0, 1.0, 5.0, 0.1]))

    # 4. Log Fit with C
    def log_model_c(x, a, b, beta, c):
        return np.log(model_stretched_exp_c(x, a, b, beta, c))
    popt_log_c, _ = curve_fit(log_model_c, dx, np.log(alpha), p0=[1.0, 0.01, 0.5, 0.001], bounds=([0.1, 1e-6, 0.1, 0.0], [2.0, 1.0, 5.0, 0.1]))

    plt.figure(figsize=(10, 6))
    plt.scatter(dx, alpha, color='y', alpha=0.6, label='Data')
    
    dx_fine = np.linspace(0, max(dx), 100)
    plt.plot(dx_fine, model_stretched_exp(dx_fine, *popt_lin), 'c-', label='Linear Fit')
    plt.plot(dx_fine, model_stretched_exp(dx_fine, *popt_log), 'r--', label='Log Fit')
    plt.plot(dx_fine, model_stretched_exp_c(dx_fine, *popt_c), 'g-.', label='Linear Fit + c')
    plt.plot(dx_fine, model_stretched_exp_c(dx_fine, *popt_log_c), 'b:', lw=2, label='Log Fit + c')
    
    plt.yscale('log')
    plt.legend()
    plt.title('S=90m Fit Comparison')
    plt.savefig('scratch/fit_test_s90.png')
    print("Done")

if __name__ == '__main__':
    main()
