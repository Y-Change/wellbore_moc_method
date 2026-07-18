import csv
import numpy as np

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def debug_s100():
    rows = []
    with open('output/analysis/decay_regression/03_extracted_peaks_csv/decay_table.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['friction_model'] == 'steady' and float(r['x1']) == 4000.0 and float(r['spacing_m']) == 100.0:
                rows.append({
                    'idx': int(r['frac_idx']),
                    'dx': float(r['delta_x']),
                    'a_2d': float(r['alpha_2d']),
                    'n': int(r['n_total'])
                })
    
    alpha = np.array([r['a_2d'] for r in rows])
    idx = np.array([r['idx'] for r in rows])
    dx = np.array([r['dx'] for r in rows])
    
    print("idx:", idx)
    print("dx:", dx)
    print("alpha:", alpha)
    
    dx_m = dx[dx > 0]
    a_m = alpha[dx > 0]
    k_dx, _, _, _ = np.linalg.lstsq(np.log(1 + dx_m)[:, None], -np.log(a_m), rcond=None)
    k_dx = k_dx[0]
    pred = (1 + dx)**(-k_dx)
    r2 = r_squared(alpha, pred)
    
    print(f"k_dx = {k_dx}")
    print(f"pred = {pred}")
    print(f"R2 = {r2}")

if __name__ == '__main__':
    debug_s100()
