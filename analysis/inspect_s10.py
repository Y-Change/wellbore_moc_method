import csv
import sys
import os

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

from paths import output_path, SERIES_DECAY_REGRESSION

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, 'data', 'decay_table.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if float(r['x1']) == 1000.0 and float(r['spacing_m']) == 10.0]
        
    print("x1=1000.0, spacing=10.0")
    for n in sorted(list(set(int(r['n_total']) for r in rows))):
        print(f"\n--- n_total = {n} ---")
        pts = [r for r in rows if int(r['n_total']) == n]
        pts.sort(key=lambda x: int(x['frac_idx']))
        for p in pts:
            print(f"idx={p['frac_idx']}, x_f={p['x_f']}, dx={p['delta_x']}, P_2d={p['P_2d']}, alpha_2d={p['alpha_2d']}")

if __name__ == '__main__':
    main()
