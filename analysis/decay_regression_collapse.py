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

def plot_divergence_vs_collapse(rows, x1, friction, alpha_key='alpha_2d'):
    spacings = sorted(list(set(r['spacing_m'] for r in rows)))
    
    sub = [r for r in rows if r['x1'] == x1]
    if not sub:
        return None
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    
    cmap = plt.cm.plasma
    
    all_idx = []
    all_alpha = []
    pts_max_all = []
    
    for i, sp in enumerate(spacings):
        pts = [r for r in sub if r['spacing_m'] == sp]
        max_n = max([r['n_total'] for r in pts])
        pts_max = [r for r in pts if r['n_total'] == max_n]
        pts_max.sort(key=lambda x: x['frac_idx'])
        pts_max_all.extend(pts_max)
        
        dx = [r['delta_x'] for r in pts_max]
        idx = [r['frac_idx'] for r in pts_max]
        alpha = [r[alpha_key] for r in pts_max]
        
        all_idx.extend(idx)
        all_alpha.extend(alpha)
        
        color = cmap(i / max(1, len(spacings)-1))
        
        ax1.plot(dx, alpha, marker='o', ls='--', lw=1.5, color=color, alpha=0.8, label=f'S = {sp}m')
        ax2.plot(idx, alpha, marker='o', ls='none', color=color, alpha=0.8, label=f'S = {sp}m')

    all_idx = np.array(all_idx)
    all_alpha = np.array(all_alpha)
    
    # Shade Divergence Envelope (Fan)
    boundary_dx = []
    boundary_alpha = []
    max_sp = max(spacings)
    min_sp = min(spacings)
    
    pts_max_sp = [r for r in pts_max_all if r['spacing_m'] == max_sp]
    boundary_dx.extend([r['delta_x'] for r in pts_max_sp])
    boundary_alpha.extend([r[alpha_key] for r in pts_max_sp])
    
    for sp in sorted(spacings, reverse=True)[1:-1]:
        pts_sp = [r for r in pts_max_all if r['spacing_m'] == sp]
        if pts_sp:
            pt = max(pts_sp, key=lambda x: x['frac_idx'])
            boundary_dx.append(pt['delta_x'])
            boundary_alpha.append(pt[alpha_key])
            
    pts_min_sp = [r for r in pts_max_all if r['spacing_m'] == min_sp]
    boundary_dx.extend([r['delta_x'] for r in reversed(pts_min_sp)])
    boundary_alpha.extend([r[alpha_key] for r in reversed(pts_min_sp)])
    
    ax1.fill(boundary_dx, boundary_alpha, color='royalblue', alpha=0.12, label='Divergence Envelope', zorder=0)
    
    # Shade Collapse Envelope (Band)
    idx_unique = sorted(list(set(all_idx)))
    min_a = [np.min(all_alpha[all_idx == i]) for i in idx_unique]
    max_a = [np.max(all_alpha[all_idx == i]) for i in idx_unique]
    ax2.fill_between(idx_unique, min_a, max_a, color='red', alpha=0.15, label='Collapse Envelope', zorder=0)
    
    mask = (all_alpha > 0) & (all_idx > 1)
    if np.any(mask):
        k, _, _, _ = np.linalg.lstsq(np.log(all_idx[mask])[:, None], -np.log(all_alpha[mask]), rcond=None)
        k = k[0]
        
        idx_dense = np.linspace(1, max(all_idx), 100)
        alpha_fit = idx_dense**(-k)
        
        pred = all_idx**(-k)
        r2 = 1 - np.sum((all_alpha - pred)**2) / np.sum((all_alpha - np.mean(all_alpha))**2)
        
        ax2.plot(idx_dense, alpha_fit, 'r-', lw=2.5, label=f'Master Curve\n(idx^{{-{k:.2f}}}, R²={r2:.3f})')
        
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    
    ax1.set_xlabel('Physical Distance $\Delta x$ [m]')
    ax1.set_ylabel(f'Relative Peak Energy ({alpha_key})')
    ax1.set_title('Divergence: Energy vs. Distance')
    
    ax2.set_xlabel('Fracture Index (Level)')
    ax2.set_title('Collapse: Energy vs. Fracture Index')
    
    ax1.legend(fontsize=9)
    ax2.legend(fontsize=9)
    
    fig.suptitle(f'Data Divergence vs. Collapse ({friction}, x1 = {x1}m)', fontsize=14, y=1.02)
    fig.tight_layout()
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, 'plots', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{friction}_collapse_vs_divergence_x1_{int(x1)}.png')
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
            p = plot_divergence_vs_collapse(rows, x1, fr)
            if p:
                print(f"Saved {p}")

if __name__ == '__main__':
    main()
