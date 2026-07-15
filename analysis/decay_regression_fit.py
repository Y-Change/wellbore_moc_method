# -*- coding: utf-8 -*-
"""研究 3：裂缝能量衰减回归与可视化。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import output_path, SERIES_DECAY_REGRESSION
from analysis.paper_plots import apply_paper_rc, save_figure, heatmap_with_contour

apply_paper_rc()

def load_decay_rows(csv_path: str) -> List[Dict]:
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

def fit_exponential_decay(rows: List[Dict], alpha_key: str) -> Dict:
    """对每个 (friction, x1, spacing, n_total) 拟合 alpha = exp(-lambda * delta_x)"""
    results = []
    
    groups = {}
    for r in rows:
        key = (r['friction_model'], r['x1'], r['spacing_m'], r['n_total'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
        
    for key, group in groups.items():
        fr, x1, sp, n = key
        group.sort(key=lambda x: x['frac_idx'])
        delta_x = np.array([r['delta_x'] for r in group])
        alpha = np.array([r[alpha_key] for r in group])
        
        mask = (alpha > 0) & (delta_x > 0)
        if not np.any(mask):
            lam = np.nan
        else:
            dx = delta_x[mask]
            y = -np.log(alpha[mask])
            lam, _, _, _ = np.linalg.lstsq(dx[:, np.newaxis], y, rcond=None)
            lam = float(lam[0])
            
        results.append({
            'friction_model': fr,
            'x1': x1,
            'spacing_m': sp,
            'n_total': n,
            'lambda': lam
        })
    return results

def plot_decay_envelopes(rows: List[Dict], alpha_key: str, save_path: str):
    """画出不同 n_total 下的典型衰减曲线（选取特定 x1 和 spacing）"""
    target_x1 = 4000.0
    target_sp = 50.0
    frictions = sorted(list(set(r['friction_model'] for r in rows)))
    
    fig, axes = plt.subplots(1, len(frictions), figsize=(4.5 * len(frictions) + 0.5, 4.0), sharey=True)
    if len(frictions) == 1:
        axes = [axes]
    
    for ax, fr in zip(axes, frictions):
        sub = [r for r in rows if r['friction_model'] == fr and r['x1'] == target_x1 and r['spacing_m'] == target_sp]
        if not sub:
            continue
            
        cmap = plt.cm.viridis
        n_totals = sorted(list(set(r['n_total'] for r in sub)))
        
        for k, n in enumerate(n_totals):
            pts = [r for r in sub if r['n_total'] == n]
            pts.sort(key=lambda x: x['frac_idx'])
            xs = [r['x_f'] for r in pts]
            ys = [r[alpha_key] for r in pts]
            
            color = cmap(k / max(1, len(n_totals)-1))
            ax.plot(xs, ys, marker='o', ls='none', color=color, label=f'n={n}')
            
            if len(xs) > 1:
                dx = np.array(xs) - xs[0]
                mask = np.array(ys) > 0
                if np.sum(mask) > 1:
                    lam, _, _, _ = np.linalg.lstsq(dx[mask, np.newaxis], -np.log(np.array(ys)[mask]), rcond=None)
                    xs_dense = np.linspace(xs[0], xs[-1], 50)
                    ys_fit = np.exp(-lam[0] * (xs_dense - xs[0]))
                    ax.plot(xs_dense, ys_fit, ls='--', color=color, alpha=0.5)
                    
        ax.set_title(f"{fr} (x1={target_x1}m, S={target_sp}m)")
        ax.set_xlabel('Depth [m]')
        if fr == 'steady':
            ax.set_ylabel(f'Relative Peak {alpha_key}')
        ax.set_yscale('log')
        ax.legend(fontsize=7)
        
    fig.tight_layout()
    save_figure(fig, save_path)

def plot_lambda_heatmap(fit_results: List[Dict], save_path: str):
    """画 lambda 随 x1 和 spacing 的热力图 (针对 n=5 的情况)"""
    target_n = 5
    frictions = sorted(list(set(r['friction_model'] for r in fit_results)))
    
    fig, axes = plt.subplots(1, len(frictions), figsize=(5.0 * len(frictions), 4.2))
    if len(frictions) == 1:
        axes = [axes]
    
    for ax, fr in zip(axes, frictions):
        sub = [r for r in fit_results if r['friction_model'] == fr and r['n_total'] == target_n]
        if not sub:
            continue
            
        x1_vals = sorted(list(set(r['x1'] for r in sub)))
        sp_vals = sorted(list(set(r['spacing_m'] for r in sub)))
        
        Z = np.full((len(sp_vals), len(x1_vals)), np.nan)
        for r in sub:
            xi = x1_vals.index(r['x1'])
            si = sp_vals.index(r['spacing_m'])
            Z[si, xi] = r['lambda']
            
        X, Y = np.meshgrid(x1_vals, sp_vals)
        heatmap_with_contour(ax, X, Y, Z, cbar_label='Decay Coeff $\lambda$', 
                             xlabel='First Frac Depth x1 [m]', ylabel='Spacing [m]', 
                             title=f'{fr} (n={target_n})')
                             
    fig.tight_layout()
    save_figure(fig, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default='alpha_2d', choices=['alpha_1d', 'alpha_2d'])
    args = parser.parse_args()
    
    csv_path = output_path(SERIES_DECAY_REGRESSION, None, 'decay_table.csv')
    if not os.path.isfile(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    rows = load_decay_rows(csv_path)
    print(f"Loaded {len(rows)} points.")
    
    fit_results = fit_exponential_decay(rows, args.alpha)
    
    json_path = output_path(SERIES_DECAY_REGRESSION, None, f'decay_lambda_{args.alpha}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(fit_results, f, indent=2, ensure_ascii=False)
        
    png_env = output_path(SERIES_DECAY_REGRESSION, None, f'decay_envelope_{args.alpha}.png')
    plot_decay_envelopes(rows, args.alpha, png_env)
    
    png_heat = output_path(SERIES_DECAY_REGRESSION, None, f'decay_lambda_heatmap_{args.alpha}.png')
    plot_lambda_heatmap(fit_results, png_heat)
    
    print(f"Done. Outputs in {os.path.dirname(png_env)}")

if __name__ == '__main__':
    main()
