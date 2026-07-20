# -*- coding: utf-8 -*-
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def load_first_frac_rows(csv_path: str):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r['frac_idx']) == 1:
                for k in ('x1', 'spacing_m', 'x_f', 'delta_x', 'P_1d', 'P_2d'):
                    r[k] = float(r[k])
                r['n_total'] = int(r['n_total'])
                rows.append(r)
    return rows

def plot_vs_x1(rows, friction, method='2d'):
    """6.1 Energy vs X1, lines are n_total, subplots are 5 selected spacing_m"""
    target_spacings = [10.0,30.0, 50.0, 70.0, 90.0]
    n_totals = sorted(list(set(r['n_total'] for r in rows)))
    p_key = f'P_{method}'
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharex=True, sharey=True)
    
    cmap = plt.cm.viridis
    
    for i, sp in enumerate(target_spacings):
        ax = axes[i]
        sub = [r for r in rows if r['spacing_m'] == sp]
        
        for j, n in enumerate(n_totals):
            pts = [r for r in sub if r['n_total'] == n]
            pts.sort(key=lambda x: x['x1'])
            if not pts:
                continue
            
            x_vals = [r['x1'] for r in pts]
            y_vals = [r[p_key] for r in pts]
            
            color = cmap(j / max(1, len(n_totals)-1))
            ax.plot(x_vals, y_vals, marker='o', ls='-', color=color, alpha=0.8, label=f'n = {n}')
            
        ax.set_title(f'Spacing $D$ = {int(sp)}m', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('First Frac Depth $x_1$ (m)')
        
    axes[0].set_ylabel(f'First Frac Absolute Response ({p_key})')
    axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    fig.suptitle(f'6.1 First Fracture Response vs. Depth ($x_1$) | {friction.capitalize()} | {method.upper()}', fontsize=14, y=1.05)
    fig.tight_layout()
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '06_first_frac_energy', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{friction}_1_vs_x1_{method}.png')
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path

def plot_vs_ntotal(rows, friction, method='2d'):
    """6.2 Energy vs n_total, lines are spacing, subplots are 5 selected X1"""
    target_x1 = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
    spacings = sorted(list(set(r['spacing_m'] for r in rows)))
    p_key = f'P_{method}'
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharex=True, sharey=True)
    
    cmap = plt.cm.plasma
    
    for i, x1 in enumerate(target_x1):
        ax = axes[i]
        sub = [r for r in rows if r['x1'] == x1]
        
        for j, sp in enumerate(spacings):
            pts = [r for r in sub if r['spacing_m'] == sp]
            pts.sort(key=lambda x: x['n_total'])
            if not pts:
                continue
            
            n_vals = [r['n_total'] for r in pts]
            y_vals = [r[p_key] for r in pts]
            
            color = cmap(j / max(1, len(spacings)-1))
            ax.plot(n_vals, y_vals, marker='s', ls='-', color=color, alpha=0.8, label=f'D = {int(sp)}m')
            
        ax.set_title(f'Depth $x_1$ = {int(x1)}m', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('Total Number of Fractures ($n$)')
        ax.set_xticks([2, 4, 6, 8])
        
    axes[0].set_ylabel(f'First Frac Absolute Response ({p_key})')
    axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=2)
    
    fig.suptitle(f'6.2 First Fracture Response vs. Total Fractures ($n$) | {friction.capitalize()} | {method.upper()}', fontsize=14, y=1.05)
    fig.tight_layout()
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '06_first_frac_energy', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{friction}_2_vs_ntotal_{method}.png')
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path

def plot_vs_spacing(rows, friction, method='2d'):
    """6.3 Energy vs Spacing, lines are n_total, subplots are 5 selected X1"""
    target_x1 = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
    n_totals = sorted(list(set(r['n_total'] for r in rows)))
    p_key = f'P_{method}'
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharex=True, sharey=True)
    
    cmap = plt.cm.cividis
    
    for i, x1 in enumerate(target_x1):
        ax = axes[i]
        sub = [r for r in rows if r['x1'] == x1]
        
        for j, n in enumerate(n_totals):
            pts = [r for r in sub if r['n_total'] == n]
            pts.sort(key=lambda x: x['spacing_m'])
            if not pts:
                continue
            
            sp_vals = [r['spacing_m'] for r in pts]
            y_vals = [r[p_key] for r in pts]
            
            color = cmap(j / max(1, len(n_totals)-1))
            ax.plot(sp_vals, y_vals, marker='^', ls='-', color=color, alpha=0.8, label=f'n = {n}')
            
        ax.set_title(f'Depth $x_1$ = {int(x1)}m', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('Fracture Spacing $D$ (m)')
        
    axes[0].set_ylabel(f'First Frac Absolute Response ({p_key})')
    axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    fig.suptitle(f'6.3 First Fracture Response vs. Spacing ($D$) | {friction.capitalize()} | {method.upper()}', fontsize=14, y=1.05)
    fig.tight_layout()
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '06_first_frac_energy', '')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{friction}_3_vs_spacing_{method}.png')
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_table.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)
        
    all_rows = load_first_frac_rows(csv_path)
    
    frictions = sorted(list(set(r['friction_model'] for r in all_rows)))
    methods = ['1d', '2d']
    
    for fr in frictions:
        fr_rows = [r for r in all_rows if r['friction_model'] == fr]
        for m in methods:
            p1 = plot_vs_x1(fr_rows, fr, method=m)
            p2 = plot_vs_ntotal(fr_rows, fr, method=m)
            p3 = plot_vs_spacing(fr_rows, fr, method=m)
            if p1: print(f"Saved {p1}")
            if p2: print(f"Saved {p2}")
            if p3: print(f"Saved {p3}")

if __name__ == '__main__':
    main()
