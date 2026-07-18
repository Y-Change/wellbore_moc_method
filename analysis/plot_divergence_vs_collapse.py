# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

from paths import output_path, SERIES_DECAY_REGRESSION
from analysis.paper_plots import apply_paper_rc, save_figure

apply_paper_rc()

def model_pidx(idx, k):
    """
    P_idx model: alpha = idx^(-k)
    This strictly satisfies alpha=1 at idx=1.
    """
    # handle arrays safely
    idx = np.asarray(idx, dtype=float)
    return idx ** (-k)

def model_stretched_exp(x, b, beta):
    """Stretched exponential function."""
    x = np.asarray(x, dtype=float)
    return np.exp(-((b * x)**beta))

def log_model_stretched_exp(x, b, beta):
    x = np.asarray(x, dtype=float)
    return -((b * x)**beta)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

def plot_divergence_collapse(df_group, friction, x1, out_dir):
    # We filter to use only runs with n_total == 8 (or max available) to have consistent curves
    n_max = df_group['n_total'].max()
    if pd.isna(n_max):
        return
    df_n = df_group[df_group['n_total'] == n_max]
    
    spacings = sorted(df_n['spacing_m'].unique())
    if len(spacings) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    ax1, ax2 = axes

    colors = plt.cm.plasma(np.linspace(0, 0.9, len(spacings)))
    
    all_dx = []
    all_idx = []
    all_a2d = []

    for i, sp in enumerate(spacings):
        df_sp = df_n[df_n['spacing_m'] == sp].sort_values('frac_idx')
        if df_sp.empty:
            continue
            
        dx = df_sp['delta_x'].values
        idx = df_sp['frac_idx'].values
        a2d = df_sp['alpha_2d'].values
        
        all_dx.extend(dx)
        all_idx.extend(idx)
        all_a2d.extend(a2d)

        ax1.plot(dx, a2d, marker='o', ls='none', color=colors[i], label=f'S = {sp:.1f}m')
        
        # Fit stretched exponential (in log space)
        if len(dx) > 2:
            try:
                popt, _ = curve_fit(
                    log_model_stretched_exp, 
                    dx, 
                    np.log(a2d), 
                    p0=[0.01, 0.5], 
                    bounds=([1e-6, 0.1], [1.0, 5.0])
                )
                dx_fit = np.linspace(0, max(dx), 100)
                a2d_fit = model_stretched_exp(dx_fit, *popt)
                ax1.plot(dx_fit, a2d_fit, ls='--', color=colors[i])
            except Exception:
                pass

        ax2.plot(idx, a2d, marker='o', ls='none', color=colors[i], label=f'S = {sp:.1f}m')

    all_dx = np.array(all_dx)
    all_idx = np.array(all_idx)
    all_a2d = np.array(all_a2d)

    # ----------------------------------------------------
    # Divergence Envelope (Convex Hull for Divergence plot)
    # ----------------------------------------------------
    import matplotlib.patches as patches
    if len(all_dx) > 3:
        points_log = np.column_stack((all_dx, np.log10(all_a2d)))
        try:
            hull = ConvexHull(points_log)
            # hull.vertices gives the indices of points forming the hull in CCW order
            hull_pts_linear = np.column_stack((all_dx[hull.vertices], all_a2d[hull.vertices]))
            
            poly = patches.Polygon(hull_pts_linear, facecolor='cornflowerblue', alpha=0.2, label='Divergence Envelope', zorder=0)
            ax1.add_patch(poly)
        except:
            pass

    # ----------------------------------------------------
    # Collapse Envelope
    # ----------------------------------------------------
    unique_idx = np.unique(all_idx)
    if len(unique_idx) > 1:
        min_col = [np.min(all_a2d[all_idx == x]) for x in unique_idx]
        max_col = [np.max(all_a2d[all_idx == x]) for x in unique_idx]
        ax2.fill_between(unique_idx, min_col, max_col, color='red', alpha=0.15, label='Collapse Envelope')

    # ----------------------------------------------------
    # Master Curve (Pidx Scheme)
    # ----------------------------------------------------
    # fit idx^(-k)
    mask = all_a2d > 0
    f_idx = all_idx[mask]
    f_a2d = all_a2d[mask]
    
    # We can fit on log-log for robustness, or directly with curve_fit
    try:
        popt, _ = curve_fit(model_pidx, f_idx, f_a2d, p0=[1.0])
        k = popt[0]
        r2 = r_squared(f_a2d, model_pidx(f_idx, k))
        
        idx_smooth = np.linspace(1, n_max, 100)
        a2d_smooth = model_pidx(idx_smooth, k)
        ax2.plot(idx_smooth, a2d_smooth, 'r-', lw=2.5, label=f'Master Curve\n(idx^{{-{k:.2f}}}, R²={r2:.3f})')
    except:
        pass

    # ----------------------------------------------------
    # Formatting
    # ----------------------------------------------------
    ax1.set_yscale('log')
    ax1.set_xlabel('Physical Distance $\\Delta x$ [m]')
    ax1.set_ylabel('Relative Peak Energy (alpha_2d)')
    ax1.set_title('Divergence: Energy vs. Distance')
    
    import matplotlib.lines as mlines
    handles, labels = ax1.get_legend_handles_labels()
    fit_line = mlines.Line2D([], [], color='gray', ls='--', label='Stretched Exp Fit')
    if 'Stretched Exp Fit' not in labels:
        handles.append(fit_line)
        labels.append('Stretched Exp Fit')
    ax1.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=8, frameon=False)
    
    # adjust the second axis slightly to make room for legend of ax1
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    ax2.set_xlabel('Fracture Index (Level)')
    ax2.set_title('Collapse: Energy vs. Fracture Index')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=8, frameon=False)
    
    box2 = ax2.get_position()
    ax2.set_position([box2.x0 + box2.width * 0.15, box2.y0, box2.width * 0.85, box2.height])
    
    fig.suptitle(f'Data Divergence vs. Collapse ({friction}, x1 = {x1}m)', fontsize=14, y=0.98)
    # plt.tight_layout() is skipped because we manual adjusted positions
    
    filename = f'{friction}_collapse_vs_divergence_x1_{int(x1)}.png'
    save_path = os.path.join(out_dir, filename)
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

def main():
    csv_path = output_path(SERIES_DECAY_REGRESSION, '03_extracted_peaks_csv', 'decay_table.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    out_dir = output_path(SERIES_DECAY_REGRESSION, '04_collapse_and_scaling_pidx', '')
    os.makedirs(out_dir, exist_ok=True)
    
    frictions = df['friction_model'].unique()
    for fr in frictions:
        df_fr = df[df['friction_model'] == fr]
        x1_list = df_fr['x1'].unique()
        for x1 in x1_list:
            df_group = df_fr[df_fr['x1'] == x1]
            plot_divergence_collapse(df_group, fr, x1, out_dir)

if __name__ == '__main__':
    main()
