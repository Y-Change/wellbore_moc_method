# -*- coding: utf-8 -*-
"""Study 2 comprehensive figures — Nature style.

4 figures, each defending a unique scientific claim:
  a) effectiveness_panels  — wlen alone governs matching rate; hop is weak
  b) resolution_curves     — FWHM is wlen-invariant; spacing_error drops 15-30s
  c) snr_cost_panels       — SNR has an optimal wlen; cost ∝ 1/hop
  d) optimal_summary       — optimal wlen* shifts longer under Brunone

Outputs PNG (300 dpi raster preview) + SVG (editable vector, primary).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'paths.py')) and os.path.isfile(
        os.path.join(_d, 'wellbore_moc.py')
    ):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _d = os.path.dirname(_d)
    if _d == os.path.dirname(_d):
        raise RuntimeError('Cannot find wellbore_moc_method root')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import output_path, SERIES_CEPSTRUM_WLEN_HOP
from analysis.paper_plots import (
    apply_paper_rc, heatmap_with_contour, save_figure,
    FRICTION_COLORS, CASE_COLORS, CASE_MARKERS,
    PALETTE, add_panel_label, style_heatmap_ax,
)

apply_paper_rc(font_size=8, axes_linewidth=1.0)

CASES = ['dual', 'triple', 'quad']
FRICTIONS = ['steady', 'brunone']
CASE_LABELS = {'dual': '2 fractures', 'triple': '3 fractures', 'quad': '4 fractures'}
CASE_NFRACS = {'dual': 2, 'triple': 3, 'quad': 4}
WLEN_MIN = 13.8  # 4L/v


def load_all() -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for fr in FRICTIONS:
        out[fr] = {}
        for ck in CASES:
            p = output_path(f"{SERIES_CEPSTRUM_WLEN_HOP}/{fr}", ck, 'metrics.json')
            if os.path.isfile(p):
                with open(p, 'r', encoding='utf-8') as f:
                    out[fr][ck] = json.load(f)
    return out


def _to_grid(results: List[Dict], key: str
             ) -> Tuple[List[float], List[float], np.ndarray]:
    wlens = sorted({r['wlen_sec'] for r in results})
    hops = sorted({r['hop_ratio'] for r in results})
    Z = np.full((len(wlens), len(hops)), np.nan)
    for r in results:
        wi = wlens.index(r['wlen_sec'])
        hi = hops.index(r['hop_ratio'])
        v = r.get(key)
        if v is not None:
            Z[wi, hi] = float(v)
    return wlens, hops, Z


def _hop_avg(results: List[Dict], key: str
             ) -> Tuple[List[float], np.ndarray]:
    wlens = sorted({r['wlen_sec'] for r in results})
    hops = sorted({r['hop_ratio'] for r in results})
    Z = np.full((len(wlens), len(hops)), np.nan)
    for r in results:
        wi = wlens.index(r['wlen_sec'])
        hi = hops.index(r['hop_ratio'])
        v = r.get(key)
        if v is not None:
            Z[wi, hi] = float(v)
    with np.errstate(all='ignore'):
        return wlens, np.nanmean(Z, axis=1)


# ── Fig a: effectiveness 6-panel heatmap ──────────────────
def plot_effectiveness_panels(all_data: Dict, save_path: str) -> None:
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(len(FRICTIONS), len(CASES) + 1,
                          width_ratios=[1, 1, 1, 0.08],
                          wspace=0.35, hspace=0.35)
    axes = np.empty((len(FRICTIONS), len(CASES)), dtype=object)
    for i in range(len(FRICTIONS)):
        for j in range(len(CASES)):
            axes[i, j] = fig.add_subplot(gs[i, j])

    last_mesh = None
    for i, fr in enumerate(FRICTIONS):
        for j, ck in enumerate(CASES):
            ax = axes[i, j]
            cm = all_data.get(fr, {}).get(ck)
            if cm is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        fontsize=8, color=PALETTE['neutral_mid'])
                ax.set_axis_off()
                continue
            wlens, hops, Z = _to_grid(cm['results'], 'n_matched_ratio')
            W, H = np.meshgrid(np.array(wlens), np.array(hops), indexing='ij')
            # RdYlGn: red=0, yellow=0.5, green=1 — matches "higher better"
            last_mesh = ax.pcolormesh(W, H, Z, cmap='RdYlGn', vmin=0, vmax=1,
                                       shading='auto')
            if Z.shape[0] >= 2 and Z.shape[1] >= 2:
                levels = np.linspace(0, 1, 6)
                cs = ax.contour(W, H, Z, levels=levels, colors='white',
                                linewidths=0.5, alpha=0.5)
                ax.clabel(cs, inline=True, fontsize=5.5, fmt='%.1f')
            # mark optimal (first 1.0)
            if not np.all(np.isnan(Z)):
                idx = np.nanargmax(Z)
                wi, hi = np.unravel_index(idx, Z.shape)
                ax.scatter([wlens[wi]], [hops[hi]], marker='*', s=60,
                           c=PALETTE['neutral_black'],
                           edgecolors='white', linewidths=0.6, zorder=10)
            # wlen_min reference — only label in first panel to avoid clutter
            ax.axvline(WLEN_MIN, color=PALETTE['neutral_dark'], ls=':',
                       lw=0.7, alpha=0.7,
                       label=r'$w_{\rm len,min}$=13.8s' if i == 0 and j == 0 else None)
            n_fracs = CASE_NFRACS[ck]
            ax.set_xlabel(r'window length $w_{\rm len}$ [s]')
            ax.set_ylabel(r'hop ratio $\rho_{\rm hop}$')
            ax.set_title(f'{fr} | n = {n_fracs}', fontweight='bold', loc='left',
                         fontsize=8)
            style_heatmap_ax(ax)
            # panel letter — shift x further left to prevent overlap with y-axis labels
            add_panel_label(ax, chr(ord('a') + i * len(CASES) + j),
                            x=-0.24, y=1.04, fontsize=9)
    # shared colorbar — placed in a subgridspec of the 4th column to center and shrink vertically
    if last_mesh is not None:
        gs_cbar = gs[:, len(CASES)].subgridspec(3, 1, height_ratios=[0.15, 0.7, 0.15])
        cax = fig.add_subplot(gs_cbar[1, 0])
        cbar = fig.colorbar(last_mesh, cax=cax)
        cbar.set_label(r'match ratio $n_{\rm matched}/n_{\rm fracs}$',
                       fontsize=8)
        cbar.ax.tick_params(labelsize=7, length=0)
        cbar.outline.set_linewidth(0.5)  # type: ignore
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.suptitle('Effectiveness: wlen dominates, hop is weak',
                 fontsize=9, fontweight='bold', y=0.99)
    save_figure(fig, save_path, pad=2.0)


# ── Fig b: resolution curves (hop-avg, wlen sweep) ────────
def plot_resolution_curves(all_data: Dict, save_path: str) -> None:
    metrics = ['mean_error_m', 'spacing_error_m', 'fwhm_m', 'sidelobe_suppression']
    ylabels = [
        r'mean position error [m]',
        r'spacing error [m]',
        r'peak FWHM [m]',
        r'sidelobe suppression',
    ]
    log_metrics = {'sidelobe_suppression'}
    fig, axes = plt.subplots(len(metrics), len(FRICTIONS),
                             figsize=(7.2, 6.4),
                             squeeze=False)
    for i, (mkey, ylabel) in enumerate(zip(metrics, ylabels)):
        for j, fr in enumerate(FRICTIONS):
            ax = axes[i, j]
            for ck in CASES:
                cm = all_data.get(fr, {}).get(ck)
                if cm is None:
                    continue
                wlens, ys = _hop_avg(cm['results'], mkey)
                # only label case in first panel to avoid repeated legends
                ax.plot(wlens, ys, marker=CASE_MARKERS[ck], ms=3.5, lw=1.4,
                        color=CASE_COLORS[ck],
                        label=CASE_LABELS[ck] if i == 0 and j == 0 else None,
                        markeredgecolor='white', markeredgewidth=0.3)
            # wlen_min reference — only label once
            ax.axvline(WLEN_MIN, color=PALETTE['neutral_mid'], ls=':',
                       lw=0.7, alpha=0.7,
                       label=r'$w_{\rm len,min}$' if i == 0 and j == 0 else None)
            if mkey in log_metrics:
                ax.set_yscale('log')
            ax.set_xlabel(r'window length $w_{\rm len}$ [s]')
            ax.set_ylabel(ylabel)
            if i == 0 and j == 0:
                ax.legend(loc='best', fontsize=7, ncol=1, framealpha=0.9)
            ax.set_title(fr, fontweight='bold', loc='left', fontsize=8)
            # tighten y limits to data
            ys_all = [np.nan] + [
                _hop_avg(all_data.get(fr, {}).get(ck, {}).get('results', []),
                         mkey)[1]
                for ck in CASES if all_data.get(fr, {}).get(ck)
            ]
            all_y = np.concatenate([np.atleast_1d(y) for y in ys_all])
            all_y = all_y[np.isfinite(all_y)]
            if len(all_y) > 0:
                ymin, ymax = np.min(all_y), np.max(all_y)
                if mkey in log_metrics and ymin > 0:
                    ax.set_ylim(ymin * 0.5, ymax * 2.0)
                else:
                    margin = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
                    ax.set_ylim(ymin - margin, ymax + margin)
            # panel letter — use larger negative x offset to clear wide y-axis labels
            add_panel_label(ax, chr(ord('a') + i * len(FRICTIONS) + j),
                            x=-0.28, y=1.04, fontsize=9)
    fig.suptitle('Resolution metrics vs wlen (hop-averaged)',
                 fontsize=9, fontweight='bold', y=0.995)
    save_figure(fig, save_path)


# ── Fig c: SNR + cost (dual-case representative) ──────────
def plot_snr_cost_panels(all_data: Dict, save_path: str) -> None:
    fig = plt.figure(figsize=(7.6, 5.2))
    gs = fig.add_gridspec(2, len(FRICTIONS) + 1,
                          width_ratios=[1, 1, 0.08],
                          wspace=0.35, hspace=0.35)
    axes = np.empty((2, len(FRICTIONS)), dtype=object)
    for i in range(2):
        for j in range(len(FRICTIONS)):
            axes[i, j] = fig.add_subplot(gs[i, j])

    mesh_snr = None
    mesh_cost = None
    for j, fr in enumerate(FRICTIONS):
        cm = all_data.get(fr, {}).get('dual')
        # top: log10 SNR
        ax = axes[0, j]
        if cm is None:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                     fontsize=8, color=PALETTE['neutral_mid'])
            ax.set_axis_off()
        else:
            wlens, hops, Z = _to_grid(cm['results'], 'snr')
            W, H = np.meshgrid(np.array(wlens), np.array(hops), indexing='ij')
            Zlog = np.log10(Z + 1)
            mesh_snr = ax.pcolormesh(W, H, Zlog, cmap='viridis', shading='auto')
            if Z.shape[0] >= 2 and Z.shape[1] >= 2:
                levels = np.linspace(np.nanmin(Zlog), np.nanmax(Zlog), 7)
                cs = ax.contour(W, H, Zlog, levels=levels, colors='white',
                                linewidths=0.5, alpha=0.5)
                ax.clabel(cs, inline=True, fontsize=5.5, fmt='%.1f')
            ax.set_xlabel(r'window length $w_{\rm len}$ [s]')
            ax.set_ylabel(r'hop ratio $\rho_{\rm hop}$')
            ax.set_title(f'{fr} | SNR', fontweight='bold', loc='left',
                         fontsize=8)
            style_heatmap_ax(ax)
            # panel letter — shift x further left to prevent overlap with y-axis labels
            add_panel_label(ax, chr(ord('a') + j),
                            x=-0.24, y=1.04, fontsize=9)

        # bottom: elapsed_s
        ax = axes[1, j]
        if cm is None:
            ax.set_axis_off()
            continue
        wlens, hops, Z = _to_grid(cm['results'], 'elapsed_s')
        W, H = np.meshgrid(np.array(wlens), np.array(hops), indexing='ij')
        mesh_cost = ax.pcolormesh(W, H, Z, cmap='magma', shading='auto')
        if Z.shape[0] >= 2 and Z.shape[1] >= 2:
            levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 7)
            cs = ax.contour(W, H, Z, levels=levels, colors='white',
                            linewidths=0.5, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=5.5, fmt='%.2f')
        ax.set_xlabel(r'window length $w_{\rm len}$ [s]')
        ax.set_ylabel(r'hop ratio $\rho_{\rm hop}$')
        ax.set_title(f'{fr} | cost', fontweight='bold', loc='left',
                     fontsize=8)
        style_heatmap_ax(ax)
        # panel letter — shift x further left to prevent overlap with y-axis labels
        add_panel_label(ax, chr(ord('a') + len(FRICTIONS) + j),
                        x=-0.24, y=1.04, fontsize=9)
    # 每行共享 colorbar — placed in a subgridspec of the 3rd column to prevent overlap
    if mesh_snr is not None:
        gs_snr = gs[0, len(FRICTIONS)].subgridspec(3, 1, height_ratios=[0.1, 0.8, 0.1])
        cax_snr = fig.add_subplot(gs_snr[1, 0])
        cbar_snr = fig.colorbar(mesh_snr, cax=cax_snr)
        cbar_snr.set_label(r'$\log_{10}$ SNR', fontsize=8)
        cbar_snr.ax.tick_params(labelsize=7, length=0)
        cbar_snr.outline.set_linewidth(0.5)  # type: ignore
    if mesh_cost is not None:
        gs_cost = gs[1, len(FRICTIONS)].subgridspec(3, 1, height_ratios=[0.1, 0.8, 0.1])
        cax_cost = fig.add_subplot(gs_cost[1, 0])
        cbar_cost = fig.colorbar(mesh_cost, cax=cax_cost)
        cbar_cost.set_label('compute time [s]', fontsize=8)
        cbar_cost.ax.tick_params(labelsize=7, length=0)
        cbar_cost.outline.set_linewidth(0.5)  # type: ignore
    fig.suptitle('SNR optimum vs wlen; cost scales as 1/hop (dual-fracture representative)',
                 fontsize=9, fontweight='bold', y=0.995)
    save_figure(fig, save_path)


# ── Fig d: optimal summary ────────────────────────────────
def plot_optimal_summary(all_data: Dict, save_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    # (a) optimal (wlen*, hop*) scatter + recommendation bands
    for fr in FRICTIONS:
        for ck in CASES:
            cm = all_data.get(fr, {}).get(ck)
            if cm is None:
                continue
            wlens, hops, Z = _to_grid(cm['results'], 'n_matched_ratio')
            if np.all(np.isnan(Z)):
                continue
            idx = np.nanargmax(Z)
            wi, hi = np.unravel_index(idx, Z.shape)
            color = FRICTION_COLORS[fr]
            ax1.scatter([wlens[wi]], [hops[hi]], marker=CASE_MARKERS[ck],
                        s=70, c=color, edgecolors='white', linewidths=0.8,
                        label=f'{fr} / n={CASE_NFRACS[ck]}')
    # recommendation bands
    ax1.axhspan(0.1, 0.25, alpha=0.12, color=PALETTE['green_3'])
    ax1.axvspan(30, 50, alpha=0.08, color=PALETTE['blue_secondary'])
    ax1.text(40, 0.27, 'recommended hop [0.1, 0.25]',
             ha='center', va='bottom', fontsize=6.5,
             color=PALETTE['neutral_dark'])
    ax1.text(82, 0.04, 'wlen [30, 50] s',
             ha='right', va='bottom', fontsize=6.5,
             color=PALETTE['blue_main'])
    ax1.set_xlabel(r'optimal $w_{\rm len}^*$ [s]')
    ax1.set_ylabel(r'optimal $\rho_{\rm hop}^*$')
    ax1.set_title('Optimal parameters', fontweight='bold', loc='left', fontsize=8)
    ax1.set_xlim([10, 85])
    ax1.set_ylim([0, 0.55])
    # legend below the plot to avoid overlapping data points in upper right
    ax1.legend(fontsize=6.5, loc='upper left', ncol=3,
               columnspacing=1.0, handletextpad=0.4,
               bbox_to_anchor=(0.0, -0.22), framealpha=0.9)
    add_panel_label(ax1, 'a', x=-0.18, y=1.04, fontsize=9)

    # (b) optimal wlen* grouped bars by metric
    metrics_focus = ['n_matched_ratio', 'mean_error_m', 'snr', 'sidelobe_suppression']
    x_pos = np.arange(len(metrics_focus))
    bar_width = 0.12
    for i, fr in enumerate(FRICTIONS):
        for j, ck in enumerate(CASES):
            best_wlens = []
            for mkey in metrics_focus:
                cm = all_data.get(fr, {}).get(ck)
                if cm is None:
                    continue
                wlens, hops, Z = _to_grid(cm['results'], mkey)
                if np.all(np.isnan(Z)):
                    continue
                from numpy import nanargmax, nanargmin
                idx = nanargmax(Z) if mkey in ('n_matched_ratio', 'snr',
                                                'sidelobe_suppression') else nanargmin(Z)
                wi, _ = np.unravel_index(idx, Z.shape)
                best_wlens.append(wlens[wi])
            if not best_wlens:
                continue
            offset = (i * len(CASES) + j - 1.5) * bar_width
            base_color = FRICTION_COLORS[fr]
            rgb = [int(base_color.lstrip('#')[k:k+2], 16) / 255.0
                   for k in (0, 2, 4)]
            alphas = [0.4, 0.65, 1.0]  # dual/triple/quad
            color = (rgb[0], rgb[1], rgb[2], alphas[j])
            ax2.bar(x_pos + offset, best_wlens, bar_width,
                    color=color, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['match\nratio', 'mean\nerror', 'SNR', 'sidelobe\nsupp.'],
                        fontsize=7)
    ax2.set_ylabel(r'optimal $w_{\rm len}^*$ [s]')
    ax2.set_title('Optimal wlen* by metric', fontweight='bold', loc='left',
                  fontsize=8)
    ax2.axhline(WLEN_MIN, color=PALETTE['neutral_dark'], ls=':', lw=0.8,
                alpha=0.8)
    ax2.text(3.4, WLEN_MIN + 1, f'$w_{{\\rm len,min}}$={WLEN_MIN:.1f}s',
             fontsize=6.5, color=PALETTE['neutral_dark'], va='bottom', ha='right')
    # custom legend (friction colors)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=FRICTION_COLORS['steady'], edgecolor='white',
              label='steady'),
        Patch(facecolor=FRICTION_COLORS['brunone'], edgecolor='white',
              label='brunone'),
    ]
    ax2.legend(handles=legend_elems, fontsize=7, loc='upper left')
    add_panel_label(ax2, 'b', x=-0.14, y=1.04, fontsize=9)

    fig.suptitle('Optimal parameters: Brunone shifts wlen* longer',
                 fontsize=9, fontweight='bold', y=1.02)
    save_figure(fig, save_path)


# ── main ──────────────────────────────────────────────────
def main() -> None:
    all_data = load_all()
    available = sum(1 for fr in FRICTIONS for ck in CASES
                    if all_data.get(fr, {}).get(ck))
    print(f'loaded {available}/{len(FRICTIONS)*len(CASES)} (friction, case)')

    out_dir = os.path.dirname(
        output_path(SERIES_CEPSTRUM_WLEN_HOP, None, '_placeholder')
    )
    figures = [
        ('study2_effectiveness_panels.png', plot_effectiveness_panels),
        ('study2_resolution_curves.png', plot_resolution_curves),
        ('study2_snr_cost_panels.png', plot_snr_cost_panels),
        ('study2_optimal_summary.png', plot_optimal_summary),
    ]
    for fname, fn in figures:
        path = os.path.join(out_dir, fname)
        fn(all_data, path)
        print(f'  -> {path}  (+SVG)')


if __name__ == '__main__':
    main()
