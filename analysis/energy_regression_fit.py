# -*- coding: utf-8 -*-
"""研究1：能量-参数回归拟合与可视化。

读 energy_table.csv → log-log 幂律 lstsq 拟合 + degree=2 多项式对照 →
输出 heatmap（E vs spacing × n_fracs，按 friction 分面）、回归曲线、R²/系数表。

主模型：
    log E = β0 + β_Cf·log Cf + β_kleak·log kleak
                + β_s·log spacing + β_n·log n_fracs + β_fr·I_brunone

运行
----
    python analysis/energy_regression_fit.py
    python analysis/energy_regression_fit.py --energy E_2d_norm
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _METHOD_ROOT not in sys.path:
    sys.path.insert(0, _METHOD_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import output_path, SERIES_ENERGY_REGRESSION
from analysis.paper_plots import apply_paper_rc, heatmap_with_contour, dual_axis_curve, save_figure
from validation.config import SPACING_PRESETS_M

apply_paper_rc()

CASE_ORDER = ['single', 'dual', 'triple', 'quad', 'quint']
CASE_NFRACS = {'single': 1, 'dual': 2, 'triple': 3, 'quad': 4, 'quint': 5}

ENERGY_CHOICES = [
    'E_1d', 'E_2d', 'E_fft', 'E_1d_peaks', 'E_2d_peaks',
    'E_1d_norm', 'E_2d_norm',
]


def load_rows(csv_path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ('spacing_m', 'Cf', 'kleak', 'sim_seconds',
                      'E_1d', 'E_2d', 'E_fft', 'E_1d_peaks', 'E_2d_peaks',
                      'E_1d_norm', 'E_2d_norm', 'fmax_eff',
                      'snr_1d', 'snr_2d'):
                if k in r and r[k] not in (None, ''):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        r[k] = np.nan
            for k in ('n_fracs', 'n_matched_1d', 'n_matched_2d'):
                if k in r and r[k] not in (None, ''):
                    try:
                        r[k] = int(float(r[k]))
                    except ValueError:
                        r[k] = 0
            rows.append(r)
    return rows


def _safe_log(x: float) -> float:
    if x is None or x <= 0 or not np.isfinite(x):
        return np.nan
    return float(np.log(x))


# ── 回归 ─────────────────────────────────────────────────
def fit_loglog_powerlaw(rows: List[Dict], energy_key: str) -> Dict:
    """log E = β0 + β_Cf·log Cf + β_kleak·log kleak + β_s·log spacing
              + β_n·log n_fracs + β_fr·I_brunone

    主网格行（Cf/kleak 不变）β_Cf/β_kleak 不可分；用主+次网格合并拟合。
    """
    X, y = [], []
    for r in rows:
        E = r.get(energy_key)
        if E is None or E <= 0 or not np.isfinite(E):
            continue
        Cf = r.get('Cf', 0.0)
        kleak = r.get('kleak', 0.0)
        spacing = r.get('spacing_m', 0.0)
        n = r.get('n_fracs', 0)
        if Cf <= 0 or kleak <= 0 or spacing <= 0 or n <= 0:
            continue
        is_brunone = 1.0 if r.get('friction_model') == 'brunone' else 0.0
        X.append([
            _safe_log(Cf), _safe_log(kleak), _safe_log(spacing),
            _safe_log(float(n)), is_brunone,
        ])
        y.append(_safe_log(E))
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    if len(X) < 6:
        return {'ok': False, 'n_samples': len(X), 'reason': 'samples < 6'}
    # 加截距列
    A = np.hstack([np.ones((len(X), 1)), X])
    coef, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    names = ['intercept', 'log_Cf', 'log_kleak', 'log_spacing', 'log_n_fracs', 'I_brunone']
    return {
        'ok': True,
        'n_samples': len(X),
        'coef': dict(zip(names, coef.tolist())),
        'r2': r2,
        'y_pred': y_pred.tolist(),
        'y_true': y.tolist(),
    }


def fit_poly2_main_grid(rows: List[Dict], energy_key: str, friction: str) -> Dict:
    """主网格（Cf/kleak 固定）上 E vs (spacing, n_fracs) 二次多项式对照。"""
    sub = [r for r in rows if r.get('friction_model') == (friction)
           and r.get('Cf', 0) > 0 and r.get('kleak', 0) > 0
           and r.get(energy_key, 0) > 0]
    if len(sub) < 6:
        return {'ok': False, 'n_samples': len(sub)}
    X, y = [], []
    for r in sub:
        s = float(r['spacing_m'])
        n = float(r['n_fracs'])
        E = float(r[energy_key])
        X.append([s, n, s * s, n * n, s * n])
        y.append(np.log(E))
    X = np.array(X); y = np.array(y)
    A = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'ok': True, 'n_samples': len(sub), 'r2': r2,
            'coef': coef.tolist()}


# ── Bootstrap 95% CI ─────────────────────────────────────
def bootstrap_coefs(
    rows: List[Dict], energy_key: str, n_boot: int = 200, seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    base = fit_loglog_powerlaw(rows, energy_key)
    if not base.get('ok'):
        return base
    n = base['n_samples']
    y_true = np.array(base['y_true'])
    y_pred = np.array(base['y_pred'])
    resid = y_true - y_pred
    X_list = []
    for r in rows:
        E = r.get(energy_key)
        if E is None or E <= 0 or not np.isfinite(E):
            continue
        Cf = r.get('Cf', 0.0); kleak = r.get('kleak', 0.0)
        spacing = r.get('spacing_m', 0.0); n_f = r.get('n_fracs', 0)
        if Cf <= 0 or kleak <= 0 or spacing <= 0 or n_f <= 0:
            continue
        is_b = 1.0 if r.get('friction_model') == 'brunone' else 0.0
        X_list.append([1.0, _safe_log(Cf), _safe_log(kleak), _safe_log(spacing),
                       _safe_log(float(n_f)), is_b])
    X = np.array(X_list)
    coefs_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = X[idx]
        yb = y_pred[idx] + resid[idx]
        try:
            c, *_ = np.linalg.lstsq(xb, yb, rcond=None)
            coefs_boot.append(c)
        except Exception:
            pass
    if not coefs_boot:
        return base
    cb = np.array(coefs_boot)
    names = ['intercept', 'log_Cf', 'log_kleak', 'log_spacing', 'log_n_fracs', 'I_brunone']
    ci = {}
    for i, nm in enumerate(names):
        ci[nm] = {
            'p2.5': float(np.percentile(cb[:, i], 2.5)),
            'p97.5': float(np.percentile(cb[:, i], 97.5)),
        }
    return {**base, 'coef_ci95': ci, 'n_boot': len(coefs_boot)}


# ── 可视化 ───────────────────────────────────────────────
def plot_energy_heatmap(
    rows: List[Dict], energy_key: str, save_path: str,
) -> None:
    """E vs (spacing × n_fracs)，按 friction 分面。"""
    frictions = sorted({r.get('friction_model', 'steady') for r in rows})
    spacings = sorted({float(r['spacing_m']) for r in rows if 'spacing_m' in r})
    nfracs = sorted({int(r['n_fracs']) for r in rows if 'n_fracs' in r})
    fig, axes = plt.subplots(1, len(frictions), figsize=(5 * len(frictions), 4.2),
                             squeeze=False)
    for j, fr in enumerate(frictions):
        ax = axes[0, j]
        Z = np.full((len(nfracs), len(spacings)), np.nan)
        for r in rows:
            if r.get('friction_model') != fr:
                continue
            try:
                si = spacings.index(float(r['spacing_m']))
                ni = nfracs.index(int(r['n_fracs']))
            except ValueError:
                continue
            v = r.get(energy_key)
            if v is not None and np.isfinite(v):
                Z[ni, si] = float(v)
        S, N = np.meshgrid(np.array(spacings), np.array(nfracs))
        heatmap_with_contour(
            ax, S, N, Z,
            cbar_label=energy_key,
            xlabel='spacing [m]', ylabel='n_fractures',
            title=f'{fr} | {energy_key}',
        )
    fig.tight_layout()
    save_figure(fig, save_path)


def plot_regression_curves(
    rows: List[Dict], energy_key: str, fit: Dict, save_path: str,
) -> None:
    """y_true vs y_pred 散点 + 1:1 线；并按 spacing 着色。"""
    if not fit.get('ok'):
        return
    y_true = np.array(fit['y_true'])
    y_pred = np.array(fit['y_pred'])
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(y_pred, y_true, s=24, alpha=0.7, edgecolors='none', c='C0')
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.0, label='1:1')
    ax.set_xlabel(r'$\log E$ predicted')
    ax.set_ylabel(r'$\log E$ true')
    ax.set_title(f"Log-log powerlaw | R²={fit['r2']:.3f} | n={fit['n_samples']}")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_path)


def plot_cfkleak_curves(rows: List[Dict], energy_key: str, save_path: str) -> None:
    """次网格：E vs Cf（按 kleak 着色）与 E vs kleak（按 Cf 着色），双联。区分 steady 与 brunone。"""
    sub = [r for r in rows if r.get('Cf', 0) > 0 and r.get('kleak', 0) > 0
           and r.get(energy_key, 0) > 0
           and r.get('case') in (None, 'triple')]
    # 区分主网格（Cf/kleak 固定）与次网格：次网格 Cf/kleak 多值
    cf_vals = sorted({round(r['Cf'], 8) for r in sub})
    kl_vals = sorted({round(r['kleak'], 8) for r in sub})
    frictions = sorted({r.get('friction_model', 'steady') for r in sub})
    if len(cf_vals) < 2 and len(kl_vals) < 2:
        return  # 次网格未跑
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    cmap = plt.cm.viridis
    
    # E vs Cf，按 kleak 着色
    for fr in frictions:
        ls = '-' if fr == 'steady' else '--'
        marker = 'o' if fr == 'steady' else 'x'
        for k, kl in enumerate(kl_vals):
            pts = [(r['Cf'], r[energy_key]) for r in sub 
                   if abs(r['kleak'] - kl) < 1e-12 and r.get('friction_model', 'steady') == fr]
            pts.sort()
            if not pts:
                continue
            xs, ys = zip(*pts)
            # 为了避免图例重复，只有 steady 时添加 label
            label = f'kleak={kl:.1e}' if fr == 'steady' else None
            ax1.plot(xs, ys, marker=marker, ls=ls, color=cmap(k / max(1, len(kl_vals) - 1)),
                     label=label)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('Cf [m²]'); ax1.set_ylabel(f'{energy_key}')
    ax1.set_title(f'{energy_key} vs Cf (solid: steady, dashed: brunone)'); ax1.legend(fontsize=7)
    
    # E vs kleak，按 Cf 着色
    for fr in frictions:
        ls = '-' if fr == 'steady' else '--'
        marker = 's' if fr == 'steady' else '^'
        for k, cf in enumerate(cf_vals):
            pts = [(r['kleak'], r[energy_key]) for r in sub 
                   if abs(r['Cf'] - cf) < 1e-12 and r.get('friction_model', 'steady') == fr]
            pts.sort()
            if not pts:
                continue
            xs, ys = zip(*pts)
            label = f'Cf={cf:.1e}' if fr == 'steady' else None
            ax2.plot(xs, ys, marker=marker, ls=ls, color=cmap(k / max(1, len(cf_vals) - 1)),
                     label=label)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('kleak [m²/s/√m]'); ax2.set_ylabel(f'{energy_key}')
    ax2.set_title(f'{energy_key} vs kleak (solid: steady, dashed: brunone)'); ax2.legend(fontsize=7)
    
    fig.tight_layout()
    save_figure(fig, save_path)


# ── JSON 汇总 ────────────────────────────────────────────
def write_fit_json(fit: Dict, poly_fits: Dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    out = {
        'powerlaw_loglog': {k: v for k, v in fit.items()
                            if k not in ('y_pred', 'y_true')},
        'poly2_main_grid': poly_fits,
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)


# ── CLI ──────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='研究1：能量-参数回归拟合')
    p.add_argument('--csv', default=None,
                   help='energy_table.csv 路径；缺省=output/analysis/energy_regression/energy_table.csv')
    p.add_argument('--energy', default='E_2d_norm', choices=ENERGY_CHOICES)
    p.add_argument('--n-boot', type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv or output_path(SERIES_ENERGY_REGRESSION, None, 'energy_table.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'CSV 不存在：{csv_path}；先跑 energy_regression.py')
    rows = load_rows(csv_path)
    print(f'读入 {len(rows)} 行')

    fit = bootstrap_coefs(rows, args.energy, n_boot=args.n_boot)
    if fit.get('ok'):
        print(f"\nLog-log powerlaw (n={fit['n_samples']}, R^2={fit['r2']:.3f}):")
        for nm, c in fit['coef'].items():
            ci = fit.get('coef_ci95', {}).get(nm, {})
            lo = ci.get('p2.5', np.nan); hi = ci.get('p97.5', np.nan)
            print(f"  {nm:12s} beta={c:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]")
    else:
        print(f"fit failed: {fit.get('reason')}")

    poly_fits = {}
    for fr in ('steady', 'brunone'):
        poly_fits[fr] = fit_poly2_main_grid(rows, args.energy, fr)
        if poly_fits[fr].get('ok'):
            print(f"  poly2 [{fr}] R^2={poly_fits[fr]['r2']:.3f}")

    # 绘图
    png_heat = output_path(SERIES_ENERGY_REGRESSION, None,
                           f'heatmap_{args.energy}.png')
    png_reg = output_path(SERIES_ENERGY_REGRESSION, None,
                          f'regression_{args.energy}.png')
    png_cfkl = output_path(SERIES_ENERGY_REGRESSION, None,
                           f'cfkleak_curves_{args.energy}.png')
    plot_energy_heatmap(rows, args.energy, png_heat)
    if fit.get('ok'):
        plot_regression_curves(rows, args.energy, fit, png_reg)
    plot_cfkleak_curves(rows, args.energy, png_cfkl)
    print(f"\n图件:\n  {png_heat}\n  {png_reg}\n  {png_cfkl}")

    json_path = output_path(SERIES_ENERGY_REGRESSION, None,
                            f'fit_{args.energy}.json')
    write_fit_json(fit, poly_fits, json_path)
    print(f"  {json_path}")


if __name__ == '__main__':
    main()
