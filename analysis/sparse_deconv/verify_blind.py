# -*- coding: utf-8 -*-
"""
验证 Level 3：BSD/L1 在【不喂真实缝数】前提下的实际能力。

动机
----
现有 l1_deconv.py / blind_sparse_deconv.py 的 solve() 都通过
`find_lambda_for_n_peaks(n_target=n_fracs)` 二分 λ，把真实缝数当先验喂进去。
这使得"检出峰数=真实缝数"成为被约束的结果，无法证明方法本身的分辨能力。

本脚本剥离该先验，做两件事：
  1. λ 全景扫描：对数网格扫 λ，记录 (n_detected, recall, precision, F1) 随 λ 的变化，
     得到 oracle-λ（事后最优）作为能力上界。
  2. 盲选 λ：用 BIC 模型定阶 与 L-curve 拐点 两种【不看缝数】的准则自动选 λ，
     评估其 recall/precision/误差，与 oracle 及 count-based（作弊）对照。

同时对 BSD（交替最小化，估计子波 h）用盲 λ 跑 AM 主循环，看 kernel 精化是否有增益。

输出
----
  output/analysis/sparse_deconv/verify_blind/{friction}/
     - blind_verify.json         汇总指标
     - lambda_sweep_{case}.png    每个 case 的 F1/峰数-vs-λ 全景
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

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

from analysis.sparse_deconv.reference_signal import (
    generate_reference_signal, extract_wavelet, compute_difference_signal
)
from analysis.sparse_deconv.solvers import fista_solve
from analysis.sparse_deconv.blind_sparse_deconv import BlindSparseDeconvolver
from analysis.sparse_deconv.evaluation import evaluate_sparse_deconv, get_match_tol_m

from moc_simulate.paths import output_path, SERIES_LEAKOFF
from moc_simulate.leakoff_multi import resolve_cases, load_timeseries_csv
from moc_simulate.config import SIM_CONFIG

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ── 卷积模型残差（与 fista 完全一致的循环卷积算子）──────────────
def _model_rss(dy: np.ndarray, dh: np.ndarray, r: np.ndarray) -> float:
    """RSS = ‖dy - (dh * r)‖²，用与 FISTA 相同的 FFT 循环卷积重构。"""
    N = len(dy)
    h_pad = np.zeros(N)
    h_pad[:len(dh)] = dh
    yhat = np.real(ifft(fft(h_pad) * fft(r)))
    return float(np.sum((yhat - dy) ** 2))


def _n_effective_peaks(r: np.ndarray) -> int:
    """r 的显著峰数（相对 1% 门限），genuinely blind：不依赖真实缝数。"""
    r_abs = np.abs(r)
    rmax = float(np.max(r_abs)) if len(r_abs) else 0.0
    if rmax <= 0:
        return 0
    from scipy.signal import find_peaks
    thr = max(1e-6, rmax * 1e-2)
    peaks, _ = find_peaks(r_abs, height=thr)
    if len(peaks) == 0:
        peaks = np.where(r_abs > thr)[0]
    return int(len(peaks))


# ── λ 全景扫描 ────────────────────────────────────────────
def sweep_lambda(dy: np.ndarray, dh: np.ndarray, depth_axis: np.ndarray,
                 x_f_true, n_grid: int = 45,
                 lambda_range=(1e-5, 5e1)) -> dict:
    """扫描 λ，记录每个 λ 下的稀疏解指标与模型选择准则量。"""
    lams = np.logspace(np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_grid)

    tol_repo = get_match_tol_m(x_f_true)           # 仓库默认容差（D20→80m，偏松）
    spacing = float(np.min(np.diff(sorted(x_f_true)))) if len(x_f_true) > 1 else 300.0
    tol_strict = max(5.0, spacing * 0.5)           # 严格容差 = 半个缝距

    rows = []
    for lam in lams:
        r = fista_solve(dy, dh, lam=lam, max_iter=1500)
        n_det = _n_effective_peaks(r)
        rss = _model_rss(dy, dh, r)
        l1 = float(np.sum(np.abs(r)))

        ev_repo = evaluate_sparse_deconv(r, depth_axis, x_f_true, match_tol_m=tol_repo)
        ev_strict = evaluate_sparse_deconv(r, depth_axis, x_f_true, match_tol_m=tol_strict)

        N = len(dy)
        bic = N * np.log(rss / N + 1e-30) + max(n_det, 1) * np.log(N)

        rows.append({
            'lambda': float(lam),
            'n_detected': int(n_det),
            'rss': rss, 'l1': l1, 'bic': float(bic),
            'recall_repo': ev_repo['recall'], 'precision_repo': ev_repo['precision'],
            'f1_repo': ev_repo['f1'], 'mean_err_repo': ev_repo['mean_error_m'],
            'recall_strict': ev_strict['recall'], 'precision_strict': ev_strict['precision'],
            'f1_strict': ev_strict['f1'], 'mean_err_strict': ev_strict['mean_error_m'],
        })

    return {'lams': lams, 'rows': rows, 'tol_repo': tol_repo, 'tol_strict': tol_strict}


def _lcurve_corner_idx(rss: np.ndarray, l1: np.ndarray) -> int:
    """L-curve 最大曲率拐点（对数坐标下的三点曲率近似）。"""
    x = np.log10(rss + 1e-30)
    y = np.log10(l1 + 1e-30)
    # 归一化到 [0,1] 消除量纲
    x = (x - x.min()) / (np.ptp(x) + 1e-30)
    y = (y - y.min()) / (np.ptp(y) + 1e-30)
    kappa = np.zeros(len(x))
    for i in range(1, len(x) - 1):
        dx1, dy1 = x[i] - x[i - 1], y[i] - y[i - 1]
        dx2, dy2 = x[i + 1] - x[i], y[i + 1] - y[i]
        cross = dx1 * dy2 - dx2 * dy1
        denom = (np.hypot(dx1, dy1) * np.hypot(dx2, dy2) * np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]))
        kappa[i] = abs(cross) / (denom + 1e-30)
    return int(np.argmax(kappa))


def analyze_case(friction: str, spacing: int, case_key: str,
                 ref_cache: dict, out_dir: str) -> dict:
    friction_key = f"{friction}_D{spacing}"
    cases = resolve_cases(friction_key)
    if case_key not in cases:
        return {}
    x_f_true = cases[case_key]['x_f_list']
    n_true = len(x_f_true)

    csv_path = output_path(f"{SERIES_LEAKOFF}/{friction_key}", case_key, 'moc_timeseries.csv')
    if not os.path.isfile(csv_path):
        print(f"  [skip] 找不到 {csv_path}")
        return {}

    fs = ref_cache['fs']; a_adj = ref_cache['a_adj']; ts = ref_cache['ts']
    t_ref = ref_cache['t_ref']; H_ref = ref_cache['H_ref']; h_wav = ref_cache['h']

    frac = load_timeseries_csv(csv_path)
    diff = compute_difference_signal(frac['t'], frac['H_wh'], t_ref, H_ref, fs, ts)
    delta_y = diff['delta_y']

    # 导数域（与 l1_deconv 一致）
    pulse_len = len(h_wav) - 10
    y_trunc = delta_y[:pulse_len]
    h_trunc = h_wav[:pulse_len]
    dy = np.diff(y_trunc, prepend=0)
    dh = np.diff(h_trunc, prepend=0)
    depth_axis = np.arange(len(dy)) / fs * a_adj / 2.0

    print(f"\n=== {friction_key}/{case_key}  真实缝数={n_true}  缝深={x_f_true} ===")
    sw = sweep_lambda(dy, dh, depth_axis, x_f_true)
    rows = sw['rows']
    lams = sw['lams']

    f1_repo = np.array([r['f1_repo'] for r in rows])
    f1_strict = np.array([r['f1_strict'] for r in rows])
    n_det = np.array([r['n_detected'] for r in rows])
    rss = np.array([r['rss'] for r in rows])
    l1 = np.array([r['l1'] for r in rows])
    bic = np.array([r['bic'] for r in rows])

    # oracle（事后最优，能力上界）
    i_oracle_repo = int(np.argmax(f1_repo))
    i_oracle_strict = int(np.argmax(f1_strict))
    # 盲选
    i_bic = int(np.argmin(bic))
    i_lc = _lcurve_corner_idx(rss, l1)

    def pack(i):
        return {
            'lambda': rows[i]['lambda'], 'n_detected': rows[i]['n_detected'],
            'recall_repo': rows[i]['recall_repo'], 'precision_repo': rows[i]['precision_repo'],
            'f1_repo': rows[i]['f1_repo'], 'mean_err_repo': rows[i]['mean_err_repo'],
            'recall_strict': rows[i]['recall_strict'], 'precision_strict': rows[i]['precision_strict'],
            'f1_strict': rows[i]['f1_strict'], 'mean_err_strict': rows[i]['mean_err_strict'],
        }

    # count-based（作弊参照）：直接取 n_detected 最接近 n_true 的 λ
    i_count = int(np.argmin([abs(r['n_detected'] - n_true) for r in rows]))

    result = {
        'friction': friction_key, 'case': case_key, 'n_true': n_true,
        'x_f_true': list(map(float, x_f_true)),
        'tol_repo_m': sw['tol_repo'], 'tol_strict_m': sw['tol_strict'],
        'oracle_repo': pack(i_oracle_repo),
        'oracle_strict': pack(i_oracle_strict),
        'blind_bic': pack(i_bic),
        'blind_lcurve': pack(i_lc),
        'count_based_cheat': pack(i_count),
    }

    # 控制台摘要
    def line(tag, i):
        r = rows[i]
        print(f"  {tag:14s} λ={r['lambda']:.2e}  峰数={r['n_detected']:2d}/{n_true}  "
              f"F1(松)={r['f1_repo']:.2f} F1(严)={r['f1_strict']:.2f}  "
              f"recall(严)={r['recall_strict']:.2f} prec(严)={r['precision_strict']:.2f}")
    line('oracle(松)', i_oracle_repo)
    line('oracle(严)', i_oracle_strict)
    line('BIC盲选', i_bic)
    line('Lcurve盲选', i_lc)
    line('count作弊', i_count)

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.semilogx(lams, f1_repo, 'o-', color='steelblue', label='F1 (松容差 %.0fm)' % sw['tol_repo'], ms=3)
    ax.semilogx(lams, f1_strict, 's-', color='crimson', label='F1 (严容差 %.0fm)' % sw['tol_strict'], ms=3)
    for i, c, lab in [(i_bic, 'green', 'BIC盲选'), (i_lc, 'purple', 'Lcurve盲选'),
                      (i_oracle_strict, 'orange', 'oracle(严)')]:
        ax.axvline(lams[i], color=c, ls='--', alpha=0.7, label=f'{lab} λ={lams[i]:.1e}')
    ax.set_ylabel('F1'); ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'{friction_key}/{case_key}  n_true={n_true}  —  F1 vs λ（不喂缝数）')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, ls='--', alpha=0.5)

    ax = axes[1]
    ax.semilogx(lams, n_det, 'o-', color='black', ms=3, label='检出峰数')
    ax.axhline(n_true, color='red', ls='--', label=f'真实缝数={n_true}')
    ax.set_ylabel('峰数'); ax.set_xlabel('λ')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    save = os.path.join(out_dir, f'lambda_sweep_{case_key}.png')
    plt.savefig(save, dpi=150); plt.close()
    print(f"  图已存: {save}")

    # ── BSD 盲版（AM 精化子波 + BIC 盲选 λ）─────────────────
    result['bsd_blind'] = _bsd_blind(delta_y, h_wav, fs, a_adj,
                                     ref_cache['L'], depth_axis, x_f_true,
                                     sw['tol_repo'], sw['tol_strict'])
    rb = result['bsd_blind']
    print(f"  {'BSD盲(BIC)':14s} λ={rb['lambda']:.2e}  峰数={rb['n_detected']:2d}/{n_true}  "
          f"F1(松)={rb['f1_repo']:.2f} F1(严)={rb['f1_strict']:.2f}  外层={rb['n_outer']}")
    return result


def _bsd_blind(delta_y, h_wav, fs, a_adj, L, depth_axis, x_f_true,
               tol_repo, tol_strict, max_outer=8) -> dict:
    """BSD 交替最小化，但每轮用 BIC 盲选 λ（不喂缝数）。"""
    bsd = BlindSparseDeconvolver(fs, a_adj, L)
    pulse_len = len(h_wav) - 10
    y_trunc = delta_y[:pulse_len]
    h = h_wav[:pulse_len].copy()
    h = h / (np.linalg.norm(h) + 1e-12)
    dy = np.diff(y_trunc, prepend=0)

    lam_grid = np.logspace(-5, np.log10(50), 30)
    r = np.zeros_like(y_trunc)
    lam_used = lam_grid[len(lam_grid) // 2]
    k = 0
    for k in range(max_outer):
        dh = np.diff(h, prepend=0)
        # BIC 盲选 λ
        best_bic, best_r, best_lam = np.inf, None, lam_used
        N = len(dy)
        for lam in lam_grid:
            rr = fista_solve(dy, dh, lam=lam, max_iter=400)
            n_det = _n_effective_peaks(rr)
            rss = _model_rss(dy, dh, rr)
            bic = N * np.log(rss / N + 1e-30) + max(n_det, 1) * np.log(N)
            if bic < best_bic:
                best_bic, best_r, best_lam = bic, rr, lam
        r_new = best_r; lam_used = best_lam
        h_new = bsd._solve_kernel_step(y_trunc, r_new, lambda_h=1e-1)
        nrm = np.linalg.norm(h_new)
        if nrm > 0:
            h_new = h_new / nrm
        r_diff = np.linalg.norm(r_new - r) / (np.linalg.norm(r_new) + 1e-12)
        h_diff = np.linalg.norm(h_new[:len(h)] - h) / (np.linalg.norm(h_new) + 1e-12)
        r, h = r_new, h_new[:len(h)]
        if k > 0 and r_diff < 1e-3 and h_diff < 1e-3:
            break

    n_det = _n_effective_peaks(r)
    ev_repo = evaluate_sparse_deconv(r, depth_axis, x_f_true, match_tol_m=tol_repo)
    ev_strict = evaluate_sparse_deconv(r, depth_axis, x_f_true, match_tol_m=tol_strict)
    return {
        'lambda': float(lam_used), 'n_detected': int(n_det), 'n_outer': k + 1,
        'recall_repo': ev_repo['recall'], 'precision_repo': ev_repo['precision'],
        'f1_repo': ev_repo['f1'], 'mean_err_repo': ev_repo['mean_error_m'],
        'recall_strict': ev_strict['recall'], 'precision_strict': ev_strict['precision'],
        'f1_strict': ev_strict['f1'], 'mean_err_strict': ev_strict['mean_error_m'],
    }


def run(friction: str = 'steady', spacing: int = 20,
        cases=('dual', 'triple', 'quad', 'quint', 'hex', 'oct')):
    print(f"[{friction}] 生成参考波形并提取子波...")
    ref = generate_reference_signal(friction=friction)
    L = ref['cfg'].wellbore_length
    ts = SIM_CONFIG['ts']
    h = extract_wavelet(ref['H_wh'], ref['fs'], ref['a_adj'], L, ts, ref['t'])
    ref_cache = {
        'fs': ref['fs'], 'a_adj': ref['a_adj'], 'ts': ts,
        't_ref': ref['t'], 'H_ref': ref['H_wh'], 'h': h, 'L': L,
    }

    out_dir = output_path('analysis/sparse_deconv/verify_blind', friction, '')
    os.makedirs(out_dir, exist_ok=True)

    all_res = []
    for ck in cases:
        res = analyze_case(friction, spacing, ck, ref_cache, out_dir)
        if res:
            all_res.append(res)

    out_json = os.path.join(out_dir, 'blind_verify.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已存: {out_json}")

    # 汇总表
    print("\n================ 汇总（严容差 = 半缝距）================")
    print(f"{'case':6s} {'n':>2s} | {'oracle_F1':>9s} {'BIC_F1':>7s} {'BIC_n':>6s} "
          f"{'Lcv_F1':>7s} {'BSD_F1':>7s} {'BSD_n':>6s}")
    for r in all_res:
        print(f"{r['case']:6s} {r['n_true']:>2d} | "
              f"{r['oracle_strict']['f1_strict']:>9.2f} "
              f"{r['blind_bic']['f1_strict']:>7.2f} "
              f"{r['blind_bic']['n_detected']:>4d}/{r['n_true']:<1d} "
              f"{r['blind_lcurve']['f1_strict']:>7.2f} "
              f"{r['bsd_blind']['f1_strict']:>7.2f} "
              f"{r['bsd_blind']['n_detected']:>4d}/{r['n_true']:<1d}")


if __name__ == '__main__':
    run(friction='steady', spacing=20,
        cases=('dual', 'triple', 'quad', 'quint', 'hex', 'oct'))
