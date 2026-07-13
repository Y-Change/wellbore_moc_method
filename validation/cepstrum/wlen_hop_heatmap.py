# -*- coding: utf-8 -*-
"""研究2：(wlen, hop) 网格扫描结果聚合 — heatmap/contour + 最优 (wlen*, hop*) 表。

读各 (friction, case) 的 metrics.json，对每个指标生成 wlen × hop heatmap，
叠 contour，并标记最优点；输出跨 case 聚合的"最优 (wlen*, hop*)"表 JSON+CSV。

运行
----
    python validation/cepstrum/wlen_hop_heatmap.py
    python validation/cepstrum/wlen_hop_heatmap.py --friction steady --cases dual,triple
"""
from __future__ import annotations

import argparse
import csv
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

from paths import output_path, SERIES_CEPSTRUM_WLEN_HOP, CASE_DUAL, CASE_TRIPLE, CASE_QUAD
from analysis.paper_plots import apply_paper_rc, heatmap_with_contour, save_figure

apply_paper_rc()

DEFAULT_CASES = [CASE_DUAL, CASE_TRIPLE, CASE_QUAD]

# 评估指标定义： (key, label, mode)
# mode='higher_better'：越大越好；'lower_better'：越小越好
METRICS = [
    ('n_matched_ratio', '匹配率 n_matched/n_fracs', 'higher_better'),
    ('mean_error_m',     '平均位置误差 [m]',         'lower_better'),
    ('max_error_m',      '最大位置误差 [m]',         'lower_better'),
    ('snr',              'SNR (peak/median)',        'higher_better'),
    ('spacing_error_m',  '峰间距误差 [m]',           'lower_better'),
    ('fwhm_m',           '峰宽 FWHM [m]',            'lower_better'),
    ('sidelobe_suppression', '旁瓣抑制比',           'higher_better'),
]


def load_case_metrics(friction: str, case: str) -> Optional[Dict]:
    path = output_path(f"{SERIES_CEPSTRUM_WLEN_HOP}/{friction}", case, 'metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _to_grid(results: List[Dict], key: str) -> Tuple[List[float], List[float], np.ndarray]:
    """results → (wlen_unique, hop_unique, Z[wlen_idx, hop_idx])。"""
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


def _best_point(Z: np.ndarray, wlens: List[float], hops: List[float],
                mode: str) -> Optional[Tuple[float, float, float]]:
    """返回 (best_wlen, best_hop, best_val)；空矩阵返回 None。"""
    if np.all(np.isnan(Z)):
        return None
    if mode == 'higher_better':
        idx = np.nanargmax(Z)
    else:
        idx = np.nanargmin(Z)
    wi, hi = np.unravel_index(idx, Z.shape)
    return float(wlens[wi]), float(hops[hi]), float(Z[wi, hi])


def plot_metric_heatmaps(
    case_metrics: Dict, friction: str, case: str, save_dir: str,
) -> List[str]:
    """对单 (friction, case) 生成所有指标的 heatmap PNG，返回路径列表。"""
    results = case_metrics['results']
    paths = []
    for key, label, mode in METRICS:
        wlens, hops, Z = _to_grid(results, key)
        if np.all(np.isnan(Z)):
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        W, H = np.meshgrid(np.array(wlens), np.array(hops), indexing='ij')
        best = _best_point(Z, wlens, hops, mode)
        heatmap_with_contour(
            ax, W, H, Z,
            cbar_label=label,
            xlabel='wlen [s]', ylabel='hop_ratio',
            title=f'{friction} | {case} | {label}',
            mark_best=best[:2] if best else None,
        )
        out = os.path.join(save_dir, f'heatmap_{key}.png')
        save_figure(fig, out)
        paths.append(out)
    return paths


def aggregate_best_table(
    all_data: Dict[str, Dict[str, Dict]],
) -> List[Dict]:
    """all_data[friction][case] = case_metrics dict。返回每 (friction, case, metric) 一行。"""
    rows: List[Dict] = []
    for fr, by_case in all_data.items():
        for case, cm in by_case.items():
            if cm is None:
                continue
            results = cm['results']
            for key, label, mode in METRICS:
                wlens, hops, Z = _to_grid(results, key)
                best = _best_point(Z, wlens, hops, mode)
                if best is None:
                    continue
                rows.append({
                    'friction': fr,
                    'case': case,
                    'metric': key,
                    'mode': mode,
                    'best_wlen_sec': best[0],
                    'best_hop_ratio': best[1],
                    'best_value': best[2],
                })
    return rows


def write_best_table(rows: List[Dict], csv_path: str, json_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['friction', 'case', 'metric', 'mode',
                                          'best_wlen_sec', 'best_hop_ratio', 'best_value'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def plot_cross_case_summary(
    all_data: Dict[str, Dict[str, Dict]], save_path: str,
) -> None:
    """跨 case 聚合：对每 (friction, metric) 画 1 张子图，叠加各 case 的
    best (wlen, hop) 点。聚焦 n_matched_ratio 与 mean_error_m 两指标。"""
    frictions = sorted(all_data.keys())
    focus_metrics = ['n_matched_ratio', 'mean_error_m']
    fig, axes = plt.subplots(len(focus_metrics), len(frictions),
                             figsize=(5 * len(frictions), 4 * len(focus_metrics)),
                             squeeze=False)
    for i, mkey in enumerate(focus_metrics):
        for j, fr in enumerate(frictions):
            ax = axes[i, j]
            for case, cm in all_data[fr].items():
                if cm is None:
                    continue
                wlens, hops, Z = _to_grid(cm['results'], mkey)
                if np.all(np.isnan(Z)):
                    continue
                mode = next(m for k, _, m in METRICS if k == mkey)
                best = _best_point(Z, wlens, hops, mode)
                if best:
                    ax.scatter([best[0]], [best[1]], s=80, label=f'{case}')
            ax.set_xlabel('best wlen [s]')
            ax.set_ylabel('best hop_ratio')
            ax.set_title(f'{fr} | {mkey}')
            ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, save_path)


# ── CLI ──────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='研究2：(wlen, hop) 热图聚合')
    p.add_argument('--friction', choices=['steady', 'brunone', 'both'], default='both')
    p.add_argument('--cases', default='all',
                   help='dual/triple/quad 逗号分隔；all=DEFAULT_CASES')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frictions = ['steady', 'brunone'] if args.friction == 'both' else [args.friction]
    case_keys = (DEFAULT_CASES if args.cases == 'all'
                 else [c.strip() for c in args.cases.split(',')])

    all_data: Dict[str, Dict[str, Dict]] = {fr: {} for fr in frictions}
    for fr in frictions:
        for ck in case_keys:
            cm = load_case_metrics(fr, ck)
            if cm is None:
                print(f"[skip] 无数据：{fr}/{ck}")
                continue
            all_data[fr][ck] = cm
            save_dir = os.path.dirname(
                output_path(f"{SERIES_CEPSTRUM_WLEN_HOP}/{fr}", ck, '_placeholder')
            )
            pngs = plot_metric_heatmaps(cm, fr, ck, save_dir)
            for p in pngs:
                print(f"  → {p}")

    rows = aggregate_best_table(all_data)
    if not rows:
        print("无可用数据，退出")
        return
    csv_path = output_path(SERIES_CEPSTRUM_WLEN_HOP, None, 'best_wlen_hop_table.csv')
    json_path = output_path(SERIES_CEPSTRUM_WLEN_HOP, None, 'best_wlen_hop_table.json')
    write_best_table(rows, csv_path, json_path)
    print(f"\n最优表:\n  {csv_path}\n  {json_path}")

    summary_png = output_path(SERIES_CEPSTRUM_WLEN_HOP, None, 'cross_case_summary.png')
    plot_cross_case_summary(all_data, summary_png)
    print(f"  {summary_png}")


if __name__ == '__main__':
    main()
