# -*- coding: utf-8 -*-
"""顶层报告入口：调 report_render 把两项研究的结果合成 output/RESEARCH_REPORT.md。

要求：
- 研究1 已跑 energy_regression.py + energy_regression_fit.py
- 研究2 已跑 wlen_hop_sweep.py + wlen_hop_heatmap.py

运行
----
    python analysis/research_report.py
    python analysis/research_report.py --energy E_2d_norm
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import os
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'paths.py')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find wellbore_moc_method root')
    _d = _parent

from paths import OUTPUT_DIR, SERIES_ENERGY_REGRESSION, SERIES_CEPSTRUM_WLEN_HOP
from analysis.report_render import (
    render_energy_section, render_wlen_hop_section, render_full_report,
)

ENERGY_KEYS = ['E_1d', 'E_2d', 'E_1d_norm', 'E_2d_norm']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='综合研究报告生成')
    p.add_argument('--energy', default='E_2d_norm',
                   help='研究1主绘图能量列名；需与 fit 脚本一致')
    p.add_argument('--output', default=None,
                   help='输出 md 路径；缺省=output/RESEARCH_REPORT.md')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 研究1 文件（4 个能量指标都生成 fit JSON）
    fit_json_paths: Dict[str, str] = {}
    for ek in ENERGY_KEYS:
        fit_json_paths[ek] = os.path.join(OUTPUT_DIR, SERIES_ENERGY_REGRESSION,
                                          f'fit_{ek}.json')
    fit_json = fit_json_paths[args.energy]
    csv_path = os.path.join(OUTPUT_DIR, SERIES_ENERGY_REGRESSION, 'energy_table.csv')
    heat_png = os.path.join(OUTPUT_DIR, SERIES_ENERGY_REGRESSION,
                            f'heatmap_{args.energy}.png')
    reg_png = os.path.join(OUTPUT_DIR, SERIES_ENERGY_REGRESSION,
                           f'regression_{args.energy}.png')
    cfkl_png = os.path.join(OUTPUT_DIR, SERIES_ENERGY_REGRESSION,
                            f'cfkleak_curves_{args.energy}.png')

    # 研究2 文件
    best_csv = os.path.join(OUTPUT_DIR, SERIES_CEPSTRUM_WLEN_HOP,
                            'best_wlen_hop_table.csv')
    best_json = os.path.join(OUTPUT_DIR, SERIES_CEPSTRUM_WLEN_HOP,
                             'best_wlen_hop_table.json')
    summary_png = os.path.join(OUTPUT_DIR, SERIES_CEPSTRUM_WLEN_HOP,
                               'cross_case_summary.png')
    case_metrics_dir = os.path.join(OUTPUT_DIR, SERIES_CEPSTRUM_WLEN_HOP)
    # 研究2 综合可视化（4 张跨指标图）
    study2_pngs = {
        'effectiveness': os.path.join(OUTPUT_DIR, SERIES_CEPSTRUM_WLEN_HOP,
                                      'study2_effectiveness_panels.png'),
        'resolution': os.path.join(OUTPUT_DIR, SERIES_CEPSTRUM_WLEN_HOP,
                                   'study2_resolution_curves.png'),
        'snr_cost': os.path.join(OUTPUT_DIR, SERIES_CEPSTRUM_WLEN_HOP,
                                 'study2_snr_cost_panels.png'),
        'optimal': os.path.join(OUTPUT_DIR, SERIES_CEPSTRUM_WLEN_HOP,
                                'study2_optimal_summary.png'),
    }

    energy_md = render_energy_section(
        fit_json, csv_path, heat_png, reg_png, cfkl_png, args.energy,
        fit_json_paths=fit_json_paths,
    )
    wlen_hop_md = render_wlen_hop_section(
        best_csv, best_json, summary_png, case_metrics_dir,
        study2_effectiveness_png=study2_pngs['effectiveness'],
        study2_resolution_png=study2_pngs['resolution'],
        study2_snr_cost_png=study2_pngs['snr_cost'],
        study2_optimal_png=study2_pngs['optimal'],
    )

    out_md = args.output or os.path.join(OUTPUT_DIR, 'RESEARCH_REPORT.md')
    render_full_report(energy_md, wlen_hop_md, out_md)
    print(f'报告已生成: {out_md}')


if __name__ == '__main__':
    main()
