# -*- coding: utf-8 -*-
"""
批量基准测试：
跑/读 MOC 仿真，计算差信号，运行 L1/BSD 算法并与倒谱结果对比。
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

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
from analysis.sparse_deconv.l1_deconv import L1Deconvolver
from analysis.sparse_deconv.evaluation import evaluate_sparse_deconv

from moc_simulate.paths import output_path, SERIES_LEAKOFF
from moc_simulate.leakoff_multi import resolve_cases, load_timeseries_csv
from moc_simulate.config import SIM_CONFIG

def run_l1_benchmark(friction: str = 'steady',
                     spacings: tuple = (5, 10, 20, 50, 100),
                     cases_to_run: tuple = ('dual', 'triple', 'quad', 'quint')):
    """
    针对特定的 friction 模式，跑基准测试
    """
    print(f"[{friction}] 生成无裂缝参考波形...")
    ref_data = generate_reference_signal(friction=friction)
    t_ref = ref_data['t']
    H_ref = ref_data['H_wh']
    fs = ref_data['fs']
    a_adj = ref_data['a_adj']
    L = ref_data['cfg'].wellbore_length
    ts = SIM_CONFIG['ts']
    
    print(f"[{friction}] 提取系统子波...")
    h = extract_wavelet(H_ref, fs, a_adj, L, ts, t_ref)
    
    deconvolver = L1Deconvolver(h, fs, a_adj)
    
    results = {}
    
    for spacing in spacings:
        friction_key = f"{friction}_D{spacing}"
        cases = resolve_cases(friction_key)
        
        results[spacing] = {}
        
        for case_key in cases_to_run:
            if case_key not in cases:
                continue
                
            x_f_true = cases[case_key]['x_f_list']
            n_fracs = len(x_f_true)
            print(f"\n--- 测试 {friction_key} / {case_key} (缝深: {x_f_true}) ---")
            
            # 读取已有的 moc_timeseries.csv
            csv_path = output_path(f"{SERIES_LEAKOFF}/{friction_key}", case_key, 'moc_timeseries.csv')
            if not os.path.isfile(csv_path):
                print(f"找不到 CSV: {csv_path}，跳过")
                continue
                
            frac_data = load_timeseries_csv(csv_path)
            t_frac = frac_data['t']
            H_frac = frac_data['H_wh']
            
            diff_data = compute_difference_signal(t_frac, H_frac, t_ref, H_ref, fs, ts)
            delta_y = diff_data['delta_y']
            
            print(f"运行 L1 反卷积...")
            l1_res = deconvolver.solve(delta_y, n_fracs=n_fracs)
            
            # 评估
            eval_res = evaluate_sparse_deconv(l1_res['r'], l1_res['depth_axis'], x_f_true)
            
            # 打印匹配结果
            print(f"  λ used: {l1_res['lambda_used']:.2e}")
            print(f"  匹配率: {eval_res['n_matched']}/{n_fracs}")
            print(f"  平均误差: {eval_res['mean_error_m']} m")
            
            for m in eval_res['matches']:
                if m['matched']:
                    print(f"    F{m['frac_id']}: {m['peak_depth_m']:.1f} m (Δ={m['error_m']:.1f}m)")
                else:
                    print(f"    F{m['frac_id']}: x (未匹配)")
                    
            print(f"运行 BSD 盲反卷积...")
            from analysis.sparse_deconv.blind_sparse_deconv import BlindSparseDeconvolver
            bsd = BlindSparseDeconvolver(fs, a_adj, L)
            bsd_res = bsd.solve(delta_y, h, n_fracs=n_fracs)
            bsd_eval = evaluate_sparse_deconv(bsd_res['r'], bsd_res['depth_axis'], x_f_true)
            print(f"  外层迭代: {bsd_res['n_outer_iter']}")
            print(f"  匹配率: {bsd_eval['n_matched']}/{n_fracs}")
            print(f"  平均误差: {bsd_eval['mean_error_m']} m")
            
            for m in bsd_eval['matches']:
                if m['matched']:
                    print(f"    F{m['frac_id']}: {m['peak_depth_m']:.1f} m (Δ={m['error_m']:.1f}m)")
                else:
                    print(f"    F{m['frac_id']}: x (未匹配)")
                    
            results[spacing][case_key] = {
                'l1': eval_res,
                'bsd': bsd_eval
            }
            
    # 输出 JSON 汇总
    out_path = output_path('analysis/sparse_deconv', friction, 'l1_benchmark.json')
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n基准测试结果已保存到: {out_path}")


if __name__ == '__main__':
    run_l1_benchmark(friction='steady', spacings=(20, 50), cases_to_run=('dual', 'triple', 'quad', 'quint'))
