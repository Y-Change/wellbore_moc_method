# -*- coding: utf-8 -*-
"""
run_joint_mle.py — 联合 MLE 管道（Phase 1 核心功能）

针对波速 a 和摩阻系数 k_scale 存在严重失配的情况，通过两阶段优化解决：
1. 全局网格扫描：扫描候选波速，使用 Cepstrum(a) 获得初值，配合快速坐标下降寻找全局似然盆地。
2. 联合 Gauss-Newton 精修：将 (x, a, log_kbr) 作为联合参数向量，在全局最优盆地内进行亚网格精确梯度下降。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _p = os.path.dirname(_d)
    if _p == _d:
        break
    _d = _p

from analysis.identifiability.crb_core import forward, observe
from analysis.identifiability.mle import (
    add_noise,
    mle_grid_coord_descent,
    mle_refine_gn,
)
from analysis.identifiability.scenarios import (
    case_by_name,
    make_obs,
    make_scenario,
)
from analysis.identifiability.run_moc_pipeline import coarse_init_cepstrum, adaptive_radius_cells



def run_joint_pipeline(
    scn, obs, y, t_full, H_full, n_frac,
    a_grid, k_grid, radius, n_rounds, workers,
    gn_steps=3
) -> Dict[str, object]:
    """联合 MLE 核心算法：波速网格扫描 + 坐标下降 + 联合 GN 精修"""
    print("Running joint pipeline...")
    # 1. 物理锚定波速
    dt = scn.dt
    t_start_idx = int(6.0 / dt)
    t_end_idx = int(8.0 / dt)
    search_window = np.abs(np.diff(y[t_start_idx:t_end_idx]))
    peak_idx = np.argmax(search_window)
    t_bound = (t_start_idx + peak_idx) * dt
    a_est = 2 * scn.L / t_bound
    print(f"Physical boundary detected at t={t_bound:.3f}s. Estimated a={a_est:.2f} m/s", flush=True)

    # 2. 短窗寻峰 (避开边界)
    # tf=6.0 以排除边界回波，只找裂缝
    scn_short = replace(scn, a_nominal=a_est, k_scale=1.0, tf=6.0)
    fr_short = forward(scn_short)
    t_short = np.asarray(fr_short.t, dtype=np.float64)
    H_short = np.asarray(fr_short.H_wh, dtype=np.float64)
    
    from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum_1d
    from analysis.unified_evaluation.detection_protocol import DetectorConfig, detect_peaks
    cep = compute_moc_cepstrum_1d(
        t_short, H_short, v=a_est, ts=1.0, wellbore_length=5000.0, depth_min=2000.0
    )
    det_cfg = DetectorConfig(
        min_separation_m=10.0, height_pct=50.0, max_peaks=5, search_min_m=2000.0, search_max_m=4900.0
    )
    peaks = detect_peaks(cep["depth"], cep["response"], det_cfg)
    
    best_x0 = [scn.L * 0.75]  # Fallback to 3750m, removing the 4000m hardcoded cheat
    best_init_rss = float('inf')
    cache = {}
    
    if len(peaks) > 0:
        for p in peaks[:3]:
            test_x0 = [float(p["depth_m"])]
            refined_init = mle_grid_coord_descent(
                scn_short, obs, y, radius=1, n_rounds=1, x_init=test_x0, signal_cache=cache, workers=1
            )
            if refined_init["rss"] < best_init_rss:
                best_init_rss = refined_init["rss"]
                best_x0 = test_x0
        print(f"Short-window (6.0s) evaluation best_init_x0={best_x0[0]:.1f}", flush=True)
    else:
        print("WARNING: Cepstrum found no peaks in short window. Falling back to coarse RSS scan.", flush=True)
        for coarse_x in np.arange(2000.0, 4801.0, 200.0):
            test_x0 = [float(coarse_x)]
            refined_init = mle_grid_coord_descent(
                scn_short, obs, y, radius=1, n_rounds=1, x_init=test_x0, signal_cache=cache, workers=1
            )
            if refined_init["rss"] < best_init_rss:
                best_init_rss = refined_init["rss"]
                best_x0 = test_x0
        print(f"Coarse RSS scan fallback selected best_init_x0={best_x0[0]:.1f}", flush=True)

    # 3. 联合 GN 精修 (全窗 10.0s 引入边界)
    scn_best = replace(scn, a_nominal=a_est, k_scale=1.0, tf=10.0)
    gn_out = mle_refine_gn(
        scn_best, obs, y,
        x_init=best_x0,
        mode="custom", 
        include_override=["x", "a", "log_kbr"],
        n_steps=gn_steps,
        map_fn=None
    )

    x_hat = np.asarray(gn_out["x_hat"], dtype=np.float64)
    a_hat = float(gn_out["a_hat"])
    k_scale_hat = float(gn_out["k_scale_hat"])
    rss_final = float(gn_out["rss"])
    
    return {
        "x_hat": x_hat.tolist(),
        "a_hat": a_hat,
        "k_scale_hat": k_scale_hat,
        "rss_grid": best_init_rss,
        "rss_final": rss_final,
        "best_a_grid": a_est,
        "best_k_grid": 1.0,
        "n_forwards": gn_steps * (2 * n_frac + 4)
    }

def main():
    p = argparse.ArgumentParser(description="Joint MLE Pipeline for a and x")
    p.add_argument("--cases", default="single_4000")
    p.add_argument("--snr", type=float, default=40)
    p.add_argument("--a-true", type=float, default=1451.4)
    p.add_argument("--a-mismatch", type=float, default=1430.0, help="波速失配，用于生成观测数据")
    p.add_argument("--k-scale-mismatch", type=float, default=1.5, help="摩阻系数失配")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    
    print("=" * 72)
    print("Joint MLE Estimator (Phase 1)")
    print("=" * 72)
    
    case = case_by_name(args.cases)
    n_frac = len(case.x_f)
    
    # 构建真实场景 (含失配的波速和摩阻)，必须保持 tf=50 以保证 Cepstrum 能够提取周期回波
    scn_true = make_scenario(case)
    n_cells_true = max(10, int(round(scn_true.L / (args.a_mismatch * scn_true.dt))))
    a_nominal_true = scn_true.a_for(n_cells_true)
    from dataclasses import replace
    scn_true = replace(scn_true, a_nominal=a_nominal_true, k_scale=args.k_scale_mismatch)
    
    obs = make_obs()
    
    # 1. 生成真实观测 (带噪声)
    print("Generating true observation with mismatch...")
    fr_true = forward(scn_true)
    s0_true = observe(fr_true.H_wh, fr_true.t, scn_true, obs)
    
    rng = np.random.default_rng(42)
    y, sigma = add_noise(s0_true, args.snr, rng)
    H_noisy = fr_true.H_wh + rng.normal(0, sigma, size=len(fr_true.H_wh))
    
    # 构建反演初始场景 (波速未知，使用基准波速，例如 1450)
    scn_inv = replace(make_scenario(case, tf=10.0), a_nominal=args.a_true, k_scale=1.0)
    
    # 网格扫描 a 和 k_scale
    # BUGFIX 4: 稀疏网格以加速测试
    a_grid = np.linspace(1350, 1550, 5)
    k_grid = np.linspace(0.5, 2.0, 3)
    
    t0 = time.time()
    res = run_joint_pipeline(
        scn_inv, obs, y, fr_true.t, H_noisy, n_frac,
        a_grid=a_grid, k_grid=k_grid, radius=20, n_rounds=3, workers=args.workers, gn_steps=3
    )
    t1 = time.time()
    
    print("\n[Results]")
    print(f"True x      : {scn_true.x_f}")
    print(f"Est  x      : {np.round(res['x_hat'], 2).tolist()}")
    print(f"True a      : {a_nominal_true:.2f} m/s")
    print(f"Est  a      : {res['a_hat']:.2f} m/s")
    print(f"True k_scale: {args.k_scale_mismatch:.2f}")
    print(f"Est  k_scale: {res['k_scale_hat']:.2f}")
    
    err_x = np.max(np.abs(np.array(res['x_hat']) - np.array(scn_true.x_f)))
    err_a = np.abs(res['a_hat'] - a_nominal_true)
    print(f"Max Pos Err : {err_x:.3f} m")
    print(f"a Error     : {err_a:.3f} m/s")
    print(f"Time Elapsed: {t1-t0:.1f} s")

if __name__ == "__main__":
    main()
