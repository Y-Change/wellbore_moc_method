# -*- coding: utf-8 -*-
"""
run_ita_refine.py — P2 两阶段 ITA 精修（最小版）在 lhs_dataset_2000 单缝子集上运行。

阶段 1（粗）：P0 层剥离输出（peeling_final）。
阶段 2（精）：在粗深度 ±10 m（步 1 m）上逐点跑单缝 MOC（tf=16s），
以停泵窗口全波形最小二乘拟合（幅值 gain 线性解出）选最优深度。

检验：全波形精修相对模板匹配是否有额外收益（正确核条件下）。

用法
----
    python -m analysis.method_migration.run_ita_refine --n-cases 60
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from analysis.method_migration import common
from analysis.method_migration.build_dictionary import (
    build_config,
    H_EXT,
)
from moc_simulate.wellbore_moc import simulate_wellbore

N_WORKERS = int(os.environ.get("MM_WORKERS", "12"))
TF_FIT = 16.0            # s 精修正演时长
CF_FIX = 1.0e-7          # 精修固定 Cf（匹配剥离的增益通道）
KLEAK_FIX = 1.0e-4
HALF_SPAN_M = 10.0
STEP_M = 1.0


def _run_fit(case_id: int, coarse_depth: float) -> Dict:
    """在 coarse ±span 网格上跑单缝 MOC 并做全波形拟合。"""
    case = common.load_case(case_id)
    cfg = build_config(tf=TF_FIT)
    intact16 = simulate_wellbore(cfg, store_full_field=False)["wellhead_head"]

    # 观测与基线对齐到同一时间轴
    t_fit = np.arange(intact16.size) * 1.0e-3
    mask = (case["t"] >= common.TS_SHUT) & (case["t"] <= TF_FIT)
    obs = case["H_wh"][mask]
    t_obs = case["t"][mask]
    base = np.interp(t_obs, t_fit, intact16)
    resid_obs = obs - base
    w = np.ones(resid_obs.size)
    w[(t_obs < 2.0)] = 0.0  # 排除停泵瞬态

    grid = np.arange(coarse_depth - HALF_SPAN_M,
                     coarse_depth + HALF_SPAN_M + STEP_M, STEP_M)
    costs = np.full(grid.size, np.inf)
    for i, d in enumerate(grid):
        res = simulate_wellbore(
            cfg, fracture_positions=[float(d)],
            fracture_Cf=[CF_FIX], fracture_kleak=[KLEAK_FIX],
            H_ext=H_EXT, store_full_field=False,
        )
        h_single = np.interp(t_obs, t_fit, res["wellhead_head"])
        tpl = h_single - base
        a = w * tpl
        b = w * resid_obs
        g = float(np.dot(a, b) / (np.dot(a, a) + 1e-15))
        cost = float(np.sum((b - g * a) ** 2))
        costs[i] = cost
    i_best = int(np.argmin(costs))
    return {
        "case_id": int(case_id),
        "coarse_depth": float(coarse_depth),
        "refined_depth": float(grid[i_best]),
        "cost_best": float(costs[i_best]),
        "grid": grid.tolist(),
        "costs": costs.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=60)
    args = ap.parse_args()

    # 单缝 case 池 + 固定种子抽样
    idx_list = common.load_index()
    singles = [s["case_id"] for s in idx_list if s["n_frac"] == 1]
    rng = np.random.RandomState(20260803)
    picks = sorted(int(c) for c in rng.choice(singles, size=args.n_cases,
                                              replace=False))

    # 粗深度 = peeling_final 单缝第一峰
    with open("output/analysis/method_migration/peeling_final/nominal_summary.json",
              encoding="utf-8") as fh:
        data = json.load(fh)
    coarse = {
        r["case_id"]: r["detected_depths"][0] if r["detected_depths"] else None
        for r in data["rows"] if r["n_frac"] == 1
    }

    tasks = []
    for cid in picks:
        cd = coarse.get(cid)
        if cd is not None:
            tasks.append((cid, cd))

    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        rows = pool.starmap(_run_fit, tasks)
    print(f"ITA refine: {len(rows)} cases, {time.time()-t0:.0f}s")

    # 误差对比
    truth = {s["case_id"]: common.parse_positions(s["positions_str"])[0]
             for s in idx_list}
    results = []
    for r in rows:
        true_d = truth[r["case_id"]]
        r["true_depth"] = float(true_d)
        r["err_coarse"] = float(abs(r["coarse_depth"] - true_d))
        r["err_refined"] = float(abs(r["refined_depth"] - true_d))
        results.append(r)

    err_c = np.array([r["err_coarse"] for r in results])
    err_r = np.array([r["err_refined"] for r in results])
    summary = {
        "n_cases": len(results),
        "coarse": {
            "median_m": float(np.median(err_c)),
            "mean_m": float(np.mean(err_c)),
            "within10m": float((err_c <= 10).mean()),
            "within5m": float((err_c <= 5).mean()),
        },
        "refined": {
            "median_m": float(np.median(err_r)),
            "mean_m": float(np.mean(err_r)),
            "within10m": float((err_r <= 10).mean()),
            "within5m": float((err_r <= 5).mean()),
        },
        "improved_frac": float((err_r < err_c).mean()),
        "worsened_frac": float((err_r > err_c).mean()),
    }
    common.save_json({"summary": summary, "rows": results},
                     "ita/ita_refine_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
