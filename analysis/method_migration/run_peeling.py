# -*- coding: utf-8 -*-
"""
run_peeling.py — P0 字典序贯剥离（层剥离 / CLEAN 式）在 lhs_dataset_2000 上运行。

算法（每 case）
--------------
1. residual = H_wh − intact（同物理字典的正演基线）
2. 迭代至多 10 次：
   a. 对全部模板（深度 × Cf）在停泵窗口内做归一化互相关，取最佳（d*, cf*, gain*）
   b. 若 corr < max(0.35×首轮corr, 0.015) 停止
   c. 若 d* 距已检出峰 < 5 m 则不重复记录
   d. residual −= gain* × template(d*, cf*)
3. 检出深度 → detection_protocol 盲评分（主容差 10 m + 容差扫描）

G0 核失配：--subset 模式下用失配字典（a/V0/roughness）重跑同一批 case。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from analysis.method_migration import common
from analysis.unified_evaluation.detection_protocol import (
    ScoringConfig,
    score_detections,
    tolerance_sweep,
)

DICT_ROOT = Path("output/analysis/method_migration/dictionary")
N_WORKERS = int(os.environ.get("MM_WORKERS", "14"))
MIN_SEP_M = 5.0
MAX_ITER = 10
STOP_REL = 0.35
STOP_ABS = 0.015


def _load_dict(variant: str) -> Dict[str, np.ndarray]:
    with np.load(DICT_ROOT / variant / "dictionary.npz") as z:
        return {
            "depth_grid": z["depth_grid"],
            "cf_grid": z["cf_grid"],
            "intact": z["intact"],
            "templates": z["templates"],
        }


def peel_signal(h_wh: np.ndarray, intact: np.ndarray, templates: np.ndarray,
                depth_grid: np.ndarray, cf_grid: np.ndarray,
                ts: float = 1.0, dt: float = 1.0e-3,
                stop_rel: float = STOP_REL, stop_abs: float = STOP_ABS,
                max_iter: int = MAX_ITER) -> List[Dict]:
    """字典序贯剥离 → 检出峰列表 {depth_m, corr, gain, cf, iter}"""
    win = np.arange(h_wh.size, dtype=int)
    win = win[(win * dt) >= ts]

    residual = h_wh.astype(np.float64).copy() - intact
    n_d, n_c, n_samp = templates.shape

    t_norms = np.sqrt((templates[:, :, win] ** 2).sum(axis=2)) + 1e-15

    det: List[Dict] = []
    corr_first: Optional[float] = None

    for it in range(max_iter):
        r_win = residual[win]
        r_norm = np.sqrt((r_win ** 2).sum()) + 1e-15
        # (n_d, n_c)
        dots = np.tensordot(templates[:, :, win], r_win, axes=([2], [0]))
        corr = dots / (t_norms * r_norm)
        corr = np.where(corr < 0.0, 0.0, corr)  # 只接受正相关
        i_d, i_c = np.unravel_index(np.argmax(corr), corr.shape)
        c_best = float(corr[i_d, i_c])

        if corr_first is None:
            corr_first = c_best
        if c_best < max(stop_rel * corr_first, stop_abs):
            break

        d_best = float(depth_grid[i_d])
        # 抛物线插值：沿深度方向用相邻模板的相关值精修（亚网格定位）
        if 0 < i_d < n_d - 1:
            c_m, c_0, c_p = (float(corr[i_d - 1, i_c]),
                             float(corr[i_d, i_c]),
                             float(corr[i_d + 1, i_c]))
            denom = c_m - 2.0 * c_0 + c_p
            if abs(denom) > 1e-12:
                d_best += (depth_grid[1] - depth_grid[0]) * 0.5 * (c_m - c_p) / denom
        if all(abs(d_best - d0["depth_m"]) >= MIN_SEP_M for d0 in det):
            gain = float(dots[i_d, i_c] / (t_norms[i_d, i_c] ** 2 + 1e-15))
            det.append({
                "depth_m": d_best,
                "corr": c_best,
                "gain": gain,
                "cf": float(cf_grid[i_c]),
                "iter": int(it + 1),
            })
        residual -= dots[i_d, i_c] / (t_norms[i_d, i_c] ** 2 + 1e-15) * templates[i_d, i_c]

    return det


def _work(args) -> Dict:
    case_id, variant, stop_rel, stop_abs = args
    case = common.load_case(case_id)
    dic = _load_dict(variant)
    det = peel_signal(
        case["H_wh"], dic["intact"], dic["templates"],
        dic["depth_grid"], dic["cf_grid"],
        stop_rel=stop_rel, stop_abs=stop_abs,
    )
    pred = [d["depth_m"] for d in det]
    true = case["x_f"]
    primary = score_detections(pred, true, 10.0)
    return {
        "case_id": case_id,
        "variant": variant,
        "n_frac": case["n_frac"],
        "spacing": common.min_spacing(true),
        "band": common.spacing_band(common.min_spacing(true)),
        "n_pred": len(pred),
        "f1": primary["f1"],
        "sep": bool(primary["separation_success"]),
        "count_err": primary["count_error"],
        "median_err_m": primary["median_error_m"],
        "detected_depths": pred,
        "peel_info": det,
        "sweep": tolerance_sweep(pred, true),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=str, default="all",
                    help="all 或 逗号分隔 case_id 列表")
    ap.add_argument("--variants", nargs="*", default=["nominal"])
    ap.add_argument("--subset-n", type=int, default=200,
                    help="G0 失配评估的固定子集大小")
    ap.add_argument("--out-tag", type=str, default="peeling")
    ap.add_argument("--stop-rel", type=float, default=STOP_REL)
    ap.add_argument("--stop-abs", type=float, default=STOP_ABS)
    args = ap.parse_args()

    idx = {s["case_id"]: s for s in common.load_index()}
    if args.cases == "all":
        case_ids = sorted(idx.keys())
    else:
        case_ids = [int(x) for x in args.cases.split(",")]

    # 固定分层子集（n_frac 1..6 各 1/6）；样本不足时退化为全部
    rng = np.random.RandomState(20260802)
    if len(case_ids) >= args.subset_n:
        stratified: List[int] = []
        for nf in range(1, 7):
            pool_ids = [c for c in case_ids if idx[c]["n_frac"] == nf]
            per = max(1, args.subset_n // 6)
            stratified += list(rng.choice(pool_ids, size=min(per, len(pool_ids)),
                                          replace=False))
        subset = sorted(int(c) for c in stratified)
    else:
        subset = sorted(case_ids)

    all_rows: Dict[str, List[Dict]] = {}
    for variant in args.variants:
        use_ids = case_ids if variant == "nominal" else subset
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            rows = pool.map(
                _work, [(c, variant, args.stop_rel, args.stop_abs)
                        for c in use_ids])
        all_rows[variant] = rows
        s = common.summarize_results(rows)
        common.save_json({"summary": s, "rows": rows, "case_ids": use_ids},
                         f"{args.out_tag}/{variant}_summary.json")
        print(f"[{variant}] n={len(rows)} F1@10={s['f1_mean']:.4f} "
              f"sep={s['sep_rate']:.4f} MAE={s['count_mae']:.3f} "
              f"({time.time()-t0:.0f}s)")

    if len(args.variants) > 1:
        rows = all_rows[args.variants[0]]
        sub_ids = {r["case_id"] for r in rows}
        g0 = {
            "subset_n": len(sub_ids),
            "case_ids": sorted(sub_ids),
            "by_variant": {
                v: common.summarize_results(
                    [r for r in all_rows[v] if r["case_id"] in sub_ids])
                for v in args.variants
            },
        }
        common.save_json(g0, f"{args.out_tag}/G0_mismatch.json")
        print("G0:", {v: round(g0["by_variant"][v]["f1_mean"], 4)
                      for v in args.variants})


if __name__ == "__main__":
    main()
