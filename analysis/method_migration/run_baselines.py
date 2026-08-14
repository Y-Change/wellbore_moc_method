# -*- coding: utf-8 -*-
"""
run_baselines.py — 在 lhs_dataset_2000 上运行基线检测器（全部 2000 case）。

* cepstrum：``compute_moc_cepstrum_1d``（深度域响应）→ 盲评分
* cwt     ：停泵后小波脊包络（时-深映射 depth = a·(t-ts)/2）→ 盲评分

输出：output/analysis/method_migration/baselines/
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pywt
from scipy import signal as scipy_signal

from analysis.method_migration import common
from analysis.unified_evaluation.detection_protocol import evaluate_blind
from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum_1d

N_WORKERS = int(os.environ.get("MM_WORKERS", "14"))


# ---------------------------------------------------------------- cepstrum
def _cepstrum_response(h: np.ndarray, t: np.ndarray) -> Dict:
    res = compute_moc_cepstrum_1d(
        t, h, v=common.A_WAVE, dt=1.0e-3, ts=common.TS_SHUT,
        wellbore_length=common.L_WELL,
    )
    return {"depth": res["depth"], "response": res["response"]}


# ---------------------------------------------------------------- cwt
def _cwt_response(h: np.ndarray, t: np.ndarray) -> Dict:
    """停泵后小波脊包络 → 深度域响应（depth = a·(t-ts)/2）。"""
    mask = t >= common.TS_SHUT
    t_w = t[mask]
    x = h[mask] - np.mean(h[mask])
    dt = float(t_w[1] - t_w[0])
    fs = 1.0 / dt

    scales = np.geomspace(2.0, 200.0, 32)
    coefs, _ = pywt.cwt(x, scales, "morl", sampling_period=dt)
    # 每尺度归一化后取脊包络
    rows_norm = np.abs(coefs) / (
        np.max(np.abs(coefs), axis=1, keepdims=True) + 1e-12
    )
    ridge = np.max(rows_norm, axis=0)

    depth_w = common.A_WAVE * (t_w - common.TS_SHUT) / 2.0
    sel = (depth_w >= common.SEARCH_WIN[0]) & (depth_w <= common.SEARCH_WIN[1])
    if sel.sum() < 50:
        return {"depth": np.array([]), "response": np.array([])}
    depth_grid = np.arange(common.SEARCH_WIN[0], common.SEARCH_WIN[1], 1.0)
    resp = np.interp(depth_grid, depth_w[sel], ridge[sel])
    return {"depth": depth_grid, "response": resp}


# ---------------------------------------------------------------- 并行
def _work(args) -> Dict:
    idx, method = args
    case = common.load_case(idx)
    t, h = case["t"], case["H_wh"]
    if method == "cepstrum":
        res = _cepstrum_response(h, t)
    else:
        res = _cwt_response(h, t)
    ev = evaluate_blind(
        res["depth"], res["response"], case["x_f"],
        detector_config=common.DET_CFG, scoring_config=common.SCORE_CFG,
    )
    row = common.row_from_eval(idx, case["n_frac"], common.min_spacing(case["x_f"]), ev)
    row["method"] = method
    return row


def run(method: str, case_ids: List[int]) -> List[Dict]:
    with mp.Pool(N_WORKERS) as pool:
        rows = pool.map(_work, [(i, method) for i in case_ids])
    return rows


def main() -> None:
    idx_list = common.load_index()
    case_ids = [s["case_id"] for s in idx_list]
    out_dir = common.OUT_ROOT / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in ["cepstrum", "cwt"]:
        t0 = time.time()
        rows = run(method, case_ids)
        summary = common.summarize_results(rows)
        common.save_json({"summary": summary, "rows": rows},
                         f"baselines/{method}_summary.json")
        with (out_dir / f"{method}_summary.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# {method} on lhs_dataset_2000 (n={summary['n_cases']})\n\n")
            fh.write(f"- F1@10m mean: {summary['f1_mean']:.4f}\n")
            fh.write(f"- separation rate: {summary['sep_rate']:.4f}\n")
            fh.write(f"- count MAE: {summary['count_mae']:.3f}\n")
            fh.write(f"- elapsed: {time.time() - t0:.0f} s\n")
        print(f"[{method}] F1={summary['f1_mean']:.4f} sep={summary['sep_rate']:.4f} "
              f"MAE={summary['count_mae']:.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
