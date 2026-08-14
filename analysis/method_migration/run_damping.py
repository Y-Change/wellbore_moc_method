# -*- coding: utf-8 -*-
"""
run_damping.py — P3 多周期模态阻尼率特征（TWD 视角）在 lhs_dataset_2000 上运行。

对每个 case：停泵后 H_wh 各谐波模态（f_n = (2n-1)·a/(4L)）带通 → Hilbert
包络 → 对数包络斜率 = 阻尼率 α_n。检验：
1. 阻尼率是否携带裂缝信息（与 n_frac / ΣCf / Σkleak / 间距 的相关性）
2. 有无裂缝（对照 intact 无裂缝信号）的阻尼率差异

输出：output/analysis/method_migration/damping/damping_summary.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import spearmanr

from analysis.method_migration import common

N_WORKERS = int(os.environ.get("MM_WORKERS", "14"))
A_WAVE = 1450.0
L_WELL = 5000.0
TS = 1.0
N_MODES = 4
INTACT = Path("output/analysis/method_migration/dictionary/nominal/dictionary.npz")


def modal_damping(h: np.ndarray, t: np.ndarray, mode_idx: int) -> float:
    """第 mode_idx 个谐波模态（1-based）的对数包络斜率（阻尼率）。"""
    f_n = (2 * mode_idx - 1) * A_WAVE / (4 * L_WELL)
    mask = t >= TS
    x = h[mask] - np.mean(h[mask])
    dt = float(t[mask][1] - t[mask][0])
    fs = 1.0 / dt

    # 带通：半带宽取相邻模态间距的 40%
    bw = 0.4 * A_WAVE / (2 * L_WELL)
    lo, hi = max(f_n - bw, 1e-4), f_n + bw
    sos = scipy_signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = scipy_signal.sosfiltfilt(sos, x)

    env = np.abs(scipy_signal.hilbert(y))
    # 用 15–45 s 的包络拟合指数衰减（避开初始瞬态）
    t_w = t[mask]
    sel = (t_w >= 15.0) & (t_w <= 45.0)
    if sel.sum() < 100:
        return float("nan")
    tt = t_w[sel]
    ee = env[sel]
    ee = np.maximum(ee, 1e-12)
    slope, _ = np.polyfit(tt - tt[0], np.log(ee), 1)
    return float(-slope)


def _work(case_id: int) -> Dict:
    case = common.load_case(case_id)
    alphas = [modal_damping(case["H_wh"], case["t"], m) for m in range(1, N_MODES + 1)]
    return {
        "case_id": int(case_id),
        "n_frac": case["n_frac"],
        "spacing": common.min_spacing(case["x_f"]),
        "band": common.spacing_band(common.min_spacing(case["x_f"])),
        "sum_Cf": float(np.log10(np.sum(case["Cf"]))),
        "sum_kleak": float(np.log10(np.sum(case["kleak"]))),
        "alpha": alphas,
    }


def _intact_alphas() -> List[float]:
    with np.load(INTACT) as z:
        intact = z["intact"]
        t = np.arange(intact.size) * 1.0e-3
    return [modal_damping(intact, t, m) for m in range(1, N_MODES + 1)]


def main() -> None:
    t0 = time.time()
    idx_list = common.load_index()
    case_ids = [s["case_id"] for s in idx_list]
    with mp.Pool(N_WORKERS) as pool:
        rows = pool.map(_work, case_ids)

    rows_sorted = sorted(rows, key=lambda r: r["case_id"])
    a = np.array([r["alpha"] for r in rows_sorted])
    n_frac = np.array([r["n_frac"] for r in rows_sorted])
    sum_cf = np.array([r["sum_Cf"] for r in rows_sorted])
    sum_kl = np.array([r["sum_kleak"] for r in rows_sorted])
    spacing = np.array([r["spacing"] for r in rows_sorted])

    intact_a = _intact_alphas()
    corr = {}
    for m in range(N_MODES):
        corr[f"mode{m+1}"] = {
            "vs_n_frac": float(spearmanr(a[:, m], n_frac).statistic),
            "vs_logsumCf": float(spearmanr(a[:, m], sum_cf).statistic),
            "vs_logsumkleak": float(spearmanr(a[:, m], sum_kl).statistic),
            "vs_spacing": float(spearmanr(a[:, m], spacing).statistic),
        }
    summary = {
        "n_cases": len(rows_sorted),
        "intact_alpha": intact_a,
        "alpha_stats": {
            f"mode{m+1}": {
                "mean": float(np.nanmean(a[:, m])),
                "median": float(np.nanmedian(a[:, m])),
                "p10": float(np.nanpercentile(a[:, m], 10)),
                "p90": float(np.nanpercentile(a[:, m], 90)),
            }
            for m in range(N_MODES)
        },
        "spearman": corr,
        "intact_vs_case_delta": {
            f"mode{m+1}": {
                "intact": intact_a[m],
                "case_median": float(np.nanmedian(a[:, m])),
                "delta": float(np.nanmedian(a[:, m]) - intact_a[m]),
            }
            for m in range(N_MODES)
        },
    }
    common.save_json({"summary": summary, "rows": rows_sorted},
                     "damping/damping_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:1500])
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
