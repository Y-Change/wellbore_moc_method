# -*- coding: utf-8 -*-
"""
run_music.py — P4 频域子空间超分辨（MUSIC）在 lhs_dataset_2000 子集上运行。

原理
----
反射序列使井口频谱 H(f) = H0(f)·(1 + Σ r_k e^{-j2πf τ_k})，τ_k = 2d_k/a。
把残差谱 R(f) = H(f) − H_intact(f) 看作 f 轴上的复指数和（频率 1/τ_k），
用 MUSIC 在 quefrency 域（→ 深度域）估计谱峰，突破 Rayleigh 带宽限制。

用法
----
    python -m analysis.method_migration.run_music --cases 0..199
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from analysis.method_migration import common
from analysis.unified_evaluation.detection_protocol import (
    evaluate_blind,
)

N_WORKERS = int(os.environ.get("MM_WORKERS", "12"))
INTACT = Path("output/analysis/method_migration/dictionary/nominal/dictionary.npz")

F_MIN, F_MAX = 0.05, 10.0     # Hz 谱窗
WIN_S = 30.0                  # s 分析窗（停泵后）
P_WIN = 14                    # MUSIC 快照窗口长（f 轴样本数）
N_SIGNAL_MAX = 6


def music_pseudo(
    r: np.ndarray,
    f: np.ndarray,
    depth_grid: np.ndarray,
    p_win: int = P_WIN,
) -> np.ndarray:
    """MUSIC 伪谱 over depth_grid。"""
    n = f.size
    snaps = []
    for i0 in range(0, n - p_win + 1, 2):
        snaps.append(r[i0:i0 + p_win])
    X = np.array(snaps).T  # (p, Nsnap)
    X = X - X.mean(axis=1, keepdims=True)
    p, ns = X.shape
    if ns < p:
        return np.full(depth_grid.size, np.nan)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    # 信号秩：奇异值显著高于噪声底的个数（动态阈值）
    s_thresh = 0.05 * s[0]
    k_sig = max(1, int(np.sum(s > s_thresh)))
    k_sig = min(k_sig, p - 2)
    if k_sig < 1:
        k_sig = 1
    En = U[:, k_sig:]  # 噪声子空间
    tau = 2.0 * depth_grid / common.A_WAVE
    f_win = f[:p_win]
    a = np.exp(-1j * 2 * np.pi * np.outer(f_win, tau))  # (p, n_tau)
    proj = En.conj().T @ a  # (p-k, n_tau)
    pseudo = 1.0 / (np.sum(np.abs(proj) ** 2, axis=0) + 1e-15)
    return pseudo


def _work(case_id: int) -> Dict:
    case = common.load_case(case_id)
    with np.load(INTACT) as z:
        intact = z["intact"]
        t_int = np.arange(intact.size) * 1.0e-3
    dt = 1.0e-3
    mask = (case["t"] >= common.TS_SHUT) & (case["t"] <= common.TS_SHUT + WIN_S)
    t_w = case["t"][mask]
    h_frac = case["H_wh"][mask] - np.mean(case["H_wh"][mask])
    h_int = np.interp(t_w, t_int, intact) - np.mean(intact)
    r_t = h_frac - h_int
    r_t = r_t * np.hanning(r_t.size)

    X = np.fft.rfft(r_t)
    f = np.fft.rfftfreq(r_t.size, dt)
    sel = (f >= F_MIN) & (f <= F_MAX)
    r_f = X[sel]
    f_sel = f[sel]

    depth_grid = np.arange(common.SEARCH_WIN[0], common.SEARCH_WIN[1], 1.0)
    pseudo = music_pseudo(r_f, f_sel, depth_grid)
    pseudo = np.nan_to_num(pseudo, nan=0.0)
    ev = evaluate_blind(
        depth_grid, pseudo, case["x_f"],
        detector_config=common.DET_CFG, scoring_config=common.SCORE_CFG,
    )
    return common.row_from_eval(
        int(case_id), case["n_frac"], common.min_spacing(case["x_f"]), ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=str, default="0..199",
                    help="case 范围：'a..b' 或逗号列表")
    args = ap.parse_args()
    if ".." in args.cases:
        lo, hi = (int(x) for x in args.cases.split(".."))
        case_ids = list(range(lo, hi + 1))
    else:
        case_ids = [int(x) for x in args.cases.split(",")]

    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        rows = pool.map(_work, case_ids)
    summary = common.summarize_results(rows)
    common.save_json({"summary": summary, "rows": rows, "case_ids": case_ids},
                     "music/music_summary.json")
    print(f"[music] n={len(rows)} F1@10={summary['f1_mean']:.4f} "
          f"sep={summary['sep_rate']:.4f} MAE={summary['count_mae']:.3f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
