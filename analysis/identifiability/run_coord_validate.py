# -*- coding: utf-8 -*-
"""
run_coord_validate.py — 坐标下降 vs 穷举网格验收（P2a 门控）。

在 dual_10m / dual_40m 上：同一噪声实现下，坐标下降与穷举网格的
位置差须 < 0.1·dx = 0.145 m（dx≈1.45 m）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.mle import (  # noqa: E402
    add_noise,
    build_position_cache,
    mle_from_cache,
    mle_grid_coord_descent,
)
from analysis.identifiability.scenarios import (  # noqa: E402
    case_by_name,
    make_obs,
    make_scenario,
)

DEFAULT_OUT = os.path.join("output", "analysis", "identifiability", "coord_validate")
GATE_M = 0.145  # 0.1 * dx


def _run_one(case_name: str, *, radius: int, snr: Optional[float],
             seed: int, workers: int, n_rounds: int) -> dict:
    case = case_by_name(case_name)
    scn = make_scenario(case)
    obs = make_obs()
    rng = np.random.default_rng(seed)

    t0 = time.time()
    cache = build_position_cache(scn, obs, radius=radius, workers=workers)
    t_cache = time.time() - t0

    y, sigma = add_noise(cache.s_truth, snr, rng)

    # 穷举
    mle_ex = mle_from_cache(cache, y, mode="A", skip_gn=True)
    x_ex = mle_ex.x_hat

    # 坐标下降：从真值起步（与效率实验一致的邻域设定）
    t1 = time.time()
    cd = mle_grid_coord_descent(
        scn, obs, y,
        radius=radius, n_rounds=n_rounds, workers=workers,
        x_init=cache.placed_truth,
    )
    t_cd = time.time() - t1
    x_cd = np.asarray(cd["x_hat"], dtype=np.float64)

    # 匹配后逐缝差
    err = np.sort(x_cd) - np.sort(x_ex)
    max_abs = float(np.max(np.abs(err)))
    passed = max_abs < GATE_M

    return {
        "case": case_name,
        "snr": "inf" if snr is None else snr,
        "radius": radius,
        "dx": float(cache.dx),
        "gate_m": GATE_M,
        "x_exhaustive": x_ex.tolist(),
        "x_coord_descent": x_cd.tolist(),
        "err_m": err.tolist(),
        "max_abs_err_m": max_abs,
        "rss_exhaustive": mle_ex.rss_grid,
        "rss_coord_descent": cd["rss"],
        "n_forwards_cd": cd["n_forwards"],
        "n_cache_exhaustive": len(cache.signals),
        "t_cache_s": t_cache,
        "t_cd_s": t_cd,
        "passed": passed,
    }


def main():
    p = argparse.ArgumentParser(description="坐标下降 vs 穷举网格验收")
    p.add_argument("--tag", default="gate")
    p.add_argument("--cases", default="dual_10m,dual_40m")
    p.add_argument("--radius", type=int, default=8)
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--snr", default="inf", help="inf 或数值 dB")
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    snr: Optional[float]
    if args.snr.lower() in ("inf", "infty", "none"):
        snr = None
    else:
        snr = float(args.snr)

    out_dir = Path(DEFAULT_OUT) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_pass = True
    for name in [c.strip() for c in args.cases.split(",") if c.strip()]:
        print(f"\n=== {name} ===")
        r = _run_one(
            name, radius=args.radius, snr=snr, seed=args.seed,
            workers=args.workers, n_rounds=args.n_rounds,
        )
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  [{status}] max|Δx|={r['max_abs_err_m']:.4f} m "
            f"(gate={GATE_M} m); rss_ex={r['rss_exhaustive']:.4e} "
            f"rss_cd={r['rss_coord_descent']:.4e}; "
            f"forwards_cd={r['n_forwards_cd']}"
        )
        if not r["passed"]:
            all_pass = False

    summary = {
        "tag": args.tag,
        "gate_m": GATE_M,
        "all_passed": all_pass,
        "results": results,
    }
    path = out_dir / "coord_validate.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n写入 {path}")
    print("总体:", "PASS" if all_pass else "FAIL")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
