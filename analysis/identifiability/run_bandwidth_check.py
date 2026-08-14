# -*- coding: utf-8 -*-
"""
run_bandwidth_check.py — 单场景高带宽 CRB 标度复核（dt=2e-4）。

回应审稿人「真实传感器可达 kHz，你只做到 20 Hz」的质疑：
在差分收敛域内提高 fc，确认 σ_x 随带宽继续下降（无信息墙）。

用法
----
    python -m analysis.identifiability.run_bandwidth_check --tag hifreq \\
        --case single_4000 --workers 14
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import (  # noqa: E402
    ObsModel,
    assemble_jacobian,
    build_jobs,
    crb_report,
    run_jobs,
)
from analysis.identifiability.scenarios import (  # noqa: E402
    case_by_name,
    make_scenario,
    scenario_summary,
)

FC_DT_LIMIT = 0.021
DEFAULT_OUT = os.path.join("output", "analysis", "identifiability", "bandwidth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hifreq")
    ap.add_argument("--case", default="single_4000")
    ap.add_argument("--dt", type=float, default=2.0e-4)
    ap.add_argument("--tf", type=float, default=20.0,
                    help="高分辨率时缩短 tf 以控制成本")
    ap.add_argument("--fc", default="20,50,100,200")
    ap.add_argument("--snr", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    fcs = [float(x) for x in args.fc.split(",")]
    case = case_by_name(args.case)
    scn = make_scenario(case, dt=args.dt, tf=args.tf)
    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"高带宽 CRB 复核  case={args.case}  dt={args.dt:g}  tf={args.tf}")
    print(f"  fc={fcs}  收敛上限 fc≤{FC_DT_LIMIT / args.dt:.0f} Hz")
    print(f"  n_cells≈{scn.n_cells_nominal()}")
    print("=" * 72)

    t0 = time.time()
    scn2, names, jobs = build_jobs(scn, include=("x", "cf", "kleak", "a"))
    print(f"  正演任务 {len(jobs)} …")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            raws = run_jobs(scn2, jobs, map_fn=ex.map)
    else:
        raws = run_jobs(scn2, jobs)
    print(f"  正演完成 ({time.time()-t0:.1f}s)")

    rows: List[Dict] = []
    for fc in fcs:
        obs = ObsModel(fc_hz=fc)
        jac = assemble_jacobian(scn2, names, jobs, raws, obs)
        rep = crb_report(jac, args.snr, nuisance=("a",))
        idx0 = names.index("x0")
        row = {
            "fc_hz": fc,
            "fc_times_dt": fc * args.dt,
            "fd_converged": (fc * args.dt) <= FC_DT_LIMIT,
            "std_x0_full_m": float(rep["std_full"][idx0]),
            "std_x0_a_known_m": float(rep["std_no_nuisance"][idx0]),
            "std_a_mps": float(rep["std_full"][names.index("a")]),
            "a_penalty_x0": float(
                rep["std_full"][idx0] / rep["std_no_nuisance"][idx0]
            ),
            "cond_full": rep["cond_normalized"],
            "n_obs": rep["n_obs"],
        }
        rows.append(row)
        flag = "OK" if row["fd_converged"] else "OUT"
        print(f"  fc={fc:6g} Hz  [{flag}]  "
              f"σ(x0|a已知)={row['std_x0_a_known_m']:.4e} m  "
              f"σ(x0|联合)={row['std_x0_full_m']:.4e} m  "
              f"代价×{row['a_penalty_x0']:.2f}")

    # 标度指数：在收敛域内对 log σ ~ α log fc 拟合
    conv = [r for r in rows if r["fd_converged"]]
    scale_fit = None
    if len(conv) >= 2:
        log_fc = np.log([r["fc_hz"] for r in conv])
        log_std = np.log([r["std_x0_a_known_m"] for r in conv])
        coef = np.polyfit(log_fc, log_std, 1)
        scale_fit = {
            "alpha": float(coef[0]),  # σ ∝ fc^α，期望 α≈-1 ~ -1.5
            "intercept": float(coef[1]),
            "n_points": len(conv),
            "interpretation": (
                "α<0 表示带宽越高定位越准，无信息墙；"
                "理论窄带脉冲定位 roughly σ∝1/(SNR·Bandwidth)"
            ),
        }
        print(f"\n  标度拟合 σ_x ∝ fc^{scale_fit['alpha']:.3f}  "
              f"(收敛域 n={len(conv)})")

    payload = {
        "case": args.case,
        "scenario": scenario_summary(scn2),
        "snr_db": args.snr,
        "dt_s": args.dt,
        "tf_s": args.tf,
        "fc_dt_limit": FC_DT_LIMIT,
        "rows": rows,
        "scale_fit": scale_fit,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = out_dir / f"{args.case}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
