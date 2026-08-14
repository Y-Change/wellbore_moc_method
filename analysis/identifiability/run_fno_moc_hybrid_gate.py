# -*- coding: utf-8 -*-
"""
run_fno_moc_hybrid_gate.py — P1b 混合路径门槛：FNO 初值 + MOC 坐标下降精修。

验收：精修后 |x_hybrid - x_oracle| 是否进入 CRB 量级（初值是否落入 MOC 吸引域）。
对照：同半径下从 FNO 初值 vs 从真值邻域；并报告初值偏差是否超出 radius·dx。

用法
----
    D:\\Anaconda\\envs\\torch24\\python.exe -m analysis.identifiability.run_fno_moc_hybrid_gate \\
        --tag smoke --cases single_4000 --snr 40 --n-mc 3 --radius 5 --workers 14
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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
    one_step_from_jacobian,
)
from analysis.identifiability.run_efficiency import compute_crb_baselines
from analysis.identifiability.run_fno_mle_gate import fno_mle, load_fno
from analysis.identifiability.scenarios import case_by_name, make_obs, make_scenario


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_snr(s: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for tok in _parse_list(s):
        if tok.lower() in ("inf", "infty", "infinity", "none"):
            out.append(None)
        else:
            out.append(float(tok))
    return out


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n] - b[:n])))


def run_case(
    case_name: str,
    *,
    model,
    device: torch.device,
    snr_list: Sequence[Optional[float]],
    n_mc: int,
    seed: int,
    workers: int,
    n_time: int,
    fno_steps: int,
    fno_lr: float,
    init_jitter_m: float,
    radius: int,
    n_rounds: int,
) -> Dict[str, object]:
    case = case_by_name(case_name)
    scn = make_scenario(case)
    obs = make_obs()
    print(f"\n=== case {case_name}  forward + CRB ===")
    t0 = time.time()
    fr = forward(scn)
    s0 = observe(fr.H_wh, fr.t, scn, obs)
    crb = compute_crb_baselines(scn, obs, workers=workers)
    jac_a = crb["jac_a"]
    placed = np.asarray(crb["placed_x"], dtype=np.float64)
    dx = float(scn.L / scn.n_cells_nominal())
    basin_m = float(radius) * dx
    print(
        f"  placed={placed.tolist()}  CRB@40A={crb['std_x_a_ref']}  "
        f"dx={dx:.3f}m  radius={radius} → basin±{basin_m:.1f}m  ({time.time()-t0:.1f}s)"
    )

    t_moc = np.asarray(fr.t, dtype=np.float64)
    scale_std = crb["scale_std"]
    rng = np.random.default_rng(seed)
    # 跨 trial 复用 MOC 正演（信号与噪声无关）
    signal_cache: Dict[Tuple[int, ...], np.ndarray] = {}

    snr_blocks = []
    for snr in snr_list:
        snr_label = "inf" if snr is None else f"{snr:g}"
        crb_std = [scale_std(ref, snr) for ref in crb["std_x_a_ref"]]
        crb_scale = float(np.mean(crb_std)) if snr is not None else float(
            np.mean(crb["std_x_a_ref"])
        )
        rows = []
        print(f"  SNR={snr_label}  n_mc={n_mc}  shared_cache={len(signal_cache)} ...")
        for i in range(n_mc):
            y, sigma = add_noise(s0, snr, rng)
            oracle = one_step_from_jacobian(jac_a, y)
            x_oracle = np.asarray(oracle["x_hat"], dtype=np.float64)

            jitter = rng.uniform(-init_jitter_m, init_jitter_m, size=placed.shape)
            x_fno_init = np.sort(placed + jitter)
            fno = fno_mle(
                model,
                y,
                x_fno_init,
                case.Cf,
                case.kleak,
                t_moc=t_moc,
                ts=scn.ts,
                dt=scn.dt,
                tf=scn.tf,
                a=scn.a_nominal,
                fc_hz=float(obs.fc_hz),
                n_time=n_time,
                device=device,
                n_steps=fno_steps,
                lr=fno_lr,
            )
            x_fno = np.asarray(fno["x_hat"], dtype=np.float64)
            init_err = _max_abs(x_fno, placed)
            in_radius = bool(init_err <= basin_m + 1e-9)

            t_ref = time.time()
            hybrid = mle_grid_coord_descent(
                scn,
                obs,
                y,
                radius=radius,
                n_rounds=n_rounds,
                x_init=x_fno,
                signal_cache=signal_cache,
                workers=workers,
            )
            x_hyb = np.asarray(hybrid["x_hat"], dtype=np.float64)
            moc_s = time.time() - t_ref

            d_fno_oracle = _max_abs(x_fno, x_oracle)
            d_hyb_oracle = _max_abs(x_hyb, x_oracle)
            d_hyb_truth = _max_abs(x_hyb, placed)
            d_fno_truth = _max_abs(x_fno, placed)

            rows.append(
                {
                    "trial": i,
                    "x_oracle": x_oracle.tolist(),
                    "x_fno": x_fno.tolist(),
                    "x_hybrid": x_hyb.tolist(),
                    "init_err_to_placed_m": init_err,
                    "fno_init_in_radius": in_radius,
                    "max_abs_fno_vs_oracle_m": d_fno_oracle,
                    "max_abs_hybrid_vs_oracle_m": d_hyb_oracle,
                    "max_abs_hybrid_vs_placed_m": d_hyb_truth,
                    "max_abs_fno_vs_placed_m": d_fno_truth,
                    "moc_n_forwards": int(hybrid["n_forwards"]),
                    "moc_wall_s": float(moc_s),
                    "fno_rss": float(fno["rss"]),
                    "hybrid_rss": float(hybrid["rss"]),
                    "sigma": float(sigma),
                }
            )
            if (i + 1) % max(1, n_mc // 5) == 0 or i == 0:
                print(
                    f"    trial {i+1}/{n_mc}  |Δ|_fno={d_fno_oracle:.3f}  "
                    f"|Δ|_hyb={d_hyb_oracle:.3f}  in_R={in_radius}  "
                    f"fwd=+{hybrid['n_forwards']} cache={len(signal_cache)}"
                )

        hyb = np.array([r["max_abs_hybrid_vs_oracle_m"] for r in rows])
        fno_d = np.array([r["max_abs_fno_vs_oracle_m"] for r in rows])
        in_r_rate = float(np.mean([r["fno_init_in_radius"] for r in rows]))
        med_hyb = float(np.median(hyb))
        med_fno = float(np.median(fno_d))
        # 吸引域门槛：精修后贴 oracle 到 CRB 量级（允许 2× 咨询）
        pass_strict = bool(med_hyb < crb_scale)
        pass_adv = bool(med_hyb < max(2.0 * crb_scale, 0.5 * dx))
        # 实用门槛：精修误差 < 1 个网格
        pass_cell = bool(med_hyb < dx)
        block = {
            "snr_db": snr_label,
            "radius_cells": radius,
            "basin_halfwidth_m": basin_m,
            "dx_m": dx,
            "crb_scale_m": crb_scale,
            "median_fno_vs_oracle_m": med_fno,
            "median_hybrid_vs_oracle_m": med_hyb,
            "mean_hybrid_vs_oracle_m": float(np.mean(hyb)),
            "fno_init_in_radius_rate": in_r_rate,
            "ratio_hybrid_to_crb": float(med_hyb / crb_scale) if crb_scale > 0 else float("inf"),
            "pass_strict_lt_crb": pass_strict,
            "pass_advisory_lt_2crb_or_half_dx": pass_adv,
            "pass_lt_one_cell": pass_cell,
            "cache_size": len(signal_cache),
            "trials": rows,
        }
        snr_blocks.append(block)
        print(
            f"  → SNR={snr_label}: med|hyb-oracle|={med_hyb:.4f} m  "
            f"med|fno-oracle|={med_fno:.4f} m  in_R={in_r_rate:.0%}  "
            f"strict={pass_strict} cell={pass_cell} adv={pass_adv}"
        )

    return {
        "case": case_name,
        "placed_x": placed.tolist(),
        "dx_m": dx,
        "radius_cells": radius,
        "basin_halfwidth_m": basin_m,
        "n_mc": n_mc,
        "n_rounds": n_rounds,
        "snr_blocks": snr_blocks,
        "pass_all_strict": all(b["pass_strict_lt_crb"] for b in snr_blocks),
        "pass_all_cell": all(b["pass_lt_one_cell"] for b in snr_blocks),
        "pass_all_advisory": all(
            b["pass_advisory_lt_2crb_or_half_dx"] for b in snr_blocks
        ),
    }


def main():
    if sys.platform.startswith("win") and hasattr(sys.stdout, "buffer") and not sys.stdout.closed:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="FNO-init + MOC-refine hybrid gate")
    p.add_argument("--tag", default="main")
    p.add_argument("--cases", default="single_4000,dual_10m")
    p.add_argument("--snr", default="40")
    p.add_argument("--n-mc", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--ckpt", default="output/models/fno_close2000/fno_surrogate_best.pt")
    p.add_argument("--n-time", type=int, default=4096)
    p.add_argument("--fno-steps", type=int, default=60)
    p.add_argument("--fno-lr", type=float, default=1.0)
    p.add_argument("--init-jitter-m", type=float, default=2.0)
    p.add_argument("--radius", type=int, default=10, help="MOC 坐标下降半径（格）")
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else Path(
        f"output/analysis/identifiability/fno_moc_hybrid/{args.tag}"
    )
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("P1b FNO-init + MOC-refine hybrid gate")
    print(f"  device={device}  ckpt={args.ckpt}")
    print(
        f"  cases={args.cases}  snr={args.snr}  n_mc={args.n_mc}  "
        f"radius={args.radius}  workers={args.workers}"
    )
    print("=" * 72)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    model = load_fno(args.ckpt, device)

    results = []
    t_all = time.time()
    for name in _parse_list(args.cases):
        r = run_case(
            name,
            model=model,
            device=device,
            snr_list=_parse_snr(args.snr),
            n_mc=args.n_mc,
            seed=args.seed,
            workers=args.workers,
            n_time=args.n_time,
            fno_steps=args.fno_steps,
            fno_lr=args.fno_lr,
            init_jitter_m=args.init_jitter_m,
            radius=args.radius,
            n_rounds=args.n_rounds,
        )
        results.append(r)
        (out / f"{name}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")

    summary = {
        "tag": args.tag,
        "ckpt": os.path.abspath(args.ckpt),
        "device": str(device),
        "radius_cells": args.radius,
        "n_mc": args.n_mc,
        "gate_note": (
            "Hybrid: x_fno=FNO-MLE init → MOC coord-descent refine. "
            "Pass if median |x_hybrid-x_oracle| < CRB (strict) or <1 cell (practical)."
        ),
        "pass_all_strict": all(r["pass_all_strict"] for r in results),
        "pass_all_cell": all(r["pass_all_cell"] for r in results),
        "pass_all_advisory": all(r["pass_all_advisory"] for r in results),
        "results": [
            {
                "case": r["case"],
                "basin_halfwidth_m": r["basin_halfwidth_m"],
                "pass_all_strict": r["pass_all_strict"],
                "pass_all_cell": r["pass_all_cell"],
                "pass_all_advisory": r["pass_all_advisory"],
                "snr_summary": [
                    {
                        "snr_db": b["snr_db"],
                        "median_fno_vs_oracle_m": b["median_fno_vs_oracle_m"],
                        "median_hybrid_vs_oracle_m": b["median_hybrid_vs_oracle_m"],
                        "fno_init_in_radius_rate": b["fno_init_in_radius_rate"],
                        "ratio_hybrid_to_crb": b["ratio_hybrid_to_crb"],
                        "pass_strict_lt_crb": b["pass_strict_lt_crb"],
                        "pass_lt_one_cell": b["pass_lt_one_cell"],
                        "pass_advisory_lt_2crb_or_half_dx": b[
                            "pass_advisory_lt_2crb_or_half_dx"
                        ],
                    }
                    for b in r["snr_blocks"]
                ],
            }
            for r in results
        ],
        "wall_s": float(time.time() - t_all),
    }
    (out / "hybrid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 72)
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2))
    for r in summary["results"]:
        print(json.dumps(r, indent=2))
    print(f"wrote {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
