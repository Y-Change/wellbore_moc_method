# -*- coding: utf-8 -*-
"""
run_moc_pipeline.py — MOC 主路径实用管线：粗初值 → MOC 坐标下降精修。

策略（项目锁定）：位置精修以 MOC 全波形为准；FNO / 倒谱仅作初值辅助。

用法
----
    python -m analysis.identifiability.run_moc_pipeline \\
        --tag smoke --cases single_4000 --init cepstrum --snr 40 --n-mc 3 --workers 14
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
from concurrent.futures import ProcessPoolExecutor

from analysis.identifiability.mle import (
    add_noise,
    mle_grid_coord_descent,
    mle_refine_gn,
    one_step_from_jacobian,
)
from analysis.identifiability.run_efficiency import compute_crb_baselines
from analysis.identifiability.scenarios import (
    A_WAVE,
    case_by_name,
    make_obs,
    make_scenario,
)
from analysis.unified_evaluation.detection_protocol import DetectorConfig, detect_peaks
from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum_1d

_P0_DICT_PATH = Path("output/analysis/method_migration/dictionary/nominal/dictionary.npz")
_P0_DICT_CACHE: Optional[Dict[str, np.ndarray]] = None


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


def _pad_depths(pred: List[float], n_frac: int, *, fallback: float = 4000.0) -> np.ndarray:
    """缝数已知时补齐峰：缺峰则在末峰后按 10 m 级联。"""
    pred = list(pred)
    if not pred:
        pred = [float(fallback)]
    while len(pred) < n_frac:
        pred.append(float(pred[-1]) + 10.0)
    return np.sort(np.asarray(pred[:n_frac], dtype=np.float64))


def coarse_init_cepstrum(
    t: np.ndarray,
    H_wh: np.ndarray,
    n_frac: int,
    *,
    v: float = A_WAVE,
    depth_min: float = 3400.0,
    depth_max: float = 4900.0,
) -> np.ndarray:
    """倒谱峰作粗初值（缝数已知：取前 n_frac 峰）。"""
    cep = compute_moc_cepstrum_1d(
        t, H_wh, v=v, ts=1.0, wellbore_length=5000.0, depth_min=depth_min,
    )
    depth = np.asarray(cep["depth"], dtype=np.float64)
    resp = np.asarray(cep["response"], dtype=np.float64)
    det_cfg = DetectorConfig(
        min_separation_m=5.0,
        height_pct=85.0,
        height_rel=0.03,
        max_peaks=max(10, n_frac + 2),
        search_min_m=depth_min,
        search_max_m=depth_max,
    )
    peaks = detect_peaks(depth, resp, det_cfg)
    depths = [float(p["depth_m"]) for p in peaks]
    if not depths:
        return _pad_depths([], n_frac, fallback=0.5 * (depth_min + depth_max))
    # 多缝：以首峰为锚，只接纳落在簇内（5–80 m）的后续峰，抑制趾端/旁瓣假峰
    selected = [depths[0]]
    for d in depths[1:]:
        if len(selected) >= n_frac:
            break
        if any(5.0 <= abs(d - s) <= 80.0 for s in selected):
            selected.append(d)
    return _pad_depths(selected, n_frac, fallback=selected[0])


def _load_p0_dict() -> Dict[str, np.ndarray]:
    global _P0_DICT_CACHE
    if _P0_DICT_CACHE is None:
        if not _P0_DICT_PATH.is_file():
            raise FileNotFoundError(_P0_DICT_PATH)
        with np.load(_P0_DICT_PATH) as z:
            _P0_DICT_CACHE = {
                "depth_grid": z["depth_grid"],
                "cf_grid": z["cf_grid"],
                "intact": z["intact"],
                "templates": z["templates"],
            }
    return _P0_DICT_CACHE


def coarse_init_p0(H_wh: np.ndarray, n_frac: int) -> np.ndarray:
    """P0 字典剥离作粗初值（缝数已知）。"""
    from analysis.method_migration.run_peeling import peel_signal

    dic = _load_p0_dict()
    det = peel_signal(
        H_wh, dic["intact"], dic["templates"], dic["depth_grid"], dic["cf_grid"],
    )
    pred = [float(d["depth_m"]) for d in det[:n_frac]]
    return _pad_depths(pred, n_frac, fallback=4000.0)

def adaptive_radius_cells(
    x_init: np.ndarray,
    *,
    dx: float,
    radius_min: int,
    radius_max: int,
    margin_cells: int,
    prior_uncertainty_m: float,
) -> int:
    """由初值不确定度量级定坐标下降半径（无真值时用 prior）。"""
    need = int(math.ceil(float(prior_uncertainty_m) / max(dx, 1e-9))) + int(margin_cells)
    return int(np.clip(max(radius_min, need), radius_min, radius_max))


def run_pipeline_once(
    scn,
    obs,
    y: np.ndarray,
    t_full: np.ndarray,
    H_full: np.ndarray,
    *,
    n_frac: int,
    init: str,
    dx: float,
    radius: int,
    n_rounds: int,
    workers: int,
    signal_cache: Dict[Tuple[int, ...], np.ndarray],
    jac_a: Optional[dict] = None,
    oracle_gn: bool = False,
    deploy_gn: bool = False,
    gn_steps: int = 1,
) -> Dict[str, object]:
    if init == "cepstrum":
        x0 = coarse_init_cepstrum(t_full, H_full, n_frac, v=scn.a_nominal)
    elif init == "p0":
        x0 = coarse_init_p0(H_full, n_frac)
    elif init == "midwell":
        x0 = np.linspace(3800.0, 4200.0, n_frac, dtype=np.float64)
    else:
        raise ValueError(f"未知 init={init}（支持 cepstrum|p0|midwell）")

    refined = mle_grid_coord_descent(
        scn,
        obs,
        y,
        radius=radius,
        n_rounds=n_rounds,
        x_init=x0,
        signal_cache=signal_cache,
        workers=workers,
    )
    x_grid = np.asarray(refined["x_hat"], dtype=np.float64)
    x_hat = x_grid.copy()

    extras: Dict[str, object] = {
        "x_init": x0.tolist(),
        "x_grid": x_grid.tolist(),
        "rss_grid": float(refined["rss"]),
        "n_forwards": int(refined["n_forwards"]),
        "radius_cells": radius,
        "gn_mode": "none",
    }
    if oracle_gn and jac_a is not None:
        # 诊断：真值处 J（非部署）
        step = one_step_from_jacobian(jac_a, y, s_ref=None, x_ref=x_grid)
        x_hat = np.asarray(step["x_hat"], dtype=np.float64)
        extras["gn_mode"] = "oracle"
        extras["x_after_gn"] = x_hat.tolist()
    elif deploy_gn:
        # 部署：在 x_grid 处重算 Jacobian + 一步（或多步）GN
        pool = None
        map_fn = None
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers)
            map_fn = pool.map
        try:
            gn_out = mle_refine_gn(
                scn,
                obs,
                y,
                x_init=x_grid,
                mode="A",
                n_steps=int(gn_steps),
                map_fn=map_fn,
            )
        finally:
            if pool is not None:
                pool.shutdown()
        x_hat = np.asarray(gn_out["x_hat"], dtype=np.float64)
        extras["gn_mode"] = "deploy"
        extras["x_after_gn"] = x_hat.tolist()
        extras["rss_after_gn"] = float(gn_out["rss"])
        extras["n_forwards"] = int(extras["n_forwards"]) + int(
            gn_out.get("n_steps", gn_steps)
        ) * (2 * n_frac + 2)  # 粗计：中心差分 + 更新后正演

    return {"x_hat": x_hat, "x_init": x0, "x_grid": x_grid, **extras}

def run_case(
    case_name: str,
    *,
    init: str,
    snr_list: Sequence[Optional[float]],
    n_mc: int,
    seed: int,
    workers: int,
    radius_min: int,
    radius_max: int,
    margin_cells: int,
    prior_uncertainty_m: float,
    n_rounds: int,
    oracle_gn: bool,
    deploy_gn: bool,
    gn_steps: int,
) -> Dict[str, object]:
    case = case_by_name(case_name)
    scn = make_scenario(case)
    obs = make_obs()
    n_frac = len(case.x_f)
    # prior<=0：按缝数自动（单缝~15 m，双缝~55 m）
    prior_m = float(prior_uncertainty_m)
    if prior_m <= 0:
        prior_m = 15.0 if n_frac <= 1 else 55.0
    print(
        f"\n=== {case_name}  init={init}  deploy_gn={deploy_gn} "
        f"oracle_gn={oracle_gn} ===",
        flush=True,
    )
    t0 = time.time()
    print("  [1/3] MOC forward ...", flush=True)
    fr = forward(scn)
    s0 = observe(fr.H_wh, fr.t, scn, obs)
    print("  [2/3] CRB baselines ...", flush=True)
    crb = compute_crb_baselines(scn, obs, workers=workers)
    jac_a = crb["jac_a"]
    placed = np.asarray(crb["placed_x"], dtype=np.float64)
    dx = float(scn.L / scn.n_cells_nominal())
    radius = adaptive_radius_cells(
        placed,
        dx=dx,
        radius_min=radius_min,
        radius_max=radius_max,
        margin_cells=margin_cells,
        prior_uncertainty_m=prior_m,
    )
    print(
        f"  placed={placed.tolist()}  dx={dx:.3f}  R={radius} "
        f"(prior±{prior_m}m)  CRB@40A={crb['std_x_a_ref']}  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    print("  [3/3] MC refine ...", flush=True)

    scale_std = crb["scale_std"]
    rng = np.random.default_rng(seed)
    cache: Dict[Tuple[int, ...], np.ndarray] = {}
    blocks = []

    for snr in snr_list:
        snr_label = "inf" if snr is None else f"{snr:g}"
        crb_scale = float(np.mean(
            [scale_std(r, snr) for r in crb["std_x_a_ref"]]
            if snr is not None
            else crb["std_x_a_ref"]
        ))
        rows = []
        print(f"  SNR={snr_label}  n_mc={n_mc} ...", flush=True)
        for i in range(n_mc):
            y, sigma = add_noise(s0, snr, rng)
            H_noisy = np.asarray(fr.H_wh, dtype=np.float64).copy()
            if sigma > 0:
                H_noisy = H_noisy + rng.normal(0.0, sigma, size=H_noisy.shape)

            oracle = one_step_from_jacobian(jac_a, y)
            x_oracle = np.asarray(oracle["x_hat"], dtype=np.float64)

            if init == "cepstrum":
                x_init_preview = coarse_init_cepstrum(
                    fr.t, H_noisy, n_frac, v=scn.a_nominal
                )
            elif init == "p0":
                x_init_preview = coarse_init_p0(H_noisy, n_frac)
            else:
                x_init_preview = np.linspace(3800.0, 4200.0, n_frac)
            print(
                f"    trial {i+1}/{n_mc} init={np.round(x_init_preview, 2).tolist()} "
                f"|e|={float(np.max(np.abs(x_init_preview - placed))):.2f}m → refine...",
                flush=True,
            )
            pipe = run_pipeline_once(
                scn,
                obs,
                y,
                fr.t,
                H_noisy,
                n_frac=n_frac,
                init=init,
                dx=dx,
                radius=radius,
                n_rounds=n_rounds,
                workers=workers,
                signal_cache=cache,
                jac_a=jac_a,
                oracle_gn=oracle_gn,
                deploy_gn=deploy_gn,
                gn_steps=gn_steps,
            )
            x_hat = np.asarray(pipe["x_hat"], dtype=np.float64)
            x_grid = np.asarray(pipe["x_grid"], dtype=np.float64)
            x_init = np.asarray(pipe["x_init"], dtype=np.float64)
            d_init = float(np.max(np.abs(x_init - placed)))
            d_grid = float(np.max(np.abs(x_grid - x_oracle)))
            d_hat = float(np.max(np.abs(x_hat - x_oracle)))
            d_truth = float(np.max(np.abs(x_hat - placed)))
            rows.append(
                {
                    "trial": i,
                    "x_init": x_init.tolist(),
                    "x_grid": x_grid.tolist(),
                    "x_hat": x_hat.tolist(),
                    "x_oracle": x_oracle.tolist(),
                    "gn_mode": pipe.get("gn_mode"),
                    "max_abs_init_vs_placed_m": d_init,
                    "max_abs_grid_vs_oracle_m": d_grid,
                    "max_abs_hat_vs_oracle_m": d_hat,
                    "max_abs_hat_vs_placed_m": d_truth,
                    "n_forwards": pipe["n_forwards"],
                    "rss_grid": pipe["rss_grid"],
                    "rss_after_gn": pipe.get("rss_after_gn"),
                    "sigma": float(sigma),
                }
            )
            print(
                f"    trial {i+1}/{n_mc} done  |init|={d_init:.2f}  "
                f"|grid-oracle|={d_grid:.3f}  |hat-oracle|={d_hat:.3f}  "
                f"gn={pipe.get('gn_mode')}  cache={len(cache)}",
                flush=True,
            )

        hats = np.array([r["max_abs_hat_vs_oracle_m"] for r in rows])
        grids = np.array([r["max_abs_grid_vs_oracle_m"] for r in rows])
        inits = np.array([r["max_abs_init_vs_placed_m"] for r in rows])
        med_hat = float(np.median(hats))
        med_grid = float(np.median(grids))
        block = {
            "snr_db": snr_label,
            "radius_cells": radius,
            "crb_scale_m": crb_scale,
            "dx_m": dx,
            "median_init_vs_placed_m": float(np.median(inits)),
            "median_grid_vs_oracle_m": med_grid,
            "median_hat_vs_oracle_m": med_hat,
            "mean_hat_vs_oracle_m": float(np.mean(hats)),
            "ratio_to_crb": float(med_hat / crb_scale) if crb_scale > 0 else float("inf"),
            "pass_strict_lt_crb": bool(med_hat < crb_scale),
            "pass_lt_one_cell": bool(med_hat < dx),
            "pass_advisory": bool(med_hat < max(2.0 * crb_scale, 0.5 * dx)),
            "cache_size": len(cache),
            "trials": rows,
        }
        blocks.append(block)
        print(
            f"  → SNR={snr_label}: med|hat-oracle|={med_hat:.4f} m  "
            f"med|grid-oracle|={med_grid:.4f} m  "
            f"med|init|={block['median_init_vs_placed_m']:.2f} m  "
            f"strict={block['pass_strict_lt_crb']} cell={block['pass_lt_one_cell']}",
            flush=True,
        )

    return {
        "case": case_name,
        "init": init,
        "placed_x": placed.tolist(),
        "radius_cells": radius,
        "prior_uncertainty_m": prior_m,
        "oracle_gn": oracle_gn,
        "deploy_gn": deploy_gn,
        "gn_steps": gn_steps,
        "n_mc": n_mc,
        "snr_blocks": blocks,
        "pass_all_cell": all(b["pass_lt_one_cell"] for b in blocks),
        "pass_all_strict": all(b["pass_strict_lt_crb"] for b in blocks),
    }

def main():
    if sys.platform.startswith("win") and hasattr(sys.stdout, "buffer") and not sys.stdout.closed:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="MOC-primary localization pipeline")
    p.add_argument("--tag", default="smoke")
    p.add_argument("--cases", default="single_4000")
    p.add_argument("--init", default="cepstrum", choices=("cepstrum", "p0", "midwell"))
    p.add_argument("--snr", default="40")
    p.add_argument("--n-mc", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--radius-min", type=int, default=8)
    p.add_argument("--radius-max", type=int, default=50)
    p.add_argument(
        "--prior-uncertainty-m",
        type=float,
        default=0.0,
        help="粗初值不确定度量级；<=0 时按缝数自动（单15 / 双55）",
    )
    p.add_argument("--margin-cells", type=int, default=3)
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument(
        "--oracle-gn",
        action="store_true",
        help="用真值处 Jacobian 做一步 GN（仅诊断；非部署）",
    )
    p.add_argument(
        "--deploy-gn",
        action="store_true",
        help="部署版：在坐标下降格点 x_grid 处重算 J 再做一步 GN",
    )
    p.add_argument("--gn-steps", type=int, default=1)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    if args.oracle_gn and args.deploy_gn:
        raise SystemExit("勿同时开启 --oracle-gn 与 --deploy-gn")

    out = Path(args.out_dir) if args.out_dir else Path(
        f"output/analysis/identifiability/moc_pipeline/{args.tag}"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MOC-primary pipeline  (FNO/cepstrum = init only)")
    print(
        f"  cases={args.cases}  init={args.init}  snr={args.snr}  n_mc={args.n_mc}  "
        f"deploy_gn={args.deploy_gn}  oracle_gn={args.oracle_gn}"
    )
    print("=" * 72)

    results = []
    t_all = time.time()
    for name in _parse_list(args.cases):
        r = run_case(
            name,
            init=args.init,
            snr_list=_parse_snr(args.snr),
            n_mc=args.n_mc,
            seed=args.seed,
            workers=args.workers,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            margin_cells=args.margin_cells,
            prior_uncertainty_m=args.prior_uncertainty_m,
            n_rounds=args.n_rounds,
            oracle_gn=args.oracle_gn,
            deploy_gn=args.deploy_gn,
            gn_steps=args.gn_steps,
        )
        results.append(r)
        (out / f"{name}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")

    summary = {
        "tag": args.tag,
        "init": args.init,
        "strategy": "MOC primary; cepstrum/FNO auxiliary init only",
        "n_mc": args.n_mc,
        "oracle_gn": args.oracle_gn,
        "deploy_gn": args.deploy_gn,
        "gn_steps": args.gn_steps,
        "pass_all_cell": all(r["pass_all_cell"] for r in results),
        "pass_all_strict": all(r["pass_all_strict"] for r in results),
        "results": [
            {
                "case": r["case"],
                "radius_cells": r["radius_cells"],
                "deploy_gn": r["deploy_gn"],
                "pass_all_cell": r["pass_all_cell"],
                "pass_all_strict": r["pass_all_strict"],
                "snr_summary": [
                    {
                        "snr_db": b["snr_db"],
                        "median_init_vs_placed_m": b["median_init_vs_placed_m"],
                        "median_grid_vs_oracle_m": b["median_grid_vs_oracle_m"],
                        "median_hat_vs_oracle_m": b["median_hat_vs_oracle_m"],
                        "ratio_to_crb": b["ratio_to_crb"],
                        "pass_lt_one_cell": b["pass_lt_one_cell"],
                        "pass_strict_lt_crb": b["pass_strict_lt_crb"],
                    }
                    for b in r["snr_blocks"]
                ],
            }
            for r in results
        ],
        "wall_s": float(time.time() - t_all),
    }
    (out / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
