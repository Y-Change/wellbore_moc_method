# -*- coding: utf-8 -*-
"""
run_mismatch_mle.py — 模型失配下的 MLE 偏差实测，并与一阶线性化预测对比。

协议
----
1. 真数据：brunone（或 steady）正演 + 可选噪声
2. 拟合模型：另一摩阻形式（steady ↔ brunone）
3. 用拟合模型的位置网格做 MLE-A，测量位置偏差
4. 与 misspecification.bias_from_residual 的一阶预测对比
5. 再把 brunone_k_scale 纳入估计（自由参数），看偏差能否被吸收

用法
----
    python -m analysis.identifiability.run_mismatch_mle --tag main \\
        --case single_4000 --radius 20 --workers 14
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import (  # noqa: E402
    forward,
    observe,
)
from analysis.identifiability.misspecification import bias_from_residual  # noqa: E402
from analysis.identifiability.mle import (  # noqa: E402
    build_position_cache,
    jacobian_at_truth,
    match_estimate_to_truth,
    mle_from_cache,
    mle_refine_gn,
)
from analysis.identifiability.scenarios import (  # noqa: E402
    FC_DEFAULT,
    case_by_name,
    make_obs,
    make_scenario,
    scenario_summary,
)

DEFAULT_OUT = os.path.join("output", "analysis", "identifiability", "mismatch_mle")


def _truth_observation(scn, obs):
    res = forward(scn)
    s = observe(res.H_wh, res.t, scn, obs)
    return res, s


def run_pair(
    case_name: str,
    *,
    truth_friction: str,
    fit_friction: str,
    fc: float,
    radius: int,
    workers: int,
    gn_steps: int,
    snr_db: Optional[float],
    seed: int,
) -> Dict[str, object]:
    case = case_by_name(case_name)
    obs = make_obs(fc)
    scn_truth = make_scenario(case, friction=truth_friction)
    scn_fit = make_scenario(case, friction=fit_friction)

    print(f"\n  真模型={truth_friction} → 拟合={fit_friction}")

    # 真观测（无噪声或加噪）
    res_t, s_truth = _truth_observation(scn_truth, obs)
    placed_truth = res_t.placed_x
    y = s_truth.copy()
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        from analysis.identifiability.mle import add_noise
        y, sigma = add_noise(s_truth, snr_db, rng)
    else:
        sigma = 0.0

    # 拟合模型在真值参数处的预测
    res_f, s_fit = _truth_observation(scn_fit, obs)
    residual = y - s_fit  # 模型误差 + 噪声

    # 一阶偏差预测（用拟合模型 Jacobian）
    map_fn = None
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers)
        map_fn = pool.map
    try:
        jac = jacobian_at_truth(scn_fit, obs, mode="A", map_fn=map_fn)
    finally:
        if pool is not None:
            pool.shutdown()

    free = [f"x{i}" for i in range(case.x_f.__len__())]
    # 无噪声时用 s_truth - s_fit；有噪声时用 y - s_fit
    pred = bias_from_residual(
        np.asarray(jac["J"]), jac["names"],
        observe(res_t.H_wh, res_t.t, scn_truth, obs) - s_fit
        if snr_db is None else residual,
        free,
    )

    # 实测：在拟合模型网格上做 MLE
    print(f"    构建拟合模型位置缓存 (radius=±{radius})…")
    t0 = time.time()
    # 缓存场景用拟合摩阻，但位置对齐到真值吸附位置
    scn_fit_aligned = replace(
        jac["scenario"],
        friction=fit_friction,
    )
    cache = build_position_cache(
        scn_fit_aligned, obs, radius=radius, workers=workers,
    )
    print(f"    缓存 {len(cache.signals)} 点 ({time.time()-t0:.1f}s)")

    # 注意：y 的长度必须与 cache 观测一致。真/拟合 tf、dt 相同则长度一致。
    m = min(len(y), len(cache.s_truth))
    y_use = y[:m]

    mle = mle_from_cache(
        cache, y_use, mode="A", gn_steps=gn_steps,
    )
    # 与真值吸附位置比较
    matched = match_estimate_to_truth(mle.x_hat, placed_truth)

    # 联合估 k_scale（仅当真模型是 brunone、拟合也是 brunone 时才有意义；
    # 对 steady↔brunone，把 k_scale 作为拟合 brunone 的自由参数试吸收）
    k_absorb = None
    if fit_friction == "brunone":
        print("    尝试把 k_scale 纳入估计以吸收偏差…")
        # 在真值位置附近，对 (x, log_kbr) 做 GN
        # 需要临时扩展：用 include x + kbr
        scn_k = replace(scn_fit_aligned, k_scale=1.0)
        try:
            refined = mle_refine_gn(
                scn_k, obs, y_use,
                x_init=mle.x_grid,
                mode="A",
                n_steps=max(1, gn_steps),
                include_override=["x", "kbr"],
            )
            mk = match_estimate_to_truth(refined["x_hat"], placed_truth)
            k_absorb = {
                "x_hat": refined["x_hat"].tolist(),
                "errors_m": mk["errors_m"].tolist(),
                "rmse_m": mk["rmse_m"],
                "k_scale_hat": refined.get("k_scale_hat"),
                "rss": refined["rss"],
                "rss_grid": mle.rss_grid,
                "rss_improvement": float(mle.rss_grid - refined["rss"]),
                "history": refined["history"],
                "note": "include=(x, log_kbr)；联合估 k_scale 试图吸收摩阻模型误差",
            }
        except Exception as exc:
            k_absorb = {"error": str(exc)}

    bias_pred = pred["bias"]
    return {
        "truth_friction": truth_friction,
        "fit_friction": fit_friction,
        "placed_truth": placed_truth.tolist(),
        "sigma": sigma,
        "snr_db": snr_db,
        "residual_norm": float(np.linalg.norm(
            observe(res_t.H_wh, res_t.t, scn_truth, obs) - s_fit
        )),
        "linearized_bias": bias_pred,
        "absorbed_frac": pred["absorbed_frac"],
        "mle_x_hat": mle.x_hat.tolist(),
        "mle_errors_m": matched["errors_m"].tolist(),
        "mle_rmse_m": matched["rmse_m"],
        "mle_bias_m": matched["errors_m"].tolist(),
        "prediction_vs_measured": {
            name: {
                "predicted": bias_pred.get(name),
                "measured": float(matched["errors_m"][int(name[1:])])
                if name.startswith("x") and int(name[1:]) < len(matched["errors_m"])
                else None,
            }
            for name in free
        },
        "k_scale_absorption": k_absorb,
        "scenario_truth": scenario_summary(scn_truth),
        "scenario_fit": scenario_summary(scn_fit_aligned),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--case", default="single_4000")
    ap.add_argument("--fc", type=float, default=FC_DEFAULT)
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--gn-steps", type=int, default=1)
    ap.add_argument("--snr", default="inf", help="inf 或 dB 值")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    snr_db: Optional[float]
    if args.snr.strip().lower() in ("inf", "infty", "none"):
        snr_db = None
    else:
        snr_db = float(args.snr)

    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("brunone", "steady"),
        ("steady", "brunone"),
        ("brunone", "brunone"),  # 匹配对照
    ]

    print("=" * 72)
    print(f"模型失配 MLE  tag={args.tag}  case={args.case}  SNR={args.snr}")
    print("=" * 72)

    all_results = []
    t0 = time.time()
    for truth_f, fit_f in pairs:
        r = run_pair(
            args.case,
            truth_friction=truth_f,
            fit_friction=fit_f,
            fc=args.fc,
            radius=args.radius,
            workers=args.workers,
            gn_steps=args.gn_steps,
            snr_db=snr_db,
            seed=args.seed,
        )
        all_results.append(r)
        print(f"    实测 RMSE={r['mle_rmse_m']:.4g} m  "
              f"一阶预测 bias_x0={r['linearized_bias'].get('x0')}  "
              f"absorbed={r['absorbed_frac']:.3f}")

    payload = {
        "case": args.case,
        "fc_hz": args.fc,
        "radius": args.radius,
        "snr_db": snr_db,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "pairs": all_results,
    }

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and np.isnan(o):
            return None
        raise TypeError(type(o))

    path = out_dir / f"{args.case}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_default)
    print(f"\n→ {path}  ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
