# -*- coding: utf-8 -*-
"""
run_highorder_efficiency.py — n=3/4/5 全波形 MLE 效率比（P2b）。

协议与 EXP-003 一致：真值处 Jacobian + 一步 GN 蒙特卡洛。
另对每个场景做无噪声坐标下降点检，确认优化器落在真值邻域。

验收：效率比 RMSE/CRB ≤ 3；超标如实记录原因。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.mle import (  # noqa: E402
    add_noise,
    jacobian_at_truth,
    match_estimate_to_truth,
    mle_grid_coord_descent,
    one_step_from_jacobian,
)
from analysis.identifiability.run_efficiency import (  # noqa: E402
    compute_crb_baselines,
)
from analysis.identifiability.scenarios import (  # noqa: E402
    case_by_name,
    highorder_cases,
    make_obs,
    make_scenario,
    scenario_summary,
)
from analysis.identifiability.crb_core import forward, observe  # noqa: E402

DEFAULT_OUT = os.path.join("output", "analysis", "identifiability", "highorder")
EFF_GATE = 3.0


def _parse_snr(s: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for part in s.split(","):
        part = part.strip().lower()
        if part in ("inf", "infty", "infinity", "none"):
            out.append(None)
        else:
            out.append(float(part))
    return out


def _snr_label(snr: Optional[float]) -> str:
    return "inf" if snr is None else f"{snr:g}"


def _summarize_errors(errs: List[np.ndarray]) -> Dict[str, object]:
    if not errs:
        return {
            "rmse_per_frac_m": [],
            "rmse_m": float("nan"),
            "n_mc": 0,
        }
    E = np.asarray(errs, dtype=np.float64)
    with np.errstate(all="ignore"):
        rmse_per = np.sqrt(np.nanmean(E ** 2, axis=0))
        rmse_all = float(np.sqrt(np.nanmean(E ** 2)))
    return {
        "rmse_per_frac_m": np.where(np.isfinite(rmse_per), rmse_per, np.nan).tolist(),
        "rmse_m": rmse_all if np.isfinite(rmse_all) else float("nan"),
        "n_mc": int(E.shape[0]),
    }


def run_case(
    case_name: str,
    *,
    snrs: List[Optional[float]],
    n_mc: int,
    seed: int,
    workers: int,
    radius_cd: int,
    do_coord_check: bool,
) -> Dict[str, object]:
    case = case_by_name(case_name)
    scn = make_scenario(case)
    obs = make_obs()
    print(f"\n===== {case_name} n={len(case.x_f)} spacing={case.spacing_m} =====")
    print("  CRB / Jacobian ...")
    t0 = time.time()
    crb = compute_crb_baselines(scn, obs, workers=workers)
    print(f"  CRB done in {time.time() - t0:.1f}s")

    # 真值观测
    res = forward(scn)
    s0 = observe(res.H_wh, res.t, scn, obs)
    x_true = np.asarray(crb["placed_x"], dtype=np.float64)

    coord_check = None
    if do_coord_check:
        print("  noise-free coord-descent spot check ...")
        t1 = time.time()
        # 故意从偏移起点出发，避免 trivial
        x_perturb = x_true + np.array(
            [(-1) ** i * min(3.0, case.spacing_m * 0.3) for i in range(len(x_true))]
        )
        cd = mle_grid_coord_descent(
            scn, obs, s0,
            radius=radius_cd, n_rounds=3, workers=workers,
            x_init=x_perturb,
        )
        x_cd = np.asarray(cd["x_hat"], dtype=np.float64)
        mk = match_estimate_to_truth(x_cd, x_true)
        coord_check = {
            "x_init": x_perturb.tolist(),
            "x_hat": x_cd.tolist(),
            "rmse_m": mk["rmse_m"],
            "errors_m": mk["errors_m"].tolist(),
            "n_forwards": cd["n_forwards"],
            "t_s": time.time() - t1,
            "within_half_dx": bool(mk["rmse_m"] < 0.5 * float(scn.dx_for(scn.n_cells_nominal()))),
        }
        print(
            f"  coord check RMSE={mk['rmse_m']:.4f} m, "
            f"forwards={cd['n_forwards']}, t={coord_check['t_s']:.1f}s"
        )

    jac_a = crb["jac_a"]
    scale_std = crb["scale_std"]
    std_x_ref = np.asarray(crb["std_x_a_ref"], dtype=np.float64)
    crb_mean_ref = float(np.mean(std_x_ref))

    by_snr = {}
    rng = np.random.default_rng(seed)
    for snr in snrs:
        label = _snr_label(snr)
        errs = []
        for _ in range(n_mc):
            y, _sig = add_noise(s0, snr, rng)
            if snr is None:
                # 无噪声：一步应为 0
                step = {"x_hat": x_true.copy()}
            else:
                step = one_step_from_jacobian(jac_a, y, s_ref=s0, x_ref=x_true)
            mk = match_estimate_to_truth(step["x_hat"], x_true)
            errs.append(np.asarray(mk["errors_m"], dtype=np.float64))
        summ = _summarize_errors(errs)
        crb_rmse = scale_std(crb_mean_ref, snr)
        rmse = float(summ["rmse_m"])
        if snr is None:
            eff = float("nan") if rmse < 1e-9 else float("inf")
        else:
            eff = rmse / crb_rmse if crb_rmse > 0 else float("nan")
        by_snr[label] = {
            "rmse_m": rmse,
            "crb_rmse_m": crb_rmse,
            "efficiency_ratio": eff,
            "rmse_per_frac_m": summ["rmse_per_frac_m"],
            "n_mc": summ["n_mc"],
            "pass_gate": bool(snr is None or (np.isfinite(eff) and eff <= EFF_GATE)),
        }
        print(
            f"  SNR={label}: RMSE={rmse:.4f} CRB={crb_rmse:.4f} "
            f"eff={eff if np.isfinite(eff) else 'nan'} "
            f"gate={'PASS' if by_snr[label]['pass_gate'] else 'FAIL'}"
        )

    # JSON 安全：去掉不可序列化
    crb_json = {
        k: v for k, v in crb.items()
        if k not in ("jac_a", "jac_b", "scale_std")
    }
    return {
        "case": case_name,
        "n_frac": len(case.x_f),
        "spacing_m": case.spacing_m,
        "scenario": scenario_summary(scn),
        "crb": crb_json,
        "std_x_a_ref_mean": crb_mean_ref,
        "coord_check": coord_check,
        "by_snr": by_snr,
        "eff_gate": EFF_GATE,
    }


def main():
    p = argparse.ArgumentParser(description="n≥3 MLE 效率比")
    p.add_argument("--tag", default="main")
    p.add_argument(
        "--cases",
        default="triple_10m,triple_40m,quad_10m,quad_40m,quint_10m,quint_40m",
    )
    p.add_argument("--snr", default="40,30,20")
    p.add_argument("--n-mc", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--radius-cd", type=int, default=8)
    p.add_argument("--skip-coord-check", action="store_true")
    args = p.parse_args()

    out_dir = Path(DEFAULT_OUT) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    snrs = _parse_snr(args.snr)
    names = [c.strip() for c in args.cases.split(",") if c.strip()]

    rows = []
    for name in names:
        row = run_case(
            name,
            snrs=snrs,
            n_mc=args.n_mc,
            seed=args.seed,
            workers=args.workers,
            radius_cd=args.radius_cd,
            do_coord_check=not args.skip_coord_check,
        )
        rows.append(row)
        (out_dir / f"{name}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )

    # 汇总表
    summary_rows = []
    all_pass = True
    for row in rows:
        for snr_label, st in row["by_snr"].items():
            summary_rows.append({
                "case": row["case"],
                "n_frac": row["n_frac"],
                "spacing_m": row["spacing_m"],
                "snr": snr_label,
                "rmse_m": st["rmse_m"],
                "crb_rmse_m": st["crb_rmse_m"],
                "efficiency_ratio": st["efficiency_ratio"],
                "pass_gate": st["pass_gate"],
                "coord_rmse_m": (
                    None if row["coord_check"] is None
                    else row["coord_check"]["rmse_m"]
                ),
            })
            if not st["pass_gate"]:
                all_pass = False

    summary = {
        "tag": args.tag,
        "eff_gate": EFF_GATE,
        "all_passed": all_pass,
        "rows": summary_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # 简图：效率比 vs n
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for spacing in (10.0, 40.0):
            for snr_label in [_snr_label(s) for s in snrs]:
                xs, ys = [], []
                for row in rows:
                    if row["spacing_m"] != spacing:
                        continue
                    st = row["by_snr"].get(snr_label)
                    if st is None:
                        continue
                    xs.append(row["n_frac"])
                    ys.append(st["efficiency_ratio"])
                if xs:
                    ax.plot(xs, ys, "o-", label=f"D={spacing:g}m SNR={snr_label}")
        ax.axhline(EFF_GATE, color="k", ls="--", lw=1, label=f"gate={EFF_GATE}")
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("n_frac")
        ax.set_ylabel("RMSE / CRB")
        ax.set_title("High-order MLE efficiency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / "eff_vs_n.png", dpi=150)
        fig.savefig(fig_dir / "eff_vs_n.svg")
        plt.close(fig)
    except Exception as e:
        print(f"  (plot skipped: {e})")

    print(f"\n写入 {out_dir}")
    print("总体门控:", "PASS" if all_pass else "FAIL")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
