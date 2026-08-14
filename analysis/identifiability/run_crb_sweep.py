# -*- coding: utf-8 -*-
"""
run_crb_sweep.py — 井筒多裂缝反问题的 CRB 扫描。

结构
----
昂贵的部分（MOC 正演）与廉价的部分（观测算子 + 矩阵求逆）分离：
  1. 对每个物理构型跑一次中心差分所需的全部正演，波形缓存到 npz；
  2. 在缓存上遍历 (传感器带宽 fc) × (SNR)，组装 Jacobian 并求 CRB。
因此增加 SNR 或带宽取值不需要重新仿真。

数值可信域（见 check_convergence.py）
------------------------------------
位置差分步长受网格约束 h = dx = a*dt，收敛要求 fc*dt <~ 0.02。
脚本会对每个 (fc, dt) 组合检查该判据，不满足时在输出中标记 converged=False。

用法
----
    # 快速构型（dt=1ms，fc<=20Hz）
    python -m analysis.identifiability.run_crb_sweep --tag quick \
        --dt 1e-3 --fc 5,10,20 --spacings 5,10,20,50,100 --friction steady

    # 高带宽构型（dt=2e-4，fc<=100Hz），仅关键间距
    python -m analysis.identifiability.run_crb_sweep --tag hifreq \
        --dt 2e-4 --fc 20,50,100 --spacings 5,10,20 --friction steady
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import (  # noqa: E402
    ObsModel,
    Scenario,
    assemble_jacobian,
    build_jobs,
    crb_report,
    linear_functional_std,
    run_jobs,
)

DEFAULT_OUT = os.path.join("output", "analysis", "identifiability")

# 差分收敛判据：fc * dt 的上限（由 check_convergence.py 标定）
FC_DT_LIMIT = 0.021


def _cache_path(root: str, key: str) -> str:
    return os.path.join(root, "raw_cache", f"{key}.npz")


def _config_key(friction: str, spacing: float, x1: float, dt: float, tf: float) -> str:
    return (f"{friction}_D{spacing:g}_x{x1:g}_dt{dt:g}_tf{tf:g}"
            .replace(".", "p").replace("-", "m"))


def simulate_config(
    scn: Scenario,
    out_root: str,
    key: str,
    *,
    workers: int,
    grid_step: int,
    log_step: float,
    a_cell_step: int,
    force: bool = False,
):
    """跑（或读取缓存）一个构型的全部差分正演。"""
    scn2, names, jobs = build_jobs(
        scn, grid_step=grid_step, log_step=log_step, a_cell_step=a_cell_step,
    )
    path = _cache_path(out_root, key)
    if os.path.exists(path) and not force:
        blob = np.load(path, allow_pickle=False)
        raws = [
            {
                "H_wh": blob[f"H{i}"],
                "t": blob["t"],
                "a_adj": float(blob["a_adj"][i]),
                "placed_x": blob["placed_x"][i],
            }
            for i in range(len(jobs))
        ]
        return scn2, names, jobs, raws, True

    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            raws = run_jobs(scn2, jobs, map_fn=ex.map)
    else:
        raws = run_jobs(scn2, jobs)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {f"H{i}": np.asarray(r["H_wh"], dtype=np.float64)
               for i, r in enumerate(raws)}
    payload["t"] = np.asarray(raws[0]["t"], dtype=np.float64)
    payload["a_adj"] = np.array([r["a_adj"] for r in raws], dtype=np.float64)
    payload["placed_x"] = np.array([r["placed_x"] for r in raws], dtype=np.float64)
    np.savez_compressed(path, **payload)
    print(f"    {len(jobs)} 次正演耗时 {elapsed:.1f}s → {os.path.basename(path)}")
    return scn2, names, jobs, raws, False


def analyze_config(
    scn: Scenario,
    names: List[str],
    jobs,
    raws,
    *,
    friction: str,
    spacing: float,
    x1: float,
    fcs: List[float],
    snrs: List[float],
    grid_step: int,
    log_step: float,
) -> List[Dict[str, object]]:
    """在缓存波形上遍历 (fc, SNR)，输出 CRB 行。"""
    rows: List[Dict[str, object]] = []
    n = scn.n_frac
    idx_x = [names.index(f"x{i}") for i in range(n)]

    # 线性泛函：缝间距（差模）与缝群质心（共模）
    c_sep = np.zeros(len(names)); c_sep[idx_x[0]] = -1.0; c_sep[idx_x[-1]] = 1.0
    c_cen = np.zeros(len(names))
    for j in idx_x:
        c_cen[j] = 1.0 / n

    for fc in fcs:
        obs = ObsModel(fc_hz=fc)
        jac = assemble_jacobian(scn, names, jobs, raws, obs,
                                grid_step=grid_step, log_step=log_step)
        diag = jac["diagnostics"]
        converged = (fc * scn.dt) <= FC_DT_LIMIT
        for snr in snrs:
            rep = crb_report(jac, snr, nuisance=("a",))
            row: Dict[str, object] = {
                "friction": friction,
                "spacing_nominal_m": spacing,
                "spacing_placed_m": float(
                    np.asarray(diag["placed_x0"])[-1] - np.asarray(diag["placed_x0"])[0]
                ),
                "x1_m": x1,
                "n_frac": n,
                "dt_s": scn.dt,
                "tf_s": scn.tf,
                "fc_hz": fc,
                "fc_times_dt": fc * scn.dt,
                "fd_converged": converged,
                "snr_db": snr,
                "sigma_m": rep["sigma"],
                "n_obs": rep["n_obs"],
                "cond_full": rep["cond_normalized"],
                "cond_a_known": rep["cond_no_nuisance"],
                "weakest_eigval": rep["weakest_eigval"],
                "a_snap_corr_rel": diag.get("a_snap_correction_rel", np.nan),
                "a_rel_step": diag.get("a_rel_step", np.nan),
                "a_phase_number": diag.get("a_phase_number", np.nan),
            }
            for i in range(n):
                j = idx_x[i]
                row[f"std_x{i}_full_m"] = rep["std_full"][j]
                row[f"std_x{i}_a_known_m"] = rep["std_no_nuisance"][j]
                row[f"std_x{i}_alone_m"] = rep["std_alone"][j]
            row["std_a_full_mps"] = rep["std_full"][names.index("a")]
            row["std_spacing_full_m"] = linear_functional_std(rep, c_sep)
            row["std_spacing_a_known_m"] = linear_functional_std(
                rep, c_sep, which="no_nuisance")
            row["std_centroid_full_m"] = linear_functional_std(rep, c_cen)
            row["std_centroid_a_known_m"] = linear_functional_std(
                rep, c_cen, which="no_nuisance")
            row["a_penalty_x0"] = (
                rep["std_full"][idx_x[0]] / rep["std_no_nuisance"][idx_x[0]]
            )
            row["a_penalty_spacing"] = (
                row["std_spacing_full_m"] / row["std_spacing_a_known_m"]
                if row["std_spacing_a_known_m"] > 0 else np.nan
            )
            row["weakest_direction"] = json.dumps(
                {nm: round(float(v), 5)
                 for nm, v in zip(names, rep["weakest_direction"])},
                ensure_ascii=False,
            )
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", help="输出子目录名")
    ap.add_argument("--friction", default="steady",
                    help="逗号分隔：steady,brunone")
    ap.add_argument("--spacings", default="5,10,20,50,100",
                    help="逗号分隔缝间距 [m]")
    ap.add_argument("--x1", type=float, default=4000.0, help="首缝深度 [m]")
    ap.add_argument("--n-frac", type=int, default=2)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--tf", type=float, default=20.0)
    ap.add_argument("--fc", default="5,10,20", help="逗号分隔传感器带宽 [Hz]")
    ap.add_argument("--snr", default="10,20,30,40,60",
                    help="逗号分隔 SNR [dB]")
    ap.add_argument("--Cf", type=float, default=1.0e-5)
    ap.add_argument("--kleak", type=float, default=1.0e-4)
    ap.add_argument("--grid-step", type=int, default=1)
    ap.add_argument("--log-step", type=float, default=0.02)
    ap.add_argument("--a-cell-step", type=int, default=1)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="忽略缓存重跑正演")
    args = ap.parse_args()

    frictions = [s.strip() for s in args.friction.split(",") if s.strip()]
    spacings = [float(s) for s in args.spacings.split(",")]
    fcs = [float(s) for s in args.fc.split(",")]
    snrs = [float(s) for s in args.snr.split(",")]

    out_root = os.path.abspath(os.path.join(args.out, args.tag))
    os.makedirs(out_root, exist_ok=True)

    bad_fc = [fc for fc in fcs if fc * args.dt > FC_DT_LIMIT]
    print("=" * 84)
    print(f"CRB 扫描  tag={args.tag}  dt={args.dt:g}s  tf={args.tf}s  "
          f"n_frac={args.n_frac}  x1={args.x1}m")
    print(f"  摩阻   : {frictions}")
    print(f"  间距   : {spacings} m")
    print(f"  带宽   : {fcs} Hz   (差分收敛上限 fc <= {FC_DT_LIMIT / args.dt:.0f} Hz)")
    print(f"  SNR    : {snrs} dB")
    print(f"  输出   : {out_root}")
    if bad_fc:
        print(f"  [警告] fc={bad_fc} 超出差分收敛域，结果行将标记 fd_converged=False")
    print("=" * 84)

    all_rows: List[Dict[str, object]] = []
    t_start = time.time()
    for friction in frictions:
        for spacing in spacings:
            key = _config_key(friction, spacing, args.x1, args.dt, args.tf)
            print(f"\n[{friction}  D={spacing:g} m]  {key}")
            xs = tuple(args.x1 + i * spacing for i in range(args.n_frac))
            scn = Scenario(
                x_f=xs,
                Cf=tuple([args.Cf] * args.n_frac),
                kleak=tuple([args.kleak] * args.n_frac),
                dt=args.dt, tf=args.tf, friction=friction,
            )
            try:
                scn2, names, jobs, raws, cached = simulate_config(
                    scn, out_root, key, workers=args.workers,
                    grid_step=args.grid_step, log_step=args.log_step,
                    a_cell_step=args.a_cell_step, force=args.force,
                )
            except ValueError as exc:
                print(f"    跳过：{exc}")
                continue
            if cached:
                print("    命中缓存")
            rows = analyze_config(
                scn2, names, jobs, raws,
                friction=friction, spacing=spacing, x1=args.x1,
                fcs=fcs, snrs=snrs,
                grid_step=args.grid_step, log_step=args.log_step,
            )
            all_rows += rows
            r = [x for x in rows if x["snr_db"] == max(snrs) and x["fc_hz"] == max(fcs)]
            if r:
                r = r[0]
                print(f"    placed spacing = {r['spacing_placed_m']:.3f} m | "
                      f"@fc={r['fc_hz']:g}Hz SNR={r['snr_db']:g}dB: "
                      f"std(x0)={r['std_x0_full_m']:.3e} m, "
                      f"std(间距)={r['std_spacing_full_m']:.3e} m, "
                      f"波速代价 x0 ×{r['a_penalty_x0']:.1f}")

    if not all_rows:
        print("\n没有产出任何结果行。")
        return

    import pandas as pd
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_root, "crb_table.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    meta = {
        "tag": args.tag,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dt_s": args.dt, "tf_s": args.tf, "x1_m": args.x1,
        "n_frac": args.n_frac, "Cf": args.Cf, "kleak": args.kleak,
        "frictions": frictions, "spacings_m": spacings,
        "fc_hz": fcs, "snr_db": snrs,
        "grid_step": args.grid_step, "log_step": args.log_step,
        "a_cell_step": args.a_cell_step,
        "fd_convergence_limit_fc_dt": FC_DT_LIMIT,
        "fc_max_hz_for_dt": FC_DT_LIMIT / args.dt,
        "n_rows": len(df),
        "elapsed_s": round(time.time() - t_start, 1),
        "note": ("观测模型：停泵后 H_wh 经零相位 Butterworth(4阶) 限带 + 白噪声；"
                 "SNR 按停泵后观测段去均值方差定义，与 stratified_bench.apply_awgn 一致。"),
    }
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"\n写出 {len(df)} 行 → {csv_path}")
    print(f"总耗时 {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
