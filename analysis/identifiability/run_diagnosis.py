# -*- coding: utf-8 -*-
"""
run_diagnosis.py — 把「信息下界」与「模型误差偏差」放在同一张表上比较。

产出的核心判断
--------------
对每个 (摩阻, 间距, 带宽, SNR)：
  - CRB_std(x)            噪声允许的最好精度
  - bias(delta_a)         波速标定误差引起的系统偏差
  - bias(delta_k)         Brunone 系数标定误差引起的系统偏差
  - bias(模型形式)        用 steady 模型拟合 brunone 数据引起的偏差
  - 临界 delta            使 bias = CRB_std 的模型误差幅度
  - absorbed_frac         模型误差被参数吸收的比例（越接近 1 越危险）

若 bias >> CRB_std，则该反问题是**偏差受限**而非**信息受限**，
提高算法/网络能力不会有收益，必须先解决模型标定。

用法
----
    python -m analysis.identifiability.run_diagnosis \
        --spacings 5,20,50 --fc 10,20 --snr 10,20,30,40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import (  # noqa: E402
    ObsModel,
    Scenario,
    assemble_jacobian,
    build_jobs,
    crb_report,
    observe,
    run_jobs,
)
from analysis.identifiability.misspecification import (  # noqa: E402
    bias_from_residual,
    critical_nuisance_error,
    nuisance_bias,
)

OUT_DEFAULT = os.path.join("output", "analysis", "identifiability", "diagnosis")


def _run_forwards(scn: Scenario, include, workers: int, grid_step: int,
                  log_step: float, a_cell_step: int):
    scn2, names, jobs = build_jobs(
        scn, include=include, grid_step=grid_step,
        log_step=log_step, a_cell_step=a_cell_step,
    )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            raws = run_jobs(scn2, jobs, map_fn=ex.map)
    else:
        raws = run_jobs(scn2, jobs)
    return scn2, names, jobs, raws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacings", default="5,20,50")
    ap.add_argument("--x1", type=float, default=4000.0)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--tf", type=float, default=20.0)
    ap.add_argument("--fc", default="10,20")
    ap.add_argument("--snr", default="10,20,30,40")
    ap.add_argument("--Cf", type=float, default=1.0e-5)
    ap.add_argument("--kleak", type=float, default=1.0e-4)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    spacings = [float(s) for s in args.spacings.split(",")]
    fcs = [float(s) for s in args.fc.split(",")]
    snrs = [float(s) for s in args.snr.split(",")]
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    # 拟合时会被估计的参数（自由）；a 与 log_kbr 视为被固定的讨厌参数
    FREE = ["x0", "x1", "log_cf0", "log_cf1", "log_kl0", "log_kl1"]
    TARGETS = ["x0", "x1"]

    print("=" * 90)
    print(f"可辨识性诊断  dt={args.dt:g}s tf={args.tf}s x1={args.x1}m  "
          f"fc={fcs}Hz  SNR={snrs}dB")
    print(f"自由参数 {FREE}；讨厌参数 a / log_kbr")
    print("=" * 90)

    rows: List[Dict[str, object]] = []
    t0 = time.time()

    for spacing in spacings:
        xs = (args.x1, args.x1 + spacing)
        cache: Dict[str, Dict] = {}
        for friction in ("steady", "brunone"):
            include = (("x", "cf", "kleak", "a", "kbr") if friction == "brunone"
                       else ("x", "cf", "kleak", "a"))
            scn = Scenario(x_f=xs, Cf=(args.Cf,) * 2, kleak=(args.kleak,) * 2,
                           dt=args.dt, tf=args.tf, friction=friction)
            scn2, names, jobs, raws = _run_forwards(
                scn, include, args.workers, 1, 0.02, 1)
            cache[friction] = {"scn": scn2, "names": names,
                               "jobs": jobs, "raws": raws}
            print(f"[D={spacing:g} m  {friction}] {len(jobs)} 次正演完成 "
                  f"({time.time() - t0:.0f}s)")

        for friction in ("steady", "brunone"):
            c = cache[friction]
            scn2, names, jobs, raws = c["scn"], c["names"], c["jobs"], c["raws"]
            for fc in fcs:
                obs = ObsModel(fc_hz=fc)
                jac = assemble_jacobian(scn2, names, jobs, raws, obs)
                J, s0 = np.asarray(jac["J"]), np.asarray(jac["s0"])

                # 模型形式误差残差：真值为 brunone，拟合模型为 steady（或反之）
                other = "brunone" if friction == "steady" else "steady"
                oc = cache[other]
                s_other = observe(np.asarray(oc["raws"][0]["H_wh"]),
                                  np.asarray(oc["raws"][0]["t"]), oc["scn"], obs)
                form_res = s_other - s0
                form = bias_from_residual(J, names, form_res, FREE)

                # 讨厌参数偏差（探针幅度）
                probes = {"a_rel_1pct": ("a", 0.01 * float(jac["diagnostics"]["a0"]))}
                if "log_kbr" in names:
                    probes["kbr_rel_10pct"] = ("log_kbr", np.log10(1.10))
                biases = {k: nuisance_bias(jac, nuisance=n, delta=d, free=FREE)
                          for k, (n, d) in probes.items()}

                for snr in snrs:
                    rep = crb_report(jac, snr, nuisance=("a", "log_kbr"))
                    crit_a = critical_nuisance_error(
                        jac, rep, nuisance="a", free=FREE, targets=TARGETS)
                    row: Dict[str, object] = {
                        "friction": friction, "spacing_m": spacing,
                        "fc_hz": fc, "snr_db": snr,
                        "sigma_m": rep["sigma"],
                        "crb_std_x0_m": rep["std_no_nuisance"][names.index("x0")],
                        "crb_std_x0_joint_m": rep["std_full"][names.index("x0")],
                        "form_bias_x0_m": form["bias"]["x0"],
                        "form_bias_x1_m": form["bias"]["x1"],
                        "form_absorbed_frac": form["absorbed_frac"],
                        "form_residual_rms_m": form["residual_norm"] / np.sqrt(s0.size),
                        "crit_da_over_a": crit_a["x0"] / float(jac["diagnostics"]["a0"]),
                    }
                    for k, b in biases.items():
                        row[f"bias_x0_{k}_m"] = b["bias"]["x0"]
                        row[f"absorbed_{k}"] = b["absorbed_frac"]
                    row["bias_over_crb_a1pct"] = (
                        abs(row["bias_x0_a_rel_1pct_m"]) / row["crb_std_x0_m"]
                    )
                    row["bias_over_crb_form"] = (
                        abs(row["form_bias_x0_m"]) / row["crb_std_x0_m"]
                    )
                    rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(out_root, "diagnosis_table.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    # ---- 控制台摘要 ----
    print("\n" + "=" * 108)
    print("摘要：位置 x0 的噪声下界 vs 模型误差偏差 [m]")
    print("=" * 108)
    hdr = (f"{'摩阻':<8}{'D[m]':>6}{'fc':>5}{'SNR':>5}"
           f"{'CRB(a已知)':>12}{'CRB(联合a)':>12}"
           f"{'偏差|δa/a=1%':>14}{'偏差|δk/k=10%':>15}{'偏差|摩阻模型错':>16}"
           f"{'临界δa/a':>11}")
    print(hdr)
    for _, r in df.iterrows():
        kb = r.get("bias_x0_kbr_rel_10pct_m", np.nan)
        print(f"{r['friction']:<8}{r['spacing_m']:>6.0f}{r['fc_hz']:>5.0f}"
              f"{r['snr_db']:>5.0f}"
              f"{r['crb_std_x0_m']:>12.3e}{r['crb_std_x0_joint_m']:>12.3e}"
              f"{r['bias_x0_a_rel_1pct_m']:>14.3e}"
              f"{(kb if kb == kb else float('nan')):>15.3e}"
              f"{r['form_bias_x0_m']:>16.3e}"
              f"{r['crit_da_over_a']:>11.2e}")

    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dt_s": args.dt, "tf_s": args.tf, "x1_m": args.x1,
            "spacings_m": spacings, "fc_hz": fcs, "snr_db": snrs,
            "free_params": FREE, "nuisance": ["a", "log_kbr"],
            "Cf": args.Cf, "kleak": args.kleak,
            "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"\n写出 {len(df)} 行 → {csv}   总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
