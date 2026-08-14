# -*- coding: utf-8 -*-
"""
run_wavespeed_mismatch.py — ±1% 波速失配对位置偏差的非线性复核（P3b）。

用 seed=42 配对几何，a=1435.5（−1%）生成少量探针 case（默认 20），
在正确 a=1450 模型下做全波形 MLE，对比 EXP-002 一阶预测 28–36 m。

若全量 2000 集已用 run_lhs_batch_simulate 生成，可 --data-dir 指向该目录。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import ObsModel, Scenario, observe  # noqa: E402
from analysis.identifiability.mle import mle_grid_coord_descent, match_estimate_to_truth  # noqa: E402
from analysis.identifiability.scenarios import (  # noqa: E402
    A_WAVE,
    CF_DEFAULT,
    DIA,
    DT,
    H0,
    H_EXT,
    KLEAK_DEFAULT,
    L,
    ROUGH,
    TF,
    TS,
    V0,
    VISC,
)
from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore  # noqa: E402

DEFAULT_OUT = os.path.join("output", "analysis", "identifiability", "wavespeed_mismatch")
A_MIS = 1435.5  # −1%
PRED_BIAS_LO = 28.0
PRED_BIAS_HI = 36.0


def _simulate_case(x_f, Cf, kleak, *, a: float, friction: str = "brunone") -> Dict:
    cfg = MocConfig(
        wellbore_length=L,
        wellbore_diameter=DIA,
        fluid_density=1000.0,
        fluid_viscosity=VISC,
        wavespeed=a,
        roughness_height=ROUGH,
        friction_model=friction,
        dt=DT,
        tf=TF,
        wellhead_bc="velocity_step",
        pump_shut_time=TS,
        initial_velocity=V0,
        initial_head=H0,
        toe_bc="reservoir",
        toe_head=H0,
    )
    res = simulate_wellbore(
        cfg,
        fracture_positions=list(x_f),
        fracture_Cf=list(Cf),
        fracture_kleak=list(kleak),
        H_ext=H_EXT,
        store_full_field=False,
    )
    return {
        "t": res["timestamps"],
        "H_wh": res["wellhead_head"],
        "x_f": np.asarray(x_f, dtype=float),
        "a_true": a,
    }


def main():
    p = argparse.ArgumentParser(description="波速失配偏差复核")
    p.add_argument("--tag", default="probe20")
    p.add_argument("--n-probe", type=int, default=20)
    p.add_argument("--a-mis", type=float, default=A_MIS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--radius", type=int, default=40,
                   help="坐标下降半径（格点）；失配偏差~30m 需覆盖")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(DEFAULT_OUT) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 探针几何：单缝扫深度 + 双缝 10m
    rng = np.random.default_rng(args.seed)
    probes = []
    depths = np.linspace(3600, 4600, args.n_probe // 2)
    for d in depths:
        probes.append({"x_f": (float(d),), "Cf": (CF_DEFAULT,), "kleak": (KLEAK_DEFAULT,), "tag": "single"})
    for _ in range(args.n_probe - len(probes)):
        d0 = float(rng.uniform(3600, 4500))
        probes.append({
            "x_f": (d0, d0 + 10.0),
            "Cf": (CF_DEFAULT, CF_DEFAULT),
            "kleak": (KLEAK_DEFAULT, KLEAK_DEFAULT),
            "tag": "dual10",
        })

    obs = ObsModel(fc_hz=20.0, order=4)
    rows = []
    for i, pr in enumerate(probes):
        print(f"[{i+1}/{len(probes)}] simulate a={args.a_mis} x={pr['x_f']}")
        t0 = time.time()
        data = _simulate_case(pr["x_f"], pr["Cf"], pr["kleak"], a=args.a_mis)
        # 观测用失配数据
        scn_obs = Scenario(
            x_f=pr["x_f"], Cf=pr["Cf"], kleak=pr["kleak"],
            L=L, diameter=DIA, a_nominal=args.a_mis, dt=DT, tf=TF, ts=TS,
            V0=V0, H0=H0, H_ext=H_EXT, toe_bc="reservoir", friction="brunone",
            viscosity=VISC, roughness=ROUGH,
        )
        # observe 需要与 Scenario 一致的时间轴；用简单截取+滤波包装
        from analysis.identifiability.crb_core import observe as obs_fn
        # 构造临时 scn 仅用于 observe 的 ts/tf
        y = obs_fn(data["H_wh"], data["t"], scn_obs, obs)

        # 拟合模型用正确 a=1450
        scn_fit = Scenario(
            x_f=pr["x_f"], Cf=pr["Cf"], kleak=pr["kleak"],
            L=L, diameter=DIA, a_nominal=A_WAVE, dt=DT, tf=TF, ts=TS,
            V0=V0, H0=H0, H_ext=H_EXT, toe_bc="reservoir", friction="brunone",
            viscosity=VISC, roughness=ROUGH,
        )
        cd = mle_grid_coord_descent(
            scn_fit, obs, y,
            radius=args.radius, n_rounds=3, workers=args.workers,
            x_init=pr["x_f"],
        )
        mk = match_estimate_to_truth(cd["x_hat"], pr["x_f"])
        # 共模偏差（平均深度误差）
        bias_mean = float(np.mean(mk["errors_m"]))
        row = {
            "probe": i,
            "tag": pr["tag"],
            "x_true": list(pr["x_f"]),
            "x_hat": cd["x_hat"].tolist(),
            "errors_m": mk["errors_m"].tolist(),
            "bias_mean_m": bias_mean,
            "rmse_m": mk["rmse_m"],
            "rss": cd["rss"],
            "n_forwards": cd["n_forwards"],
            "t_s": time.time() - t0,
            "a_data": args.a_mis,
            "a_model": A_WAVE,
            "a_rel_err": (args.a_mis - A_WAVE) / A_WAVE,
        }
        rows.append(row)
        print(f"  bias_mean={bias_mean:.2f} m RMSE={mk['rmse_m']:.2f} m")

    biases = np.array([r["bias_mean_m"] for r in rows], dtype=float)
    med = float(np.median(np.abs(biases)))
    # 一阶预测量级：取绝对值中位与 [28,36] 比较
    ratio_lo = med / PRED_BIAS_HI
    ratio_hi = med / PRED_BIAS_LO
    # 验收：比值 < 2（与预测区间任一端）
    pass_gate = med < 2.0 * PRED_BIAS_HI

    summary = {
        "tag": args.tag,
        "a_data": args.a_mis,
        "a_model": A_WAVE,
        "n_probe": len(rows),
        "median_abs_bias_m": med,
        "mean_bias_m": float(np.mean(biases)),
        "pred_range_m": [PRED_BIAS_LO, PRED_BIAS_HI],
        "ratio_to_pred_lo": float(med / PRED_BIAS_LO),
        "ratio_to_pred_hi": float(med / PRED_BIAS_HI),
        "pass_gate_ratio_lt2": pass_gate,
        "rows": rows,
    }
    (out_dir / "wavespeed_mismatch.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nmedian |bias| = {med:.2f} m; pred 28–36 m; pass={pass_gate}")
    print(f"写入 {out_dir}")
    if not pass_gate:
        sys.exit(1)


if __name__ == "__main__":
    main()
