# -*- coding: utf-8 -*-
"""
run_landscape.py — 似然面（失配面）扫描：检验全局多峰性与波速-深度简并。

为什么需要
----------
Fisher 信息矩阵与 CRB 都是**局部**量，只描述真值附近的曲率。即使局部信息充足，
似然面仍可能存在多个深谷（例如把某条缝的反射错认成另一条缝的），导致任何优化
器或网络都收敛到错误的盆地。CRB 对这种失败完全盲视，必须单独检验。

两种扫描
--------
profile1d : 固定其余参数，在很宽的深度范围扫描 x0，考察
            (a) 是否存在与全局最小竞争的远处局部极小；
            (b) 全局盆地宽度与 CRB 预测的曲率是否一致。
grid2d    : 在 (x0, 波速) 平面扫描，直接显示深度-波速简并谷的形状与走向。

失配统计量
----------
    chi2(theta) = || s(theta) - s_true ||^2 / sigma^2
输出 delta_chi2 = chi2 - min(chi2)。对高斯噪声，delta_chi2 ~ 1 对应 1-sigma；
若某个远处局部极小的 delta_chi2 只有个位数，说明它在噪声下与真解无法区分。
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
    forward,
    noise_std_from_snr,
    observe,
)

OUT_DEFAULT = os.path.join("output", "analysis", "identifiability", "landscape")


def _job(args):
    scn, n_cells, x_f = args
    r = forward(scn, n_cells=int(n_cells), x_f=x_f)
    return {"H_wh": r.H_wh, "t": r.t, "a_adj": r.a_adj,
            "placed_x": r.placed_x, "n_cells": r.n_cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["profile1d", "grid2d", "both"])
    ap.add_argument("--friction", default="steady", choices=["steady", "brunone"])
    ap.add_argument("--spacing", type=float, default=20.0)
    ap.add_argument("--x1", type=float, default=4000.0)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--tf", type=float, default=20.0)
    ap.add_argument("--fc", type=float, default=20.0)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--scan-lo", type=float, default=3400.0, help="x0 扫描下限 [m]")
    ap.add_argument("--scan-hi", type=float, default=4600.0, help="x0 扫描上限 [m]")
    ap.add_argument("--scan-stride", type=int, default=2, help="扫描步长（网格数）")
    ap.add_argument("--grid2d-halfwidth", type=float, default=80.0,
                    help="grid2d 中 x0 偏移半宽 [m]")
    ap.add_argument("--grid2d-arel", type=float, default=0.01,
                    help="grid2d 中波速相对偏移半宽")
    ap.add_argument("--grid2d-na", type=int, default=13)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    tag = f"{args.friction}_D{args.spacing:g}_fc{args.fc:g}_snr{args.snr:g}"

    scn = Scenario(
        x_f=(args.x1, args.x1 + args.spacing),
        Cf=(1.0e-5, 1.0e-5), kleak=(1.0e-4, 1.0e-4),
        dt=args.dt, tf=args.tf, friction=args.friction,
    )
    obs = ObsModel(fc_hz=args.fc)
    N0 = scn.n_cells_nominal()
    dx0 = scn.dx_for(N0)
    idx_true = np.round(np.asarray(scn.x_f) / dx0).astype(int)
    x_true = idx_true * dx0

    print("=" * 84)
    print(f"似然面扫描  {tag}")
    print(f"  真值 x = {np.array2string(x_true, precision=2)} m  "
          f"(网格 N={N0}, dx={dx0:.4f} m, a={scn.a_for(N0):.3f} m/s)")
    print("=" * 84)

    t0 = time.time()
    truth = _job((scn, N0, x_true))
    s_true = observe(truth["H_wh"], truth["t"], scn, obs)
    sigma = noise_std_from_snr(s_true, args.snr)
    print(f"sigma = {sigma:.4e} m  (SNR={args.snr:g} dB, n_obs={s_true.size})")

    results: Dict[str, object] = {
        "tag": tag, "friction": args.friction, "spacing_m": args.spacing,
        "x1_m": args.x1, "fc_hz": args.fc, "snr_db": args.snr,
        "sigma_m": float(sigma), "dt_s": args.dt, "tf_s": args.tf,
        "x_true_m": x_true.tolist(), "a_true_mps": float(scn.a_for(N0)),
        "n_obs": int(s_true.size),
    }

    ex = ProcessPoolExecutor(max_workers=args.workers)

    # ---------- 1D 深度剖面 ----------
    if args.mode in ("profile1d", "both"):
        i_lo = max(1, int(round(args.scan_lo / dx0)))
        i_hi = min(N0 - 1, int(round(args.scan_hi / dx0)))
        idxs = [i for i in range(i_lo, i_hi + 1, args.scan_stride)
                if i != idx_true[1]]
        jobs = [(scn, N0, np.array([i * dx0, x_true[1]])) for i in idxs]
        print(f"\n[profile1d] {len(jobs)} 次正演 "
              f"(x0 从 {i_lo * dx0:.0f} 到 {i_hi * dx0:.0f} m, 步长 {args.scan_stride * dx0:.2f} m)")
        outs = list(ex.map(_job, jobs))
        xs = np.array([i * dx0 for i in idxs])
        chi2 = np.array([
            float(np.sum((observe(o["H_wh"], o["t"], scn, obs) - s_true) ** 2))
            / sigma ** 2 for o in outs
        ])
        d_chi2 = chi2 - chi2.min()
        results["profile1d"] = {"x0_m": xs.tolist(), "delta_chi2": d_chi2.tolist()}

        # 局部极小识别
        loc = [k for k in range(1, len(d_chi2) - 1)
               if d_chi2[k] < d_chi2[k - 1] and d_chi2[k] < d_chi2[k + 1]]
        loc.sort(key=lambda k: d_chi2[k])
        print(f"    全局最小在 x0 = {xs[int(np.argmin(d_chi2))]:.2f} m "
              f"(真值 {x_true[0]:.2f} m)")
        print(f"    共 {len(loc)} 个局部极小；最强的若干个：")
        print("        x0[m]      Δχ²        与真值距离[m]")
        for k in loc[:8]:
            print(f"      {xs[k]:9.2f}  {d_chi2[k]:12.4e}   {xs[k] - x_true[0]:+9.2f}")
        comp = [k for k in loc if d_chi2[k] < 9.0 and abs(xs[k] - x_true[0]) > 3 * dx0]
        results["profile1d"]["n_competing_minima_dchi2_lt9"] = len(comp)
        print(f"    与真解无法区分（Δχ²<9 且距真值>{3 * dx0:.1f} m）的竞争极小：{len(comp)} 个")
        print(f"    ({time.time() - t0:.0f}s)")

    # ---------- 2D (x0, a) ----------
    if args.mode in ("grid2d", "both"):
        m_max = int(round(args.grid2d_arel * N0))
        ms = np.unique(np.round(np.linspace(-m_max, m_max, args.grid2d_na)).astype(int))
        half = int(round(args.grid2d_halfwidth / dx0))
        offs = list(range(-half, half + 1, args.scan_stride))
        jobs, meta = [], []
        for m in ms:
            n_cells = N0 - int(m)          # N 减小 -> a 增大
            dxm = scn.L / n_cells
            for o in offs:
                # 整个缝群刚性平移（共模）。只平移首缝会在正 δa 一侧撞上次缝，
                # 且共模才是与波速真正简并的方向：缝间距是波速不变量。
                xf = x_true + o * dx0
                jobs.append((scn, n_cells, xf))
                meta.append((int(m), o, scn.L / (n_cells * scn.dt), dxm))
        print(f"\n[grid2d] {len(jobs)} 次正演 "
              f"({len(ms)} 个波速 × {len(offs)} 个深度偏移)")
        outs = list(ex.map(_job, jobs))
        rec = []
        for (m, o, a_val, dxm), out in zip(meta, outs):
            s = observe(out["H_wh"], out["t"], scn, obs)
            rec.append({
                "m": m, "a_mps": a_val, "a_rel": a_val / scn.a_for(N0) - 1.0,
                "x0_offset_m": o * dx0,
                "x0_placed_m": float(out["placed_x"][0]),
                "chi2": float(np.sum((s - s_true) ** 2) / sigma ** 2),
            })
        c = np.array([r["chi2"] for r in rec])
        for r, v in zip(rec, c - c.min()):
            r["delta_chi2"] = float(v)
        results["grid2d"] = rec

        # 每个波速下的最优深度 -> 简并谷的走向
        print("    简并谷：每个波速下的最佳共模平移量")
        print("       δa/a        最佳Δx[m]      该处Δχ²     预测 x0*δa/a [m]")
        valley = []
        for m in ms:
            sub = [r for r in rec if r["m"] == int(m)]
            best = min(sub, key=lambda r: r["chi2"])
            pred = x_true[0] * best["a_rel"]
            valley.append({"a_rel": best["a_rel"], "dx0": best["x0_offset_m"],
                           "dchi2": best["delta_chi2"], "pred": pred})
            print(f"    {best['a_rel']:+9.4%}  {best['x0_offset_m']:+11.2f}  "
                  f"{best['delta_chi2']:12.4e}  {pred:+14.2f}")
        results["grid2d_valley"] = valley
        print(f"    ({time.time() - t0:.0f}s)")

    ex.shutdown()
    path = os.path.join(out_root, f"{tag}.json")
    if os.path.exists(path):   # 合并已有扫描（profile1d / grid2d 可分次运行）
        with open(path, encoding="utf-8") as fh:
            prev = json.load(fh)
        prev.update(results)
        results = prev
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n写出 → {path}   总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
