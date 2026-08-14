# -*- coding: utf-8 -*-
"""
check_convergence.py — 有限差分导数的可信域标定。

背景
----
MOC 以 Courant=1 运行，裂缝位置被吸附到网格，dx = L/N（dt=1 ms 时约 1.45 m）。
位置导数只能用 h = k*dx 的中心差分近似。该近似成立的条件是：观测信号在
时间尺度 2h/a 上足够光滑，即传感器带宽 fc 必须满足

        h << a / (4 * fc)          <=>        fc << a / (4h)

本脚本对同一批正演结果施加不同 fc，定位「差分收敛」的 (fc, dx) 可信域，
从而确定后续 CRB 扫描允许使用的最大带宽。

用法
----
    python -m analysis.identifiability.check_convergence --friction steady --dt 1e-3
    python -m analysis.identifiability.check_convergence --friction steady --dt 2.5e-4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import (  # noqa: E402
    ObsModel,
    Scenario,
    forward,
    observe,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", default="steady", choices=["steady", "brunone"])
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--tf", type=float, default=20.0)
    ap.add_argument("--spacing", type=float, default=10.0)
    ap.add_argument("--x1", type=float, default=4000.0)
    ap.add_argument("--steps", type=str, default="1,2,4,8")
    ap.add_argument("--fc", type=str, default="10,20,40,80,160,320")
    ap.add_argument(
        "--a-steps", type=str, default="1,2,4,8",
        help="波速扰动的网格数偏移 m；|da/a| = m/N。改变 a 等价于时间轴伸缩，"
             "导数随 t 线性增长，故 m 必须足够小以保证 (m/N)*tf*fc << 1",
    )
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    fcs = [float(s) for s in args.fc.split(",")]
    a_steps_arg = [int(s) for s in args.a_steps.split(",")]

    scn = Scenario(
        x_f=(args.x1, args.x1 + args.spacing),
        Cf=(1.0e-5, 1.0e-5),
        kleak=(1.0e-4, 1.0e-4),
        dt=args.dt,
        tf=args.tf,
        friction=args.friction,
    )
    n0 = scn.n_cells_nominal()
    dx0 = scn.dx_for(n0)
    idx0 = np.round(np.asarray(scn.x_f) / dx0).astype(int)
    x0 = idx0 * dx0

    print("=" * 88)
    print(f"差分收敛域标定  friction={args.friction}  dt={args.dt:g} s  tf={args.tf} s")
    print(f"网格 N={n0}  dx={dx0:.4f} m  a_adj={scn.a_for(n0):.4f} m/s")
    print(f"缝位置（吸附后）= {np.array2string(x0, precision=3)}  间距={x0[1] - x0[0]:.3f} m")
    print("=" * 88)

    # ---- 一次性跑完所有需要的正演 ----
    t0 = time.time()
    runs: Dict[str, np.ndarray] = {}
    print("\n运行正演 ...")

    base = forward(scn, n_cells=n0, x_f=x0)
    runs["base"] = base.H_wh
    tgrid = base.t
    print(f"  base done ({time.time() - t0:.1f}s)")

    for k in steps:
        for sgn in (+1, -1):
            xx = x0.copy()
            xx[0] = (idx0[0] + sgn * k) * dx0
            r = forward(scn, n_cells=n0, x_f=xx)
            runs[f"x0{'+' if sgn > 0 else '-'}{k}"] = r.H_wh
        print(f"  x0 +/- {k} cells done ({time.time() - t0:.1f}s)")

    obs_meta: Dict[str, object] = {}
    a_cell_steps = sorted({max(1, m) for m in a_steps_arg})
    for m in a_cell_steps:
        for sgn in (+1, -1):
            r = forward(scn, n_cells=n0 - sgn * m, x_f=x0)
            runs[f"a{'+' if sgn > 0 else '-'}{m}"] = r.H_wh
            obs_meta[f"a{'+' if sgn > 0 else '-'}{m}"] = (r.a_adj, r.placed_x.copy())
        print(f"  a +/- {m} cells done ({time.time() - t0:.1f}s)")

    print(f"正演总耗时 {time.time() - t0:.1f}s，共 {len(runs)} 次\n")

    # ---- 在不同 fc 下评估收敛性 ----
    print("[A] 位置导数 ds/dx0：不同带宽 fc 下的差分步长收敛性")
    print("    列 = 相邻步长之间的相对差 ||J_k - J_{k/2}|| / ||J_{k/2}||")
    header = "    fc[Hz]  a/(4fc)[m]  h/lam  " + "  ".join(
        f"{steps[i - 1]}->{steps[i]:<2d}" for i in range(1, len(steps))
    ) + "     ||J_1||"
    print(header)
    conv_table: List[Dict[str, object]] = []
    for fc in fcs:
        obs = ObsModel(fc_hz=fc)
        cols = {}
        for k in steps:
            sp = observe(runs[f"x0+{k}"], tgrid, scn, obs)
            sm = observe(runs[f"x0-{k}"], tgrid, scn, obs)
            cols[k] = (sp - sm) / (2.0 * k * dx0)
        rels = []
        for i in range(1, len(steps)):
            a_, b_ = cols[steps[i - 1]], cols[steps[i]]
            rels.append(float(np.linalg.norm(b_ - a_) / np.linalg.norm(a_)))
        lam = scn.a_for(n0) / (4.0 * fc)
        row = f"    {fc:6.0f}  {lam:10.3f}  {dx0 / lam:5.2f}  " + "  ".join(
            f"{r:6.2%}" for r in rels
        ) + f"   {np.linalg.norm(cols[steps[0]]):10.4e}"
        print(row)
        conv_table.append({"fc": fc, "rel": rels, "lam": lam})

    print("\n    判据：h/lam = dx*4fc/a 应 << 1；相邻步长相对差 < 10% 视为收敛。")

    # ---- 波速导数 ----
    print("\n[B] 波速导数 ds/da（含吸附修正）：不同 fc 与扰动步长")
    print("    线性区判据 phi = (m/N)*tf*fc << 1（末端相位漂移相对于最高频周期）")
    print("    " + "  ".join(
        f"m={m}: |da/a|={m / n0:.3%}, phi@fc_max={m / n0 * args.tf * max(fcs):.2f}"
        for m in a_cell_steps))
    print("    fc[Hz]   " + "  ".join(f"m={m:<7d}" for m in a_cell_steps)
          + "  相邻相对差")
    for fc in fcs:
        obs = ObsModel(fc_hz=fc)
        norms, cols = [], []
        for m in a_cell_steps:
            ap_, (a_p, px_p) = runs[f"a+{m}"], obs_meta[f"a+{m}"]
            am_, (a_m, px_m) = runs[f"a-{m}"], obs_meta[f"a-{m}"]
            sp = observe(ap_, tgrid, scn, obs)
            sm = observe(am_, tgrid, scn, obs)
            da = a_p - a_m
            col = (sp - sm) / da
            # 吸附修正：扣除因 dx 改变导致裂缝重新吸附的位置漂移贡献
            dpl = px_p - px_m
            k_ref = steps[0]
            for i in range(scn.n_frac):
                if i == 0:
                    jx = (observe(runs[f"x0+{k_ref}"], tgrid, scn, obs)
                          - observe(runs[f"x0-{k_ref}"], tgrid, scn, obs)) / (2.0 * k_ref * dx0)
                    col = col - jx * (dpl[i] / da)
            cols.append(col)
            norms.append(float(np.linalg.norm(col)))
        rels = [float(np.linalg.norm(cols[i] - cols[i - 1]) / np.linalg.norm(cols[i - 1]))
                for i in range(1, len(cols))]
        print(f"    {fc:6.0f}   " + "  ".join(f"{v:9.3e}" for v in norms)
              + "    " + "  ".join(f"{r:6.2%}" for r in rels))

    print(f"\n总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
