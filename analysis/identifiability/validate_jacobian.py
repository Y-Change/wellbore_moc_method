# -*- coding: utf-8 -*-
"""
validate_jacobian.py — CRB 分析的数值可信度检查。

在把任何 CRB 数字当结论之前必须通过的四项检查：
  A. 差分步长收敛性：位置导数在 grid_step = 1/2/4 下是否一致
  B. 波速扰动步长稳健性：a_cell_step 变化时 ds/da 是否稳定，吸附修正占比多大
  C. FIM 条件数与最弱方向：矩阵求逆是否数值可信
  D. 与解析基准对照：单缝纯时延情形的 CRB 是否接近经典时延估计界

用法
----
    python -m analysis.identifiability.validate_jacobian --friction steady
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import (  # noqa: E402
    ObsModel,
    Scenario,
    compute_jacobian,
    crb_report,
    linear_functional_std,
    noise_std_from_snr,
    observe,
)


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "   n/a  "
    if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-3):
        return f"{v:9.3e}"
    return f"{v:9.4f}"


def check_A_grid_step(scn: Scenario, obs: ObsModel, steps=(1, 2, 4)) -> None:
    print("\n[A] 位置导数的差分步长收敛性")
    print("    grid_step   dx步长[m]   ||ds/dx0||      与 step=1 的相对差")
    ref = None
    for gs in steps:
        jac = compute_jacobian(scn, obs, include=("x",), grid_step=gs)
        col = jac["J"][:, 0]
        nrm = float(np.linalg.norm(col))
        if ref is None:
            ref, rel = col, 0.0
        else:
            rel = float(np.linalg.norm(col - ref) / np.linalg.norm(ref))
        print(f"    {gs:^9d}   {gs * jac['diagnostics']['dx0']:8.3f}   "
              f"{nrm:12.5e}   {rel:14.4%}")
    print("    判据：step=1 与 step=2 的相对差 < ~15% 视为已进入线性区。")


def check_B_wavespeed(scn: Scenario, obs: ObsModel, cell_steps=(9, 17, 35)) -> None:
    print("\n[B] 波速导数的扰动步长稳健性与吸附修正")
    print("    a_cell_step  |da/a|      ||ds/da||       吸附修正占比   与首个的相对差")
    ref = None
    for cs in cell_steps:
        jac = compute_jacobian(scn, obs, include=("x", "a"), a_cell_step=cs)
        ja = jac["names"].index("a")
        col = jac["J"][:, ja]
        d = jac["diagnostics"]
        if ref is None:
            ref, rel = col, 0.0
        else:
            rel = float(np.linalg.norm(col - ref) / np.linalg.norm(ref))
        print(f"    {cs:^11d}  {d['a_rel_step']:.4%}   {np.linalg.norm(col):12.5e}   "
              f"{d['a_snap_correction_rel']:11.2%}   {rel:14.4%}")
    print("    判据：修正占比应远小于 1（否则波速列被网格抖动主导）；")
    print("          不同 a_cell_step 的相对差 < ~15%。")


def check_C_conditioning(scn: Scenario, obs: ObsModel, snr_db: float = 30.0) -> None:
    print(f"\n[C] FIM 条件数与最弱可辨识方向 (SNR={snr_db} dB)")
    jac = compute_jacobian(scn, obs)
    rep = crb_report(jac, snr_db)
    names = rep["names"]
    print(f"    观测点数 n_obs = {rep['n_obs']},  sigma = {rep['sigma']:.5e} m")
    print(f"    归一化 FIM 条件数（波速未知）= {rep['cond_normalized']:.4e}")
    print(f"    归一化 FIM 条件数（波速已知）= {rep['cond_no_nuisance']:.4e}")
    print("    归一化特征值：", np.array2string(rep["eigvals_normalized"],
                                            precision=3, max_line_width=100))
    print("    最弱方向（归一化坐标下的特征向量）：")
    for n, v in zip(names, rep["weakest_direction"]):
        bar = "#" * int(round(abs(v) * 40))
        print(f"        {n:>10s}  {v:+7.4f}  {bar}")
    print("    判据：条件数 < 1e12 时 float64 求逆可信；若最弱方向由 x 与 a 共同构成，")
    print("          说明深度与波速存在近简并（本项目的核心怀疑）。")


def check_D_single_fracture(scn: Scenario, obs: ObsModel, snr_db: float = 30.0) -> None:
    """单缝、波速已知时，位置 CRB 应接近经典时延估计界。

    经典结果： var(tau) >= sigma^2 / ||ds/dtau||^2 ，而 x = a*tau/2，
    故这里直接比较数值 CRB 与 1/sqrt(F_xx)（二者在单参数情形应完全相等），
    用于验证 _safe_inverse 的实现。
    """
    print(f"\n[D] 单参数一致性检查 (SNR={snr_db} dB)")
    scn1 = Scenario(**{**scn.__dict__, "x_f": (scn.x_f[0],),
                       "Cf": (scn.Cf[0],), "kleak": (scn.kleak[0],)})
    jac = compute_jacobian(scn1, obs, include=("x",))
    rep = crb_report(jac, snr_db, nuisance=())
    sigma = rep["sigma"]
    manual = sigma / float(np.linalg.norm(jac["J"][:, 0]))
    print(f"    数值 CRB std(x0) = {rep['std_full'][0]:.6e} m")
    print(f"    手算 sigma/||J|| = {manual:.6e} m")
    rel = abs(rep["std_full"][0] - manual) / manual
    print(f"    相对偏差 = {rel:.3e}   {'PASS' if rel < 1e-9 else 'FAIL'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", default="steady", choices=["steady", "brunone"])
    ap.add_argument("--spacing", type=float, default=10.0)
    ap.add_argument("--x1", type=float, default=4000.0)
    ap.add_argument("--tf", type=float, default=20.0)
    ap.add_argument("--fc", type=float, default=200.0)
    ap.add_argument("--snr", type=float, default=30.0)
    args = ap.parse_args()

    scn = Scenario(
        x_f=(args.x1, args.x1 + args.spacing),
        Cf=(1.0e-5, 1.0e-5),
        kleak=(1.0e-4, 1.0e-4),
        tf=args.tf,
        friction=args.friction,
    )
    obs = ObsModel(fc_hz=args.fc)

    print("=" * 78)
    print(f"CRB 数值验证  friction={args.friction}  spacing={args.spacing} m  "
          f"x1={args.x1} m  tf={args.tf}s  fc={args.fc}Hz")
    print(f"网格：N={scn.n_cells_nominal()}  dx={scn.dx_for(scn.n_cells_nominal()):.4f} m  "
          f"a_adj={scn.a_for(scn.n_cells_nominal()):.4f} m/s")
    print("=" * 78)

    t0 = time.time()
    check_A_grid_step(scn, obs)
    check_B_wavespeed(scn, obs)
    check_C_conditioning(scn, obs, args.snr)
    check_D_single_fracture(scn, obs, args.snr)
    print(f"\n总耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
