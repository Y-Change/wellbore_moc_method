# -*- coding: utf-8 -*-
"""
probe_realised_k.py — 实测 MOC 仿真中真正生效的 Brunone 系数 k 的分布。

背景
----
`analysis/brunone_spacing_effect/` 的 k×D 矩阵是把 k 当作外部给定常数扫描的，
而 `run_lhs_batch_simulate.py` 生成的数据集（friction="brunone"）里 k 由
`brunone_k(Re)` 按局部瞬时 Re 逐点计算。二者不是同一个量，因此无法直接用
Rayleigh 可分辨性矩阵去查表判断 LHS 数据集处于哪一档频散强度。

本脚本跑一次带全场输出的仿真，统计停泵后管内 (t, x) 上实际取到的 k 分布，
用于把 LHS 数据集定位到 k×D 矩阵的对应档位。

注意：npz 数据集只存井口 Q，而井口停泵后 Q≡0，无法从数据文件反推 k，
必须重跑仿真取内部速度场。

用法
----
    python -m analysis.brunone_spacing_effect.probe_realised_k
"""
from __future__ import annotations

import argparse

import numpy as np

from moc_simulate.config import SIM_CONFIG, WELL_CONFIG
from moc_simulate.wellbore_moc import (
    MocConfig,
    brunone_k,
    brunone_k_vec,
    simulate_wellbore,
)

# lhs_dataset_2000 的典型样本：4 缝、约 8 m 间距（该数据集间距中位数 8.31 m），
# Cf / kleak 取 LHS 对数范围的中部。
DEFAULT_X_F = (4100.0, 4108.0, 4116.0, 4124.0)
DEFAULT_LOG10_CF = -7.1
DEFAULT_LOG10_KLEAK = -4.5
DEFAULT_H_EXT = 100.0


def describe_k_of_re() -> None:
    D = WELL_CONFIG["wellbore_diameter"]
    nu = WELL_CONFIG["fluid_viscosity"]
    V0 = WELL_CONFIG["V0"]

    print("=== k(Re) 特性 ===")
    print(f"  Re(V=V0={V0}) = {V0 * D / nu:.4g}  ->  k = {brunone_k(V0 * D / nu):.5f}")
    print(f"  层流分支 (Re<2000)             ->  k = {brunone_k(1000.0):.5f}")
    print(f"  Re=2000 对应 |V|               =  {2000 * nu / D:.5f} m/s")
    for re in (5e3, 1e4, 3e4, 6e4, 1e5, 1.4e5):
        print(f"    Re={re:>9.4g}  k={brunone_k(re):.5f}")


def probe(tf: float, x_f: tuple[float, ...]) -> None:
    D = WELL_CONFIG["wellbore_diameter"]
    nu = WELL_CONFIG["fluid_viscosity"]
    a = WELL_CONFIG["wavespeed"]

    cfg = MocConfig(
        wellbore_length=WELL_CONFIG["L"],
        wellbore_diameter=D,
        fluid_density=WELL_CONFIG["fluid_density"],
        fluid_viscosity=nu,
        wavespeed=a,
        roughness_height=WELL_CONFIG["roughness_height"],
        friction_model="brunone",
        dt=SIM_CONFIG["dt"],
        tf=tf,
        pump_shut_time=SIM_CONFIG["ts"],
        initial_velocity=WELL_CONFIG["V0"],
        initial_head=WELL_CONFIG["H0"],
        theta=WELL_CONFIG["theta"],
    )
    print(f"\n=== 仿真 (tf={cfg.tf}s, N={cfg.N}, 缝={list(x_f)}) ===")

    res = simulate_wellbore(
        cfg,
        fracture_positions=list(x_f),
        fracture_Cf=[10.0**DEFAULT_LOG10_CF] * len(x_f),
        fracture_kleak=[10.0**DEFAULT_LOG10_KLEAK] * len(x_f),
        H_ext=DEFAULT_H_EXT,
        store_full_field=True,
    )

    t = res["timestamps"]
    post = t >= cfg.pump_shut_time
    absV = np.abs(res["velocity"][post])
    re = absV * D / nu
    k = brunone_k_vec(re)

    print("\n=== 停泵后管内 |V| ===")
    for q in (5, 25, 50, 75, 95):
        print(f"  P{q:<2d} |V| = {np.percentile(absV, q):.5f} m/s")
    print(f"  落在层流分支 (Re<2000) 的 (t,x) 占比: {np.mean(re < 2000.0) * 100:.1f}%")

    print("\n=== 实际生效的 Brunone k 分布（停泵后全场 (t,x)）===")
    for q in (5, 25, 50, 75, 95):
        print(f"  P{q:<2d} k = {np.percentile(k, q):.5f}")
    print(f"  算术平均 k       = {k.mean():.5f}")

    # 静止节点对波动能量几乎无贡献，用 |V| 加权更能代表"波看到的" k
    weight = absV / max(absV.sum(), 1e-30)
    print(f"  |V| 加权平均 k   = {float((k * weight).sum()):.5f}")

    print("\n=== 参考尺度 ===")
    print(f"  a/(2*B_coh), B_coh=18.9 Hz -> {a / (2 * 18.9):.2f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf", type=float, default=8.0, help="仿真时长 [s]")
    args = parser.parse_args()

    describe_k_of_re()
    probe(args.tf, DEFAULT_X_F)


if __name__ == "__main__":
    main()
