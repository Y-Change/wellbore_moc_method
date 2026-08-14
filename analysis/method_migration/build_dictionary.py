# -*- coding: utf-8 -*-
"""
build_dictionary.py — 生成层剥离（P0）的单缝 MOC 响应字典。

字典项
------
* intact   ：无裂缝井筒响应（tf=50s，brunone，与 lhs_dataset_2000 同配置）
* template ：单缝响应 − intact，深度网格 3500–4800 m（步长 20 m）× Cf∈{1e-7,1e-6}
              （kleak 固定 1e-4，即模板核的第三维简化）
* G0 失配变体：wavespeed×1.02 / V0×1.10 / roughness×2.0（核不确定代理）

输出：output/analysis/method_migration/dictionary/ 下各 variant .npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore

A_WAVE = 1450.0
V0 = 1.0
ROUGH = 4.5e-5
TF = 50.0
DT = 1.0e-3
TS = 1.0
L = 5000.0
DIA = 0.1397
RHO = 1000.0
NU = 1.0e-6
H0 = 300.0
H_EXT = 100.0

DEPTH_GRID = np.arange(3500.0, 4800.0 + 1.0, 10.0)   # 131 点（默认精细网格）
CF_GRID = np.array([1.0e-7, 1.0e-6])
KLEAK_FIX = 1.0e-4

MISMATCHES = {
    "nominal": {},
    "a_plus2pct": {"wavespeed": A_WAVE * 1.02},
    "V0_plus10pct": {"initial_velocity": V0 * 1.10},
    "roughness_x2": {"roughness_height": ROUGH * 2.0},
}

OUT_ROOT = Path("output/analysis/method_migration/dictionary")
N_WORKERS = int(os.environ.get("MM_WORKERS", "14"))


def build_config(**overrides) -> MocConfig:
    base = dict(
        wellbore_length=L,
        wellbore_diameter=DIA,
        fluid_density=RHO,
        fluid_viscosity=NU,
        wavespeed=A_WAVE,
        roughness_height=ROUGH,
        friction_model="brunone",
        dt=DT,
        tf=TF,
        wellhead_bc="velocity_step",
        pump_shut_time=TS,
        initial_velocity=V0,
        initial_head=H0,
        theta=0.0,
        toe_bc="reservoir",
        toe_head=H0,
    )
    base.update(overrides)
    return MocConfig(**base)


def _run_single(args: Tuple[str, float, float, float]) -> Dict[str, np.ndarray]:
    """跑单缝 case：variant, depth, cf, kleak → {depth, cf, kleak, H_wh}"""
    variant, depth, cf, kleak = args
    ov = MISMATCHES[variant]
    cfg = build_config(**ov)
    res = simulate_wellbore(
        cfg,
        fracture_positions=[depth],
        fracture_Cf=[cf],
        fracture_kleak=[kleak],
        H_ext=H_EXT,
        store_full_field=False,
    )
    return {
        "variant": variant,
        "depth": depth,
        "cf": cf,
        "kleak": kleak,
        "H_wh": np.asarray(res["wellhead_head"], dtype=np.float64),
    }


def _run_intact(variant: str) -> np.ndarray:
    ov = MISMATCHES[variant]
    cfg = build_config(**ov)
    res = simulate_wellbore(cfg, store_full_field=False)
    return np.asarray(res["wellhead_head"], dtype=np.float64)


def build_variant(variant: str, depth_step: float = 10.0) -> None:
    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.arange(3500.0, 4800.0 + 1.0, depth_step)

    t0 = time.time()
    intact = _run_intact(variant)
    print(f"[{variant}] intact done ({time.time()-t0:.0f}s)")

    tasks = [(variant, float(d), float(c), KLEAK_FIX)
             for d in grid for c in CF_GRID]
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(_run_single, tasks)

    # 组装 (n_depth, n_cf, n_samp)
    n_d = len(grid)
    n_c = len(CF_GRID)
    n_samp = intact.size
    templates = np.zeros((n_d, n_c, n_samp), dtype=np.float64)
    for r in results:
        di = int(np.where(grid == r["depth"])[0][0])
        ci = int(np.where(CF_GRID == r["cf"])[0][0])
        templates[di, ci] = r["H_wh"] - intact

    np.savez_compressed(
        out_dir / "dictionary.npz",
        depth_grid=grid,
        cf_grid=CF_GRID,
        kleak_fix=KLEAK_FIX,
        intact=intact,
        templates=templates,
        variant=variant,
    )
    print(f"[{variant}] done ({time.time()-t0:.0f}s): "
          f"{n_d}x{n_c} templates, {templates.nbytes/1e6:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(MISMATCHES.keys()))
    ap.add_argument("--depth-step", type=float, default=10.0,
                    help="深度网格步长 [m]（G0 失配变体建议 20 提速）")
    args = ap.parse_args()
    for v in args.variants:
        build_variant(v, args.depth_step)


if __name__ == "__main__":
    main()
