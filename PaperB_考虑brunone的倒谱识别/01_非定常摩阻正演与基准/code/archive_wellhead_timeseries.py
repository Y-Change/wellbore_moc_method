# -*- coding: utf-8 -*-
"""Archive wellhead traces used by Paper B Figures 2, 6(a) and 7 into the evidence pack.

Writes gzipped CSVs under 01_.../data/timeseries/ so plotting no longer depends on
output/analysis/brunone_spacing_effect/.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

_d = os.path.dirname(os.path.abspath(__file__))
_paperb = os.path.dirname(os.path.dirname(_d))
_root = os.path.dirname(_paperb)
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.config import FRACTURE_CONFIG, WELL_CONFIG
from moc_simulate.wellbore_moc import MocConfig, brunone_k_vec, simulate_wellbore
import moc_simulate.wellbore_moc as wm

OUT_DIR = os.path.join(_paperb, "01_非定常摩阻正演与基准", "data", "timeseries")
os.makedirs(OUT_DIR, exist_ok=True)

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
DT = 0.001
TF = 50.0
TS = 1.0


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cfg(friction_model: str) -> MocConfig:
    return MocConfig(
        wellbore_length=WELL_CONFIG["L"],
        wellbore_diameter=WELL_CONFIG["wellbore_diameter"],
        fluid_density=WELL_CONFIG["fluid_density"],
        fluid_viscosity=WELL_CONFIG["fluid_viscosity"],
        wavespeed=WAVESPEED,
        roughness_height=WELL_CONFIG["roughness_height"],
        friction_model=friction_model,
        dt=DT,
        tf=TF,
        wellhead_bc="velocity_step",
        pump_shut_time=TS,
        initial_velocity=WELL_CONFIG["V0"],
        initial_head=WELL_CONFIG["H0"],
        theta=WELL_CONFIG["theta"],
        toe_bc="reservoir",
        toe_head=WELL_CONFIG["H0"],
    )


def _patch_constant_k(k_val: float):
    orig_k, orig_vec = wm.brunone_k, wm.brunone_k_vec

    def mock_k(Re: float) -> float:
        return k_val

    def mock_vec(Re_arr: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(Re_arr, dtype=np.float64), k_val)

    wm.brunone_k = mock_k
    wm.brunone_k_vec = mock_vec
    return orig_k, orig_vec


def _restore(orig_k, orig_vec):
    wm.brunone_k = orig_k
    wm.brunone_k_vec = orig_vec


def _save_wh(stem: str, t, H, Q) -> str:
    path = os.path.join(OUT_DIR, f"{stem}.csv.gz")
    pd.DataFrame({"t": t, "H_wh": H, "Q_wh": Q}).to_csv(path, index=False, compression="gzip")
    return path


def run_case(stem: str, k: float, x_f):
    out_path = os.path.join(OUT_DIR, f"{stem}.csv.gz")
    if os.path.exists(out_path):
        print(f"skip existing {stem}")
        return out_path
    friction = "steady" if k == 0.0 else "brunone"
    orig = None
    if k > 0.0:
        orig = _patch_constant_k(k)
    cfg = _cfg(friction)
    n_frac = len(x_f)
    t0 = time.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=list(x_f),
        fracture_Cf=[FRACTURE_CONFIG["Cf"]] * n_frac,
        fracture_kleak=[FRACTURE_CONFIG["kleak"]] * n_frac,
        H_ext=FRACTURE_CONFIG["H_ext"],
        store_full_field=False,
    )
    print(f"  {stem} done in {time.time() - t0:.1f}s")
    if orig is not None:
        _restore(*orig)
    Q = res["wellhead_velocity"] * cfg.area
    return _save_wh(stem, res["timestamps"], res["wellhead_head"], Q)


def probe_dynamic_k_n1():
    """Archive realised k(Re) on the n=1 paper case using spatial snapshots."""
    csv_path = os.path.join(
        _paperb, "05_雷诺数动态k与物性敏感性", "data", "realised_k_n1_snapshots.csv"
    )
    if os.path.exists(csv_path):
        print("skip existing realised_k table")
        return csv_path

    snaps = list(np.round(np.arange(1.0, 50.0 + 1e-9, 0.5), 6))
    cfg = _cfg("brunone")
    D = cfg.wellbore_diameter
    nu = cfg.fluid_viscosity
    t0 = time.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=[FRAC_X1],
        fracture_Cf=[FRACTURE_CONFIG["Cf"]],
        fracture_kleak=[FRACTURE_CONFIG["kleak"]],
        H_ext=FRACTURE_CONFIG["H_ext"],
        store_full_field=False,
        snapshot_times=snaps,
    )
    print(f"  dynamic k probe done in {time.time() - t0:.1f}s, snapshots={len(res['snapshots'])}")

    V_list, t_list = [], []
    for rec in res["snapshots"].values():
        if rec["t"] + 1e-12 < TS:
            continue
        V_list.append(np.asarray(rec["V"], dtype=np.float64))
        t_list.append(rec["t"])
    V = np.vstack(V_list)
    absV = np.abs(V)
    Re = absV * D / nu
    k = brunone_k_vec(Re.ravel()).reshape(Re.shape)

    rows = [{
        "case": "n1_X1=4100_dynamic_k(Re)",
        "nu_m2s": nu,
        "D_m": D,
        "n_snapshots_post_shut": int(Re.shape[0]),
        "n_nodes": int(Re.shape[1]),
        "Re_median": float(np.median(Re)),
        "Re_P95": float(np.percentile(Re, 95)),
        "Re_max": float(np.max(Re)),
        "k_median": float(np.median(k)),
        "k_P95": float(np.percentile(k, 95)),
        "k_max": float(np.max(k)),
        "k_mean": float(np.mean(k)),
        "frac_Re_lt_2000": float(np.mean(Re < 2000.0)),
        "update_rule": "every interior node, every MOC step n>=2 (probe stats from 0.5 s snapshots)",
        "Re_lt_1_k": 0.0,
        "k_laminar_1_to_2000": float(np.sqrt(4.76e-3) / 2.0),
        "discontinuous_at_Re_2000": True,
    }]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def write_manifest(paths):
    rows = []
    for p in paths:
        if p is None or not os.path.exists(p):
            continue
        rel = os.path.relpath(p, _paperb)
        rows.append({
            "relative_path": rel.replace("\\", "/"),
            "bytes": os.path.getsize(p),
            "sha256": _sha256(p),
        })
    man = os.path.join(OUT_DIR, "SHA256.csv")
    pd.DataFrame(rows).to_csv(man, index=False)
    print(f"wrote {man}")
    return man


def main():
    paths = []
    n1_cases = [
        ("n1_k0", 0.0, [FRAC_X1]),
        ("n1_k0.01", 0.01, [FRAC_X1]),
        ("n1_k0.02", 0.02, [FRAC_X1]),
        ("n1_k0.05", 0.05, [FRAC_X1]),
    ]
    n4_cases = [
        ("n4_D20_k0.01", 0.01, [4100.0, 4120.0, 4140.0, 4160.0]),
        ("n4_D5_k0.01", 0.01, [4100.0, 4105.0, 4110.0, 4115.0]),
    ]
    for stem, k, xf in n1_cases + n4_cases:
        print(f"Running {stem} ...")
        paths.append(run_case(stem, k, xf))
    paths.append(probe_dynamic_k_n1())
    write_manifest(paths)


if __name__ == "__main__":
    main()
