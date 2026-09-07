# -*- coding: utf-8 -*-
"""Solver-side memory consistency (tiny MOC + rec vs direct).

This is NOT a neural-operator memory-channel vs MOC z_l(x,t) gate.
Pilot NPZ have no interior snapshots, so the M3 pointwise <2% network
gate remains pending.  Do not batch-generate training snaps here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from paths import DATA_AUDIT_DIR, STAGE1_CODE, ensure_dirs

# Stage-1 config_io imports `paths.SCHEMA_DIR`.  The Stage-2 paths module is
# already cached, so drop it before importing the solver stack.
_stage2_paths = sys.modules.pop("paths")
sys.path.insert(0, str(STAGE1_CODE))
import goldens  # noqa: E402
import memory_kernel as mk  # noqa: E402
import moc_solver as ms  # noqa: E402
sys.modules["paths"] = _stage2_paths


THRESH = 0.02


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.linalg.norm(b)
    if den <= 1e-30:
        return float("nan"), True
    return float(np.linalg.norm(a - b) / den), False


def rec_algebra_and_fit():
    dt = 1.0e-3
    n_steps = 400
    t = dt * np.arange(n_steps + 1)
    V = 0.4 * np.sin(2.0 * np.pi * 3.0 * t)
    V[:40] = 0.0
    dV = np.diff(V)
    nu, D = 1.0e-6, 0.10
    Re = 800.0  # laminar → Zielke
    fz, fp = ms.get_kernel_fits(12, 1e-8, 1e2)
    m, n = mk.kernel_for_reynolds(Re, fz, fp)
    rk = mk.RecursiveKernel.build(m, n, nu, D, dt)
    Ju_rec = mk.convolve_recursive(dV, rk)
    Wbar_exp = mk.expsum_step_weights(m, n, nu, D, dt, n_steps)
    Ju_exp = mk.convolve_direct(dV, Wbar_exp, rk.w)
    _, Wint = mk.exact_W_for_reynolds(Re)
    Wbar_z = mk.direct_step_weights(Wint, nu, D, dt, n_steps)
    Ju_zielke = mk.convolve_direct(dV, Wbar_z, rk.w)
    rel_alg, silent_alg = _rel(Ju_rec[40:], Ju_exp[40:])
    rel_fit, silent_fit = _rel(Ju_rec[40:], Ju_zielke[40:])
    return {
        "name": "rec_algebra_and_zielke_fit",
        "rel_l2_rec_vs_expsum": rel_alg,
        "rel_l2_rec_vs_zielke": rel_fit,
        "silent": silent_alg or silent_fit,
        "pass": (not silent_alg) and rel_alg <= THRESH,
        "fit_is_report_only": True,
        "n_steps": n_steps,
        "Re": Re,
        "note": "gate = recursion vs same exponential-sum convolution; Zielke gap is M=12 fit residual, not a solver bug",
    }


def tiny_moc_z_consistency():
    well = goldens.joukowsky_case(V0=1.0)
    num = ms.NumericsSpec(
        Nx_ref=48, Cr_target=1.0, friction_model="zvb_rec",
        n_periods=0.5, T_total=1.2, node_decim=8, snap_every=1,
        do_energy=False, kernel_M=12,
    )
    res = ms.simulate(well, num)
    inode = int(res.grid.seg_off[0] + max(res.grid.seg_N[0] // 2, 1))
    A = well.A
    V = np.asarray(res.snap_Q[:, inode], dtype=np.float64) / A
    dt = float(res.meta["dt"])
    Re = abs(float(res.snap_Q[0, inode])) * well.D / (well.nu * A)
    fz, fp = ms.get_kernel_fits(12, 1e-8, 1e2)
    m, n = mk.kernel_for_reynolds(Re, fz, fp)
    rk = mk.RecursiveKernel.build(m, n, well.nu, well.D, dt)
    z = np.zeros(m.size)
    for k in range(1, V.size):
        z = rk.E * z + rk.Phi * (V[k] - V[k - 1])
    z_true = np.asarray(res.final_state["z"][:, inode], dtype=np.float64)
    rel, silent = _rel(z, z_true)
    return {
        "name": "tiny_joukowsky_snap_recon_vs_final_z",
        "rel_l2_z": rel,
        "silent": silent,
        "pass": (not silent) and rel <= THRESH,
        "inode": inode,
        "n_snap": int(res.snap_Q.shape[0]),
        "n_steps": int(res.meta["n_steps"]),
        "wall_s": float(res.meta["wall_clock_s"]),
        "snap_H_present": bool(res.snap_H.size),
        "note": "reconstructs z at one interior node from snap_Q; not a network memory gate",
    }


def main():
    ensure_dirs()
    rows = [rec_algebra_and_fit(), tiny_moc_z_consistency()]
    n_fail = sum(0 if r["pass"] else 1 for r in rows)
    out = {
        "status": "PASS" if n_fail == 0 else "FAIL",
        "n_fail": n_fail,
        "threshold": THRESH,
        "rows": rows,
        "cannot_certify": [
            "neural_memory_channel_vs_MOC_zl_xt_pointwise",
            "full_field_memory_on_pilot_npz",
        ],
        "scope": "solver-side consistency only; pilot training NPZ have snap_every=0",
    }
    dump_json(out, DATA_AUDIT_DIR / "memory_moc_consistency.json")
    print(json.dumps({k: out[k] for k in ("status", "n_fail", "rows")}, indent=2, default=str))
    if n_fail:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
