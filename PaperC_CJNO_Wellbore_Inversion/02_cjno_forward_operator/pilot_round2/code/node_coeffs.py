# -*- coding: utf-8 -*-
"""Real discrete coefficients from per-perf physics + previous state.

Never reads current-time target pc. a0/c0 come from (q^n, p_c^n, F^n).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from paths import STAGE1_CODE

_stage2_paths = sys.modules.get("paths")
sys.path.insert(0, str(STAGE1_CODE))
from node_newton import branch_coefficients  # noqa: E402
if _stage2_paths is not None:
    sys.modules["paths"] = _stage2_paths

G = 9.80665


def darcy_R(f, dx, D, A, g):
    """R such that (R/2) Q|Q| is the trapezoidal friction head (see moc_solver)."""
    return f * dx / (2.0 * g * D * A * A)


def collect_branch_coeffs(dt, I_f, R_f, C_f, G_l, p_res, q_n, p_c_n, F_n,
                          inertia_scheme=1, storage_scheme=0, stiff_guard=1):
    I_f = np.atleast_1d(np.asarray(I_f, dtype=np.float64))
    R_f = np.atleast_1d(np.asarray(R_f, dtype=np.float64))
    C_f = np.atleast_1d(np.asarray(C_f, dtype=np.float64))
    G_l = np.atleast_1d(np.asarray(G_l, dtype=np.float64))
    q_n = np.atleast_1d(np.asarray(q_n, dtype=np.float64))
    p_c_n = np.atleast_1d(np.asarray(p_c_n, dtype=np.float64))
    F_n = np.atleast_1d(np.asarray(F_n, dtype=np.float64))
    M = I_f.size
    c1 = np.zeros(M)
    c0 = np.zeros(M)
    a1 = np.zeros(M)
    a0 = np.zeros(M)
    guard = 0
    for m in range(M):
        cc1, cc0, aa1, aa0, g = branch_coefficients(
            float(dt), float(I_f[m]), float(R_f[m]), float(C_f[m]), float(G_l[m]),
            float(p_res), float(q_n[m]), float(p_c_n[m]), float(F_n[m]),
            int(inertia_scheme), int(storage_scheme), int(stiff_guard),
        )
        c1[m], c0[m], a1[m], a0[m] = cc1, cc0, aa1, aa0
        guard += int(g)
    return {"c1": c1, "c0": c0, "a1": a1, "a0": a0, "stiff_guard": int(guard)}


def pack_from_physical(phys_cluster: Dict, dt: float, q_n, p_c_n, F_n,
                       CP, CM, BL, BR, RL, RR, z_j, rho_g,
                       H_init, H_scale, Q_scale, p_scale, eps_q=1e-8) -> Dict:
    """phys_cluster holds K, kappa, I_f, R_f, C_f, G_l, p_res — all real."""
    br = collect_branch_coeffs(dt, phys_cluster["I_f"], phys_cluster["R_f"],
                               phys_cluster["C_f"], phys_cluster["G_l"],
                               phys_cluster["p_res"], q_n, p_c_n, F_n)
    return {
        "CP": np.float64(CP), "CM": np.float64(CM),
        "BL": np.float64(BL), "BR": np.float64(BR),
        "RL": np.float64(RL), "RR": np.float64(RR),
        "z_j": np.float64(z_j), "rho_g": np.float64(rho_g),
        "K": np.asarray(phys_cluster["K"], dtype=np.float64),
        "kappa": np.asarray(phys_cluster["kappa"], dtype=np.float64),
        "eps_q": float(eps_q),
        "c1": br["c1"], "c0": br["c0"], "a1": br["a1"], "a0": br["a0"],
        "H_init": np.float64(H_init), "q_init": np.asarray(q_n, dtype=np.float64),
        "H_scale": float(H_scale), "Q_scale": float(Q_scale), "p_scale": float(p_scale),
        "stiff_guard": br["stiff_guard"],
        "dt": float(dt),
    }


def update_pc_q(q_new, a0, a1, c0, c1):
    """Free-roll state update from affine branch maps (own prediction)."""
    q_new = np.asarray(q_new, dtype=np.float64)
    pc = a0 + a1 * q_new
    F = c1 * q_new + c0
    return q_new, pc, F
