# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from node_coeffs import collect_branch_coeffs

G = 9.80665


def _base_phys(M=4, hetero=False):
    if hetero:
        K = np.array([1.0e10, 3.0e10, 8.0e9, 5.0e10][:M], dtype=np.float64)
        kappa = np.array([2e6, 0.0, 8e5, 4e6][:M], dtype=np.float64)
        I_f = np.array([4e5, 1e5, 8e5, 3e5][:M], dtype=np.float64)
        R_f = np.array([2e7, 5e6, 3e7, 1e7][:M], dtype=np.float64)
        C_f = np.array([2e-6, 8e-7, 3e-6, 1e-6][:M], dtype=np.float64)
        G_l = np.array([5e-10, 2e-10, 8e-10, 1e-9][:M], dtype=np.float64)
    else:
        K = np.full(M, 2.0e10)
        kappa = np.full(M, 1.0e6)
        I_f = np.full(M, 4.0e5)
        R_f = np.full(M, 2.0e7)
        C_f = np.full(M, 2.0e-6)
        G_l = np.full(M, 5.0e-10)
    return {
        "K": K, "kappa": kappa, "I_f": I_f, "R_f": R_f, "C_f": C_f, "G_l": G_l,
        "p_res": 4.0e7, "Cd": np.full(M, 0.7),
    }


def _pack(phys, dt, q_n, pc_n, F_n, CP, CM, BL, BR, RL, RR, z_j, H_init,
          inertia=1, storage=0, stiff_guard=1, H_scale=4000.0, Q_scale=0.05, p_scale=4e7):
    br = collect_branch_coeffs(dt, phys["I_f"], phys["R_f"], phys["C_f"], phys["G_l"],
                               phys["p_res"], q_n, pc_n, F_n, inertia, storage, stiff_guard)
    return {
        "CP": np.float64(CP), "CM": np.float64(CM),
        "BL": np.float64(BL), "BR": np.float64(BR),
        "RL": np.float64(RL), "RR": np.float64(RR),
        "z_j": np.float64(z_j), "rho_g": np.float64(1000.0 * G),
        "K": np.asarray(phys["K"], dtype=np.float64),
        "kappa": np.asarray(phys["kappa"], dtype=np.float64),
        "eps_q": 1e-8,
        "c1": br["c1"], "c0": br["c0"], "a1": br["a1"], "a0": br["a0"],
        "H_init": np.float64(H_init), "q_init": np.asarray(q_n, dtype=np.float64),
        "H_scale": H_scale, "Q_scale": Q_scale, "p_scale": p_scale,
        "stiff_guard": br["stiff_guard"],
        "dt": float(dt),
        "phys": phys,
        "q_n": np.asarray(q_n, dtype=np.float64),
        "pc_n": np.asarray(pc_n, dtype=np.float64),
        "F_n": np.asarray(F_n, dtype=np.float64),
        "inertia_scheme": inertia, "storage_scheme": storage,
    }


def scenario(name: str) -> dict:
    dt = 0.0028
    if name == "nominal":
        p = _base_phys(4, hetero=False)
        M = 4
        s = _pack(p, dt, np.full(M, 0.002), np.full(M, 4.05e7), np.full(M, 1e5),
                  5200.0, 4100.0, 80.0, 80.0, 0.4, 0.4, -2200.0, 4650.0)
        s["expect"] = {"stiff_guard": 0}
    elif name == "stiff_storage":
        p = _base_phys(4)
        p["C_f"] = np.full(4, 1.0e-12)
        p["G_l"] = np.full(4, 1.0e-8)
        M = 4
        s = _pack(p, dt, np.full(M, 0.002), np.full(M, 4.05e7), np.full(M, 1e5),
                  5200.0, 4100.0, 80.0, 80.0, 0.4, 0.4, -2200.0, 4650.0)
        s["expect"] = {"stiff_guard_gt": 0, "Gl_dt_over_Cf_gt": 1.0}
    elif name == "reverse_flow":
        p = _base_phys(4)
        M = 4
        s = _pack(p, dt, np.full(M, -0.002), np.full(M, 4.05e7), np.full(M, -1e5),
                  1800.0, 1700.0, 80.0, 80.0, 0.4, 0.4, 0.0, 1750.0)
        s["expect"] = {"q_all_negative": True, "q_max": 0.0}
    elif name == "near_zero":
        p = _base_phys(2)
        M = 2
        q_n = np.zeros(M)
        pc = np.full(M, 4.0e7)
        Fn = np.zeros(M)
        tmp = _pack(p, dt, q_n, pc, Fn, 2000.0, 2000.0, 80.0, 80.0, 0.0, 0.0, 0.0, 2000.0)
        p_eq = tmp["a0"] + tmp["c0"]
        H = float(np.mean(p_eq) / tmp["rho_g"] + 0.0)
        s = _pack(p, dt, q_n, pc, Fn, H + 0.5, H - 0.5, 80.0, 80.0, 0.0, 0.0, 0.0, H)
        s["expect"] = {"q_abs_max": 1e-6}
        s["near_zero_band"] = 1e-6
    elif name == "hetero_perf":
        p = _base_phys(4, hetero=True)
        M = 4
        s = _pack(p, dt, np.array([0.003, 0.001, 0.002, 0.0005]), np.full(M, 4.05e7),
                  np.full(M, 1e5), 5200.0, 4100.0, 80.0, 80.0, 0.4, 0.4, -2200.0, 4650.0)
        s["expect"] = {"hetero": True}
    elif name == "cr_lt1_friction":
        p = _base_phys(3)
        M = 3
        s = _pack(p, dt, np.full(M, 0.0015), np.full(M, 4.02e7), np.full(M, 8e4),
                  5000.0, 4200.0, 95.0, 70.0, 1.2, 0.8, -1800.0, 4600.0)
        s["expect"] = {"RL_nonzero": True}
    elif name == "storage_exp_branch":
        p = _base_phys(3)
        M = 3
        s = _pack(p, dt, np.full(M, 0.002), np.full(M, 4.05e7), np.full(M, 1e5),
                  5100.0, 4000.0, 80.0, 80.0, 0.2, 0.2, -2000.0, 4500.0,
                  inertia=1, storage=1, stiff_guard=0)
        s["expect"] = {"storage_scheme": 1, "stiff_guard": 0}
    elif name == "tikhonov_illcond":
        p = _base_phys(2)
        p["K"] = np.array([1.0e4, 1.0e20])
        p["kappa"] = np.array([0.0, 0.0])
        p["C_f"] = np.array([1e-3, 1e-16])
        M = 2
        s = _pack(p, dt, np.array([0.01, 1e-12]), np.array([4e7, 4e7]), np.zeros(M),
                  4800.0, 4700.0, 80.0, 80.0, 0.0, 0.0, -1000.0, 4750.0,
                  H_scale=4000.0, Q_scale=0.05, p_scale=1.0)
        s["expect"] = {"may_tikhonov": True}
    else:
        raise KeyError(name)
    s["name"] = name
    return s


ALL_NAMES = [
    "nominal", "stiff_storage", "reverse_flow", "near_zero",
    "hetero_perf", "cr_lt1_friction", "storage_exp_branch", "tikhonov_illcond",
]
