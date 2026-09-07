# -*- coding: utf-8 -*-
"""Node-layer gates: forward vs MOC, dimensionless residual, IFT vs central FD."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from interface_newton import (
    COND_LIMIT, ImplicitNodeLayer, cluster_newton, make_branch_pack_from_state,
    reduced_residual_and_jac,
)
from paths import DATA_AUDIT_DIR, STAGE1_CODE, ensure_dirs

sys.path.insert(0, str(STAGE1_CODE))
from node_newton import solve_cluster_node

G = 9.80665


def _pack_tensors(base, requires=()):
    skip = {"name", "guard", "tol"}
    t = {}
    for k, v in base.items():
        if k in skip:
            continue
        if k in ("eps_q", "H_scale", "Q_scale", "p_scale"):
            t[k] = float(v)
        else:
            x = torch.as_tensor(v, dtype=torch.float64)
            if k in requires:
                x = x.clone().requires_grad_(True)
            t[k] = x
    return t


def _scenario(name="nominal"):
    M = 4
    K = np.full(M, 2.0e10)
    kappa = np.full(M, 1.0e6)
    I_f = np.full(M, 4.0e5)
    R_f = np.full(M, 2.0e7)
    C_f = np.full(M, 2.0e-6)
    G_l = np.full(M, 5.0e-10)
    p_res = np.full(M, 4.0e7)
    dt = 0.0028
    q_n = np.full(M, 0.002)
    pc_n = np.full(M, 4.05e7)
    Fn = np.full(M, 1.0e5)
    if name == "stiff_storage":
        C_f = np.full(M, 2.0e-9)
        G_l = np.full(M, 3.0e-9)
    if name == "reverse_flow":
        q_n = np.full(M, -0.003)
    if name == "near_zero":
        q_n = np.full(M, 1e-10)
    if name == "many_perf":
        M = 12
        K = np.full(M, 2.0e10)
        kappa = np.full(M, 1.0e6)
        I_f = np.full(M, 4.0e5)
        R_f = np.full(M, 2.0e7)
        C_f = np.full(M, 2.0e-6)
        G_l = np.full(M, 5.0e-10)
        p_res = np.full(M, 4.0e7)
        q_n = np.full(M, 0.001)
        pc_n = np.full(M, 4.05e7)
        Fn = np.full(M, 1.0e5)
    c1, c0, a1, a0, guard = make_branch_pack_from_state(
        dt, I_f, R_f, C_f, G_l, p_res, q_n, pc_n, Fn)
    H_scale, Q_scale, p_scale = 4000.0, 0.05, 4.0e7
    return {
        "name": name, "guard": guard,
        "CP": 5200.0, "CM": 4100.0, "BL": 80.0, "BR": 80.0, "RL": 0.4, "RR": 0.4,
        "z_j": -2200.0, "rho_g": 1000.0 * G,
        "K": K, "kappa": kappa, "eps_q": 1e-8,
        "c1": c1, "c0": c0, "a1": a1, "a0": a0,
        "H_init": 4650.0, "q_init": q_n.copy(),
        "H_scale": H_scale, "Q_scale": Q_scale, "p_scale": p_scale,
    }


def _forward_np(s):
    q = s["q_init"].copy()
    dq = np.zeros_like(q)
    H, Qm, Qp, it, ok, rinf = solve_cluster_node(
        s["CP"], s["BL"], s["RL"], s["CM"], s["BR"], s["RR"], s["z_j"], s["rho_g"],
        q.size, s["K"], s["kappa"], s["eps_q"], s["c1"], s["c0"], s["a1"], s["a0"],
        q, dq, s["H_init"], s["H_scale"], s["Q_scale"], s["p_scale"], 1e-8, 100, 100,
    )
    return H, Qm, Qp, q, it, ok, rinf


def fd_grad(s, key, eps):
    def loss_of(sc):
        H, Qm, Qp, q, _, _, _ = _forward_np(sc)
        return H + 0.3 * Qm - 0.2 * Qp + 0.1 * q.sum()

    sc_p, sc_m = dict(s), dict(s)
    if isinstance(s[key], np.ndarray):
        vp, vm = s[key].copy(), s[key].copy()
        vp.ravel()[0] += eps
        vm.ravel()[0] -= eps
        sc_p[key], sc_m[key] = vp, vm
    else:
        sc_p[key] = s[key] + eps
        sc_m[key] = s[key] - eps
    return (loss_of(sc_p) - loss_of(sc_m)) / (2.0 * eps)


def autograd_grad(s, key):
    t = _pack_tensors(s, requires=(key,))
    H, Qm, Qp, q, stat = cluster_newton(
        t["CP"], t["CM"], t["BL"], t["BR"], t["RL"], t["RR"], t["z_j"], t["rho_g"],
        t["K"], t["kappa"], t["eps_q"], t["c1"], t["c0"], t["a1"], t["a0"],
        t["H_init"], t["q_init"], t["H_scale"], t["Q_scale"], t["p_scale"],
    )
    loss = H + 0.3 * Qm - 0.2 * Qp + 0.1 * q.sum()
    loss.backward()
    g = t[key].grad
    if g is None:
        return 0.0
    if g.ndim == 0:
        return float(g)
    return float(g.reshape(-1)[0])


def main():
    ensure_dirs()
    names = ["nominal", "stiff_storage", "reverse_flow", "near_zero", "many_perf"]
    rows = []
    n_fail = 0
    for name in names:
        s = _scenario(name)
        Hn, Qmn, Qpn, qn, it, ok, rinf = _forward_np(s)
        t = _pack_tensors(s, requires=("CP", "CM", "K"))
        H, Qm, Qp, q, stat = cluster_newton(
            t["CP"], t["CM"], t["BL"], t["BR"], t["RL"], t["RR"], t["z_j"], t["rho_g"],
            t["K"], t["kappa"], t["eps_q"], t["c1"], t["c0"], t["a1"], t["a0"],
            t["H_init"], t["q_init"], t["H_scale"], t["Q_scale"], t["p_scale"],
        )
        r, J, _, _ = reduced_residual_and_jac(
            float(H), q.detach().numpy(), s["CP"], s["BL"], s["RL"], s["CM"], s["BR"], s["RR"],
            s["z_j"], s["rho_g"], s["K"], s["kappa"], s["eps_q"], s["c1"], s["c0"], s["a1"], s["a0"],
            s["Q_scale"], s["p_scale"],
        )
        fwd_H = abs(float(H) - Hn) / max(abs(Hn), 1.0)
        resid = float(np.max(np.abs(r)))
        cond = float(np.linalg.cond(J))
        grads = {}
        smooth = name != "near_zero"
        for key, eps in (("CP", 1e-3), ("CM", 1e-3), ("K", 1e6)):
            fd = fd_grad(s, key, eps)
            ag = autograd_grad(s, key)
            ref = max(abs(fd), 1e-12)
            rel = abs(ag - fd) / ref
            abs_e = abs(ag - fd)
            grads[key] = {"autograd": ag, "fd": fd, "rel": rel, "abs": abs_e,
                          "near_zero_ref": abs(fd) < 1e-8}
            if smooth and abs(fd) >= 1e-8 and rel > 1e-4:
                n_fail += 1
            if smooth and abs(fd) < 1e-8 and abs_e > 1e-6:
                n_fail += 1
        if resid >= 1e-8:
            n_fail += 1
        if fwd_H > 1e-12:
            n_fail += 1
        row = {
            "name": name, "forward_match_H_rel": fwd_H, "resid_inf_dimless": resid,
            "newton_ok": bool(ok), "n_iter": int(stat[0]), "cond": cond,
            "tikhonov": bool(cond > COND_LIMIT), "stiff_guard": int(s["guard"]),
            "n_perf": int(s["K"].size), "q_mean": float(np.mean(qn)),
            "grads": grads, "smooth": smooth,
            "pass_resid": resid < 1e-8,
            "pass_grad": all(
                (g["rel"] < 1e-4) if (smooth and not g["near_zero_ref"]) else True
                for g in grads.values()
            ),
        }
        rows.append(row)
        print(f"{name:16s} resid={resid:.3e} cond={cond:.3e} fwdH={fwd_H:.3e} "
              f"dCP rel={grads['CP']['rel']:.3e} ok={ok}")

    metrics = {
        "n_scenarios": len(rows),
        "n_fail": n_fail,
        "status": "PASS" if n_fail == 0 else "FAIL",
        "thresholds": {"resid_inf": 1e-8, "grad_rel_smooth": 1e-4},
        "rows": rows,
        "notes": {
            "near_zero": "orifice K q|q| is C0 at q=0; FD is not a smooth derivative and is reported separately",
            "ift": "VJP on reduced (H,q) Jacobian; Tikhonov if cond>1e8",
            "dtype": "float64 forward and backward",
        },
    }
    dump_json(metrics, DATA_AUDIT_DIR / "node_validation_metrics.json")
    print(json.dumps({"status": metrics["status"], "n_fail": n_fail}, indent=2))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
