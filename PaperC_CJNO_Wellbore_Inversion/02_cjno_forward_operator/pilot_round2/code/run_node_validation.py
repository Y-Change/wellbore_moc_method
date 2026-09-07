# -*- coding: utf-8 -*-
"""Real-coefficient node coverage + physical-parameter gradient chain."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from ift_layer import COND_LIMIT, cluster_newton, reduced_residual_and_jac
from node_coeffs import collect_branch_coeffs
from node_scenarios import ALL_NAMES, scenario
from paths import MANIFEST_DIR, STAGE1_CODE, TABLE_DIR, ensure_dirs

sys.path.insert(0, str(STAGE1_CODE))
from node_newton import solve_cluster_node  # noqa: E402

G = 9.80665


def _forward_np(s):
    q = s["q_init"].copy()
    dq = np.zeros_like(q)
    H, Qm, Qp, it, ok, rinf = solve_cluster_node(
        s["CP"], s["BL"], s["RL"], s["CM"], s["BR"], s["RR"], s["z_j"], s["rho_g"],
        q.size, s["K"], s["kappa"], s["eps_q"], s["c1"], s["c0"], s["a1"], s["a0"],
        q, dq, s["H_init"], s["H_scale"], s["Q_scale"], s["p_scale"], 1e-8, 100, 100,
    )
    return H, Qm, Qp, q, it, ok, rinf


def _ift(s, requires=()):
    def t(k):
        x = torch.as_tensor(s[k], dtype=torch.float64)
        if k in requires:
            x = x.clone().requires_grad_(True)
        return x
    keys = ("CP", "CM", "BL", "BR", "RL", "RR", "z_j", "rho_g", "K", "kappa",
            "c1", "c0", "a1", "a0", "H_init", "q_init")
    ten = {k: t(k) if k not in ("eps_q",) else s[k] for k in keys}
    H, Qm, Qp, q, stat = cluster_newton(
        ten["CP"], ten["CM"], ten["BL"], ten["BR"], ten["RL"], ten["RR"],
        ten["z_j"], ten["rho_g"], ten["K"], ten["kappa"], s["eps_q"],
        ten["c1"], ten["c0"], ten["a1"], ten["a0"],
        ten["H_init"], ten["q_init"], s["H_scale"], s["Q_scale"], s["p_scale"],
    )
    return H, Qm, Qp, q, stat, ten


def _loss_np(s):
    H, Qm, Qp, q, _, _, _ = _forward_np(s)
    return H + 0.3 * Qm - 0.2 * Qp + 0.1 * q.sum()


def fd_key(s, key, eps):
    sp, sm = dict(s), dict(s)
    if isinstance(s[key], np.ndarray):
        vp, vm = s[key].copy(), s[key].copy()
        vp.ravel()[0] += eps
        vm.ravel()[0] -= eps
        sp[key], sm[key] = vp, vm
    else:
        sp[key] = s[key] + eps
        sm[key] = s[key] - eps
    return (_loss_np(sp) - _loss_np(sm)) / (2.0 * eps)


def ag_key(s, key):
    H, Qm, Qp, q, stat, ten = _ift(s, requires=(key,))
    loss = H + 0.3 * Qm - 0.2 * Qp + 0.1 * q.sum()
    loss.backward()
    g = ten[key].grad
    if g is None:
        return 0.0
    return float(g.reshape(-1)[0]) if g.ndim else float(g)


def rebuild_from_phys(s, I_f=None, Cd_scale=None):
    phys = dict(s["phys"])
    if I_f is not None:
        phys["I_f"] = np.asarray(I_f, dtype=np.float64)
    if Cd_scale is not None:
        phys["K"] = s["phys"]["K"] / (Cd_scale ** 2)
    br = collect_branch_coeffs(
        s["dt"], phys["I_f"], phys["R_f"], phys["C_f"], phys["G_l"],
        phys["p_res"], s["q_n"], s["pc_n"], s["F_n"],
        s["inertia_scheme"], s["storage_scheme"], 1,
    )
    out = dict(s)
    out["phys"] = phys
    out["K"] = np.asarray(phys["K"], dtype=np.float64)
    out["c1"], out["c0"], out["a1"], out["a0"] = br["c1"], br["c0"], br["a1"], br["a0"]
    out["stiff_guard"] = br["stiff_guard"]
    return out


def physical_fd(s, kind, eps):
    if kind == "I_f":
        If = s["phys"]["I_f"].copy()
        scale = max(abs(If[0]), 1.0)
        Ifp, Ifm = If.copy(), If.copy()
        Ifp[0] += eps * scale
        Ifm[0] -= eps * scale
        return (_loss_np(rebuild_from_phys(s, I_f=Ifp)) - _loss_np(rebuild_from_phys(s, I_f=Ifm))) / (2 * eps * scale)
    if kind == "Cd":
        return (_loss_np(rebuild_from_phys(s, Cd_scale=1 + eps)) - _loss_np(rebuild_from_phys(s, Cd_scale=1 - eps))) / (2 * eps)
    raise KeyError(kind)


def physical_ag(s, kind):
    """Autograd through K or c1 after packing; chain starts at discrete coeffs."""
    key = "K" if kind == "Cd" else "c1"
    H, Qm, Qp, q, stat, ten = _ift(s, requires=(key,))
    loss = H + 0.3 * Qm - 0.2 * Qp + 0.1 * q.sum()
    loss.backward()
    g = ten[key].grad
    if g is None:
        return 0.0 if kind != "Cd" else (0.0, None)
    if kind == "Cd":
        return g.detach().cpu().numpy()
    return float(g.reshape(-1)[0])


def check_expect(name, s, q_sol):
    exp = s.get("expect", {})
    fails = []
    if "stiff_guard_gt" in exp and s["stiff_guard"] <= exp["stiff_guard_gt"]:
        fails.append(f"stiff_guard={s['stiff_guard']} not > {exp['stiff_guard_gt']}")
    if name == "stiff_storage":
        ratio = float(np.min(s["phys"]["G_l"] * s["dt"] / np.maximum(s["phys"]["C_f"], 1e-30)))
        if ratio <= 1.0:
            fails.append(f"G_l dt/C_f={ratio} not > 1")
        if s["stiff_guard"] <= 0:
            fails.append("stiff_guard not triggered")
    if exp.get("q_all_negative") and not np.all(q_sol < 0.0):
        fails.append(f"reverse not realized q={q_sol}")
    if "q_abs_max" in exp and float(np.max(np.abs(q_sol))) > exp["q_abs_max"]:
        fails.append(f"near-zero band missed |q|_max={np.max(np.abs(q_sol))}")
    return fails


def main():
    ensure_dirs()
    rows = []
    n_fail = 0
    for name in ALL_NAMES:
        s = scenario(name)
        Hn, Qmn, Qpn, qn, it, ok, rinf = _forward_np(s)
        H, Qm, Qp, q, stat, _ = _ift(s)
        r, J, _, _ = reduced_residual_and_jac(
            float(H), q.detach().numpy(), s["CP"], s["BL"], s["RL"], s["CM"], s["BR"], s["RR"],
            s["z_j"], s["rho_g"], s["K"], s["kappa"], s["eps_q"],
            s["c1"], s["c0"], s["a1"], s["a0"], s["Q_scale"], s["p_scale"],
        )
        resid = float(np.max(np.abs(r)))
        cond = float(np.linalg.cond(J)) if np.isfinite(J).all() else float("inf")
        q_sol = q.detach().numpy()
        exp_fail = check_expect(name, s, q_sol)
        smooth = name not in ("near_zero",)
        grads = {}
        for key, eps in (("CP", 1e-3), ("CM", 1e-3), ("K", 1e6)):
            fd = fd_key(s, key, eps)
            ag = ag_key(s, key)
            ref = max(abs(fd), 1e-12)
            rel = abs(ag - fd) / ref
            grads[key] = {"autograd": ag, "fd": fd, "rel": rel, "abs": abs(ag - fd),
                          "near_zero_ref": abs(fd) < 1e-8}
            if smooth and abs(fd) >= 1e-8 and rel > 1e-4:
                n_fail += 1
        # physical chain on nominal/hetero
        phys_g = {}
        if name in ("nominal", "hetero_perf"):
            for kind, eps in (("I_f", 1e-4), ("Cd", 1e-4)):
                fd = physical_fd(s, kind, eps)
                ag = physical_ag(s, kind)
                # Cd AG is dL/dK[0]; FD is dL/dCd_scale. Convert FD to dL/dK via K ∝ Cd_scale^{-2}
                if kind == "Cd":
                    # all perfs scaled: dK/ds = -2 K, dL/ds = (dL/dK)·(dK/ds)
                    ag_on_s = float(np.sum(np.asarray(ag) * (-2.0 * s["K"])))
                    relp = abs(ag_on_s - fd) / max(abs(fd), 1e-12)
                    phys_g[kind] = {"fd_dL_dCdscale": fd,
                                    "ag_dL_dCdscale": ag_on_s, "rel": relp}
                    if abs(fd) >= 1e-8 and relp > 1e-4:
                        n_fail += 1
                else:
                    # I_f FD vs dL/dc1 * dc1/dI_f ; report both, require FD nonzero
                    phys_g[kind] = {"fd_dL_dIf_scaled": fd, "ag_dL_dc1_0": ag}
                    if abs(fd) < 1e-14:
                        n_fail += 1
                        phys_g[kind]["chain_dead"] = True
        if resid >= 1e-8:
            n_fail += 1
        if abs(float(H) - Hn) / max(abs(Hn), 1.0) > 1e-12:
            n_fail += 1
        if exp_fail:
            n_fail += 1
        row = {
            "name": name, "newton_ok": bool(ok), "resid_inf": resid, "cond": cond,
            "tikhonov": bool(cond > COND_LIMIT), "stiff_guard": int(s["stiff_guard"]),
            "q": q_sol.tolist(), "q_mean": float(q_sol.mean()),
            "expect_fail": exp_fail, "grads": grads, "phys_chain": phys_g,
            "pass_resid": resid < 1e-8,
            "note_qq_abs": "q|q| is C1 at 0 (deriv 2|q| continuous); d2 not C0. "
                           "Regularized tortuosity is C^inf. stiff_guard switch is discrete.",
        }
        rows.append(row)
        print(f"{name:20s} resid={resid:.3e} guard={s['stiff_guard']} "
              f"qmean={q_sol.mean():+.3e} ok={ok} exp={exp_fail or 'ok'}")

    coverage = {
        "stiff_guard_triggered": any(r["stiff_guard"] > 0 for r in rows),
        "reverse_realized": any(r["name"] == "reverse_flow" and not r["expect_fail"] for r in rows),
        "near_zero_realized": any(r["name"] == "near_zero" and not r["expect_fail"] for r in rows),
        "hetero_covered": any(r["name"] == "hetero_perf" for r in rows),
        "friction_cr_branch": any(r["name"] == "cr_lt1_friction" for r in rows),
        "storage_exp_branch": any(r["name"] == "storage_exp_branch" for r in rows),
        "tikhonov_seen": any(r["tikhonov"] for r in rows),
    }
    cov_fail = [k for k, v in coverage.items() if not v]
    n_fail += len(cov_fail)
    metrics = {
        "n_fail": n_fail,
        "status": "PASS" if n_fail == 0 else "FAIL",
        "thresholds": {"resid_inf": 1e-8, "grad_rel_smooth": 1e-4},
        "coverage": coverage,
        "coverage_fail": cov_fail,
        "rows": rows,
    }
    dump_json(metrics, MANIFEST_DIR / "branch_coverage.json")
    dump_json(metrics, TABLE_DIR / "node_gradient_validation.json")
    dump_json(metrics, MANIFEST_DIR / "node_gradient_validation.json")
    print("status", metrics["status"], "n_fail", n_fail, "coverage_fail", cov_fail)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
