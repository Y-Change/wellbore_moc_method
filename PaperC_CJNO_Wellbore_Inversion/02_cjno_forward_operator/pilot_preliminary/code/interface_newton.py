# -*- coding: utf-8 -*-
"""Differentiable implicit cluster node (pilot prototype).

Unknowns after eliminating affine constraints
--------------------------------------------
Pressure-trace equality is already H^- = H^+ = H.
C+/C− give Q^-(H), Q^+(H) explicitly (trapezoidal friction).
Branch ODEs give p_c = a0 + a1 q,  p_in = p_c + c1 q + c0.
Reduced independent variables ũ = (H, q_1, …, q_M).
Reduced residuals (same SI families as node_newton.py):

    r0 = Q^-(H) − Q^+(H) − Σ q_m
    r_m = p_w(H) − K_m q|q| − κ_m φ(q) − (c1 q + c0) − (a0 + a1 q)

IFT is applied to this (M+1) system in float64.  Forward Newton calls the
Stage-1 solver so the discrete root matches MOC.  Backward uses a linear
solve on J, never an explicit inverse.  cond(J) > 1e8 triggers Tikhonov.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import STAGE1_CODE

sys.path.insert(0, str(STAGE1_CODE))
from node_newton import (  # noqa: E402
    branch_coefficients, cluster_mass_function, flow_from_characteristic,
    perf_dresidual, perf_residual, solve_cluster_node, tort_phi,
)

G = 9.80665
COND_LIMIT = 1.0e8
TIKHONOV_LAMBDA = 1.0e-8


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def reduced_residual_and_jac(H, q, CP, BL, RL, CM, BR, RR, z_j, rho_g,
                             K, kappa, eps_q, c1, c0, a1, a0,
                             Q_scale, p_scale):
    """Dimensionless reduced residual r̃ and Jacobian ∂r̃/∂ũ (M+1)."""
    q = np.asarray(q, dtype=np.float64).ravel()
    M = q.size
    Qm = flow_from_characteristic(CP - H, BL, RL)
    Qp = flow_from_characteristic(H - CM, BR, RR)
    dQm = -1.0 / (BL + RL * abs(Qm) + 1e-30)
    dQp = 1.0 / (BR + RR * abs(Qp) + 1e-30)
    p_w = rho_g * (H - z_j)
    r = np.zeros(M + 1, dtype=np.float64)
    J = np.zeros((M + 1, M + 1), dtype=np.float64)
    r[0] = (Qm - Qp - q.sum()) / Q_scale
    J[0, 0] = (dQm - dQp) / Q_scale
    J[0, 1:] = -1.0 / Q_scale
    for m in range(M):
        r[m + 1] = perf_residual(q[m], p_w, K[m], kappa[m], eps_q,
                                 c1[m], c0[m], a1[m], a0[m]) / p_scale
        J[m + 1, 0] = rho_g / p_scale
        J[m + 1, m + 1] = perf_dresidual(q[m], K[m], kappa[m], eps_q, c1[m], a1[m]) / p_scale
    return r, J, Qm, Qp


def reduced_residual_theta_partials(H, q, CP, BL, RL, CM, BR, RR, z_j, rho_g,
                                    K, kappa, eps_q, c1, c0, a1, a0,
                                    Q_scale, p_scale) -> Dict[str, np.ndarray]:
    """∂r̃/∂θ at fixed ũ, for IFT. Keys used by the backward."""
    q = np.asarray(q, dtype=np.float64).ravel()
    M = q.size
    Qm = flow_from_characteristic(CP - H, BL, RL)
    Qp = flow_from_characteristic(H - CM, BR, RR)
    # Q = 2|D| / (B + sqrt(B^2 + 2 R |D|)) with sign
    def dQ_dD(D, B, R, Q):
        return 1.0 / (B + R * abs(Q) + 1e-30)

    def dQ_dB(D, B, R, Q):
        return -Q / (B + R * abs(Q) + 1e-30)

    def dQ_dR(D, B, R, Q):
        return -0.5 * Q * abs(Q) / (B + R * abs(Q) + 1e-30)

    dQm_dCP = dQ_dD(CP - H, BL, RL, Qm)
    dQp_dCM = -dQ_dD(H - CM, BR, RR, Qp)
    out = {
        "CP": np.zeros(M + 1), "CM": np.zeros(M + 1),
        "BL": np.zeros(M + 1), "BR": np.zeros(M + 1),
        "RL": np.zeros(M + 1), "RR": np.zeros(M + 1),
        "z_j": np.zeros(M + 1), "rho_g": np.zeros(M + 1),
        "K": np.zeros((M + 1, M)), "kappa": np.zeros((M + 1, M)),
        "c1": np.zeros((M + 1, M)), "c0": np.zeros((M + 1, M)),
        "a1": np.zeros((M + 1, M)), "a0": np.zeros((M + 1, M)),
    }
    out["CP"][0] = dQm_dCP / Q_scale
    out["CM"][0] = -dQp_dCM / Q_scale
    out["BL"][0] = dQ_dB(CP - H, BL, RL, Qm) / Q_scale
    out["BR"][0] = -dQ_dB(H - CM, BR, RR, Qp) / Q_scale
    out["RL"][0] = dQ_dR(CP - H, BL, RL, Qm) / Q_scale
    out["RR"][0] = -dQ_dR(H - CM, BR, RR, Qp) / Q_scale
    p_w = rho_g * (H - z_j)
    out["z_j"][1:] = (-rho_g) / p_scale
    out["rho_g"][1:] = (H - z_j) / p_scale
    for m in range(M):
        out["K"][m + 1, m] = -(q[m] * abs(q[m])) / p_scale
        out["kappa"][m + 1, m] = -tort_phi(q[m], eps_q) / p_scale
        out["c1"][m + 1, m] = -q[m] / p_scale
        out["c0"][m + 1, m] = -1.0 / p_scale
        out["a1"][m + 1, m] = -q[m] / p_scale
        out["a0"][m + 1, m] = -1.0 / p_scale
    return out


@dataclass
class NodeSolveStat:
    n_iter: int
    ok: bool
    resid_inf: float
    cond: float
    tikhonov: bool
    n_fail: int


class _ClusterNewtonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, CP, CM, BL, BR, RL, RR, z_j, rho_g, K, kappa, eps_q,
                c1, c0, a1, a0, H_init, q_init, H_scale, Q_scale, p_scale,
                tol, max_iter, max_inner):
        CP_n, CM_n = float(CP), float(CM)
        BL_n, BR_n, RL_n, RR_n = float(BL), float(BR), float(RL), float(RR)
        z_n, rg = float(z_j), float(rho_g)
        K_n = _as_np(K).ravel()
        kap = _as_np(kappa).ravel()
        c1n, c0n = _as_np(c1).ravel(), _as_np(c0).ravel()
        a1n, a0n = _as_np(a1).ravel(), _as_np(a0).ravel()
        q_work = _as_np(q_init).ravel().copy()
        dq_work = np.zeros_like(q_work)
        M = q_work.size
        H, Qm, Qp, nit, ok, rinf = solve_cluster_node(
            CP_n, BL_n, RL_n, CM_n, BR_n, RR_n, z_n, rg, M, K_n, kap, float(eps_q),
            c1n, c0n, a1n, a0n, q_work, dq_work, float(H_init),
            float(H_scale), float(Q_scale), float(p_scale),
            float(tol), int(max_iter), int(max_inner),
        )
        r, J, Qm2, Qp2 = reduced_residual_and_jac(
            H, q_work, CP_n, BL_n, RL_n, CM_n, BR_n, RR_n, z_n, rg,
            K_n, kap, float(eps_q), c1n, c0n, a1n, a0n,
            float(Q_scale), float(p_scale),
        )
        cond = float(np.linalg.cond(J)) if np.isfinite(J).all() else float("inf")
        tikh = cond > COND_LIMIT or not np.isfinite(cond)
        ctx.save_for_backward(torch.as_tensor([H], dtype=torch.float64),
                              torch.as_tensor(q_work, dtype=torch.float64),
                              CP.detach().to(torch.float64), CM.detach().to(torch.float64),
                              BL.detach().to(torch.float64), BR.detach().to(torch.float64),
                              RL.detach().to(torch.float64), RR.detach().to(torch.float64),
                              z_j.detach().to(torch.float64), rho_g.detach().to(torch.float64),
                              K.detach().to(torch.float64), kappa.detach().to(torch.float64),
                              c1.detach().to(torch.float64), c0.detach().to(torch.float64),
                              a1.detach().to(torch.float64), a0.detach().to(torch.float64))
        ctx.aux = dict(eps_q=float(eps_q), H_scale=float(H_scale), Q_scale=float(Q_scale),
                       p_scale=float(p_scale), Qm=float(Qm), Qp=float(Qp),
                       cond=cond, tikh=tikh, ok=bool(ok), nit=int(nit), rinf=float(rinf), J=J)
        H_t = torch.tensor(H, dtype=torch.float64)
        Qm_t = torch.tensor(Qm, dtype=torch.float64)
        Qp_t = torch.tensor(Qp, dtype=torch.float64)
        q_t = torch.as_tensor(q_work, dtype=torch.float64)
        stat = torch.tensor([float(nit), 1.0 if ok else 0.0, float(rinf), cond,
                             1.0 if tikh else 0.0], dtype=torch.float64)
        return H_t, Qm_t, Qp_t, q_t, stat

    @staticmethod
    def backward(ctx, gH, gQm, gQp, gq, gstat):
        (H_t, q_t, CP, CM, BL, BR, RL, RR, z_j, rho_g, K, kappa, c1, c0, a1, a0) = ctx.saved_tensors
        aux = ctx.aux
        H = float(H_t)
        q = _as_np(q_t)
        J = aux["J"]
        Q_scale, p_scale = aux["Q_scale"], aux["p_scale"]
        Qm, Qp = aux["Qm"], aux["Qp"]
        # dL/dũ including explicit Q(H)
        BL_n, RL_n, BR_n, RR_n = float(BL), float(RL), float(BR), float(RR)
        dQm_dH = -1.0 / (BL_n + RL_n * abs(Qm) + 1e-30)
        dQp_dH = 1.0 / (BR_n + RR_n * abs(Qp) + 1e-30)
        v = np.zeros(q.size + 1, dtype=np.float64)
        v[0] = float(gH) + float(gQm) * dQm_dH + float(gQp) * dQp_dH
        v[1:] = _as_np(gq).ravel()
        Jw = J.copy()
        if aux["tikh"]:
            Jw = Jw + TIKHONOV_LAMBDA * np.eye(Jw.shape[0])
        try:
            lam = np.linalg.solve(Jw.T, v)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(Jw.T, v, rcond=None)[0]
        parts = reduced_residual_theta_partials(
            H, q, float(CP), BL_n, RL_n, float(CM), BR_n, RR_n,
            float(z_j), float(rho_g), _as_np(K), _as_np(kappa), aux["eps_q"],
            _as_np(c1), _as_np(c0), _as_np(a1), _as_np(a0), Q_scale, p_scale,
        )
        # dL/dθ = − (∂r/∂θ)^T λ  + explicit Q(θ)|_H terms
        def vjp_scalar(name, extra=0.0):
            return torch.tensor(-float(np.dot(parts[name], lam)) + extra, dtype=torch.float64)

        def vjp_vec(name):
            return torch.as_tensor(-parts[name].T @ lam, dtype=torch.float64)

        def dQ_dD(Q, B, R):
            return 1.0 / (B + R * abs(Q) + 1e-30)

        gCP = vjp_scalar("CP", extra=float(gQm) * dQ_dD(Qm, BL_n, RL_n))
        gCM = vjp_scalar("CM", extra=float(gQp) * (-dQ_dD(Qp, BR_n, RR_n)))
        gBL = vjp_scalar("BL", extra=float(gQm) * (-Qm / (BL_n + RL_n * abs(Qm) + 1e-30)))
        gBR = vjp_scalar("BR", extra=float(gQp) * (-Qp / (BR_n + RR_n * abs(Qp) + 1e-30)))
        gRL = vjp_scalar("RL", extra=float(gQm) * (-0.5 * Qm * abs(Qm) / (BL_n + RL_n * abs(Qm) + 1e-30)))
        gRR = vjp_scalar("RR", extra=float(gQp) * (-0.5 * Qp * abs(Qp) / (BR_n + RR_n * abs(Qp) + 1e-30)))
        zeros_like = lambda t: torch.zeros_like(t)
        return (gCP, gCM, gBL, gBR, gRL, gRR, vjp_scalar("z_j"), vjp_scalar("rho_g"),
                vjp_vec("K").reshape_as(K), vjp_vec("kappa").reshape_as(kappa),
                None, vjp_vec("c1").reshape_as(c1), vjp_vec("c0").reshape_as(c0),
                vjp_vec("a1").reshape_as(a1), vjp_vec("a0").reshape_as(a0),
                None, None, None, None, None, None, None, None)


def cluster_newton(CP, CM, BL, BR, RL, RR, z_j, rho_g, K, kappa, eps_q,
                   c1, c0, a1, a0, H_init, q_init, H_scale, Q_scale, p_scale,
                   tol=1e-8, max_iter=100, max_inner=100):
    """All tensors float64.  Returns H, Qm, Qp, q, stat."""
    args = [x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float64)
            for x in (CP, CM, BL, BR, RL, RR, z_j, rho_g, K, kappa, eps_q,
                      c1, c0, a1, a0, H_init, q_init, H_scale, Q_scale, p_scale)]
    return _ClusterNewtonFn.apply(*args, float(tol), int(max_iter), int(max_inner))


class ImplicitNodeLayer(nn.Module):
    """Hard implicit node. Forward+backward stay in float64."""

    def __init__(self, tol=1e-8, max_iter=100, max_inner=100):
        super().__init__()
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.max_inner = int(max_inner)
        self.last_stats = []

    def forward_one(self, pack: Dict) -> Dict:
        H, Qm, Qp, q, stat = cluster_newton(
            pack["CP"], pack["CM"], pack["BL"], pack["BR"], pack["RL"], pack["RR"],
            pack["z_j"], pack["rho_g"], pack["K"], pack["kappa"], pack["eps_q"],
            pack["c1"], pack["c0"], pack["a1"], pack["a0"],
            pack["H_init"], pack["q_init"], pack["H_scale"], pack["Q_scale"], pack["p_scale"],
            self.tol, self.max_iter, self.max_inner,
        )
        self.last_stats.append(NodeSolveStat(int(stat[0]), bool(stat[1] > 0.5),
                                             float(stat[2]), float(stat[3]),
                                             bool(stat[4] > 0.5), 0 if stat[1] > 0.5 else 1))
        return {"H": H, "Qm": Qm, "Qp": Qp, "q": q, "sq": q.sum(), "stat": stat}

    def residual_dimless(self, pack, H, q) -> torch.Tensor:
        r, _, _, _ = reduced_residual_and_jac(
            float(H.detach()), _as_np(q), float(pack["CP"].detach()),
            float(pack["BL"].detach()), float(pack["RL"].detach()),
            float(pack["CM"].detach()), float(pack["BR"].detach()), float(pack["RR"].detach()),
            float(pack["z_j"].detach()), float(pack["rho_g"].detach()),
            _as_np(pack["K"]), _as_np(pack["kappa"]), float(pack["eps_q"]),
            _as_np(pack["c1"]), _as_np(pack["c0"]), _as_np(pack["a1"]), _as_np(pack["a0"]),
            float(pack["Q_scale"]), float(pack["p_scale"]),
        )
        return torch.as_tensor(r, dtype=torch.float64)


def make_branch_pack_from_state(dt, I_f, R_f, C_f, G_l, p_res, q_n, p_c_n, F_n,
                                inertia_scheme=1, storage_scheme=0, stiff_guard=1):
    M = len(np.atleast_1d(I_f))
    c1 = np.zeros(M); c0 = np.zeros(M); a1 = np.zeros(M); a0 = np.zeros(M)
    guard = 0
    for m in range(M):
        cc1, cc0, aa1, aa0, g = branch_coefficients(
            float(dt), float(np.atleast_1d(I_f)[m]), float(np.atleast_1d(R_f)[m]),
            float(np.atleast_1d(C_f)[m]), float(np.atleast_1d(G_l)[m]),
            float(np.atleast_1d(p_res)[m]), float(np.atleast_1d(q_n)[m]),
            float(np.atleast_1d(p_c_n)[m]), float(np.atleast_1d(F_n)[m]),
            int(inertia_scheme), int(storage_scheme), int(stiff_guard),
        )
        c1[m], c0[m], a1[m], a0[m] = cc1, cc0, aa1, aa0
        guard += int(g)
    return c1, c0, a1, a0, guard
