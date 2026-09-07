# -*- coding: utf-8 -*-
"""Production Tensor rollout. Same discrete formulas as moc_twin; no mini_loop gate.

Hard node outputs enter later propagation. No detach/NumPy on tensors that
must carry time gradients. Newton failure raises and blocks the train step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from ift_layer import cluster_newton, reduced_residual_and_jac
from moc_twin import _setup
from stage1_bridge import F_DEFINITION, ms

G = 9.80665


class NewtonFailedError(RuntimeError):
    pass


def _t64(x, device=None, req=False):
    if torch.is_tensor(x):
        y = x.to(dtype=torch.float64, device=device)
        return y
    return torch.tensor(x, dtype=torch.float64, device=device, requires_grad=req)


def exp_trap_weights_t(h: torch.Tensor):
    # same series/closed form as node_newton.exp_trap_weights
    small = h < 1e-3
    wB_s = 0.5 - h / 6.0 + h * h / 24.0
    wA_s = 0.5 - h / 3.0 + h * h / 8.0
    E_s = 1.0 - h + 0.5 * h * h - h * h * h / 6.0
    E = torch.exp(-h)
    p1 = (1.0 - E) / h.clamp_min(1e-30)
    wB = p1 - (p1 - E) / h.clamp_min(1e-30)
    wA = p1 - wB
    return (
        torch.where(small, E_s, E),
        torch.where(small, wA_s, wA),
        torch.where(small, wB_s, wB),
    )


def branch_coefficients_t(dt, I_f, R_f, C_f, G_l, p_res, q_n, p_c_n, F_n,
                          inertia_scheme=1, storage_scheme=0, stiff_guard=1):
    """Tensor affine maps. Discrete stiff_guard is a hard switch (reported separately)."""
    dt = _t64(dt, I_f.device)
    guard = torch.zeros((), dtype=torch.float64, device=I_f.device)
    if inertia_scheme == 0:
        c1 = R_f + I_f / dt
        c0 = -I_f * q_n / dt
    else:
        h = R_f * dt / I_f.clamp_min(1e-30)
        E, wA, wB = exp_trap_weights_t(h)
        c1 = I_f / (dt * wB.clamp_min(1e-30))
        c0 = -c1 * E * q_n - (wA / wB.clamp_min(1e-30)) * F_n
    use_exp = storage_scheme == 1
    if storage_scheme == 0 and stiff_guard:
        # discrete: if any branch would oscillate, switch that branch
        ratio = G_l * dt / C_f.clamp_min(1e-30)
        switch = (C_f <= 0) | (ratio > 1.0)
        guard = switch.to(torch.float64).sum()
        use_exp_t = switch
    else:
        use_exp_t = torch.zeros_like(C_f, dtype=torch.bool)
        if use_exp:
            use_exp_t = torch.ones_like(C_f, dtype=torch.bool)
    dn = C_f / dt + 0.5 * G_l
    a1_cn = 0.5 / dn.clamp_min(1e-30)
    a0_cn = ((C_f / dt - 0.5 * G_l) * p_c_n + 0.5 * q_n + G_l * p_res) / dn.clamp_min(1e-30)
    a1_alg = 1.0 / G_l.clamp_min(1e-30)
    a0_alg = p_res * torch.ones_like(q_n)
    h = G_l * dt / C_f.clamp_min(1e-30)
    E, wA, wB = exp_trap_weights_t(h)
    a1_ex = dt / C_f.clamp_min(1e-30) * wB
    a0_ex = p_res + (p_c_n - p_res) * E + dt / C_f.clamp_min(1e-30) * q_n * wA
    a1_e = torch.where(C_f <= 0, a1_alg, a1_ex)
    a0_e = torch.where(C_f <= 0, a0_alg, a0_ex)
    a1 = torch.where(use_exp_t, a1_e, a1_cn)
    a0 = torch.where(use_exp_t, a0_e, a0_cn)
    return c1, c0, a1, a0, guard


def f_lookup_t(Q, Re_coef, logRe_min, dlogRe, fvals):
    Re = torch.abs(Q) * Re_coef
    Re = torch.clamp(Re, min=1e-30)
    x = (torch.log10(Re) - logRe_min) / dlogRe
    n = fvals.numel()
    x = torch.clamp(x, 0.0, float(n - 1) - 1e-12)
    k = torch.floor(x).to(torch.int64)
    w = x - k.to(x.dtype)
    return fvals[k] * (1.0 - w) + fvals[k + 1] * w


def wellhead_Q_t(t, Q0, t_s, t_c):
    t = t if torch.is_tensor(t) else _t64(t, Q0.device)
    Q0 = Q0 if torch.is_tensor(Q0) else _t64(Q0, t.device)
    closed = t >= (t_s + t_c)
    xi = (t - t_s) / max(float(t_c), 1e-30)
    qmid = 0.5 * Q0 * (1.0 + torch.cos(torch.tensor(np.pi, dtype=torch.float64, device=Q0.device) * xi))
    return torch.where(t < t_s, Q0, torch.where(closed | (t_c <= 0), torch.zeros_like(Q0), qmid))


def foot_t(Hn, Qn, Ju, i, neigh, Cr):
    if abs(float(Cr) - 1.0) < 1e-15:
        return Hn[neigh], Qn[neigh], Ju[neigh]
    w = Cr
    return (1.0 - w) * Hn[i] + w * Hn[neigh], (1.0 - w) * Qn[i] + w * Qn[neigh], (1.0 - w) * Ju[i] + w * Ju[neigh]


class ProductionMOC:
    """One Tensor implementation used for forward, long-roll, and gradient gates."""

    def __init__(self, well, num, friction_model="darcy", device="cpu"):
        if friction_model not in ("none", "darcy"):
            raise ValueError("production Tensor path in this round: none/darcy")
        self.well = well
        self.num = num
        self.friction_model = friction_model
        self.device = torch.device(device)
        self.S = _setup(well, num, friction_model)
        self.grid = self.S["grid"]
        self.dt = float(self.S["dt"])
        self.F_definition = dict(F_DEFINITION)
        self.last_resid = []
        self.last_guard = 0.0
        self.n_fail = 0
        ftab = self.S["ftab"]
        self.fvals = _t64(ftab.values, self.device)
        self.logRe_min = float(ftab.logRe_min)
        self.dlogRe = float(ftab.dlogRe)
        self.Re_coef = float(self.S["Re_coef"])
        self.reset()

    def reset(self):
        S = self.S
        d = self.device
        self.H = _t64(S["H0"], d)
        self.Q = _t64(S["Q0"], d)
        self.Ju = torch.zeros_like(self.H)
        self.q = _t64(S["q0"], d)
        self.pc = _t64(S["pc0"], d)
        self.F = _t64(S["F0"], d)
        self.pK = _t64(S["pK"], d)
        self.pkappa = _t64(S["pkappa"], d)
        self.pIf = _t64(S["pIf"], d)
        self.pRf = _t64(S["pRf"], d)
        self.pCf = _t64(S["pCf"], d)
        self.pGl = _t64(S["pGl"], d)
        self.ppres = _t64(S["ppres"], d)
        self.cl_z = _t64(S["cl_z"], d)
        self.seg_B = _t64(S["seg_B"], d)
        self.seg_Rcoef = _t64(S["seg_Rcoef"], d)
        self.rho_g = _t64(S["rho_g"], d)
        self.A = _t64(S["A"], d)
        self.H_scale = float(S["H_scale"])
        self.Q_scale = float(S["Q_scale"])
        self.p_scale = float(S["p_scale"])
        self.eps_q = float(S["eps_q"])
        self.time_n = 0

    def _f(self, Qv):
        if self.S["fric_mode"] == 0:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        return f_lookup_t(Qv, self.Re_coef, self.logRe_min, self.dlogRe, self.fvals)

    def p_head(self):
        return self.rho_g * self.H[0]

    def step(self, require_ok: bool = True) -> dict:
        """Advance one time level. Hard node writes H/Q used later in this same step
        only at the node copies; subsequent steps use those values for feet."""
        grid = self.grid
        dt = self.dt
        nseg = grid.nseg
        Hn, Qn, Ju = self.H, self.Q, self.Ju
        H = self.H.clone()
        Q = self.Q.clone()
        t = (self.time_n + 1) * dt
        trap = self.S["trap"]
        # interiors
        for j in range(nseg):
            o = int(grid.seg_off[j])
            N = int(grid.seg_N[j])
            Cr = float(grid.seg_Cr[j])
            B = self.seg_B[j]
            Rc = self.seg_Rcoef[j]
            for i in range(o + 1, o + N):
                HA, QA, JA = foot_t(Hn, Qn, Ju, i, i - 1, Cr)
                HB, QB, JB = foot_t(Hn, Qn, Ju, i, i + 1, Cr)
                RA = self._f(QA) * Rc
                RB = self._f(QB) * Rc
                JuA = B * dt * self.A * JA
                JuB = B * dt * self.A * JB
                if trap:
                    RP = self._f(Qn[i]) * Rc
                    CP = HA + B * QA - 0.5 * RA * QA * torch.abs(QA) - JuA
                    CM = HB - B * QB + 0.5 * RB * QB * torch.abs(QB) + JuB
                    # explicit Q from characteristic, one substitution
                    D = CP - CM
                    twoB = 2.0 * B
                    twoR = 2.0 * RP
                    # quadratic formula matching flow_from_characteristic
                    aD = torch.abs(D)
                    disc = torch.sqrt(twoB * twoB + 2.0 * twoR * aD)
                    QP = torch.where(twoR <= 0, D / twoB, torch.sign(D) * 2.0 * aD / (twoB + disc))
                    RP = self._f(QP) * Rc
                    twoR = 2.0 * RP
                    disc = torch.sqrt(twoB * twoB + 2.0 * twoR * torch.abs(CP - CM))
                    D = CP - CM
                    QP = torch.where(twoR <= 0, D / twoB, torch.sign(D) * 2.0 * torch.abs(D) / (twoB + disc))
                    HP = CP - B * QP - 0.5 * RP * QP * torch.abs(QP)
                else:
                    CP = HA + B * QA - RA * QA * torch.abs(QA) - JuA
                    CM = HB - B * QB + RB * QB * torch.abs(QB) + JuB
                    QP = (CP - CM) / (2.0 * B)
                    HP = 0.5 * (CP + CM)
                H[i] = HP
                Q[i] = QP
        # wellhead
        Cr = float(grid.seg_Cr[0])
        B = self.seg_B[0]
        Rc = self.seg_Rcoef[0]
        i = int(grid.seg_off[0])
        HB, QB, JB = foot_t(Hn, Qn, Ju, i, i + 1, Cr)
        RB = self._f(QB) * Rc
        JuB = B * dt * self.A * JB
        QP = wellhead_Q_t(_t64(t, self.device), _t64(self.well.Q0, self.device),
                          self.well.t_s, self.well.t_c)
        if trap:
            RP = self._f(QP) * Rc
            CM = HB - B * QB + 0.5 * RB * QB * torch.abs(QB) + JuB
            HP = CM + B * QP + 0.5 * RP * QP * torch.abs(QP)
        else:
            CM = HB - B * QB + RB * QB * torch.abs(QB) + JuB
            HP = CM + B * QP
        H[i] = HP
        Q[i] = QP
        # toe
        j = nseg - 1
        o = int(grid.seg_off[j])
        N = int(grid.seg_N[j])
        Cr = float(grid.seg_Cr[j])
        B = self.seg_B[j]
        Rc = self.seg_Rcoef[j]
        i = o + N
        HA, QA, JA = foot_t(Hn, Qn, Ju, i, i - 1, Cr)
        RA = self._f(QA) * Rc
        JuA = B * dt * self.A * JA
        if trap:
            CP = HA + B * QA - 0.5 * RA * QA * torch.abs(QA) - JuA
        else:
            CP = HA + B * QA - RA * QA * torch.abs(QA) - JuA
        H[i] = CP
        Q[i] = torch.zeros((), dtype=torch.float64, device=self.device)
        # clusters — hard Newton writes H/Q that later steps read
        q_new = self.q.clone()
        pc_new = self.pc.clone()
        F_new = self.F.clone()
        resid_step = 0.0
        for jc in range(self.S["ncl"]):
            iL = grid.cluster_left_index(jc)
            iR = grid.cluster_right_index(jc)
            CrL = float(grid.seg_Cr[jc])
            BL = self.seg_B[jc]
            RcL = self.seg_Rcoef[jc]
            HA, QA, JA = foot_t(Hn, Qn, Ju, iL, iL - 1, CrL)
            RA = self._f(QA) * RcL
            JuA = BL * dt * self.A * JA
            CrR = float(grid.seg_Cr[jc + 1])
            BR = self.seg_B[jc + 1]
            RcR = self.seg_Rcoef[jc + 1]
            HB, QB, JB = foot_t(Hn, Qn, Ju, iR, iR + 1, CrR)
            RB = self._f(QB) * RcR
            JuB = BR * dt * self.A * JB
            if trap:
                RPL = self._f(Qn[iL]) * RcL
                RPR = self._f(Qn[iR]) * RcR
                CP = HA + BL * QA - 0.5 * RA * QA * torch.abs(QA) - JuA
                CM = HB - BR * QB + 0.5 * RB * QB * torch.abs(QB) + JuB
            else:
                RPL = torch.zeros((), dtype=torch.float64, device=self.device)
                RPR = torch.zeros((), dtype=torch.float64, device=self.device)
                CP = HA + BL * QA - RA * QA * torch.abs(QA) - JuA
                CM = HB - BR * QB + RB * QB * torch.abs(QB) + JuB
            m0, m1 = int(self.S["perf_off"][jc]), int(self.S["perf_off"][jc + 1])
            c1, c0, a1, a0, guard = branch_coefficients_t(
                dt, self.pIf[m0:m1], self.pRf[m0:m1], self.pCf[m0:m1], self.pGl[m0:m1],
                self.ppres[m0:m1], self.q[m0:m1], self.pc[m0:m1], self.F[m0:m1],
                self.S["inertia"], self.S["storage"], self.S["stiff_guard"],
            )
            self.last_guard += float(guard.detach())
            Hj, Qm, Qp, qj, stat = cluster_newton(
                CP, CM, BL, BR, RPL, RPR, self.cl_z[jc], self.rho_g,
                self.pK[m0:m1], self.pkappa[m0:m1], self.eps_q,
                c1, c0, a1, a0, Hn[iL], self.q[m0:m1],
                self.H_scale, self.Q_scale, self.p_scale,
                self.num.tol_dimless, self.num.max_iter, self.num.max_inner,
            )
            ok = bool(float(stat[1].detach()) > 0.5)
            rinf = float(stat[2].detach())
            resid_step = max(resid_step, rinf)
            if not ok:
                self.n_fail += 1
                if require_ok:
                    raise NewtonFailedError(
                        f"real Newton failure at step={self.time_n+1} cluster={jc} rinf={rinf}"
                    )
            H[iL] = Hj
            H[iR] = Hj
            Q[iL] = Qm
            Q[iR] = Qp
            q_new[m0:m1] = qj
            pc_new[m0:m1] = a0 + a1 * qj
            F_new[m0:m1] = c1 * qj + c0
        self.H, self.Q = H, Q
        self.q, self.pc, self.F = q_new, pc_new, F_new
        self.time_n += 1
        self.last_resid.append(resid_step)
        return {"p_head": self.p_head(), "ok": True, "resid": resid_step}

    def unroll(self, n_steps: int, require_ok: bool = True) -> torch.Tensor:
        ps = [self.p_head()]
        for _ in range(int(n_steps)):
            self.step(require_ok=require_ok)
            ps.append(self.p_head())
        return torch.stack(ps)


def production_train_step(roll: ProductionMOC, n_steps: int, target: torch.Tensor):
    """One optimizer-facing step. Real Newton failure blocks (does not inject)."""
    roll.reset()
    pred = roll.unroll(n_steps, require_ok=True)
    loss = ((pred - target) ** 2).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss; step blocked")
    return loss
