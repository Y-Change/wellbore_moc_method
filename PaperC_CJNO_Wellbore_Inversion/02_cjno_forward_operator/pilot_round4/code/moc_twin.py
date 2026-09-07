# -*- coding: utf-8 -*-
"""NumPy discrete twin of Stage-1 MOC: none / Darcy / zvb_rec.

Uses Stage-1 primitives for grid, IC, f, wellhead Q, branch maps, Newton, and
the frozen kernel. Same-source zero error is implementation consistency, not
an independent physics proof.

Archives ALL clusters: flat per-perf arrays + offsets, true IC, time, left/right
traces, spatial H/Q, and ZVB memory. Does not mark full state from cluster-0 only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from kernel_loader import FrozenKernel, load_frozen_kernel
from stage1_bridge import (
    F_DEFINITION, SHARED_PRIMITIVES, branch_coefficients, build_grid,
    flow_from_characteristic, mk, ms, solve_cluster_node,
)

G = 9.80665


def _foot(Hn, Qn, Ju, i, neigh, Cr):
    if Cr == 1.0:
        return float(Hn[neigh]), float(Qn[neigh]), float(Ju[neigh])
    w = float(Cr)
    return (
        (1.0 - w) * float(Hn[i]) + w * float(Hn[neigh]),
        (1.0 - w) * float(Qn[i]) + w * float(Qn[neigh]),
        (1.0 - w) * float(Ju[i]) + w * float(Ju[neigh]),
    )


@dataclass
class TwinResult:
    t: np.ndarray
    p_head: np.ndarray
    Q_head: np.ndarray
    H_final: np.ndarray
    Q_final: np.ndarray
    q_flat: np.ndarray
    pc_flat: np.ndarray
    F_flat: np.ndarray
    perf_off: np.ndarray
    q_hist: np.ndarray
    pc_hist: np.ndarray
    F_hist: np.ndarray
    H_left: np.ndarray
    H_right: np.ndarray
    Q_left: np.ndarray
    Q_right: np.ndarray
    H0: np.ndarray
    Q0: np.ndarray
    q0: np.ndarray
    pc0: np.ndarray
    F0: np.ndarray
    z_final: np.ndarray
    Ju_final: np.ndarray
    n_fail: int
    newton_calls: int
    resid_max: float
    dt: float
    grid_summary: dict
    friction_model: str
    friction_time_scheme: str
    kernel_audit: dict
    F_definition: dict = field(default_factory=lambda: dict(F_DEFINITION))
    shared_primitives: List[str] = field(default_factory=lambda: list(SHARED_PRIMITIVES))
    notes: List[str] = field(default_factory=list)
    full_state_exported: bool = True


def _build_kernels(well, grid, ss, Re_coef, M, frozen: FrozenKernel):
    nseg = grid.nseg
    kern_E = np.ones((nseg, M))
    kern_Phi = np.zeros((nseg, M))
    kern_w = np.zeros(nseg)
    seg_kernel = []
    seg_Re = np.abs(ss["Qseg"]) * Re_coef
    for j in range(nseg):
        Re_j = float(seg_Re[j])
        m_, n_ = mk.kernel_for_reynolds(Re_j, frozen.fz, frozen.fp)
        rk = mk.RecursiveKernel.build(m_, n_, well.nu, well.D, grid.dt)
        kern_E[j] = rk.E
        kern_Phi[j] = rk.Phi
        kern_w[j] = rk.w
        seg_kernel.append("zielke" if Re_j < mk.RE_LAMINAR_TURBULENT else "vardy_brown")
    return {
        "E": kern_E, "Phi": kern_Phi, "w": kern_w,
        "seg_Re": seg_Re, "seg_kernel": seg_kernel, "M": M,
    }


def setup(well, num, friction_model: str, frozen: Optional[FrozenKernel] = None):
    A = well.A
    g = well.g
    rho_g = well.rho * g
    grid = build_grid(well.L, well.a, well.x_clusters, num.Nx_ref, num.Cr_target)
    dt = grid.dt
    period = 4.0 * well.L / well.a
    T_total = num.T_total if num.T_total is not None else well.t_s + num.n_periods * period
    n_steps = int(math.ceil(T_total / dt))
    fric_mode = ms.FRICTION_MODES[friction_model]
    ftab = ms.FrictionTable.build(well.roughness / well.D)
    if fric_mode == 0:
        ftab = type(ftab)(logRe_min=ftab.logRe_min, dlogRe=ftab.dlogRe, values=np.zeros_like(ftab.values))
    Re_coef = well.D / (well.nu * A)
    nseg = grid.nseg
    ncl = grid.n_clusters
    seg_a = np.full(nseg, well.a)
    seg_B = seg_a / (g * A)
    seg_Rcoef = seg_a * dt / (2.0 * g * well.D * A * A)
    eps_q = num.eps_q_rel * max(abs(well.Q0), 1e-6)
    ss = ms.solve_steady_state(well, grid, ftab, eps_q)
    H0 = np.empty(grid.n_nodes)
    Q0arr = np.empty(grid.n_nodes)
    Hstart = ss["H_wh"]
    for j in range(nseg):
        o = grid.seg_off[j]
        N = grid.seg_N[j]
        Qj = ss["Qseg"][j]
        Q0arr[o:o + N + 1] = Qj
        drop_cell = ms._steady_head_drop(Qj, grid.seg_dx[j], ftab, Re_coef, well.D, A, g)
        H0[o:o + N + 1] = Hstart - drop_cell * np.arange(N + 1)
        Hstart = H0[o + N]
    if well.toe_bc == "dead_end":
        Q0arr[-1] = 0.0
    cl_z = np.array([well.z_of_x(c.x) for c in well.clusters], dtype=float)
    cl_nperf = np.array([c.nperf for c in well.clusters], dtype=np.int64)
    perf_off = np.zeros(ncl + 1, dtype=np.int64)
    if ncl:
        perf_off[1:] = np.cumsum(cl_nperf)
    ntot = int(perf_off[-1])

    def cat(attr):
        return np.concatenate([getattr(c, attr) for c in well.clusters]) if ncl else np.zeros(0)

    pK, pkappa, pIf, pRf, pCf, pGl = (cat(a) for a in ("K", "kappa", "I_f", "R_f", "C_f", "G_l"))
    ppres = np.concatenate([np.full(c.nperf, c.p_res) for c in well.clusters]) if ncl else np.zeros(0)
    q0 = np.concatenate(ss["qs"]) if ncl else np.zeros(0)
    pc0 = ppres + q0 / pGl if ncl else np.zeros(0)
    pin0 = pc0 + pRf * q0 if ncl else np.zeros(0)
    F0 = pin0 - pc0
    Q_scale = max(abs(well.Q0), A * 0.1)
    H_scale = float(seg_B[0]) * Q_scale
    p_scale = rho_g * H_scale
    kern = None
    kernel_audit = {"used": False, "used_online_refit": False}
    if fric_mode == 2:
        frozen = frozen or load_frozen_kernel()
        if frozen.used_online_refit:
            raise RuntimeError("online refit is forbidden")
        kern = _build_kernels(well, grid, ss, Re_coef, int(num.kernel_M), frozen)
        kernel_audit = dict(frozen.audit)
        kernel_audit["used"] = True
        kernel_audit["seg_kernel"] = kern["seg_kernel"]
        kernel_audit["seg_Re"] = kern["seg_Re"].tolist()
    elif fric_mode == 3:
        raise RuntimeError("zvb_direct / fit-kernel branch is forbidden in Round 4")
    return {
        "grid": grid, "dt": dt, "n_steps": n_steps, "T_total": T_total, "period": period,
        "fric_mode": fric_mode, "ftab": ftab, "Re_coef": Re_coef, "A": A, "g": g, "rho_g": rho_g,
        "seg_B": seg_B, "seg_a": seg_a, "seg_Rcoef": seg_Rcoef,
        "H0": H0, "Q0": Q0arr, "ss": ss,
        "cl_z": cl_z, "perf_off": perf_off, "ncl": ncl, "ntot": ntot,
        "pK": pK, "pkappa": pkappa, "pIf": pIf, "pRf": pRf, "pCf": pCf, "pGl": pGl, "ppres": ppres,
        "q0": q0, "pc0": pc0, "F0": F0,
        "H_scale": H_scale, "Q_scale": Q_scale, "p_scale": p_scale, "eps_q": eps_q,
        "trap": 1 if num.friction_time_scheme == "trapezoidal" else 0,
        "inertia": ms.INERTIA_SCHEMES[num.inertia_scheme],
        "storage": ms.STORAGE_SCHEMES[num.storage_scheme],
        "stiff_guard": 1 if num.stiff_guard else 0,
        "kern": kern, "kernel_audit": kernel_audit, "M": int(num.kernel_M),
        "frozen": frozen,
    }


def run_twin(well, num, friction_model: str = "darcy", frozen: Optional[FrozenKernel] = None,
             require_ok: bool = False) -> TwinResult:
    if friction_model not in ("none", "darcy", "zvb_rec"):
        raise ValueError(f"twin supports none/darcy/zvb_rec; got {friction_model}")
    S = setup(well, num, friction_model, frozen=frozen)
    grid = S["grid"]
    dt = S["dt"]
    n_steps = S["n_steps"]
    H = S["H0"].copy()
    Q = S["Q0"].copy()
    Ju = np.zeros_like(H)
    q = S["q0"].copy()
    pc = S["pc0"].copy()
    Fb = S["F0"].copy()
    M = S["M"]
    z = np.zeros((M, grid.n_nodes))
    V = Q / S["A"]
    nseg = grid.nseg
    ncl = S["ncl"]
    ntot = S["ntot"]
    notes = [
        "Cr<1 uses linear space-line interpolation at characteristic feet (same as moc_solver).",
        "If L_j/a < dt, Stage-1 build_grid reduces dt so max Cr<=1; this twin does not clip delay to 1 step.",
        f"F definition: {F_DEFINITION['discrete_def']}",
        "Same-source zero error vs Stage-1 is implementation consistency, not independent physics.",
    ]
    if grid.dt_reduced:
        notes.append("grid.dt_reduced=True: global dt was reduced so max Cr<=1.")
    if S["fric_mode"] == 2:
        notes.append("ZVB recursive memory updates Ju after each step and feeds the next C±.")
    t = np.zeros(n_steps + 1)
    p_head = np.zeros(n_steps + 1)
    Q_head = np.zeros(n_steps + 1)
    p_head[0] = S["rho_g"] * H[0]
    Q_head[0] = Q[0]
    q_hist = np.zeros((n_steps + 1, ntot))
    pc_hist = np.zeros((n_steps + 1, ntot))
    F_hist = np.zeros((n_steps + 1, ntot))
    H_left = np.zeros((n_steps + 1, ncl))
    H_right = np.zeros((n_steps + 1, ncl))
    Q_left = np.zeros((n_steps + 1, ncl))
    Q_right = np.zeros((n_steps + 1, ncl))

    def _record(k):
        q_hist[k] = q
        pc_hist[k] = pc
        F_hist[k] = Fb
        for jc in range(ncl):
            iL = grid.cluster_left_index(jc)
            iR = grid.cluster_right_index(jc)
            H_left[k, jc] = H[iL]
            H_right[k, jc] = H[iR]
            Q_left[k, jc] = Q[iL]
            Q_right[k, jc] = Q[iR]

    _record(0)
    n_fail = 0
    newton_calls = 0
    resid_max = 0.0
    trap = S["trap"]
    ftab = S["ftab"]
    Re_coef = S["Re_coef"]
    A = S["A"]

    def f_of(Qv):
        if S["fric_mode"] == 0:
            return 0.0
        return float(ms._f_lookup(float(Qv), Re_coef, ftab.logRe_min, ftab.dlogRe, ftab.values))

    for n in range(1, n_steps + 1):
        t[n] = n * dt
        Hn = H.copy()
        Qn = Q.copy()
        Jun = Ju.copy()
        for j in range(nseg):
            o = int(grid.seg_off[j])
            N = int(grid.seg_N[j])
            Cr = float(grid.seg_Cr[j])
            B = float(S["seg_B"][j])
            Rc = float(S["seg_Rcoef"][j])
            for i in range(o + 1, o + N):
                HA, QA, JA = _foot(Hn, Qn, Jun, i, i - 1, Cr)
                HB, QB, JB = _foot(Hn, Qn, Jun, i, i + 1, Cr)
                RA = f_of(QA) * Rc
                RB = f_of(QB) * Rc
                JuA = B * dt * A * JA
                JuB = B * dt * A * JB
                if trap:
                    RP = f_of(Qn[i]) * Rc
                    CP = HA + B * QA - 0.5 * RA * QA * abs(QA) - JuA
                    CM = HB - B * QB + 0.5 * RB * QB * abs(QB) + JuB
                    QP = flow_from_characteristic(CP - CM, 2.0 * B, 2.0 * RP)
                    if S["fric_mode"] != 0:
                        RP = f_of(QP) * Rc
                        QP = flow_from_characteristic(CP - CM, 2.0 * B, 2.0 * RP)
                    HP = CP - B * QP - 0.5 * RP * QP * abs(QP)
                else:
                    CP = HA + B * QA - RA * QA * abs(QA) - JuA
                    CM = HB - B * QB + RB * QB * abs(QB) + JuB
                    QP = (CP - CM) / (2.0 * B)
                    HP = 0.5 * (CP + CM)
                H[i] = HP
                Q[i] = QP
        Cr = float(grid.seg_Cr[0])
        B = float(S["seg_B"][0])
        Rc = float(S["seg_Rcoef"][0])
        i = int(grid.seg_off[0])
        HB, QB, JB = _foot(Hn, Qn, Jun, i, i + 1, Cr)
        RB = f_of(QB) * Rc
        JuB = B * dt * A * JB
        QP = float(ms._wellhead_Q(t[n], well.Q0, well.t_s, well.t_c, ms.RAMP_TYPES[well.ramp]))
        if trap:
            RP = f_of(QP) * Rc
            CM = HB - B * QB + 0.5 * RB * QB * abs(QB) + JuB
            HP = CM + B * QP + 0.5 * RP * QP * abs(QP)
        else:
            CM = HB - B * QB + RB * QB * abs(QB) + JuB
            HP = CM + B * QP
        H[i] = HP
        Q[i] = QP
        j = nseg - 1
        o = int(grid.seg_off[j])
        N = int(grid.seg_N[j])
        Cr = float(grid.seg_Cr[j])
        B = float(S["seg_B"][j])
        Rc = float(S["seg_Rcoef"][j])
        i = o + N
        HA, QA, JA = _foot(Hn, Qn, Jun, i, i - 1, Cr)
        RA = f_of(QA) * Rc
        JuA = B * dt * A * JA
        if trap:
            CP = HA + B * QA - 0.5 * RA * QA * abs(QA) - JuA
        else:
            CP = HA + B * QA - RA * QA * abs(QA) - JuA
        H[i] = CP
        Q[i] = 0.0
        c1 = np.zeros(ntot)
        c0 = np.zeros(ntot)
        a1 = np.zeros(ntot)
        a0 = np.zeros(ntot)
        q_work = q.copy()
        for jc in range(ncl):
            iL = grid.cluster_left_index(jc)
            iR = grid.cluster_right_index(jc)
            CrL = float(grid.seg_Cr[jc])
            BL = float(S["seg_B"][jc])
            RcL = float(S["seg_Rcoef"][jc])
            HA, QA, JA = _foot(Hn, Qn, Jun, iL, iL - 1, CrL)
            RA = f_of(QA) * RcL
            JuA = BL * dt * A * JA
            CrR = float(grid.seg_Cr[jc + 1])
            BR = float(S["seg_B"][jc + 1])
            RcR = float(S["seg_Rcoef"][jc + 1])
            HB, QB, JB = _foot(Hn, Qn, Jun, iR, iR + 1, CrR)
            RB = f_of(QB) * RcR
            JuB = BR * dt * A * JB
            if trap:
                RPL = f_of(Qn[iL]) * RcL
                RPR = f_of(Qn[iR]) * RcR
                CP = HA + BL * QA - 0.5 * RA * QA * abs(QA) - JuA
                CM = HB - BR * QB + 0.5 * RB * QB * abs(QB) + JuB
            else:
                RPL = 0.0
                RPR = 0.0
                CP = HA + BL * QA - RA * QA * abs(QA) - JuA
                CM = HB - BR * QB + RB * QB * abs(QB) + JuB
            m0, m1 = int(S["perf_off"][jc]), int(S["perf_off"][jc + 1])
            nperf = m1 - m0
            for m in range(m0, m1):
                cc1, cc0, aa1, aa0, _g = branch_coefficients(
                    dt, S["pIf"][m], S["pRf"][m], S["pCf"][m], S["pGl"][m], S["ppres"][m],
                    q[m], pc[m], Fb[m], S["inertia"], S["storage"], S["stiff_guard"],
                )
                c1[m], c0[m], a1[m], a0[m] = cc1, cc0, aa1, aa0
            newton_calls += 1
            Hj, Qm, Qp, it, ok, r_inf = solve_cluster_node(
                CP, BL, RPL, CM, BR, RPR, S["cl_z"][jc], S["rho_g"], nperf,
                S["pK"][m0:m1], S["pkappa"][m0:m1], S["eps_q"],
                c1[m0:m1], c0[m0:m1], a1[m0:m1], a0[m0:m1], q_work[m0:m1],
                np.zeros(nperf), Hn[iL], S["H_scale"], S["Q_scale"], S["p_scale"],
                num.tol_dimless, num.max_iter, num.max_inner,
            )
            resid_max = max(resid_max, float(r_inf))
            if not ok:
                n_fail += 1
                if require_ok:
                    raise RuntimeError(f"twin Newton fail step={n} cluster={jc} rinf={r_inf}")
            H[iL] = Hj
            H[iR] = Hj
            Q[iL] = Qm
            Q[iR] = Qp
            for m in range(m0, m1):
                q[m] = q_work[m]
                pc[m] = a0[m] + a1[m] * q[m]
                pin = pc[m] + c1[m] * q[m] + c0[m]
                Fb[m] = pin - pc[m]
        if S["fric_mode"] == 2:
            V_prev = V
            V = Q / A
            dV = V - V_prev
            kern = S["kern"]
            for j in range(nseg):
                o = int(grid.seg_off[j])
                N = int(grid.seg_N[j])
                sl = slice(o, o + N + 1)
                z[:, sl] = kern["E"][j, :, None] * z[:, sl] + kern["Phi"][j, :, None] * dV[sl]
                Ju[sl] = kern["w"][j] * z[:, sl].sum(axis=0)
        p_head[n] = S["rho_g"] * H[0]
        Q_head[n] = Q[0]
        _record(n)
    return TwinResult(
        t=t, p_head=p_head, Q_head=Q_head, H_final=H, Q_final=Q,
        q_flat=q, pc_flat=pc, F_flat=Fb, perf_off=S["perf_off"].copy(),
        q_hist=q_hist, pc_hist=pc_hist, F_hist=F_hist,
        H_left=H_left, H_right=H_right, Q_left=Q_left, Q_right=Q_right,
        H0=S["H0"].copy(), Q0=S["Q0"].copy(), q0=S["q0"].copy(),
        pc0=S["pc0"].copy(), F0=S["F0"].copy(),
        z_final=z, Ju_final=Ju,
        n_fail=n_fail, newton_calls=newton_calls, resid_max=resid_max, dt=dt,
        grid_summary=grid.summary(), friction_model=friction_model,
        friction_time_scheme=num.friction_time_scheme,
        kernel_audit=S["kernel_audit"], notes=notes, full_state_exported=True,
    )


def short_window_seconds(well) -> float:
    """Valve action plus first-cluster reflection back to the wellhead."""
    x1 = float(well.x_clusters[0]) if len(well.x_clusters) else well.L
    return float(well.t_s + well.t_c + 2.0 * x1 / well.a)


def long_window_seconds(well) -> float:
    """Round-3 registered physical window."""
    period = 4.0 * well.L / well.a
    x1 = float(well.x_clusters[0]) if len(well.x_clusters) else well.L
    return float(well.t_s + well.t_c + 2.0 * period + 2.0 * x1 / well.a)


def make_num(nx: int, T: Optional[float] = None, n_periods: Optional[float] = None,
             friction="darcy", cr=1.0, max_iter=100):
    kwargs = dict(Nx_ref=int(nx), Cr_target=float(cr), friction_model=friction,
                  node_decim=1, do_energy=False, snap_every=0, max_iter=int(max_iter))
    if T is not None:
        kwargs["T_total"] = float(T)
        kwargs["n_periods"] = 0.25
    else:
        kwargs["n_periods"] = float(n_periods if n_periods is not None else 2.0)
    return ms.NumericsSpec(**kwargs)


def grid_inventory(well, nx: int, cr: float = 1.0) -> dict:
    grid = build_grid(well.L, well.a, well.x_clusters, int(nx), float(cr))
    return {
        "Nx_ref": int(nx),
        "n_nodes": int(grid.n_nodes),
        "nseg": int(grid.nseg),
        "seg_N": grid.seg_N.tolist(),
        "seg_dx": grid.seg_dx.tolist(),
        "seg_Cr": grid.seg_Cr.tolist(),
        "seg_len": grid.seg_len.tolist(),
        "dt": float(grid.dt),
        "dt_reduced": bool(grid.dt_reduced),
        "Cr0": float(grid.seg_Cr[0]),
        "Cr0_is_one": bool(abs(grid.seg_Cr[0] - 1.0) < 1e-12),
        "note": "Nx_ref is not the total cell count",
    }
