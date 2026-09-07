# -*- coding: utf-8 -*-
"""Deterministic MOC solver for a wellbore with multiple perforation-cluster nodes.

Governing equations (H = piezometric head, Q = flow; 攻关执行方案_v2 §2.1):
    Q_t + g A H_x + f Q|Q| / (2 D A) + J_{u,Q} = 0,      H_t + (a^2/(g A)) Q_x = 0
Compatibility equations along dx/dt = ±a with B = a/(gA) (§2.2):
    C+:  H_P + B Q_P [+ (R_P/2) Q_P|Q_P|] = C_P = H_A + B Q_A - (R_A/2)[R_A] Q_A|Q_A| - B dt J_{u,Q,A}
    C-:  H_P - B Q_P [- (R_P/2) Q_P|Q_P|] = C_M = H_B - B Q_B + (R_B/2)[R_B] Q_B|Q_B| + B dt J_{u,Q,B}
with R = f * (a dt) / (2 g D A^2) (= f dx / (2 g D A^2) at Cr = 1).  Bracketed terms
belong to the second-order (trapezoidal) friction option; without them the scheme is
the explicit "friction at the foot" form.  Both keep the ±B Q_P term (unit tested).

Grid: segment-wise dx with clusters exactly on nodes (grid.py), a single dt, linear
space-line interpolation at the feet where Cr_j < 1.

Friction branches (§2.4; never mixed):
    'none'      : frictionless (analytic goldens)
    'darcy'     : quasi-steady Darcy–Weisbach only
    'zvb_rec'   : Darcy + Zielke/Vardy–Brown via M-term recursive kernel (production)
    'zvb_direct': Darcy + Zielke/Vardy–Brown via direct history convolution (reference)
    'brunone'   : Darcy + Brunone J_u = k (V_t - a sgn(V V_x) V_x), lagged source on fixed ±a

Energy audit (§2.6) is computed inside the time loop when requested.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numba import njit

import memory_kernel as mk
from friction import FrictionTable, brunone_k
from grid import Grid, build_grid
from node_newton import (branch_coefficients, flow_from_characteristic, solve_cluster_node,
                         steady_perf_flow, tort_phi)

G_STD = 9.80665

FRICTION_MODES = {"none": 0, "darcy": 1, "zvb_rec": 2, "zvb_direct": 3, "brunone": 4}
RAMP_TYPES = {"step": 0, "cosine": 1, "linear": 2, "gauss_pulse": 3}
INERTIA_SCHEMES = {"backward_euler": 0, "exp_trapezoidal": 1}
STORAGE_SCHEMES = {"crank_nicolson": 0, "exp_trapezoidal": 1}
BRUNONE_CONVENTIONS = {"plan_v2": 0, "vitkovsky2000": 1}


# ============================================================================
# numba kernels
# ============================================================================


@njit(cache=True, fastmath=False)
def _f_lookup(Q, Re_coef, logRe_min, dlogRe, fvals):
    Re = abs(Q) * Re_coef
    if Re < 1e-30:
        Re = 1e-30
    x = (math.log10(Re) - logRe_min) / dlogRe
    n = fvals.size
    if x <= 0.0:
        return fvals[0]
    if x >= n - 1:
        return fvals[n - 1]
    k = int(x)
    w = x - k
    return fvals[k] * (1.0 - w) + fvals[k + 1] * w


@njit(cache=True, fastmath=False)
def _wellhead_Q(t, Q0, t_s, t_c, ramp):
    if ramp == 3:
        # Gaussian flow pulse of amplitude Q0 centred at t_s with width t_c (zero steady flow)
        x = (t - t_s) / t_c
        return Q0 * math.exp(-x * x) if abs(x) < 8.0 else 0.0
    if t < t_s:
        return Q0
    if ramp == 0 or t_c <= 0.0:
        return 0.0
    if t >= t_s + t_c:
        return 0.0
    x = (t - t_s) / t_c
    if ramp == 1:
        return Q0 * 0.5 * (1.0 + math.cos(math.pi * x))
    return Q0 * (1.0 - x)


@njit(cache=True, fastmath=False)
def _node_Vx(V, seg_off, seg_N, seg_dx, Vx):
    """Centered dV/dx inside segments, one-sided at segment ends."""
    nseg = seg_N.size
    for j in range(nseg):
        o = seg_off[j]
        N = seg_N[j]
        dx = seg_dx[j]
        if N == 1:
            Vx[o] = (V[o + 1] - V[o]) / dx
            Vx[o + 1] = Vx[o]
            continue
        Vx[o] = (V[o + 1] - V[o]) / dx
        Vx[o + N] = (V[o + N] - V[o + N - 1]) / dx
        for i in range(o + 1, o + N):
            Vx[i] = (V[i + 1] - V[i - 1]) / (2.0 * dx)


@njit(cache=True, fastmath=False)
def _energy_state(H, Q, q, pc, seg_off, seg_N, seg_dx, seg_a, A, rho, g, H_ref, pIf, pCf, ppres):
    e_wb = 0.0
    for j in range(seg_N.size):
        o = seg_off[j]
        N = seg_N[j]
        dx = seg_dx[j]
        aj = seg_a[j]
        cH = rho * g * g * A / (2.0 * aj * aj)
        for i in range(o, o + N + 1):
            w = dx if (i > o and i < o + N) else 0.5 * dx
            e_wb += w * (rho * Q[i] * Q[i] / (2.0 * A) + cH * (H[i] - H_ref) * (H[i] - H_ref))
    e_br = 0.0
    for m in range(q.size):
        e_br += 0.5 * pIf[m] * q[m] * q[m] + 0.5 * pCf[m] * (pc[m] - ppres[m]) * (pc[m] - ppres[m])
    return e_wb, e_br


@njit(cache=True, fastmath=False)
def _branch_power(H, q, pc, seg_off, seg_N, cl_z, perf_off, pK, pkappa, pRf, pGl, ppres, rho_g, H_ref, eps_q):
    d_perf = 0.0
    d_Rf = 0.0
    d_leak = 0.0
    p_src = 0.0
    for jc in range(cl_z.size):
        for m in range(perf_off[jc], perf_off[jc + 1]):
            d_perf += pK[m] * q[m] * q[m] * abs(q[m]) + pkappa[m] * q[m] * tort_phi(q[m], eps_q)
            d_Rf += pRf[m] * q[m] * q[m]
            d_leak += pGl[m] * (pc[m] - ppres[m]) * (pc[m] - ppres[m])
            p_src += (ppres[m] - rho_g * (H_ref - cl_z[jc])) * q[m]
    return d_perf, d_Rf, d_leak, p_src


@njit(cache=True, fastmath=False)
def _memory_storage(V, z, seg_off, seg_N, seg_dx, kern_w, kern_alpha, kern_beta, rho, A):
    """KYP storage function of the recursive kernel (§2.6 backup path):
    E_mem = rho A sum_i w_i sum_l (w/(2 alpha_l beta_l)) (z_l - alpha_l V)^2  [J].
    With it, d/dt(E_tot + E_mem) = P_ports - D_steady - D_perf - ... - rho A int (w/alpha) z^2 dx <= P_ports.
    """
    e = 0.0
    M = z.shape[0]
    for j in range(seg_N.size):
        o = seg_off[j]
        N = seg_N[j]
        dx = seg_dx[j]
        w = kern_w[j]
        if w == 0.0:
            continue
        for i in range(o, o + N + 1):
            wi = dx if (i > o and i < o + N) else 0.5 * dx
            s = 0.0
            for l in range(M):
                d = z[l, i] - kern_alpha[j, l] * V[i]
                s += w / (2.0 * kern_alpha[j, l] * kern_beta[j, l]) * d * d
            e += rho * A * wi * s
    return e


@njit(cache=True, fastmath=False)
def _stored_volume(H, seg_off, seg_N, seg_dx, seg_a, g, A):
    vs = 0.0
    for j in range(seg_N.size):
        o = seg_off[j]
        N = seg_N[j]
        dx = seg_dx[j]
        c = g * A / (seg_a[j] * seg_a[j])
        for i in range(o, o + N + 1):
            w = dx if (i > o and i < o + N) else 0.5 * dx
            vs += w * c * H[i]
    return vs


@njit(cache=True, fastmath=False)
def _record_nodes(r, H, Q, q, pc, z, seg_off, seg_N, perf_off, M, node_H, node_Qm, node_Qp, node_sq, node_pc, node_z):
    for jc in range(perf_off.size - 1):
        iL = seg_off[jc] + seg_N[jc]
        iR = seg_off[jc + 1]
        node_H[r, jc] = H[iL]
        node_Qm[r, jc] = Q[iL]
        node_Qp[r, jc] = Q[iR]
        s = 0.0
        pcs = 0.0
        for m in range(perf_off[jc], perf_off[jc + 1]):
            s += q[m]
            pcs += pc[m]
        node_sq[r, jc] = s
        node_pc[r, jc] = pcs / max(1, perf_off[jc + 1] - perf_off[jc])
        for l in range(M):
            node_z[r, jc * 2 * M + l] = z[l, iL]
            node_z[r, jc * 2 * M + M + l] = z[l, iR]


@njit(cache=True, fastmath=False)
def run_moc_core(
    # grid
    seg_off, seg_N, seg_dx, seg_Cr, seg_B, seg_a, seg_Rcoef,
    # fluid / geometry
    A, rho, g, Re_coef, f_logRe_min, f_dlogRe, f_vals,
    # friction mode & schemes
    fric_mode, fric_time_scheme, brunone_conv,
    kern_E, kern_Phi, kern_w, kern_alpha, kern_beta, Wbar, seg_kb,
    # clusters
    cl_z, cl_nperf, perf_off, pK, pkappa, pIf, pRf, pCf, pGl, ppres,
    inertia_scheme, storage_scheme, stiff_guard, eps_q,
    # boundary / time
    Q0, t_s, t_c, ramp, toe_bc, H_toe, dt, n_steps,
    # initial state
    H0, Qs0, z0, q0, pc0, pin0,
    # scales / tolerances
    H_scale, Q_scale, p_scale, tol_dimless, max_iter, max_inner,
    # energy
    do_energy, H_ref,
    # recording
    node_decim, snap_every, n_fail_max,
):
    n_nodes = H0.size
    nseg = seg_N.size
    ncl = cl_nperf.size
    M = kern_E.shape[1]
    nperf_tot = pK.size

    H = H0.copy()
    Q = Qs0.copy()
    Hn = H.copy()
    Qn = Q.copy()
    V = Q / A
    V_prev = V.copy()          # V^{n-1} for Brunone
    Vx = np.zeros(n_nodes)
    z = z0.copy()              # (M, n_nodes)
    Ju = np.zeros(n_nodes)     # unsteady friction acceleration at nodes (m/s^2), level n
    n_hist = n_steps + 1 if fric_mode == 3 else 1
    hist = np.zeros((n_hist, n_nodes))

    q = q0.copy()
    pc = pc0.copy()
    pin = pin0.copy()
    Fb = pin - pc                 # (p_in - p_c)^n
    q_work = np.zeros(nperf_tot)
    dq_work = np.zeros(nperf_tot)
    c1 = np.zeros(nperf_tot)
    c0 = np.zeros(nperf_tot)
    a1 = np.zeros(nperf_tot)
    a0 = np.zeros(nperf_tot)

    # ---- recording buffers
    p_head = np.zeros(n_steps + 1)
    Q_head = np.zeros(n_steps + 1)
    n_rec = n_steps // node_decim + 1
    node_H = np.zeros((n_rec, ncl))
    node_Qm = np.zeros((n_rec, ncl))
    node_Qp = np.zeros((n_rec, ncl))
    node_sq = np.zeros((n_rec, ncl))
    node_pc = np.zeros((n_rec, ncl))
    node_z = np.zeros((n_rec, ncl * 2 * M))
    n_snap = (n_steps // snap_every + 1) if snap_every > 0 else 0
    snap_H = np.zeros((n_snap, n_nodes))
    snap_Q = np.zeros((n_snap, n_nodes))
    snap_step = np.zeros(n_snap, dtype=np.int64)
    E_wb = np.zeros(n_steps + 1)
    E_br = np.zeros(n_steps + 1)
    E_mem = np.zeros(n_steps + 1)
    W_port = np.zeros(n_steps + 1)
    W_wall_s = np.zeros(n_steps + 1)
    W_wall_u = np.zeros(n_steps + 1)
    W_perf = np.zeros(n_steps + 1)
    W_Rf = np.zeros(n_steps + 1)
    W_leak = np.zeros(n_steps + 1)
    W_src = np.zeros(n_steps + 1)
    mass_resid = np.zeros(n_steps + 1)
    newton_iters = np.zeros(n_steps + 1, dtype=np.int64)
    newton_resid = np.zeros(n_steps + 1)
    fail_log = np.zeros((n_fail_max, 2), dtype=np.int64)
    n_fail = 0
    n_guard = 0
    max_newton_iter = 0

    rho_g = rho * g

    # ---- initial records
    p_head[0] = rho_g * H[0]
    Q_head[0] = Q[0]
    _record_nodes(0, H, Q, q, pc, z, seg_off, seg_N, perf_off, M, node_H, node_Qm, node_Qp, node_sq, node_pc, node_z)
    if snap_every > 0:
        snap_H[0, :] = H
        snap_Q[0, :] = Q
        snap_step[0] = 0
    if do_energy:
        e_wb, e_br = _energy_state(H, Q, q, pc, seg_off, seg_N, seg_dx, seg_a, A, rho, g, H_ref, pIf, pCf, ppres)
        E_wb[0] = e_wb
        E_br[0] = e_br
        if fric_mode == 2:
            E_mem[0] = _memory_storage(V, z, seg_off, seg_N, seg_dx, kern_w, kern_alpha, kern_beta, rho, A)
    d_perf_n, d_Rf_n, d_leak_n, p_src_n = _branch_power(H, q, pc, seg_off, seg_N, cl_z, perf_off, pK, pkappa, pRf, pGl,
                                                        ppres, rho_g, H_ref, eps_q)
    P_wh_n = rho_g * (H[0] - H_ref) * Q[0] - rho_g * (H[n_nodes - 1] - H_ref) * Q[n_nodes - 1]
    sq_tot_n = 0.0
    for m in range(nperf_tot):
        sq_tot_n += q[m]
    Vst_n = _stored_volume(H, seg_off, seg_N, seg_dx, seg_a, g, A)

    # ============================ time loop ============================
    for n in range(1, n_steps + 1):
        t = n * dt
        Hn[:] = H
        Qn[:] = Q
        if fric_mode == 4:
            _node_Vx(V, seg_off, seg_N, seg_dx, Vx)
        d_wall_s = 0.0
        d_wall_u = 0.0

        # ---------------- interior nodes of every segment ----------------
        for j in range(nseg):
            o = seg_off[j]
            N = seg_N[j]
            Cr = seg_Cr[j]
            B = seg_B[j]
            Rc = seg_Rcoef[j]
            aj = seg_a[j]
            kb = seg_kb[j]
            for i in range(o + 1, o + N):
                # feet
                if Cr == 1.0:
                    HA = Hn[i - 1]; QA = Qn[i - 1]; JA = Ju[i - 1]
                    HB = Hn[i + 1]; QB = Qn[i + 1]; JB = Ju[i + 1]
                else:
                    HA = (1.0 - Cr) * Hn[i] + Cr * Hn[i - 1]
                    QA = (1.0 - Cr) * Qn[i] + Cr * Qn[i - 1]
                    JA = (1.0 - Cr) * Ju[i] + Cr * Ju[i - 1]
                    HB = (1.0 - Cr) * Hn[i] + Cr * Hn[i + 1]
                    QB = (1.0 - Cr) * Qn[i] + Cr * Qn[i + 1]
                    JB = (1.0 - Cr) * Ju[i] + Cr * Ju[i + 1]
                if fric_mode == 4:
                    # Brunone at the feet (lagged): V_t from n-1 -> n, V_x centered at nodes
                    if Cr == 1.0:
                        VtA = (V[i - 1] - V_prev[i - 1]) / dt; VxA = Vx[i - 1]; VA = V[i - 1]
                        VtB = (V[i + 1] - V_prev[i + 1]) / dt; VxB = Vx[i + 1]; VB = V[i + 1]
                    else:
                        VA = (1.0 - Cr) * V[i] + Cr * V[i - 1]
                        VpA = (1.0 - Cr) * V_prev[i] + Cr * V_prev[i - 1]
                        VtA = (VA - VpA) / dt
                        VxA = (1.0 - Cr) * Vx[i] + Cr * Vx[i - 1]
                        VB = (1.0 - Cr) * V[i] + Cr * V[i + 1]
                        VpB = (1.0 - Cr) * V_prev[i] + Cr * V_prev[i + 1]
                        VtB = (VB - VpB) / dt
                        VxB = (1.0 - Cr) * Vx[i] + Cr * Vx[i + 1]
                    if brunone_conv == 0:
                        sA = 1.0 if VA * VxA >= 0.0 else -1.0
                        sB = 1.0 if VB * VxB >= 0.0 else -1.0
                        JA = kb * (VtA - aj * sA * VxA)
                        JB = kb * (VtB - aj * sB * VxB)
                    else:
                        sA = 1.0 if VA >= 0.0 else -1.0
                        sB = 1.0 if VB >= 0.0 else -1.0
                        JA = kb * (VtA + aj * sA * abs(VxA))
                        JB = kb * (VtB + aj * sB * abs(VxB))
                fA = _f_lookup(QA, Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
                fB = _f_lookup(QB, Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
                RA = fA * Rc
                RB = fB * Rc
                JuA = B * dt * A * JA
                JuB = B * dt * A * JB
                if fric_time_scheme == 1:
                    fP = _f_lookup(Qn[i], Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
                    RP = fP * Rc
                    CP = HA + B * QA - 0.5 * RA * QA * abs(QA) - JuA
                    CM = HB - B * QB + 0.5 * RB * QB * abs(QB) + JuB
                    QP = flow_from_characteristic(CP - CM, 2.0 * B, 2.0 * RP)   # 2B Q + RP Q|Q| = CP - CM
                    if fric_mode != 0:
                        RP = _f_lookup(QP, Re_coef, f_logRe_min, f_dlogRe, f_vals) * Rc
                        QP = flow_from_characteristic(CP - CM, 2.0 * B, 2.0 * RP)
                    HP = CP - B * QP - 0.5 * RP * QP * abs(QP)
                    dHp = 0.5 * RA * QA * abs(QA) + 0.5 * RP * QP * abs(QP)
                    dHm = 0.5 * RB * QB * abs(QB) + 0.5 * RP * QP * abs(QP)
                else:
                    CP = HA + B * QA - RA * QA * abs(QA) - JuA
                    CM = HB - B * QB + RB * QB * abs(QB) + JuB
                    QP = (CP - CM) / (2.0 * B)
                    HP = 0.5 * (CP + CM)
                    dHp = RA * QA * abs(QA)
                    dHm = RB * QB * abs(QB)
                H[i] = HP
                Q[i] = QP
                if do_energy:
                    # each characteristic carries half of its cell's dissipation (exact pairing identity)
                    d_wall_s += 0.5 * rho_g * (dHp * 0.5 * (QA + QP) + dHm * 0.5 * (QB + QP))
                    d_wall_u += 0.5 * rho_g * (JuA * 0.5 * (QA + QP) + JuB * 0.5 * (QB + QP))

        # ---------------- wellhead (segment 0, node 0): Q prescribed ----------------
        j = 0
        o = seg_off[0]
        Cr = seg_Cr[0]; B = seg_B[0]; Rc = seg_Rcoef[0]; aj = seg_a[0]; kb = seg_kb[0]
        i = o
        if Cr == 1.0:
            HB = Hn[i + 1]; QB = Qn[i + 1]; JB = Ju[i + 1]
        else:
            HB = (1.0 - Cr) * Hn[i] + Cr * Hn[i + 1]
            QB = (1.0 - Cr) * Qn[i] + Cr * Qn[i + 1]
            JB = (1.0 - Cr) * Ju[i] + Cr * Ju[i + 1]
        if fric_mode == 4:
            VB = (1.0 - Cr) * V[i] + Cr * V[i + 1]
            VpB = (1.0 - Cr) * V_prev[i] + Cr * V_prev[i + 1]
            VtB = (VB - VpB) / dt
            VxB = (1.0 - Cr) * Vx[i] + Cr * Vx[i + 1]
            if brunone_conv == 0:
                sB = 1.0 if VB * VxB >= 0.0 else -1.0
                JB = kb * (VtB - aj * sB * VxB)
            else:
                sB = 1.0 if VB >= 0.0 else -1.0
                JB = kb * (VtB + aj * sB * abs(VxB))
        fB = _f_lookup(QB, Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
        RB = fB * Rc
        JuB = B * dt * A * JB
        QP = _wellhead_Q(t, Q0, t_s, t_c, ramp)
        if fric_time_scheme == 1:
            fP = _f_lookup(QP, Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
            RP = fP * Rc
            CM = HB - B * QB + 0.5 * RB * QB * abs(QB) + JuB
            HP = CM + B * QP + 0.5 * RP * QP * abs(QP)
            dHm = 0.5 * RB * QB * abs(QB) + 0.5 * RP * QP * abs(QP)
        else:
            CM = HB - B * QB + RB * QB * abs(QB) + JuB
            HP = CM + B * QP
            dHm = RB * QB * abs(QB)
        H[i] = HP
        Q[i] = QP
        if do_energy:
            d_wall_s += 0.5 * rho_g * dHm * 0.5 * (QB + QP)
            d_wall_u += 0.5 * rho_g * JuB * 0.5 * (QB + QP)

        # ---------------- toe (last node): dead end Q = 0 ----------------
        j = nseg - 1
        o = seg_off[j]; N = seg_N[j]
        Cr = seg_Cr[j]; B = seg_B[j]; Rc = seg_Rcoef[j]; aj = seg_a[j]; kb = seg_kb[j]
        i = o + N
        if Cr == 1.0:
            HA = Hn[i - 1]; QA = Qn[i - 1]; JA = Ju[i - 1]
        else:
            HA = (1.0 - Cr) * Hn[i] + Cr * Hn[i - 1]
            QA = (1.0 - Cr) * Qn[i] + Cr * Qn[i - 1]
            JA = (1.0 - Cr) * Ju[i] + Cr * Ju[i - 1]
        if fric_mode == 4:
            VA = (1.0 - Cr) * V[i] + Cr * V[i - 1]
            VpA = (1.0 - Cr) * V_prev[i] + Cr * V_prev[i - 1]
            VtA = (VA - VpA) / dt
            VxA = (1.0 - Cr) * Vx[i] + Cr * Vx[i - 1]
            if brunone_conv == 0:
                sA = 1.0 if VA * VxA >= 0.0 else -1.0
                JA = kb * (VtA - aj * sA * VxA)
            else:
                sA = 1.0 if VA >= 0.0 else -1.0
                JA = kb * (VtA + aj * sA * abs(VxA))
        fA = _f_lookup(QA, Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
        RA = fA * Rc
        JuA = B * dt * A * JA
        if fric_time_scheme == 1:
            fP = _f_lookup(Qn[i], Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
            RP = fP * Rc
            CP = HA + B * QA - 0.5 * RA * QA * abs(QA) - JuA
            if toe_bc == 0:
                QP = 0.0
                HP = CP
            else:
                HP = H_toe
                QP = flow_from_characteristic(CP - H_toe, B, RP)
                if fric_mode != 0:
                    RP = _f_lookup(QP, Re_coef, f_logRe_min, f_dlogRe, f_vals) * Rc
                    QP = flow_from_characteristic(CP - H_toe, B, RP)
            dHp = 0.5 * RA * QA * abs(QA) + 0.5 * RP * QP * abs(QP)
        else:
            CP = HA + B * QA - RA * QA * abs(QA) - JuA
            if toe_bc == 0:
                QP = 0.0
                HP = CP
            else:
                HP = H_toe
                QP = (CP - H_toe) / B
            dHp = RA * QA * abs(QA)
        H[i] = HP
        Q[i] = QP
        if do_energy:
            d_wall_s += 0.5 * rho_g * dHp * 0.5 * (QA + QP)
            d_wall_u += 0.5 * rho_g * JuA * 0.5 * (QA + QP)

        # ---------------- cluster nodes ----------------
        it_step = 0
        res_step = 0.0
        for jc in range(ncl):
            iL = seg_off[jc] + seg_N[jc]
            iR = seg_off[jc + 1]
            # left foot (segment jc)
            CrL = seg_Cr[jc]; BL = seg_B[jc]; RcL = seg_Rcoef[jc]; aL = seg_a[jc]; kbL = seg_kb[jc]
            if CrL == 1.0:
                HA = Hn[iL - 1]; QA = Qn[iL - 1]; JA = Ju[iL - 1]
            else:
                HA = (1.0 - CrL) * Hn[iL] + CrL * Hn[iL - 1]
                QA = (1.0 - CrL) * Qn[iL] + CrL * Qn[iL - 1]
                JA = (1.0 - CrL) * Ju[iL] + CrL * Ju[iL - 1]
            if fric_mode == 4:
                VA = (1.0 - CrL) * V[iL] + CrL * V[iL - 1]
                VpA = (1.0 - CrL) * V_prev[iL] + CrL * V_prev[iL - 1]
                VtA = (VA - VpA) / dt
                VxA = (1.0 - CrL) * Vx[iL] + CrL * Vx[iL - 1]
                if brunone_conv == 0:
                    sA = 1.0 if VA * VxA >= 0.0 else -1.0
                    JA = kbL * (VtA - aL * sA * VxA)
                else:
                    sA = 1.0 if VA >= 0.0 else -1.0
                    JA = kbL * (VtA + aL * sA * abs(VxA))
            fA = _f_lookup(QA, Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
            RA = fA * RcL
            JuA = BL * dt * A * JA
            # right foot (segment jc+1)
            CrR = seg_Cr[jc + 1]; BR = seg_B[jc + 1]; RcR = seg_Rcoef[jc + 1]; aR = seg_a[jc + 1]; kbR = seg_kb[jc + 1]
            if CrR == 1.0:
                HB = Hn[iR + 1]; QB = Qn[iR + 1]; JB = Ju[iR + 1]
            else:
                HB = (1.0 - CrR) * Hn[iR] + CrR * Hn[iR + 1]
                QB = (1.0 - CrR) * Qn[iR] + CrR * Qn[iR + 1]
                JB = (1.0 - CrR) * Ju[iR] + CrR * Ju[iR + 1]
            if fric_mode == 4:
                VB = (1.0 - CrR) * V[iR] + CrR * V[iR + 1]
                VpB = (1.0 - CrR) * V_prev[iR] + CrR * V_prev[iR + 1]
                VtB = (VB - VpB) / dt
                VxB = (1.0 - CrR) * Vx[iR] + CrR * Vx[iR + 1]
                if brunone_conv == 0:
                    sB = 1.0 if VB * VxB >= 0.0 else -1.0
                    JB = kbR * (VtB - aR * sB * VxB)
                else:
                    sB = 1.0 if VB >= 0.0 else -1.0
                    JB = kbR * (VtB + aR * sB * abs(VxB))
            fB = _f_lookup(QB, Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0
            RB = fB * RcR
            JuB = BR * dt * A * JB
            if fric_time_scheme == 1:
                RPL = (_f_lookup(Qn[iL], Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0) * RcL
                RPR = (_f_lookup(Qn[iR], Re_coef, f_logRe_min, f_dlogRe, f_vals) if fric_mode != 0 else 0.0) * RcR
                CP = HA + BL * QA - 0.5 * RA * QA * abs(QA) - JuA
                CM = HB - BR * QB + 0.5 * RB * QB * abs(QB) + JuB
            else:
                RPL = 0.0
                RPR = 0.0
                CP = HA + BL * QA - RA * QA * abs(QA) - JuA
                CM = HB - BR * QB + RB * QB * abs(QB) + JuB
            # branch coefficients
            m0 = perf_off[jc]
            m1 = perf_off[jc + 1]
            nperf = m1 - m0
            for m in range(m0, m1):
                cc1, cc0, aa1, aa0, gflag = branch_coefficients(dt, pIf[m], pRf[m], pCf[m], pGl[m], ppres[m],
                                                                q[m], pc[m], Fb[m], inertia_scheme, storage_scheme, stiff_guard)
                c1[m] = cc1; c0[m] = cc0; a1[m] = aa1; a0[m] = aa0
                n_guard += gflag
                q_work[m] = q[m]
            Hj, Qm, Qp, it, ok, r_inf = solve_cluster_node(
                CP, BL, RPL, CM, BR, RPR, cl_z[jc], rho_g, nperf, pK[m0:m1], pkappa[m0:m1], eps_q,
                c1[m0:m1], c0[m0:m1], a1[m0:m1], a0[m0:m1], q_work[m0:m1], dq_work[m0:m1],
                Hn[iL], H_scale, Q_scale, p_scale, tol_dimless, max_iter, max_inner)
            it_step += it
            if r_inf > res_step:
                res_step = r_inf
            if not ok:
                if n_fail < n_fail_max:
                    fail_log[n_fail, 0] = n
                    fail_log[n_fail, 1] = jc
                n_fail += 1
            H[iL] = Hj
            H[iR] = Hj
            Q[iL] = Qm
            Q[iR] = Qp
            for m in range(m0, m1):
                q[m] = q_work[m]
                pc[m] = a0[m] + a1[m] * q[m]
                pin[m] = pc[m] + c1[m] * q[m] + c0[m]
                Fb[m] = pin[m] - pc[m]
            if do_energy:
                if fric_time_scheme == 1:
                    dHp = 0.5 * RA * QA * abs(QA) + 0.5 * RPL * Qm * abs(Qm)
                    dHm = 0.5 * RB * QB * abs(QB) + 0.5 * RPR * Qp * abs(Qp)
                else:
                    dHp = RA * QA * abs(QA)
                    dHm = RB * QB * abs(QB)
                d_wall_s += 0.5 * rho_g * (dHp * 0.5 * (QA + Qm) + dHm * 0.5 * (QB + Qp))
                d_wall_u += 0.5 * rho_g * (JuA * 0.5 * (QA + Qm) + JuB * 0.5 * (QB + Qp))
        newton_iters[n] = it_step
        newton_resid[n] = res_step
        if it_step > max_newton_iter:
            max_newton_iter = it_step

        # ---------------- velocities & memory states ----------------
        V_prev[:] = V
        for i in range(n_nodes):
            V[i] = Q[i] / A
        if fric_mode == 2:
            for j in range(nseg):
                o = seg_off[j]; N = seg_N[j]; w = kern_w[j]
                for i in range(o, o + N + 1):
                    dV = V[i] - V_prev[i]
                    s = 0.0
                    for l in range(M):
                        z[l, i] = kern_E[j, l] * z[l, i] + kern_Phi[j, l] * dV
                        s += z[l, i]
                    Ju[i] = w * s
        elif fric_mode == 3:
            for i in range(n_nodes):
                hist[n - 1, i] = V[i] - V_prev[i]
            for j in range(nseg):
                o = seg_off[j]; N = seg_N[j]; w = kern_w[j]
                for i in range(o, o + N + 1):
                    s = 0.0
                    for k in range(n):
                        s += Wbar[j, k] * hist[n - 1 - k, i]
                    Ju[i] = w * s

        # ---------------- records ----------------
        p_head[n] = rho_g * H[0]
        Q_head[n] = Q[0]
        if n % node_decim == 0:
            _record_nodes(n // node_decim, H, Q, q, pc, z, seg_off, seg_N, perf_off, M,
                          node_H, node_Qm, node_Qp, node_sq, node_pc, node_z)
        if snap_every > 0 and n % snap_every == 0:
            r = n // snap_every
            snap_H[r, :] = H
            snap_Q[r, :] = Q
            snap_step[r] = n
        # global volume balance: dVst/dt - (Q_wh - Q_toe - sum q)
        Vst = _stored_volume(H, seg_off, seg_N, seg_dx, seg_a, g, A)
        sq_tot = 0.0
        for m in range(nperf_tot):
            sq_tot += q[m]
        mass_resid[n] = ((Vst - Vst_n) / dt - 0.5 * ((Q[0] - Q[n_nodes - 1] - sq_tot)
                                                     + (Qn[0] - Qn[n_nodes - 1] - sq_tot_n))) / Q_scale
        Vst_n = Vst
        sq_tot_n = sq_tot
        if do_energy:
            e_wb, e_br = _energy_state(H, Q, q, pc, seg_off, seg_N, seg_dx, seg_a, A, rho, g, H_ref, pIf, pCf, ppres)
            E_wb[n] = e_wb
            E_br[n] = e_br
            if fric_mode == 2:
                E_mem[n] = _memory_storage(V, z, seg_off, seg_N, seg_dx, kern_w, kern_alpha, kern_beta, rho, A)
            d_perf, d_Rf, d_leak, p_src = _branch_power(H, q, pc, seg_off, seg_N, cl_z, perf_off, pK, pkappa, pRf, pGl,
                                                        ppres, rho_g, H_ref, eps_q)
            P_wh = rho_g * (H[0] - H_ref) * Q[0] - rho_g * (H[n_nodes - 1] - H_ref) * Q[n_nodes - 1]
            W_port[n] = W_port[n - 1] + 0.5 * dt * (P_wh + P_wh_n)
            W_wall_s[n] = W_wall_s[n - 1] + d_wall_s * dt
            W_wall_u[n] = W_wall_u[n - 1] + d_wall_u * dt
            W_perf[n] = W_perf[n - 1] + 0.5 * dt * (d_perf + d_perf_n)
            W_Rf[n] = W_Rf[n - 1] + 0.5 * dt * (d_Rf + d_Rf_n)
            W_leak[n] = W_leak[n - 1] + 0.5 * dt * (d_leak + d_leak_n)
            W_src[n] = W_src[n - 1] + 0.5 * dt * (p_src + p_src_n)
            P_wh_n = P_wh
            d_perf_n = d_perf; d_Rf_n = d_Rf; d_leak_n = d_leak; p_src_n = p_src

    return (p_head, Q_head, node_H, node_Qm, node_Qp, node_sq, node_pc, node_z,
            snap_H, snap_Q, snap_step,
            E_wb, E_br, E_mem, W_port, W_wall_s, W_wall_u, W_perf, W_Rf, W_leak, W_src,
            mass_resid, newton_iters, newton_resid, fail_log, n_fail, n_guard, max_newton_iter,
            H, Q, z, q, pc, pin)


# ============================================================================
# Python-level specification and driver
# ============================================================================


@dataclass
class ClusterSpec:
    x: float                      # measured depth along wellbore [m]
    K: np.ndarray                 # per perf orifice coefficient rho/(2 Cd^2 A^2) [Pa s^2 m^-6]
    kappa: np.ndarray             # per perf tortuosity coefficient [Pa s^0.5 m^-1.5]
    I_f: np.ndarray               # [Pa s^2 m^-3]
    R_f: np.ndarray               # [Pa s m^-3]
    C_f: np.ndarray               # [m^3 Pa^-1]
    G_l: np.ndarray               # [m^3 Pa^-1 s^-1]
    p_res: float                  # [Pa]

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.K = np.asarray(self.K, dtype=float)
        self.kappa = np.asarray(self.kappa, dtype=float)
        self.I_f = np.asarray(self.I_f, dtype=float)
        self.R_f = np.asarray(self.R_f, dtype=float)
        self.C_f = np.asarray(self.C_f, dtype=float)
        self.G_l = np.asarray(self.G_l, dtype=float)
        self.p_res = float(self.p_res)

    @property
    def nperf(self) -> int:
        return int(self.K.size)

    @staticmethod
    def from_perf_params(x: float, Cd: np.ndarray, A_perf: np.ndarray, kappa: np.ndarray, I_f: np.ndarray,
                         R_f: np.ndarray, C_f: np.ndarray, G_l: np.ndarray, p_res: float, rho: float) -> "ClusterSpec":
        Cd_a = np.asarray(Cd, dtype=float)
        A_a = np.asarray(A_perf, dtype=float)
        den = 2.0 * Cd_a ** 2 * A_a ** 2
        # deleted-cluster OOD sets Cd*A = 0 → finite but impassable orifice (K huge)
        K = np.divide(rho, den, out=np.full(den.shape, 1.0e30), where=den > 0.0)
        return ClusterSpec(x=x, K=K, kappa=np.asarray(kappa, float), I_f=np.asarray(I_f, float),
                           R_f=np.asarray(R_f, float), C_f=np.asarray(C_f, float), G_l=np.asarray(G_l, float),
                           p_res=float(p_res))


@dataclass
class WellSpec:
    L: float
    D: float
    a: float
    rho: float
    nu: float
    roughness: float
    TVD: float
    clusters: List[ClusterSpec]
    Q0: float
    t_s: float
    t_c: float
    ramp: str = "cosine"
    g: float = G_STD
    toe_bc: str = "dead_end"      # dead_end | reservoir (reservoir only without clusters: analytic goldens)
    H_toe: float = 0.0            # constant head at the toe when toe_bc == 'reservoir'

    @property
    def A(self) -> float:
        return math.pi * self.D ** 2 / 4.0

    def z_of_x(self, x: float) -> float:
        return -min(x, self.TVD)

    @property
    def x_clusters(self) -> np.ndarray:
        return np.array([c.x for c in self.clusters], dtype=float)


@dataclass
class NumericsSpec:
    Nx_ref: int = 1024
    Cr_target: float = 1.0
    friction_model: str = "zvb_rec"        # none | darcy | zvb_rec | zvb_direct | brunone
    friction_time_scheme: str = "trapezoidal"   # explicit | trapezoidal
    brunone_convention: str = "vitkovsky2000"  # locked in friction_brunone.yaml; plan_v2 is the contrast arm
    kernel_M: int = 12
    kernel_tau_min: float = 1e-8
    kernel_tau_max: float = 1e2
    inertia_scheme: str = "exp_trapezoidal"
    storage_scheme: str = "crank_nicolson"
    stiff_guard: bool = True
    eps_q_rel: float = 1e-6
    tol_dimless: float = 1e-8
    max_iter: int = 100
    max_inner: int = 100
    n_periods: float = 20.0                # window length in units of 4L/a (after t_s)
    T_total: Optional[float] = None        # overrides n_periods if given
    node_decim: int = 1
    snap_every: int = 0
    do_energy: bool = True
    n_fail_max: int = 1000


@dataclass
class MOCResult:
    t: np.ndarray
    p_head: np.ndarray
    Q_head: np.ndarray
    node_t: np.ndarray
    node_H: np.ndarray
    node_Qm: np.ndarray
    node_Qp: np.ndarray
    node_sq: np.ndarray
    node_pc: np.ndarray
    node_z: np.ndarray
    snap_t: np.ndarray
    snap_H: np.ndarray
    snap_Q: np.ndarray
    energy: Dict[str, np.ndarray]
    mass_resid: np.ndarray
    newton_iters: np.ndarray
    newton_resid: np.ndarray
    fail_log: np.ndarray
    n_fail: int
    n_guard: int
    grid: Grid
    meta: Dict = field(default_factory=dict)
    final_state: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def H_head(self) -> np.ndarray:
        return self.p_head / self.meta["rho_g"]


# ---- kernel cache -----------------------------------------------------------
_KERNEL_CACHE: Dict = {}


def get_kernel_fits(M: int, tau_min: float, tau_max: float):
    """Frozen kernel from configs/friction_zielke.yaml when available (same M and domain), else fit."""
    key = (M, tau_min, tau_max)
    if key not in _KERNEL_CACHE:
        fits = None
        try:
            from config_io import load_yaml
            cfg = load_yaml("friction_zielke.yaml")
            dom = cfg["production_path"]["fit"]["domain_tau"]
            if M in cfg["fits_by_M"] and abs(dom[0] - tau_min) < 1e-30 and abs(dom[1] - tau_max) < 1e-12:
                fits = mk.kernel_fits_from_config(cfg, M)
        except Exception:
            fits = None
        if fits is None:
            fz = mk.fit_zielke_kernel(M=M, tau_min=tau_min, tau_max=tau_max)
            fp = mk.fit_powerlaw_kernel(M=M, tau_min=tau_min, tau_max=min(tau_max, 1.0))
            fits = (fz, fp)
        _KERNEL_CACHE[key] = fits
    return _KERNEL_CACHE[key]


def set_kernel_fits(M: int, tau_min: float, tau_max: float, fz: mk.KernelFit, fp: mk.KernelFit) -> None:
    _KERNEL_CACHE[(M, tau_min, tau_max)] = (fz, fp)


# ---- steady state -----------------------------------------------------------
def _steady_head_drop(Q: float, seg_len: float, ftab: FrictionTable, Re_coef: float, D: float, A: float, g: float) -> float:
    f = float(_f_lookup(Q, Re_coef, ftab.logRe_min, ftab.dlogRe, ftab.values))
    return f * seg_len * Q * abs(Q) / (2.0 * g * D * A * A)


def solve_steady_state(well: WellSpec, grid: Grid, ftab: FrictionTable, eps_q: float,
                       tol_p: float = 1e-6, tol_q: float = 1e-14, max_iter: int = 200) -> Dict:
    """Exact discrete steady state for pumping at Q0 with a dead-end toe.

    Unknown: head at cluster 1; everything downstream follows recursively; the
    residual is the flow left below the last cluster (must vanish) — monotone in
    H_1, solved by safeguarded bisection/secant.
    """
    A = well.A
    g = well.g
    rho_g = well.rho * g
    Re_coef = well.D / (well.nu * A)
    ncl = len(well.clusters)
    seg_len = grid.seg_len

    def sweep(H1: float):
        Hc = np.empty(ncl)
        qs = []
        Qseg = np.empty(ncl + 1)
        Qseg[0] = well.Q0
        Hc[0] = H1
        Hj = H1
        for jc, cl in enumerate(well.clusters):
            if jc > 0:
                Hj = Hj - _steady_head_drop(Qseg[jc], seg_len[jc], ftab, Re_coef, well.D, A, g)
                Hc[jc] = Hj
            p_w = rho_g * (Hj - well.z_of_x(cl.x))
            qv = np.empty(cl.nperf)
            for m in range(cl.nperf):
                qv[m], _, ok = steady_perf_flow(p_w, cl.K[m], cl.kappa[m], eps_q, cl.R_f[m], cl.G_l[m], cl.p_res,
                                                tol_p, tol_q, max_iter)
            qs.append(qv)
            Qseg[jc + 1] = Qseg[jc] - qv.sum()
        return Qseg, Hc, qs

    if ncl == 0:
        if well.toe_bc == "reservoir":
            H_wh = well.H_toe + _steady_head_drop(well.Q0, seg_len[0], ftab, Re_coef, well.D, A, g)
            return {"H_wh": H_wh, "Qseg": np.array([well.Q0]), "Hc": np.array([]), "qs": [], "converged": True}
        if abs(well.Q0) > 0.0:
            raise ValueError("dead-end toe without clusters admits no steady flow: set Q0 = 0 or toe_bc='reservoir'")
        return {"H_wh": well.H_toe, "Qseg": np.array([0.0]), "Hc": np.array([]), "qs": [], "converged": True}
    if well.toe_bc != "dead_end":
        raise NotImplementedError("reservoir toe is only supported for single-pipe goldens (no clusters)")

    # bracket H1: residual r(H1) = Qseg[-1] decreasing in H1
    H_lo = well.clusters[0].p_res / rho_g + well.z_of_x(well.clusters[0].x)   # p_w = p_res -> q = 0 -> r = Q0 > 0
    r_lo = sweep(H_lo)[0][-1]
    step = 100.0
    H_hi = H_lo + step
    r_hi = sweep(H_hi)[0][-1]
    k = 0
    while r_hi > 0.0 and k < 200:
        H_lo, r_lo = H_hi, r_hi
        step *= 2.0
        H_hi = H_hi + step
        r_hi = sweep(H_hi)[0][-1]
        k += 1
    converged = False
    for _ in range(300):
        # Illinois/regula-falsi with bisection safeguard
        if r_lo - r_hi != 0.0:
            H_mid = H_hi - r_hi * (H_hi - H_lo) / (r_hi - r_lo)
            if not (H_lo < H_mid < H_hi):
                H_mid = 0.5 * (H_lo + H_hi)
        else:
            H_mid = 0.5 * (H_lo + H_hi)
        r_mid = sweep(H_mid)[0][-1]
        if abs(r_mid) < 1e-13 * max(well.Q0, 1e-12) or (H_hi - H_lo) < 1e-12 * max(abs(H_hi), 1.0):
            converged = True
            H1 = H_mid
            break
        if r_mid > 0.0:
            H_lo, r_lo = H_mid, r_mid
        else:
            H_hi, r_hi = H_mid, r_mid
        H1 = H_mid
    Qseg, Hc, qs = sweep(H1)
    H_wh = H1 + _steady_head_drop(well.Q0, seg_len[0], ftab, Re_coef, well.D, A, g)
    return {"H_wh": H_wh, "Qseg": Qseg, "Hc": Hc, "qs": qs, "converged": converged, "H1": H1}


# ---- main driver -------------------------------------------------------------
def simulate(well: WellSpec, num: NumericsSpec, kernel_fits=None, verbose: bool = False) -> MOCResult:
    t_start = time.perf_counter()
    A = well.A
    g = well.g
    rho_g = well.rho * g
    grid = build_grid(well.L, well.a, well.x_clusters, num.Nx_ref, num.Cr_target)
    dt = grid.dt
    period = 4.0 * well.L / well.a
    T_total = num.T_total if num.T_total is not None else well.t_s + num.n_periods * period
    n_steps = int(math.ceil(T_total / dt))
    nseg = grid.nseg
    ncl = grid.n_clusters

    fric_mode = FRICTION_MODES[num.friction_model]
    ftab = FrictionTable.build(well.roughness / well.D)
    if fric_mode == 0:
        ftab = FrictionTable(logRe_min=ftab.logRe_min, dlogRe=ftab.dlogRe, values=np.zeros_like(ftab.values))
    Re_coef = well.D / (well.nu * A)

    seg_a = np.full(nseg, well.a)
    seg_B = seg_a / (g * A)
    seg_Rcoef = seg_a * dt / (2.0 * g * well.D * A * A)

    # ---- steady state (a Gaussian pulse excitation starts from rest: Q_steady = 0)
    eps_q = num.eps_q_rel * max(abs(well.Q0), 1e-6)
    if well.ramp == "gauss_pulse":
        from dataclasses import replace as _replace
        ss = solve_steady_state(_replace(well, Q0=0.0), grid, ftab, eps_q)
    else:
        ss = solve_steady_state(well, grid, ftab, eps_q)
    H0 = np.empty(grid.n_nodes)
    Q0arr = np.empty(grid.n_nodes)
    bounds = np.concatenate([[0.0], well.x_clusters, [well.L]])
    Hstart = ss["H_wh"]
    for j in range(nseg):
        o = grid.seg_off[j]
        N = grid.seg_N[j]
        Qj = ss["Qseg"][j]
        Q0arr[o:o + N + 1] = Qj
        drop_cell = _steady_head_drop(Qj, grid.seg_dx[j], ftab, Re_coef, well.D, A, g)
        H0[o:o + N + 1] = Hstart - drop_cell * np.arange(N + 1)
        Hstart = H0[o + N]
    if well.toe_bc == "dead_end":
        Q0arr[-1] = 0.0          # toe: Q = 0 exactly

    # ---- perforation arrays
    cl_z = np.array([well.z_of_x(c.x) for c in well.clusters], dtype=float)
    cl_nperf = np.array([c.nperf for c in well.clusters], dtype=np.int64)
    perf_off = np.zeros(ncl + 1, dtype=np.int64)
    perf_off[1:] = np.cumsum(cl_nperf)
    ntot = int(perf_off[-1])

    def cat(attr):
        return np.concatenate([getattr(c, attr) for c in well.clusters]) if ncl else np.zeros(0)

    pK, pkappa, pIf, pRf, pCf, pGl = (cat(a) for a in ("K", "kappa", "I_f", "R_f", "C_f", "G_l"))
    ppres = np.concatenate([np.full(c.nperf, c.p_res) for c in well.clusters]) if ncl else np.zeros(0)
    q0 = np.concatenate(ss["qs"]) if ncl else np.zeros(0)
    pc0 = ppres + q0 / pGl if ncl else np.zeros(0)
    pin0 = pc0 + pRf * q0 if ncl else np.zeros(0)

    # ---- kernels (per segment by initial Re)
    M = num.kernel_M
    kern_E = np.ones((nseg, M))
    kern_Phi = np.zeros((nseg, M))
    kern_alpha = np.ones((nseg, M))
    kern_beta = np.ones((nseg, M))
    kern_w = np.zeros(nseg)
    Wbar = np.zeros((nseg, 1))
    seg_Re = np.abs(ss["Qseg"]) * Re_coef
    seg_kernel = []
    if fric_mode in (2, 3):
        if kernel_fits is None:
            kernel_fits = get_kernel_fits(M, num.kernel_tau_min, num.kernel_tau_max)
        fz, fp = kernel_fits
        if fric_mode == 3:
            Wbar = np.zeros((nseg, n_steps + 1))
        for j in range(nseg):
            Re_j = float(seg_Re[j])
            m_, n_ = mk.kernel_for_reynolds(Re_j, fz, fp)
            rk = mk.RecursiveKernel.build(m_, n_, well.nu, well.D, dt)
            kern_E[j] = rk.E
            kern_Phi[j] = rk.Phi
            kern_alpha[j] = rk.alpha
            kern_beta[j] = rk.beta
            kern_w[j] = rk.w
            seg_kernel.append("zielke" if Re_j < mk.RE_LAMINAR_TURBULENT else "vardy_brown")
            if fric_mode == 3:
                _, Wint = mk.exact_W_for_reynolds(Re_j)
                Wbar[j, :] = mk.direct_step_weights(Wint, well.nu, well.D, dt, n_steps + 1)
    seg_kb = np.zeros(nseg)
    if fric_mode == 4:
        for j in range(nseg):
            seg_kb[j] = brunone_k(float(seg_Re[j]))

    z0 = np.zeros((M, grid.n_nodes))

    # ---- scales
    Q_scale = max(abs(well.Q0), A * 0.1)
    H_scale = float(seg_B[0]) * Q_scale
    p_scale = rho_g * H_scale
    H_ref = float(np.mean([c.p_res / rho_g + well.z_of_x(c.x) for c in well.clusters])) if ncl else float(well.H_toe)

    out = run_moc_core(
        grid.seg_off, grid.seg_N, grid.seg_dx, grid.seg_Cr, seg_B, seg_a, seg_Rcoef,
        A, well.rho, g, Re_coef, ftab.logRe_min, ftab.dlogRe, ftab.values,
        fric_mode, 1 if num.friction_time_scheme == "trapezoidal" else 0, BRUNONE_CONVENTIONS[num.brunone_convention],
        kern_E, kern_Phi, kern_w, kern_alpha, kern_beta, Wbar, seg_kb,
        cl_z, cl_nperf, perf_off, pK, pkappa, pIf, pRf, pCf, pGl, ppres,
        INERTIA_SCHEMES[num.inertia_scheme], STORAGE_SCHEMES[num.storage_scheme], 1 if num.stiff_guard else 0, eps_q,
        well.Q0, well.t_s, well.t_c, RAMP_TYPES[well.ramp], 1 if well.toe_bc == "reservoir" else 0, well.H_toe, dt, n_steps,
        H0, Q0arr, z0, q0, pc0, pin0,
        H_scale, Q_scale, p_scale, num.tol_dimless, num.max_iter, num.max_inner,
        1 if num.do_energy else 0, H_ref,
        num.node_decim, num.snap_every, num.n_fail_max,
    )
    (p_head, Q_head, node_H, node_Qm, node_Qp, node_sq, node_pc, node_z,
     snap_H, snap_Q, snap_step,
     E_wb, E_br, E_mem, W_port, W_wall_s, W_wall_u, W_perf, W_Rf, W_leak, W_src,
     mass_resid, newton_iters, newton_resid, fail_log, n_fail, n_guard, max_newton_iter,
     Hf, Qf, zf, qf, pcf, pinf) = out
    wall = time.perf_counter() - t_start
    t = dt * np.arange(n_steps + 1)
    energy = {
        "E_wb": E_wb, "E_br": E_br, "E_mem": E_mem, "W_port": W_port, "W_wall_steady": W_wall_s,
        "W_wall_unsteady": W_wall_u, "W_perf": W_perf, "W_Rf": W_Rf, "W_leak": W_leak, "W_src": W_src,
    } if num.do_energy else {}
    meta = {
        "dt": dt, "n_steps": n_steps, "T_total": T_total, "period_4L_a": period,
        "rho_g": rho_g, "A": A, "H_ref": H_ref, "H_scale": H_scale, "Q_scale": Q_scale, "p_scale": p_scale,
        "steady": {"H_wh": ss["H_wh"], "p_wh": ss["H_wh"] * rho_g, "Qseg": ss["Qseg"].tolist(),
                   "converged": bool(ss["converged"])},
        "seg_Re": seg_Re.tolist(), "seg_kernel": seg_kernel, "seg_kb": seg_kb.tolist(),
        "friction_model": num.friction_model, "friction_time_scheme": num.friction_time_scheme,
        "kernel_M": M, "Nx_ref": num.Nx_ref, "Cr_target": num.Cr_target,
        "wall_clock_s": wall, "max_newton_iter_per_step": int(max_newton_iter),
        "newton_fail_count": int(n_fail), "stiff_guard_count": int(n_guard),
        "max_newton_resid_dimless": float(newton_resid.max()) if ncl else 0.0,
        "max_global_mass_resid_dimless": float(np.max(np.abs(mass_resid))),
        "grid": grid.summary(),
    }
    return MOCResult(t=t, p_head=p_head, Q_head=Q_head, node_t=t[::num.node_decim][: node_H.shape[0]],
                     node_H=node_H, node_Qm=node_Qm, node_Qp=node_Qp, node_sq=node_sq, node_pc=node_pc, node_z=node_z,
                     snap_t=snap_step * dt, snap_H=snap_H, snap_Q=snap_Q, energy=energy, mass_resid=mass_resid,
                     newton_iters=newton_iters, newton_resid=newton_resid, fail_log=fail_log[:min(n_fail, num.n_fail_max)],
                     n_fail=int(n_fail), n_guard=int(n_guard), grid=grid, meta=meta,
                     final_state={"H": Hf, "Q": Qf, "z": zf, "q": qf, "p_c": pcf, "p_in": pinf})


def energy_balance(res: MOCResult) -> Dict[str, float]:
    """Integral energy audit r_E over the whole window and passivity checks (§2.6)."""
    if not res.energy:
        return {}
    e = res.energy
    E_tot = e["E_wb"] + e["E_br"]                 # mechanical energy (memory variables excluded, §2.6)
    E_mem = e["E_mem"]                            # KYP storage of the recursive kernel (zero unless zvb_rec)
    W_diss = e["W_wall_steady"] + e["W_wall_unsteady"] + e["W_perf"] + e["W_Rf"] + e["W_leak"]
    resid = e["W_port"] + e["W_src"] - (E_tot - E_tot[0]) - W_diss
    E_ref = float(np.max(E_tot))
    t = res.t
    # passive window: after wellhead flow is zero
    shut = np.where(res.Q_head == 0.0)[0]
    out = {
        "E_ref": E_ref,
        "energy_imbalance_rel_final": float(abs(resid[-1]) / E_ref) if E_ref > 0 else 0.0,
        "energy_imbalance_rel_max": float(np.max(np.abs(resid)) / E_ref) if E_ref > 0 else 0.0,
        "W_unsteady_min_over_time": float(np.min(e["W_wall_unsteady"])),
        "W_unsteady_final": float(e["W_wall_unsteady"][-1]),
        "E_mem_initial": float(E_mem[0]),
        # positive-realness bound: int J V dt >= -E_mem(0)  (KYP storage with cross term)
        "W_unsteady_plus_Emem0_min": float(np.min(e["W_wall_unsteady"] + E_mem[0] - E_mem)),
    }
    if shut.size > 2:
        i0 = int(shut[0])
        E_s = E_tot[i0] if E_tot[i0] > 0 else E_ref
        dE = np.diff(E_tot[i0:])
        dEm = np.diff((E_tot + E_mem)[i0:])
        out["passive_window_start_s"] = float(t[i0])
        out["passive_max_dE_rel"] = float(np.max(dE) / E_s)
        out["passive_violations_1e-8"] = int(np.sum(dE / E_s > 1e-8))
        out["passive_max_dE_with_Emem_rel"] = float(np.max(dEm) / E_s)
        out["passive_violations_with_Emem_1e-8"] = int(np.sum(dEm / E_s > 1e-8))
        out["passive_energy_drop_total_rel"] = float((E_tot[i0] - E_tot[-1]) / E_s)
    return out
