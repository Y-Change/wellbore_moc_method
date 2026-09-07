# -*- coding: utf-8 -*-
"""Per-cluster damped Newton closure of the perforation/fracture node (§2.3).

Unknowns at a cluster node (time level n+1):  H, Q^-, Q^+, and per perforation m
q_m, p_in,m, p_c,m.  Equations (5 families, all SI):

    C+ (upstream segment):   H + B_L Q^- + (R_L/2) Q^-|Q^-| = C_P
    C- (downstream segment): H - B_R Q^+ - (R_R/2) Q^+|Q^+| = C_M
    mass:                    Q^- - Q^+ = sum_m q_m
    orifice (per perf):      p_w - p_in,m = K_p,m q_m|q_m| + kappa_m phi(q_m),   p_w = rho g (H - z_j)
    inertia/resistance:      p_in,m - p_c,m = I_f,m dq_m/dt + R_f,m q_m
    storage/leak-off:        C_f,m dp_c,m/dt = q_m - G_l,m (p_c,m - p_res)

phi(q) = q / sqrt(|q| + eps_q) is the regularised sgn(q)|q|^{1/2} tortuosity law
(exact for |q| >> eps_q; finite slope at q = 0).  The (R/2) Q|Q| terms are the
second-order (trapezoidal) friction contribution at the head of the characteristic;
they vanish for the explicit ("friction at the foot") scheme.

Time integration of the branch ODEs is expressed through the affine relations
    p_in - p_c = c1 q^{n+1} + c0        (inertia/resistance ODE)
    p_c        = a0 + a1 q^{n+1}        (storage ODE: Crank–Nicolson or exponential-trapezoidal)
whose coefficients are prepared by ``branch_coefficients`` for each step.

Solution strategy (robust "damped Newton" with bracketing):
  * for a given H every perforation equation Phi_m(q) = 0 is strictly monotone
    in q -> unique root, safeguarded Newton (Newton step + bisection fallback);
  * the mass equation g(H) = Q^-(H) - Q^+(H) - sum_m q_m(H) is strictly
    decreasing in H -> unique root, safeguarded Newton with analytic derivative.
Convergence is declared only if the *full* dimensionless residual vector of the
3 + 3M equations satisfies ||r||_inf < tol (default 1e-8); failures are counted
and reported (never silently accepted).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def phi1_scalar(h):
    if h < 1e-6:
        return 1.0 - 0.5 * h + h * h / 6.0
    return (1.0 - math.exp(-h)) / h


@njit(cache=True, fastmath=False)
def exp_trap_weights(h):
    """Return (E, wA, wB): exact step weights for  y' = -lam y + F/I with F linear,
    y^{n+1} = E y^n + (dt/I) [F^n wA + F^{n+1} wB],  h = lam dt.
    wA = phi1 - wB,  wB = phi1 - (phi1 - e^{-h})/h.
    """
    if h < 1e-3:
        wB = 0.5 - h / 6.0 + h * h / 24.0
        wA = 0.5 - h / 3.0 + h * h / 8.0
        E = 1.0 - h + 0.5 * h * h - h * h * h / 6.0
    else:
        E = math.exp(-h)
        p1 = (1.0 - E) / h
        wB = p1 - (p1 - E) / h
        wA = p1 - wB
    return E, wA, wB


@njit(cache=True, fastmath=False)
def tort_phi(q, eps):
    return q / math.sqrt(abs(q) + eps)


@njit(cache=True, fastmath=False)
def tort_dphi(q, eps):
    aq = abs(q)
    return (0.5 * aq + eps) / (aq + eps) ** 1.5


@njit(cache=True, fastmath=False)
def flow_from_characteristic(D, B, R):
    """Solve B Q + (R/2) Q|Q| = D for Q (monotone).  R = 0 -> Q = D/B."""
    if D == 0.0:
        return 0.0
    aD = abs(D)
    if R <= 0.0:
        Q = aD / B
    else:
        Q = 2.0 * aD / (B + math.sqrt(B * B + 2.0 * R * aD))
    return Q if D > 0.0 else -Q


# ----------------------------------------------------------------------------
# branch ODE coefficients
# ----------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def branch_coefficients(dt, I_f, R_f, C_f, G_l, p_res, q_n, p_c_n, F_n,
                        inertia_scheme, storage_scheme, stiff_guard):
    """Affine coefficients for one perforation branch at the new time level.

    inertia_scheme : 0 = backward Euler (theta=1), 1 = exponential-trapezoidal (exact for linear forcing)
    storage_scheme : 0 = Crank–Nicolson, 1 = exponential-trapezoidal
    stiff_guard    : if 1 and storage_scheme==0 and G_l dt / C_f > 1 (CN would oscillate),
                     switch this branch to the exponential integrator (flag returned).
    Returns (c1, c0, a1, a0, guard_triggered).
    F_n = (p_in - p_c)^n  (needed by the exponential-trapezoidal inertia scheme).
    """
    # ---- inertia / resistance:  p_in - p_c = c1 q + c0
    if inertia_scheme == 0 or I_f <= 0.0:
        c1 = R_f + I_f / dt
        c0 = -I_f * q_n / dt
    else:
        h = R_f * dt / I_f
        E, wA, wB = exp_trap_weights(h)
        # q^{n+1} = E q^n + (dt/I)[F^n wA + F^{n+1} wB]  ->  F^{n+1} = (I/(dt wB)) q^{n+1} - (I/(dt wB)) E q^n - (wA/wB) F^n
        c1 = I_f / (dt * wB)
        c0 = -c1 * E * q_n - (wA / wB) * F_n
    # ---- storage / leak-off:  p_c = a0 + a1 q
    guard = 0
    use_exp = storage_scheme == 1
    if storage_scheme == 0 and stiff_guard == 1:
        if C_f <= 0.0 or G_l * dt / C_f > 1.0:
            use_exp = True
            guard = 1
    if not use_exp:
        dn = C_f / dt + 0.5 * G_l
        a1 = 0.5 / dn
        a0 = ((C_f / dt - 0.5 * G_l) * p_c_n + 0.5 * q_n + G_l * p_res) / dn
    else:
        if C_f <= 0.0:
            # algebraic limit: p_c = p_res + q / G_l
            a1 = 1.0 / G_l
            a0 = p_res
        else:
            h = G_l * dt / C_f
            E, wA, wB = exp_trap_weights(h)
            a1 = dt / C_f * wB
            a0 = p_res + (p_c_n - p_res) * E + dt / C_f * q_n * wA
    return c1, c0, a1, a0, guard


# ----------------------------------------------------------------------------
# perforation scalar equation
# ----------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def perf_residual(q, p_w, K, kappa, eps_q, c1, c0, a1, a0):
    return p_w - K * q * abs(q) - kappa * tort_phi(q, eps_q) - (c1 * q + c0) - (a0 + a1 * q)


@njit(cache=True, fastmath=False)
def perf_dresidual(q, K, kappa, eps_q, c1, a1):
    return -(2.0 * K * abs(q) + kappa * tort_dphi(q, eps_q) + c1 + a1)


@njit(cache=True, fastmath=False)
def solve_perf_flow(p_w, K, kappa, eps_q, c1, c0, a1, a0, q_init, tol_p, tol_q, max_iter):
    """Safeguarded Newton for Phi(q) = 0 (strictly decreasing).  Returns (q, dq/dp_w, iters, ok)."""
    P = p_w - c0 - a0
    s_lin = c1 + a1
    q_lin = P / s_lin
    lo = min(0.0, q_lin)
    hi = max(0.0, q_lin)
    # Phi(lo) >= 0 >= Phi(hi)
    q = q_init
    if q < lo or q > hi:
        q = 0.5 * (lo + hi)
    ok = False
    it = 0
    for it in range(1, max_iter + 1):
        f = perf_residual(q, p_w, K, kappa, eps_q, c1, c0, a1, a0)
        if abs(f) < tol_p:
            ok = True
            break
        if f > 0.0:
            lo = q
        else:
            hi = q
        df = perf_dresidual(q, K, kappa, eps_q, c1, a1)
        q_new = q - f / df
        if q_new <= lo or q_new >= hi or not math.isfinite(q_new):
            q_new = 0.5 * (lo + hi)
        if abs(q_new - q) < tol_q:
            q = q_new
            f = perf_residual(q, p_w, K, kappa, eps_q, c1, c0, a1, a0)
            ok = abs(f) < tol_p * 1e3 or abs(hi - lo) < tol_q
            break
        q = q_new
    dq_dpw = -1.0 / perf_dresidual(q, K, kappa, eps_q, c1, a1)
    return q, dq_dpw, it, ok


# ----------------------------------------------------------------------------
# cluster node
# ----------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def cluster_mass_function(H, CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                          c1, c0, a1, a0, q_out, dq_out, tol_p, tol_q, max_inner):
    """g(H) = Q^- - Q^+ - sum q_m(H) and its derivative; fills q_out / dq_out (dq/dp_w)."""
    Qm = flow_from_characteristic(CP - H, BL, RL)
    Qp = flow_from_characteristic(H - CM, BR, RR)
    dQm = -1.0 / (BL + RL * abs(Qm))
    dQp = 1.0 / (BR + RR * abs(Qp))
    p_w = rho_g * (H - z_j)
    sq = 0.0
    dsq = 0.0
    inner_ok = True
    inner_it = 0
    for m in range(nperf):
        q, dq, it, ok = solve_perf_flow(p_w, K[m], kappa[m], eps_q, c1[m], c0[m], a1[m], a0[m],
                                        q_out[m], tol_p, tol_q, max_inner)
        q_out[m] = q
        dq_out[m] = dq
        sq += q
        dsq += dq
        inner_it += it
        if not ok:
            inner_ok = False
    g = Qm - Qp - sq
    dg = dQm - dQp - rho_g * dsq
    return g, dg, Qm, Qp, inner_ok, inner_it


@njit(cache=True, fastmath=False)
def solve_cluster_node(CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                       c1, c0, a1, a0, q_work, dq_work, H_init, H_scale, Q_scale, p_scale,
                       tol_dimless, max_iter, max_inner):
    """Solve the full node closure.  Returns (H, Q^-, Q^+, iters, ok, resid_inf).

    q_work (nperf,) must hold initial guesses on entry and receives q_m on exit.
    """
    tol_p = tol_dimless * p_scale * 1e-3
    tol_q = tol_dimless * Q_scale * 1e-3
    H = H_init
    g, dg, Qm, Qp, inner_ok, _ = cluster_mass_function(H, CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                                                       c1, c0, a1, a0, q_work, dq_work, tol_p, tol_q, max_inner)
    # ---- bracket the root (g decreasing in H)
    step = max(H_scale, 1e-6)
    if g > 0.0:
        lo = H
        hi = H + step
        g_hi, _, _, _, _, _ = cluster_mass_function(hi, CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                                                    c1, c0, a1, a0, q_work, dq_work, tol_p, tol_q, max_inner)
        k = 0
        while g_hi > 0.0 and k < 200:
            lo = hi
            step *= 2.0
            hi = hi + step
            g_hi, _, _, _, _, _ = cluster_mass_function(hi, CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                                                        c1, c0, a1, a0, q_work, dq_work, tol_p, tol_q, max_inner)
            k += 1
    else:
        hi = H
        lo = H - step
        g_lo, _, _, _, _, _ = cluster_mass_function(lo, CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                                                    c1, c0, a1, a0, q_work, dq_work, tol_p, tol_q, max_inner)
        k = 0
        while g_lo < 0.0 and k < 200:
            hi = lo
            step *= 2.0
            lo = lo - step
            g_lo, _, _, _, _, _ = cluster_mass_function(lo, CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                                                        c1, c0, a1, a0, q_work, dq_work, tol_p, tol_q, max_inner)
            k += 1
    # ---- safeguarded Newton on H
    ok = False
    it = 0
    for it in range(1, max_iter + 1):
        g, dg, Qm, Qp, inner_ok, _ = cluster_mass_function(H, CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                                                           c1, c0, a1, a0, q_work, dq_work, tol_p, tol_q, max_inner)
        if g > 0.0:
            lo = H
        else:
            hi = H
        if abs(g) < tol_dimless * Q_scale * 1e-2 and inner_ok:
            ok = True
            break
        H_new = H - g / dg
        if H_new <= lo or H_new >= hi or not math.isfinite(H_new):
            H_new = 0.5 * (lo + hi)      # bisection fallback (damping)
        if abs(H_new - H) < tol_dimless * H_scale * 1e-4:
            H = H_new
            g, dg, Qm, Qp, inner_ok, _ = cluster_mass_function(H, CP, BL, RL, CM, BR, RR, z_j, rho_g, nperf, K, kappa, eps_q,
                                                               c1, c0, a1, a0, q_work, dq_work, tol_p, tol_q, max_inner)
            ok = inner_ok
            break
        H = H_new
    # ---- full dimensionless residual of all 3 + 3M equations
    r_inf = abs(Qm - Qp - q_work[:nperf].sum()) / Q_scale
    rcp = abs(H + BL * Qm + 0.5 * RL * Qm * abs(Qm) - CP) / H_scale
    rcm = abs(H - BR * Qp - 0.5 * RR * Qp * abs(Qp) - CM) / H_scale
    if rcp > r_inf:
        r_inf = rcp
    if rcm > r_inf:
        r_inf = rcm
    p_w = rho_g * (H - z_j)
    for m in range(nperf):
        r_or = abs(perf_residual(q_work[m], p_w, K[m], kappa[m], eps_q, c1[m], c0[m], a1[m], a0[m])) / p_scale
        if r_or > r_inf:
            r_inf = r_or
    if r_inf > tol_dimless:
        ok = False
    return H, Qm, Qp, it, ok, r_inf


# ----------------------------------------------------------------------------
# steady-state helpers (used by the initial condition solver)
# ----------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def steady_perf_flow(p_w, K, kappa, eps_q, R_f, G_l, p_res, tol_p, tol_q, max_iter):
    """Steady branch: p_w - p_res = K q|q| + kappa phi(q) + R_f q + q/G_l."""
    q, dq, it, ok = solve_perf_flow(p_w, K, kappa, eps_q, R_f, 0.0, 1.0 / G_l, p_res, 0.0, tol_p, tol_q, max_iter)
    return q, dq, ok
