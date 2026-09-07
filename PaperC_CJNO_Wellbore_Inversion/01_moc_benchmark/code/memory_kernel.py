# -*- coding: utf-8 -*-
"""Frequency-dependent (memory) wall friction: Zielke / Vardy–Brown kernels.

Implements 攻关执行方案_v2 §2.4(a)/(b):

(a) Reference path — direct history convolution with the *piecewise complete*
    weighting function W(tau):
        Zielke (laminar):  small-tau sqrt-series (tau < 0.02) + large-tau
                           exponential series (tau >= 0.02);
        Vardy–Brown (turbulent, smooth pipe):
                           W = A* exp(-tau/C*) / sqrt(tau),  A* = 1/(2 sqrt(pi)),
                           C* = 12.86 / Re**kappa, kappa = log10(15.29 / Re**0.0567).
    Non-dimensional time tau = 4 nu t / D**2.

    J_{u,V}(t) = (16 nu / D**2) * int_0^t W(4 nu (t-s)/D**2) dV/ds ds   [m s^-2]
    J_{u,Q} = A * J_{u,V}                                               [m^3 s^-2]

(b) Production path — exponential-sum (recursive) kernel with M terms,
        W(tau) ~= sum_l m_l exp(-n_l tau),  m_l, n_l > 0  (positivity enforced)
    which in dimensional time gives
        dz_l/dt = -beta_l z_l + alpha_l dV/dt,   J_{u,V} = w * sum_l z_l
        beta_l = 4 nu n_l / D**2,  alpha_l = m_l,  w = 16 nu / D**2.
    With piecewise-linear V the recursion is exact:
        z_l^{n+1} = exp(-beta_l dt) z_l^n + alpha_l * phi1(beta_l dt) * (V^{n+1}-V^n),
        phi1(h) = (1 - exp(-h)) / h.

    Vardy–Brown scaling law: because W_VB = [A*/sqrt(tau)] * exp(-tau/C*),
    an exponential-sum fit {m_l, n_l} of the pure power law A*/sqrt(tau)
    yields the *exact* VB exponential sum {m_l, n_l + 1/C*(Re)} for any Re.
    Hence a single power-law fit covers the whole turbulent Re range with no
    binning error (the alternative Re-binning route of §2.4(b)).

Both paths expose *step-averaged* weights so that the direct convolution and
the recursion integrate the same piecewise-linear velocity history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf, logsumexp

# --------------------------------------------------------------------------
# Zielke (1968) piecewise complete weighting function
# --------------------------------------------------------------------------
ZIELKE_TAU_SPLIT = 0.02
# small-tau series: W = sum_i m_i tau^{(i-2)/2}, i = 1..6
ZIELKE_M_SMALL = np.array([0.282095, -1.25, 1.057855, 0.9375, 0.396696, -0.351563])
ZIELKE_POW_SMALL = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
# large-tau series: W = sum_i exp(-n_i tau); n_i = j_{2,i}^2 (squares of J_2 zeros)
ZIELKE_N_LARGE = np.array([26.3744, 70.8493, 135.0198, 218.9216, 322.5544])

A_STAR = 1.0 / (2.0 * math.sqrt(math.pi))  # 0.28209479...
RE_LAMINAR_TURBULENT = 2300.0


def zielke_W(tau: np.ndarray) -> np.ndarray:
    """Piecewise complete Zielke weighting function W(tau), tau > 0."""
    tau = np.asarray(tau, dtype=np.float64)
    out = np.empty_like(tau)
    small = tau < ZIELKE_TAU_SPLIT
    ts = tau[small]
    if ts.size:
        out[small] = sum(m * ts ** p for m, p in zip(ZIELKE_M_SMALL, ZIELKE_POW_SMALL))
    tl = tau[~small]
    if tl.size:
        out[~small] = np.exp(-np.outer(tl, ZIELKE_N_LARGE)).sum(axis=1)
    return out


def zielke_logW(tau: np.ndarray) -> np.ndarray:
    """log W(tau) evaluated stably (large tau via logsumexp)."""
    tau = np.asarray(tau, dtype=np.float64)
    out = np.empty_like(tau)
    small = tau < ZIELKE_TAU_SPLIT
    if small.any():
        out[small] = np.log(zielke_W(tau[small]))
    tl = tau[~small]
    if tl.size:
        out[~small] = logsumexp(-np.outer(tl, ZIELKE_N_LARGE), axis=1)
    return out


def zielke_W_integral(tau: np.ndarray) -> np.ndarray:
    """Antiderivative I(tau) = int_0^tau W(s) ds (piecewise analytic)."""
    tau = np.asarray(tau, dtype=np.float64)
    out = np.empty_like(tau)
    ts = ZIELKE_TAU_SPLIT

    def small_int(t):
        return sum(m * t ** (p + 1.0) / (p + 1.0) for m, p in zip(ZIELKE_M_SMALL, ZIELKE_POW_SMALL))

    def large_int_from_split(t):
        # int_split^t sum exp(-n s) ds
        return sum((math.exp(-n * ts) - np.exp(-n * t)) / n for n in ZIELKE_N_LARGE)

    small = tau < ts
    out[small] = small_int(tau[small])
    out[~small] = small_int(np.array(ts)) + large_int_from_split(tau[~small])
    return out


# --------------------------------------------------------------------------
# Vardy–Brown (2003/2004) turbulent smooth-pipe weighting function
# --------------------------------------------------------------------------
def vardy_brown_Cstar(Re: float) -> float:
    """Shear decay coefficient C*(Re) of Vardy & Brown for smooth pipes."""
    Re = float(Re)
    if Re < RE_LAMINAR_TURBULENT:
        raise ValueError("Vardy–Brown C* is only defined for turbulent Re (>= 2300)")
    kappa = math.log10(15.29 / Re ** 0.0567)
    return 12.86 / Re ** kappa


def vardy_brown_W(tau: np.ndarray, Re: float) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    C = vardy_brown_Cstar(Re)
    return A_STAR * np.exp(-tau / C) / np.sqrt(tau)


def vardy_brown_W_integral(tau: np.ndarray, Re: float) -> np.ndarray:
    """int_0^tau A* exp(-s/C)/sqrt(s) ds = A* sqrt(pi C) erf(sqrt(tau/C))."""
    tau = np.asarray(tau, dtype=np.float64)
    C = vardy_brown_Cstar(Re)
    return A_STAR * math.sqrt(math.pi * C) * erf(np.sqrt(tau / C))


def powerlaw_W(tau: np.ndarray) -> np.ndarray:
    """Re-independent small-time limit A*/sqrt(tau) (common to Zielke and VB)."""
    return A_STAR / np.sqrt(np.asarray(tau, dtype=np.float64))


# --------------------------------------------------------------------------
# Exponential-sum kernel
# --------------------------------------------------------------------------
def exp_sum_W(tau: np.ndarray, m: np.ndarray, n: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    return np.exp(-np.outer(tau, n)) @ m


def exp_sum_logW(tau: np.ndarray, m: np.ndarray, n: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    return logsumexp(-np.outer(tau, n) + np.log(m)[None, :], axis=1)


def exp_sum_W_integral(tau: np.ndarray, m: np.ndarray, n: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    return ((1.0 - np.exp(-np.outer(tau, n))) / n[None, :]) @ m


@dataclass
class KernelFit:
    """Result of an exponential-sum fit (dimensionless tau domain)."""
    name: str
    m: np.ndarray              # weights  (M,)  > 0
    n: np.ndarray              # decay rates (M,) > 0
    tau_min: float
    tau_max: float
    tau_eff_max: float         # last tau with non-negligible W actually used in the fit
    max_rel_err: float         # max |W_fit/W - 1| over the weighted domain
    rms_log_err: float
    n_points: int

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "M": int(self.m.size),
            "m": [float(v) for v in self.m],
            "n": [float(v) for v in self.n],
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "tau_eff_max": self.tau_eff_max,
            "max_rel_err": self.max_rel_err,
            "rms_log_err": self.rms_log_err,
            "n_points": self.n_points,
            "positivity_ok": bool(np.all(self.m > 0) and np.all(self.n > 0)),
        }


def assert_positive_kernel(m: np.ndarray, n: np.ndarray) -> None:
    """§2.4(b)/§2.6: positivity of w_l, alpha_l, beta_l guarantees Re[H(jw)] >= 0."""
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(n))):
        raise ValueError("kernel parameters must be finite")
    if np.any(m <= 0.0) or np.any(n <= 0.0):
        raise ValueError("positivity violated: all m_l (alpha_l w_l) and n_l (beta_l) must be > 0")


def fit_exponential_sum(
    logW_func: Callable[[np.ndarray], np.ndarray],
    M: int = 10,
    tau_min: float = 1e-8,
    tau_max: float = 1e2,
    n_points: int = 600,
    negligible_ratio: float = 1e-16,
    n_lo_init: float | None = None,
    n_hi_init: float | None = None,
    seed: int = 0,
    n_restarts: int = 2,
    minimax_iters: int = 6,
    name: str = "kernel",
) -> KernelFit:
    """Positivity-constrained log least-squares fit of W(tau) by sum_l m_l exp(-n_l tau).

    Positivity is enforced structurally through the log-parametrisation
    m_l = exp(u_l), n_l = exp(v_l).  The residual is log(W_fit) - log(W) on a
    log-spaced tau grid; points where W < negligible_ratio * W(tau_min) are
    dropped because they are below double-precision relevance (Zielke's
    exponential tail underflows long before tau = 1e2).
    """
    tau = np.logspace(math.log10(tau_min), math.log10(tau_max), n_points)
    logW = logW_func(tau)
    keep = logW > (logW[0] + math.log(negligible_ratio))
    tau_k = tau[keep]
    logW_k = logW[keep]
    tau_eff_max = float(tau_k[-1])

    if n_lo_init is None:
        n_lo_init = 0.1 / tau_eff_max
    if n_hi_init is None:
        n_hi_init = 10.0 / tau_min

    log_tau = np.log(tau_k)

    def _logfit(theta):
        u, v = theta[:M], theta[M:]
        # exponents: u_l - tau * exp(v_l); computed in log space (no m overflow)
        expo = u[None, :] - np.exp(log_tau[:, None] + v[None, :])
        mx = expo.max(axis=1, keepdims=True)
        p = np.exp(expo - mx)
        s = p.sum(axis=1, keepdims=True)
        return (mx[:, 0] + np.log(s[:, 0])), p / s

    def residual(theta):
        lf, _ = _logfit(theta)
        return lf - logW_k

    def jacobian(theta):
        v = theta[M:]
        _, p = _logfit(theta)
        # d/du_l = p_l ; d/dv_l = -tau n_l p_l
        du = p
        dv_ = -p * np.exp(log_tau[:, None] + v[None, :])
        return np.concatenate([du, dv_], axis=1)

    lo = np.concatenate([np.full(M, -40.0), np.full(M, math.log(1e-4 / tau_eff_max))])
    hi = np.concatenate([np.full(M, 40.0), np.full(M, math.log(1e3 / tau_min))])

    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        v0 = np.linspace(math.log(n_lo_init), math.log(n_hi_init), M)
        if r > 0:
            v0 = v0 + rng.normal(0.0, 0.3, size=M)
        dv = v0[1] - v0[0]
        # Laplace-quadrature initial weights for tau^{-1/2}: m = A*/sqrt(pi) sqrt(n) dv
        u0 = np.log(A_STAR / math.sqrt(math.pi) * np.exp(0.5 * v0) * dv)
        theta0 = np.clip(np.concatenate([u0, v0]), lo + 1e-6, hi - 1e-6)
        sol = least_squares(residual, theta0, jac=jacobian, bounds=(lo, hi), method="trf",
                            x_scale="jac", max_nfev=4000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if best is None or sol.cost < best.cost:
            best = sol

    # Minimax refinement (IRLS towards the L_p norm, p -> large) so that the
    # *maximum* relative error — the quantity gated by §3.3 — is minimised
    # rather than the RMS log error.
    theta = best.x.copy()
    p_norm = 8.0
    for _ in range(minimax_iters):
        r = residual(theta)
        wgt = (np.abs(r) + 1e-6) ** ((p_norm - 2.0) / 2.0)
        wgt /= wgt.max()
        sol = least_squares(lambda th: wgt * residual(th), theta,
                            jac=lambda th: wgt[:, None] * jacobian(th),
                            bounds=(lo, hi), method="trf", x_scale="jac",
                            max_nfev=2000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
        r_new = residual(sol.x)
        if np.max(np.abs(r_new)) < np.max(np.abs(r)):
            theta = sol.x
        else:
            break
    u, v = theta[:M], theta[M:]
    m, n = np.exp(u), np.exp(v)
    order = np.argsort(n)
    m, n = m[order], n[order]
    assert_positive_kernel(m, n)
    res = residual(theta)
    max_rel = float(np.max(np.abs(np.exp(res) - 1.0)))
    rms = float(np.sqrt(np.mean(res ** 2)))
    return KernelFit(name=name, m=m, n=n, tau_min=tau_min, tau_max=tau_max,
                     tau_eff_max=tau_eff_max, max_rel_err=max_rel, rms_log_err=rms,
                     n_points=int(tau_k.size))


def fit_zielke_kernel(M: int = 10, tau_min: float = 1e-8, tau_max: float = 1e2, **kw) -> KernelFit:
    return fit_exponential_sum(zielke_logW, M=M, tau_min=tau_min, tau_max=tau_max,
                               name="zielke_laminar", **kw)


def fit_powerlaw_kernel(M: int = 10, tau_min: float = 1e-8, tau_max: float = 1.0, **kw) -> KernelFit:
    """Fit of A*/sqrt(tau); combined with the VB scaling law n_l -> n_l + 1/C*(Re).

    tau_max = 1 is sufficient: for Re >= 2300, C* <= 6.6e-3 so exp(-tau/C*) < 1e-65
    at tau = 1; the VB kernel beyond that is numerically zero for any Re.
    """
    return fit_exponential_sum(lambda t: np.log(powerlaw_W(t)), M=M, tau_min=tau_min,
                               tau_max=tau_max, name="powerlaw_for_vardy_brown", **kw)


def kernel_for_reynolds(Re: float, zielke_fit: KernelFit, powerlaw_fit: KernelFit) -> Tuple[np.ndarray, np.ndarray]:
    """Return (m, n) of the dimensionless exponential sum appropriate for Re."""
    if Re < RE_LAMINAR_TURBULENT:
        return zielke_fit.m.copy(), zielke_fit.n.copy()
    C = vardy_brown_Cstar(Re)
    return powerlaw_fit.m.copy(), powerlaw_fit.n + 1.0 / C


def exact_W_for_reynolds(Re: float) -> Tuple[Callable, Callable]:
    """(W, int_0^tau W) for the reference (direct convolution) path."""
    if Re < RE_LAMINAR_TURBULENT:
        return zielke_W, zielke_W_integral
    return (lambda t: vardy_brown_W(t, Re)), (lambda t: vardy_brown_W_integral(t, Re))


# --------------------------------------------------------------------------
# Dimensional recursion coefficients and step-averaged direct weights
# --------------------------------------------------------------------------
def phi1(h: np.ndarray) -> np.ndarray:
    """(1 - exp(-h)) / h with a series for small h."""
    h = np.asarray(h, dtype=np.float64)
    out = np.empty_like(h)
    small = h < 1e-6
    out[small] = 1.0 - 0.5 * h[small]
    out[~small] = (1.0 - np.exp(-h[~small])) / h[~small]
    return out


@dataclass
class RecursiveKernel:
    """Dimensional coefficients for the M-term recursion at fixed (nu, D, dt)."""
    E: np.ndarray        # exp(-beta_l dt)          (M,)
    Phi: np.ndarray      # alpha_l * phi1(beta_l dt) (M,)
    w: float             # 16 nu / D^2              [1/s]
    beta: np.ndarray     # (M,) [1/s]
    alpha: np.ndarray    # (M,) dimensionless

    @staticmethod
    def build(m: np.ndarray, n: np.ndarray, nu: float, D: float, dt: float) -> "RecursiveKernel":
        assert_positive_kernel(m, n)
        beta = 4.0 * nu * n / D ** 2
        alpha = m.copy()
        E = np.exp(-beta * dt)
        Phi = alpha * phi1(beta * dt)
        return RecursiveKernel(E=E, Phi=Phi, w=16.0 * nu / D ** 2, beta=beta, alpha=alpha)


def direct_step_weights(W_integral: Callable[[np.ndarray], np.ndarray], nu: float, D: float,
                        dt: float, n_steps: int) -> np.ndarray:
    """Step-averaged weights  Wbar_k = (1/dt) int_{k dt}^{(k+1) dt} W(4 nu u / D^2) du.

    J^{n+1} = (16 nu / D^2) * sum_{k=0}^{n} Wbar_k * (V^{n+1-k} - V^{n-k}).
    """
    scale = 4.0 * nu / D ** 2
    edges = scale * dt * np.arange(n_steps + 1, dtype=np.float64)
    I = W_integral(edges)
    return (I[1:] - I[:-1]) / (scale * dt)


def expsum_step_weights(m: np.ndarray, n: np.ndarray, nu: float, D: float, dt: float, n_steps: int) -> np.ndarray:
    """Same step-averaged weights for the exponential-sum kernel (for W-level comparison)."""
    return direct_step_weights(lambda t: exp_sum_W_integral(t, m, n), nu, D, dt, n_steps)


def convolve_direct(dV: np.ndarray, Wbar: np.ndarray, w: float) -> np.ndarray:
    """Reference J_u history for a single point: J^{n} = w * sum_k Wbar_k dV^{n-k}.

    dV[j] = V^{j+1} - V^{j}; returns J at levels 1..len(dV).  O(N^2).
    """
    N = dV.size
    J = np.empty(N)
    for i in range(N):
        J[i] = w * np.dot(Wbar[: i + 1], dV[i::-1])
    return J


def convolve_recursive(dV: np.ndarray, rk: RecursiveKernel) -> np.ndarray:
    """Recursive J_u history for a single point with the same dV history."""
    N = dV.size
    z = np.zeros(rk.E.size)
    J = np.empty(N)
    for i in range(N):
        z = rk.E * z + rk.Phi * dV[i]
        J[i] = rk.w * z.sum()
    return J


# --------------------------------------------------------------------------
# Frozen kernel configuration (configs/friction_zielke.yaml)
# --------------------------------------------------------------------------
def build_kernel_config(M_production: int = 12, M_candidates=(10, 11, 12), tau_min: float = 1e-8,
                        tau_max: float = 1e2, gate_rel_err: float = 0.02) -> Dict:
    """Fit all candidate M and assemble the YAML content with the production choice."""
    fits = {}
    for M in M_candidates:
        fz = fit_zielke_kernel(M=M, tau_min=tau_min, tau_max=tau_max)
        fp = fit_powerlaw_kernel(M=M, tau_min=tau_min, tau_max=min(tau_max, 1.0))
        fits[M] = {"zielke_laminar": fz.as_dict(), "powerlaw_for_vardy_brown": fp.as_dict(),
                   "passes_gate": bool(fz.max_rel_err < gate_rel_err and fp.max_rel_err < gate_rel_err)}
    cfg = {
        "schema_version": 1,
        "model": "zielke_vardy_brown_memory_convolution",
        "nondimensional_time": "tau = 4 nu t / D^2",
        "reference_path": {
            "type": "direct_history_convolution_piecewise_complete_W",
            "zielke_small_tau_series": {"tau_split": ZIELKE_TAU_SPLIT, "m": ZIELKE_M_SMALL.tolist(),
                                        "powers": ZIELKE_POW_SMALL.tolist()},
            "zielke_large_tau_exponents": ZIELKE_N_LARGE.tolist(),
            "vardy_brown": {"A_star": A_STAR, "C_star": "12.86 / Re^kappa, kappa = log10(15.29 / Re^0.0567)",
                            "valid_Re": [2300.0, 1.0e8]},
            "laminar_turbulent_switch_Re": RE_LAMINAR_TURBULENT,
            "step_weights": "Wbar_k = (1/dt) int_{k dt}^{(k+1) dt} W(4 nu u / D^2) du (analytic antiderivatives)",
            "complexity": "O(N_x N_t^2) time, O(N_x N_t) memory — goldens and kernel-fit reference only",
            "prehistory": "z_l(0) = 0 <=> fully developed steady flow before the disturbance; non-steady prehistory must be supplied via z_l(0)",
        },
        "production_path": {
            "type": "exponential_recursive_kernel",
            "M": M_production,
            "recursion": "dz_l/dt = -beta_l z_l + alpha_l dV/dt,  J_uV = w sum_l z_l,  w = 16 nu/D^2, alpha_l = m_l, beta_l = 4 nu n_l / D^2",
            "exact_step_update": "z^{n+1} = exp(-beta dt) z^n + alpha phi1(beta dt) (V^{n+1} - V^n)",
            "positivity_constraint": "m_l, n_l > 0 enforced structurally (log-parametrised fit) + assertion after fit",
            "fit": {"objective": "log least squares + minimax (IRLS) refinement", "domain_tau": [tau_min, tau_max],
                    "powerlaw_domain_tau": [tau_min, min(tau_max, 1.0)],
                    "negligible_weight_rule": "points with W < 1e-16 W(tau_min) carry zero weight (below double-precision relevance)",
                    "gate_max_rel_err": gate_rel_err},
            "reynolds_dependence": "VB scaling law: {m_l, n_l + 1/C*(Re)} exact for every turbulent Re (no binning error); laminar uses the Zielke fit",
            "complexity": "O(N_x M N_t)",
        },
        "fits_by_M": fits,
    }
    if 10 in fits and M_production in fits:
        z10 = 100 * fits[10]["zielke_laminar"]["max_rel_err"]
        p10 = 100 * fits[10]["powerlaw_for_vardy_brown"]["max_rel_err"]
        zP = 100 * fits[M_production]["zielke_laminar"]["max_rel_err"]
        pP = 100 * fits[M_production]["powerlaw_for_vardy_brown"]["max_rel_err"]
        cfg["repair_record"] = (
            f"M=10 (design value of §2.4(b)) does not meet the <2% fit gate on tau in [{tau_min:g}, {tau_max:g}] "
            f"(Zielke {z10:.2f}%, power law {p10:.2f}%); M={M_production} adopted (Zielke {zP:.2f}%, power law {pP:.2f}%). "
            "Repair order per §3.4: 量纲 -> CFL/插值 -> 节点闭合 -> 记忆核 (this item)."
        )
    return cfg


def kernel_fits_from_config(cfg: Dict, M: int | None = None) -> Tuple[KernelFit, KernelFit]:
    M = M or cfg["production_path"]["M"]
    f = cfg["fits_by_M"][M]

    def mk_fit(d):
        return KernelFit(name=d["name"], m=np.array(d["m"]), n=np.array(d["n"]), tau_min=d["tau_min"],
                         tau_max=d["tau_max"], tau_eff_max=d["tau_eff_max"], max_rel_err=d["max_rel_err"],
                         rms_log_err=d["rms_log_err"], n_points=d["n_points"])

    fz, fp = mk_fit(f["zielke_laminar"]), mk_fit(f["powerlaw_for_vardy_brown"])
    assert_positive_kernel(fz.m, fz.n)
    assert_positive_kernel(fp.m, fp.n)
    return fz, fp


# --------------------------------------------------------------------------
# Error decomposition helpers (§2.4(b): kernel fit error vs. convolution error)
# --------------------------------------------------------------------------
def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def phase_lag_deg(x: np.ndarray, ref: np.ndarray, dt: float, fmin: float, fmax: float,
                  energy_fraction: float = 0.9) -> Tuple[float, float]:
    """Maximum |phase difference| (deg) between x and ref over the band [fmin,fmax]
    restricted to the frequencies that carry `energy_fraction` of ref's band energy.

    Returns (max_abs_phase_deg, phase_at_peak_deg).
    """
    n = len(ref)
    X = np.fft.rfft(x - x.mean())
    R = np.fft.rfft(ref - ref.mean())
    f = np.fft.rfftfreq(n, dt)
    band = (f >= fmin) & (f <= fmax)
    P = np.abs(R[band]) ** 2
    if P.sum() == 0:
        return 0.0, 0.0
    order = np.argsort(P)[::-1]
    csum = np.cumsum(P[order]) / P.sum()
    sel = order[: int(np.searchsorted(csum, energy_fraction)) + 1]
    cross = X[band][sel] * np.conj(R[band][sel])
    ph = np.degrees(np.angle(cross))
    peak = np.degrees(np.angle(cross[0]))
    return float(np.max(np.abs(ph))), float(peak)
