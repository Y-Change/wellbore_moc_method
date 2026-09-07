# -*- coding: utf-8 -*-
"""Steady (quasi-steady Darcy–Weisbach) friction and Brunone coefficient.

* Steady/quasi-steady Darcy friction factor: Churchill (1977) universal
  correlation, smooth across laminar / transition / turbulent regimes
  (f -> 64/Re as Re -> 0, so f Q|Q| -> 64 nu A Q / D is linear and finite).
  A log-spaced lookup table is used inside the numba kernels.

* Brunone coefficient from Vardy's shear-decay coefficient (Bergant et al. 2001):
      k = sqrt(C*) / 2,
      C* = 0.00476                                (laminar, Re < 2300)
      C* = 7.41 / Re**(log10(14.3 / Re**0.05))    (turbulent)
  The convention J_u = k (V_t - a sgn(V V_x) V_x) (coefficient k, not k/2) is
  locked in configs/friction_brunone.yaml (攻关执行方案_v2 §2.4(c)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def churchill_f(Re: np.ndarray, rel_rough: float) -> np.ndarray:
    """Churchill (1977) friction factor, valid for all Re > 0."""
    Re = np.maximum(np.asarray(Re, dtype=np.float64), 1e-12)
    A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * rel_rough))) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)


@dataclass
class FrictionTable:
    """Lookup of f(log10 Re) on a uniform grid for fast numba evaluation."""
    logRe_min: float
    dlogRe: float
    values: np.ndarray   # f at logRe_min + k*dlogRe
    f_laminar_coeff: float = 64.0

    @staticmethod
    def build(rel_rough: float, logRe_min: float = -2.0, logRe_max: float = 9.0, n: int = 4096) -> "FrictionTable":
        logRe = np.linspace(logRe_min, logRe_max, n)
        f = churchill_f(10.0 ** logRe, rel_rough)
        return FrictionTable(logRe_min=logRe_min, dlogRe=float(logRe[1] - logRe[0]), values=f)


def brunone_Cstar(Re: float) -> float:
    Re = float(Re)
    if Re < 2300.0:
        return 0.00476
    return 7.41 / Re ** (math.log10(14.3 / Re ** 0.05))


def brunone_k(Re: float) -> float:
    """k = sqrt(C*)/2 (Bergant, Simpson & Vítkovský 2001; Vardy & Brown C*)."""
    return math.sqrt(brunone_Cstar(Re)) / 2.0


def brunone_characteristic_speeds(a: float, k: float, sgn_VVx: int) -> tuple[float, float]:
    """Eigen-speeds of the linearised system when the Brunone term
    J_u = k (V_t - a sgn(V V_x) V_x) is merged into the principal part.

    sgn(V V_x) = +1 :  (1+k) V_t + g H_x - k a V_x = 0  ->  speeds (-a, a/(1+k))
    sgn(V V_x) = -1 :  (1+k) V_t + g H_x + k a V_x = 0  ->  speeds (+a, -a/(1+k))
    Returned sorted ascending. (攻关执行方案_v2 §2.4(c))
    """
    g = 9.80665
    s = float(sgn_VVx)
    # system U_t + M U_x = 0, U = (V, H):
    # V_t = (s k a/(1+k)) V_x - (g/(1+k)) H_x ;  H_t = -(a^2/g) V_x
    M = np.array([[-s * k * a / (1.0 + k), g / (1.0 + k)],
                  [a * a / g, 0.0]])
    lam = np.sort(np.linalg.eigvals(M).real)
    return float(lam[0]), float(lam[1])
