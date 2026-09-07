# -*- coding: utf-8 -*-
"""Frequency-domain transfer-matrix model (TMM) for cross-validation of the MOC (§3.3).

Small-signal, p–Q variables (Z_c = rho a / A, §2.1 impedance discipline: p–Q pairs only
with Z_c, never with the head-form B = a/(gA)).

Line of length L (x increasing downhole, Q positive downhole):
    [p_u; Q_u] = [[cosh(gamma L),  Z sinh(gamma L)],
                  [sinh(gamma L)/Z, cosh(gamma L)]] [p_d; Q_d]
with gamma = sqrt(Z_s Y_p), Z = sqrt(Z_s / Y_p), series impedance per length
    Z_s(w) = j w rho / A + R' + Z_u'(w),     shunt admittance per length  Y_p = j w A / (rho a^2)
where R' is the linearised steady-friction resistance and Z_u' the memory-kernel
impedance  (rho/A) * w_k * sum_l alpha_l j w / (j w + beta_l)  (positive real).

Shunt node (cluster) with branch impedance
    Z_b(s) = R_perf + R_f + s I_f + 1 / (s C_f + G_l),  R_perf = 2 K_p |qbar|   (§2.3 rule 1)
per perforation; perforations of one cluster act in parallel: Y_b = sum_m 1/Z_b,m.
    p continuous,  Q_u = Q_d + Y_b p.

Reflection / transmission of a pressure wave at a single shunt (both sides same Z_c):
    Gamma = -Z_c / (2 Z_b + Z_c),   T = 1 + Gamma.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class BranchLin:
    R_perf: float
    R_f: float
    I_f: float
    C_f: float
    G_l: float

    def Z(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=complex)
        return self.R_perf + self.R_f + s * self.I_f + 1.0 / (s * self.C_f + self.G_l)


def shunt_reflection(Z_c: float, Z_b: np.ndarray):
    Gamma = -Z_c / (2.0 * Z_b + Z_c)
    return Gamma, 1.0 + Gamma


@dataclass
class TMMWell:
    seg_len: np.ndarray                 # (nseg,)  wellhead segment first
    a: float
    rho: float
    A: float
    branches: List[List[BranchLin]]     # per cluster: list of perforation branches
    R_prime: float = 0.0                # linearised steady friction per length [Pa s m^-4]
    kernel: Optional[dict] = None       # {'w': float, 'alpha': (M,), 'beta': (M,)} for unsteady friction

    @property
    def Z_c(self) -> float:
        return self.rho * self.a / self.A

    def series_impedance(self, w: np.ndarray) -> np.ndarray:
        s = 1j * w
        Zs = s * self.rho / self.A + self.R_prime
        if self.kernel is not None:
            k = self.kernel
            Zs = Zs + (self.rho / self.A) * k["w"] * np.sum(
                k["alpha"][None, :] * s[:, None] / (s[:, None] + k["beta"][None, :]), axis=1)
        return Zs

    def wellhead_state(self, w: np.ndarray):
        """Propagate (p, Q) = (1, 0) from the dead-end toe to the wellhead."""
        w = np.asarray(w, dtype=float)
        s = 1j * w
        Zs = self.series_impedance(w)
        Yp = s * self.A / (self.rho * self.a ** 2)
        gamma = np.sqrt(Zs * Yp)
        Z = np.sqrt(Zs / Yp)
        p = np.ones_like(s)
        Q = np.zeros_like(s)
        nseg = len(self.seg_len)
        for j in range(nseg - 1, -1, -1):
            gl = gamma * self.seg_len[j]
            ch, sh = np.cosh(gl), np.sinh(gl)
            p, Q = ch * p + Z * sh * Q, sh / Z * p + ch * Q
            if j > 0:
                Yb = np.zeros_like(s)
                for br in self.branches[j - 1]:
                    Yb = Yb + 1.0 / br.Z(s)
                Q = Q + Yb * p
        return p, Q

    def input_impedance(self, w: np.ndarray) -> np.ndarray:
        p, Q = self.wellhead_state(w)
        return p / Q

    def natural_frequencies(self, f_max: float, n_grid: int = 200000, f_min: float = 1e-3) -> np.ndarray:
        """Frequencies (Hz) where |Q_wellhead| has a local minimum (closed wellhead: Q = 0),
        refined by parabolic interpolation of log|Q|."""
        f = np.linspace(f_min, f_max, n_grid)
        p, Q = self.wellhead_state(2.0 * np.pi * f)
        m = np.log(np.abs(Q) + 1e-300)
        idx = np.where((m[1:-1] < m[:-2]) & (m[1:-1] < m[2:]))[0] + 1
        out = []
        for i in idx:
            y0, y1, y2 = m[i - 1], m[i], m[i + 1]
            den = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            out.append(f[i] + delta * (f[1] - f[0]))
        return np.array(out)


def spectral_peaks(x: np.ndarray, dt: float, f_max: float, n_peaks: int = 12, pad: int = 8,
                   f_min: float = 1e-3) -> np.ndarray:
    """Frequencies of the strongest spectral peaks of x (Hann window, zero padding,
    parabolic interpolation in log magnitude)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    win = np.hanning(n)
    N = int(2 ** np.ceil(np.log2(n * pad)))
    X = np.abs(np.fft.rfft(x * win, N))
    f = np.fft.rfftfreq(N, dt)
    band = (f >= f_min) & (f <= f_max)
    m = np.log(X + 1e-300)
    idx = np.where((m[1:-1] > m[:-2]) & (m[1:-1] > m[2:]))[0] + 1
    idx = idx[band[idx]]
    idx = idx[np.argsort(X[idx])[::-1]][:n_peaks]
    out = []
    for i in idx:
        y0, y1, y2 = m[i - 1], m[i], m[i + 1]
        den = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        out.append(f[i] + delta * (f[1] - f[0]))
    return np.sort(np.array(out))


def match_frequencies(f_ref: np.ndarray, f_test: np.ndarray, rel_tol: float = 0.05):
    """Match each reference frequency to the closest test frequency; return relative errors."""
    out = []
    for fr in f_ref:
        if f_test.size == 0:
            out.append(np.nan)
            continue
        k = int(np.argmin(np.abs(f_test - fr)))
        rel = (f_test[k] - fr) / fr
        out.append(rel if abs(rel) <= rel_tol else np.nan)
    return np.array(out)
