# -*- coding: utf-8 -*-
"""Golden-case builders for the Stage-1 acceptance gates (§3.3).

All cases are built from configs/physics.yaml so that the gate table and the
data manifests trace back to the same frozen numbers.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

import moc_solver as ms
from config_io import load_yaml

G = 9.80665


def reference_well_dict() -> Dict:
    return load_yaml("physics.yaml")["reference_well"]


def make_clusters(x_clusters: List[float], nperf: int, Cd: float, d_perf: float, kappa: float, I_f: float,
                  R_f: float, C_f: float, G_l: float, p_res: float, rho: float) -> List[ms.ClusterSpec]:
    A_perf = math.pi * d_perf ** 2 / 4.0
    out = []
    for x in x_clusters:
        M = nperf
        out.append(ms.ClusterSpec.from_perf_params(
            x=float(x), Cd=np.full(M, Cd), A_perf=np.full(M, A_perf), kappa=np.full(M, kappa), I_f=np.full(M, I_f),
            R_f=np.full(M, R_f), C_f=np.full(M, C_f), G_l=np.full(M, G_l), p_res=p_res, rho=rho))
    return out


def reference_case(spacing: float | None = None, N: int | None = None, **overrides) -> ms.WellSpec:
    """The typical multi-cluster case of physics.yaml (optionally with a different spacing / N)."""
    r = reference_well_dict()
    c = r["clusters"]
    N = N or int(c["N"])
    s = spacing if spacing is not None else float(c["spacing"])
    x1 = r["L"] - c["L_toe"] - (N - 1) * s
    xs = [x1 + j * s for j in range(N)]
    clusters = make_clusters(xs, int(c["perfs_per_cluster"]), c["Cd"], c["d_perf"], c["kappa"], c["I_f"], c["R_f"],
                             c["C_f"], c["G_l"], c["p_res"], r["rho"])
    kw = dict(L=r["L"], D=r["D"], a=r["a"], rho=r["rho"], nu=r["nu"], roughness=r["roughness"], TVD=r["TVD"],
              clusters=clusters, Q0=r["Q0"], t_s=r["t_s"], t_c=r["t_c"], ramp=r["ramp"], toe_bc=r["toe_bc"])
    kw.update(overrides)
    return ms.WellSpec(**kw)


def joukowsky_case(V0: float = 2.0) -> ms.WellSpec:
    """Frictionless single pipe, upstream valve (wellhead) step closure, constant-head toe."""
    r = reference_well_dict()
    A = math.pi * r["D"] ** 2 / 4.0
    return ms.WellSpec(L=r["L"], D=r["D"], a=r["a"], rho=r["rho"], nu=r["nu"], roughness=r["roughness"], TVD=r["TVD"],
                       clusters=[], Q0=V0 * A, t_s=1.0, t_c=0.0, ramp="step", toe_bc="reservoir", H_toe=1000.0)


def shunt_case(R_f: float = 2e7, G_l: float = 1e-8, x_c: float = 2000.0, Q_pulse: float = 1e-3, t_c: float = 0.02) -> ms.WellSpec:
    """Single resistive shunt (linear branch Z_b = R_f + 1/G_l), Gaussian flow pulse from rest."""
    r = reference_well_dict()
    cl = ms.ClusterSpec(x=x_c, K=np.zeros(1), kappa=np.zeros(1), I_f=np.zeros(1), R_f=np.array([R_f]),
                        C_f=np.zeros(1), G_l=np.array([G_l]), p_res=r["clusters"]["p_res"])
    return ms.WellSpec(L=r["L"], D=r["D"], a=r["a"], rho=r["rho"], nu=r["nu"], roughness=r["roughness"], TVD=r["TVD"],
                       clusters=[cl], Q0=Q_pulse, t_s=0.5, t_c=t_c, ramp="gauss_pulse")


def rci_case(R_f: float = 1e8, I_f: float = 5e5, C_f: float = 1e-6, G_l: float = 2e-10, x_c: float = 2000.0,
             Q_pulse: float = 1e-3, t_c: float = 0.01) -> ms.WellSpec:
    """Single linear RCI branch (K = 0), broadband Gaussian pulse: q/p_w vs 1/Z_b(jw)."""
    r = reference_well_dict()
    cl = ms.ClusterSpec(x=x_c, K=np.zeros(1), kappa=np.zeros(1), I_f=np.array([I_f]), R_f=np.array([R_f]),
                        C_f=np.array([C_f]), G_l=np.array([G_l]), p_res=r["clusters"]["p_res"])
    return ms.WellSpec(L=r["L"], D=r["D"], a=r["a"], rho=r["rho"], nu=r["nu"], roughness=r["roughness"], TVD=r["TVD"],
                       clusters=[cl], Q0=Q_pulse, t_s=0.5, t_c=t_c, ramp="gauss_pulse")


def tmm_case(N: int = 4, s: float = 25.0, L_toe: float = 60.0, R_f: float = 1e8, I_f: float = 5e5, C_f: float = 1e-6,
             G_l: float = 2e-10, Q0: float = 0.02, t_c: float = 0.05) -> ms.WellSpec:
    """Frictionless wellbore with N linear (K = 0) RCI shunts for the transfer-matrix resonance check."""
    r = reference_well_dict()
    x1 = r["L"] - L_toe - (N - 1) * s
    clusters = [ms.ClusterSpec(x=x1 + j * s, K=np.zeros(1), kappa=np.zeros(1), I_f=np.array([I_f]), R_f=np.array([R_f]),
                               C_f=np.array([C_f]), G_l=np.array([G_l]), p_res=r["clusters"]["p_res"]) for j in range(N)]
    return ms.WellSpec(L=r["L"], D=r["D"], a=r["a"], rho=r["rho"], nu=r["nu"], roughness=r["roughness"], TVD=r["TVD"],
                       clusters=clusters, Q0=Q0, t_s=1.0, t_c=t_c, ramp="cosine")


def bergant2001_case(V0: float = 0.1) -> ms.WellSpec:
    """Bergant, Simpson & Vítkovský (2001) laboratory apparatus, Case 1 (laminar, Re = 1870).

    L = 37.23 m, D = 22.1 mm, a = 1319 m/s, upstream constant-head tank H_r = 32 m, downstream ball valve
    closed in t_c = 9 ms.  In the solver's frame the valve is the *upstream* boundary (x = 0) and the
    reservoir the *toe*; flow is therefore reversed (Q0 < 0: fluid moves from the toe towards the valve).
    The pressure at the valve (x = 0) is the observable.  Water at 15.4 °C: nu ≈ 1.14e-6 m^2/s.
    """
    L, D, a = 37.23, 0.0221, 1319.0
    nu = 1.14e-6
    A = math.pi * D ** 2 / 4.0
    return ms.WellSpec(L=L, D=D, a=a, rho=999.0, nu=nu, roughness=1.5e-6, TVD=0.0, clusters=[],
                       Q0=-V0 * A, t_s=0.2, t_c=0.009, ramp="linear", toe_bc="reservoir", H_toe=32.0)
