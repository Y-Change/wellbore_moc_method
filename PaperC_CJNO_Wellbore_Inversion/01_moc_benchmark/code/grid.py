# -*- coding: utf-8 -*-
"""Segment-wise (locally non-uniform) MOC grid with clusters placed exactly on nodes.

攻关执行方案_v2 §3.1 "簇节点落网策略（硬承诺）": cluster positions sampled
continuously must not be quantised to a uniform grid (±dx/2 -> up to 2dx/a ≈ 5.7 ms
arrival error).  Instead the wellbore is split into segments between clusters
(and the two ends); every segment gets its own uniform spacing dx_j = L_j / N_j so
that each cluster sits exactly on a node.  A single global time step is used:

    dt = Cr_target * dx_0 / a            (dx_0: wellhead segment, the longest path)
    N_j = max(1, floor(L_j / dx_0))  ->  dx_j >= dx_0  ->  Cr_j = a dt / dx_j <= Cr_target

Segments with Cr_j < 1 use linear space-line interpolation at the characteristic
feet (numerical dissipation/dispersion per Ghidaoui & Karney 1994 — reported in
the stage conclusion and in the s_min = 10 m convergence check).  With
Cr_target = 1 the wellhead segment is interpolation-free (Cr_0 = 1 exactly).

If a segment is shorter than dx_0 (only possible when a cluster spacing is below
the reference spacing) N_j = 1 and dt is reduced globally so that max Cr_j <= 1
(flagged in ``Grid.dt_reduced``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class Grid:
    L: float
    a: float
    Nx_ref: int
    Cr_target: float
    x_clusters: np.ndarray            # sorted cluster positions [m]
    seg_len: np.ndarray               # (nseg,)
    seg_N: np.ndarray                 # cells per segment (nseg,) int
    seg_dx: np.ndarray                # (nseg,)
    seg_Cr: np.ndarray                # (nseg,)
    seg_off: np.ndarray               # node offset of each segment in the flat array (nseg,) int
    dt: float
    n_nodes: int
    x_nodes: np.ndarray               # (n_nodes,) coordinate of every stored node (cluster nodes appear twice)
    dt_reduced: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def nseg(self) -> int:
        return int(self.seg_N.size)

    @property
    def n_clusters(self) -> int:
        return int(self.x_clusters.size)

    def cluster_left_index(self, j: int) -> int:
        """Flat index of the upstream-side copy of cluster j (end of segment j)."""
        return int(self.seg_off[j] + self.seg_N[j])

    def cluster_right_index(self, j: int) -> int:
        """Flat index of the downstream-side copy of cluster j (start of segment j+1)."""
        return int(self.seg_off[j + 1])

    def summary(self) -> dict:
        return {
            "L": self.L, "a": self.a, "Nx_ref": self.Nx_ref, "Cr_target": self.Cr_target,
            "dt": self.dt, "n_nodes": self.n_nodes, "nseg": self.nseg,
            "seg_len": self.seg_len.tolist(), "seg_N": self.seg_N.tolist(),
            "seg_dx": self.seg_dx.tolist(), "seg_Cr": self.seg_Cr.tolist(),
            "dt_reduced": self.dt_reduced, "notes": list(self.notes),
        }


def build_grid(L: float, a: float, x_clusters: Sequence[float], Nx_ref: int, Cr_target: float,
               min_cells_per_segment: int = 1) -> Grid:
    xc = np.sort(np.asarray(x_clusters, dtype=np.float64))
    if xc.size and (xc[0] <= 0.0 or xc[-1] >= L):
        raise ValueError("cluster positions must lie strictly inside (0, L)")
    if xc.size > 1 and np.any(np.diff(xc) <= 0.0):
        raise ValueError("cluster positions must be strictly increasing")
    bounds = np.concatenate([[0.0], xc, [L]])
    seg_len = np.diff(bounds)
    dx_ref = L / float(Nx_ref)

    notes: List[str] = []
    # wellhead segment defines the interpolation-free reference spacing
    N0 = max(1, int(round(seg_len[0] / dx_ref)))
    dx0 = seg_len[0] / N0
    dt = Cr_target * dx0 / a

    seg_N = np.empty(seg_len.size, dtype=np.int64)
    seg_N[0] = N0
    for j in range(1, seg_len.size):
        seg_N[j] = max(min_cells_per_segment, int(np.floor(seg_len[j] / dx0 + 1e-12)))
    seg_dx = seg_len / seg_N
    seg_Cr = a * dt / seg_dx
    dt_reduced = False
    if np.any(seg_Cr > 1.0 + 1e-12):
        # a segment shorter than dx0: shrink dt so that max Cr = Cr_target
        dt = Cr_target * seg_dx.min() / a
        seg_Cr = a * dt / seg_dx
        dt_reduced = True
        notes.append("dt reduced globally because a segment is shorter than the reference spacing")
    # snap Cr values that are 1 up to round-off to exactly 1 (no interpolation)
    seg_Cr = np.where(np.abs(seg_Cr - 1.0) < 1e-12, 1.0, seg_Cr)

    seg_off = np.zeros(seg_len.size, dtype=np.int64)
    for j in range(1, seg_len.size):
        seg_off[j] = seg_off[j - 1] + seg_N[j - 1] + 1
    n_nodes = int(seg_off[-1] + seg_N[-1] + 1)
    x_nodes = np.empty(n_nodes)
    for j in range(seg_len.size):
        x_nodes[seg_off[j]: seg_off[j] + seg_N[j] + 1] = bounds[j] + seg_dx[j] * np.arange(seg_N[j] + 1)

    return Grid(L=L, a=a, Nx_ref=Nx_ref, Cr_target=Cr_target, x_clusters=xc, seg_len=seg_len,
                seg_N=seg_N, seg_dx=seg_dx, seg_Cr=seg_Cr, seg_off=seg_off, dt=dt,
                n_nodes=n_nodes, x_nodes=x_nodes, dt_reduced=dt_reduced, notes=notes)


def interpolation_amplitude_factor(Cr: float, k_dx: float) -> float:
    """Per-step amplitude factor of linear space-line interpolation for a mode
    with wavenumber k (k*dx given), cf. Ghidaoui & Karney (1994):
        |G|^2 = 1 - 2 Cr (1 - Cr) (1 - cos(k dx)).
    """
    G2 = 1.0 - 2.0 * Cr * (1.0 - Cr) * (1.0 - np.cos(k_dx))
    return float(np.sqrt(max(G2, 0.0)))


def interpolation_dissipation_budget(grid: Grid, freqs_hz: Sequence[float]) -> List[dict]:
    """Amplitude loss per single traversal of each segment for the given frequencies.

    Returned per (segment, frequency): points per wavelength, |G| per step,
    and the amplitude factor after N_j steps (one traversal of the segment).
    """
    out = []
    for j in range(grid.nseg):
        for f in freqs_hz:
            lam = grid.a / f
            k_dx = 2.0 * np.pi * grid.seg_dx[j] / lam
            G = interpolation_amplitude_factor(float(grid.seg_Cr[j]), k_dx)
            out.append({
                "segment": j, "freq_hz": float(f), "dx": float(grid.seg_dx[j]), "Cr": float(grid.seg_Cr[j]),
                "points_per_wavelength": float(lam / grid.seg_dx[j]),
                "amp_factor_per_step": G,
                "amp_factor_per_traversal": float(G ** int(grid.seg_N[j])),
            })
    return out
