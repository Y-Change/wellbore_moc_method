# -*- coding: utf-8 -*-
"""Deterministic characteristic loop (integration reference, not full CJ-NO).

Path: valve → delay-line C+/C− → real Newton node → reflect/transmit → wellhead.
Hard Newton writes H,Q that later return to the wellhead.
One-step teacher uses known previous state; free roll uses own q,pc,F.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from dataset import InitialState, PhysicalInputs, cosine_valve_Q, orifice_K
from ift_layer import cluster_newton
from node_coeffs import collect_branch_coeffs, darcy_R, update_pc_q


def interp_hist(hist: np.ndarray, n: int, delay: float, default: float) -> float:
    """Sample hist at n-delay using values written on 0..n-1 only."""
    tau = float(n) - float(delay)
    if tau <= 0.0:
        return float(default)
    i0 = int(np.floor(tau))
    i0 = max(0, min(i0, n - 1))
    i1 = min(i0 + 1, n - 1)
    w = tau - i0
    return float((1.0 - w) * hist[i0] + w * hist[i1])


def interp_list(xs, n: int, delay: float, default):
    tau = float(n) - float(delay)
    if tau <= 0.0:
        return default
    i0 = int(np.floor(tau))
    i0 = max(0, min(i0, n - 1))
    i1 = min(i0 + 1, n - 1)
    w = tau - i0
    return (1.0 - w) * xs[i0] + w * xs[i1]


@dataclass
class LoopResult:
    p_wh: np.ndarray
    H_wh: np.ndarray
    H_nodes: np.ndarray
    Qm: np.ndarray
    Qp: np.ndarray
    sq: np.ndarray
    q_perf: List[np.ndarray]
    pc_perf: List[np.ndarray]
    resid: np.ndarray
    n_hard: int
    n_fail: int
    ok: bool
    mode: str


class CharacteristicLoop:
    def __init__(self, phys: PhysicalInputs, dt: float, n_steps: int, use_friction: bool = True):
        self.phys = phys
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        xs = np.concatenate([[0.0], phys.xs, [phys.L]])
        self.xs = xs
        self.dx = np.diff(xs)
        self.delay = self.dx / phys.a / self.dt
        self.B = float(phys.B)
        f = 0.02
        self.R = darcy_R(f, self.dx, phys.D, phys.A, 9.80665) if use_friction else np.zeros_like(self.dx)

    def _CP(self, H, Qr, R):
        return H + self.B * Qr - 0.5 * R * Qr * abs(Qr)

    def _CM(self, H, Ql, R):
        return H - self.B * Ql + 0.5 * R * Ql * abs(Ql)

    def rollout(self, Q_valve: np.ndarray, initial: InitialState,
                require_ok: bool = True, teacher=None) -> LoopResult:
        """teacher: optional dict with H,q,pc,F lists at each step (one-step)."""
        p = self.phys
        N = p.N
        nT = self.n_steps
        mode = "one_step_teacher" if teacher is not None else "free_roll"
        H_wh = np.zeros(nT)
        p_wh = np.zeros(nT)
        Hn = np.zeros((nT, N))
        Qm = np.zeros((nT, N))
        Qp = np.zeros((nT, N))
        sq = np.zeros((nT, N))
        resid = np.zeros((nT, N))
        q_perf = [np.zeros((nT, p.Cd[j].size)) for j in range(N)]
        pc_perf = [np.zeros((nT, p.Cd[j].size)) for j in range(N)]
        nS = N + 2
        Hh = np.zeros((nS, nT))
        Qrh = np.zeros((nS, nT))
        Qlh = np.zeros((nS, nT))
        CPh = np.zeros((nS, nT))
        CMh = np.zeros((nS, nT))
        H0w = initial.p0_wh / p.rho_g
        Hh[0, 0] = H0w
        for j in range(N):
            Hh[j + 1, 0] = float(initial.H0_nodes[j])
            Qlh[j + 1, 0] = float(initial.Qm0[j])
            Qrh[j + 1, 0] = float(initial.Qp0[j])
        Hh[N + 1, 0] = float(initial.H0_nodes[-1])
        Qrh[0, 0] = initial.Q0_wh
        Qlh[0, 0] = initial.Q0_wh
        for s in range(nS):
            CPh[s, 0] = self._CP(Hh[s, 0], Qrh[s, 0], self.R[min(s, len(self.R) - 1)] if s < nS - 1 else 0.0)
            CMh[s, 0] = self._CM(Hh[s, 0], Qlh[s, 0], self.R[max(s - 1, 0)] if s > 0 else self.R[0])
        q_state = [a.copy() for a in initial.q0]
        pc_state = [a.copy() for a in initial.pc0]
        F_state = [a.copy() for a in initial.F0]
        n_hard = 0
        n_fail = 0
        H_scale = 4000.0
        Q_scale = max(abs(p.Q0), 1e-3)
        p_scale = p.rho_g * H_scale
        H_wh[0] = Hh[0, 0]
        p_wh[0] = H_wh[0] * p.rho_g
        for j in range(N):
            q_perf[j][0] = q_state[j]
            pc_perf[j][0] = pc_state[j]
            Hn[0, j] = Hh[j + 1, 0]
            Qm[0, j] = Qlh[j + 1, 0]
            Qp[0, j] = Qrh[j + 1, 0]
            sq[0, j] = float(q_state[j].sum())
        for n in range(1, nT):
            Qv = float(Q_valve[n])
            CM0 = interp_hist(CMh[1], n, self.delay[0], CMh[1, 0])
            Hw = CM0 + self.B * Qv
            Hh[0, n] = Hw
            Qrh[0, n] = Qv
            Qlh[0, n] = Qv
            CPh[0, n] = self._CP(Hw, Qv, self.R[0])
            CMh[0, n] = self._CM(Hw, Qv, self.R[0])
            H_wh[n] = Hw
            p_wh[n] = Hw * p.rho_g
            for j in range(N):
                s = j + 1
                CP = interp_hist(CPh[s - 1], n, self.delay[s - 1], CPh[s - 1, 0])
                CM = interp_hist(CMh[s + 1], n, self.delay[s], CMh[s + 1, 0])
                if teacher is not None:
                    q_use = teacher["q"][j][n - 1]
                    pc_use = teacher["pc"][j][n - 1]
                    F_use = teacher["F"][j][n - 1]
                    H_use = teacher["H"][j][n - 1]
                else:
                    q_use, pc_use, F_use = q_state[j], pc_state[j], F_state[j]
                    H_use = Hh[s, n - 1]
                br = collect_branch_coeffs(
                    self.dt, p.I_f[j], p.R_f[j], p.C_f[j], p.G_l[j], float(p.p_res[j]),
                    q_use, pc_use, F_use,
                )
                H, Qmm, Qpp, q, stat = cluster_newton(
                    torch.tensor(CP, dtype=torch.float64),
                    torch.tensor(CM, dtype=torch.float64),
                    torch.tensor(self.B, dtype=torch.float64),
                    torch.tensor(self.B, dtype=torch.float64),
                    torch.tensor(0.0, dtype=torch.float64),
                    torch.tensor(0.0, dtype=torch.float64),
                    torch.tensor(p.z_of_x(p.xs[j]), dtype=torch.float64),
                    torch.tensor(p.rho_g, dtype=torch.float64),
                    torch.as_tensor(p.K[j], dtype=torch.float64),
                    torch.as_tensor(p.kappa[j], dtype=torch.float64),
                    1e-8,
                    torch.as_tensor(br["c1"], dtype=torch.float64),
                    torch.as_tensor(br["c0"], dtype=torch.float64),
                    torch.as_tensor(br["a1"], dtype=torch.float64),
                    torch.as_tensor(br["a0"], dtype=torch.float64),
                    torch.tensor(float(H_use), dtype=torch.float64),
                    torch.as_tensor(q_use, dtype=torch.float64),
                    H_scale, Q_scale, p_scale,
                )
                n_hard += 1
                ok = bool(stat[1] > 0.5)
                if not ok:
                    n_fail += 1
                    if require_ok:
                        raise RuntimeError(
                            f"Newton failed at n={n} cluster={j} resid={float(stat[2])}"
                        )
                qn = q.detach().cpu().numpy()
                q_state[j], pc_state[j], F_state[j] = update_pc_q(
                    qn, br["a0"], br["a1"], br["c0"], br["c1"]
                )
                Hh[s, n] = float(H)
                Qlh[s, n] = float(Qmm)
                Qrh[s, n] = float(Qpp)
                CPh[s, n] = self._CP(Hh[s, n], Qrh[s, n], self.R[s] if s < len(self.R) else 0.0)
                CMh[s, n] = self._CM(Hh[s, n], Qlh[s, n], self.R[s - 1])
                Hn[n, j] = float(H)
                Qm[n, j] = float(Qmm)
                Qp[n, j] = float(Qpp)
                sq[n, j] = float(qn.sum())
                resid[n, j] = float(stat[2])
                q_perf[j][n] = q_state[j]
                pc_perf[j][n] = pc_state[j]
            CPt = interp_hist(CPh[N], n, self.delay[N], CPh[N, 0])
            Hh[N + 1, n] = CPt
            Qlh[N + 1, n] = 0.0
            Qrh[N + 1, n] = 0.0
            CPh[N + 1, n] = self._CP(CPt, 0.0, 0.0)
            CMh[N + 1, n] = self._CM(CPt, 0.0, self.R[N])
        return LoopResult(
            p_wh=p_wh, H_wh=H_wh, H_nodes=Hn, Qm=Qm, Qp=Qp, sq=sq,
            q_perf=q_perf, pc_perf=pc_perf, resid=resid,
            n_hard=n_hard, n_fail=n_fail, ok=n_fail == 0, mode=mode,
        )


def synthetic_single_cluster(x=800.0, L=1600.0, a=1000.0, D=0.08, Q0=0.03, t_c=0.2,
                             M=2, hetero=False) -> PhysicalInputs:
    rho = 1000.0
    Cd = np.array([0.8, 0.6] if hetero else [0.7, 0.7], dtype=np.float64)
    A = np.array([1.2e-4, 8e-5] if hetero else [1e-4, 1e-4], dtype=np.float64)
    kap = np.array([2e6, 5e5] if hetero else [1e6, 1e6], dtype=np.float64)
    If = np.full(M, 4e5)
    Rf = np.full(M, 2e7)
    Cf = np.array([3e-6, 1e-6] if hetero else [2e-6, 2e-6], dtype=np.float64)
    Gl = np.full(M, 5e-10)
    return PhysicalInputs(
        case_id="synthetic_single", L=L, D=D, a=a, rho=rho, nu=1e-6,
        roughness=1.5e-5, TVD=0.0, Q0=Q0, t_s=1.0, t_c=t_c,
        xs=np.array([x]), Cd=[Cd], A_perf=[A], kappa=[kap],
        I_f=[If], R_f=[Rf], C_f=[Cf], G_l=[Gl],
        p_res=np.array([4.0e7]), K=[orifice_K(rho, Cd, A)], N=1, M=M,
    )


def synthetic_initial(phys: PhysicalInputs) -> InitialState:
    H0 = float(phys.p_res[0] / phys.rho_g + phys.z_of_x(phys.xs[0]))
    m = phys.Cd[0].size
    q0 = np.full(m, phys.Q0 / m)
    return InitialState(
        source="physical_steady_equal_split",
        p0_wh=H0 * phys.rho_g, Q0_wh=phys.Q0,
        H0_nodes=np.array([H0]), Qm0=np.array([phys.Q0]), Qp0=np.array([0.0]),
        q0=[q0], pc0=[np.full(m, float(phys.p_res[0]))], F0=[np.zeros(m)],
        note="synthetic hydrostatic + equal-split q",
    )


def torch_mini_loop(K, kappa, c1, c0, a1, a0, Q_valve, delay_L, delay_R, B,
                    z_j, rho_g, H0_wh, H0_c, q0, H_scale, Q_scale, p_scale,
                    require_ok=True):
    """Differentiable 1-cluster 2-segment unroll. K may require grad."""
    nT = int(Q_valve.shape[0])
    H_wh = [H0_wh]
    H_c = [H0_c]
    Qm = [torch.zeros((), dtype=torch.float64)]
    Qp = [torch.zeros((), dtype=torch.float64)]
    H_toe = [H0_c]
    q_state = q0
    n_hard = 0
    resid_max = torch.zeros((), dtype=torch.float64)
    for n in range(1, nT):
        CM_wh = interp_list([H_c[k] - B * Qm[k] for k in range(n)], n, delay_L, H_c[0] - B * Qm[0])
        Hw = CM_wh + B * Q_valve[n]
        CP = interp_list([H_wh[k] + B * Q_valve[k] for k in range(n)], n, delay_L, H_wh[0] + B * Q_valve[0])
        CM = interp_list(H_toe, n, delay_R, H_toe[0])
        H, Qmm, Qpp, q, stat = cluster_newton(
            CP, CM, B, B,
            torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64),
            z_j, rho_g, K, kappa, 1e-8, c1, c0, a1, a0,
            H_c[-1], q_state, H_scale, Q_scale, p_scale,
        )
        n_hard += 1
        if require_ok and float(stat[1]) < 0.5:
            raise RuntimeError(f"Newton failed in torch mini-loop n={n} resid={float(stat[2])}")
        resid_max = torch.maximum(resid_max, stat[2])
        H_wh.append(Hw)
        H_c.append(H)
        Qm.append(Qmm)
        Qp.append(Qpp)
        CPt = interp_list([H_c[k] + B * Qp[k] for k in range(len(H_c))], n, delay_R, H_c[0])
        H_toe.append(CPt)
        q_state = q
    return {
        "H_wh": torch.stack(H_wh),
        "p_wh": torch.stack(H_wh) * rho_g,
        "H_c": torch.stack(H_c),
        "n_hard": n_hard,
        "resid_max": resid_max,
    }


def loop_from_bundle(bundle, n_steps=None, require_ok=True) -> LoopResult:
    q = bundle.query
    n = n_steps or q.t.size
    loop = CharacteristicLoop(bundle.physical, q.dt, n)
    Qv = cosine_valve_Q(q.t[:n], bundle.physical.Q0, bundle.physical.t_s, bundle.physical.t_c)
    return loop.rollout(Qv, bundle.initial, require_ok=require_ok)
