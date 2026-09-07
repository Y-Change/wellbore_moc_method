# -*- coding: utf-8 -*-
"""Contract-compliant adapter.

predict() inputs: physical_inputs + initial_state + query_grid.
targets are loaded separately and must not be passed into predict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt

from paths import ROUND1_MANIFEST

G = 9.80665
N_MAX = 12
M_MAX = 16
T_S_GENERATOR = 1.0


def _j(arr) -> dict:
    return json.loads(str(arr))


def orifice_K(rho, Cd, A):
    Cd = np.asarray(Cd, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    den = 2.0 * Cd ** 2 * A ** 2
    return np.divide(rho, den, out=np.full(den.shape, 1.0e30), where=den > 0.0)


def antialias_resample(t, y, t_q, cutoff_frac=0.4):
    """Low-pass then linear sample onto t_q. Does not invent node truth."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t_q = np.asarray(t_q, dtype=np.float64)
    if t.size < 8:
        return np.interp(t_q, t, y)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    dt_q = float(np.median(np.diff(t_q))) if t_q.size > 1 else dt
    fs_q = 1.0 / max(dt_q, 1e-12)
    fc = cutoff_frac * 0.5 * fs_q
    wn = min(0.99, max(1e-4, fc / (0.5 * fs)))
    b, a = butter(4, wn, btype="low")
    yf = filtfilt(b, a, y)
    return np.interp(t_q, t, yf)


def cosine_valve_Q(t, Q0, t_s, t_c):
    t = np.asarray(t, dtype=np.float64)
    Q = np.full_like(t, Q0)
    Q[t >= t_s + t_c] = 0.0
    mid = (t >= t_s) & (t < t_s + t_c)
    if t_c > 0:
        xi = (t[mid] - t_s) / t_c
        Q[mid] = 0.5 * Q0 * (1.0 + np.cos(np.pi * xi))
    else:
        Q[t >= t_s] = 0.0
    return Q


@dataclass
class PhysicalInputs:
    case_id: str
    L: float
    D: float
    a: float
    rho: float
    nu: float
    roughness: float
    TVD: float
    Q0: float
    t_s: float
    t_c: float
    xs: np.ndarray
    Cd: List[np.ndarray]
    A_perf: List[np.ndarray]
    kappa: List[np.ndarray]
    I_f: List[np.ndarray]
    R_f: List[np.ndarray]
    C_f: List[np.ndarray]
    G_l: List[np.ndarray]
    p_res: np.ndarray
    K: List[np.ndarray]
    N: int
    M: int
    assumed_fixed: Dict = field(default_factory=lambda: {
        "toe_bc": "dead_end", "ramp": "cosine", "t_s_locked": T_S_GENERATOR,
        "friction_closed_loop": "darcy_foot",
        "memory": "node_time_recursive_only",
    })

    @property
    def A(self) -> float:
        return float(np.pi * self.D ** 2 / 4.0)

    @property
    def rho_g(self) -> float:
        return self.rho * G

    @property
    def period(self) -> float:
        return 4.0 * self.L / self.a

    @property
    def B(self) -> float:
        return self.a / (G * self.A)

    def z_of_x(self, x: float) -> float:
        return -min(float(x), self.TVD)


@dataclass
class InitialState:
    source: str
    p0_wh: float
    Q0_wh: float
    H0_nodes: np.ndarray
    Qm0: np.ndarray
    Qp0: np.ndarray
    q0: List[np.ndarray]
    pc0: List[np.ndarray]
    F0: List[np.ndarray]
    note: str


@dataclass
class QueryGrid:
    t: np.ndarray
    t_rel: np.ndarray
    tau: np.ndarray
    mask: np.ndarray
    dt: float
    nyquist_hz: float
    source_dt: float
    source_nyquist_hz: float
    t_s: float
    t_c: float
    t_valve_end: float
    valve: np.ndarray
    free_response: np.ndarray
    protocol: Dict


@dataclass
class Targets:
    p_head: np.ndarray
    Q_head: np.ndarray
    p_pert: np.ndarray
    Q_pert: np.ndarray
    node_t: np.ndarray
    node_H: np.ndarray
    node_Qm: np.ndarray
    node_Qp: np.ndarray
    node_sq: np.ndarray
    node_pc: np.ndarray
    node_mask_t: np.ndarray
    silent_p: bool
    silent_Q: bool
    H0_nodes: np.ndarray


@dataclass
class CaseBundle:
    physical: PhysicalInputs
    initial: InitialState
    query: QueryGrid
    targets: Targets
    meta: dict
    params: dict
    source_file: str
    source_sha256: str


def load_round1_manifest(path: Optional[Path] = None) -> dict:
    p = path or ROUND1_MANIFEST
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_physical(params: dict) -> PhysicalInputs:
    w = params["well"]
    clusters = params["clusters"]
    N = int(params["N"])
    rho = float(w["rho"])
    Cd, A, kap, If, Rf, Cf, Gl, Ks, pres, xs = [], [], [], [], [], [], [], [], [], []
    for cl in clusters[:N]:
        cd = np.asarray(cl["Cd"], dtype=np.float64)
        ap = np.asarray(cl["A_perf"], dtype=np.float64)
        Cd.append(cd)
        A.append(ap)
        kap.append(np.asarray(cl["kappa"], dtype=np.float64))
        If.append(np.asarray(cl["I_f"], dtype=np.float64))
        Rf.append(np.asarray(cl["R_f"], dtype=np.float64))
        Cf.append(np.asarray(cl["C_f"], dtype=np.float64))
        Gl.append(np.asarray(cl["G_l"], dtype=np.float64))
        Ks.append(orifice_K(rho, cd, ap))
        pres.append(float(cl["p_res"]))
        xs.append(float(cl["x"]))
    M = int(Cd[0].size) if Cd else 0
    return PhysicalInputs(
        case_id="", L=float(w["L"]), D=float(w["D"]), a=float(w["a"]), rho=rho,
        nu=float(w["nu"]), roughness=float(w["roughness"]),
        TVD=float(params.get("TVD", w.get("TVD", 0.0))),
        Q0=float(w["Q0"]), t_s=T_S_GENERATOR, t_c=float(w["t_c"]),
        xs=np.asarray(xs, dtype=np.float64), Cd=Cd, A_perf=A, kappa=kap,
        I_f=If, R_f=Rf, C_f=Cf, G_l=Gl, p_res=np.asarray(pres, dtype=np.float64),
        K=Ks, N=N, M=M,
    )


def _initial_from_t0(phys: PhysicalInputs, z, node_t) -> InitialState:
    """t=0 observation. Per-perf q/pc cannot be recovered from mean/sum → equal split."""
    p0 = float(np.asarray(z["p_head"])[0])
    Q0 = float(np.asarray(z["Q_head"])[0])
    nt = np.asarray(node_t, dtype=np.float64)
    i0 = int(np.argmin(np.abs(nt - 0.0))) if nt.size else 0
    H0 = np.asarray(z["node_H"][i0], dtype=np.float64)[:phys.N]
    Qm0 = np.asarray(z["node_Qm"][i0], dtype=np.float64)[:phys.N]
    Qp0 = np.asarray(z["node_Qp"][i0], dtype=np.float64)[:phys.N]
    sq = np.asarray(z["node_sq"][i0], dtype=np.float64)[:phys.N]
    pc = np.asarray(z["node_pc"][i0], dtype=np.float64)[:phys.N]
    q0, pc0, F0 = [], [], []
    for j in range(phys.N):
        m = phys.Cd[j].size
        q0.append(np.full(m, sq[j] / max(m, 1)))
        pc0.append(np.full(m, pc[j]))
        F0.append(np.zeros(m))
    return InitialState(
        source="t0_observation", p0_wh=p0, Q0_wh=Q0, H0_nodes=H0, Qm0=Qm0, Qp0=Qp0,
        q0=q0, pc0=pc0, F0=F0,
        note="per-perf q/pc equal-split of cluster sum/mean; not a heterogeneous reconstruction",
    )


def make_query(phys: PhysicalInputs, dt_q: float, n: int, source_dt: float,
               cutoff_frac=0.4) -> QueryGrid:
    t_rel = dt_q * np.arange(n, dtype=np.float64)
    t = phys.t_s + t_rel
    tau = t_rel / max(phys.period, 1e-12)
    valve = (t >= phys.t_s) & (t < phys.t_s + phys.t_c)
    free = t >= phys.t_s + phys.t_c
    return QueryGrid(
        t=t, t_rel=t_rel, tau=tau, mask=np.ones(n, dtype=np.float64),
        dt=dt_q, nyquist_hz=0.5 / dt_q, source_dt=source_dt,
        source_nyquist_hz=0.5 / source_dt if source_dt > 0 else float("nan"),
        t_s=phys.t_s, t_c=phys.t_c, t_valve_end=phys.t_s + phys.t_c,
        valve=valve.astype(np.float64), free_response=free.astype(np.float64),
        protocol={"dt_s": dt_q, "n_samples": n, "t_origin": "valve_start",
                  "antialias_frac_nyquist": cutoff_frac,
                  "coord": "t_rel = t - t_s and tau = t_rel / (4L/a)"},
    )


class Round2Dataset:
    def __init__(self, split: str, cfg: dict, manifest: Optional[dict] = None,
                 case_ids: Optional[List[str]] = None):
        if split not in ("train", "val", "all_usable"):
            raise ValueError(f"split {split} blocked this round (test is viewed holdout)")
        self.split = split
        self.cfg = cfg
        self.manifest = manifest or load_round1_manifest()
        rows = [c for c in self.manifest["cases"]
                if split == "all_usable" or c["split"] == split]
        if case_ids is not None:
            want = set(case_ids)
            rows = [c for c in rows if c["case_id"] in want]
            rows.sort(key=lambda c: case_ids.index(c["case_id"]))
        if not rows:
            raise RuntimeError(f"no cases in split={split}")
        self.rows = rows
        self.qcfg = cfg["query"]
        self._cache: Dict[int, CaseBundle] = {}

    def __len__(self):
        return len(self.rows)

    def ids(self) -> List[str]:
        return [r["case_id"] for r in self.rows]

    def _load(self, row: dict) -> CaseBundle:
        z = np.load(row["source_file"], allow_pickle=True)
        params = _j(z["params_json"])
        meta = _j(z["meta_json"])
        phys = _parse_physical(params)
        phys.case_id = row["case_id"]
        t_src = np.asarray(z["t"], dtype=np.float64)
        p_src = np.asarray(z["p_head"], dtype=np.float64)
        Q_src = np.asarray(z["Q_head"], dtype=np.float64)
        source_dt = float(np.median(np.diff(t_src))) if t_src.size > 1 else float(meta["dt"])
        query = make_query(phys, float(self.qcfg["dt_s"]), int(self.qcfg["n_samples"]),
                           source_dt, float(self.qcfg.get("antialias_frac_nyquist", 0.4)))
        p = antialias_resample(t_src, p_src, query.t, float(self.qcfg.get("antialias_frac_nyquist", 0.4)))
        Q = antialias_resample(t_src, Q_src, query.t, float(self.qcfg.get("antialias_frac_nyquist", 0.4)))
        init = _initial_from_t0(phys, z, z["node_t"])
        nt = np.asarray(z["node_t"], dtype=np.float64)
        m = (nt >= query.t[0]) & (nt <= query.t[-1])
        nti = nt[m]
        def take(name):
            a = np.asarray(z[name], dtype=np.float64)[m]
            out = np.zeros((nti.size, N_MAX), dtype=np.float64)
            out[:, :phys.N] = a[:, :phys.N]
            return out
        from metrics import is_silent_pert
        p_pert = p - init.p0_wh
        Q_pert = Q - init.Q0_wh
        targets = Targets(
            p_head=p, Q_head=Q, p_pert=p_pert, Q_pert=Q_pert,
            node_t=nti, node_H=take("node_H"), node_Qm=take("node_Qm"),
            node_Qp=take("node_Qp"), node_sq=take("node_sq"), node_pc=take("node_pc"),
            node_mask_t=np.ones(nti.size, dtype=np.float64),
            silent_p=is_silent_pert(p_pert, p), silent_Q=is_silent_pert(Q_pert, Q),
            H0_nodes=init.H0_nodes,
        )
        return CaseBundle(physical=phys, initial=init, query=query, targets=targets,
                          meta=meta, params=params, source_file=row["source_file"],
                          source_sha256=row["source_sha256"])

    def __getitem__(self, i: int) -> CaseBundle:
        i = int(i)
        if i not in self._cache:
            self._cache[i] = self._load(self.rows[i])
        return self._cache[i]


def nested_order(manifest: dict, subset_seed: int) -> List[str]:
    train = [c["case_id"] for c in manifest["cases"] if c["split"] == "train"]
    rng = np.random.default_rng(int(subset_seed))
    perm = rng.permutation(len(train))
    return [train[int(k)] for k in perm]


def nested_ids(order: List[str], n) -> List[str]:
    if n == "all" or n is None:
        return list(order)
    return list(order[:int(n)])


def predict_arrays(bundle: CaseBundle) -> dict:
    """Arrays allowed into predict(). No future waveforms."""
    p = bundle.physical
    q = bundle.query
    ini = bundle.initial
    return {
        "case_id": p.case_id,
        "L": p.L, "D": p.D, "a": p.a, "rho": p.rho, "nu": p.nu,
        "roughness": p.roughness, "TVD": p.TVD, "Q0": p.Q0, "t_s": p.t_s, "t_c": p.t_c,
        "N": p.N, "M": p.M, "xs": p.xs, "p_res": p.p_res,
        "K": p.K, "Cd": p.Cd, "A_perf": p.A_perf, "kappa": p.kappa,
        "I_f": p.I_f, "R_f": p.R_f, "C_f": p.C_f, "G_l": p.G_l,
        "B": p.B, "A": p.A, "rho_g": p.rho_g, "period": p.period,
        "z_j": np.array([p.z_of_x(x) for x in p.xs], dtype=np.float64),
        "t": q.t, "t_rel": q.t_rel, "tau": q.tau, "mask": q.mask,
        "dt": q.dt, "t_valve_end": q.t_valve_end,
        "Q_valve": cosine_valve_Q(q.t, p.Q0, p.t_s, p.t_c),
        "p0_wh": ini.p0_wh, "Q0_wh": ini.Q0_wh,
        "H0_nodes": ini.H0_nodes, "Qm0": ini.Qm0, "Qp0": ini.Qp0,
        "q0": ini.q0, "pc0": ini.pc0, "F0": ini.F0,
        "init_source": ini.source,
    }


def pack_features(inp: dict, norm: dict) -> np.ndarray:
    """Fixed-length physical feature vector (no future labels)."""
    xs = np.zeros(N_MAX, dtype=np.float64)
    xs[:inp["N"]] = np.asarray(inp["xs"]) / max(inp["L"], 1.0)
    sumK = np.zeros(N_MAX, dtype=np.float64)
    sumCf = np.zeros(N_MAX, dtype=np.float64)
    for j, Kj in enumerate(inp["K"][:N_MAX]):
        sumK[j] = np.log10(np.clip(np.mean(Kj), 1.0, 1e40))
        sumCf[j] = np.log10(np.clip(np.mean(inp["C_f"][j]), 1e-16, 1.0))
    well = np.array([
        inp["L"] / 4000.0, inp["D"] / 0.1, inp["a"] / 1400.0, inp["rho"] / 1000.0,
        np.log10(max(inp["nu"], 1e-8)), inp["Q0"] / 0.05, inp["t_c"] / 1.0,
        inp["N"] / 12.0,
    ], dtype=np.float64)
    raw = np.concatenate([well, xs, sumK, sumCf])
    mu = np.asarray(norm["feat_mean"], dtype=np.float64)
    sd = np.asarray(norm["feat_std"], dtype=np.float64)
    return ((raw - mu) / sd).astype(np.float32)


def compute_norm(ds: Round2Dataset, max_cases: int = 256) -> dict:
    n = min(len(ds), max_cases)
    pstd, feats = [], []
    for i in range(n):
        b = ds[i]
        pstd.append(float(np.std(b.targets.p_pert)) + 1e-8)
        dummy_norm = {"feat_mean": np.zeros(8 + 3 * N_MAX), "feat_std": np.ones(8 + 3 * N_MAX)}
        feats.append(pack_features(predict_arrays(b), dummy_norm) * np.asarray(dummy_norm["feat_std"])
                     + np.asarray(dummy_norm["feat_mean"]))
    F = np.stack(feats, 0)
    return {
        "p_pert_scale": float(np.median(pstd)),
        "feat_mean": F.mean(0).tolist(),
        "feat_std": np.clip(F.std(0), 1e-8, None).tolist(),
        "n_cases_used": int(n),
        "case_ids": ds.ids()[:n],
        "silent_rel": 1e-6,
        "query": ds.qcfg,
    }


FORBIDDEN_PREDICT_KEYS = (
    "p_head", "Q_head", "p_pert", "Q_pert",
    "node_H", "node_Qm", "node_Qp", "node_pc", "node_sq",
)


def assert_predict_clean(inp: dict) -> None:
    leak = [k for k in FORBIDDEN_PREDICT_KEYS if k in inp]
    if leak:
        raise RuntimeError(f"predict inputs contain future/target keys: {leak}")


def collate_predict(bundles, norm, pad_extra=0):
    """Batch predict tensors. Padding is appended after the valid grid and masked."""
    feats, taus, masks, ids = [], [], [], []
    T0 = bundles[0].query.tau.size
    T = T0 + int(pad_extra)
    dt_tau = float(np.median(np.diff(bundles[0].query.tau))) if T0 > 1 else 0.0
    for b in bundles:
        inp = predict_arrays(b)
        assert_predict_clean(inp)
        if b.query.tau.size != T0:
            raise RuntimeError("query length mismatch; protocol requires a shared grid")
        tau = np.zeros(T, dtype=np.float64)
        mask = np.zeros(T, dtype=np.float64)
        tau[:T0] = b.query.tau
        mask[:T0] = 1.0
        if pad_extra:
            tau[T0:] = b.query.tau[-1] + dt_tau * np.arange(1, pad_extra + 1)
        feats.append(pack_features(inp, norm))
        taus.append(tau.astype(np.float32))
        masks.append(mask.astype(np.float32))
        ids.append(b.physical.case_id)
    return {
        "feat": np.stack(feats, 0),
        "tau": np.stack(taus, 0),
        "mask": np.stack(masks, 0),
        "case_ids": ids,
        "pad_extra": int(pad_extra),
    }


def collate_targets(bundles, pad_extra=0):
    T0 = bundles[0].query.tau.size
    T = T0 + int(pad_extra)
    p, q, silent = [], [], []
    for b in bundles:
        pp = np.zeros(T, dtype=np.float64)
        qq = np.zeros(T, dtype=np.float64)
        pp[:T0] = b.targets.p_pert
        qq[:T0] = b.targets.Q_pert
        p.append(pp)
        q.append(qq)
        silent.append(b.targets.silent_p)
    return {
        "p_pert": np.stack(p, 0),
        "Q_pert": np.stack(q, 0),
        "silent_p": np.asarray(silent, dtype=bool),
    }
