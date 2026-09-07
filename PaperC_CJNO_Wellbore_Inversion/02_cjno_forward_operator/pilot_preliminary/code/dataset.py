# -*- coding: utf-8 -*-
"""Read-only adapter: wellhead + saved node traces/memory. No full-field interpolation."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MANIFEST_DIR, CACHE_DIR, STAGE1_MANIFEST

G = 9.80665
N_MAX = 12
F_CLUSTER = 6  # x_norm, sum_CdA, C_f, G_l, R_f, I_f


def _j(arr) -> dict:
    return json.loads(str(arr))


@dataclass
class CaseSample:
    case_id: str
    split: str
    N: int
    well: np.ndarray          # [L, D, a, rho, nu, Q0, t_c, TVD]
    cluster: np.ndarray       # (N_MAX, F) padded
    mask: np.ndarray          # (N_MAX,)
    xs: np.ndarray            # (N,)
    t: np.ndarray             # wellhead time (possibly windowed/decimated)
    p_head: np.ndarray
    Q_head: np.ndarray
    p_pert: np.ndarray        # p - p[0]
    Q_pert: np.ndarray
    node_t: np.ndarray
    node_H: np.ndarray        # (Tn, N_MAX)
    node_Qm: np.ndarray
    node_Qp: np.ndarray
    node_sq: np.ndarray
    node_pc: np.ndarray
    node_z: np.ndarray        # (Tn, N_MAX, 2, Mz) padded
    p0: float
    Q0: float
    H0: float
    rho_g: float
    period: float
    dt: float
    dt_node: float
    nyquist_wh: float
    nyquist_node: float
    scales: Dict[str, float]
    meta: dict
    params: dict


def load_research_manifest(path: Optional[Path] = None) -> dict:
    p = path or (MANIFEST_DIR / "pilot_manifest.json")
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    params = _j(z["params_json"])
    meta = _j(z["meta_json"])
    return {"z": z, "params": params, "meta": meta}


def _cluster_table(params: dict, L: float) -> tuple:
    N = int(params["N"])
    feat = np.zeros((N_MAX, F_CLUSTER), dtype=np.float64)
    mask = np.zeros((N_MAX,), dtype=np.float64)
    xs = np.array(params["xs"], dtype=np.float64)
    for j, cl in enumerate(params["clusters"][:N_MAX]):
        eq = cl["eq"]
        feat[j] = [
            xs[j] / max(L, 1.0),
            float(eq["sum_CdA"]),
            float(eq["C_f"]),
            float(eq["G_l"]),
            float(eq["R_f"]),
            float(eq["I_f"]),
        ]
        mask[j] = 1.0
    return feat, mask, xs


def window_and_decimate(t, y, t0, t1, decim: int):
    m = (t >= t0) & (t <= t1)
    return t[m][::decim], y[m][::decim]


class PilotDataset:
    """Supervision = saved wellhead + node traces. Never treats node interpolation as MOC field."""

    def __init__(self, split: str, manifest: Optional[dict] = None,
                 window_periods: float = 4.0, wellhead_decim: int = 4,
                 node_max_len: int = 256, after_shut_in: bool = True):
        if split not in ("train", "val", "test", "all_usable"):
            raise ValueError(split)
        self.split = split
        self.manifest = manifest or load_research_manifest()
        self.window_periods = float(window_periods)
        self.wellhead_decim = int(wellhead_decim)
        self.node_max_len = int(node_max_len)
        self.after_shut_in = after_shut_in
        self.rows = [c for c in self.manifest["cases"]
                     if split == "all_usable" or c["split"] == split]
        if not self.rows:
            raise RuntimeError(f"no cases in split={split}; run audit_pilot.py first")
        self.norm = None
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def ids(self) -> List[str]:
        return [r["case_id"] for r in self.rows]

    def _load_row(self, row: dict) -> CaseSample:
        blob = _parse_npz(Path(row["source_file"]))
        z, params, meta = blob["z"], blob["params"], blob["meta"]
        t = np.asarray(z["t"], dtype=np.float64)
        p = np.asarray(z["p_head"], dtype=np.float64)
        Q = np.asarray(z["Q_head"], dtype=np.float64)
        nt = np.asarray(z["node_t"], dtype=np.float64)
        period = float(meta["period_4L_a"])
        t_s = 1.0
        t0 = t_s if self.after_shut_in else float(t[0])
        t1 = t0 + self.window_periods * period
        tw, pw = window_and_decimate(t, p, t0, t1, self.wellhead_decim)
        _, Qw = window_and_decimate(t, Q, t0, t1, self.wellhead_decim)
        nm = (nt >= t0) & (nt <= t1)
        nti = nt[nm]
        if nti.size > self.node_max_len:
            idx = np.linspace(0, nti.size - 1, self.node_max_len).astype(int)
        else:
            idx = np.arange(nti.size)
        nti = nti[idx]
        def pad_node(a):
            a = np.asarray(a, dtype=np.float64)[nm][idx]
            out = np.zeros((a.shape[0], N_MAX), dtype=np.float64)
            ncl = a.shape[1]
            out[:, :ncl] = a
            return out
        nH = pad_node(z["node_H"])
        nQm = pad_node(z["node_Qm"])
        nQp = pad_node(z["node_Qp"])
        nsq = pad_node(z["node_sq"])
        npc = pad_node(z["node_pc"])
        N = int(params["N"])
        Mz = int(meta.get("kernel_M", 12))
        nz = np.asarray(z["node_z"], dtype=np.float64)[nm][idx]
        # (Tn, N*2*Mz) → (Tn, N_MAX, 2, Mz)
        nz_p = np.zeros((nz.shape[0], N_MAX, 2, Mz), dtype=np.float64)
        if nz.size:
            raw = nz.reshape(nz.shape[0], N, 2, Mz)
            nz_p[:, :N] = raw
        L = float(params["well"]["L"])
        feat, mask, xs = _cluster_table(params, L)
        well = np.array([
            L, float(params["well"]["D"]), float(params["well"]["a"]),
            float(params["well"]["rho"]), float(params["well"]["nu"]),
            float(params["well"]["Q0"]), float(params["well"]["t_c"]),
            float(params.get("TVD", 0.0)),
        ], dtype=np.float64)
        rho_g = float(meta.get("rho_g", params["well"]["rho"] * G))
        p0 = float(p[0])
        Q0 = float(Q[0])
        dt = float(np.median(np.diff(tw))) if tw.size > 1 else float(meta["dt"]) * self.wellhead_decim
        dt_n = float(np.median(np.diff(nti))) if nti.size > 1 else float("nan")
        return CaseSample(
            case_id=row["case_id"], split=row["split"], N=N, well=well, cluster=feat, mask=mask,
            xs=xs, t=tw, p_head=pw, Q_head=Qw, p_pert=pw - p0, Q_pert=Qw - Q0,
            node_t=nti, node_H=nH, node_Qm=nQm, node_Qp=nQp, node_sq=nsq, node_pc=npc,
            node_z=nz_p, p0=p0, Q0=Q0, H0=p0 / rho_g, rho_g=rho_g, period=period,
            dt=dt, dt_node=dt_n,
            nyquist_wh=0.5 / dt if dt > 0 else float("nan"),
            nyquist_node=0.5 / dt_n if dt_n > 0 else float("nan"),
            scales={
                "H": float(meta.get("H_scale", 1.0)),
                "Q": float(meta.get("Q_scale", 1.0)),
                "p": float(meta.get("p_scale", rho_g)),
            },
            meta=meta, params=params,
        )

    def __getitem__(self, i: int) -> CaseSample:
        i = int(i)
        if i not in self._cache:
            self._cache[i] = self._load_row(self.rows[i])
        return self._cache[i]

    def subset(self, n: int, seed: int = 42) -> "PilotDataset":
        rng = np.random.default_rng(seed)
        if n >= len(self.rows):
            return self
        pick = rng.choice(len(self.rows), size=n, replace=False)
        out = PilotDataset.__new__(PilotDataset)
        out.split = self.split
        out.manifest = self.manifest
        out.window_periods = self.window_periods
        out.wellhead_decim = self.wellhead_decim
        out.node_max_len = self.node_max_len
        out.after_shut_in = self.after_shut_in
        out.rows = [self.rows[int(k)] for k in pick]
        out.norm = self.norm
        out._cache = {}
        return out

    def compute_train_norm(self, max_cases: int = 256, seed: int = 20260906) -> dict:
        """Train-split only. Never peek val/test."""
        if self.split != "train":
            raise RuntimeError("normalization statistics must be computed on train")
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self.rows))
        if len(idx) > max_cases:
            idx = rng.choice(idx, size=max_cases, replace=False)
        pstd, qstd, hstd = [], [], []
        well = []
        cl = []
        for i in idx:
            s = self[int(i)]
            pstd.append(float(np.std(s.p_pert)) + 1e-8)
            qstd.append(float(np.std(s.Q_pert)) + 1e-8)
            hstd.append(float(np.std(s.node_H[:, :s.N] - s.H0)) + 1e-8)
            well.append(s.well)
            cl.append(s.cluster[s.mask > 0])
        W = np.stack(well, 0)
        C = np.concatenate(cl, 0) if cl else np.zeros((1, F_CLUSTER))
        # zero-signal rule: if std < 1e-8 * typical, mark channel silent (do not inflate epsilon)
        self.norm = {
            "p_pert_scale": float(np.median(pstd)),
            "Q_pert_scale": float(np.median(qstd)),
            "H_pert_scale": float(np.median(hstd)),
            "well_mean": W.mean(0).tolist(),
            "well_std": np.clip(W.std(0), 1e-12, None).tolist(),
            "cluster_mean": C.mean(0).tolist(),
            "cluster_std": np.clip(C.std(0), 1e-12, None).tolist(),
            "zero_signal_rule": "relative L2 uses ||u||_2 in the denominator; if ||u_pert||_2 / ||u||_2 < 1e-6 the channel is reported as silent, not divided by a larger epsilon",
            "n_cases_used": int(len(idx)),
            "split": "train",
            "window_periods": self.window_periods,
            "wellhead_decim": self.wellhead_decim,
        }
        return self.norm


def collate_batch(samples: List[CaseSample], norm: dict) -> dict:
    """Pad variable-length wellhead to max T in batch. Nodes already padded in N."""
    T = max(s.t.size for s in samples)
    Tn = max(s.node_t.size for s in samples)
    B = len(samples)
    def zeros(*shp):
        return np.zeros(shp, dtype=np.float32)

    batch = {
        "case_id": [s.case_id for s in samples],
        "N": np.array([s.N for s in samples], dtype=np.int64),
        "mask": np.stack([s.mask for s in samples]).astype(np.float32),
        "cluster": np.stack([s.cluster for s in samples]).astype(np.float32),
        "well": np.stack([s.well for s in samples]).astype(np.float32),
        "t": zeros(B, T), "p_pert": zeros(B, T), "Q_pert": zeros(B, T),
        "p_head": zeros(B, T), "wh_mask": zeros(B, T),
        "node_t": zeros(B, Tn),
        "node_H": zeros(B, Tn, N_MAX), "node_Qm": zeros(B, Tn, N_MAX),
        "node_Qp": zeros(B, Tn, N_MAX), "node_sq": zeros(B, Tn, N_MAX),
        "node_pc": zeros(B, Tn, N_MAX),
        "node_mask_t": zeros(B, Tn),
        "p0": np.array([s.p0 for s in samples], dtype=np.float32),
        "H0": np.array([s.H0 for s in samples], dtype=np.float32),
        "Q0": np.array([s.Q0 for s in samples], dtype=np.float32),
        "rho_g": np.array([s.rho_g for s in samples], dtype=np.float32),
        "period": np.array([s.period for s in samples], dtype=np.float32),
        "L": np.array([s.well[0] for s in samples], dtype=np.float32),
        "a": np.array([s.well[2] for s in samples], dtype=np.float32),
    }
    wm = np.array(norm["well_mean"], dtype=np.float32)
    ws = np.array(norm["well_std"], dtype=np.float32)
    cm = np.array(norm["cluster_mean"], dtype=np.float32)
    cs = np.array(norm["cluster_std"], dtype=np.float32)
    batch["well_n"] = ((batch["well"] - wm) / ws).astype(np.float32)
    batch["cluster_n"] = ((batch["cluster"] - cm) / cs).astype(np.float32) * batch["mask"][..., None]
    for i, s in enumerate(samples):
        n = s.t.size
        batch["t"][i, :n] = (s.t - s.t[0]) / max(s.period, 1e-6)
        batch["p_pert"][i, :n] = s.p_pert
        batch["Q_pert"][i, :n] = s.Q_pert
        batch["p_head"][i, :n] = s.p_head
        batch["wh_mask"][i, :n] = 1.0
        m = s.node_t.size
        batch["node_t"][i, :m] = (s.node_t - s.t[0]) / max(s.period, 1e-6) if s.t.size else s.node_t
        batch["node_H"][i, :m] = s.node_H[:m]
        batch["node_Qm"][i, :m] = s.node_Qm[:m]
        batch["node_Qp"][i, :m] = s.node_Qp[:m]
        batch["node_sq"][i, :m] = s.node_sq[:m]
        batch["node_pc"][i, :m] = s.node_pc[:m]
        batch["node_mask_t"][i, :m] = 1.0
    batch["p_pert_scale"] = float(norm["p_pert_scale"])
    batch["Q_pert_scale"] = float(norm["Q_pert_scale"])
    batch["H_pert_scale"] = float(norm["H_pert_scale"])
    return batch
