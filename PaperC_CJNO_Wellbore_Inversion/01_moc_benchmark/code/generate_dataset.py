# -*- coding: utf-8 -*-
"""Pilot / formal dataset generator (攻关执行方案_v2 §3.1).

    python generate_dataset.py --pilot                 # 2000 maximin-LHS
    python generate_dataset.py --pilot --n 20 --smoke  # coverage smoke
    python generate_dataset.py --formal                # refused unless --smoke or --confirm-30k
    python generate_dataset.py --formal --smoke        # tiny Sobol + 8 OOD types, simulate
    python generate_dataset.py --formal --design-only  # full 20k/3k/3k/4k designs, no MOC
    python generate_dataset.py --formal --confirm-30k  # full MOC (explicit; do not launch casually)

Formal Sobol expansion is *blocked* until goldens/metrics.json has no FAIL
and the Newton pilot gate is PASS.  Failures are kept, never deleted (R4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))

import goldens
import moc_solver as ms
import observation as obsmod
from audit import RunAudit, dump_json, sha256_file
from config_io import load_yaml, validate_against_schema
from paths import GOLDEN_DIR, MANIFEST_DIR, OOD_DIR, PILOT_DIR, SOBOL_DIR, ensure_dirs

G = 9.80665

TRAIN_KEYS = [
    "L", "D", "a", "rho", "nu", "roughness", "TVD_frac", "Q0", "t_c",
    "N_clusters", "spacing", "spacing_jitter_frac", "L_toe", "perfs_per_cluster",
    "Cd", "d_perf", "perf_heterogeneity_cv", "kappa",
    "I_f_cluster", "R_f_cluster", "C_f_cluster", "G_l_cluster",
    "cluster_heterogeneity_cv", "p_res_grad",
]

OOD_TYPES = [
    "dense_spacing", "sparse_spacing", "many_clusters",
    "tail_viscosity", "tail_compliance", "uncooccurring",
    "deleted_cluster", "impedance_jump",
]


def _transform(u: float, spec: dict):
    lo, hi = spec["low"], spec["high"]
    dist = spec["dist"]
    if dist == "uniform":
        return lo + u * (hi - lo)
    if dist == "loguniform":
        return 10 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo)))
    if dist == "int_uniform":
        return int(lo + u * (hi - lo + 1 - 1e-12))
    raise ValueError(dist)


def _loguniform_draw(rng: np.random.Generator, lo: float, hi: float) -> float:
    return 10 ** float(rng.uniform(math.log10(lo), math.log10(hi)))


def maximin_lhs(n: int, d: int, seed: int, n_candidates: int) -> np.ndarray:
    best, best_sep = None, -1.0
    for k in range(n_candidates):
        samp = qmc.LatinHypercube(d=d, seed=seed + 17 * k, optimization="random-cd").random(n)
        sep = float("inf")
        for i in range(0, n, 256):
            a = samp[i:i + 256]
            d2 = np.min((a[:, None, :] - samp[None, :, :]) ** 2, axis=2)
            np.fill_diagonal(d2[:a.shape[0], i:i + a.shape[0]], np.inf)
            for r in range(a.shape[0]):
                d2[r, i + r] = np.inf
            sep = min(sep, float(np.sqrt(d2.min())))
        if sep > best_sep:
            best, best_sep = samp, sep
    return best


def sobol_owen(n: int, d: int, seed: int) -> np.ndarray:
    """Owen-scrambled Sobol; generate the next power-of-two block and slice."""
    m = max(1, int(math.ceil(math.log2(max(n, 2)))))
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    return np.asarray(engine.random_base2(m)[:n], dtype=float)


def iter_unit_rows(n: int, d: int, seed: int, extra_factor: int = 8) -> Iterator[np.ndarray]:
    primary = sobol_owen(n, d, seed)
    extra = sobol_owen(max(n, extra_factor * n), d, seed + 1009)
    for row in list(primary) + list(extra):
        yield np.asarray(row, dtype=float)


def make_group_id(c: dict) -> str:
    """井 × 簇布局 × 激励事件；同一 case 的全部时间窗必须共享该 id（R8）。"""
    w = c["well"]
    well_part = f"{w['L']:.2f}|{w['D']:.5f}|{w['a']:.2f}|{w['rho']:.2f}|{w['nu']:.6e}"
    layout_part = f"{c['N']}|" + ",".join(f"{float(x):.3f}" for x in c["xs"])
    exc_part = f"{w['Q0']:.5f}|{w['t_c']:.5f}"
    raw = f"{well_part}#{layout_part}#{exc_part}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _ood_override_params(p: dict, ood_type: str, rng: np.random.Generator, domain: dict) -> dict:
    ood = domain["ood"]
    p = dict(p)
    if ood_type == "dense_spacing":
        spec = ood["dense_spacing"]["spacing"]
        p["spacing"] = float(spec["low"] + rng.random() * (spec["high"] - spec["low"]))
        p["spacing_jitter_frac"] = min(float(p["spacing_jitter_frac"]), 0.05)
    elif ood_type == "sparse_spacing":
        spec = ood["sparse_spacing"]["spacing"]
        p["spacing"] = float(spec["low"] + rng.random() * (spec["high"] - spec["low"]))
    elif ood_type == "many_clusters":
        spec = ood["many_clusters"]["N_clusters"]
        p["N_clusters"] = int(spec["low"] + rng.integers(0, int(spec["high"] - spec["low"] + 1)))
    elif ood_type == "tail_viscosity":
        spec = ood["tail_viscosity"]["nu"]
        p["nu"] = _loguniform_draw(rng, spec["low"], spec["high"])
    elif ood_type == "tail_compliance":
        spec = ood["tail_compliance"]["C_f_cluster"]
        p["C_f_cluster"] = _loguniform_draw(rng, spec["low"], spec["high"])
    elif ood_type == "uncooccurring":
        # 训练域边缘联合：高 Q0 × 密簇下沿 × 高 I_f（方案：训练域内未共现组合）
        tr = domain["train"]
        p["Q0"] = 0.85 * tr["Q0"]["high"] + 0.15 * tr["Q0"]["low"]
        p["Q0"] = float(p["Q0"] + 0.15 * (tr["Q0"]["high"] - p["Q0"]) * rng.random())
        p["spacing"] = float(tr["spacing"]["low"] + rng.random() * 2.0)  # 12–14 m
        p["I_f_cluster"] = _loguniform_draw(rng, 0.3 * tr["I_f_cluster"]["high"], tr["I_f_cluster"]["high"])
    elif ood_type in ("deleted_cluster", "impedance_jump"):
        pass
    else:
        raise ValueError(ood_type)
    return p


def _finish_ood_clusters(c: dict, ood_type: Optional[str], rng: np.random.Generator) -> dict:
    if not ood_type:
        c["ood_type"] = None
        return c
    clusters = c["clusters"]
    if ood_type == "deleted_cluster":
        j = int(rng.integers(0, len(clusters)))
        clusters[j]["Cd"] = np.zeros_like(np.asarray(clusters[j]["Cd"], dtype=float))
        clusters[j]["eq"]["sum_CdA"] = 0.0
        c["deleted_index"] = j
    elif ood_type == "impedance_jump":
        factor = 3.0 if rng.random() < 0.5 else (1.0 / 3.0)
        cl = clusters[0]
        cl["Cd"] = np.clip(np.asarray(cl["Cd"], dtype=float) * factor, 0.1, 2.5)
        cl["eq"]["sum_CdA"] = float(np.sum(cl["Cd"] * np.asarray(cl["A_perf"], dtype=float)))
        c["impedance_jump_factor"] = float(factor)
    c["ood_type"] = ood_type
    return c


def draw_case(u: np.ndarray, rng: np.random.Generator, domain: dict,
              ood_type: Optional[str] = None) -> Optional[dict]:
    spec = domain["train"]
    p = {k: _transform(float(u[i]), spec[k]) for i, k in enumerate(TRAIN_KEYS)}
    if ood_type:
        p = _ood_override_params(p, ood_type, rng, domain)
    N = int(p["N_clusters"])
    M = int(p["perfs_per_cluster"])
    s = float(p["spacing"])
    jit = float(p["spacing_jitter_frac"])
    spaces = s * (1.0 + jit * (2.0 * rng.random(max(N - 1, 0)) - 1.0))
    if N == 1:
        span = 0.0
    else:
        span = float(spaces.sum())
    x1 = p["L"] - p["L_toe"] - span
    if x1 <= 50.0:
        return None
    xs = [x1]
    for ds in spaces:
        xs.append(xs[-1] + float(ds))
    if xs[-1] >= p["L"] - 5.0:
        return None
    if N > 1 and min(np.diff(xs)) < domain["rejection_rules"]["min_cluster_spacing"] - 1e-9:
        return None
    tvd = float(p["TVD_frac"]) * max(x1 - 200.0, 100.0)
    tvd = min(tvd, x1 - 50.0)
    if tvd <= 0:
        return None
    A_perf = math.pi * p["d_perf"] ** 2 / 4.0
    cv_p = float(p["perf_heterogeneity_cv"])
    cv_c = float(p["cluster_heterogeneity_cv"])
    clusters = []
    for x in xs:
        scale_c = float(np.exp(rng.normal(0.0, cv_c))) if cv_c > 0 else 1.0
        Cd = p["Cd"] * (1.0 + cv_p * rng.normal(0.0, 1.0, size=M))
        Cd = np.clip(Cd, 0.4, 1.1)
        A = np.full(M, A_perf)
        I_f = np.full(M, p["I_f_cluster"] * M * scale_c)
        R_f = np.full(M, p["R_f_cluster"] * M * scale_c)
        C_f = np.full(M, p["C_f_cluster"] / M * scale_c)
        G_l = np.full(M, p["G_l_cluster"] / M * scale_c)
        kappa = np.full(M, p["kappa"])
        p_res = p["p_res_grad"] * tvd
        clusters.append({
            "x": float(x), "p_res": float(p_res),
            "Cd": Cd, "A_perf": A, "kappa": kappa, "I_f": I_f, "R_f": R_f, "C_f": C_f, "G_l": G_l,
            "eq": {"sum_CdA": float(np.sum(Cd * A)), "C_f": float(C_f.sum()), "G_l": float(G_l.sum()),
                   "R_f": float(1.0 / np.sum(1.0 / R_f)), "I_f": float(1.0 / np.sum(1.0 / I_f))},
        })
    c = {
        "well": {k: float(p[k]) for k in ("L", "D", "a", "rho", "nu", "roughness", "Q0", "t_c")},
        "TVD": float(tvd), "N": N, "M": M, "spacing": s, "xs": xs, "clusters": clusters,
        "draw": {k: (int(p[k]) if k in ("N_clusters", "perfs_per_cluster") else float(p[k]))
                 for k in TRAIN_KEYS},
    }
    return _finish_ood_clusters(c, ood_type, rng)


def draw_n_cases(n: int, seed: int, domain: dict, ood_type: Optional[str] = None) -> List[dict]:
    rng = np.random.default_rng(seed)
    drawn: List[dict] = []
    n_try = 0
    for row in iter_unit_rows(n, len(TRAIN_KEYS), seed):
        n_try += 1
        c = draw_case(row, rng, domain, ood_type=ood_type)
        if c is None:
            continue
        drawn.append(c)
        if len(drawn) >= n:
            break
    if len(drawn) < n:
        raise RuntimeError(f"only accepted {len(drawn)}/{n} draws (ood={ood_type}, tried={n_try})")
    return drawn


def assign_group_splits(cases: List[dict], n_train: int, n_val: int, n_id_test: int,
                        seed: int) -> Tuple[List[str], dict]:
    groups: Dict[str, List[int]] = defaultdict(list)
    gids_ordered = []
    for i, c in enumerate(cases):
        gid = make_group_id(c)
        c["group_id"] = gid
        if gid not in groups:
            gids_ordered.append(gid)
        groups[gid].append(i)
    rng = np.random.default_rng(seed)
    gids = list(gids_ordered)
    rng.shuffle(gids)
    splits = [""] * len(cases)
    counts = {"train": 0, "val": 0, "id_test": 0}
    targets = [("train", n_train), ("val", n_val), ("id_test", n_id_test)]
    ti = 0
    leftovers: List[int] = []
    for g in gids:
        idxs = groups[g]
        name, tgt = targets[ti]
        if counts[name] >= tgt and ti < len(targets) - 1:
            ti += 1
            name, tgt = targets[ti]
        if counts[name] < tgt:
            for i in idxs:
                splits[i] = name
            counts[name] += len(idxs)
        else:
            leftovers.extend(idxs)
    for i in leftovers:
        splits[i] = "train"
        counts["train"] += 1
    return splits, counts


def split_leakage(rows: List[dict]) -> List[dict]:
    seen: Dict[str, str] = {}
    leaks = []
    for r in rows:
        g, s = r["group_id"], r["split"]
        if g in seen and seen[g] != s:
            leaks.append({"group_id": g, "split_a": seen[g], "split_b": s})
        else:
            seen[g] = s
    return leaks


def case_to_well(c: dict) -> ms.WellSpec:
    w = c["well"]
    cls = [ms.ClusterSpec.from_perf_params(
        x=cl["x"], Cd=cl["Cd"], A_perf=cl["A_perf"], kappa=cl["kappa"], I_f=cl["I_f"],
        R_f=cl["R_f"], C_f=cl["C_f"], G_l=cl["G_l"], p_res=cl["p_res"], rho=w["rho"])
        for cl in c["clusters"]]
    return ms.WellSpec(L=w["L"], D=w["D"], a=w["a"], rho=w["rho"], nu=w["nu"], roughness=w["roughness"],
                       TVD=c["TVD"], clusters=cls, Q0=w["Q0"], t_s=1.0, t_c=w["t_c"], ramp="cosine",
                       toe_bc="dead_end")


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def case_schema_record(case_id: str, c: dict, split: str, seed: int, numerics: dict) -> dict:
    w = c["well"]
    return {
        "case_id": case_id,
        "group_id": c.get("group_id") or make_group_id(c),
        "split": split,
        "ood_type": c.get("ood_type"),
        "seed": int(seed),
        "well": {"L": w["L"], "D": w["D"], "a": w["a"], "rho": w["rho"], "nu": w["nu"],
                 "roughness": w["roughness"], "TVD": c["TVD"], "toe_bc": "dead_end"},
        "clusters": [
            {"x": float(cl["x"]), "p_res": float(cl["p_res"]),
             "perfs": {k: np.asarray(cl[k], dtype=float).tolist()
                       for k in ("Cd", "A_perf", "kappa", "I_f", "R_f", "C_f", "G_l")},
             "cluster_equivalents": {k: float(cl["eq"][k]) for k in cl["eq"]}}
            for cl in c["clusters"]
        ],
        "excitation": {"Q0": float(w["Q0"]), "t_s": 1.0, "t_c": float(w["t_c"]), "ramp": "cosine"},
        "numerics": numerics,
    }


def friction_record(friction: str, phy: dict, zielke: dict, bru: dict) -> dict:
    return {
        "model": friction,
        "kernel_M": int(zielke["production_path"]["M"]),
        "brunone_convention": bru["locked_convention"],
        "friction_time_scheme": phy["numerics"]["friction_time_scheme"],
        "zielke_fit_domain_tau": list(zielke["production_path"]["fit"]["domain_tau"]),
    }


def _simulate_one(job: dict) -> dict:
    """Worker: build, simulate, save.  Returns a manifest row (no fat arrays)."""
    case_id = job["case_id"]
    c = job["case"]
    seed = int(job["seed"])
    nx = int(job["nx"])
    cr = float(job["cr"])
    friction = job["friction"]
    n_periods = float(job["n_periods"])
    out_dir = Path(job["out_dir"])
    split = job.get("split", "pilot")
    ood_type = job.get("ood_type", c.get("ood_type"))
    snap_every = int(job.get("snap_every", 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    well = case_to_well(c)
    phy = load_yaml("physics.yaml")
    zielke = load_yaml("friction_zielke.yaml")
    bru = load_yaml("friction_brunone.yaml")
    data_cfg = load_yaml("data.yaml")
    num = ms.NumericsSpec(
        Nx_ref=nx, Cr_target=cr, friction_model=friction,
        friction_time_scheme=phy["numerics"]["friction_time_scheme"],
        inertia_scheme=phy["numerics"]["inertia_scheme"],
        storage_scheme=phy["numerics"]["storage_scheme"],
        stiff_guard=bool(phy["numerics"]["storage_stiff_guard"]),
        n_periods=n_periods, do_energy=True,
        kernel_M=int(zielke["production_path"]["M"]),
        brunone_convention=bru["locked_convention"],
        node_decim=16, snap_every=snap_every,
    )
    grid_dt = cr * (well.L / nx) / well.a
    if well.t_c < 5.0 * grid_dt:
        well.t_c = 5.0 * grid_dt
    try:
        res = ms.simulate(well, num)
    except Exception as exc:
        return {"case_id": case_id, "group_id": c.get("group_id") or case_id, "split": split,
                "ood_type": ood_type, "file": "", "sha256": "", "status": "error",
                "newton_fail_count": -1, "wall_clock_s": 0.0, "error": repr(exc), "params": c}
    p_wh = float(res.meta["steady"]["p_wh"])
    rules = phy["sampling_domain"]["rejection_rules"]
    if not (rules["p_wh_steady_min"] <= p_wh <= rules["p_wh_steady_max"]):
        return {"case_id": case_id, "group_id": c.get("group_id") or case_id, "split": split,
                "ood_type": ood_type, "file": "", "sha256": "", "status": "rejected",
                "newton_fail_count": int(res.n_fail), "wall_clock_s": res.meta["wall_clock_s"],
                "p_wh": p_wh, "params": c}
    numerics = {"Nx_ref": nx, "Cr_target": cr, "friction_model": friction,
                "kernel_M": int(zielke["production_path"]["M"]),
                "friction_time_scheme": phy["numerics"]["friction_time_scheme"],
                "window_periods": n_periods}
    rec = case_schema_record(case_id, c, split, seed, numerics)
    fric = friction_record(friction, phy, zielke, bru)
    ospec = obsmod.observation_spec(data_cfg, snr_db=float("inf"))
    try:
        t_obs, p_obs = obsmod.apply_observation_chain(res.t, res.p_head, ospec)
    except Exception:
        t_obs, p_obs = np.array([]), np.array([])
    path = out_dir / f"{case_id}.npz"
    payload = dict(
        t=res.t.astype(np.float32), p_head=res.p_head.astype(np.float32),
        Q_head=res.Q_head.astype(np.float32),
        node_t=res.node_t.astype(np.float32),
        node_H=res.node_H.astype(np.float32), node_Qm=res.node_Qm.astype(np.float32),
        node_Qp=res.node_Qp.astype(np.float32), node_sq=res.node_sq.astype(np.float32),
        node_pc=res.node_pc.astype(np.float32), node_z=res.node_z.astype(np.float32),
        mass_resid=res.mass_resid.astype(np.float32),
        newton_resid=res.newton_resid.astype(np.float32),
        t_obs=t_obs, p_obs=p_obs,
        params_json=np.array(json.dumps(c, default=_js)),
        grid_json=np.array(json.dumps(res.grid.summary(), default=_js)),
        friction_json=np.array(json.dumps(fric, default=_js)),
        observation_json=np.array(json.dumps(ospec, default=_js)),
        case_json=np.array(json.dumps(rec, default=_js)),
        meta_json=np.array(json.dumps(res.meta, default=_js)),
        seed=np.int64(seed),
    )
    if snap_every > 0 and res.snap_H.size:
        payload["snap_t"] = res.snap_t.astype(np.float32)
        payload["snap_H"] = res.snap_H.astype(np.float32)
        payload["snap_Q"] = res.snap_Q.astype(np.float32)
    np.savez_compressed(path, **payload)
    status = "newton_failed" if res.n_fail > 0 else "ok"
    gid = rec["group_id"]
    return {"case_id": case_id, "group_id": gid, "split": split, "ood_type": ood_type,
            "file": str(path.as_posix()), "sha256": sha256_file(path),
            "status": status, "newton_fail_count": int(res.n_fail),
            "wall_clock_s": float(res.meta["wall_clock_s"]),
            "p_wh": p_wh, "max_mass": float(res.meta["max_global_mass_resid_dimless"]),
            "xs": c["xs"], "N": c["N"]}


def _run_jobs(jobs: List[dict], audit: RunAudit, n_workers: int) -> List[dict]:
    rows = []
    if n_workers <= 1:
        w = goldens.joukowsky_case(V0=1.0)
        ms.simulate(w, ms.NumericsSpec(Nx_ref=32, Cr_target=1.0, friction_model="none",
                                       n_periods=0.4, do_energy=False))
        for job in jobs:
            rows.append(_simulate_one(job))
            audit.log(f"  {rows[-1]['case_id']} {rows[-1]['status']} nfail={rows[-1]['newton_fail_count']}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_simulate_one, job): job["case_id"] for job in jobs}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {"case_id": futs[fut], "group_id": futs[fut], "split": "error",
                           "ood_type": None, "file": "", "sha256": "", "status": "error",
                           "newton_fail_count": -1, "wall_clock_s": 0.0, "error": repr(exc)}
                rows.append(row)
                audit.log(f"  {row['case_id']} {row['status']} nfail={row.get('newton_fail_count')}")
    return rows


def gates_allow_formal() -> Tuple[bool, str]:
    p = GOLDEN_DIR / "metrics.json"
    if not p.exists():
        return False, "goldens/metrics.json missing — run validate.py first"
    m = json.loads(p.read_text(encoding="utf-8"))
    fails = [k for k, g in m.get("gates", {}).items() if g.get("status") == "FAIL"]
    newt = m.get("gates", {}).get("newton_convergence_pilot2000", {})
    if fails:
        return False, f"FAIL gates: {fails}"
    if newt.get("status") != "PASS":
        return False, "Newton pilot gate is not PASS"
    return True, "ok"


def ood_constraint_report(cases: List[dict], domain: dict) -> dict:
    """Audit that each OOD type actually left the training envelope (or the declared joint)."""
    tr = domain["train"]
    out = {}
    for typ in OOD_TYPES:
        subset = [c for c in cases if c.get("ood_type") == typ]
        out[typ] = {"n": len(subset), "ok": True, "notes": ""}
        if not subset:
            out[typ]["ok"] = False
            out[typ]["notes"] = "empty"
            continue
        if typ == "dense_spacing":
            ss = [float(c["spacing"]) for c in subset]
            ok = all(10.0 - 1e-9 <= s < 12.0 + 1e-9 for s in ss)
            out[typ].update(ok=ok, min=min(ss), max=max(ss), notes="spacing in [10,12)")
        elif typ == "sparse_spacing":
            ss = [float(c["spacing"]) for c in subset]
            ok = all(40.0 - 1e-9 <= s <= 60.0 + 1e-9 for s in ss)
            out[typ].update(ok=ok, min=min(ss), max=max(ss), notes="spacing in [40,60]")
        elif typ == "many_clusters":
            ns = [int(c["N"]) for c in subset]
            ok = all(13 <= n <= 16 for n in ns)
            out[typ].update(ok=ok, min=min(ns), max=max(ns), notes="N in [13,16]")
        elif typ == "tail_viscosity":
            vs = [float(c["well"]["nu"]) for c in subset]
            ok = all(v > tr["nu"]["high"] - 1e-15 for v in vs)
            out[typ].update(ok=ok, min=min(vs), max=max(vs), notes="nu > train high")
        elif typ == "tail_compliance":
            vs = [float(c["draw"]["C_f_cluster"]) for c in subset]
            ok = all(v > tr["C_f_cluster"]["high"] - 1e-15 for v in vs)
            out[typ].update(ok=ok, min=min(vs), max=max(vs), notes="C_f > train high")
        elif typ == "uncooccurring":
            q = [float(c["well"]["Q0"]) for c in subset]
            s = [float(c["spacing"]) for c in subset]
            i = [float(c["draw"]["I_f_cluster"]) for c in subset]
            ok = (all(qi >= 0.07 for qi in q) and all(si <= 14.0 + 1e-9 for si in s)
                  and all(ii >= 0.2 * tr["I_f_cluster"]["high"] for ii in i))
            out[typ].update(ok=ok, notes="high Q0 + dense-edge spacing + high I_f")
        elif typ == "deleted_cluster":
            ok = all(any(abs(float(cl["eq"]["sum_CdA"])) < 1e-18 for cl in c["clusters"])
                     for c in subset)
            out[typ].update(ok=ok, notes="at least one cluster sum_CdA = 0")
        elif typ == "impedance_jump":
            fac = [float(c.get("impedance_jump_factor", 0.0)) for c in subset]
            ok = all(abs(f - 3.0) < 1e-12 or abs(f - 1.0 / 3.0) < 1e-12 for f in fac)
            out[typ].update(ok=ok, notes="first-cluster Cd scaled by 3 or 1/3")
    return out


def build_formal_design(smoke: bool = False, n_override: Optional[int] = None) -> dict:
    """Owen-scrambled Sobol ID pool + 8 OOD types; group split; no MOC."""
    data_cfg = load_yaml("data.yaml")
    phy = load_yaml("physics.yaml")
    domain = phy["sampling_domain"]
    formal = data_cfg["formal"]
    seeds = list(formal["seeds"])
    if smoke:
        sm = formal["smoke"]
        n_train = int(sm["n_train"])
        n_val = int(sm["n_val"])
        n_id_test = int(sm["n_id_test"])
        n_ood = int(sm["n_ood"])
        if n_override is not None:
            n_train, n_val, n_id_test = n_override, 0, 0
    else:
        n_train = int(formal["n_train"])
        n_val = int(formal["n_val"])
        n_id_test = int(formal["n_id_test"])
        n_ood = int(formal["n_ood"])
        if n_override is not None:
            raise ValueError("--n with full formal is refused; use --smoke or --design-only defaults")
    n_id = n_train + n_val + n_id_test
    id_seed = int(seeds[0])
    id_cases = draw_n_cases(n_id, id_seed, domain, ood_type=None)
    splits, split_counts = assign_group_splits(id_cases, n_train, n_val, n_id_test, seed=id_seed + 7)
    records = []
    for i, (c, spl) in enumerate(zip(id_cases, splits)):
        c["group_id"] = c.get("group_id") or make_group_id(c)
        records.append({
            "case_id": f"{'smoke' if smoke else 'id'}_{i:05d}",
            "split": spl,
            "ood_type": None,
            "seed": id_seed + i,
            "case": c,
        })
    n_per = max(1, n_ood // len(OOD_TYPES))
    remainder = n_ood - n_per * len(OOD_TYPES)
    ood_cases = []
    k = 0
    for ti, typ in enumerate(OOD_TYPES):
        nt = n_per + (1 if ti < remainder else 0)
        seed_t = int(seeds[1 + (ti % max(1, len(seeds) - 1))]) + 17 * ti
        drawn = draw_n_cases(nt, seed_t, domain, ood_type=typ)
        for j, c in enumerate(drawn):
            c["group_id"] = make_group_id(c)
            rec = {
                "case_id": f"{'smoke_' if smoke else ''}ood_{typ}_{j:04d}",
                "split": "ood",
                "ood_type": typ,
                "seed": seed_t + j,
                "case": c,
            }
            records.append(rec)
            ood_cases.append(c)
            k += 1
    leak = split_leakage([{"group_id": r["case"].get("group_id") or make_group_id(r["case"]),
                           "split": r["split"]} for r in records])
    ood_rep = ood_constraint_report(ood_cases, domain)
    return {
        "smoke": smoke,
        "design": "sobol_owen_scrambled",
        "seeds": seeds,
        "n_id": n_id,
        "n_ood": len(ood_cases),
        "split_counts": {**split_counts, "ood": len(ood_cases)},
        "leakage": leak,
        "ood_constraints": ood_rep,
        "records": records,
    }


def _jobs_from_design(design: dict, nx: int, cr: float, friction: str, n_periods: float,
                      snap_fraction: float) -> List[dict]:
    jobs = []
    n_rec = len(design["records"])
    snap_idx = set()
    if snap_fraction > 0 and n_rec:
        n_snap = max(1, int(round(snap_fraction * n_rec)))
        rng = np.random.default_rng(0)
        snap_idx = set(rng.choice(n_rec, size=min(n_snap, n_rec), replace=False).tolist())
    for i, rec in enumerate(design["records"]):
        split = rec["split"]
        ood_type = rec["ood_type"]
        if split == "ood":
            out_dir = OOD_DIR / ("smoke" if design["smoke"] else "cases") / str(ood_type)
        else:
            out_dir = SOBOL_DIR / ("smoke" if design["smoke"] else "cases") / split
        jobs.append({
            "case_id": rec["case_id"],
            "case": rec["case"],
            "seed": rec["seed"],
            "nx": nx, "cr": cr, "friction": friction, "n_periods": n_periods,
            "out_dir": str(out_dir),
            "split": split,
            "ood_type": ood_type,
            "snap_every": 50 if i in snap_idx else 0,
        })
    return jobs


def _write_design_sidecar(design: dict, dest: Path) -> Path:
    slim = {k: design[k] for k in design if k != "records"}
    slim["records"] = [{
        "case_id": r["case_id"], "split": r["split"], "ood_type": r["ood_type"],
        "seed": r["seed"], "group_id": r["case"].get("group_id"),
        "N": r["case"]["N"], "spacing": r["case"]["spacing"],
        "xs": r["case"]["xs"], "L": r["case"]["well"]["L"],
        "nu": r["case"]["well"]["nu"],
        "C_f_cluster": r["case"]["draw"]["C_f_cluster"],
        "Q0": r["case"]["well"]["Q0"],
        "I_f_cluster": r["case"]["draw"]["I_f_cluster"],
        "deleted_index": r["case"].get("deleted_index"),
        "impedance_jump_factor": r["case"].get("impedance_jump_factor"),
    } for r in design["records"]]
    dump_json(slim, dest)
    return dest


def run_formal(smoke: bool, design_only: bool, n_workers: int,
               n_override: Optional[int] = None) -> dict:
    ensure_dirs()
    data_cfg = load_yaml("data.yaml")
    formal = data_cfg["formal"]
    if smoke:
        nx = int(formal["smoke"]["numerics"]["Nx_ref"])
        cr = float(formal["smoke"]["numerics"]["Cr_target"])
        friction = formal["smoke"]["numerics"]["friction_model"]
        n_periods = float(formal["smoke"]["window_periods"])
        snap_frac = 0.25
    else:
        nx = int(formal["numerics"]["Nx_ref"])
        cr = float(formal["numerics"]["Cr_target"])
        friction = formal["numerics"]["friction_model"]
        n_periods = float(data_cfg["pilot"]["window_periods"])
        snap_frac = float(data_cfg["storage"]["full_field_snapshots"]["case_fraction"])
    resolved = {"mode": "formal", "smoke": smoke, "design_only": design_only,
                "nx": nx, "cr": cr, "friction": friction, "n_periods": n_periods}
    audit = RunAudit("s1_formal", seed=int(formal["seeds"][0]), resolved_config=resolved)
    audit.log(f"building formal design smoke={smoke} design_only={design_only}")
    design = build_formal_design(smoke=smoke, n_override=n_override)
    audit.log(f"design n_id={design['n_id']} n_ood={design['n_ood']} splits={design['split_counts']}")
    if design["leakage"]:
        raise RuntimeError(f"R8 split leakage: {design['leakage']}")
    if not all(v["ok"] for v in design["ood_constraints"].values()):
        raise RuntimeError(f"OOD constraints failed: {design['ood_constraints']}")
    dname = "design_smoke.json" if smoke else "design_formal.json"
    dest_design = (SOBOL_DIR / "smoke" / dname) if smoke else (SOBOL_DIR / dname)
    if smoke:
        (SOBOL_DIR / "smoke").mkdir(parents=True, exist_ok=True)
        (OOD_DIR / "smoke").mkdir(parents=True, exist_ok=True)
    _write_design_sidecar(design, dest_design)
    audit.log(f"wrote {dest_design}")

    coverage = {
        "mode": "formal", "smoke": smoke, "design_only": design_only,
        "n_id": design["n_id"], "n_ood": design["n_ood"],
        "split_counts": design["split_counts"],
        "leakage": design["leakage"],
        "ood_constraints": design["ood_constraints"],
        "design_path": str(dest_design.as_posix()),
    }
    if design_only:
        dump_json(coverage, dest_design.parent / ("coverage_design_smoke.json" if smoke
                                                   else "coverage_design.json"))
        audit.write_metrics({"stage": 1, "formal_design": coverage})
        audit.close()
        return coverage

    jobs = _jobs_from_design(design, nx, cr, friction, n_periods, snap_frac)
    audit.log(f"simulating {len(jobs)} cases workers={n_workers}")
    rows = _run_jobs(jobs, audit, n_workers)
    n_ok = sum(r["status"] == "ok" for r in rows)
    n_nf = sum(r["status"] == "newton_failed" for r in rows)
    n_err = sum(r["status"] == "error" for r in rows)
    n_rej = sum(r["status"] == "rejected" for r in rows)
    n_ran = n_ok + n_nf
    rate = n_ok / n_ran if n_ran else 0.0
    leak = split_leakage(rows)
    manifest = {
        "manifest_version": "v1_formal_smoke" if smoke else "v1_formal",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": {"code_digest": audit.meta["code_digest"], "config_digest": audit.meta["config_digest"],
                      "git_commit": audit.meta["git_commit"], "run_id": audit.run_id,
                      "frozen": False},
        "design": {"type": "sobol_owen_scrambled", "seed": int(formal["seeds"][0]),
                   "seeds": list(formal["seeds"]), "n": len(rows),
                   "n_requested": design["n_id"] + design["n_ood"]},
        "splits": {k: int(sum(r["split"] == k for r in rows))
                   for k in ("train", "val", "id_test", "ood")},
        "numerics": {"Nx_ref": nx, "Cr_target": cr, "friction_model": friction,
                     "window_periods": n_periods, "smoke": smoke},
        "cases": [{k: r.get(k) for k in ("case_id", "group_id", "split", "ood_type", "file",
                                         "sha256", "status", "newton_fail_count", "wall_clock_s")}
                  for r in rows],
        "failures": [r for r in rows if r["status"] in ("newton_failed", "error")],
        "leakage": leak,
        "digest": "",
    }
    raw = json.dumps(manifest, sort_keys=True, default=str).encode()
    manifest["digest"] = hashlib.sha256(raw).hexdigest()
    mname = "manifest_formal_smoke.json" if smoke else "manifest_formal.json"
    dump_json(manifest, MANIFEST_DIR / mname)
    try:
        validate_against_schema(manifest, "manifest.schema.json")
    except Exception as exc:
        audit.log(f"schema warning: {exc}")
    coverage.update({
        "n_ok": n_ok, "n_newton_failed": n_nf, "n_rejected": n_rej, "n_error": n_err,
        "newton_case_convergence_rate": rate, "leakage_after_sim": leak,
        "manifest": str((MANIFEST_DIR / mname).as_posix()),
    })
    dump_json(coverage, dest_design.parent / ("coverage_smoke.json" if smoke else "coverage.json"))
    audit.write_metrics({"stage": 1, "formal": coverage, "manifest_digest": manifest["digest"]})
    audit.close()
    return coverage


def run_pilot(n: int, smoke: bool, n_workers: int, seed: Optional[int] = None) -> dict:
    ensure_dirs()
    data_cfg = load_yaml("data.yaml")
    phy = load_yaml("physics.yaml")
    seed = seed if seed is not None else int(data_cfg["pilot"]["seed"])
    n_cand = 2 if smoke else int(data_cfg["pilot"]["n_candidates"])
    nx = 256 if smoke else int(data_cfg["pilot"]["numerics"]["Nx_ref"])
    cr = float(data_cfg["pilot"]["numerics"]["Cr_target"])
    friction = data_cfg["pilot"]["numerics"]["friction_model"]
    n_periods = 3.0 if smoke else float(data_cfg["pilot"]["window_periods"])
    out_dir = PILOT_DIR / ("smoke" if smoke else "cases")

    resolved = {"n": n, "smoke": smoke, "seed": seed, "nx": nx, "cr": cr,
                "friction": friction, "n_periods": n_periods}
    audit = RunAudit("s1_pilot", seed=seed, resolved_config=resolved)
    audit.log(f"building maximin-LHS n={n} d={len(TRAIN_KEYS)} candidates={n_cand}")
    U = maximin_lhs(n, len(TRAIN_KEYS), seed, n_cand)
    rng = np.random.default_rng(seed)
    drawn = []
    n_try = 0
    extra = qmc.LatinHypercube(d=len(TRAIN_KEYS), seed=seed + 999).random(max(n, 8 * n))
    pool = list(U) + list(extra)
    for row in pool:
        n_try += 1
        c = draw_case(np.asarray(row), rng, phy["sampling_domain"])
        if c is None:
            continue
        drawn.append(c)
        if len(drawn) >= n:
            break
    audit.log(f"accepted {len(drawn)} / tried {n_try} (rejections={n_try - len(drawn)})")

    jobs = []
    for i, c in enumerate(drawn):
        c["group_id"] = make_group_id(c)
        jobs.append({
            "case_id": f"pilot_{i:05d}", "case": c, "seed": seed + i,
            "nx": nx, "cr": cr, "friction": friction, "n_periods": n_periods,
            "out_dir": str(out_dir), "split": "pilot", "ood_type": None, "snap_every": 0,
        })

    rows = _run_jobs(jobs, audit, n_workers)

    n_ok = sum(r["status"] == "ok" for r in rows)
    n_nf = sum(r["status"] == "newton_failed" for r in rows)
    n_err = sum(r["status"] == "error" for r in rows)
    n_rej = sum(r["status"] == "rejected" for r in rows)
    n_ran = n_ok + n_nf
    rate = n_ok / n_ran if n_ran else 0.0
    audit.log(f"ok={n_ok} newton_failed={n_nf} rejected={n_rej} error={n_err}  rate={rate:.6f}")

    manifest = {
        "manifest_version": "v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": {"code_digest": audit.meta["code_digest"], "config_digest": audit.meta["config_digest"],
                      "git_commit": audit.meta["git_commit"], "run_id": audit.run_id,
                      "frozen": bool(rate > 0.999 and n_err == 0 and not smoke)},
        "design": {"type": "maximin_lhs", "seed": seed, "n": len(rows), "n_requested": n,
                   "n_candidates": n_cand},
        "splits": {"pilot": len(rows)},
        "numerics": {"Nx_ref": nx, "Cr_target": cr, "friction_model": friction,
                     "window_periods": n_periods, "smoke": smoke},
        "cases": [{k: r[k] for k in ("case_id", "group_id", "split", "file", "sha256", "status",
                                     "newton_fail_count", "wall_clock_s")} for r in rows],
        "failures": [r for r in rows if r["status"] in ("newton_failed", "error")],
        "digest": "",
    }
    raw = json.dumps(manifest, sort_keys=True, default=str).encode()
    manifest["digest"] = hashlib.sha256(raw).hexdigest()
    dest = MANIFEST_DIR / ("manifest_v1_smoke.json" if smoke else "manifest_v1.json")
    dump_json(manifest, dest)
    try:
        validate_against_schema(manifest, "manifest.schema.json")
    except Exception as exc:
        audit.log(f"schema warning: {exc}")

    Ns = [r.get("N") for r in rows if r.get("N") is not None]
    coverage = {
        "n_ok": n_ok, "n_newton_failed": n_nf, "n_rejected": n_rej, "n_error": n_err,
        "newton_case_convergence_rate": rate,
        "N_clusters_hist": {str(k): int(sum(n == k for n in Ns)) for k in sorted(set(Ns))},
        "min_spacing": float(min((min(np.diff(r["xs"])) for r in rows if r.get("xs") and len(r["xs"]) > 1),
                                 default=float("nan"))),
    }
    dump_json(coverage, PILOT_DIR / ("coverage_smoke.json" if smoke else "coverage.json"))

    mp = GOLDEN_DIR / "metrics.json"
    if mp.exists() and not smoke:
        m = json.loads(mp.read_text(encoding="utf-8"))
        m.setdefault("gates", {})["newton_convergence_pilot2000"] = {
            "value": rate, "threshold": 0.999, "comparison": ">",
            "status": "PASS" if rate > 0.999 else "FAIL",
            "Cr": 1.0,
            "notes": f"ok={n_ok} failed={n_nf} error={n_err} rejected={n_rej}; failures retained",
        }
        m["n_fail"] = sum(g.get("status") == "FAIL" for g in m["gates"].values())
        dump_json(m, mp)
        dump_json(m, Path(audit.dir) / "metrics.json")

    audit.write_metrics({"stage": 1, "pilot": coverage, "manifest_digest": manifest["digest"]})
    audit.close()
    return coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--confirm-30k", action="store_true", dest="confirm_30k",
                    help="explicitly launch the full 30k MOC expansion")
    ap.add_argument("--design-only", action="store_true", dest="design_only",
                    help="write Sobol+OOD designs and split tables without running MOC")
    args = ap.parse_args()
    if args.formal:
        ok, msg = gates_allow_formal()
        if not ok:
            print(f"REFUSED formal expansion: {msg}")
            print("Per §3.3 / §3.4: stop data expansion until every gate PASSes.")
            sys.exit(2)
        if args.confirm_30k and args.smoke:
            ap.error("--confirm-30k and --smoke are mutually exclusive")
        if args.confirm_30k and args.design_only:
            ap.error("--confirm-30k launches MOC; drop --design-only")
        if not args.smoke and not args.design_only and not args.confirm_30k:
            print("Gates are PASS. Formal Sobol+OOD generator is implemented.")
            print("Refusing to silently launch 30k MOC cases.")
            print("Use --formal --smoke           (tiny simulate)")
            print("    --formal --design-only     (full 20k/3k/3k/4k designs, no MOC)")
            print("    --formal --confirm-30k     (full MOC; ~15× pilot wall-clock)")
            sys.exit(3)
        cov = run_formal(smoke=args.smoke, design_only=args.design_only,
                         n_workers=args.workers, n_override=args.n if args.smoke else None)
        print(json.dumps(cov, indent=2, default=str))
        ood_ok = all(v.get("ok", False) for v in cov.get("ood_constraints", {}).values())
        sys.exit(0 if ood_ok and not cov.get("leakage") else 1)
    if not args.pilot:
        ap.error("specify --pilot or --formal")
    n = args.n if args.n is not None else (20 if args.smoke else int(load_yaml("data.yaml")["pilot"]["n"]))
    cov = run_pilot(n=n, smoke=args.smoke, n_workers=args.workers)
    print(json.dumps(cov, indent=2))
    sys.exit(0 if cov["newton_case_convergence_rate"] > 0.999 or args.smoke else 1)


if __name__ == "__main__":
    main()
