# -*- coding: utf-8 -*-
"""
docs/moc_v2_technical_report/run_sensitivity_study.py

Production-grade MOC_V2 Field Sensitivity Pipeline & Nature-Grade Figure Generator
Executes Base Case and 7 Sensitivity Topics (42 simulation runs) using multi-process
parallel execution, computes 1D real cepstrum and 2D continuous cepstrograms (Hamming window),
and renders Figure 1-7 in both 300 DPI PNG and vector SVG format (14 files total).

Topics:
- Base Case: L=5000m, D=0.1397m, a=1450m/s, V0=1.0m/s, H0=300m, Hext=100m,
             3 fractures at [4500, 4510, 4520]m, Cf=0.01m^2, kleak=1.0e-4 m^2.5/s,
             Kp=5.43e5 s^2/m^5, tc=0.1s (cosine ramp, Brunone friction).
- Topic 1: Fracture Count Nc in [1, 2, 3, 4, 5, 6, 7, 8], start at 4500m, 10m spacing, w=1/Nc, tc=0.1s (8 runs)
- Topic 2: Fracture Spacing d in [5, 10, 15, 20, 25, 30, 50, 80]m, tc=0.1s (8 runs)
- Topic 3: Compliance Cf in [0.002, 0.005, 0.010, 0.020, 0.030]m^2, tc=0.1s (5 runs)
- Topic 4: Leak-off kleak in [0.2, 0.6, 1.0, 3.0, 10.0] * 1e-4 m^2.5/s, tc=0.1s (5 runs)
- Topic 5: Perforation Kp in [1.5, 3.5, 5.43, 10.0, 25.0] * 1e5 s^2/m^5, tc=0.1s (5 runs)
- Topic 6: Ramp Closure tc in [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]s (6 runs)
- Topic 7: Intake Capacity Combinations (5 typical cases, tc=0.1s)
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moc_simulate.v2 import MocV2Config, simulate_v2
from moc_simulate.v2.signal import (
    compute_cepstrum_1d,
    compute_cepstrogram_2d,
    quefrency_to_distance,
)


def _setup_matplotlib():
    """Deferred import and configuration of Matplotlib to keep worker processes lightweight."""
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",     # Keep text as editable <text> in vector SVG
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


# ==============================================================================
# 1. Simulation Case Specification Data Structure
# ==============================================================================

@dataclass
class SimCaseSpec:
    case_id: str
    topic_id: int
    label: str
    positions: List[float]
    Cf: List[float]
    kleak: List[float]
    Kp: List[float]
    weights: List[float]
    tc: float
    calc_2d: bool = False
    is_base_case: bool = False

    def param_hash(self) -> str:
        """Deterministic fingerprint of physics parameters for result caching."""
        spec_tuple = (
            tuple(round(p, 4) for p in self.positions),
            tuple(round(c, 6) for c in self.Cf),
            tuple(round(k, 8) for k in self.kleak),
            tuple(round(kp, 2) for kp in self.Kp),
            tuple(round(w, 4) for w in self.weights),
            round(self.tc, 4),
        )
        return hashlib.sha256(repr(spec_tuple).encode("utf-8")).hexdigest()[:16]


def build_manifest() -> List[SimCaseSpec]:
    """Construct specifications for Base Case and Topics 1 to 7 (42 runs)."""
    manifest: List[SimCaseSpec] = []

    # --- Topic 1: Fracture Count (Nc in [1..8], 10m spacing from 4500m) ---
    for nc in [1, 2, 3, 4, 5, 6, 7, 8]:
        pos = [4500.0 + 10.0 * i for i in range(nc)]
        w = [1.0 / nc] * nc
        cf = [0.01] * nc
        kleak = [1.0e-4] * nc
        kp = [5.43e5] * nc
        is_base = (nc == 3)
        calc_2d = (nc == 1 or nc == 8)
        label = f"$N_c = {nc}$" + (" (Base)" if is_base else "")
        manifest.append(
            SimCaseSpec(
                case_id=f"T1_Nc{nc}",
                topic_id=1,
                label=label,
                positions=pos,
                Cf=cf,
                kleak=kleak,
                Kp=kp,
                weights=w,
                tc=0.1,
                calc_2d=calc_2d,
                is_base_case=is_base,
            )
        )

    # --- Topic 2: Fracture Spacing (3 fractures, d in [5, 10, 15, 20, 25, 30, 50, 80]m) ---
    for d in [5, 10, 15, 20, 25, 30, 50, 80]:
        pos = [4500.0, 4500.0 + float(d), 4500.0 + 2.0 * float(d)]
        w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        cf = [0.01, 0.01, 0.01]
        kleak = [1.0e-4, 1.0e-4, 1.0e-4]
        kp = [5.43e5, 5.43e5, 5.43e5]
        is_base = (d == 10)
        calc_2d = (d == 10 or d == 50)
        label = f"$d = {d}\\,\\mathrm{{m}}$" + (" (Base)" if is_base else "")
        manifest.append(
            SimCaseSpec(
                case_id=f"T2_d{d}",
                topic_id=2,
                label=label,
                positions=pos,
                Cf=cf,
                kleak=kleak,
                Kp=kp,
                weights=w,
                tc=0.1,
                calc_2d=calc_2d,
                is_base_case=is_base,
            )
        )

    # --- Topic 3: Compliance Cf in [0.002, 0.005, 0.010, 0.020, 0.030] m^2 ---
    for cf_val in [0.002, 0.005, 0.010, 0.020, 0.030]:
        pos = [4500.0, 4510.0, 4520.0]
        w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        cf = [cf_val, cf_val, cf_val]
        kleak = [1.0e-4, 1.0e-4, 1.0e-4]
        kp = [5.43e5, 5.43e5, 5.43e5]
        is_base = abs(cf_val - 0.010) < 1e-6
        calc_2d = (abs(cf_val - 0.002) < 1e-6 or abs(cf_val - 0.030) < 1e-6)
        label = f"$C_f = {cf_val:g}\\,\\mathrm{{m^2}}$" + (" (Base)" if is_base else "")
        manifest.append(
            SimCaseSpec(
                case_id=f"T3_Cf{cf_val:.3f}",
                topic_id=3,
                label=label,
                positions=pos,
                Cf=cf,
                kleak=kleak,
                Kp=kp,
                weights=w,
                tc=0.1,
                calc_2d=calc_2d,
                is_base_case=is_base,
            )
        )

    # --- Topic 4: Leak-off kleak in [0.2, 0.6, 1.0, 3.0, 10.0] * 1e-4 m^2.5/s ---
    for kl_mult in [0.2, 0.6, 1.0, 3.0, 10.0]:
        kl_val = kl_mult * 1.0e-4
        pos = [4500.0, 4510.0, 4520.0]
        w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        cf = [0.01, 0.01, 0.01]
        kleak = [kl_val, kl_val, kl_val]
        kp = [5.43e5, 5.43e5, 5.43e5]
        is_base = abs(kl_mult - 1.0) < 1e-6
        calc_2d = (abs(kl_mult - 0.2) < 1e-6 or abs(kl_mult - 10.0) < 1e-6)
        tag = " (Fault)" if abs(kl_mult - 10.0) < 1e-6 else (" (Base)" if is_base else "")
        label = f"$k_{{leak}} = {kl_mult:g}\\times 10^{{-4}}$" + tag
        manifest.append(
            SimCaseSpec(
                case_id=f"T4_kleak{kl_mult:g}",
                topic_id=4,
                label=label,
                positions=pos,
                Cf=cf,
                kleak=kleak,
                Kp=kp,
                weights=w,
                tc=0.1,
                calc_2d=calc_2d,
                is_base_case=is_base,
            )
        )

    # --- Topic 5: Perforation Impedance Kp in [1.5, 3.5, 5.43, 10.0, 25.0] * 1e5 s^2/m^5 ---
    # corresponding holes: 16, 8, 6, 4, 2
    kp_info = [
        (1.5, "16 holes"),
        (3.5, "8 holes"),
        (5.43, "6 holes, Base"),
        (10.0, "4 holes"),
        (25.0, "2 holes"),
    ]
    for kp_mult, hole_desc in kp_info:
        kp_val = kp_mult * 1.0e5
        pos = [4500.0, 4510.0, 4520.0]
        w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        cf = [0.01, 0.01, 0.01]
        kleak = [1.0e-4, 1.0e-4, 1.0e-4]
        kp = [kp_val, kp_val, kp_val]
        is_base = abs(kp_mult - 5.43) < 1e-6
        calc_2d = (abs(kp_mult - 1.5) < 1e-6 or abs(kp_mult - 25.0) < 1e-6)
        label = f"$K_p = {kp_mult:g}\\times 10^5$ ({hole_desc})"
        manifest.append(
            SimCaseSpec(
                case_id=f"T5_Kp{kp_mult:g}",
                topic_id=5,
                label=label,
                positions=pos,
                Cf=cf,
                kleak=kleak,
                Kp=kp,
                weights=w,
                tc=0.1,
                calc_2d=calc_2d,
                is_base_case=is_base,
            )
        )

    # --- Topic 6: Pump Shutoff Ramp Duration tc in [0.0, 0.05, 0.1, 0.5, 1.0, 2.0] s ---
    for tc_val in [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]:
        pos = [4500.0, 4510.0, 4520.0]
        w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        cf = [0.01, 0.01, 0.01]
        kleak = [1.0e-4, 1.0e-4, 1.0e-4]
        kp = [5.43e5, 5.43e5, 5.43e5]
        is_base = abs(tc_val - 0.1) < 1e-6
        calc_2d = (abs(tc_val - 0.1) < 1e-6 or abs(tc_val - 2.0) < 1e-6)
        tag = " (Step)" if abs(tc_val - 0.0) < 1e-6 else (" (Base)" if is_base else "")
        label = f"$t_c = {tc_val:g}\\,\\mathrm{{s}}$" + tag
        manifest.append(
            SimCaseSpec(
                case_id=f"T6_tc{tc_val:g}",
                topic_id=6,
                label=label,
                positions=pos,
                Cf=cf,
                kleak=kleak,
                Kp=kp,
                weights=w,
                tc=tc_val,
                calc_2d=calc_2d,
                is_base_case=is_base,
            )
        )

    # --- Topic 7: Intake Capacity Combinations (5 Typical Cases) ---
    t7_specs = [
        (
            "Case 7.1",
            "[Med, Med, Med] (Base)",
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.010, 0.010, 0.010],
            [1.0e-4, 1.0e-4, 1.0e-4],
            [5.43e5, 5.43e5, 5.43e5],
            True,   # calc_2d
            True,   # is_base
        ),
        (
            "Case 7.2",
            "[High, Med, Low] Heel-dom",
            [0.60, 0.25, 0.15],
            [0.020, 0.010, 0.005],
            [2.2e-4, 1.0e-4, 0.5e-4],
            [2.5e5, 5.43e5, 10.0e5],
            False,
            False,
        ),
        (
            "Case 7.3",
            "[High, Low, High] Saddle",
            [0.45, 0.10, 0.45],
            [0.018, 0.004, 0.018],
            [2.0e-4, 0.4e-4, 2.0e-4],
            [2.5e5, 12.0e5, 2.5e5],
            False,
            False,
        ),
        (
            "Case 7.4",
            "[Low, Med, High] Toe-dom",
            [0.20, 0.25, 0.55],
            [0.008, 0.010, 0.022],
            [0.8e-4, 1.0e-4, 2.5e-4],
            [6.0e5, 5.43e5, 2.0e5],
            False,
            False,
        ),
        (
            "Case 7.5",
            "[Dead, Med, High] Plugged",
            [0.01, 0.39, 0.60],
            [0.0005, 0.010, 0.022],
            [0.05e-4, 1.0e-4, 2.5e-4],
            [8.0e7, 5.43e5, 2.0e5],
            True,   # calc_2d
            False,
        ),
    ]
    for cid, clabel, w, cf, kl, kp, c2d, is_base in t7_specs:
        manifest.append(
            SimCaseSpec(
                case_id=f"T7_{cid.replace(' ', '_')}",
                topic_id=7,
                label=f"{cid}: {clabel}",
                positions=[4500.0, 4510.0, 4520.0],
                Cf=cf,
                kleak=kl,
                Kp=kp,
                weights=w,
                tc=0.1,
                calc_2d=c2d,
                is_base_case=is_base,
            )
        )

    return manifest


CURRENT_STCT_TAG = "kaiser_wlen30_hop1"


# ==============================================================================
# 2. Worker Function for Parallel Execution
# ==============================================================================

def _worker_execute_sim(spec_dict: dict) -> dict:
    """Independent worker function to execute a single forward MOC simulation with disk caching."""
    spec = SimCaseSpec(**spec_dict)
    t_start = time.time()
    h = spec.param_hash()

    cache_dir = REPO_ROOT / "docs" / "moc_v2_technical_report" / ".sim_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{h}.npz"

    # Check cache first
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                has_sim = "timestamps" in data and "wellhead_head" in data
                stct_tag = str(data["stct_tag"][0]) if "stct_tag" in data else ""
                has_2d = (
                    "ceps2d_matrix" in data
                    and data["ceps2d_matrix"] is not None
                    and data["ceps2d_matrix"].size > 0
                    and stct_tag == CURRENT_STCT_TAG
                )
                if has_sim and ((not spec.calc_2d) or has_2d):
                    return {
                        "case_id": spec.case_id,
                        "param_hash": h,
                        "timestamps": data["timestamps"],
                        "wellhead_head": data["wellhead_head"],
                        "ceps1d_distance": data["ceps1d_distance"],
                        "ceps1d_amp": data["ceps1d_amp"],
                        "ceps2d_time_centers": data["ceps2d_time_centers"] if has_2d else None,
                        "ceps2d_distances": data["ceps2d_distances"] if has_2d else None,
                        "ceps2d_matrix": data["ceps2d_matrix"] if has_2d else None,
                        "a_adj": float(np.squeeze(data["a_adj"])),
                        "t_elapsed": 0.0,
                    }
                elif has_sim:
                    # Fast path: recompute Kaiser 2D cepstrogram without rerunning MOC forward solve
                    timestamps = data["timestamps"]
                    wh_head = data["wellhead_head"]
                    a_adj = float(np.squeeze(data["a_adj"]))
                    ceps1d_dist = data["ceps1d_distance"]
                    ceps1d_amp = data["ceps1d_amp"]

                    ceps2d = None
                    if spec.calc_2d:
                        ceps2d = compute_cepstrogram_2d(
                            timestamps,
                            wh_head,
                            wavespeed=a_adj,
                            fs=1000.0,
                            ts=1.0,
                            window_len_s=30.0,
                            hop_len_s=1.0,
                            window="kaiser",
                            max_distance=5000.0,
                        )

                    save_dict = {
                        "timestamps": timestamps,
                        "wellhead_head": wh_head,
                        "ceps1d_distance": ceps1d_dist,
                        "ceps1d_amp": ceps1d_amp,
                        "a_adj": np.array([a_adj]),
                        "stct_tag": np.array([CURRENT_STCT_TAG]),
                    }
                    if ceps2d is not None:
                        save_dict["ceps2d_time_centers"] = ceps2d["time_centers"]
                        save_dict["ceps2d_distances"] = ceps2d["distances"]
                        save_dict["ceps2d_matrix"] = ceps2d["cepstrogram"]

                    np.savez_compressed(cache_path, **save_dict)
                    return {
                        "case_id": spec.case_id,
                        "param_hash": h,
                        "timestamps": timestamps,
                        "wellhead_head": wh_head,
                        "ceps1d_distance": ceps1d_dist,
                        "ceps1d_amp": ceps1d_amp,
                        "ceps2d_time_centers": ceps2d["time_centers"] if ceps2d is not None else None,
                        "ceps2d_distances": ceps2d["distances"] if ceps2d is not None else None,
                        "ceps2d_matrix": ceps2d["cepstrogram"] if ceps2d is not None else None,
                        "a_adj": a_adj,
                        "t_elapsed": time.time() - t_start,
                    }
        except Exception:
            pass

    cfg = MocV2Config(
        wellbore_length=5000.0,
        wellbore_diameter=0.1397,
        wavespeed=1450.0,
        initial_velocity=1.0,
        initial_head=300.0,
        pump_shut_time=1.0,
        pump_closure_duration=spec.tc,
        ramp_type="cosine",
        friction_model="brunone",
        brunone_k_scale=1.0,
        toe_bc="dead_end",
        store_full_field=False,
        dt=0.001,
        tf=60.0,
    )

    res = simulate_v2(
        cfg=cfg,
        fracture_positions=spec.positions,
        fracture_Cf=spec.Cf,
        fracture_kleak=spec.kleak,
        fracture_inflow_weights=spec.weights,
        fracture_Kp=spec.Kp,
        H_ext=100.0,
    )

    timestamps = res["timestamps"]
    wh_head = res["wellhead_head"]

    # Verify no NaN or Inf
    if not np.all(np.isfinite(wh_head)):
        raise RuntimeError(f"Simulation {spec.case_id} produced NaN/Inf values!")

    # 1D Real Cepstrum via compute_cepstrum_1d
    ceps1d = compute_cepstrum_1d(
        timestamps,
        wh_head,
        wavespeed=cfg.a_adj,
        fs=1000.0,
        ts=1.0,
        window="hamming",
        derivative=True,
        derivative_order=1,
        max_distance=5000.0,
    )

    # 2D Continuous Cepstrogram (STCT) via compute_cepstrogram_2d with Kaiser window (wlen=30.0s, hop=1.0s)
    ceps2d = None
    if spec.calc_2d:
        ceps2d = compute_cepstrogram_2d(
            timestamps,
            wh_head,
            wavespeed=cfg.a_adj,
            fs=1000.0,
            ts=1.0,
            window_len_s=30.0,
            hop_len_s=1.0,
            window="kaiser",
            max_distance=5000.0,
        )

    t_elapsed = time.time() - t_start

    # Save to disk cache
    save_dict = {
        "timestamps": timestamps,
        "wellhead_head": wh_head,
        "ceps1d_distance": ceps1d["distance"],
        "ceps1d_amp": ceps1d["cepstrum"],
        "a_adj": np.array([cfg.a_adj]),
        "stct_tag": np.array([CURRENT_STCT_TAG]),
    }
    if ceps2d is not None:
        save_dict["ceps2d_time_centers"] = ceps2d["time_centers"]
        save_dict["ceps2d_distances"] = ceps2d["distances"]
        save_dict["ceps2d_matrix"] = ceps2d["cepstrogram"]

    np.savez_compressed(cache_path, **save_dict)

    return {
        "case_id": spec.case_id,
        "param_hash": h,
        "timestamps": timestamps,
        "wellhead_head": wh_head,
        "ceps1d_distance": ceps1d["distance"],
        "ceps1d_amp": ceps1d["cepstrum"],
        "ceps2d_time_centers": ceps2d["time_centers"] if ceps2d is not None else None,
        "ceps2d_distances": ceps2d["distances"] if ceps2d is not None else None,
        "ceps2d_matrix": ceps2d["cepstrogram"] if ceps2d is not None else None,
        "a_adj": cfg.a_adj,
        "t_elapsed": t_elapsed,
    }


# ==============================================================================
# 3. Parallel Simulation Pipeline
# ==============================================================================

def run_simulation_pipeline(
    manifest: List[SimCaseSpec], max_workers: int = 6
) -> Dict[str, dict]:
    """
    Executes all distinct MOC simulations in parallel.
    Identical parameter cases share results via param_hash deduplication and disk cache.
    """
    print(f"[{time.strftime('%X')}] Initializing simulation pipeline for {len(manifest)} cases...")

    # Group specifications by unique parameter hash
    unique_specs: Dict[str, SimCaseSpec] = {}
    hash_to_case_ids: Dict[str, List[str]] = {}
    needs_2d_hash: Dict[str, bool] = {}

    for spec in manifest:
        h = spec.param_hash()
        if h not in unique_specs:
            unique_specs[h] = spec
            hash_to_case_ids[h] = [spec.case_id]
            needs_2d_hash[h] = spec.calc_2d
        else:
            hash_to_case_ids[h].append(spec.case_id)
            if spec.calc_2d:
                needs_2d_hash[h] = True

    # Ensure any shared run needing 2D computation has calc_2d enabled
    for h, spec in unique_specs.items():
        if needs_2d_hash[h]:
            spec.calc_2d = True

    print(
        f"[{time.strftime('%X')}] Deduplicated {len(manifest)} cases into "
        f"{len(unique_specs)} distinct physical forward runs."
    )

    cache_dir = REPO_ROOT / "docs" / "moc_v2_technical_report" / ".sim_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results_by_hash: Dict[str, dict] = {}
    tasks_to_run: List[dict] = []

    for h, spec in unique_specs.items():
        cache_path = cache_dir / f"{h}.npz"
        loaded = False
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=True) as data:
                    has_sim = "timestamps" in data and "wellhead_head" in data
                    stct_tag = str(data["stct_tag"][0]) if "stct_tag" in data else ""
                    has_2d = (
                        "ceps2d_matrix" in data
                        and data["ceps2d_matrix"] is not None
                        and data["ceps2d_matrix"].size > 0
                        and stct_tag == CURRENT_STCT_TAG
                    )
                    if has_sim and ((not spec.calc_2d) or has_2d):
                        results_by_hash[h] = {
                            "case_id": spec.case_id,
                            "param_hash": h,
                            "timestamps": data["timestamps"],
                            "wellhead_head": data["wellhead_head"],
                            "ceps1d_distance": data["ceps1d_distance"],
                            "ceps1d_amp": data["ceps1d_amp"],
                            "ceps2d_time_centers": data["ceps2d_time_centers"] if has_2d else None,
                            "ceps2d_distances": data["ceps2d_distances"] if has_2d else None,
                            "ceps2d_matrix": data["ceps2d_matrix"] if has_2d else None,
                            "a_adj": float(np.squeeze(data["a_adj"])),
                            "t_elapsed": 0.0,
                        }
                        loaded = True
            except Exception:
                pass
        if not loaded:
            tasks_to_run.append(asdict(spec))

    print(
        f"[{time.strftime('%X')}] Cache status: {len(results_by_hash)} runs cached, "
        f"{len(tasks_to_run)} runs remaining to execute."
    )

    t_start = time.time()
    if tasks_to_run:
        actual_workers = min(len(tasks_to_run), max_workers)
        print(
            f"[{time.strftime('%X')}] Launching ProcessPoolExecutor with {actual_workers} worker processes..."
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_to_hash = {
                executor.submit(_worker_execute_sim, task_dict): task_dict["case_id"]
                for task_dict in tasks_to_run
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_hash):
                cid = future_to_hash[future]
                try:
                    res = future.result()
                    results_by_hash[res["param_hash"]] = res
                    completed += 1
                    print(
                        f"[{time.strftime('%X')}] [{completed}/{len(tasks_to_run)}] "
                        f"Run {cid} completed in {res['t_elapsed']:.1f}s | "
                        f"Head range: [{res['wellhead_head'].min():.1f}m, {res['wellhead_head'].max():.1f}m]"
                    )
                except Exception as e:
                    print(f"ERROR executing {cid}: {e}", file=sys.stderr)
                    raise e

    t_total = time.time() - t_start
    print(
        f"[{time.strftime('%X')}] All {len(unique_specs)} forward simulations available in "
        f"{t_total:.1f}s ({t_total / 60.0:.2f} min)!"
    )

    # Map results back to all 42 case IDs
    final_results: Dict[str, dict] = {}
    for spec in manifest:
        h = spec.param_hash()
        base_res = results_by_hash[h]
        final_results[spec.case_id] = {
            "spec": spec,
            "timestamps": base_res["timestamps"],
            "wellhead_head": base_res["wellhead_head"],
            "ceps1d_distance": base_res["ceps1d_distance"],
            "ceps1d_amp": base_res["ceps1d_amp"],
            "ceps2d_time_centers": base_res["ceps2d_time_centers"],
            "ceps2d_distances": base_res["ceps2d_distances"],
            "ceps2d_matrix": base_res["ceps2d_matrix"],
            "a_adj": base_res["a_adj"],
        }

    return final_results


# ==============================================================================
# 4. Nature-Grade Multi-Panel Scientific Figure Generator
# ==============================================================================

def render_topic_figure(
    topic_id: int,
    fig_title: str,
    case_specs: List[SimCaseSpec],
    results: Dict[str, dict],
    case_c_id: str,
    case_c_title: str,
    case_d_id: str,
    case_d_title: str,
    output_dir: Path,
    out_basename: str,
    dist_min: float = 4400.0,
    dist_max: float = 4650.0,
    extra_basenames: Optional[List[str]] = None,
):
    """
    Renders a unified Nature-grade 4-panel composite figure with faceted row subplots:
    - Left Column:
        - Panel a: 60s full-time waveform with inset zoom of early valve closure (0-15s).
        - Panel b: 1D real cepstrum depth spectra split row-by-row into stacked subplots
                   (1 strip per condition), sharing the horizontal depth axis, with dashed lines
                   marking true fracture positions.
    - Right Column:
        - Panel c: 2D continuous Rainbow cepstrogram (Case 1), X=Depth [m], Y=Time [s].
        - Panel d: 2D continuous Rainbow cepstrogram (Case 2), X=Depth [m], Y=Time [s].
    STRICTLY NO text detection criteria in panels c and d.
    Exports both 300 DPI PNG and vector SVG (<300 KB via rasterized mesh).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    n_cases = len(case_specs)
    if n_cases <= 5:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    else:
        colors = plt.cm.viridis(np.linspace(0.05, 0.88, n_cases))

    # Figure geometry: 12.0 inches wide by 9.6 inches tall
    fig = plt.figure(figsize=(12.0, 9.6))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.08, 1.0], wspace=0.25,
        left=0.07, right=0.96, top=0.94, bottom=0.06
    )

    gs_left = gs[0].subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.34)
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)

    # --- Left Upper: Panel a (Waveform 0-60s with Inset Zoom 0-15s) ---
    ax_a = fig.add_subplot(gs_left[0])
    for i, spec in enumerate(case_specs):
        res = results[spec.case_id]
        t = res["timestamps"]
        h = res["wellhead_head"]
        lw = 1.3 if spec.is_base_case else 1.0
        ax_a.plot(t, h, label=spec.label, color=colors[i], lw=lw, alpha=0.9)

    ax_a.axhline(300.0, color="#7f7f7f", ls=":", lw=0.8, alpha=0.7)
    ax_a.axhline(100.0, color="#b0b0b0", ls=":", lw=0.8, alpha=0.7)
    ax_a.set_xlim(0.0, 60.0)
    ax_a.set_ylim(50.0, 560.0)
    ax_a.set_xlabel("Time $t$ (s)")
    ax_a.set_ylabel(r"Wellhead Pressure Head $H_{\mathrm{wh}}$ (m)")
    ax_a.set_title("a  Wellhead Pressure Head Waveform (0–60 s)", loc="left", fontweight="bold", fontsize=8.5)
    ax_a.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=5.8,
        ncol=1 if n_cases <= 5 else 2,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#d0d0d0",
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.3,
    )

    # Inset zoom (0 to 15 s)
    ax_ins = inset_axes(
        ax_a,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.58, 0.64, 0.39, 0.32),
        bbox_transform=ax_a.transAxes,
    )
    ax_ins.patch.set_facecolor("white")
    ax_ins.patch.set_alpha(0.96)
    ax_ins.patch.set_edgecolor("#d0d0d0")
    for i, spec in enumerate(case_specs):
        res = results[spec.case_id]
        t = res["timestamps"]
        h = res["wellhead_head"]
        m_ins = t <= 15.0
        lw = 1.2 if spec.is_base_case else 0.9
        ax_ins.plot(t[m_ins], h[m_ins], color=colors[i], lw=lw, alpha=0.85)
    ax_ins.set_xlim(0.0, 15.0)
    ax_ins.set_xticks([0.0, 5.0, 10.0, 15.0])
    ax_ins.locator_params(axis="y", nbins=4)
    ax_ins.set_xlabel("Time $t$ (s)", fontsize=5.8, labelpad=1.5)
    ax_ins.set_ylabel(r"$H_{\mathrm{wh}}$ (m)", fontsize=5.8, labelpad=1.5)
    ax_ins.tick_params(axis="both", labelsize=5.5, pad=1.5)
    ax_ins.set_title("Early Closure Zoom (0–15 s)", fontsize=6.5, pad=2, fontweight="bold")

    # --- Left Lower: Panel b (1D Real Cepstrum Depth Spectrum - Row-by-Row Faceted Subplots) ---
    gs_b = gs_left[1].subgridspec(n_cases, 1, hspace=0.12)
    for i, spec in enumerate(case_specs):
        ax_b_i = fig.add_subplot(gs_b[i])
        res = results[spec.case_id]
        dist = res["ceps1d_distance"]
        amp = res["ceps1d_amp"]

        mask = (dist >= dist_min) & (dist <= dist_max)
        sub_dist = dist[mask]
        sub_amp = amp[mask]
        norm_amp = sub_amp / (np.max(np.abs(sub_amp)) + 1e-12)

        lw = 1.2 if spec.is_base_case else 0.9
        ax_b_i.plot(sub_dist, norm_amp, color=colors[i], lw=lw, alpha=0.9)
        ax_b_i.axhline(0.0, color="#b0b0b0", ls=":", lw=0.6, alpha=0.7)

        # Mark true fracture positions for this specific case
        for xf in spec.positions:
            if dist_min <= xf <= dist_max:
                ax_b_i.axvline(xf, color="#d9534f", ls="--", lw=0.9, alpha=0.75)

        # Label tag inside strip
        ax_b_i.text(
            0.015,
            0.74,
            spec.label,
            transform=ax_b_i.transAxes,
            fontsize=6.2,
            fontweight="bold",
            color=colors[i],
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d0d0d0", lw=0.5, alpha=0.92),
        )

        ax_b_i.set_xlim(dist_min, dist_max)
        ax_b_i.set_ylim(-1.10, 1.10)
        ax_b_i.set_yticks([])

        if i == 0:
            ax_b_i.set_title("b  1D Real Cepstrum Depth Spectrum (Faceted by Case)", loc="left", fontweight="bold", fontsize=8.5)

        if i == n_cases - 1:
            ax_b_i.set_xlabel("Acoustic Reflection Depth $x$ (m)", fontsize=8.0)
            ax_b_i.tick_params(labelbottom=True, labelsize=7.0)
        else:
            ax_b_i.tick_params(labelbottom=False)

    # --- Right Column: Panels c & d (2D Continuous Rainbow Cepstrograms, X=Depth, Y=Time) ---
    def plot_ceps2d_panel(ax, case_id: str, panel_label: str, panel_title: str):
        res = results[case_id]
        spec = res["spec"]
        time_centers = res["ceps2d_time_centers"]
        distances = res["ceps2d_distances"]
        matrix = res["ceps2d_matrix"]

        if matrix is None:
            # Fallback computation with Kaiser window (wlen=30.0s, hop=1.0s)
            ceps2d = compute_cepstrogram_2d(
                res["timestamps"],
                res["wellhead_head"],
                wavespeed=res["a_adj"],
                fs=1000.0,
                ts=1.0,
                window_len_s=30.0,
                hop_len_s=1.0,
                window="kaiser",
                max_distance=5000.0,
            )
            time_centers = ceps2d["time_centers"]
            distances = ceps2d["distances"]
            matrix = ceps2d["cepstrogram"]

        m_dist = (distances >= dist_min) & (distances <= dist_max)
        sub_dist = distances[m_dist]
        sub_matrix = matrix[:, m_dist]

        energy = np.abs(sub_matrix)
        v_min = np.percentile(energy, 2)
        v_max = np.percentile(energy, 98)
        if v_max <= v_min:
            v_max = v_min + 1e-6

        # Rasterized mesh with X=Depth, Y=Time, Rainbow colormap
        mesh = ax.pcolormesh(
            sub_dist,
            time_centers,
            energy,
            shading="auto",
            cmap="rainbow",
            vmin=v_min,
            vmax=v_max,
            rasterized=True,
        )

        # Vertical dashed lines at true fracture locations
        for xf in spec.positions:
            if dist_min <= xf <= dist_max:
                ax.axvline(xf, color="white", ls="--", lw=1.1, alpha=0.9)

        # STRICTLY NO TEXT DETECTION LABELS!
        ax.set_xlim(dist_min, dist_max)
        ax.set_ylim(time_centers[0], time_centers[-1])
        ax.set_xlabel("Acoustic Reflection Depth $x$ (m)", fontsize=8.0)
        ax.set_ylabel("Time Window Center $t$ (s)", fontsize=8.0)
        ax.set_title(f"{panel_label}  {panel_title}", loc="left", fontweight="bold", fontsize=8.0)

        cbar = plt.colorbar(mesh, ax=ax, pad=0.02, shrink=0.88)
        cbar.set_label("Cepstral Intensity", fontsize=7.0)
        cbar.ax.tick_params(labelsize=6.5)

    ax_c = fig.add_subplot(gs_right[0])
    ax_d = fig.add_subplot(gs_right[1])
    plot_ceps2d_panel(ax_c, case_c_id, "c", case_c_title)
    plot_ceps2d_panel(ax_d, case_d_id, "d", case_d_title)

    # Super-title with overall topic definition
    fig.suptitle(fig_title, fontsize=10.0, fontweight="bold", y=0.98)

    # Export both 300 DPI PNG and vector SVG
    out_png = output_dir / f"{out_basename}.png"
    out_svg = output_dir / f"{out_basename}.svg"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(
        f"[{time.strftime('%X')}] Saved {out_png.name} ({out_png.stat().st_size / 1024:.1f} KB) "
        f"and {out_svg.name} ({out_svg.stat().st_size / 1024:.1f} KB)"
    )

    # Export any alias filenames for compatibility
    if extra_basenames:
        import shutil
        for alias in extra_basenames:
            shutil.copyfile(out_png, output_dir / f"{alias}.png")
            shutil.copyfile(out_svg, output_dir / f"{alias}.svg")
            print(f"[{time.strftime('%X')}] Also created alias figure: {alias}.{{png,svg}}")


# ==============================================================================
# 5. Main Execution Orchestrator
# ==============================================================================

def main():
    print("======================================================================")
    print("  MOC_V2 7 SENSITIVITY STUDIES SIMULATION PIPELINE & FIGURE GENERATOR ")
    print("======================================================================")
    start_time = time.time()

    # Output directory
    output_dir = REPO_ROOT / "docs" / "moc_v2_technical_report" / "sensitivity_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target figure directory: {output_dir}")

    # Build manifest (42 cases)
    manifest = build_manifest()
    print(f"Assembled {len(manifest)} simulation specifications across Topics 1 to 7.")

    # Execute all simulations in parallel (memory-safe worker count)
    results = run_simulation_pipeline(manifest, max_workers=5)

    # Verification: Confirm all 42 simulations completed without NaN/Inf
    print("\n--- Numerical Verification of Simulation Results ---")
    verified_count = 0
    for spec in manifest:
        res = results[spec.case_id]
        wh = res["wellhead_head"]
        assert np.all(np.isfinite(wh)), f"NaN/Inf found in {spec.case_id}"
        verified_count += 1
    print(f"VERIFIED: All {verified_count}/{len(manifest)} simulation runs completed with ZERO NaN/Inf.\n")

    # ==========================================================================
    # Render Figures 1 to 7
    # ==========================================================================
    print("--- Generating Nature-Grade Figures 1 to 7 (PNG + SVG) ---")
    _setup_matplotlib()

    # --- Figure 1: Fracture Count Sensitivity (Topic 1) ---
    t1_specs = [s for s in manifest if s.topic_id == 1]
    render_topic_figure(
        topic_id=1,
        fig_title="Figure 1 | Parametric Sensitivity to Fracture Cluster Count ($N_c = 1 \\sim 8$)",
        case_specs=t1_specs,
        results=results,
        case_c_id="T1_Nc1",
        case_c_title="Single Cluster ($N_c=1, x_f=4500\\,\\mathrm{m}$)",
        case_d_id="T1_Nc8",
        case_d_title="Dense Array ($N_c=8, 4500\\sim 4570\\,\\mathrm{m}$)",
        output_dir=output_dir,
        out_basename="fig1_fracture_count_sensitivity",
        dist_min=4400.0,
        dist_max=4650.0,
    )

    # --- Figure 2: Fracture Spacing Sensitivity (Topic 2) ---
    t2_specs = [s for s in manifest if s.topic_id == 2]
    render_topic_figure(
        topic_id=2,
        fig_title="Figure 2 | Parametric Sensitivity to Inter-Cluster Spacing ($d = 5 \\sim 80\\,\\mathrm{m}$)",
        case_specs=t2_specs,
        results=results,
        case_c_id="T2_d10",
        case_c_title="Close Spacing ($d=10\\,\\mathrm{m}$, Strong Reverberation)",
        case_d_id="T2_d50",
        case_d_title="Wide Spacing ($d=50\\,\\mathrm{m}$, Isolated Arrivals)",
        output_dir=output_dir,
        out_basename="fig2_fracture_spacing_sensitivity",
        dist_min=4400.0,
        dist_max=4720.0,
    )

    # --- Figure 3: Fracture Compliance Sensitivity (Topic 3) ---
    t3_specs = [s for s in manifest if s.topic_id == 3]
    render_topic_figure(
        topic_id=3,
        fig_title="Figure 3 | Parametric Sensitivity to Fracture Compliance Capacity ($C_f = 0.002 \\sim 0.030\\,\\mathrm{m^2}$)",
        case_specs=t3_specs,
        results=results,
        case_c_id="T3_Cf0.002",
        case_c_title="Stiff Fracture ($C_f=0.002\\,\\mathrm{m^2}$, Fast Decay)",
        case_d_id="T3_Cf0.030",
        case_d_title="High Compliance ($C_f=0.030\\,\\mathrm{m^2}$, Large Rebound)",
        output_dir=output_dir,
        out_basename="fig3_compliance_sensitivity",
        dist_min=4400.0,
        dist_max=4650.0,
    )

    # --- Figure 4: Quasi-Darcy Leak-off Sensitivity (Topic 4) ---
    t4_specs = [s for s in manifest if s.topic_id == 4]
    render_topic_figure(
        topic_id=4,
        fig_title="Figure 4 | Parametric Sensitivity to Quasi-Darcy Leak-off ($k_{leak} = (0.2 \\sim 10.0)\\times 10^{-4}\\,\\mathrm{m^{2.5}/s}$)",
        case_specs=t4_specs,
        results=results,
        case_c_id="T4_kleak0.2",
        case_c_title="Tight Formation ($k_{leak}=0.2\\times 10^{-4}$, High Rebound)",
        case_d_id="T4_kleak10",
        case_d_title="Fault-Intersected ($k_{leak}=10.0\\times 10^{-4}$, Strong Drainage)",
        output_dir=output_dir,
        out_basename="fig4_leakoff_sensitivity",
        dist_min=4400.0,
        dist_max=4650.0,
    )

    # --- Figure 5: Perforation Flow Resistance Sensitivity (Topic 5) ---
    t5_specs = [s for s in manifest if s.topic_id == 5]
    render_topic_figure(
        topic_id=5,
        fig_title="Figure 5 | Parametric Sensitivity to Perforation Throttling Impedance ($K_p = (1.5 \\sim 25.0)\\times 10^5\\,\\mathrm{s^2/m^5}$)",
        case_specs=t5_specs,
        results=results,
        case_c_id="T5_Kp1.5",
        case_c_title="16 Holes ($K_p=1.5\\times 10^5$, Acoustic Short-Circuit)",
        case_d_id="T5_Kp25",
        case_d_title="2 Holes ($K_p=25.0\\times 10^5$, Severe Choking)",
        output_dir=output_dir,
        out_basename="fig5_perforation_sensitivity",
        extra_basenames=["fig5_perforation_impedance_sensitivity"],
        dist_min=4400.0,
        dist_max=4650.0,
    )

    # --- Figure 6: Pump Shutoff Ramp Duration Sensitivity (Topic 6) ---
    t6_specs = [s for s in manifest if s.topic_id == 6]
    render_topic_figure(
        topic_id=6,
        fig_title="Figure 6 | Parametric Sensitivity to Valve Shutoff Ramp Duration ($t_c = 0.0 \\sim 2.0\\,\\mathrm{s}$)",
        case_specs=t6_specs,
        results=results,
        case_c_id="T6_tc0.1",
        case_c_title="Base Fast Closure ($t_c=0.1\\,\\mathrm{s}$, Wideband High-Resolution)",
        case_d_id="T6_tc2",
        case_d_title="Slow Cosine Ramp ($t_c=2.0\\,\\mathrm{s}$, Low-Pass Filtered)",
        output_dir=output_dir,
        out_basename="fig6_ramp_closure_sensitivity",
        extra_basenames=["fig6_pump_shutoff_ramp_sensitivity"],
        dist_min=4400.0,
        dist_max=4650.0,
    )

    # --- Figure 7: Multi-Cluster Intake Capacity Combinations (Topic 7) ---
    t7_specs = [s for s in manifest if s.topic_id == 7]
    render_topic_figure(
        topic_id=7,
        fig_title="Figure 7 | Acoustic Diagnostics of 5 Typical Multi-Cluster Inflow Capacity Regimes",
        case_specs=t7_specs,
        results=results,
        case_c_id="T7_Case_7.1",
        case_c_title="Case 7.1 Uniform Baseline [Med, Med, Med]",
        case_d_id="T7_Case_7.5",
        case_d_title="Case 7.5 Sand-Plugged Heel [Dead, Med, High]",
        output_dir=output_dir,
        out_basename="fig7_intake_capacity_combinations",
        dist_min=4400.0,
        dist_max=4650.0,
    )

    # ==========================================================================
    # Deliverable Verification
    # ==========================================================================
    print("\n--- Final Deliverable Audit ---")
    required_figures = [
        "fig1_fracture_count_sensitivity.png",
        "fig1_fracture_count_sensitivity.svg",
        "fig2_fracture_spacing_sensitivity.png",
        "fig2_fracture_spacing_sensitivity.svg",
        "fig3_compliance_sensitivity.png",
        "fig3_compliance_sensitivity.svg",
        "fig4_leakoff_sensitivity.png",
        "fig4_leakoff_sensitivity.svg",
        "fig5_perforation_sensitivity.png",
        "fig5_perforation_sensitivity.svg",
        "fig6_ramp_closure_sensitivity.png",
        "fig6_ramp_closure_sensitivity.svg",
        "fig7_intake_capacity_combinations.png",
        "fig7_intake_capacity_combinations.svg",
    ]

    all_exist = True
    for fname in required_figures:
        fpath = output_dir / fname
        if not fpath.exists():
            print(f"FAILED: Missing deliverable figure: {fpath}", file=sys.stderr)
            all_exist = False
        else:
            sz = fpath.stat().st_size
            if sz == 0:
                print(f"FAILED: Zero-size figure file: {fpath}", file=sys.stderr)
                all_exist = False
            else:
                ext = fname.split(".")[-1].upper()
                print(f"  [OK] {fname:42s} | Size: {sz / 1024:6.1f} KB ({ext})")

    assert all_exist, "One or more deliverable figures are missing or empty!"

    elapsed_total = time.time() - start_time
    print(f"\nSUCCESS: Pipeline executed in {elapsed_total:.1f}s ({elapsed_total / 60.0:.2f} min).")
    print(f"All {len(manifest)} simulations completed with ZERO NaN/Inf and all 14 figures successfully created.")


if __name__ == "__main__":
    main()
