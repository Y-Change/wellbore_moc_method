"""Core validation for the Paper A design-constrained regularizer.

The script uses the steady Paper A grid for calibration/geometry transfer and
generates a small negative-control matrix with inactive or weak designed
clusters.  Active truth is used only after profile correction for scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "moc_simulate").is_dir():
            return parent
    raise RuntimeError("Repository root not found")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.unified_evaluation.detection_protocol import (  # noqa: E402
    DetectorConfig,
    detect_peaks,
    score_detections,
    tolerance_sweep,
)
from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum  # noqa: E402
from moc_simulate.config import FRACTURE_CONFIG, SIM_CONFIG, WELL_CONFIG  # noqa: E402
from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore  # noqa: E402

from design_constrained_regularization import DesignConstrainedRegularizer  # noqa: E402


PAPERA = REPO_ROOT / "PaperA井口多裂缝水击响应"
GRID_ROOT = PAPERA / "01_几何网格"
GRID_NPZ = next(p for p in GRID_ROOT.iterdir() if p.is_dir() and "npz" in p.name.lower())
OUT_ROOT = PAPERA / "文章撰写" / "稿件" / "data" / "design_regularization"
NEG_ROOT = OUT_ROOT / "negative_controls" / "waveforms"
FIG_ROOT = PAPERA / "文章撰写" / "稿件" / "图表数据"

X1_CAL = 3000.0
X1_TEST = (2000.0, 2500.0, 3500.0, 4000.0, 4500.0)
S_GRID = tuple(float(v) for v in range(10, 101, 10))
N_GRID = tuple(range(3, 9))
METHODS = ("raw", "design_only", "amplitude_only", "focus_only", "full")


def _npz_name(x1: float, spacing: float, n: int) -> str:
    return f"steady_x1_{int(x1)}_sp_{int(spacing)}_n_{n}.npz"


def _profile_from_npz(path: Path) -> Dict[str, object]:
    with np.load(path) as data:
        t = np.asarray(data["t_sim"], dtype=float)
        h = np.asarray(data["H_wh"], dtype=float)
        v = float(np.atleast_1d(data["v"])[0])
        fs = float(np.atleast_1d(data["fs"])[0])
        ts = float(np.atleast_1d(data["ts"])[0])
        length = float(np.atleast_1d(data["L"])[0])
        if "x_perf_aligned" in data:
            x_perf = np.asarray(data["x_perf_aligned"], dtype=float)
        else:
            x_perf = np.asarray(data["x_f_aligned"], dtype=float)
        active_mask = (
            np.asarray(data["active_mask"], dtype=bool)
            if "active_mask" in data
            else np.ones(x_perf.size, dtype=bool)
        )
        weak_index = int(np.atleast_1d(data["weak_index"])[0]) if "weak_index" in data else -1
        pattern = str(np.atleast_1d(data["pattern"])[0]) if "pattern" in data else "all_active"

    out = compute_moc_cepstrum(
        t,
        h,
        v,
        fs=fs,
        ts=ts,
        wellbore_length=length,
        wlen_sec=30.0,
        hop_sec=5.0,
        win_type="hamming",
    )
    depth = np.asarray(out["depth"], dtype=float)
    profile = -np.sum(np.asarray(out["C"], dtype=float), axis=1)
    return {
        "depth": depth,
        "raw_profile": profile,
        "perforation_positions": x_perf,
        "active_positions": x_perf[active_mask],
        "active_mask": active_mask,
        "weak_index": weak_index,
        "pattern": pattern,
    }


def _build_config(required_length: float) -> MocConfig:
    w, s = WELL_CONFIG, SIM_CONFIG
    length = max(float(w["L"]), required_length)
    tf = max(float(s["tf"]), 2.0 * length / float(w["wavespeed"]) + float(s["ts"]))
    return MocConfig(
        wellbore_length=length,
        wellbore_diameter=float(w["wellbore_diameter"]),
        fluid_density=float(w["fluid_density"]),
        fluid_viscosity=float(w["fluid_viscosity"]),
        wavespeed=float(w["wavespeed"]),
        roughness_height=float(w["roughness_height"]),
        friction_model="steady",
        dt=float(s["dt"]),
        tf=tf,
        wellhead_bc="velocity_step",
        pump_shut_time=float(s["ts"]),
        initial_velocity=float(w["V0"]),
        initial_head=float(w["H0"]),
        theta=float(w["theta"]),
        toe_bc="reservoir",
        toe_head=float(w["H0"]),
    )


def _negative_case_specs() -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for x1 in (2500.0, 4000.0):
        for spacing in (10.0, 30.0, 80.0):
            for n in (4, 8):
                middle = n // 2
                for pattern, altered in (
                    ("middle_inactive", middle),
                    ("tail_inactive", n - 1),
                    ("middle_weak", middle),
                ):
                    specs.append({
                        "x1": x1,
                        "spacing": spacing,
                        "n": n,
                        "pattern": pattern,
                        "altered_index": altered,
                    })
    return specs


def _negative_path(spec: Dict[str, object]) -> Path:
    return NEG_ROOT / (
        f"steady_x1_{int(spec['x1'])}_sp_{int(spec['spacing'])}_"
        f"n_{int(spec['n'])}_{spec['pattern']}.npz"
    )


def _simulate_negative_case(spec: Dict[str, object]) -> str:
    path = _negative_path(spec)
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x1 = float(spec["x1"])
    spacing = float(spec["spacing"])
    n = int(spec["n"])
    pattern = str(spec["pattern"])
    altered = int(spec["altered_index"])
    x_perf = np.asarray([x1 + i * spacing for i in range(n)], dtype=float)
    cf = np.full(n, float(FRACTURE_CONFIG["Cf"]), dtype=float)
    kleak = np.full(n, float(FRACTURE_CONFIG["kleak"]), dtype=float)
    active_mask = np.ones(n, dtype=bool)
    weak_index = -1
    if "inactive" in pattern:
        cf[altered] = 0.0
        kleak[altered] = 0.0
        active_mask[altered] = False
    else:
        cf[altered] *= 0.25
        kleak[altered] *= 0.25
        weak_index = altered

    cfg = _build_config(float(x_perf.max()) + 500.0)
    res = simulate_wellbore(
        cfg,
        fracture_positions=x_perf.tolist(),
        fracture_Cf=cf.tolist(),
        fracture_kleak=kleak.tolist(),
        H_ext=float(FRACTURE_CONFIG["H_ext"]),
        store_full_field=False,
    )
    aligned = np.asarray([res["x_grid"][i] for i in res["fracture_indices"]], dtype=float)
    np.savez_compressed(
        path,
        t_sim=res["timestamps"],
        H_wh=res["wellhead_head"],
        x_perf_aligned=aligned,
        active_mask=active_mask,
        weak_index=np.asarray([weak_index], dtype=int),
        pattern=np.asarray([pattern]),
        Cf=cf,
        kleak=kleak,
        v=np.asarray([cfg.a_adj]),
        L=np.asarray([cfg.wellbore_length]),
        fs=np.asarray([1.0 / cfg.dt_adj]),
        ts=np.asarray([cfg.pump_shut_time]),
    )
    return str(path)


def generate_negative_controls(max_workers: int) -> None:
    specs = _negative_case_specs()
    missing = [s for s in specs if not _negative_path(s).exists()]
    if not missing:
        print("[negative controls] all 36 waveforms already exist")
        return
    print(f"[negative controls] generating {len(missing)} waveforms with {max_workers} workers")
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_simulate_negative_case, spec) for spec in missing]
        for future in as_completed(futures):
            future.result()
            done += 1
            print(f"  {done}/{len(missing)}")


def _detector_for_case(x_perf: np.ndarray, spacing: float) -> DetectorConfig:
    margin = max(30.0, spacing)
    return DetectorConfig(
        min_separation_m=5.0,
        height_pct=85.0,
        height_rel=0.03,
        height_abs=0.0,
        prominence_rel=0.0,
        max_peaks=10,
        search_min_m=float(x_perf.min() - margin),
        search_max_m=float(x_perf.max() + margin),
    )


def _outside_energy_ratio(depth: np.ndarray, profile: np.ndarray, x_perf: np.ndarray) -> float:
    inside = np.zeros(depth.size, dtype=bool)
    for position in x_perf:
        inside |= np.abs(depth - position) <= 10.0
    energy = np.square(np.asarray(profile, dtype=float))
    total = float(np.sum(energy))
    return float(np.sum(energy[~inside]) / total) if total > 0 else 0.0


def _score_profile(
    method: str,
    depth: np.ndarray,
    profile: np.ndarray,
    x_perf: np.ndarray,
    active_truth: np.ndarray,
    spacing: float,
    inactive_positions: np.ndarray,
    weak_position: float | None,
) -> Dict[str, object]:
    cfg = _detector_for_case(x_perf, spacing)
    detected = detect_peaks(depth, profile, cfg)
    predicted = [float(p["depth_m"]) for p in detected]
    primary = score_detections(predicted, active_truth, 10.0)
    inactive_hits = 0
    for position in inactive_positions:
        if any(abs(position - pred) <= 10.0 for pred in predicted):
            inactive_hits += 1
    weak_hit = None
    if weak_position is not None:
        weak_hit = any(abs(weak_position - pred) <= 10.0 for pred in predicted)
    return {
        "method": method,
        "n_pred": primary["n_pred"],
        "tp": primary["tp"],
        "fp": primary["fp"],
        "fn": primary["fn"],
        "precision": primary["precision"],
        "recall": primary["recall"],
        "f1_10m": primary["f1"],
        "separation_success": primary["separation_success"],
        "count_error": primary["count_error"],
        "median_error_m": primary["median_error_m"],
        "inactive_hits": inactive_hits,
        "n_inactive": int(inactive_positions.size),
        "weak_hit": weak_hit,
        "outside_energy_ratio": _outside_energy_ratio(depth, profile, x_perf),
        "detected_depths": json.dumps(predicted),
        "tolerance_sweep": json.dumps(tolerance_sweep(predicted, active_truth)),
    }


def _case_profiles(result: Dict[str, object]) -> Dict[str, np.ndarray]:
    return {
        "raw": np.asarray(result["raw_profile"], dtype=float),
        "design_only": np.asarray(result["design_only_profile"], dtype=float),
        "amplitude_only": np.asarray(result["amplitude_only_profile"], dtype=float),
        "focus_only": np.asarray(result["focus_only_profile"], dtype=float),
        "full": np.asarray(result["regularized_profile"], dtype=float),
    }


def _spacing_band(spacing: float) -> str:
    if spacing == 10.0:
        return "S=10"
    if spacing <= 40.0:
        return "S=20-40"
    return "S=50-100"


def _evaluate_case(
    regularizer: DesignConstrainedRegularizer,
    profile_case: Dict[str, object],
    case_id: str,
    x1: float,
    spacing: float,
    n: int,
    source: str,
) -> List[Dict[str, object]]:
    depth = np.asarray(profile_case["depth"], dtype=float)
    raw = np.asarray(profile_case["raw_profile"], dtype=float)
    x_perf = np.asarray(profile_case["perforation_positions"], dtype=float)
    active = np.asarray(profile_case["active_positions"], dtype=float)
    active_mask = np.asarray(profile_case["active_mask"], dtype=bool)
    inactive = x_perf[~active_mask]
    weak_index = int(profile_case["weak_index"])
    weak_position = float(x_perf[weak_index]) if weak_index >= 0 else None
    result = regularizer.apply_correction(depth, raw, x_perf, spacing)
    rows = []
    for method, profile in _case_profiles(result).items():
        row = _score_profile(
            method, depth, profile, x_perf, active, spacing, inactive, weak_position
        )
        row.update({
            "case_id": case_id,
            "source": source,
            "pattern": str(profile_case["pattern"]),
            "x1": x1,
            "spacing_m": spacing,
            "spacing_band": _spacing_band(spacing),
            "n_design": n,
            "n_active": int(active.size),
        })
        rows.append(row)
    return rows


def _load_calibration_cases() -> List[Dict[str, object]]:
    cases = []
    reference = _profile_from_npz(GRID_NPZ / _npz_name(X1_CAL, 20.0, 1))
    reference["spacing_m"] = 20.0
    cases.append(reference)
    for spacing in S_GRID:
        for n in N_GRID:
            case = _profile_from_npz(GRID_NPZ / _npz_name(X1_CAL, spacing, n))
            case["spacing_m"] = spacing
            cases.append(case)
    return cases


def _paired_bootstrap_delta(rows: pd.DataFrame, baseline: str, n_boot: int = 4000) -> Dict[str, float]:
    pivot = rows.pivot_table(index="case_id", columns="method", values="f1_10m", aggfunc="first")
    pivot = pivot.dropna(subset=["full", baseline])
    delta = (pivot["full"] - pivot[baseline]).to_numpy(dtype=float)
    if delta.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(20260831)
    samples = rng.choice(delta, size=(n_boot, delta.size), replace=True).mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def _summarize(rows: pd.DataFrame, calibration: object) -> Dict[str, object]:
    test = rows[rows["source"] == "geometry_transfer"]
    neg = rows[rows["source"] == "negative_control"]
    method_summary = {}
    for method, group in rows.groupby("method"):
        errors = pd.to_numeric(group["median_error_m"], errors="coerce")
        method_summary[method] = {
            "n_rows": int(len(group)),
            "mean_f1_10m": float(group["f1_10m"].mean()),
            "separation_rate": float(group["separation_success"].mean()),
            "count_mae": float(group["count_error"].abs().mean()),
            "median_localization_error_m": float(errors.median()) if errors.notna().any() else None,
            "median_outside_energy_ratio": float(group["outside_energy_ratio"].median()),
        }

    baseline_means = test[test["method"].isin(["raw", "design_only"])].groupby("method")["f1_10m"].mean()
    best_baseline = str(baseline_means.idxmax())
    bootstrap = _paired_bootstrap_delta(test, best_baseline)

    neg_full = neg[neg["method"] == "full"]
    inactive_den = int(neg_full["n_inactive"].sum())
    inactive_rate = float(neg_full["inactive_hits"].sum() / inactive_den) if inactive_den else 0.0
    raw_error = pd.to_numeric(test[test["method"] == "raw"]["median_error_m"], errors="coerce").median()
    full_error = pd.to_numeric(test[test["method"] == "full"]["median_error_m"], errors="coerce").median()
    localization_delta = float(full_error - raw_error) if np.isfinite(raw_error) and np.isfinite(full_error) else None
    band_means = test.groupby(["spacing_band", "method"])["f1_10m"].mean().unstack()
    improved_bands = 0
    for _, record in band_means.iterrows():
        best = max(float(record.get("raw", -np.inf)), float(record.get("design_only", -np.inf)))
        if float(record.get("full", -np.inf)) > best:
            improved_bands += 1

    ablation_means = test.groupby("method")["f1_10m"].mean()
    full_mean = float(ablation_means.get("full", np.nan))
    critical_ablation_better = all(
        full_mean > float(ablation_means.get(method, np.inf))
        for method in ("amplitude_only", "focus_only")
    )
    gate_checks = {
        "f1_gain_at_least_0.05": bool(bootstrap["mean"] >= 0.05),
        "paired_bootstrap_ci_low_above_zero": bool(bootstrap["ci_low"] > 0.0),
        "inactive_false_activation_at_most_0.05": bool(inactive_rate <= 0.05),
        "median_localization_degradation_at_most_1m": bool(
            localization_delta is not None and localization_delta <= 1.0
        ),
        "improves_at_least_two_spacing_bands": bool(improved_bands >= 2),
        "full_better_than_key_ablations": bool(critical_ablation_better),
    }
    return {
        "calibration": {
            "gamma_by_spacing": calibration.gamma_by_spacing,
            "gain_low": calibration.gain_low,
            "gain_high": calibration.gain_high,
            "focus_fwhm_m": calibration.focus_fwhm_m,
        },
        "n_geometry_transfer_cases": int(test["case_id"].nunique()),
        "n_negative_control_cases": int(neg["case_id"].nunique()),
        "method_summary": method_summary,
        "best_baseline": best_baseline,
        "paired_f1_delta_vs_best_baseline": bootstrap,
        "inactive_false_activation_rate": inactive_rate,
        "median_localization_delta_full_minus_raw_m": localization_delta,
        "improved_spacing_bands": improved_bands,
        "gate_checks": gate_checks,
        "promote_to_main_method": bool(all(gate_checks.values())),
    }


def _plot_summary(rows: pd.DataFrame, summary: Dict[str, object]) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.linewidth": 0.75,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    colors = {
        "raw": "#7F8C8D",
        "design_only": "#1B4F72",
        "amplitude_only": "#D35400",
        "focus_only": "#6C3483",
        "full": "#117864",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    test = rows[rows["source"] == "geometry_transfer"]
    means = test.groupby("method")["f1_10m"].mean().reindex(METHODS)
    axes[0].bar(np.arange(len(METHODS)), means, color=[colors[m] for m in METHODS])
    axes[0].set_xticks(np.arange(len(METHODS)))
    axes[0].set_xticklabels(["Raw", "Design", "Amp.", "Focus", "Full"], rotation=30, ha="right")
    axes[0].set_ylabel("Mean F1 @ 10 m")
    axes[0].set_title("(a) Geometry transfer")

    band = test.groupby(["spacing_band", "method"])["f1_10m"].mean().unstack()
    order = ["S=10", "S=20-40", "S=50-100"]
    x = np.arange(len(order))
    display_names = {"raw": "Raw", "design_only": "Design only", "full": "Full"}
    for method in ("raw", "design_only", "full"):
        axes[1].plot(
            x,
            band.reindex(order)[method],
            marker="o",
            label=display_names[method],
            color=colors[method],
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["10", "20-40", "50-100"])
    axes[1].set_xlabel("Spacing S (m)")
    axes[1].set_ylabel("Mean F1 @ 10 m")
    axes[1].set_title("(b) Spacing strata")
    axes[1].legend(frameon=False, fontsize=6.5)

    neg = rows[(rows["source"] == "negative_control") & (rows["method"].isin(["raw", "design_only", "full"]))]
    inactive = neg.groupby("method").apply(
        lambda g: g["inactive_hits"].sum() / max(g["n_inactive"].sum(), 1),
        include_groups=False,
    ).reindex(["raw", "design_only", "full"])
    axes[2].bar(np.arange(3), inactive, color=[colors[m] for m in ["raw", "design_only", "full"]])
    axes[2].set_xticks(np.arange(3))
    axes[2].set_xticklabels(["Raw", "Design", "Full"])
    axes[2].set_ylabel("Inactive-cluster activation rate")
    axes[2].set_title("(c) Negative controls")
    fig.suptitle("Core validation of design-constrained regularization", fontsize=9.0)

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    stem = FIG_ROOT / "Figure_Design_Regularization_Core_Validation"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--skip-negative-generation", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.skip_negative_generation:
        generate_negative_controls(args.workers)

    print("[1/4] calibration profiles")
    calibration_cases = _load_calibration_cases()
    regularizer = DesignConstrainedRegularizer()
    calibration = regularizer.calibrate(calibration_cases)
    print(calibration)

    print("[2/4] geometry-transfer validation")
    rows: List[Dict[str, object]] = []
    count = 0
    for x1 in X1_TEST:
        for spacing in S_GRID:
            for n in N_GRID:
                path = GRID_NPZ / _npz_name(x1, spacing, n)
                case = _profile_from_npz(path)
                case_id = f"grid_x1_{int(x1)}_S_{int(spacing)}_n_{n}"
                rows.extend(_evaluate_case(regularizer, case, case_id, x1, spacing, n, "geometry_transfer"))
                count += 1
                if count % 50 == 0:
                    print(f"  {count}/300")

    print("[3/4] negative controls")
    for spec in _negative_case_specs():
        path = _negative_path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Negative-control waveform missing: {path}")
        case = _profile_from_npz(path)
        case_id = path.stem
        rows.extend(_evaluate_case(
            regularizer,
            case,
            case_id,
            float(spec["x1"]),
            float(spec["spacing"]),
            int(spec["n"]),
            "negative_control",
        ))

    print("[4/4] summaries and promotion gate")
    df = pd.DataFrame(rows)
    summary = _summarize(df, calibration)
    df.to_csv(OUT_ROOT / "validation_case_metrics.csv", index=False)
    with (OUT_ROOT / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _plot_summary(df, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
