# -*- coding: utf-8 -*-
"""Blind peak detection, one-to-one matching, calibration, and oracle checks."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from neural_operator.dccdm_pipeline import write_json
from .config import DataConfig, DetectorConfig, dataclass_from_dict
from .data import DirectInverseDataset, load_manifest, spacing_band, time_to_depth_m


def detect_events(
    probability: np.ndarray,
    valid_mask: np.ndarray,
    time_axis: np.ndarray,
    threshold: float,
    data_config: DataConfig,
    detector_config: DetectorConfig,
    target_count: int | None = None,
) -> List[Dict]:
    probability = np.asarray(probability, dtype=float).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    time_axis = np.asarray(time_axis, dtype=float).reshape(-1)
    if not (len(probability) == len(valid_mask) == len(time_axis)):
        raise ValueError("probability, mask, and time axis lengths differ")
    masked = np.where(valid_mask, probability, 0.0)
    dt = float(np.median(np.diff(time_axis)))
    bin_depth = data_config.wavespeed_m_s * dt / 2.0
    minimum_distance = max(1, int(np.ceil(detector_config.minimum_separation_m / bin_depth)))
    search_height = threshold
    if target_count is not None and target_count > 0:
        # Allow weaker candidates so the count head can request missing peaks.
        search_height = min(threshold, max(0.05, 0.5 * threshold))
    peaks, properties = find_peaks(
        masked,
        height=search_height,
        prominence=detector_config.prominence,
        distance=minimum_distance,
    )
    events = [
        {
            "index": int(index),
            "time_s": float(time_axis[index]),
            "depth_m": float(time_to_depth_m(time_axis[index], data_config)),
            "probability": float(masked[index]),
            "prominence": float(properties["prominences"][position]),
        }
        for position, index in enumerate(peaks)
    ]
    if target_count is None or target_count <= 0:
        return events
    events = sorted(events, key=lambda event: event["probability"], reverse=True)
    return sorted(events[: int(target_count)], key=lambda event: event["index"])


def match_events(detected: Sequence[Dict], true_depths: np.ndarray, tolerance_m: float) -> Dict:
    true_depths = np.asarray(true_depths, dtype=float)
    detected_depths = np.asarray([event["depth_m"] for event in detected], dtype=float)
    pairs = []
    if len(true_depths) and len(detected_depths):
        cost = np.abs(true_depths[:, None] - detected_depths[None, :])
        rows, cols = linear_sum_assignment(cost)
        pairs = [
            (int(row), int(col), float(cost[row, col]))
            for row, col in zip(rows, cols)
            if cost[row, col] <= tolerance_m
        ]
    tp = len(pairs)
    fp = len(detected) - tp
    fn = len(true_depths) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = [pair[2] for pair in pairs]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "depth_errors_m": errors,
        "matches": [
            {
                "true_index": row,
                "detected_index": col,
                "depth_error_m": error,
            }
            for row, col, error in pairs
        ],
    }


def _aggregate_event_totals(totals: Dict[str, int]) -> Dict:
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def summarize_case_rows(rows: Sequence[Dict]) -> Dict:
    physical_totals = {"tp": 0, "fp": 0, "fn": 0}
    grid_totals = {"tp": 0, "fp": 0, "fn": 0}
    all_errors: List[float] = []
    count_errors: List[int] = []
    exact_counts = 0
    for row in rows:
        for key in physical_totals:
            physical_totals[key] += row["physical"][key]
            grid_totals[key] += row["grid"][key]
        all_errors.extend(row["physical"]["depth_errors_m"])
        count_errors.append(int(row["count_absolute_error"]))
        exact_counts += int(row["predicted_count"] == row["n_frac"])
    return {
        "n_cases": len(rows),
        "physical": _aggregate_event_totals(physical_totals),
        "grid": _aggregate_event_totals(grid_totals),
        "exact_count_accuracy": exact_counts / max(len(rows), 1),
        "exact_count_cases": exact_counts,
        "count_mae": float(np.mean(count_errors)) if count_errors else None,
        "median_depth_error_m": float(np.median(all_errors)) if all_errors else None,
        "p95_depth_error_m": float(np.percentile(all_errors, 95)) if all_errors else None,
    }


def stratify_metrics(rows: Sequence[Dict]) -> Dict:
    by_n_frac: Dict[str, List[Dict]] = {}
    by_spacing_band: Dict[str, List[Dict]] = {}
    by_n_and_band: Dict[str, List[Dict]] = {}
    for row in rows:
        n_key = f"n{row['n_frac']}"
        band = row.get("spacing_band") or spacing_band(int(row["n_frac"]), float(row["min_spacing_m"]))
        combo = f"{n_key}:{band}"
        by_n_frac.setdefault(n_key, []).append(row)
        by_spacing_band.setdefault(band, []).append(row)
        by_n_and_band.setdefault(combo, []).append(row)
    return {
        "by_n_frac": {key: summarize_case_rows(group) for key, group in sorted(by_n_frac.items())},
        "by_spacing_band": {
            key: summarize_case_rows(group) for key, group in sorted(by_spacing_band.items())
        },
        "by_n_frac_and_spacing_band": {
            key: summarize_case_rows(group) for key, group in sorted(by_n_and_band.items())
        },
    }


def evaluate_bundle(
    bundle: Dict[str, np.ndarray],
    threshold: float,
    data_config: DataConfig,
    detector_config: DetectorConfig,
    use_count_prior: bool = True,
) -> Dict:
    rows = []
    dt = float(np.median(np.diff(bundle["time_axis"][0])))
    grid_tolerance_m = detector_config.grid_tolerance_bins * data_config.wavespeed_m_s * dt / 2.0
    has_count_prior = use_count_prior and "predicted_count" in bundle
    for index, case_id in enumerate(bundle["case_id"]):
        count = int(bundle["n_frac"][index])
        true_depths = bundle["x_f"][index, :count]
        target_count = None
        if has_count_prior:
            prior = int(bundle["predicted_count"][index])
            if prior > 0:
                target_count = prior
        detected = detect_events(
            bundle["probability"][index],
            bundle["valid_time_mask"][index],
            bundle["time_axis"][index],
            threshold,
            data_config,
            detector_config,
            target_count=target_count,
        )
        physical = match_events(detected, true_depths, detector_config.physical_tolerance_m)
        grid = match_events(detected, true_depths, grid_tolerance_m)
        predicted_count = len(detected)
        spacing = float(bundle["min_spacing_m"][index])
        band = spacing_band(count, spacing, data_config.spacing_regime)
        rows.append({
            "case_id": str(case_id),
            "n_frac": count,
            "min_spacing_m": spacing,
            "spacing_band": band,
            "predicted_count": predicted_count,
            "count_head_prior": target_count,
            "count_absolute_error": abs(predicted_count - count),
            "physical": physical,
            "grid": grid,
            "detected": detected,
        })

    summary = summarize_case_rows(rows)
    return {
        "threshold": threshold,
        **summary,
        "strata": stratify_metrics(rows),
        "per_case": rows,
    }


def calibrate_threshold(
    bundle: Dict[str, np.ndarray],
    data_config: DataConfig,
    detector_config: DetectorConfig,
) -> Dict:
    candidates = []
    for threshold in detector_config.threshold_grid:
        metrics = evaluate_bundle(bundle, threshold, data_config, detector_config)
        candidates.append(metrics)
    best = max(
        candidates,
        key=lambda item: (
            item["physical"]["f1"],
            item["physical"]["precision"],
            item["threshold"],
        ),
    )
    return {"threshold": best["threshold"], "validation_metrics": best, "candidates": candidates}


def oracle_bundle(dataset: DirectInverseDataset) -> Dict[str, np.ndarray]:
    case_ids = []
    probabilities = []
    targets = []
    masks = []
    times = []
    depths = []
    n_frac = []
    spacings = []
    for index in range(len(dataset)):
        sample = dataset[index]
        case_ids.append(sample["case_id"])
        target = sample["event_target"].numpy()
        probabilities.append(target)
        targets.append(target)
        masks.append(sample["valid_time_mask"].numpy())
        times.append(sample["time_axis"].numpy())
        depths.append(sample["x_f_m"].numpy())
        n_frac.append(sample["n_frac"].numpy())
        spacings.append(sample["min_spacing_m"].numpy())
    return {
        "case_id": np.asarray(case_ids),
        "probability": np.stack(probabilities),
        "event_target": np.stack(targets),
        "valid_time_mask": np.stack(masks),
        "time_axis": np.stack(times),
        "x_f": np.stack(depths),
        "n_frac": np.asarray(n_frac),
        "min_spacing_m": np.asarray(spacings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate direct inverse artifacts or oracle targets")
    parser.add_argument("--mode", choices=("oracle", "artifact"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", default="val128")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    data_config = dataclass_from_dict(DataConfig, manifest["data_config"])
    if manifest.get("detector_config"):
        detector_config = dataclass_from_dict(DetectorConfig, manifest["detector_config"])
    else:
        detector_config = DetectorConfig()
    if args.mode == "oracle":
        bundle = oracle_bundle(DirectInverseDataset(args.manifest, args.selection))
    else:
        if not args.artifact:
            raise ValueError("artifact mode requires --artifact")
        loaded = np.load(args.artifact, allow_pickle=False)
        bundle = {key: loaded[key] for key in loaded.files}
    metrics = evaluate_bundle(bundle, args.threshold, data_config, detector_config)
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_case"}, indent=2))


if __name__ == "__main__":
    main()
