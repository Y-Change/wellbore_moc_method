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
from .config import DataConfig, DetectorConfig
from .data import DirectInverseDataset, load_manifest, time_to_depth_m


def detect_events(
    probability: np.ndarray,
    valid_mask: np.ndarray,
    time_axis: np.ndarray,
    threshold: float,
    data_config: DataConfig,
    detector_config: DetectorConfig,
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
    peaks, properties = find_peaks(
        masked,
        height=threshold,
        prominence=detector_config.prominence,
        distance=minimum_distance,
    )
    return [
        {
            "index": int(index),
            "time_s": float(time_axis[index]),
            "depth_m": float(time_to_depth_m(time_axis[index], data_config)),
            "probability": float(masked[index]),
            "prominence": float(properties["prominences"][position]),
        }
        for position, index in enumerate(peaks)
    ]


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


def evaluate_bundle(
    bundle: Dict[str, np.ndarray],
    threshold: float,
    data_config: DataConfig,
    detector_config: DetectorConfig,
) -> Dict:
    rows = []
    physical_totals = {"tp": 0, "fp": 0, "fn": 0}
    grid_totals = {"tp": 0, "fp": 0, "fn": 0}
    all_errors = []
    exact_counts = 0
    count_errors = []
    dt = float(np.median(np.diff(bundle["time_axis"][0])))
    grid_tolerance_m = detector_config.grid_tolerance_bins * data_config.wavespeed_m_s * dt / 2.0
    for index, case_id in enumerate(bundle["case_id"]):
        count = int(bundle["n_frac"][index])
        true_depths = bundle["x_f"][index, :count]
        detected = detect_events(
            bundle["probability"][index],
            bundle["valid_time_mask"][index],
            bundle["time_axis"][index],
            threshold,
            data_config,
            detector_config,
        )
        physical = match_events(detected, true_depths, detector_config.physical_tolerance_m)
        grid = match_events(detected, true_depths, grid_tolerance_m)
        for key in physical_totals:
            physical_totals[key] += physical[key]
            grid_totals[key] += grid[key]
        all_errors.extend(physical["depth_errors_m"])
        predicted_count = len(detected)
        exact_counts += int(predicted_count == count)
        count_errors.append(abs(predicted_count - count))
        spacing = float(bundle["min_spacing_m"][index])
        rows.append({
            "case_id": str(case_id),
            "n_frac": count,
            "min_spacing_m": spacing,
            "predicted_count": predicted_count,
            "count_absolute_error": abs(predicted_count - count),
            "physical": physical,
            "grid": grid,
            "detected": detected,
        })

    def aggregate(totals: Dict[str, int]) -> Dict:
        precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
        recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
        return {
            **totals,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }

    return {
        "threshold": threshold,
        "n_cases": len(rows),
        "physical": aggregate(physical_totals),
        "grid": aggregate(grid_totals),
        "exact_count_accuracy": exact_counts / max(len(rows), 1),
        "exact_count_cases": exact_counts,
        "count_mae": float(np.mean(count_errors)) if count_errors else None,
        "median_depth_error_m": float(np.median(all_errors)) if all_errors else None,
        "p95_depth_error_m": float(np.percentile(all_errors, 95)) if all_errors else None,
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
    data_config = DataConfig(**manifest["data_config"])
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
