# -*- coding: utf-8 -*-
"""Quantitative, blind evaluation for DCCDM fracture-map predictions."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError("Cannot find wellbore_moc_method root")
    _d = _parent

from neural_operator.dccdm_pipeline import json_safe, write_json


DEFAULT_FIXED_TOLERANCE_S = 0.15
DEFAULT_STRICT_TOLERANCE_S = 0.05
DEFAULT_PEAK_HEIGHT = 1.0
DEFAULT_PEAK_PROMINENCE = 0.5


def detect_positive_peaks(
    response: np.ndarray,
    time_axis: np.ndarray,
    *,
    height: float = DEFAULT_PEAK_HEIGHT,
    prominence: float = DEFAULT_PEAK_PROMINENCE,
    min_distance_bins: int = 1,
) -> List[Dict]:
    """Blind positive-only detector with fixed, case-independent settings."""
    response = np.asarray(response, dtype=float).reshape(-1)
    time_axis = np.asarray(time_axis, dtype=float).reshape(-1)
    if len(response) != len(time_axis):
        raise ValueError("response and time_axis must have the same length")
    positive = np.clip(response, 0.0, None)
    peaks, properties = find_peaks(
        positive,
        height=height,
        prominence=prominence,
        distance=max(1, int(min_distance_bins)),
    )
    order = np.argsort(positive[peaks])[::-1]
    return [
        {
            "index": int(index),
            "time_s": float(time_axis[index]),
            "amplitude": float(response[index]),
            "prominence": float(properties["prominences"][position]),
            "rank": rank,
        }
        for rank, position in enumerate(order, start=1)
        for index in [peaks[position]]
    ]


def match_peaks_one_to_one(
    detected: Sequence[Dict],
    true_times: np.ndarray,
    true_depths: np.ndarray,
    true_amplitudes: np.ndarray,
    *,
    tolerance_s: float,
    wavespeed: float,
    pump_shut_time: float = 1.0,
) -> Dict:
    detected_times = np.asarray([peak["time_s"] for peak in detected], dtype=float)
    n_true = len(true_times)
    n_detected = len(detected_times)
    matches: List[Dict] = []

    if n_true and n_detected:
        cost = np.abs(true_times[:, None] - detected_times[None, :])
        large = tolerance_s * 1.0e6 + 1.0
        assignment_cost = np.where(cost <= tolerance_s, cost, large)
        rows, cols = linear_sum_assignment(assignment_cost)
        pairs = {
            int(row): int(col)
            for row, col in zip(rows, cols)
            if cost[row, col] <= tolerance_s
        }
    else:
        pairs = {}

    time_errors: List[float] = []
    depth_errors: List[float] = []
    amplitude_errors: List[float] = []
    for true_index in range(n_true):
        detected_index = pairs.get(true_index)
        if detected_index is None:
            matches.append({
                "true_index": true_index,
                "true_time_s": float(true_times[true_index]),
                "true_depth_m": float(true_depths[true_index]),
                "true_amplitude": float(true_amplitudes[true_index]),
                "detected_index": None,
                "detected_time_s": None,
                "detected_depth_m": None,
                "detected_amplitude": None,
                "time_error_s": None,
                "depth_error_m": None,
                "amplitude_error": None,
                "matched": False,
            })
            continue
        peak = detected[detected_index]
        time_error = abs(float(peak["time_s"]) - float(true_times[true_index]))
        depth = (float(peak["time_s"]) - pump_shut_time) * wavespeed / 2.0
        depth_error = abs(depth - float(true_depths[true_index]))
        amplitude_error = abs(float(peak["amplitude"]) - float(true_amplitudes[true_index]))
        time_errors.append(time_error)
        depth_errors.append(depth_error)
        amplitude_errors.append(amplitude_error)
        matches.append({
            "true_index": true_index,
            "true_time_s": float(true_times[true_index]),
            "true_depth_m": float(true_depths[true_index]),
            "true_amplitude": float(true_amplitudes[true_index]),
            "detected_index": detected_index,
            "detected_time_s": float(peak["time_s"]),
            "detected_depth_m": depth,
            "detected_amplitude": float(peak["amplitude"]),
            "time_error_s": time_error,
            "depth_error_m": depth_error,
            "amplitude_error": amplitude_error,
            "matched": True,
        })

    tp = len(pairs)
    fp = n_detected - tp
    fn = n_true - tp
    precision = tp / n_detected if n_detected else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_true": n_true,
        "n_detected": n_detected,
        "peak_count_absolute_error": abs(n_detected - n_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time_error_mean_s": float(np.mean(time_errors)) if time_errors else None,
        "time_error_median_s": float(np.median(time_errors)) if time_errors else None,
        "time_error_max_s": float(np.max(time_errors)) if time_errors else None,
        "depth_error_mean_m": float(np.mean(depth_errors)) if depth_errors else None,
        "depth_error_median_m": float(np.median(depth_errors)) if depth_errors else None,
        "depth_error_max_m": float(np.max(depth_errors)) if depth_errors else None,
        "matched_amplitude_mae": float(np.mean(amplitude_errors)) if amplitude_errors else None,
        "matched_amplitude_rmse": float(np.sqrt(np.mean(np.square(amplitude_errors)))) if amplitude_errors else None,
        "matches": matches,
    }


def hoyer_sparsity(values: np.ndarray) -> float:
    values = np.abs(np.asarray(values, dtype=float).reshape(-1))
    if len(values) <= 1 or np.linalg.norm(values) == 0:
        return 1.0
    return float((np.sqrt(len(values)) - values.sum() / np.linalg.norm(values)) / (np.sqrt(len(values)) - 1.0))


def evaluate_case(
    prediction: np.ndarray,
    true_map: np.ndarray,
    time_axis: np.ndarray,
    x_f: np.ndarray,
    Cf: np.ndarray,
    *,
    wavespeed: float,
    pump_shut_time: float,
    target_scale: float,
    peak_height: float,
    peak_prominence: float,
    min_distance_bins: int,
    fixed_tolerance_s: float,
    strict_tolerance_s: float,
) -> Dict:
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    true_map = np.asarray(true_map, dtype=float).reshape(-1)
    time_axis = np.asarray(time_axis, dtype=float).reshape(-1)
    finite = np.isfinite(prediction)
    finite_prediction = np.where(finite, prediction, 0.0)
    positive = np.clip(finite_prediction, 0.0, None)
    detected = detect_positive_peaks(
        positive,
        time_axis,
        height=peak_height,
        prominence=peak_prominence,
        min_distance_bins=min_distance_bins,
    )
    true_times = pump_shut_time + 2.0 * np.asarray(x_f, dtype=float) / wavespeed
    true_amplitudes = np.log10(np.maximum(np.asarray(Cf, dtype=float), 1.0e-12)) + 12.0

    map_error = finite_prediction - true_map
    true_mass = float(np.clip(true_map, 0.0, None).sum())
    predicted_mass = float(positive.sum())
    physical = {
        "finite_fraction": float(finite.mean()),
        "minimum": float(np.nanmin(prediction)) if len(prediction) else None,
        "maximum": float(np.nanmax(prediction)) if len(prediction) else None,
        "negative_fraction": float((finite_prediction < 0).mean()),
        "above_target_scale_fraction": float((finite_prediction > target_scale).mean()),
    }
    map_metrics = {
        "mae": float(np.mean(np.abs(map_error))),
        "rmse": float(np.sqrt(np.mean(np.square(map_error)))),
        "positive_mass": predicted_mass,
        "positive_mass_relative_error": abs(predicted_mass - true_mass) / max(true_mass, 1.0e-12),
    }
    sparsity = {
        "fraction_above_threshold": float((positive >= peak_height).mean()),
        "detected_peak_count": len(detected),
        "positive_mass": predicted_mass,
        "hoyer": hoyer_sparsity(positive),
    }
    fixed = match_peaks_one_to_one(
        detected, true_times, np.asarray(x_f), true_amplitudes,
        tolerance_s=fixed_tolerance_s, wavespeed=wavespeed,
        pump_shut_time=pump_shut_time,
    )
    strict = match_peaks_one_to_one(
        detected, true_times, np.asarray(x_f), true_amplitudes,
        tolerance_s=strict_tolerance_s, wavespeed=wavespeed,
        pump_shut_time=pump_shut_time,
    )
    return {
        "physical_validity": physical,
        "map": map_metrics,
        "sparsity": sparsity,
        "detected_peaks": detected,
        "fixed_tolerance": fixed,
        "strict_tolerance": strict,
    }


def _mean_defined(rows: Sequence[Dict], path: Tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def aggregate_case_metrics(rows: Sequence[Dict]) -> Dict:
    aggregate: Dict[str, Dict] = {}
    for tolerance_key in ("fixed_tolerance", "strict_tolerance"):
        tp = sum(row[tolerance_key]["tp"] for row in rows)
        fp = sum(row[tolerance_key]["fp"] for row in rows)
        fn = sum(row[tolerance_key]["fn"] for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        aggregate[tolerance_key] = {
            "micro_tp": tp,
            "micro_fp": fp,
            "micro_fn": fn,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "macro_precision": _mean_defined(rows, (tolerance_key, "precision")),
            "macro_recall": _mean_defined(rows, (tolerance_key, "recall")),
            "macro_f1": _mean_defined(rows, (tolerance_key, "f1")),
            "mean_depth_error_m": _mean_defined(rows, (tolerance_key, "depth_error_mean_m")),
            "median_depth_error_m": _mean_defined(rows, (tolerance_key, "depth_error_median_m")),
        }
    aggregate["physical_validity"] = {
        key: _mean_defined(rows, ("physical_validity", key))
        for key in ("finite_fraction", "minimum", "maximum", "negative_fraction", "above_target_scale_fraction")
    }
    aggregate["map"] = {
        key: _mean_defined(rows, ("map", key))
        for key in ("mae", "rmse", "positive_mass", "positive_mass_relative_error")
    }
    aggregate["sparsity"] = {
        key: _mean_defined(rows, ("sparsity", key))
        for key in ("fraction_above_threshold", "detected_peak_count", "positive_mass", "hoyer")
    }
    aggregate["n_cases"] = len(rows)
    return aggregate


def evaluate_prediction_bundle(
    *,
    true_maps: np.ndarray,
    raw_predictions: np.ndarray,
    postprocessed_predictions: np.ndarray,
    time_axes: np.ndarray,
    x_f_padded: np.ndarray,
    Cf_padded: np.ndarray,
    n_frac: np.ndarray,
    case_ids: Sequence[str],
    wavespeed: float = 1450.0,
    pump_shut_time: float = 1.0,
    target_scale: float = 25.0,
    peak_height: float = DEFAULT_PEAK_HEIGHT,
    peak_prominence: float = DEFAULT_PEAK_PROMINENCE,
    min_distance_bins: int = 1,
    fixed_tolerance_s: float = DEFAULT_FIXED_TOLERANCE_S,
    strict_tolerance_s: float = DEFAULT_STRICT_TOLERANCE_S,
) -> Dict:
    variants = {
        "raw": raw_predictions,
        "postprocessed": postprocessed_predictions,
        "zero_baseline": np.zeros_like(raw_predictions),
    }
    per_case: Dict[str, List[Dict]] = {key: [] for key in variants}
    for variant, predictions in variants.items():
        for index, prediction in enumerate(predictions):
            count = int(n_frac[index])
            metrics = evaluate_case(
                prediction=np.asarray(prediction).reshape(-1),
                true_map=np.asarray(true_maps[index]).reshape(-1),
                time_axis=np.asarray(time_axes[index]).reshape(-1),
                x_f=x_f_padded[index, :count],
                Cf=Cf_padded[index, :count],
                wavespeed=wavespeed,
                pump_shut_time=pump_shut_time,
                target_scale=target_scale,
                peak_height=peak_height,
                peak_prominence=peak_prominence,
                min_distance_bins=min_distance_bins,
                fixed_tolerance_s=fixed_tolerance_s,
                strict_tolerance_s=strict_tolerance_s,
            )
            metrics["case_id"] = str(case_ids[index])
            metrics["variant"] = variant
            per_case[variant].append(metrics)
    return {
        "config": {
            "wavespeed": wavespeed,
            "pump_shut_time": pump_shut_time,
            "target_scale": target_scale,
            "peak_height": peak_height,
            "peak_prominence": peak_prominence,
            "min_distance_bins": min_distance_bins,
            "fixed_tolerance_s": fixed_tolerance_s,
            "fixed_tolerance_m": fixed_tolerance_s * wavespeed / 2.0,
            "strict_tolerance_s": strict_tolerance_s,
            "strict_tolerance_m": strict_tolerance_s * wavespeed / 2.0,
        },
        "aggregate": {variant: aggregate_case_metrics(rows) for variant, rows in per_case.items()},
        "per_case": per_case,
    }


def write_evaluation(evaluation_dir: str, result: Dict, manifest: Dict) -> None:
    os.makedirs(evaluation_dir, exist_ok=False)
    write_json(os.path.join(evaluation_dir, "evaluation_manifest.json"), manifest)
    write_json(
        os.path.join(evaluation_dir, "metrics_summary.json"),
        {"config": result["config"], "aggregate": result["aggregate"]},
    )
    csv_path = os.path.join(evaluation_dir, "per_case_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fields = [
            "variant", "case_id", "finite_fraction", "minimum", "maximum",
            "negative_fraction", "above_target_scale_fraction", "map_mae", "map_rmse",
            "fraction_above_threshold", "hoyer", "n_detected",
            "fixed_precision", "fixed_recall", "fixed_f1", "fixed_depth_error_mean_m",
            "strict_precision", "strict_recall", "strict_f1", "strict_depth_error_mean_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, rows in result["per_case"].items():
            for row in rows:
                writer.writerow({
                    "variant": variant,
                    "case_id": row["case_id"],
                    "finite_fraction": row["physical_validity"]["finite_fraction"],
                    "minimum": row["physical_validity"]["minimum"],
                    "maximum": row["physical_validity"]["maximum"],
                    "negative_fraction": row["physical_validity"]["negative_fraction"],
                    "above_target_scale_fraction": row["physical_validity"]["above_target_scale_fraction"],
                    "map_mae": row["map"]["mae"],
                    "map_rmse": row["map"]["rmse"],
                    "fraction_above_threshold": row["sparsity"]["fraction_above_threshold"],
                    "hoyer": row["sparsity"]["hoyer"],
                    "n_detected": row["sparsity"]["detected_peak_count"],
                    "fixed_precision": row["fixed_tolerance"]["precision"],
                    "fixed_recall": row["fixed_tolerance"]["recall"],
                    "fixed_f1": row["fixed_tolerance"]["f1"],
                    "fixed_depth_error_mean_m": row["fixed_tolerance"]["depth_error_mean_m"],
                    "strict_precision": row["strict_tolerance"]["precision"],
                    "strict_recall": row["strict_tolerance"]["recall"],
                    "strict_f1": row["strict_tolerance"]["f1"],
                    "strict_depth_error_mean_m": row["strict_tolerance"]["depth_error_mean_m"],
                })


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved DCCDM prediction artifacts")
    parser.add_argument("artifact", help="predictions.npz or legacy val_results.npz")
    parser.add_argument("--output-root", default="output/dccdm/evaluations")
    parser.add_argument("--peak-height", type=float, default=DEFAULT_PEAK_HEIGHT)
    parser.add_argument("--peak-prominence", type=float, default=DEFAULT_PEAK_PROMINENCE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.artifact, allow_pickle=False)
    required = {
        "true_map", "raw_prediction", "postprocessed_prediction", "time_axis",
        "x_f", "Cf", "n_frac", "case_id",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(
            f"artifact missing {sorted(missing)}; regenerate it with the schema-v2 inference workflow"
        )
    result = evaluate_prediction_bundle(
        true_maps=data["true_map"],
        raw_predictions=data["raw_prediction"],
        postprocessed_predictions=data["postprocessed_prediction"],
        time_axes=data["time_axis"],
        x_f_padded=data["x_f"],
        Cf_padded=data["Cf"],
        n_frac=data["n_frac"],
        case_ids=[str(value) for value in data["case_id"]],
        peak_height=args.peak_height,
        peak_prominence=args.peak_prominence,
    )
    evaluation_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-seed{args.seed}"
    evaluation_dir = os.path.join(args.output_root, evaluation_id)
    manifest = {
        "evaluation_id": evaluation_id,
        "artifact": os.path.abspath(args.artifact),
        "seed": args.seed,
        "config": result["config"],
    }
    write_evaluation(evaluation_dir, result, manifest)
    print(f"Evaluation written to {evaluation_dir}")


if __name__ == "__main__":
    main()
