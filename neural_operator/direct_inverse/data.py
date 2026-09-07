# -*- coding: utf-8 -*-
"""Data audit, immutable splits, and inverse-facing dataset adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError("Cannot find project root")
    _d = _parent
PROJECT_ROOT = _d

from neural_operator.dataset_surrogate import FracturingMOCSurrogateDataset
from neural_operator.dccdm_pipeline import json_safe, write_json
from neural_operator.direct_inverse.config import (
    DATA_SCHEMA,
    DATA_SCHEMA_CLOSE2000,
    DATA_SCHEMA_CLOSE2000_N8192,
    DATA_SCHEMA_CLOSE2000_N8192_PHYS3,
    DATA_SCHEMA_CLOSE2000_N16384,
    DATA_SCHEMA_CLOSE2000_N16384_WIDEFWHM,
    DataConfig,
    FeatureConfig,
    TargetConfig,
    close2000_detector_config,
    close2000_target_config,
    profile_configs,
    resampled_bin_depth_m,
)
from moc_simulate.cepstrum_mocdata import real_cepstrum_1d
from scipy.signal import hilbert


def event_time_s(depth_m: np.ndarray | float, config: DataConfig) -> np.ndarray:
    return config.pump_shut_time_s + 2.0 * np.asarray(depth_m, dtype=float) / config.wavespeed_m_s


def time_to_depth_m(time_s: np.ndarray | float, config: DataConfig) -> np.ndarray:
    return (np.asarray(time_s, dtype=float) - config.pump_shut_time_s) * config.wavespeed_m_s / 2.0


def minimum_spacing_m(depths: np.ndarray) -> float:
    depths = np.sort(np.asarray(depths, dtype=float))
    return float(np.min(np.diff(depths))) if len(depths) > 1 else float("inf")


def spacing_band(n_frac: int, spacing_m: float, regime: str = "nominal50") -> str:
    if n_frac == 1:
        return "singleton"
    if regime == "close5_20":
        if spacing_m < 10.0:
            return "5-10"
        if spacing_m < 15.0:
            return "10-15"
        return "15-20"
    if spacing_m < 50.0:
        return "lt50"
    if spacing_m < 75.0:
        return "50-75"
    if spacing_m < 100.0:
        return "75-100"
    return "ge100"


def construct_event_target(
    time_axis: np.ndarray,
    fracture_depths_m: np.ndarray,
    data_config: DataConfig,
    target_config: TargetConfig,
) -> np.ndarray:
    time_axis = np.asarray(time_axis, dtype=np.float32)
    arrivals = event_time_s(fracture_depths_m, data_config)
    target = np.zeros_like(time_axis, dtype=np.float32)
    for arrival in arrivals:
        gaussian = np.exp(
            -((time_axis - float(arrival)) ** 2) / (2.0 * target_config.sigma_time_s**2)
        ).astype(np.float32)
        target = np.maximum(target, gaussian)
    return target


def construct_search_mask(time_axis: np.ndarray, config: DataConfig) -> np.ndarray:
    start, end = event_time_s(np.asarray(config.fracture_zone_m), config)
    return ((time_axis >= start) & (time_axis <= end)).astype(bool)


def compute_envelope_channel(pressure: np.ndarray) -> np.ndarray:
    """Analytic-signal envelope |hilbert(H)| on the resampled waveform."""
    pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
    demeaned = pressure - np.mean(pressure)
    return np.abs(hilbert(demeaned)).astype(np.float32)


def compute_cepstrum_time_channel(
    pressure: np.ndarray,
    time_axis: np.ndarray,
    data_config: DataConfig,
    *,
    as_response: bool = True,
) -> np.ndarray:
    """Map 1D real cepstrum C(q) onto the arrival-time axis t = t_s + q.

    Fracture-related quefrency peaks then sit near the same time indices used by
    the event heatmap labels (two-way travel).
    """
    pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
    time_axis = np.asarray(time_axis, dtype=np.float64).reshape(-1)
    if len(pressure) != len(time_axis):
        raise ValueError("pressure and time_axis lengths differ")
    if len(time_axis) < 2:
        raise ValueError("time_axis must have length >= 2")
    dt = float(np.median(np.diff(time_axis)))
    fs = 1.0 / dt
    cepstrum, quefrency = real_cepstrum_1d(pressure, fs)
    values = -cepstrum if as_response else cepstrum
    arrival_time = data_config.pump_shut_time_s + quefrency
    aligned = np.interp(time_axis, arrival_time, values, left=0.0, right=0.0)
    return aligned.astype(np.float32)


def build_observation_channels(
    pressure: np.ndarray,
    time_axis: np.ndarray,
    data_config: DataConfig,
    feature_config: FeatureConfig,
) -> np.ndarray:
    """Stack observation channels as float32 array [C, T]."""
    pressure = np.asarray(pressure, dtype=np.float32).reshape(-1)
    channels = [pressure]
    if feature_config.include_envelope:
        channels.append(compute_envelope_channel(pressure))
    if feature_config.include_cepstrum:
        channels.append(
            compute_cepstrum_time_channel(
                pressure,
                time_axis,
                data_config,
                as_response=feature_config.cepstrum_as_response,
            )
        )
    stacked = np.stack(channels, axis=0)
    if stacked.shape[0] != feature_config.n_channels:
        raise ValueError(
            f"feature channels {stacked.shape[0]} != configured n_channels={feature_config.n_channels}"
        )
    return stacked


def _source_digest(rows: Sequence[Dict]) -> str:
    payload = "\n".join(
        f"{row['case_id']}\t{row['relative_path']}\t{row['n_frac']}\t{row['min_spacing_m']}"
        for row in sorted(rows, key=lambda value: value["case_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_cases(data_dir: str, config: DataConfig) -> Tuple[List[Dict], Dict]:
    all_files = sorted(
        os.path.join(os.path.abspath(data_dir), filename)
        for filename in os.listdir(data_dir)
        if filename.startswith("case_") and filename.endswith(".npz")
    )
    rows: List[Dict] = []
    seen = set()
    failures: List[Dict] = []
    for path in all_files:
        try:
            with np.load(path, allow_pickle=False) as npz:
                required = ("t", "H_wh", "x_f", "Cf", "kleak", "n_frac", "tf")
                missing = [key for key in required if key not in npz]
                if missing:
                    raise ValueError(f"missing fields {missing}")
                t = np.asarray(npz["t"], dtype=float)
                pressure = np.asarray(npz["H_wh"], dtype=float)
                x_f = np.asarray(npz["x_f_aligned"] if "x_f_aligned" in npz else npz["x_f"], dtype=float)
                cf = np.asarray(npz["Cf"], dtype=float)
                kleak = np.asarray(npz["kleak"], dtype=float)
                n_frac = int(npz["n_frac"])
                tf = float(npz["tf"])
                if t.ndim != 1 or pressure.ndim != 1 or len(t) != len(pressure):
                    raise ValueError("invalid t/H_wh shapes")
                if len(t) < 2 or not np.all(np.diff(t) > 0):
                    raise ValueError("time axis is not strictly increasing")
                if not (len(x_f) == len(cf) == len(kleak) == n_frac):
                    raise ValueError("fracture arrays do not match n_frac")
                for name, values in (("t", t), ("H_wh", pressure), ("x_f", x_f), ("Cf", cf), ("kleak", kleak)):
                    if not np.isfinite(values).all():
                        raise ValueError(f"{name} contains NaN/Inf")
                if abs(tf - config.expected_tf_s) > 1.0e-6:
                    raise ValueError(f"unexpected tf={tf}, expected {config.expected_tf_s}")
                case_id = os.path.splitext(os.path.basename(path))[0]
                if case_id in seen:
                    raise ValueError("duplicate case ID")
                seen.add(case_id)
                spacing = minimum_spacing_m(x_f)
                nominal = n_frac == 1 or spacing >= config.nominal_min_spacing_m
                rows.append({
                    "case_id": case_id,
                    "relative_path": os.path.relpath(path, PROJECT_ROOT).replace("\\", "/"),
                    "n_frac": n_frac,
                    "min_spacing_m": spacing,
                    "spacing_band": spacing_band(n_frac, spacing, config.spacing_regime),
                    "nominal": nominal,
                })
        except Exception as error:
            failures.append({"path": path, "error": str(error)})

    nominal_count = sum(row["nominal"] for row in rows)
    challenge_count = len(rows) - nominal_count
    audit = {
        "schema": config.schema,
        "data_dir": os.path.abspath(data_dir),
        "total_files": len(all_files),
        "valid_cases": len(rows),
        "invalid_cases": len(failures),
        "nominal_cases": nominal_count,
        "challenge_cases": challenge_count,
        "source_digest": _source_digest(rows),
        "failures": failures,
    }
    rows = json_safe(rows)
    audit = json_safe(audit)
    expected = {
        "total_files": config.expected_total_count,
        "valid_cases": config.expected_total_count,
        "invalid_cases": 0,
        "nominal_cases": config.expected_nominal_count,
        "challenge_cases": config.expected_challenge_count,
    }
    mismatches = {
        key: {"observed": audit[key], "expected": value}
        for key, value in expected.items()
        if audit[key] != value
    }
    audit["gate_passed"] = not mismatches
    audit["mismatches"] = mismatches
    return rows, audit


def _allocate_counts(group_sizes: Dict[str, int], total: int) -> Dict[str, int]:
    population = sum(group_sizes.values())
    exact = {key: total * size / population for key, size in group_sizes.items()}
    counts = {key: int(np.floor(value)) for key, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(group_sizes, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _stratified_take(rows: Sequence[Dict], total: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        groups[f"n{row['n_frac']}:{row['spacing_band']}"] .append(row)
    counts = _allocate_counts({key: len(value) for key, value in groups.items()}, total)
    selected: List[Dict] = []
    remaining: List[Dict] = []
    for group_index, key in enumerate(sorted(groups)):
        group = sorted(groups[key], key=lambda row: row["case_id"])
        rng = np.random.RandomState(seed + group_index * 104729)
        order = rng.permutation(len(group))
        shuffled = [group[index] for index in order]
        selected.extend(shuffled[: counts[key]])
        remaining.extend(shuffled[counts[key] :])
    return sorted(selected, key=lambda row: row["case_id"]), sorted(remaining, key=lambda row: row["case_id"])


def _overfit_band_plan(regime: str) -> Tuple[str, list]:
    if regime == "close5_20":
        overfit1_band = "15-20"
        desired = [
            (n_frac, band)
            for n_frac in range(1, 7)
            for band in (("singleton",) if n_frac == 1 else ("5-10", "10-15", "15-20"))
        ]
        return overfit1_band, desired
    overfit1_band = "50-75"
    desired = [
        (n_frac, band)
        for n_frac in range(1, 7)
        for band in (("singleton",) if n_frac == 1 else ("50-75", "75-100", "ge100"))
    ]
    return overfit1_band, desired


def build_split_manifest(
    rows: Sequence[Dict],
    config: DataConfig,
    target: TargetConfig,
    detector_config=None,
    feature_config: FeatureConfig | None = None,
) -> Dict:
    nominal = [dict(row) for row in rows if row["nominal"]]
    challenge = [dict(row, split="challenge") for row in rows if not row["nominal"]]
    test, remaining = _stratified_take(nominal, config.test_count, config.split_seed + 2)
    validation, train = _stratified_take(remaining, config.validation_count, config.split_seed + 1)
    if len(train) != config.train_count:
        raise RuntimeError(f"train count {len(train)} != {config.train_count}")
    for split, split_rows in (("train", train), ("validation", validation), ("test", test)):
        for row in split_rows:
            row["split"] = split

    overfit1_band, desired = _overfit_band_plan(config.spacing_regime)
    overfit1_candidates = [
        row for row in train if row["n_frac"] > 1 and row["spacing_band"] == overfit1_band
    ]
    if not overfit1_candidates:
        raise RuntimeError(f"no overfit1 candidate in {overfit1_band} band")
    overfit1 = [overfit1_candidates[0]["case_id"]]

    overfit16: List[Dict] = []
    used = set()
    for n_frac, band in desired:
        candidates = [
            row for row in train
            if row["case_id"] not in used and row["n_frac"] == n_frac and row["spacing_band"] == band
        ]
        if candidates and len(overfit16) < 16:
            overfit16.append(candidates[0])
            used.add(candidates[0]["case_id"])
    for row in train:
        if len(overfit16) >= 16:
            break
        if row["case_id"] not in used:
            overfit16.append(row)
            used.add(row["case_id"])

    train512, _ = _stratified_take(train, min(512, len(train)), config.split_seed + 512)
    train512_ids = {row["case_id"] for row in train512}
    for row in overfit16:
        if row["case_id"] not in train512_ids:
            replace_index = next(
                index for index in range(len(train512) - 1, -1, -1)
                if train512[index]["case_id"] not in {value["case_id"] for value in overfit16}
            )
            train512[replace_index] = row
            train512_ids.add(row["case_id"])
    val128, _ = _stratified_take(validation, min(128, len(validation)), config.split_seed + 128)

    all_rows = sorted(train + validation + test + challenge, key=lambda row: row["case_id"])
    if detector_config is None:
        detector_payload = asdict(close2000_detector_config()) if config.spacing_regime == "close5_20" else None
    else:
        detector_payload = asdict(detector_config)
    if feature_config is None:
        feature_config = FeatureConfig()
    manifest = {
        "schema": config.schema,
        "data_config": asdict(config),
        "target_config": {
            **asdict(target),
            "fwhm_time_s": target.fwhm_time_s,
            "sigma_time_s": target.sigma_time_s,
        },
        "feature_config": asdict(feature_config),
        "detector_config": detector_payload,
        "source_digest": _source_digest(rows),
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "challenge": len(challenge),
        },
        "subsets": {
            "overfit1": overfit1,
            "overfit16": [row["case_id"] for row in sorted(overfit16, key=lambda value: value["case_id"])],
            "train512": [row["case_id"] for row in sorted(train512, key=lambda value: value["case_id"])],
            "val128": [row["case_id"] for row in sorted(val128, key=lambda value: value["case_id"])],
        },
        "cases": all_rows,
    }
    return manifest


def load_manifest(path: str) -> Dict:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    schema = manifest.get("schema")
    if schema not in (
        DATA_SCHEMA,
        DATA_SCHEMA_CLOSE2000,
        DATA_SCHEMA_CLOSE2000_N8192,
        DATA_SCHEMA_CLOSE2000_N8192_PHYS3,
        DATA_SCHEMA_CLOSE2000_N16384,
        DATA_SCHEMA_CLOSE2000_N16384_WIDEFWHM,
    ):
        raise ValueError(f"unsupported manifest schema: {schema}")
    return manifest


def selected_case_ids(manifest: Dict, selection: str) -> List[str]:
    if selection in manifest["subsets"]:
        return list(manifest["subsets"][selection])
    if selection in ("train", "validation", "test", "challenge"):
        return [row["case_id"] for row in manifest["cases"] if row["split"] == selection]
    raise KeyError(f"unknown selection: {selection}")


class DirectInverseDataset(Dataset):
    """Inverse view: raw pressure observation plus narrow event target and audit metadata."""

    def __init__(self, manifest_path: str, selection: str):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        self.data_config = DataConfig(**{
            key: value for key, value in self.manifest["data_config"].items()
            if key in DataConfig.__dataclass_fields__
        })
        target_values = {
            key: value for key, value in self.manifest["target_config"].items()
            if key in TargetConfig.__dataclass_fields__
        }
        self.target_config = TargetConfig(**target_values)
        feature_payload = self.manifest.get("feature_config") or {}
        self.feature_config = FeatureConfig(**{
            key: value for key, value in feature_payload.items()
            if key in FeatureConfig.__dataclass_fields__
        })
        self.case_ids = selected_case_ids(self.manifest, selection)
        case_rows = {row["case_id"]: row for row in self.manifest["cases"]}
        self.rows = [case_rows[case_id] for case_id in self.case_ids]
        self.base = FracturingMOCSurrogateDataset(
            data_dir=os.path.dirname(os.path.join(PROJECT_ROOT, self.rows[0]["relative_path"])),
            n_time_target=self.data_config.seq_length,
            split="train",
            train_ratio=1.0,
            seed=0,
            wavespeed=self.data_config.wavespeed_m_s,
            ts=self.data_config.pump_shut_time_s,
        )
        self.base_index = {
            os.path.splitext(os.path.basename(path))[0]: index
            for index, path in enumerate(self.base.files)
        }
        missing = set(self.case_ids) - set(self.base_index)
        if missing:
            raise KeyError(f"manifest cases missing from dataset: {sorted(missing)[:5]}")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        case_id = self.case_ids[index]
        base_index = self.base_index[case_id]
        surrogate_input, pressure = self.base[base_index]
        del surrogate_input
        metadata = self.base.get_case_metadata(base_index)
        time_axis = metadata["time_axis"].astype(np.float32)
        pressure_1d = pressure.numpy().reshape(-1).astype(np.float32)
        observation = build_observation_channels(
            pressure_1d, time_axis, self.data_config, self.feature_config
        )
        event_target = construct_event_target(
            time_axis, metadata["x_f"], self.data_config, self.target_config
        )
        valid_mask = construct_search_mask(time_axis, self.data_config)
        maximum = self.data_config.max_fractures
        x_f = np.full(maximum, np.nan, dtype=np.float32)
        cf = np.full(maximum, np.nan, dtype=np.float32)
        kleak = np.full(maximum, np.nan, dtype=np.float32)
        event_time = np.full(maximum, np.nan, dtype=np.float32)
        event_bin = np.full(maximum, -1, dtype=np.int64)
        event_valid = np.zeros(maximum, dtype=bool)
        count = metadata["n_frac"]
        x_f[:count] = metadata["x_f"]
        cf[:count] = metadata["Cf"]
        kleak[:count] = metadata["kleak"]
        arrivals = event_time_s(metadata["x_f"], self.data_config).astype(np.float32)
        event_time[:count] = arrivals
        event_bin[:count] = np.asarray([int(np.argmin(np.abs(time_axis - value))) for value in arrivals])
        event_valid[:count] = True
        return {
            "observation": torch.from_numpy(observation),
            "event_target": torch.from_numpy(event_target[None, :]),
            "valid_time_mask": torch.from_numpy(valid_mask[None, :]),
            "time_axis": torch.from_numpy(time_axis),
            "event_bin": torch.from_numpy(event_bin),
            "event_time_s": torch.from_numpy(event_time),
            "x_f_m": torch.from_numpy(x_f),
            "Cf": torch.from_numpy(cf),
            "kleak": torch.from_numpy(kleak),
            "event_valid_mask": torch.from_numpy(event_valid),
            "n_frac": torch.tensor(count, dtype=torch.int64),
            "min_spacing_m": torch.tensor(minimum_spacing_m(metadata["x_f"]), dtype=torch.float32),
            "tf_s": torch.tensor(metadata["tf"], dtype=torch.float32),
            "case_id": case_id,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data and freeze direct-inverse split manifest")
    parser.add_argument("--data-dir", default="output/lhs_dataset/data")
    parser.add_argument("--output-root", default="output/direct_inverse/manifests")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        choices=(
            "nominal50",
            "close2000",
            "close2000_n8192",
            "close2000_n8192_phys3",
            "close2000_n16384",
            "close2000_n16384_widefwhm",
        ),
        default="nominal50",
    )
    parser.add_argument("--min-spacing-m", type=float, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    args = parser.parse_args()

    profiles = profile_configs(args.profile, args.split_seed)
    config = profiles["data"]
    from dataclasses import replace
    replacements = {}
    if args.min_spacing_m is not None:
        replacements["nominal_min_spacing_m"] = args.min_spacing_m
    if args.seq_length is not None:
        replacements["seq_length"] = args.seq_length
    if replacements:
        config = replace(config, **replacements)
    target = profiles["target"]
    detector = profiles["detector"]
    feature = profiles["feature"]
    # CLI seq_length override: realign bin-tied configs unless the profile
    # deliberately decouples FWHM from the grid (widefwhm).
    if args.profile.startswith("close2000") and args.seq_length is not None:
        bin_m = resampled_bin_depth_m(config.seq_length, config.expected_tf_s, config.wavespeed_m_s)
        detector = close2000_detector_config(bin_m)
        if "widefwhm" not in args.profile:
            target = close2000_target_config(bin_m)
    rows, audit = audit_cases(args.data_dir, config)
    os.makedirs(args.output_root, exist_ok=True)
    audit_path = os.path.join(args.output_root, "dataset-audit.json")
    write_json(audit_path, audit)
    if not audit["gate_passed"]:
        raise RuntimeError(f"dataset audit failed; see {audit_path}: {audit['mismatches']}")
    manifest = build_split_manifest(
        rows, config, target, detector_config=detector, feature_config=feature
    )
    manifest_path = os.path.join(args.output_root, f"{config.schema}-seed{args.split_seed}.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise FileExistsError(f"existing immutable manifest differs: {manifest_path}")
    else:
        write_json(manifest_path, manifest)
    print(f"Audit written to {audit_path}")
    print(f"Manifest written to {manifest_path}")
    print(f"Counts: {manifest['counts']}")
    if args.profile.startswith("close2000"):
        bin_m = resampled_bin_depth_m(config.seq_length, config.expected_tf_s, config.wavespeed_m_s)
        print(
            f"seq_length={config.seq_length} "
            f"bin_depth_m={bin_m:.4f} "
            f"fwhm_depth_m={target.fwhm_depth_m:.4f} "
            f"n_channels={feature.n_channels}"
        )

if __name__ == "__main__":
    main()
