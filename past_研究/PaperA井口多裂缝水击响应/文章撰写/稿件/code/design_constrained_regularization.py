"""Design-constrained cepstral-profile regularization for Paper A.

This module deliberately separates calibration/application from evaluation.
``apply_correction`` accepts the measured profile and the known perforation
design, but it does not accept active-fracture truth.  Truth is reserved for
the validation script.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence

import numpy as np
from scipy.optimize import curve_fit, linear_sum_assignment


def _ensure_repo_root() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "analysis").is_dir() and (parent / "moc_simulate").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    raise RuntimeError("Repository root not found")


_ensure_repo_root()

from analysis.unified_evaluation.detection_protocol import DetectorConfig, detect_peaks


@dataclass(frozen=True)
class RegularizationCalibration:
    gamma_by_spacing: Dict[float, float]
    gain_low: float
    gain_high: float
    focus_fwhm_m: float


def _exp_envelope(k0: np.ndarray, amplitude: float, gamma: float) -> np.ndarray:
    return amplitude * np.exp(-gamma * k0)


def _local_peak_values(
    depth: np.ndarray,
    profile: np.ndarray,
    perforation_positions: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    radius = min(15.0, 0.49 * float(spacing_m))
    values = []
    for position in perforation_positions:
        mask = np.abs(depth - position) <= radius
        values.append(float(np.max(profile[mask])) if np.any(mask) else np.nan)
    return np.asarray(values, dtype=float)


def _estimate_fwhm(depth: np.ndarray, profile: np.ndarray, center: float) -> float:
    mask = np.abs(depth - center) <= 20.0
    if np.count_nonzero(mask) < 5:
        return 5.0
    x = depth[mask]
    y = profile[mask]
    peak_index = int(np.argmax(y))
    peak = float(y[peak_index])
    baseline = float(np.percentile(y, 10.0))
    half = baseline + 0.5 * (peak - baseline)
    above = np.flatnonzero(y >= half)
    if above.size < 2:
        return 5.0
    width = float(x[above[-1]] - x[above[0]])
    return float(np.clip(width, 1.0, 20.0))


class DesignConstrainedRegularizer:
    """Calibrated, deterministic profile regularizer with no truth input."""

    def __init__(self) -> None:
        self.calibration: RegularizationCalibration | None = None

    def calibrate(self, calibration_cases: Sequence[Mapping[str, object]]) -> RegularizationCalibration:
        """Calibrate spacing-dependent envelopes and bounded gains.

        Each calibration case must contain ``depth``, ``raw_profile``,
        ``perforation_positions`` and ``spacing_m``.  Calibration cases are
        assumed to be the predefined all-active calibration set.
        """
        gamma_samples: Dict[float, List[float]] = {}
        fitted_cases: List[tuple[float, np.ndarray]] = []
        fwhm_samples: List[float] = []

        for case in calibration_cases:
            depth = np.asarray(case["depth"], dtype=float)
            profile = np.asarray(case["raw_profile"], dtype=float)
            x_perf = np.asarray(case["perforation_positions"], dtype=float)
            spacing = float(case["spacing_m"])
            peaks = _local_peak_values(depth, profile, x_perf, spacing)

            if x_perf.size == 1:
                fwhm_samples.append(_estimate_fwhm(depth, profile, float(x_perf[0])))
                continue
            if x_perf.size < 3 or not np.all(np.isfinite(peaks)) or np.any(peaks <= 0.0):
                continue
            try:
                params, _ = curve_fit(
                    _exp_envelope,
                    np.arange(x_perf.size, dtype=float),
                    peaks,
                    p0=[float(peaks[0]), 0.4],
                    bounds=([1e-12, 0.0], [np.inf, 5.0]),
                    maxfev=8000,
                )
            except (RuntimeError, ValueError):
                continue
            gamma = float(params[1])
            gamma_samples.setdefault(spacing, []).append(gamma)
            fitted_cases.append((spacing, peaks))

        if not gamma_samples:
            raise ValueError("No valid multi-cluster calibration cases were supplied")

        gamma_by_spacing = {
            float(spacing): float(np.mean(values))
            for spacing, values in sorted(gamma_samples.items())
        }

        gain_samples: List[float] = []
        for spacing, peaks in fitted_cases:
            gamma = gamma_by_spacing[spacing]
            expected = peaks[0] * np.exp(-gamma * np.arange(peaks.size, dtype=float))
            gain_samples.extend((expected / np.maximum(peaks, 1e-12)).tolist())

        if gain_samples:
            gain_low = float(np.clip(np.percentile(gain_samples, 5.0), 0.25, 1.0))
            gain_high = float(np.clip(np.percentile(gain_samples, 95.0), 1.0, 4.0))
        else:
            gain_low, gain_high = 0.5, 2.0
        focus_fwhm = float(np.median(fwhm_samples)) if fwhm_samples else 5.0
        focus_fwhm = float(np.clip(focus_fwhm, 1.0, 20.0))

        self.calibration = RegularizationCalibration(
            gamma_by_spacing=gamma_by_spacing,
            gain_low=gain_low,
            gain_high=gain_high,
            focus_fwhm_m=focus_fwhm,
        )
        return self.calibration

    def apply_correction(
        self,
        depth: Sequence[float],
        raw_profile: Sequence[float],
        perforation_positions: Sequence[float],
        cluster_spacing_m: float,
    ) -> Dict[str, object]:
        """Apply bounded correction using only the profile and design layout."""
        if self.calibration is None:
            raise RuntimeError("calibrate() must be called before apply_correction()")

        depth_arr = np.asarray(depth, dtype=float)
        raw = np.asarray(raw_profile, dtype=float)
        x_perf = np.asarray(perforation_positions, dtype=float)
        if depth_arr.shape != raw.shape:
            raise ValueError("depth and raw_profile must have identical shapes")
        if x_perf.size == 0:
            raise ValueError("perforation_positions must not be empty")

        search_margin = max(30.0, float(cluster_spacing_m))
        detector_cfg = DetectorConfig(
            min_separation_m=5.0,
            height_pct=85.0,
            height_rel=0.03,
            height_abs=0.0,
            prominence_rel=0.0,
            max_peaks=10,
            search_min_m=float(x_perf.min() - search_margin),
            search_max_m=float(x_perf.max() + search_margin),
        )
        candidates = detect_peaks(depth_arr, raw, detector_cfg)
        candidate_depths = np.asarray([p["depth_m"] for p in candidates], dtype=float)

        matches: List[Dict[str, float | int]] = []
        tolerance_m = 10.0
        if x_perf.size and candidate_depths.size:
            cost = np.abs(x_perf[:, None] - candidate_depths[None, :])
            rows, cols = linear_sum_assignment(cost)
            for design_index, candidate_index in zip(rows, cols):
                error = float(cost[design_index, candidate_index])
                if error <= tolerance_m:
                    matches.append({
                        "design_index": int(design_index),
                        "candidate_index": int(candidate_index),
                        "design_depth_m": float(x_perf[design_index]),
                        "candidate_depth_m": float(candidate_depths[candidate_index]),
                        "error_m": error,
                    })

        gamma = self._gamma_for_spacing(float(cluster_spacing_m))
        match_by_design = {int(m["design_index"]): m for m in matches}
        first_match = match_by_design.get(0)
        first_amplitude = None
        if first_match is not None:
            first_amplitude = float(candidates[int(first_match["candidate_index"])]["response"])

        fwhm = self.calibration.focus_fwhm_m
        sigma = max(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 0.25)
        focus = np.zeros_like(raw)
        design_weight = np.zeros_like(raw)
        gain_num = np.zeros_like(raw)
        gain_den = np.zeros_like(raw)
        cluster_gains: List[Dict[str, float | int | bool]] = []

        design_sigma = max(sigma, 2.0)
        for design_index, position in enumerate(x_perf):
            design_weight = np.maximum(
                design_weight,
                np.exp(-0.5 * ((depth_arr - position) / design_sigma) ** 2),
            )
            match = match_by_design.get(design_index)
            if match is None:
                cluster_gains.append({
                    "design_index": design_index,
                    "matched": False,
                    "gain": 1.0,
                })
                continue
            candidate = candidates[int(match["candidate_index"])]
            candidate_depth = float(candidate["depth_m"])
            observed = max(float(candidate["response"]), 1e-12)
            if first_amplitude is None:
                gain = 1.0
            else:
                expected = first_amplitude * np.exp(-gamma * design_index)
                gain = float(np.clip(
                    expected / observed,
                    self.calibration.gain_low,
                    self.calibration.gain_high,
                ))
            window = np.exp(-0.5 * ((depth_arr - candidate_depth) / sigma) ** 2)
            focus = np.maximum(focus, window)
            gain_num += gain * window
            gain_den += window
            cluster_gains.append({
                "design_index": design_index,
                "matched": True,
                "gain": gain,
                "observed_amplitude": observed,
                "candidate_depth_m": candidate_depth,
            })

        gain_field = np.ones_like(raw)
        covered = gain_den > 1e-12
        gain_field[covered] = gain_num[covered] / gain_den[covered]
        amplitude_only = raw * gain_field
        focus_only = raw * focus
        regularized = raw * focus * gain_field
        design_only = raw * design_weight

        return {
            "depth": depth_arr,
            "raw_profile": raw,
            "regularized_profile": regularized,
            "design_only_profile": design_only,
            "amplitude_only_profile": amplitude_only,
            "focus_only_profile": focus_only,
            "design_weight": design_weight,
            "focus_kernel": focus,
            "gain_field": gain_field,
            "candidate_peaks": candidates,
            "matches": matches,
            "cluster_gains": cluster_gains,
            "gamma_cal": gamma,
            "detector_config": detector_cfg.__dict__.copy(),
        }

    def _gamma_for_spacing(self, spacing_m: float) -> float:
        assert self.calibration is not None
        spacings = np.asarray(sorted(self.calibration.gamma_by_spacing), dtype=float)
        values = np.asarray(
            [self.calibration.gamma_by_spacing[float(s)] for s in spacings],
            dtype=float,
        )
        return float(np.interp(spacing_m, spacings, values))
