from __future__ import annotations

import inspect
import unittest

import numpy as np

from design_constrained_regularization import DesignConstrainedRegularizer


def _gaussian_profile(depth: np.ndarray, centers: list[float], amplitudes: list[float]) -> np.ndarray:
    profile = np.zeros_like(depth)
    for center, amplitude in zip(centers, amplitudes):
        profile += amplitude * np.exp(-0.5 * ((depth - center) / 2.0) ** 2)
    return profile


class DesignConstrainedRegularizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.depth = np.linspace(950.0, 1150.0, 2001)
        self.x_perf = np.asarray([1000.0, 1020.0, 1040.0, 1060.0])
        calibration = []
        for gamma in (0.35, 0.4, 0.45):
            amplitudes = np.exp(-gamma * np.arange(4))
            calibration.append({
                "depth": self.depth,
                "raw_profile": _gaussian_profile(self.depth, self.x_perf.tolist(), amplitudes.tolist()),
                "perforation_positions": self.x_perf,
                "spacing_m": 20.0,
            })
        calibration.append({
            "depth": self.depth,
            "raw_profile": _gaussian_profile(self.depth, [1000.0], [1.0]),
            "perforation_positions": np.asarray([1000.0]),
            "spacing_m": 20.0,
        })
        self.regularizer = DesignConstrainedRegularizer()
        self.regularizer.calibrate(calibration)

    def test_public_application_interface_has_no_truth_argument(self) -> None:
        parameters = inspect.signature(self.regularizer.apply_correction).parameters
        forbidden = {"x_f_true", "active_positions", "active_mask", "truth"}
        self.assertTrue(forbidden.isdisjoint(parameters))

    def test_unmatched_design_cluster_is_not_synthesized(self) -> None:
        raw = _gaussian_profile(self.depth, [1000.0, 1020.0, 1060.0], [1.0, 0.7, 0.3])
        result = self.regularizer.apply_correction(self.depth, raw, self.x_perf, 20.0)
        unmatched = [g for g in result["cluster_gains"] if not g["matched"]]
        self.assertTrue(any(g["design_index"] == 2 for g in unmatched))
        near_missing = np.abs(self.depth - 1040.0) <= 2.0
        self.assertLess(float(np.max(result["regularized_profile"][near_missing])), 1e-6)

    def test_application_is_deterministic(self) -> None:
        raw = _gaussian_profile(self.depth, self.x_perf.tolist(), [1.0, 0.7, 0.45, 0.3])
        first = self.regularizer.apply_correction(self.depth, raw, self.x_perf, 20.0)
        second = self.regularizer.apply_correction(self.depth, raw, self.x_perf, 20.0)
        np.testing.assert_allclose(first["regularized_profile"], second["regularized_profile"])


if __name__ == "__main__":
    unittest.main()

