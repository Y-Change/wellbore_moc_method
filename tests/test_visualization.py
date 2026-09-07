# -*- coding: utf-8 -*-
"""可视化适配层的快速回归测试。"""
from __future__ import annotations

import unittest
import time

import numpy as np

from moc_simulate.visualization import (
    VisualizationRunConfig,
    export_pair_json,
    export_pair_npz,
    export_summary_png,
    run_visualization_pair,
    start_live_visualization,
    validate_run_config,
)
from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore


class VisualizationAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VisualizationRunConfig(
            total_time_s=0.2,
            shut_in_time_s=0.05,
            wavespeed_mps=100.0,
            initial_velocity_mps=0.5,
            fracture_count=2,
            first_fracture_m=100.0,
            fracture_spacing_m=20.0,
            fracture_compliance_m2=1.0e-5,
            leakoff_coefficient=1.0e-4,
            external_head_m=100.0,
            time_step_s=0.01,
            max_display_frames=21,
        )

    def test_pair_has_finite_matching_spatial_frames(self) -> None:
        pair = run_visualization_pair(self.config)
        self.assertEqual(pair.steady.head_frames.shape, pair.brunone.head_frames.shape)
        self.assertEqual(pair.steady.velocity_frames.shape, pair.brunone.velocity_frames.shape)
        self.assertEqual(pair.steady.fracture_heads.shape[1], 2)
        self.assertEqual(pair.brunone.fracture_flows.shape[1], 2)
        self.assertTrue(np.isfinite(pair.steady.head_frames).all())
        self.assertTrue(np.isfinite(pair.brunone.head_frames).all())

    def test_steady_adapter_matches_direct_solver(self) -> None:
        pair = run_visualization_pair(self.config)
        direct_cfg = MocConfig(
            wellbore_length=5000.0,
            wellbore_diameter=0.1397,
            fluid_density=1000.0,
            fluid_viscosity=1.0e-6,
            wavespeed=self.config.wavespeed_mps,
            roughness_height=4.5e-5,
            friction_model="steady",
            dt=self.config.time_step_s,
            tf=self.config.total_time_s,
            wellhead_bc="velocity_step",
            pump_shut_time=self.config.shut_in_time_s,
            initial_velocity=self.config.initial_velocity_mps,
            initial_head=300.0,
            theta=0.0,
            toe_bc="reservoir",
            toe_head=300.0,
        )
        direct = simulate_wellbore(
            direct_cfg,
            fracture_positions=self.config.fracture_positions_m,
            fracture_Cf=[self.config.fracture_compliance_m2] * 2,
            fracture_kleak=[self.config.leakoff_coefficient] * 2,
            H_ext=self.config.external_head_m,
            store_full_field=False,
        )
        np.testing.assert_allclose(pair.steady.wellhead_head, direct["wellhead_head"])

    def test_grid_collision_is_rejected(self) -> None:
        invalid = VisualizationRunConfig(
            **{**self.config.__dict__, "fracture_spacing_m": 0.01}
        )
        with self.assertRaisesRegex(ValueError, "同一节点"):
            validate_run_config(invalid)

    def test_exports_include_visualization_artifacts(self) -> None:
        pair = run_visualization_pair(self.config)
        self.assertIn(b"run_config", export_pair_json(pair))
        self.assertGreater(len(export_pair_npz(pair)), 100)
        self.assertTrue(export_summary_png(pair).startswith(b"\x89PNG"))

    def test_live_pair_streams_complete_pressure_frames(self) -> None:
        live = start_live_visualization(self.config, max_frames=8)
        deadline = time.monotonic() + 8.0
        snapshot = live.snapshot()
        while not snapshot.is_finished and time.monotonic() < deadline:
            time.sleep(0.01)
            snapshot = live.snapshot()

        self.assertTrue(snapshot.is_finished, snapshot.error)
        self.assertIsNone(snapshot.error)
        for model in (snapshot.steady, snapshot.brunone):
            self.assertGreater(len(model.frame_t), 1)
            self.assertEqual(model.head_frames.shape, (len(model.frame_t), len(snapshot.x_grid)))
            self.assertEqual(len(model.wellhead_t), len(model.wellhead_head))
            self.assertAlmostEqual(model.frame_t[-1], self.config.total_time_s, places=7)
            self.assertTrue(np.isfinite(model.head_frames).all())


if __name__ == "__main__":
    unittest.main()
