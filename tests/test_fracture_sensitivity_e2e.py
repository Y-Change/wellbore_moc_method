# -*- coding: utf-8 -*-
"""
tests/test_fracture_sensitivity_e2e.py

End-to-End Acceptance Test Suite for Fracture Parameter Sensitivity Ablation Experiment.
Directly verifies all five acceptance criteria tiers specified in ORIGINAL_REQUEST.md and PROJECT.md:

Tier 1: Simulation Convergence & Time Series Completeness
        - Existence of output/fracture_parameter_sensitivity/data/case_*.npz and timeseries_csv/case_*.csv
        - 100% convergence across all generated cases
        - Zero NaN / Inf in all arrays and time series
        - Valid shapes (len(t) >= 100) and non-trivial water hammer amplitudes (np.ptp(H_wh) > 1.0 m)

Tier 2: Schema moc_lhs_v2.1 Compatibility & Physical Constraints
        - Presence of all 41 mandatory metadata and array keys
        - schema_version == 'moc_lhs_v2.1'
        - friction in ['steady', 'brunone']
        - Absolute pressure margin min(Hf - H_ext) > 0 (or min(Hf_ss - H_ext) > 0)
        - Alias consistency: x_f == x_f_aligned, x_f_raw == x_f_requested, grid_index == fracture_indices,
          Cf == compliance_head_m2, kleak == kleak_equiv, dt == dt_adj, wavespeed == wavespeed_adj
        - Physical validity: Cf >= 0, kleak >= 0, wi >= 0 with sum(wi) == 1.0 (for n_frac > 0)

Tier 3: Metrics Table & Summary JSON Completeness
        - output/fracture_parameter_sensitivity/tables/sensitivity_metrics.csv:
          * Contains Joukowsky drop, max gradient, 5-window RMS attenuation rate,
            FFT high-frequency (>1.5Hz) energy ratio, 1D real cepstrum peak depth and amplitude,
            and 2D sliding-window cepstrum spatial resolution.
          * Finite numeric values, zero NaN/Inf, covering both steady and brunone friction.
        - output/fracture_parameter_sensitivity/tables/sensitivity_summary.json:
          * Valid JSON with sensitivity rankings and impact levels for key parameters.

Tier 4: Publication Figures Verification
        - output/fracture_parameter_sensitivity/figures/ containing at least 4 figures:
          * fig1_wavefront_step_gradient.png (.svg)
          * fig2_envelope_rms_decay.png (.svg)
          * fig3_frequency_spectral_dissipation.png (.svg)
          * fig4_cepstrum_rayleigh_resolution.png (.svg)
        - Verified DPI >= 200 via PIL Image headers / dimensions
        - Companion SVG vector files present, non-empty, valid XML/SVG tags

Tier 5: Academic Research Report Completeness
        - output/fracture_parameter_sensitivity/README.md:
          * File size >= 5000 bytes
          * Contains all 8 required chapter headings:
            1. Executive Summary
            2. Physical Models & Discretization (or Physical Models and Discretization)
            3. Experimental Design
            4. Waveform Step & Gradient Sensitivity
            5. Dual Friction Attenuation & Damping Confusion
            6. Frequency Dissipation & Energy Ratios
            7. Cepstrum Peak Response & Rayleigh Resolution Limit
            8. Implications for Deep Neural Operator Inversion

Execution Modes:
- Strict mode (default): Asserts artifact presence and physical properties. Fails if files are missing.
- Progressive mode: Set environment variable E2E_ALLOW_SKIP=1 to skip tests gracefully when awaiting upstream steps.
"""

import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Authoritative Contract Constants
# ---------------------------------------------------------------------------

SCHEMA_V2_1_MANDATORY_KEYS: Set[str] = {
    # 1. Version & reproducibility (3)
    "schema_version",
    "seed",
    "case_id",
    # 2. Grid & acoustic parameters (10)
    "N",
    "wellbore_length",
    "dx",
    "dt_requested",
    "dt_adj",
    "dt",
    "wavespeed_requested",
    "wavespeed_adj",
    "wavespeed",
    "wavespeed_nominal",
    # 3. Spatial coordinates & alignment (6)
    "x_f_requested",
    "x_f_aligned",
    "x_f",
    "x_f_raw",
    "fracture_indices",
    "grid_index",
    # 4. Wellhead observation time series (3)
    "t",
    "H_wh",
    "Q_wh",
    # 5. Boundary & operating conditions (7)
    "friction",
    "brunone_k_scale",
    "toe_bc",
    "initial_head",
    "initial_velocity",
    "H_ext",
    "tf",
    # 6. Fracture physical properties (7)
    "n_frac",
    "Rp",
    "compliance_head_m2",
    "Cf",
    "inflow_weight",
    "kleak_equiv",
    "kleak",
    # 7. Steady-state closure & diagnostics (5)
    "Hw_ss",
    "Hf_ss",
    "Qf_ss",
    "Qin_ss",
    "alpha_dirichlet",
}

REQUIRED_FIGURE_NAMES: List[str] = [
    "fig1_wavefront_step_gradient",
    "fig2_envelope_rms_decay",
    "fig3_frequency_spectral_dissipation",
    "fig4_cepstrum_rayleigh_resolution",
]

REQUIRED_REPORT_CHAPTERS: Dict[str, str] = {
    "Executive Summary": r"executive\s+summary",
    "Physical Models & Discretization": r"physical\s+models?\s*(?:&|and)\s*discretization",
    "Experimental Design": r"experimental\s+design",
    "Waveform Step & Gradient Sensitivity": r"waveform\s+step\s*(?:&|and)\s*gradient\s+sensitivity",
    "Dual Friction Attenuation & Damping Confusion": r"dual\s+friction\s+attenuation\s*(?:&|and)\s*damping\s+confusion",
    "Frequency Dissipation & Energy Ratios": r"frequency\s+dissipation\s*(?:&|and)\s*energy\s+ratios?",
    "Cepstrum Peak Response & Rayleigh Resolution Limit": r"cepstrum\s+peak\s+response\s*(?:&|and)\s*rayleigh\s+resolution\s+limit",
    "Implications for Deep Neural Operator Inversion": r"implications\s+for\s+deep\s+neural\s+operator\s+inversion",
}

# ---------------------------------------------------------------------------
# Helper Navigation and Verification Functions
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_sensitivity_dir() -> Path:
    """Return the root output directory for fracture parameter sensitivity experiments."""
    return get_project_root() / "output" / "fracture_parameter_sensitivity"


def check_artifact(condition: bool, failure_msg: str, artifact_name: str = "artifact") -> None:
    """
    Check an artifact condition.
    If condition is False:
      - If E2E_ALLOW_SKIP=1, skip the test with diagnostic context.
      - Otherwise, fail with an explicit AssertionError.
    """
    if not condition:
        if os.environ.get("E2E_ALLOW_SKIP", "").lower() in ("1", "true", "yes"):
            pytest.skip(f"[PROGRESSIVE SKIP] Pending downstream {artifact_name}: {failure_msg}")
        else:
            pytest.fail(f"[E2E DEFECT] {failure_msg}")


def get_npz_case_paths() -> List[Path]:
    """Retrieve all case NPZ files in output/fracture_parameter_sensitivity/data/."""
    data_dir = get_sensitivity_dir() / "data"
    if not data_dir.exists():
        return []
    return sorted(list(data_dir.glob("case_*.npz")))


def get_csv_case_paths() -> List[Path]:
    """Retrieve all case CSV files in output/fracture_parameter_sensitivity/timeseries_csv/."""
    csv_dir = get_sensitivity_dir() / "timeseries_csv"
    if not csv_dir.exists():
        return []
    return sorted(list(csv_dir.glob("case_*.csv")))


# ---------------------------------------------------------------------------
# Test Suite Contract Verification (Self-Contained Invariant Test)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_e2e_contract_and_metadata_spec():
    """
    Verify test suite contract definitions:
    1. Schema v2.1 contains exactly 41 required keys.
    2. Exactly 4 publication figure plates are defined.
    3. Exactly 8 academic report chapter patterns are defined.
    4. Project root and output paths are valid and well-formed.
    """
    assert len(SCHEMA_V2_1_MANDATORY_KEYS) == 41, (
        f"Contract mismatch: Expected exactly 41 schema keys, found {len(SCHEMA_V2_1_MANDATORY_KEYS)}"
    )
    assert len(REQUIRED_FIGURE_NAMES) == 4
    assert len(REQUIRED_REPORT_CHAPTERS) == 8
    assert get_project_root().exists()
    assert (get_project_root() / "tests").exists()


# ---------------------------------------------------------------------------
# Tier 1: Simulation Convergence & Time Series Completeness
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier1
class TestTier1SimulationConvergence:
    """Acceptance Tier 1: Output existence, 100% convergence, and finite non-trivial signals."""

    def test_tier1_output_directories_exist(self):
        """Verify data and timeseries_csv output directories exist."""
        sens_dir = get_sensitivity_dir()
        data_dir = sens_dir / "data"
        csv_dir = sens_dir / "timeseries_csv"

        check_artifact(
            sens_dir.exists(),
            f"Sensitivity output directory does not exist: {sens_dir}",
            artifact_name="output directory"
        )
        check_artifact(
            data_dir.exists(),
            f"Data directory does not exist: {data_dir}",
            artifact_name="data directory"
        )
        check_artifact(
            csv_dir.exists(),
            f"CSV timeseries directory does not exist: {csv_dir}",
            artifact_name="timeseries_csv directory"
        )

    def test_tier1_case_files_exist_and_counts_match(self):
        """Verify NPZ and CSV simulation files exist and are paired."""
        npz_files = get_npz_case_paths()
        csv_files = get_csv_case_paths()

        check_artifact(
            len(npz_files) > 0,
            f"No case_*.npz simulation files found in {get_sensitivity_dir() / 'data'}. "
            f"Expected batch simulation cases (e.g. >= 10 cases).",
            artifact_name="NPZ files"
        )
        check_artifact(
            len(csv_files) > 0,
            f"No case_*.csv simulation files found in {get_sensitivity_dir() / 'timeseries_csv'}.",
            artifact_name="CSV files"
        )
        check_artifact(
            len(npz_files) == len(csv_files),
            f"File count mismatch: {len(npz_files)} NPZ files vs {len(csv_files)} CSV files.",
            artifact_name="file count parity"
        )

    def test_tier1_npz_files_convergence_and_finite(self):
        """Verify 100% convergence, zero NaN/Inf, valid shape, and non-trivial amplitude in all NPZ files."""
        npz_files = get_npz_case_paths()
        check_artifact(len(npz_files) > 0, "No NPZ files to verify for Tier 1 convergence.", "NPZ dataset")

        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception as e:
                pytest.fail(f"Failed to load NPZ file {npz_path.name}: {e}")

            # 1. Zero NaN / Inf in core observation arrays
            for arr_name in ["t", "H_wh", "Q_wh"]:
                assert arr_name in data, f"Key '{arr_name}' missing from {npz_path.name}"
                arr = np.asarray(data[arr_name])
                assert arr.ndim == 1, f"{arr_name} in {npz_path.name} must be 1D, got ndim={arr.ndim}"
                assert len(arr) >= 100, f"{arr_name} in {npz_path.name} length {len(arr)} < 100"
                assert np.all(np.isfinite(arr)), f"Non-finite (NaN or Inf) detected in {arr_name} in {npz_path.name}"

            # Array length consistency
            assert len(data["H_wh"]) == len(data["t"]), (
                f"Length mismatch in {npz_path.name}: len(H_wh)={len(data['H_wh'])} != len(t)={len(data['t'])}"
            )
            assert len(data["Q_wh"]) == len(data["t"]), (
                f"Length mismatch in {npz_path.name}: len(Q_wh)={len(data['Q_wh'])} != len(t)={len(data['t'])}"
            )

            # 2. Time monotonically increasing
            t = np.asarray(data["t"], dtype=float)
            dt_diff = np.diff(t)
            assert np.all(dt_diff > 0.0), f"Time array 't' in {npz_path.name} is not strictly monotonic"

            # 3. Non-trivial water hammer wave amplitude (not a dead/flatline signal)
            H_wh = np.asarray(data["H_wh"], dtype=float)
            ptp_head = float(np.ptp(H_wh))
            assert ptp_head > 1.0, (
                f"Trivial head oscillation in {npz_path.name}: ptp(H_wh) = {ptp_head:.3f} m <= 1.0 m"
            )

            # 4. Convergence status flag if present
            if "status" in data:
                assert str(data["status"]).upper() == "PASS", (
                    f"Simulation status in {npz_path.name} is '{data['status']}', expected 'PASS'"
                )
            if "convergence_status" in data:
                assert str(data["convergence_status"]).upper() == "PASS", (
                    f"Convergence status in {npz_path.name} is '{data['convergence_status']}', expected 'PASS'"
                )

    def test_tier1_csv_timeseries_validity(self):
        """Verify timeseries CSV files are non-empty, parse cleanly, and contain no NaN/Inf."""
        csv_files = get_csv_case_paths()
        check_artifact(len(csv_files) > 0, "No CSV files to verify for Tier 1.", "timeseries CSV")

        for csv_path in csv_files:
            assert csv_path.stat().st_size > 100, f"CSV file {csv_path.name} is too small or empty"
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                pytest.fail(f"Failed to read CSV file {csv_path.name}: {e}")

            # Verify mandatory observation columns
            for col in ["t", "H_wh", "Q_wh"]:
                assert col in df.columns, f"Mandatory column '{col}' missing from {csv_path.name}"
                assert len(df[col]) >= 100, f"Insufficient rows in {csv_path.name}: {len(df[col])} < 100"
                assert not df[col].isna().any(), f"NaN values detected in column '{col}' of {csv_path.name}"
                assert not np.isinf(df[col].to_numpy()).any(), f"Inf values in column '{col}' of {csv_path.name}"


# ---------------------------------------------------------------------------
# Tier 2: Schema moc_lhs_v2.1 Compatibility & Physical Constraints
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier2
class TestTier2SchemaCompatibility:
    """Acceptance Tier 2: Schema moc_lhs_v2.1 41-key compliance and physical viability."""

    def test_tier2_all_41_keys_present_in_every_npz(self):
        """Verify that every case NPZ file contains all 41 mandatory schema keys."""
        npz_files = get_npz_case_paths()
        check_artifact(len(npz_files) > 0, "No NPZ files found for Tier 2 schema audit.", "NPZ dataset")

        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as data:
                present_keys = set(data.files)
                missing_keys = SCHEMA_V2_1_MANDATORY_KEYS - present_keys
                assert not missing_keys, (
                    f"File {npz_path.name} is missing {len(missing_keys)} schema keys: {sorted(missing_keys)}"
                )

    def test_tier2_schema_version_and_friction_models(self):
        """Verify schema_version=='moc_lhs_v2.1' and friction model is 'steady' or 'brunone'."""
        npz_files = get_npz_case_paths()
        check_artifact(len(npz_files) > 0, "No NPZ files found for Tier 2.", "NPZ dataset")

        found_frictions = set()
        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as data:
                schema_ver = str(data["schema_version"])
                assert schema_ver == "moc_lhs_v2.1", (
                    f"Invalid schema_version '{schema_ver}' in {npz_path.name}, expected 'moc_lhs_v2.1'"
                )

                friction = str(data["friction"]).lower()
                assert friction in ("steady", "brunone"), (
                    f"Invalid friction model '{friction}' in {npz_path.name}, expected 'steady' or 'brunone'"
                )
                found_frictions.add(friction)

        # Batch simulation must cover both steady and brunone for dual friction ablation
        assert "steady" in found_frictions, "No 'steady' friction cases found in dataset"
        assert "brunone" in found_frictions, "No 'brunone' friction cases found in dataset"

    def test_tier2_positive_head_margin(self):
        """Verify positive pressure head margin: min(Hf - H_ext) > 0 (prevents unphysical cavitation/shutoff)."""
        npz_files = get_npz_case_paths()
        check_artifact(len(npz_files) > 0, "No NPZ files found for Tier 2 margin check.", "NPZ dataset")

        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as data:
                H_ext = float(data["H_ext"])

                # Check steady-state fracture head margin
                if "Hf_ss" in data:
                    Hf_ss = np.asarray(data["Hf_ss"], dtype=float)
                    min_margin_ss = float(np.min(Hf_ss - H_ext))
                    assert min_margin_ss > 0.0, (
                        f"Negative or zero steady head margin in {npz_path.name}: "
                        f"min(Hf_ss - H_ext) = {min_margin_ss:.4f} m <= 0.0"
                    )

                # Check transient fracture head margin if transient array is stored
                if "Hf" in data:
                    Hf = np.asarray(data["Hf"], dtype=float)
                    min_margin_trans = float(np.min(Hf - H_ext))
                    assert min_margin_trans > 0.0, (
                        f"Transient head margin violation in {npz_path.name}: "
                        f"min(Hf(t) - H_ext) = {min_margin_trans:.4f} m <= 0.0"
                    )

                # Check explicit margin diagnostic if recorded
                if "min_head_margin_m" in data:
                    min_recorded = float(data["min_head_margin_m"])
                    assert min_recorded > 0.0, (
                        f"Recorded min_head_margin_m in {npz_path.name} = {min_recorded} <= 0"
                    )

    def test_tier2_aliases_and_physical_constraints(self):
        """Verify alias consistency and physical parameter constraints in every NPZ."""
        npz_files = get_npz_case_paths()
        check_artifact(len(npz_files) > 0, "No NPZ files found for Tier 2 alias check.", "NPZ dataset")

        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as data:
                # 1. Alias consistency
                np.testing.assert_allclose(
                    data["x_f"], data["x_f_aligned"],
                    err_msg=f"Alias mismatch x_f != x_f_aligned in {npz_path.name}"
                )
                np.testing.assert_allclose(
                    data["x_f_raw"], data["x_f_requested"],
                    err_msg=f"Alias mismatch x_f_raw != x_f_requested in {npz_path.name}"
                )
                np.testing.assert_array_equal(
                    data["grid_index"], data["fracture_indices"],
                    err_msg=f"Alias mismatch grid_index != fracture_indices in {npz_path.name}"
                )
                np.testing.assert_allclose(
                    data["Cf"], data["compliance_head_m2"],
                    err_msg=f"Alias mismatch Cf != compliance_head_m2 in {npz_path.name}"
                )
                np.testing.assert_allclose(
                    data["kleak"], data["kleak_equiv"],
                    err_msg=f"Alias mismatch kleak != kleak_equiv in {npz_path.name}"
                )
                assert float(data["dt"]) == float(data["dt_adj"]), (
                    f"Alias mismatch dt != dt_adj in {npz_path.name}"
                )
                assert float(data["wavespeed"]) == float(data["wavespeed_adj"]), (
                    f"Alias mismatch wavespeed != wavespeed_adj in {npz_path.name}"
                )

                # 2. Physical bounds
                Cf = np.asarray(data["compliance_head_m2"], dtype=float)
                kleak = np.asarray(data["kleak_equiv"], dtype=float)
                Rp = np.asarray(data["Rp"], dtype=float)
                wi = np.asarray(data["inflow_weight"], dtype=float)
                n_frac = int(data["n_frac"])

                assert np.all(Cf >= 0.0), f"Negative compliance detected in {npz_path.name}"
                assert np.all(kleak >= 0.0), f"Negative leakoff detected in {npz_path.name}"
                assert np.all(Rp >= 0.0), f"Negative Rp detected in {npz_path.name}"

                if n_frac > 0:
                    assert np.all(wi >= 0.0), f"Negative inflow weight in {npz_path.name}"
                    assert abs(float(np.sum(wi)) - 1.0) < 1e-6, (
                        f"Inflow weights do not sum to 1.0 in {npz_path.name}: sum={np.sum(wi)}"
                    )


# ---------------------------------------------------------------------------
# Tier 3: Metrics Table & Summary JSON Completeness
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier3
class TestTier3MetricsCompleteness:
    """Acceptance Tier 3: Structured sensitivity metrics CSV and summary JSON verification."""

    def test_tier3_metrics_csv_existence_and_schema(self):
        """Verify sensitivity_metrics.csv exists, parses, and contains all required feature columns."""
        csv_path = get_sensitivity_dir() / "tables" / "sensitivity_metrics.csv"
        check_artifact(csv_path.exists(), f"Metrics table not found at {csv_path}", "sensitivity_metrics.csv")
        assert csv_path.stat().st_size > 200, f"Metrics table at {csv_path} is empty or trivial"

        df = pd.read_csv(csv_path)
        assert len(df) >= 10, f"Metrics table has only {len(df)} rows, expected >= 10 cases"

        cols = [c.lower() for c in df.columns]

        # 1. Joukowsky drop
        has_joukowsky = any("jouk" in c for c in cols)
        assert has_joukowsky, f"No Joukowsky drop column found in metrics table. Columns: {df.columns.tolist()}"

        # 2. Wavefront max gradient
        has_gradient = any("grad" in c or "slope" in c for c in cols)
        assert has_gradient, f"No wavefront max gradient column found. Columns: {df.columns.tolist()}"

        # 3. 5-window RMS head attenuation rate
        has_rms = any("rms" in c or "alpha_rms" in c for c in cols)
        assert has_rms, f"No 5-window RMS attenuation column found. Columns: {df.columns.tolist()}"

        # 4. FFT high frequency >1.5Hz energy ratio
        has_high_freq = any("high_freq" in c or "power_high" in c for c in cols)
        assert has_high_freq, f"No high-frequency energy ratio column found. Columns: {df.columns.tolist()}"

        # 5. 1D real cepstrum peak depth and amplitude
        has_cep_1d_depth = any(("ceps" in c or "cep" in c) and "depth" in c for c in cols)
        has_cep_1d_amp = any(("ceps" in c or "cep" in c) and "amp" in c for c in cols)
        assert has_cep_1d_depth, f"No 1D cepstrum peak depth column found. Columns: {df.columns.tolist()}"
        assert has_cep_1d_amp, f"No 1D cepstrum peak amplitude column found. Columns: {df.columns.tolist()}"

        # 6. 2D sliding window cepstrum spatial resolution
        has_cep_2d_res = any(
            ("2d" in c or "rayleigh" in c or "spatial" in c or "fwhm" in c) for c in cols
        )
        assert has_cep_2d_res, f"No 2D cepstrum spatial resolution column found. Columns: {df.columns.tolist()}"

    def test_tier3_metrics_csv_data_validity(self):
        """Verify numeric completeness, zero NaN/Inf in metrics, and dual friction coverage."""
        csv_path = get_sensitivity_dir() / "tables" / "sensitivity_metrics.csv"
        check_artifact(csv_path.exists(), f"Metrics table not found at {csv_path}", "sensitivity_metrics.csv")

        df = pd.read_csv(csv_path)

        # Check friction column covers both steady and brunone
        friction_col = None
        for col in df.columns:
            if "friction" in col.lower():
                friction_col = col
                break
        assert friction_col is not None, f"No friction model column found in {df.columns.tolist()}"

        frictions_present = set(df[friction_col].astype(str).str.lower().unique())
        assert "steady" in frictions_present, f"'steady' friction missing from metrics: {frictions_present}"
        assert "brunone" in frictions_present, f"'brunone' friction missing from metrics: {frictions_present}"

        # Check all numeric columns for NaN / Inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 5, f"Too few numeric columns in metrics table: {numeric_cols.tolist()}"
        for col in numeric_cols:
            assert not df[col].isna().any(), f"NaN found in metrics column '{col}'"
            assert not np.isinf(df[col].to_numpy()).any(), f"Inf found in metrics column '{col}'"

    def test_tier3_summary_json_rankings_and_impact_levels(self):
        """Verify sensitivity_summary.json exists, parses, and contains sensitivity rankings."""
        json_path = get_sensitivity_dir() / "tables" / "sensitivity_summary.json"
        check_artifact(json_path.exists(), f"Summary JSON not found at {json_path}", "sensitivity_summary.json")
        assert json_path.stat().st_size > 100, f"Summary JSON at {json_path} is empty"

        with open(json_path, "r", encoding="utf-8") as f:
            try:
                summary = json.load(f)
            except Exception as e:
                pytest.fail(f"Invalid JSON format in {json_path}: {e}")

        # Check for sensitivity rankings section
        rankings_key = None
        for key in ["sensitivity_rankings", "rankings", "parameter_rankings", "importance_ranking"]:
            if key in summary:
                rankings_key = key
                break
        assert rankings_key is not None, (
            f"No sensitivity rankings section found in summary JSON. Keys: {list(summary.keys())}"
        )

        rankings_data = summary[rankings_key]
        assert isinstance(rankings_data, (dict, list)), (
            f"Rankings data should be dict or list, got {type(rankings_data)}"
        )
        assert len(rankings_data) > 0, "Sensitivity rankings section is empty"


# ---------------------------------------------------------------------------
# Tier 4: Publication Figures Verification
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier4
class TestTier4PublicationFigures:
    """Acceptance Tier 4: Verification of the 4 required publication figure plates (PNG >= 200 DPI, SVG)."""

    def test_tier4_all_four_png_figures_exist_and_non_empty(self):
        """Verify all 4 required PNG figure plates exist and are non-empty in output/fracture_parameter_sensitivity/figures/."""
        fig_dir = get_sensitivity_dir() / "figures"
        check_artifact(fig_dir.exists(), f"Figures directory does not exist: {fig_dir}", "figures directory")

        for fig_base in REQUIRED_FIGURE_NAMES:
            png_file = fig_dir / f"{fig_base}.png"
            check_artifact(
                png_file.exists(),
                f"Required figure plate missing: {png_file}",
                artifact_name=f"{fig_base}.png"
            )
            assert png_file.stat().st_size >= 5000, (
                f"Figure file {png_file.name} is too small ({png_file.stat().st_size} bytes < 5000 bytes)"
            )

    def test_tier4_figure_dpi_greater_than_200(self):
        """Verify image resolution >= 200 DPI via PIL Image headers and physical dimensions."""
        fig_dir = get_sensitivity_dir() / "figures"
        check_artifact(fig_dir.exists(), f"Figures directory not found: {fig_dir}", "figures directory")

        for fig_base in REQUIRED_FIGURE_NAMES:
            png_file = fig_dir / f"{fig_base}.png"
            check_artifact(png_file.exists(), f"Figure file missing: {png_file}", artifact_name=png_file.name)

            with Image.open(png_file) as img:
                assert img.format == "PNG", f"{png_file.name} is not a valid PNG image (format={img.format})"
                assert img.width >= 400 and img.height >= 400, (
                    f"{png_file.name} dimensions too small: {img.width}x{img.height}"
                )

                dpi = img.info.get("dpi")
                if dpi is not None:
                    if isinstance(dpi, (tuple, list)):
                        dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
                    else:
                        dpi_x = dpi_y = float(dpi)
                    assert dpi_x >= 200.0 and dpi_y >= 200.0, (
                        f"{png_file.name} DPI is {dpi}, below required threshold of 200 DPI"
                    )
                else:
                    # Fallback verification: publication dimensions for >= 200 DPI
                    total_pixels = img.width * img.height
                    min_required_pixels = 1200 * 800  # ~6x4 in at 200 DPI
                    assert total_pixels >= min_required_pixels, (
                        f"{png_file.name} missing DPI header and has insufficient pixels "
                        f"({img.width}x{img.height} = {total_pixels} < {min_required_pixels}) for 200 DPI"
                    )

    def test_tier4_companion_svg_files_exist_and_valid(self):
        """Verify companion vector SVG files exist, are non-empty, and contain valid SVG markup."""
        fig_dir = get_sensitivity_dir() / "figures"
        check_artifact(fig_dir.exists(), f"Figures directory not found: {fig_dir}", "figures directory")

        for fig_base in REQUIRED_FIGURE_NAMES:
            svg_file = fig_dir / f"{fig_base}.svg"
            check_artifact(
                svg_file.exists(),
                f"Companion SVG vector figure missing: {svg_file}",
                artifact_name=f"{fig_base}.svg"
            )
            assert svg_file.stat().st_size >= 1000, (
                f"SVG file {svg_file.name} size {svg_file.stat().st_size} < 1000 bytes"
            )
            with open(svg_file, "r", encoding="utf-8", errors="ignore") as f:
                svg_content = f.read()
            assert "<svg" in svg_content.lower() and "</svg>" in svg_content.lower(), (
                f"{svg_file.name} does not contain valid SVG root tags"
            )


# ---------------------------------------------------------------------------
# Tier 5: Academic Research Report Completeness
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier5
class TestTier5AcademicReport:
    """Acceptance Tier 5: Comprehensive academic research report (README.md) validation."""

    def test_tier5_readme_file_exists_and_size_ge_5000(self):
        """Verify output/fracture_parameter_sensitivity/README.md exists and size >= 5000 bytes."""
        readme_path = get_sensitivity_dir() / "README.md"
        check_artifact(
            readme_path.exists(),
            f"Academic report README.md not found at {readme_path}",
            artifact_name="academic report README.md"
        )
        file_size = readme_path.stat().st_size
        assert file_size >= 5000, (
            f"Academic report {readme_path} size {file_size} bytes is below the 5000 bytes threshold"
        )

    def test_tier5_all_eight_chapter_headings_present(self):
        """Verify that all 8 required chapter headings are present in the academic report."""
        readme_path = get_sensitivity_dir() / "README.md"
        check_artifact(readme_path.exists(), f"Academic report not found at {readme_path}", "README.md")

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        missing_chapters = []
        for chapter_name, pattern in REQUIRED_REPORT_CHAPTERS.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if not match:
                missing_chapters.append(chapter_name)

        assert not missing_chapters, (
            f"Academic report README.md is missing {len(missing_chapters)} required chapter headings: {missing_chapters}"
        )

    def test_tier5_substantive_chapter_content(self):
        """Verify chapters contain substantive technical text rather than empty headers."""
        readme_path = get_sensitivity_dir() / "README.md"
        check_artifact(readme_path.exists(), f"Academic report not found at {readme_path}", "README.md")

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Find positions of all matched chapters
        positions: List[Tuple[str, int]] = []
        for chapter_name, pattern in REQUIRED_REPORT_CHAPTERS.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                positions.append((chapter_name, match.start()))

        positions.sort(key=lambda x: x[1])
        for i in range(len(positions) - 1):
            ch_curr, pos_curr = positions[i]
            ch_next, pos_next = positions[i + 1]
            section_len = pos_next - pos_curr
            assert section_len >= 150, (
                f"Chapter '{ch_curr}' appears empty or superficial: section length is only {section_len} characters."
            )
