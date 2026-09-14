# -*- coding: utf-8 -*-
"""
tests/test_challenger_adversarial_stress.py
-------------------------------------------
Independent Empirical Adversarial Stress Test Suite (Challenger 2).

Verifies three fundamental physical/methodological dimensions:
1. Rayleigh Spatial Resolution Limit:
   - Quantitative evaluation of 1D and 2D cepstrum on sub-resolution spacing (delta_x = 5m)
     vs resolvable spacing (delta_x = 20m, 35m, 50m) against theoretical delta_d_min approx a / (2 * B_coh).
   - Verifies peak merging and contrast collapse at sub-resolution spacing.
   - Audits whether production code used genuine peak resolution or metadata shortcut.
2. Anti-Leakage Blind Peak Detection:
   - Verifies primary fracture identification under strictly blind conditions (zero coordinate leakage).
   - Evaluates across all 84 simulation cases, proving >=90% success rate on identifiable cases.
   - Identifies physical failure mode at micro-compliance (C_H = 1e-7 m^2).
   - Quantifies the Brunone boundary layer clock skew (+4.54 m).
   - Audits ground-truth leakage in production extract_features.py (lines 277-279).
3. Damping Confusion Zone:
   - Verifies quantitative distinguishability between pipe wall shear dissipation (Brunone)
     and fracture leakoff dissipation (k_leak) via high-frequency spectral ratios (R_high).
   - Demonstrates that R_high breaks the time-domain alpha_RMS degeneracy.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "output" / "fracture_parameter_sensitivity" / "data"
METRICS_CSV = PROJECT_ROOT / "output" / "fracture_parameter_sensitivity" / "tables" / "sensitivity_metrics.csv"


def compute_coherent_bandwidth(
    H_wh: np.ndarray, t: np.ndarray, dt: float, a_adj: float, L: float, dr_db: float = 80.0
) -> Tuple[float, float]:
    mask_fft = t >= 0.55
    x = H_wh[mask_fft] - np.mean(H_wh[mask_fft])
    n = len(x)
    freqs = fftfreq(n, dt)[: n // 2]
    spec = np.abs(fft(x)[: n // 2]) * (2.0 / n)

    f0 = a_adj / (4.0 * L)
    eps = 10.0 ** (-dr_db / 20.0)
    peak_mag = float(np.max(spec))
    thr = eps * peak_mag
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else f0
    max_gap_bins = max(1, int(round(0.5 * f0 / df)))

    above = spec >= thr
    last = 0
    gap = 0
    for i in range(len(spec)):
        if above[i]:
            last = i
            gap = 0
        else:
            gap += 1
            if i > 0 and gap > max_gap_bins:
                break
    B_coh = float(freqs[last])
    delta_d_min = float(a_adj / (2.0 * B_coh)) if B_coh > 0 else float(L)
    return B_coh, delta_d_min


def compute_1d_cepstrum(H_wh: np.ndarray, t: np.ndarray, dt: float, a_adj: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = t >= 0.5
    x = H_wh[mask] - np.mean(H_wh[mask])
    n = len(x)
    fs = 1.0 / dt
    spec = fft(x)
    log_spec = np.log(np.abs(spec) + 1e-12)
    raw_ceps = np.real(ifft(log_spec))
    rown = n // 2 + 1
    resp = -raw_ceps[:rown]
    depth = (np.arange(rown) / fs) * a_adj / 2.0
    return depth, resp


def test_rayleigh_spatial_resolution_sub_vs_resolvable():
    assert DATA_DIR.exists(), f"Data dir not found: {DATA_DIR}"

    spacing_test_cases = [
        (50, "steady", 5.0, False),
        (51, "brunone", 5.0, False),
        (54, "steady", 20.0, True),
        (55, "brunone", 20.0, True),
        (56, "steady", 35.0, True),
        (57, "brunone", 35.0, True),
        (58, "steady", 50.0, True),
        (59, "brunone", 50.0, True),
    ]

    for cid, frict, dx, expect_resolved in spacing_test_cases:
        npz_file = DATA_DIR / f"case_{cid:05d}.npz"
        assert npz_file.exists(), f"File missing: {npz_file}"

        with np.load(npz_file) as d:
            t = np.asarray(d["t"], dtype=float)
            H_wh = np.asarray(d["H_wh"], dtype=float)
            dt = float(d["dt_adj"])
            a_adj = float(d["wavespeed_adj"])
            L = float(d["wellbore_length"])
            xf = np.asarray(d["x_f_aligned"], dtype=float)

        B_coh, delta_d_min = compute_coherent_bandwidth(H_wh, t, dt, a_adj, L)
        depth, resp = compute_1d_cepstrum(H_wh, t, dt, a_adj)

        assert 5.0 <= delta_d_min <= 15.0, f"Case {cid}: unexpected delta_d_min={delta_d_min:.2f}"

        # Examine region enclosing all 3 fractures with allowance for clock skew
        clock_skew = 5.0 if frict == "brunone" else 0.0
        cluster_start = xf[0] - 2.0 + clock_skew
        cluster_end = xf[-1] + 2.0 + clock_skew

        sub = (depth >= cluster_start) & (depth <= cluster_end)
        d_sub = depth[sub]
        r_sub = resp[sub]

        depth_bin = d_sub[1] - d_sub[0]
        min_bins = max(1, int(round(2.5 / depth_bin)))
        pks, props = find_peaks(r_sub, height=max(np.max(r_sub) * 0.20, 0.0005), distance=min_bins)

        if not expect_resolved:
            # Sub-resolution spacing: dx = 5.0 m < delta_d_min (approx 6.1 - 6.8 m)
            # The 3 fractures must merge into a single dominant peak inside the cluster span
            assert dx < delta_d_min, f"Case {cid}: dx={dx} should be < delta_d_min={delta_d_min}"
            assert len(pks) == 1, (
                f"Case {cid}: Expected exactly 1 merged peak inside cluster span at dx=5m, got {len(pks)}"
            )
        else:
            # Resolvable spacing: dx = 20, 35, 50 m > delta_d_min
            assert dx > delta_d_min, f"Case {cid}: dx={dx} should be > delta_d_min={delta_d_min}"
            # Must resolve multiple distinct peaks
            assert len(pks) >= 2, f"Case {cid}: expected >=2 peaks at dx={dx}m, got {len(pks)}"


def test_audit_production_rayleigh_implementation_shortcut():
    df = pd.read_csv(METRICS_CSV)
    spacing_cases = df[df["group"] == "oat_spacing"]

    for _, row in spacing_cases.iterrows():
        cid = int(row["case_id"])
        npz_file = DATA_DIR / f"case_{cid:05d}.npz"
        with np.load(npz_file) as d:
            xf = np.asarray(d["x_f_aligned"], dtype=float)
        min_sp = float(np.min(np.diff(xf)))
        d_min = float(row["ceps_2d_spatial_resolution_m"])
        reported_success = int(row["ceps_2d_separation_success"])

        expected_shortcut = 0 if min_sp < d_min * 0.85 else 1
        assert reported_success == expected_shortcut, (
            f"Case {cid}: separation_success ({reported_success}) did not match metadata formula"
        )


def test_anti_leakage_blind_peak_detection():
    df = pd.read_csv(METRICS_CSV)
    blind_results = []
    
    for _, row in df.iterrows():
        cname = row["case_name"]
        npz_file = DATA_DIR / f"{cname}.npz"
        with np.load(npz_file) as d:
            t = np.asarray(d["t"], dtype=float)
            H_wh = np.asarray(d["H_wh"], dtype=float)
            dt = float(d["dt_adj"])
            a_adj = float(d["wavespeed_adj"])
            L = float(d["wellbore_length"])
            xf = np.asarray(d["x_f_aligned"], dtype=float)

        depth, resp = compute_1d_cepstrum(H_wh, t, dt, a_adj)

        valid = (depth >= 100.0) & (depth <= L - 50.0)
        depth_v = depth[valid]
        resp_v = resp[valid]

        rmax = float(np.max(resp_v))
        pct_thr = float(np.percentile(resp_v, 90))
        thresh = max(pct_thr, 0.05 * rmax)
        dist_bins = max(1, int(round(5.0 / (depth_v[1] - depth_v[0]))))

        peaks, props = find_peaks(resp_v, height=thresh, distance=dist_bins)

        if len(peaks) > 0:
            top_order = np.argsort(props["peak_heights"])[::-1]
            rank1_idx = peaks[top_order[0]]
            blind_depth = float(depth_v[rank1_idx])
            blind_amp = float(resp_v[rank1_idx])
        else:
            blind_depth = float(depth_v[np.argmax(resp_v)])
            blind_amp = float(np.max(resp_v))

        min_dist_to_cluster = min(abs(blind_depth - f) for f in xf)
        dist_to_xf0 = abs(blind_depth - xf[0])

        blind_results.append({
            "case_id": int(row["case_id"]),
            "friction": str(row["friction"]),
            "group": str(row["group"]),
            "param_name": str(row["param_name"]),
            "param_value": str(row["param_value"]),
            "xf0": float(xf[0]),
            "blind_depth": blind_depth,
            "dist_to_xf0": dist_to_xf0,
            "min_dist_to_cluster": min_dist_to_cluster,
            "success": min_dist_to_cluster <= 10.0,
        })

    res_df = pd.DataFrame(blind_results)
    
    overall_success_rate = res_df["success"].mean()
    assert overall_success_rate >= 0.90, f"Blind detection success rate {overall_success_rate:.2%} < 90%"

    failures = res_df[~res_df["success"]]
    assert len(failures) <= 6, f"Unexpected number of failures: {len(failures)}"
    for _, f_row in failures.iterrows():
        assert ("1e-07" in str(f_row["param_value"]) or "1e-06" in str(f_row["param_value"])), (
            f"Failure in unpredicted case: {f_row['case_id']} ({f_row['param_name']}={f_row['param_value']})"
        )

    successful_cases = res_df[res_df["success"]]
    steady_cases = successful_cases[successful_cases["friction"] == "steady"]
    brunone_cases = successful_cases[successful_cases["friction"] == "brunone"]

    # Steady cases have sub-bin localization precision (< 0.20 m)
    mean_cluster_dist_steady = steady_cases["min_dist_to_cluster"].mean()
    assert mean_cluster_dist_steady < 0.20, f"Steady mean cluster distance {mean_cluster_dist_steady:.3f}m >= 0.20m"

    # Brunone cases exhibit systematic clock skew (+4.54 m +/- 0.6 m)
    mean_cluster_dist_brunone = brunone_cases["min_dist_to_cluster"].mean()
    assert 4.0 <= mean_cluster_dist_brunone <= 5.2, (
        f"Brunone clock skew {mean_cluster_dist_brunone:.3f}m outside expected [4.0, 5.2]m window"
    )



def test_audit_production_ground_truth_leakage():
    df = pd.read_csv(METRICS_CSV)
    c10 = df[df["case_id"] == 10].iloc[0]
    assert c10["ceps_1d_depth_error_m"] < 5.0

    npz_10 = DATA_DIR / "case_00010.npz"
    with np.load(npz_10) as d:
        depth, resp = compute_1d_cepstrum(d["H_wh"], d["t"], float(d["dt_adj"]), float(d["wavespeed_adj"]))
    valid = (depth >= 100.0) & (depth <= 4950.0)
    blind_peak_depth = depth[valid][np.argmax(resp[valid])]
    assert abs(blind_peak_depth - 2980.0) < 5.0, "True highest peak in Case 10 must be near 2980m"
    assert abs(blind_peak_depth - 4000.0) > 900.0, "True blind detection failed by >900m due to micro-compliance!"


def test_damping_confusion_zone_and_spectral_ratio_separation():
    df = pd.read_csv(METRICS_CSV)
    kdf = df[df["group"] == "oat_kleak"]

    c21 = kdf[kdf["case_id"] == 21].iloc[0]
    c24 = kdf[kdf["case_id"] == 24].iloc[0]

    alpha_diff_pct_A = abs(c21["rms_decay_alpha_per_s"] - c24["rms_decay_alpha_per_s"]) / c24["rms_decay_alpha_per_s"] * 100.0
    rhigh_diff_pct_A = abs(c21["high_freq_power_ratio_pct"] - c24["high_freq_power_ratio_pct"]) / c24["high_freq_power_ratio_pct"] * 100.0

    assert alpha_diff_pct_A < 7.0, f"Pair A: alpha diff {alpha_diff_pct_A:.2f}% expected < 7%"
    assert rhigh_diff_pct_A > 20.0, f"Pair A: R_high diff {rhigh_diff_pct_A:.2f}% expected > 20%"

    c29 = kdf[kdf["case_id"] == 29].iloc[0]
    c30 = kdf[kdf["case_id"] == 30].iloc[0]

    alpha_diff_pct_B = abs(c29["rms_decay_alpha_per_s"] - c30["rms_decay_alpha_per_s"]) / c30["rms_decay_alpha_per_s"] * 100.0
    rhigh_diff_pct_B = abs(c29["high_freq_power_ratio_pct"] - c30["high_freq_power_ratio_pct"]) / c30["high_freq_power_ratio_pct"] * 100.0

    assert alpha_diff_pct_B < 1.0, f"Pair B: alpha diff {alpha_diff_pct_B:.2f}% expected < 1.0%"
    assert rhigh_diff_pct_B > 150.0, f"Pair B: R_high diff {rhigh_diff_pct_B:.2f}% expected > 150%"


if __name__ == "__main__":
    print("Running independent adversarial stress test suite...")
    test_rayleigh_spatial_resolution_sub_vs_resolvable()
    print("[PASS] test_rayleigh_spatial_resolution_sub_vs_resolvable")
    test_audit_production_rayleigh_implementation_shortcut()
    print("[PASS] test_audit_production_rayleigh_implementation_shortcut")
    test_anti_leakage_blind_peak_detection()
    print("[PASS] test_anti_leakage_blind_peak_detection")
    test_audit_production_ground_truth_leakage()
    print("[PASS] test_audit_production_ground_truth_leakage")
    test_damping_confusion_zone_and_spectral_ratio_separation()
    print("[PASS] test_damping_confusion_zone_and_spectral_ratio_separation")
    print("All independent challenger tests PASSED successfully!")
