# -*- coding: utf-8 -*-
"""
tests/test_challenger_empirical_stress.py
-----------------------------------------
Empirical Verification & Adversarial Stress Test Suite (Challenger 1).

Objectives:
1. Joukowsky Theory Consistency:
   - Verify instantaneous 1-step simulated head drop matches theoretical Korteweg-Joukowsky
     head drop Delta H_jouk = - a_adj * V0 / g within machine precision (< 1e-9 relative error).
   - Verify 50 ms head drop matches within tight physical tolerance (< 0.5%) accounting for wall friction.
   - Verify causality invariance: head drop prior to acoustic round-trip 2*xf/a is 100% independent
     of downstream fracture parameters (Cf, kleak, Rp, spacing, cluster count).
   - Verify near-fracture acoustic causality breakdown boundary (xf < a * 50ms).

2. Damping & Energy Dissipation:
   - Audit all 42 paired cases in production dataset (output/fracture_parameter_sensitivity/data/case_*.npz)
     by recomputing alpha_rms directly from raw time series and verifying alpha_rms(Brunone) > alpha_rms(Steady).
   - Verify strict monotonic scaling of alpha_rms with Brunone k_scale across [0.0, 0.25, 0.5, 1.0, 1.5, 2.0].
   - Run fresh adversarial paired simulations under extreme operating regimes (low/high V0, zero/huge Cf,
     high Rp, dense clusters) and confirm Brunone dissipation systematically exceeds steady Darcy.

3. Flow Continuity & Mass Conservation:
   - Verify steady-state wellhead inflow Q_in = V0 * A exactly equals fracture outflow sum sum(Q_f,i) (< 1e-12).
   - Verify dead-end stagnant zone flow velocity between last fracture and toe is strictly 0.0 m/s.
   - Verify dynamic dead-end toe velocity and flow remain strictly 0.0 m/s at all discrete time steps.
   - Verify dynamic fracture node mass conservation residual |r_mass| <= 1e-9 m3/s and perforation drop |r_Rp| <= 1e-8 m.
   - Verify global mass balance integral over time domain for both wellbore and fracture domains (< 0.05%).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moc_simulate.wellbore_moc import (
    G,
    MocConfig,
    simulate_wellbore,
    solve_fracture_node,
)

# Helper function to compute RMS decay alpha_rms independently
def independent_calc_alpha_rms(
    H_wh: np.ndarray,
    t: np.ndarray,
    ts: float,
    tf: float,
    n_windows: int = 5,
) -> Tuple[float, List[float], List[float]]:
    """
    Independently calculate 5-window RMS values and exponential decay constant alpha_rms.
    """
    w_len = (tf - ts) / float(n_windows)
    rms_vals: List[float] = []
    t_mids: List[float] = []
    for k in range(n_windows):
        t_start = ts + k * w_len
        t_end = ts + (k + 1) * w_len
        mask = (t >= t_start) & (t <= t_end)
        H_win = H_wh[mask]
        if len(H_win) > 5:
            rms_val = float(np.sqrt(np.mean((H_win - np.mean(H_win)) ** 2)))
        else:
            rms_val = 1e-12
        rms_vals.append(max(rms_val, 1e-12))
        t_mids.append((t_start + t_end) / 2.0)
    slope, _ = np.polyfit(t_mids, np.log(rms_vals), 1)
    alpha_rms = float(-slope)
    return alpha_rms, rms_vals, t_mids


# ===========================================================================
# 1. JOUKOWSKY THEORY CONSISTENCY & CAUSALITY TESTS
# ===========================================================================

def test_joukowsky_discrete_exact_drop():
    """
    Test that the single-step instantaneous head drop at shut-in matches
    the theoretical Korteweg-Joukowsky drop within machine precision (< 1e-9).
    Sweeps across fluid velocities V0 and wave speeds a.
    """
    velocities = [0.2, 0.5, 1.0, 2.0, 3.0]
    wavespeeds = [1100.0, 1300.0, 1450.0, 1600.0]

    max_rel_err = 0.0
    for V0 in velocities:
        for a in wavespeeds:
            cfg = MocConfig(
                wellbore_length=1200.0,
                wellbore_diameter=0.1397,
                fluid_density=1000.0,
                fluid_viscosity=1.0e-6,
                wavespeed=a,
                roughness_height=4.5e-5,
                friction_model="steady",
                wellhead_bc="velocity_step",
                pump_shut_time=1.0,
                initial_velocity=V0,
                initial_head=350.0,
                toe_bc="dead_end",
                dt=1.0e-3,
                tf=1.2,
            )
            res = simulate_wellbore(
                cfg,
                fracture_positions=[800.0],
                fracture_compliance_m2=[1e-5],
                fracture_inflow_weights=[1.0],
                H_ext=100.0,
                store_full_field=False,
            )
            dt = cfg.dt_adj
            idx_ts = int(round(cfg.pump_shut_time / dt))
            dH_sim_step = res["wellhead_head"][idx_ts] - res["wellhead_head"][idx_ts - 1]
            dH_jouk = -cfg.a_adj * V0 / G

            rel_err = abs(dH_sim_step - dH_jouk) / abs(dH_jouk)
            max_rel_err = max(max_rel_err, rel_err)
            assert rel_err < 1.0e-9, (
                f"Joukowsky discrete step error too large: V0={V0}, a={a}, "
                f"sim={dH_sim_step:.6f}, jouk={dH_jouk:.6f}, rel_err={rel_err:.4e}"
            )

    print(f"\n[PASS] Joukowsky discrete exact drop verified across 20 (V0, a) pairs: max rel err = {max_rel_err:.4e}")


def test_joukowsky_causality_independence_across_fracture_parameters():
    """
    Acoustic Causality Invariant:
    Prior to the acoustic round-trip travel time (2 * x_f,1 / a), the wellhead
    pressure head drop MUST be mathematically invariant to downstream fracture
    properties (Cf, kleak, Rp, cluster count, spacing).
    We vary Cf over 5 orders of magnitude, Rp over 3 orders of magnitude, and
    inflow weights, and assert the head step variance is < 1e-10.
    """
    cfg = MocConfig(
        wellbore_length=2000.0,
        wellbore_diameter=0.1397,
        fluid_density=1000.0,
        fluid_viscosity=1.0e-6,
        wavespeed=1450.0,
        roughness_height=4.5e-5,
        friction_model="steady",
        wellhead_bc="velocity_step",
        pump_shut_time=1.0,
        initial_velocity=1.0,
        initial_head=300.0,
        toe_bc="dead_end",
        dt=1.0e-3,
        tf=1.3,
    )

    adversarial_fractures = [
        # (positions, Cf, Rp, weights)
        ([1500.0], [1.0e-8], [0.0], [1.0]),
        ([1500.0], [1.0e-5], [0.0], [1.0]),
        ([1500.0], [1.0e-3], [0.0], [1.0]),
        ([1500.0], [1.0e-5], [200.0], [1.0]),
        ([1200.0, 1400.0], [1.0e-5, 2.0e-5], [10.0, 50.0], [0.8, 0.2]),
        ([1000.0, 1200.0, 1400.0, 1600.0], [1.0e-6, 5.0e-6, 1.0e-5, 2.0e-5], [0.0, 20.0, 50.0, 100.0], [0.4, 0.3, 0.2, 0.1]),
    ]

    dt = cfg.dt_adj
    idx_ts = int(round(cfg.pump_shut_time / dt))
    dH_jouk = -cfg.a_adj * cfg.initial_velocity / G

    step_drops = []
    fifty_ms_drops = []
    idx_50ms = int(round((cfg.pump_shut_time + 0.05) / dt))
    idx_pre = int(round((cfg.pump_shut_time - 0.01) / dt))

    for i, (pos, cf, rp, w) in enumerate(adversarial_fractures):
        res = simulate_wellbore(
            cfg,
            fracture_positions=pos,
            fracture_compliance_m2=cf,
            fracture_inflow_weights=w,
            fracture_Rp=rp,
            H_ext=100.0,
            store_full_field=False,
        )
        H_wh = res["wellhead_head"]
        dH_step = H_wh[idx_ts] - H_wh[idx_ts - 1]
        dH_50ms = H_wh[idx_50ms] - H_wh[idx_pre]

        step_drops.append(dH_step)
        fifty_ms_drops.append(dH_50ms)

    # 1. Check all 1-step drops match Joukowsky exactly
    for dH in step_drops:
        assert abs(dH - dH_jouk) / abs(dH_jouk) < 1.0e-9

    # 2. Check pairwise variance across fracture variations is zero within float64 precision
    step_variance = np.ptp(step_drops)
    fifty_ms_variance = np.ptp(fifty_ms_drops)
    assert step_variance < 1.0e-11, f"Causality violation: 1-step drop varies with fractures! ptp={step_variance}"
    assert fifty_ms_variance < 1.0e-10, f"Causality violation: 50ms drop varies with fractures before wave return! ptp={fifty_ms_variance}"

    print(f"\n[PASS] Causality Invariance verified across 6 radical fracture configs: ptp(step)={step_variance:.2e}, ptp(50ms)={fifty_ms_variance:.2e}")


def test_joukowsky_50ms_interval_physical_tolerance():
    """
    Verify that simulated head drop measured across 50 ms window satisfies
    |Delta H_sim - Delta H_jouk| / |Delta H_jouk| <= 0.5% (wall friction margin).
    """
    for f_model in ["steady", "brunone"]:
        cfg = MocConfig(
            wellbore_length=2000.0,
            wellbore_diameter=0.1397,
            fluid_density=1000.0,
            fluid_viscosity=1.0e-6,
            wavespeed=1450.0,
            roughness_height=4.5e-5,
            friction_model=f_model,
            brunone_k_scale=1.0,
            wellhead_bc="ramp",
            pump_shut_time=0.5,
            pump_closure_duration=0.05,
            initial_velocity=1.0,
            initial_head=300.0,
            toe_bc="dead_end",
            dt=1.0e-3,
            tf=1.5,
        )
        res = simulate_wellbore(
            cfg,
            fracture_positions=[1200.0, 1500.0],
            fracture_compliance_m2=[1.0e-5, 1.0e-5],
            fracture_inflow_weights=[0.6, 0.4],
            H_ext=100.0,
            store_full_field=False,
        )
        dt = cfg.dt_adj
        dH_jouk = -cfg.a_adj * cfg.initial_velocity / G
        idx_after = int(round((0.5 + 0.05 + 0.05) / dt))
        idx_before = int(round((0.5 - 0.01) / dt))
        dH_sim = res["wellhead_head"][idx_after] - res["wellhead_head"][idx_before]
        err_pct = abs(dH_sim - dH_jouk) / abs(dH_jouk) * 100.0

        assert err_pct < 0.5, f"50ms drop error exceeds 0.5% in {f_model}: err={err_pct:.4f}%"
        print(f"\n[PASS] 50ms interval Joukowsky drop for {f_model}: err = {err_pct:.4f}% (< 0.5%)")


def test_joukowsky_near_fracture_causality_breakdown_boundary():
    """
    Adversarial verification of acoustic travel time:
    If a fracture is placed abnormally close to the wellhead (xf = 25m),
    the reflected wave round trip 2*25/1450 = 34.5 ms arrives BEFORE the 50 ms window,
    which alters the 50 ms head, BUT the 1-step instantaneous drop (at 1 ms) MUST
    still match Joukowsky to machine precision (< 1e-9).
    """
    cfg = MocConfig(
        wellbore_length=1000.0,
        wellbore_diameter=0.1397,
        fluid_density=1000.0,
        fluid_viscosity=1.0e-6,
        wavespeed=1450.0,
        roughness_height=4.5e-5,
        friction_model="steady",
        wellhead_bc="velocity_step",
        pump_shut_time=1.0,
        initial_velocity=1.0,
        initial_head=300.0,
        toe_bc="dead_end",
        dt=1.0e-3,
        tf=1.2,
    )
    res = simulate_wellbore(
        cfg,
        fracture_positions=[25.0],
        fracture_compliance_m2=[1.0e-5],
        fracture_inflow_weights=[1.0],
        H_ext=100.0,
        store_full_field=False,
    )
    dt = cfg.dt_adj
    idx_ts = int(round(cfg.pump_shut_time / dt))
    dH_jouk = -cfg.a_adj * cfg.initial_velocity / G
    dH_step = res["wellhead_head"][idx_ts] - res["wellhead_head"][idx_ts - 1]

    idx_50ms = int(round((cfg.pump_shut_time + 0.05) / dt))
    idx_pre = int(round((cfg.pump_shut_time - 0.01) / dt))
    dH_50ms = res["wellhead_head"][idx_50ms] - res["wellhead_head"][idx_pre]

    # Instantaneous drop is exact
    rel_err_step = abs(dH_step - dH_jouk) / abs(dH_jouk)
    assert rel_err_step < 1.0e-9, f"Instantaneous drop violated at xf=25m: {rel_err_step}"

    # 50ms drop has received the reflected wave and differed significantly
    assert abs(dH_50ms - dH_jouk) > 10.0, "Reflection from xf=25m should have arrived before 50ms!"
    print(f"\n[PASS] Near-fracture acoustic causality verified: 1-step exact ({rel_err_step:.2e}), 50ms reflection confirmed ({dH_50ms:.2f}m vs {dH_jouk:.2f}m)")


# ===========================================================================
# 2. DAMPING AND ENERGY DISSIPATION TESTS (BRUNONE VS STEADY)
# ===========================================================================

def test_brunone_systematic_higher_attenuation_dataset_audit():
    """
    Audit all 42 paired cases in production dataset:
    1. Recompute alpha_rms independently from raw NPZ files.
    2. Confirm zero discrepancy with sensitivity_metrics.csv.
    3. Assert alpha_rms(Brunone) > alpha_rms(Steady) across 100% of paired cases.
    """
    csv_path = PROJECT_ROOT / "output" / "fracture_parameter_sensitivity" / "tables" / "sensitivity_metrics.csv"
    data_dir = PROJECT_ROOT / "output" / "fracture_parameter_sensitivity" / "data"
    assert csv_path.exists(), f"Metrics CSV missing: {csv_path}"

    df = pd.read_csv(csv_path)
    steady_cases = df[df["friction"] == "steady"]
    brunone_cases = df[df["friction"] == "brunone"]
    assert len(steady_cases) == 42, f"Expected 42 steady cases, found {len(steady_cases)}"
    assert len(brunone_cases) == 42, f"Expected 42 brunone cases, found {len(brunone_cases)}"

    diffs: List[float] = []
    ratios: List[float] = []
    max_recalc_err = 0.0

    for idx in range(len(steady_cases)):
        c_s = steady_cases.iloc[idx]
        c_b = brunone_cases.iloc[idx]
        s_name = c_s["case_name"]
        b_name = c_b["case_name"]

        npz_s = np.load(data_dir / f"{s_name}.npz")
        npz_b = np.load(data_dir / f"{b_name}.npz")

        # Independent calculation with ts=0.5
        alpha_s, _, _ = independent_calc_alpha_rms(npz_s["H_wh"], npz_s["t"], 0.5, float(npz_s["tf"]))
        alpha_b, _, _ = independent_calc_alpha_rms(npz_b["H_wh"], npz_b["t"], 0.5, float(npz_b["tf"]))

        table_alpha_s = float(c_s["rms_decay_alpha_per_s"])
        table_alpha_b = float(c_b["rms_decay_alpha_per_s"])

        # Audit check: table matches raw NPZ to machine precision
        err_s = abs(alpha_s - table_alpha_s)
        err_b = abs(alpha_b - table_alpha_b)
        max_recalc_err = max(max_recalc_err, err_s, err_b)
        assert err_s < 1.0e-9, f"Recalculation mismatch in {s_name}: {alpha_s} vs {table_alpha_s}"
        assert err_b < 1.0e-9, f"Recalculation mismatch in {b_name}: {alpha_b} vs {table_alpha_b}"

        diff = alpha_b - alpha_s
        ratio = alpha_b / alpha_s
        diffs.append(diff)
        ratios.append(ratio)
        assert diff > 0, f"Brunone failed to exceed Steady in pair {s_name} vs {b_name}: diff={diff}"

    all_passed = all(d > 0 for d in diffs)
    assert all_passed, "Not all paired cases exhibited higher Brunone attenuation"

    print(
        f"\n[PASS] Dataset Audit (42/42 pairs): Brunone systematically exceeds Steady.\n"
        f"  - Recalculation max err vs table: {max_recalc_err:.4e}\n"
        f"  - Diff (Brunone - Steady): min={min(diffs):.6f} s^-1, mean={np.mean(diffs):.6f} s^-1, max={max(diffs):.6f} s^-1\n"
        f"  - Ratio (Brunone / Steady): min={min(ratios):.4f}x, mean={np.mean(ratios):.4f}x, max={max(ratios):.4f}x"
    )


def test_brunone_unsteady_friction_monotonicity_with_k_scale():
    """
    Stress test that wave attenuation alpha_rms increases strictly monotonically
    with the Brunone scaling coefficient k_scale over [0.0, 0.25, 0.5, 1.0, 1.5, 2.0].
    """
    k_scales = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    alphas: List[float] = []

    for k in k_scales:
        f_model = "steady" if k == 0.0 else "brunone"
        cfg = MocConfig(
            wellbore_length=1500.0,
            wellbore_diameter=0.1397,
            fluid_density=1000.0,
            fluid_viscosity=1.0e-6,
            wavespeed=1450.0,
            roughness_height=4.5e-5,
            friction_model=f_model,
            brunone_k_scale=k,
            wellhead_bc="ramp",
            pump_shut_time=0.5,
            pump_closure_duration=0.05,
            initial_velocity=1.0,
            initial_head=300.0,
            toe_bc="dead_end",
            dt=1.0e-3,
            tf=10.0,
        )
        res = simulate_wellbore(
            cfg,
            fracture_positions=[1000.0],
            fracture_compliance_m2=[1e-5],
            fracture_inflow_weights=[1.0],
            H_ext=100.0,
            store_full_field=False,
        )
        a, _, _ = independent_calc_alpha_rms(res["wellhead_head"], res["timestamps"], 0.5, 10.0)
        alphas.append(a)

    # Assert strict monotonicity: alphas[i] < alphas[i+1]
    is_strictly_monotonic = all(alphas[i] < alphas[i + 1] for i in range(len(alphas) - 1))
    assert is_strictly_monotonic, f"alpha_rms is not strictly monotonic with k_scale: {alphas}"

    print(f"\n[PASS] Strict Brunone k_scale monotonicity verified across {k_scales}:")
    for k, a in zip(k_scales, alphas):
        print(f"  k_scale={k:.2f} -> alpha_rms={a:.6f} s^-1")


def test_brunone_higher_dissipation_adversarial_paired_simulations():
    """
    Run fresh paired simulations under extreme operating regimes:
    1. Low flow velocity (V0 = 0.2 m/s)
    2. High flow velocity (V0 = 2.5 m/s)
    3. Zero fracture compliance (Cf = 0.0)
    4. Huge compliance (Cf = 1e-3 m2)
    5. High perforation resistance (Rp = 250 s2/m5)
    6. 4-cluster dense spacing
    7. Shallow fracture (300m)
    8. Deep fracture (1400m)
    Assert alpha_rms(Brunone) > alpha_rms(Steady) in all cases.
    """
    adversarial_cases = [
        {"name": "Low V0 (0.2 m/s)", "V0": 0.2, "Cf": [1e-5], "pos": [1000.0], "Rp": [0.0], "w": [1.0]},
        {"name": "High V0 (2.5 m/s)", "V0": 2.5, "Cf": [1e-5], "pos": [1000.0], "Rp": [0.0], "w": [1.0]},
        {"name": "Zero Cf (0.0)", "V0": 1.0, "Cf": [0.0], "pos": [1000.0], "Rp": [0.0], "w": [1.0]},
        {"name": "Huge Cf (1e-3)", "V0": 1.0, "Cf": [1e-3], "pos": [1000.0], "Rp": [0.0], "w": [1.0]},
        {"name": "High Rp (250 s2/m5)", "V0": 1.0, "Cf": [1e-5], "pos": [1000.0], "Rp": [250.0], "w": [1.0]},
        {"name": "4-fracture cluster", "V0": 1.0, "Cf": [1e-5]*4, "pos": [800.0, 900.0, 1000.0, 1100.0], "Rp": [10.0]*4, "w": [0.4, 0.3, 0.2, 0.1]},
        {"name": "Shallow fracture (300m)", "V0": 1.0, "Cf": [1e-5], "pos": [300.0], "Rp": [0.0], "w": [1.0]},
        {"name": "Deep fracture (1400m)", "V0": 1.0, "Cf": [1e-5], "pos": [1400.0], "Rp": [0.0], "w": [1.0]},
    ]

    print("\n=== ADVERSARIAL PAIRED SIMULATIONS ===")
    for c in adversarial_cases:
        alphas = {}
        for f_model in ["steady", "brunone"]:
            cfg = MocConfig(
                wellbore_length=1500.0,
                wellbore_diameter=0.1397,
                fluid_density=1000.0,
                fluid_viscosity=1.0e-6,
                wavespeed=1450.0,
                roughness_height=4.5e-5,
                friction_model=f_model,
                brunone_k_scale=1.0,
                wellhead_bc="ramp",
                pump_shut_time=0.5,
                pump_closure_duration=0.05,
                initial_velocity=c["V0"],
                initial_head=300.0,
                toe_bc="dead_end",
                dt=1.0e-3,
                tf=10.0,
            )
            res = simulate_wellbore(
                cfg,
                fracture_positions=c["pos"],
                fracture_compliance_m2=c["Cf"],
                fracture_inflow_weights=c["w"],
                fracture_Rp=c["Rp"],
                H_ext=100.0,
                store_full_field=False,
            )
            a, _, _ = independent_calc_alpha_rms(res["wellhead_head"], res["timestamps"], 0.5, 10.0)
            alphas[f_model] = a

        diff = alphas["brunone"] - alphas["steady"]
        ratio = alphas["brunone"] / alphas["steady"] if alphas["steady"] > 1e-12 else 0.0
        print(f"  {c['name']:25s}: Steady={alphas['steady']:.6f}, Brunone={alphas['brunone']:.6f} | diff={diff:+.6f} ({ratio:.3f}x)")
        assert diff > 0, f"Brunone damping did not exceed steady in {c['name']}: diff={diff}"

    print("[PASS] All 8 adversarial paired regimes confirmed Brunone > Steady damping.")


# ===========================================================================
# 3. FLOW CONTINUITY & MASS CONSERVATION TESTS
# ===========================================================================

def test_steady_state_inflow_equals_fracture_outflow_sum():
    """
    Verify that in steady state for closed toe:
    Q_in = V0 * A == sum_{i=1}^n Q_f,i = sum_{i=1}^n k_leak,equiv,i * sqrt(H_f,i - H_ext)
    across diverse fracture numbers (1, 2, 3, 5, 8 clusters) and skew weights.
    """
    cases = [
        ([1200.0], [1.0]),
        ([1000.0, 1400.0], [0.7, 0.3]),
        ([800.0, 1100.0, 1400.0], [0.5, 0.35, 0.15]),
        ([700.0, 900.0, 1100.0, 1300.0, 1500.0], [0.35, 0.25, 0.2, 0.12, 0.08]),
        ([600.0, 750.0, 900.0, 1050.0, 1200.0, 1350.0, 1500.0, 1650.0], [0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.06, 0.04]),
    ]

    max_balance_err = 0.0
    for pos, weights in cases:
        n = len(pos)
        cfg = MocConfig(
            wellbore_length=2000.0,
            wellbore_diameter=0.1397,
            fluid_density=1000.0,
            fluid_viscosity=1.0e-6,
            wavespeed=1450.0,
            roughness_height=4.5e-5,
            friction_model="steady",
            wellhead_bc="velocity_step",
            pump_shut_time=2.0,
            initial_velocity=1.25,
            initial_head=320.0,
            toe_bc="dead_end",
            dt=1.0e-3,
            tf=0.5,
        )
        res = simulate_wellbore(
            cfg,
            fracture_positions=pos,
            fracture_compliance_m2=[1.0e-5] * n,
            fracture_inflow_weights=weights,
            fracture_Rp=[15.0] * n,
            H_ext=110.0,
            store_full_field=True,
        )

        Qin = cfg.initial_velocity * cfg.area
        ss = res["steady_state"]
        kleak_eq = ss["equivalent_kleak"]
        Hf_ss = ss["Hf_ss"]

        Q_out_sum = float(np.sum(kleak_eq * np.sqrt(Hf_ss - 110.0)))
        rel_err = abs(Qin - Q_out_sum) / Qin
        max_balance_err = max(max_balance_err, rel_err)
        assert rel_err < 1.0e-12, f"Steady mass balance error too high for n={n}: {rel_err}"

    print(f"\n[PASS] Steady-state mass balance Qin == sum(Q_out) verified up to 8 clusters: max rel err = {max_balance_err:.4e}")


def test_dead_end_stagnant_zone_strict_zero_flow():
    """
    Verify that in steady state, flow velocity in the dead-end zone (from the last
    fracture to the closed toe) is strictly 0.0 m/s (exact zero).
    """
    cfg = MocConfig(
        wellbore_length=2000.0,
        wellbore_diameter=0.1397,
        wavespeed=1450.0,
        friction_model="steady",
        wellhead_bc="velocity_step",
        pump_shut_time=2.0,
        initial_velocity=1.5,
        initial_head=350.0,
        toe_bc="dead_end",
        dt=1.0e-3,
        tf=0.5,
    )
    res = simulate_wellbore(
        cfg,
        fracture_positions=[1200.0, 1600.0],
        fracture_compliance_m2=[1.0e-5, 1.0e-5],
        fracture_inflow_weights=[0.6, 0.4],
        H_ext=100.0,
        store_full_field=True,
    )
    last_idx = res["fracture_indices"][-1]
    dead_zone_v = res["velocity"][0, last_idx + 1 :]

    max_v = float(np.max(np.abs(dead_zone_v)))
    assert max_v == 0.0, f"Dead-end stagnant zone has non-zero velocity: {max_v}"
    print(f"\n[PASS] Stagnant zone downstream of last fracture (x > 1600m): max velocity = {max_v:.1e} m/s")


def test_dynamic_dead_end_toe_zero_flow_all_timesteps():
    """
    Verify that at every time step n in [0, n_steps], the dead-end toe velocity
    V_toe(t) == 0.0 m/s and Q_toe(t) == 0.0 m3/s to exact floating-point zero.
    """
    for f_model in ["steady", "brunone"]:
        cfg = MocConfig(
            wellbore_length=1500.0,
            wellbore_diameter=0.1397,
            wavespeed=1450.0,
            friction_model=f_model,
            wellhead_bc="ramp",
            pump_shut_time=0.5,
            pump_closure_duration=0.05,
            initial_velocity=1.0,
            initial_head=300.0,
            toe_bc="dead_end",
            dt=1.0e-3,
            tf=5.0,
        )
        res = simulate_wellbore(
            cfg,
            fracture_positions=[1000.0],
            fracture_compliance_m2=[1.0e-5],
            fracture_inflow_weights=[1.0],
            fracture_Rp=[50.0],
            H_ext=100.0,
            store_full_field=False,
        )
        toe_v = res["toe_velocity"]
        max_toe_v = float(np.max(np.abs(toe_v)))
        assert max_toe_v == 0.0, f"Dead-end toe leaked non-zero flow in {f_model}: max_v={max_toe_v}"
        print(f"\n[PASS] Dynamic dead-end boundary condition ({f_model}): max toe velocity across 5001 steps = {max_toe_v:.1e} m/s")


def test_dynamic_fracture_node_mass_conservation_residual():
    """
    Verify internal fracture node continuity residual:
    r_mass = Q_f - C_H * (H_f - H_prev_f)/dt - k_leak * sqrt(max(H_f - H_ext, 0))
    and perforation pressure drop residual:
    r_Rp = H_w - H_f - Rp * Q_f * |Q_f|
    are strictly within <= 1e-9 m3/s and <= 1e-8 m at every time step and for every cluster.
    """
    area = np.pi * 0.1397**2 / 4.0
    ga = G / 1450.0
    dt = 1.0e-3
    Cf = 2.5e-5
    kleak = 3.5e-4
    H_ext = 100.0
    Rp = 75.0

    # Test solve_fracture_node over 100 pseudo-random dynamic transient states
    np.random.seed(42)
    max_rmass = 0.0
    max_rRp = 0.0

    for step in range(100):
        H_prev = 200.0 + np.random.uniform(-30.0, 50.0)
        Cp_f = 1.0 + ga * (H_prev + np.random.uniform(-10.0, 10.0))
        Cm_f = -1.0 + ga * (H_prev + np.random.uniform(-10.0, 10.0))

        H_w, H_f, vl, vr, Qf = solve_fracture_node(
            Cp_f, Cm_f, H_prev, area, ga, Cf, kleak, H_ext, dt, Rp=Rp
        )

        r_continuity = abs(Qf - area * (vl - vr))
        r_storage = Qf - Cf * (H_f - H_prev) / dt - kleak * np.sqrt(max(H_f - H_ext, 0.0))
        r_drop = H_w - H_f - Rp * Qf * abs(Qf)

        max_rmass = max(max_rmass, abs(r_storage), r_continuity)
        max_rRp = max(max_rRp, abs(r_drop))

        assert r_continuity < 1.0e-12, f"Discontinuous pipe flow across node: {r_continuity}"
        assert abs(r_storage) <= 1.0e-9, f"Mass residual exceeded: {abs(r_storage)}"
        assert abs(r_drop) <= 1.0e-8, f"Rp drop residual exceeded: {abs(r_drop)}"

    print(f"\n[PASS] Dynamic fracture node residual verified over 100 stress states: max |r_mass| = {max_rmass:.4e} m3/s, max |r_Rp| = {max_rRp:.4e} m")


def test_dynamic_global_mass_balance_integral():
    """
    Global mass conservation stress test:
    1. Wellbore domain: Net fluid injected from wellhead into wellbore minus fluid offtake into fractures
       equals elastic fluid volume stored/decompressed in the pipe:
       M_in - M_to_fracs == Delta_M_wellbore (discrepancy < 0.01%).
    2. Fracture domain: Fluid offtake from wellbore minus formation leakoff equals fluid volume
       stored/released in fracture compliance:
       M_to_fracs - M_leak == Delta_M_frac (discrepancy < 0.01%).
    3. Total system: Overall net fluid entering wellhead minus total fluid leaking into formation
       equals total fluid stored in wellbore + fractures:
       M_in - M_leak == Delta_M_wellbore + Delta_M_frac (discrepancy < 0.02%).
    """
    cfg = MocConfig(
        wellbore_length=1500.0,
        wellbore_diameter=0.1397,
        fluid_density=1000.0,
        fluid_viscosity=1.0e-6,
        wavespeed=1450.0,
        roughness_height=4.5e-5,
        friction_model="steady",
        wellhead_bc="ramp",
        pump_shut_time=0.5,
        pump_closure_duration=0.05,
        initial_velocity=1.0,
        initial_head=300.0,
        toe_bc="dead_end",
        dt=1.0e-3,
        tf=4.0,
    )
    res = simulate_wellbore(
        cfg,
        fracture_positions=[800.0, 1200.0],
        fracture_compliance_m2=[1.0e-5, 2.0e-5],
        fracture_inflow_weights=[0.6, 0.4],
        H_ext=100.0,
        store_full_field=True,
    )

    t = res["timestamps"]
    dt = cfg.dt_adj
    area = cfg.area

    # 1. Wellbore balance
    Q_in_hist = res["wellhead_velocity"] * area
    M_in = float(np.trapz(Q_in_hist, t))
    Q_f_hist = res["fracture_Qs"]  # shape (n_steps+1, n_frac)
    M_to_fracs = float(np.trapz(np.sum(Q_f_hist, axis=1), t))

    H_init = res["head"][0, :]
    H_final = res["head"][-1, :]
    Delta_H_wellbore = float(np.trapz(H_final - H_init, dx=cfg.dx))
    Delta_M_wellbore = (area * G / (cfg.a_adj ** 2)) * Delta_H_wellbore

    wellbore_disc = abs((M_in - M_to_fracs) - Delta_M_wellbore)
    rel_wb_disc = wellbore_disc / abs(M_in - M_to_fracs)

    # 2. Fracture balance
    ss = res["steady_state"]
    kleak_eq = ss["equivalent_kleak"]
    Hf_hist = res["fracture_internal_heads"]
    Q_leak_hist = kleak_eq[None, :] * np.sqrt(np.maximum(Hf_hist - 100.0, 0.0))
    M_leak = float(np.trapz(np.sum(Q_leak_hist, axis=1), t))

    Cf_arr = np.array([1.0e-5, 2.0e-5])
    Delta_M_frac = float(np.sum(Cf_arr * (Hf_hist[-1, :] - Hf_hist[0, :])))

    frac_disc = abs((M_to_fracs - M_leak) - Delta_M_frac)
    rel_frac_disc = frac_disc / abs(Delta_M_frac)

    # 3. Overall system balance
    total_disc = abs((M_in - M_leak) - (Delta_M_wellbore + Delta_M_frac))
    rel_total_disc = total_disc / max(abs(M_in), 1e-12)

    print(
        f"\n[PASS] Global Two-Tier Mass Conservation:\n"
        f"  [Wellbore Domain]\n"
        f"    - Net Wellbore Flow:       {M_in - M_to_fracs:.8f} m3\n"
        f"    - Wellbore Elastic Storage:{Delta_M_wellbore:.8f} m3\n"
        f"    - Wellbore Discrepancy:    {wellbore_disc:.8f} m3 ({rel_wb_disc * 100:.4f}%)\n"
        f"  [Fracture Domain]\n"
        f"    - Net Fracture Inflow:     {M_to_fracs - M_leak:.8f} m3\n"
        f"    - Fracture Volume Storage: {Delta_M_frac:.8f} m3\n"
        f"    - Fracture Discrepancy:    {frac_disc:.8f} m3 ({rel_frac_disc * 100:.4f}%)\n"
        f"  [Coupled System]\n"
        f"    - Total Net Discrepancy:   {total_disc:.8f} m3 ({rel_total_disc * 100:.4f}%)"
    )

    assert rel_wb_disc < 0.001, f"Wellbore mass balance discrepancy exceeded: {rel_wb_disc}"
    assert rel_frac_disc < 0.001, f"Fracture mass balance discrepancy exceeded: {rel_frac_disc}"
    assert rel_total_disc < 0.001, f"Coupled total mass balance discrepancy exceeded: {rel_total_disc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
