# -*- coding: utf-8 -*-
"""CJ-AlphaNet 离线标签物理量单测。"""
from __future__ import annotations

import numpy as np

from PaperC_CJNO_Wellbore_Inversion.src.label_physics import (
    ALPHA_ACTIVE_THR,
    A_HAT_MAX,
    A_HAT_MIN,
    active_flags,
    build_sample_labels,
    equivalent_admittance,
    estimate_round_trip_period,
    inflow_density_field,
    log10_y_over_y0,
    renormalize_alpha,
    wavespeed_from_first_cluster,
)


def test_renormalize_and_active_flags():
    a = np.array([0.5, 0.45, 0.03, 0.02])
    rn = renormalize_alpha(a)
    assert np.isclose(rn.sum(), 1.0)
    flags = active_flags(a, threshold=0.04)
    assert list(flags) == [1, 1, 0, 0]
    assert list(active_flags(np.zeros(3))) == [0, 0, 0]
    assert ALPHA_ACTIVE_THR == 0.04


def test_wavespeed_from_first_cluster_roundtrip():
    x1 = 4700.0
    t1 = 2.0 * x1 / 1450.0
    a_hat = wavespeed_from_first_cluster(x1, t1)
    assert abs(a_hat - 1450.0) < 1.0
    assert A_HAT_MIN <= wavespeed_from_first_cluster(x1, 1.0e-6) <= A_HAT_MAX
    assert A_HAT_MIN <= wavespeed_from_first_cluster(x1, 100.0) <= A_HAT_MAX


def test_estimate_period_recovers_synthetic_echo():
    dt = 0.001
    x1 = 4700.0
    a_true = 1450.0
    t1 = 2.0 * x1 / a_true
    t = np.arange(0.0, 60.0, dt)
    # 双程回波脉冲串
    h = np.zeros_like(t)
    for k in range(1, 6):
        h += np.exp(-0.5 * ((t - k * t1) / 0.04) ** 2)
    t_est, ok = estimate_round_trip_period(h, dt=dt, x1=x1)
    assert ok
    assert abs(t_est - t1) < 0.05
    a_hat = wavespeed_from_first_cluster(x1, t_est)
    assert abs(a_hat - a_true) < 30.0


def test_estimate_period_fallback_first_echo_single_pulse():
    dt = 0.001
    x1 = 4700.0
    a_true = 1450.0
    t1 = 2.0 * x1 / a_true
    t = np.arange(0.0, 60.0, dt)
    h = np.exp(-0.5 * ((t - t1) / 0.02) ** 2)
    t_est, ok = estimate_round_trip_period(h, dt=dt, x1=x1)
    assert ok
    assert abs(t_est - t1) < 0.08
    assert abs(wavespeed_from_first_cluster(x1, t_est) - a_true) < 50.0


def test_screenout_admittance_much_smaller_than_open_cluster():
    omega = 2.0 * np.pi / 6.5
    y_open = equivalent_admittance(q=0.005, kleak=1.0e-4, kp=5.4e5, cf=0.01, omega=omega)
    y_dead = equivalent_admittance(q=1.0e-4, kleak=1.0e-6, kp=8.0e7, cf=0.0005, omega=omega)
    y_zero = equivalent_admittance(q=0.0, kleak=1.0e-4, kp=5.4e5, cf=0.01, omega=omega)
    assert y_open > 0.0
    assert np.isfinite(y_dead)
    assert y_dead < y_open
    assert y_zero == 0.0
    y0 = 1.0e-4
    assert log10_y_over_y0(y_dead, y0) < log10_y_over_y0(y_open, y0)


def test_inflow_density_integrates_to_one():
    pos = np.array([4510.0, 4540.0, 4580.0])
    alpha = np.array([0.5, 0.3, 0.2])
    grid, field = inflow_density_field(pos, alpha, L=5000.0, n_grid=500, sigma_m=20.0)
    dx = 5000.0 / 499.0
    integ = float(np.sum(field) * dx)
    assert abs(integ - 1.0) < 1.0e-3
    assert grid[0] == 0.0 and abs(grid[-1] - 5000.0) < 1.0e-3
    # 质量应落在趾端附近
    mass_toe = float(np.sum(field[grid >= 4400.0]) * dx)
    assert mass_toe > 0.9


def test_build_sample_labels_x1_active_and_screenout_yeq():
    dt = 0.001
    ts = 1.0
    t = np.arange(0.0, 61.0 + 0.5 * dt, dt)
    x_echo = 4700.0
    a_true = 1450.0
    t1 = 2.0 * x_echo / a_true
    h = np.zeros_like(t)
    t_post = t - ts
    for k in range(1, 6):
        h += np.exp(-0.5 * ((t_post - k * t1) / 0.04) ** 2)

    pos = np.array([4780.0, 4700.0, 4730.0])
    alpha = np.array([0.50, 0.47, 0.03])
    q_ss = np.array([0.006, 0.0055, 1.0e-6])
    kleak = np.array([1.0e-4, 1.2e-4, 2.0e-6])
    kp = np.array([5.4e5, 4.0e5, 8.0e7])
    cf = np.array([0.012, 0.010, 0.0004])

    lab = build_sample_labels(
        timestamps=t,
        wellhead_head=h,
        positions=pos,
        alpha=alpha,
        q_ss=q_ss,
        kleak=kleak,
        kp=kp,
        cf=cf,
        ts=ts,
    )
    assert float(lab["x1"]) == 4700.0
    assert list(lab["y_active"]) == [1, 1, 0]
    assert float(lab["Y_eq"][2]) == 0.0
    assert float(lab["Y_eq"][0]) > 0.0
    assert A_HAT_MIN <= float(lab["a_hat"]) <= A_HAT_MAX
    assert abs(float(lab["a_hat"]) - a_true) < 40.0
    assert int(lab["a_hat_ok"]) == 1
