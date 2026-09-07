# -*- coding: utf-8 -*-
"""First-arrival metric unit tests: delay, amplitude, cross-period mismatch."""
from __future__ import annotations

import numpy as np

from metrics import first_arrival, first_arrival_error, is_silent_pert, rel_l2


def _pulse(t, t0, amp=1.0, w=0.15):
    return amp * np.exp(-0.5 * ((t - t0) / w) ** 2)


def test_known_delay():
    t = np.linspace(0, 10, 1001)
    y = _pulse(t, 3.0)
    yhat = _pulse(t, 3.4)
    e = first_arrival_error(t, yhat, y, t_valve_end=2.0, period=4.0)
    assert e["matched"]
    assert abs(e["dt_s"] - 0.4) < 0.02
    assert abs(e["dphi_over_4La"] - 0.1) < 0.01


def test_amplitude_change_same_time():
    t = np.linspace(0, 10, 1001)
    y = _pulse(t, 3.0, amp=2.0)
    yhat = _pulse(t, 3.0, amp=1.0)
    e = first_arrival_error(t, yhat, y, 2.0, 4.0)
    assert e["matched"]
    assert abs(e["dt_s"]) < 0.02
    assert abs(e["amp_rel"] - 0.5) < 0.05


def test_cross_period_mismatch_not_counted():
    t = np.linspace(0, 20, 2001)
    y = _pulse(t, 3.0)
    # peak only in the next period
    yhat = _pulse(t, 8.5)
    e = first_arrival_error(t, yhat, y, 2.0, 4.0)
    assert not e["matched"]
    assert e["tag"].startswith("hat_")
    assert not np.isfinite(e["dphi_over_4La"])


def test_no_peak_tagged():
    t = np.linspace(0, 10, 401)
    y = np.zeros_like(t)
    r = first_arrival(t, y, 2.0, 6.0)
    assert r["tag"] == "no_peak"


def test_silent_is_target_only():
    y = np.ones(100)
    pert = np.zeros(100)
    assert is_silent_pert(pert, y)
    m = rel_l2(np.ones(100) * 3, pert)
    # prediction must not decide silence
    assert is_silent_pert(pert, y) and m["silent"]


if __name__ == "__main__":
    test_known_delay()
    test_amplitude_change_same_time()
    test_cross_period_mismatch_not_counted()
    test_no_peak_tagged()
    test_silent_is_target_only()
    print("test_metrics PASS")
