# -*- coding: utf-8 -*-
"""Round-3 metric counterexamples + original first-arrival regressions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from metrics import (
    batch_first_arrival, classify_silence, first_arrival_pair, first_event_in_window,
    rel_l2, summarize_phase_errors,
)


def _pulse(t, t0, amp=1.0, w=0.15):
    return amp * np.exp(-0.5 * ((t - t0) / w) ** 2)


def test_negative_trough():
    t = np.linspace(0, 10, 1001)
    y = -_pulse(t, 3.0)
    ev = first_event_in_window(t, y, 2.0, 6.0, polarity=-1)
    assert ev["ok"] and abs(ev["t_event"] - 3.0) < 0.02
    assert ev["amp"] < 0


def test_multiple_positive_peaks_take_earliest():
    t = np.linspace(0, 10, 2001)
    y = _pulse(t, 3.0, amp=0.5) + _pulse(t, 4.5, amp=2.0)
    ev = first_event_in_window(t, y, 2.0, 6.0, polarity=1)
    assert ev["ok"]
    assert abs(ev["t_event"] - 3.0) < 0.03
    assert ev["n_candidates"] >= 2
    assert ev["n_later_extrema"] >= 1


def test_multiple_negative_troughs_take_earliest():
    t = np.linspace(0, 10, 2001)
    y = -_pulse(t, 3.2, amp=0.4) - _pulse(t, 5.0, amp=1.5)
    ev = first_event_in_window(t, y, 2.0, 6.0, polarity=-1)
    assert ev["ok"] and abs(ev["t_event"] - 3.2) < 0.03


def test_wrong_polarity():
    t = np.linspace(0, 10, 1001)
    y = -_pulse(t, 3.0)
    ev = first_event_in_window(t, y, 2.0, 6.0, polarity=1)
    assert not ev["ok"]
    assert ev["reason"] == "polarity_mismatch"


def test_strong_peak_outside_window_ignored():
    t = np.linspace(0, 10, 2001)
    y = _pulse(t, 1.2, amp=10.0) + _pulse(t, 3.0, amp=1.0)
    ev = first_event_in_window(t, y, 2.0, 6.0, polarity=1)
    assert ev["ok"] and abs(ev["t_event"] - 3.0) < 0.03


def test_truncated_window():
    t = np.linspace(0, 4.0, 401)
    y = _pulse(t, 3.0)
    ev = first_event_in_window(t, y, 2.0, 6.0, polarity=1)
    assert not ev["ok"]
    assert ev["reason"] == "incomplete_window"


def test_true_silence_from_target_only():
    target = np.zeros(200)
    pred = np.ones(200) * 1e6
    assert classify_silence(target, physical_scale=1e5, rel_thresh=1e-6)
    # huge prediction must not flip silence
    assert classify_silence(target, 1e5)
    r = rel_l2(pred, target, silent=True)
    assert np.isfinite(r) and r > 1.0


def test_large_pred_error_not_dropped_as_silent():
    target = np.ones(100) * 10.0
    pred = np.ones(100) * 1e9
    assert not classify_silence(target, physical_scale=10.0, rel_thresh=1e-6)
    r = rel_l2(pred, target, silent=False)
    assert r > 1e6


def test_signed_phase_cancellation_not_used_as_gate():
    recs = [
        {"matched": True, "abs_s": 0.4, "signed_s": 0.4, "reason": "ok"},
        {"matched": True, "abs_s": 0.4, "signed_s": -0.4, "reason": "ok"},
        {"matched": False, "abs_s": None, "signed_s": None, "reason": "ambiguous_match"},
    ]
    s = summarize_phase_errors(recs)
    assert s["n_matched"] == 2
    assert s["coverage"] == 2 / 3
    assert abs(s["mean_signed_s"]) < 1e-12
    assert abs(s["mean_abs_s"] - 0.4) < 1e-12


def test_ambiguous_match_excluded():
    t = np.linspace(0, 10, 1001)
    y = _pulse(t, 3.0)
    # two pred peaks equidistant from t=3.0
    yhat = _pulse(t, 2.6) + _pulse(t, 3.4)
    rec = first_arrival_pair(t, yhat, y, 2.0, 4.0, polarity=1)
    assert not rec["matched"]
    assert rec["reason"] == "ambiguous_match"
    s = summarize_phase_errors([rec])
    assert s["n_matched"] == 0
    assert s["mean_abs_s"] is None


def test_known_delay_abs_and_signed():
    t = np.linspace(0, 10, 1001)
    y = _pulse(t, 3.0)
    yhat = _pulse(t, 3.4)
    e = first_arrival_pair(t, yhat, y, 2.0, 4.0)
    assert e["matched"]
    assert abs(e["signed_s"] - 0.4) < 0.02
    assert abs(e["abs_s"] - 0.4) < 0.02


def test_cross_period_not_counted():
    t = np.linspace(0, 20, 2001)
    y = _pulse(t, 3.0)
    yhat = _pulse(t, 8.5)
    e = first_arrival_pair(t, yhat, y, 2.0, 4.0)
    assert not e["matched"]
    assert "pred" in e["reason"]


def test_definitions_are_labeled():
    t = np.linspace(0, 10, 501)
    y = _pulse(t, 3.0)
    ev = first_event_in_window(t, y, 2.0, 6.0)
    assert ev["definition"] == "first_peak_after_valve_in_window"
    assert ev["not_definition"] == "first_node_reflection_arrival"


def test_batch_coverage():
    t = np.linspace(0, 10, 1001)
    y = _pulse(t, 3.0)
    yhat = _pulse(t, 3.2)
    s = batch_first_arrival([t, t], [yhat, np.zeros_like(t)], [y, y], [2.0, 2.0], [4.0, 4.0])
    assert s["n_total"] == 2
    assert s["n_matched"] == 1
    assert abs(s["coverage"] - 0.5) < 1e-12


if __name__ == "__main__":
    for fn in [test_negative_trough, test_multiple_positive_peaks_take_earliest,
               test_multiple_negative_troughs_take_earliest, test_wrong_polarity,
               test_strong_peak_outside_window_ignored, test_truncated_window,
               test_true_silence_from_target_only, test_large_pred_error_not_dropped_as_silent,
               test_signed_phase_cancellation_not_used_as_gate, test_ambiguous_match_excluded,
               test_known_delay_abs_and_signed, test_cross_period_not_counted,
               test_definitions_are_labeled, test_batch_coverage]:
        fn()
        print("ok", fn.__name__)
    print("test_metrics PASS")
