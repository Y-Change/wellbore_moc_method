# -*- coding: utf-8 -*-
"""Round-3 wellhead metrics.

Definitions (do not conflate):
  first_peak_after_valve: first extremum of the requested polarity inside
      (t_valve_end, t_valve_end + one_period], after the cosine shut-in.
  first_node_reflection_arrival: travel-time of a node-generated wave to the
      wellhead (x_j / a).  That is a different physical event and is NOT
      computed by this module.

Silence is decided from the TARGET and a frozen physical scale only.
A large-error prediction is never reclassified as silent.
Failed / ambiguous matches are excluded from success means; coverage is reported.
Acceptance uses absolute phase error; signed error is reported but not averaged
into a cancelling mean for the gate.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _as1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def rel_l2(pred, target, silent: bool = False, silent_from_target_only: bool = True) -> float:
    pred = _as1d(pred)
    target = _as1d(target)
    if silent:
        # Frozen physical scale is the target energy; prediction never enters the denom.
        den = float(np.linalg.norm(target))
        if den < 1e-30:
            den = 1.0
        return float(np.linalg.norm(pred - target) / den)
    den = float(np.linalg.norm(target))
    if den < 1e-30:
        return float("nan")
    return float(np.linalg.norm(pred - target) / den)


def classify_silence(target, physical_scale: float, rel_thresh: float = 1e-6) -> bool:
    """Silence from target + frozen physical scale. Prediction is not used."""
    target = _as1d(target)
    scale = float(abs(physical_scale))
    if scale < 1e-30:
        scale = 1.0
    return float(np.linalg.norm(target) / scale) < rel_thresh


def _local_extrema(y: np.ndarray, polarity: int) -> np.ndarray:
    """Return indices of strict local extrema of the requested polarity.

    polarity = +1 keeps peaks (y[i] > neighbors and y[i] > 0)
    polarity = -1 keeps troughs (y[i] < neighbors and y[i] < 0)
    polarity = 0 keeps both (sign of y[i] decides peak vs trough)
    """
    y = _as1d(y)
    if y.size < 3:
        return np.zeros(0, dtype=int)
    left = y[1:-1] - y[:-2]
    right = y[1:-1] - y[2:]
    if polarity > 0:
        keep = (left > 0.0) & (right > 0.0) & (y[1:-1] > 0.0)
    elif polarity < 0:
        keep = (left < 0.0) & (right < 0.0) & (y[1:-1] < 0.0)
    else:
        keep = ((left > 0.0) & (right > 0.0) & (y[1:-1] > 0.0)) | (
            (left < 0.0) & (right < 0.0) & (y[1:-1] < 0.0)
        )
    return np.nonzero(keep)[0] + 1


def first_event_in_window(
    t,
    y,
    t0: float,
    t1: float,
    polarity: int = 1,
    require_complete_window: bool = True,
) -> dict:
    t = _as1d(t)
    y = _as1d(y)
    rec = {
        "ok": False,
        "reason": "init",
        "t_event": None,
        "amp": None,
        "n_candidates": 0,
        "window_complete": bool(t.size and t[-1] + 1e-15 >= t1 and t[0] - 1e-15 <= t0),
        "polarity": int(polarity),
        "definition": "first_peak_after_valve_in_window",
        "not_definition": "first_node_reflection_arrival",
    }
    if require_complete_window and not rec["window_complete"]:
        rec["reason"] = "incomplete_window"
        return rec
    in_win = (t > t0) & (t <= t1)
    if not np.any(in_win):
        rec["reason"] = "incomplete_window"
        rec["window_complete"] = False
        return rec
    idx = np.nonzero(in_win)[0]
    y_w = y[idx]
    loc = _local_extrema(y_w, polarity)
    rec["n_candidates"] = int(loc.size)
    if loc.size == 0:
        rec["reason"] = "no_peak" if polarity >= 0 else "no_trough"
        # polarity mismatch: opposite extrema exist
        opp = _local_extrema(y_w, -polarity if polarity != 0 else 1)
        if polarity != 0 and opp.size:
            rec["reason"] = "polarity_mismatch"
            rec["n_opposite"] = int(opp.size)
        return rec
    # Earliest qualifying extremum is the first-peak definition; extra later
    # oscillations are expected and are not themselves an ambiguous match.
    i0 = idx[int(loc[0])]
    rec["ok"] = True
    rec["reason"] = "ok"
    rec["t_event"] = float(t[i0])
    rec["amp"] = float(y[i0])
    rec["n_later_extrema"] = int(loc.size - 1)
    rec["candidate_times"] = [float(t[idx[int(j)]]) for j in loc]
    return rec


def signed_and_abs_phase(t_pred: Optional[float], t_ref: Optional[float]) -> dict:
    if t_pred is None or t_ref is None:
        return {"signed_s": None, "abs_s": None, "ok": False}
    signed = float(t_pred) - float(t_ref)
    return {"signed_s": signed, "abs_s": abs(signed), "ok": True}


def summarize_phase_errors(records: List[dict]) -> dict:
    """Coverage-aware summary. Ambiguous/failed never enter the success mean.

    The gate statistic is mean absolute error, not mean signed error.
    """
    matched = [r for r in records if r.get("matched") is True]
    n = len(records)
    n_ok = len(matched)
    abs_err = [float(r["abs_s"]) for r in matched if r.get("abs_s") is not None]
    signed = [float(r["signed_s"]) for r in matched if r.get("signed_s") is not None]
    reasons: Dict[str, int] = {}
    for r in records:
        reasons[r.get("reason", "unknown")] = reasons.get(r.get("reason", "unknown"), 0) + 1
    out = {
        "n_total": n,
        "n_matched": n_ok,
        "coverage": (n_ok / n) if n else 0.0,
        "mean_abs_s": float(np.mean(abs_err)) if abs_err else None,
        "mean_signed_s": float(np.mean(signed)) if signed else None,
        "note": "acceptance uses mean_abs_s; mean_signed_s can cancel and is not a gate",
        "reasons": reasons,
    }
    return out


def first_arrival_pair(t, pred, target, t_valve_end: float, period: float, polarity: int = 1) -> dict:
    t0 = float(t_valve_end)
    t1 = float(t_valve_end + period)
    ref = first_event_in_window(t, target, t0, t1, polarity=polarity)
    pr = first_event_in_window(t, pred, t0, t1, polarity=polarity)
    rec = {
        "ref": ref,
        "pred": pr,
        "matched": False,
        "reason": "init",
        "signed_s": None,
        "abs_s": None,
        "definition": "first_peak_after_valve",
        "not_definition": "node_reflection_first_arrival",
    }
    if not ref["ok"]:
        rec["reason"] = f"ref_{ref['reason']}"
        return rec
    if not pr["ok"]:
        rec["reason"] = f"pred_{pr['reason']}"
        return rec
    # Ambiguous match: two or more pred candidates equally close to the ref event.
    pred_times = np.asarray(pr.get("candidate_times") or [pr["t_event"]], dtype=np.float64)
    d = np.abs(pred_times - float(ref["t_event"]))
    if pred_times.size >= 2:
        order = np.argsort(d)
        if abs(float(d[order[0]]) - float(d[order[1]])) <= 1e-12:
            rec["reason"] = "ambiguous_match"
            rec["signed_s"] = None
            rec["abs_s"] = None
            return rec
    ph = signed_and_abs_phase(pr["t_event"], ref["t_event"])
    rec.update(ph)
    rec["matched"] = True
    rec["reason"] = "ok"
    return rec


def batch_first_arrival(t_list, pred_list, target_list, t_valve_end, periods, polarity=1) -> dict:
    recs = []
    for t, pr, tg, tv, per in zip(t_list, pred_list, target_list, t_valve_end, periods):
        recs.append(first_arrival_pair(t, pr, tg, tv, per, polarity=polarity))
    summary = summarize_phase_errors(recs)
    summary["records"] = recs
    return summary
