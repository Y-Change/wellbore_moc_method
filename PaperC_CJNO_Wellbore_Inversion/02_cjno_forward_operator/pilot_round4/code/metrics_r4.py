# -*- coding: utf-8 -*-
"""Round-4 metric repairs. Thresholds are frozen; they are not fit to predictions."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

SIGNIFICANCE_REL = 1e-3          # |amp| / frozen_physical_scale
NOISE_REL = 1e-4                 # precursor / frozen_physical_scale
FLAT_TOP_REL = 1e-3              # plateau half-width vs |amp|
SILENT_REL = 1e-6
FLOAT32_DEN_FLOOR = 1e-12        # physical-scale-normalized floor after float32 square


def _as1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def classify_silence(target, physical_scale: float, rel_thresh: float = SILENT_REL) -> bool:
    target = _as1d(target)
    scale = float(abs(physical_scale))
    if scale < 1e-30:
        scale = 1.0
    return float(np.linalg.norm(target) / scale) < rel_thresh


def rel_l2(pred, target, silent: bool = False) -> float:
    pred = _as1d(pred)
    target = _as1d(target)
    if silent:
        den = float(np.linalg.norm(target))
        if den < 1e-30:
            den = 1.0
        return float(np.linalg.norm(pred - target) / den)
    den = float(np.linalg.norm(target))
    if den < 1e-30:
        return float("nan")
    return float(np.linalg.norm(pred - target) / den)


def loss_b2_energy_equal(pred, p_pert, mask, silent, target_l2, physical_scale=None):
    """Per-case energy-normalized SE. Denominator is target-only, float64-safe.

    All-silent batch returns a finite skip contract (main=0, skip_step=True).
    Mixed-silent: silent cases excluded from the equal-weight mean.
    """
    err = (pred["p_pert"] - p_pert) * mask
    se = (err.to(torch.float64) ** 2).sum(dim=1)
    t64 = target_l2.to(torch.float64)
    if physical_scale is None:
        floor = FLOAT32_DEN_FLOOR
    else:
        scale = physical_scale.to(torch.float64).reshape(-1)
        floor = (FLOAT32_DEN_FLOOR * scale).clamp_min(FLOAT32_DEN_FLOOR)
    den = torch.maximum(t64 ** 2, floor ** 2 if not torch.is_tensor(floor) else floor ** 2)
    if torch.is_tensor(floor) and floor.ndim == 0:
        den = torch.maximum(t64 ** 2, (floor ** 2).expand_as(t64))
    per = se / den
    active = (~silent) & (mask.sum(dim=1) > 0)
    if active.any():
        main = per[active].mean().to(pred["p_pert"].dtype)
        skip_step = False
    else:
        main = se.new_zeros(()).to(pred["p_pert"].dtype)
        skip_step = True
    silent_abs = se[silent].mean() if silent.any() else se.new_zeros(())
    silent_abs = silent_abs.to(pred["p_pert"].dtype)
    return main, silent_abs, per.to(pred["p_pert"].dtype), skip_step


def grads_finite(params) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _local_extrema(y: np.ndarray, polarity: int) -> np.ndarray:
    y = _as1d(y)
    if y.size < 3:
        return np.zeros(0, dtype=int)
    left = y[1:-1] - y[:-2]
    right = y[1:-1] - y[2:]
    if polarity > 0:
        keep = (left > 0.0) & (right >= 0.0) & (y[1:-1] > 0.0)
    elif polarity < 0:
        keep = (left < 0.0) & (right <= 0.0) & (y[1:-1] < 0.0)
    else:
        keep = ((left > 0.0) & (right >= 0.0) & (y[1:-1] > 0.0)) | (
            (left < 0.0) & (right <= 0.0) & (y[1:-1] < 0.0)
        )
    return np.nonzero(keep)[0] + 1


def first_event_in_window(t, y, t0, t1, polarity=1, physical_scale=1.0,
                          require_complete_window=True) -> dict:
    """First significant extremum. Thresholds frozen vs physical_scale, not |pred|."""
    t = _as1d(t)
    y = _as1d(y)
    scale = float(abs(physical_scale)) if abs(physical_scale) > 1e-30 else 1.0
    sig = SIGNIFICANCE_REL * scale
    noise = NOISE_REL * scale
    rec = {
        "ok": False,
        "reason": "init",
        "t_event": None,
        "amp": None,
        "n_candidates": 0,
        "window_complete": bool(t.size and t[-1] + 1e-15 >= t1 and t[0] - 1e-15 <= t0),
        "polarity": int(polarity),
        "flat_top": False,
        "precursor_ignored": 0,
        "significance_rel": SIGNIFICANCE_REL,
        "noise_rel": NOISE_REL,
        "definition": "first_significant_peak_after_valve_in_window",
        "not_definition": "first_node_reflection_arrival",
        "threshold_source": "frozen_physical_scale_not_prediction_amplitude",
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
        opp = _local_extrema(y_w, -polarity if polarity != 0 else 1)
        if polarity != 0 and opp.size:
            rec["reason"] = "polarity_mismatch"
            rec["n_opposite"] = int(opp.size)
        return rec
    kept = []
    ignored = 0
    for j in loc:
        amp = float(y_w[int(j)])
        if abs(amp) < sig:
            ignored += 1
            continue
        if abs(amp) < noise:
            ignored += 1
            continue
        kept.append(int(j))
    rec["precursor_ignored"] = ignored
    if not kept:
        rec["reason"] = "no_significant_peak"
        return rec
    i_loc = kept[0]
    i0 = idx[i_loc]
    rec["ok"] = True
    rec["reason"] = "ok"
    rec["t_event"] = float(t[i0])
    rec["amp"] = float(y[i0])
    rec["n_later_extrema"] = int(len(kept) - 1)
    rec["candidate_times"] = [float(t[idx[int(j)]]) for j in kept]
    # flat-top: neighbors within FLAT_TOP_REL*|amp|
    amp = abs(float(y[i0]))
    half = FLAT_TOP_REL * max(amp, sig)
    k = i0
    while k + 1 < y.size and abs(y[k + 1] - y[i0]) <= half:
        k += 1
    j = i0
    while j - 1 >= 0 and abs(y[j - 1] - y[i0]) <= half:
        j -= 1
    rec["flat_top"] = (k - j) >= 2
    rec["flat_top_n"] = int(k - j + 1)
    return rec
