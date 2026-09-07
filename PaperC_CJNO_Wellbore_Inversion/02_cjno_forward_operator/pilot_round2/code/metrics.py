# -*- coding: utf-8 -*-
"""Evaluation contract: first-arrival matching, silent channels, SI after undoing scales."""
from __future__ import annotations

import numpy as np


SILENT_REL = 1.0e-6


def rel_l2(a, b, mask=None, silent_rel=SILENT_REL):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is not None:
        m = np.asarray(mask) > 0.5
        a, b = a[m], b[m]
    den = np.linalg.norm(b)
    num = np.linalg.norm(a - b)
    state = np.linalg.norm(b + a)
    silent = den <= silent_rel * max(state, 1.0)
    return {
        "rel": float(num / den) if den > 0 else float("nan"),
        "abs": float(num),
        "silent": bool(silent),
        "den": float(den),
    }


def is_silent_pert(pert, state, silent_rel=SILENT_REL) -> bool:
    """Target-only. Must not depend on predictions."""
    pn = np.linalg.norm(np.asarray(pert, dtype=float))
    sn = np.linalg.norm(np.asarray(state, dtype=float))
    return bool(pn <= silent_rel * max(sn, 1.0))


def _local_peaks(y, min_prom_frac=0.15):
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.array([], dtype=int)
    prom = min_prom_frac * (np.max(np.abs(y)) + 1e-30)
    idx = []
    for i in range(1, y.size - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and abs(y[i]) >= prom:
            if abs(y[i]) >= abs(y[i - 1]) and abs(y[i]) >= abs(y[i + 1]):
                idx.append(i)
    return np.asarray(idx, dtype=int)


def first_arrival(t, y, t_lo, t_hi, polarity=None):
    """Match the first same-polarity local peak inside [t_lo, t_hi].

    Returns time, amplitude, and a failure tag:
      ok | no_peak | multi_peak_ambiguous | empty_window
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (t >= t_lo) & (t <= t_hi)
    if not np.any(m):
        return {"t": float("nan"), "amp": float("nan"), "tag": "empty_window", "n_peaks": 0}
    tw, yw = t[m], y[m]
    peaks = _local_peaks(yw)
    if polarity is not None and peaks.size:
        keep = np.sign(yw[peaks]) == np.sign(polarity) or np.sign(polarity) == 0
        # sign==0: keep all
        if np.sign(polarity) != 0:
            peaks = peaks[np.sign(yw[peaks]) == np.sign(polarity)]
    if peaks.size == 0:
        return {"t": float("nan"), "amp": float("nan"), "tag": "no_peak", "n_peaks": 0}
    i0 = int(peaks[0])
    tag = "ok" if peaks.size == 1 else "ok_first_of_many"
    if peaks.size > 3:
        tag = "multi_peak_ambiguous"
    return {"t": float(tw[i0]), "amp": float(yw[i0]), "tag": tag, "n_peaks": int(peaks.size)}


def first_arrival_error(t, y_hat, y_true, t_valve_end, period):
    """Physical first-arrival: window (t_valve_end, t_valve_end + period)."""
    t_lo = float(t_valve_end)
    t_hi = float(t_valve_end + period)
    ref = first_arrival(t, y_true, t_lo, t_hi, polarity=None)
    if ref["tag"] in ("empty_window", "no_peak"):
        return {"tag": "ref_" + ref["tag"], "dphi_over_4La": float("nan"),
                "dt_s": float("nan"), "amp_rel": float("nan"), "matched": False}
    pol = np.sign(ref["amp"])
    hat = first_arrival(t, y_hat, t_lo, t_hi, polarity=pol)
    if hat["tag"] in ("empty_window", "no_peak"):
        return {"tag": "hat_" + hat["tag"], "dphi_over_4La": float("nan"),
                "dt_s": float("nan"), "amp_rel": float("nan"), "matched": False,
                "ref_t": ref["t"], "hat_t": float("nan")}
    dt = hat["t"] - ref["t"]
    return {
        "tag": "ok" if hat["tag"].startswith("ok") and ref["tag"].startswith("ok") else "multi_peak_ambiguous",
        "dphi_over_4La": float(dt / period),
        "dt_s": float(dt),
        "amp_rel": float(abs(hat["amp"] - ref["amp"]) / (abs(ref["amp"]) + 1e-30)),
        "matched": True,
        "ref_t": ref["t"],
        "hat_t": hat["t"],
        "ref_n": ref["n_peaks"],
        "hat_n": hat["n_peaks"],
    }


def summarize_first_arrival(rows):
    ok = [r for r in rows if r.get("matched")]
    fail = [r for r in rows if not r.get("matched")]
    if ok:
        dphi = np.array([r["dphi_over_4La"] for r in ok], dtype=float)
        stats = {"n_ok": len(ok), "n_fail": len(fail),
                 "dphi_mean": float(dphi.mean()), "dphi_p50": float(np.median(dphi)),
                 "dphi_p90": float(np.quantile(dphi, 0.9)), "dphi_max": float(dphi.max())}
    else:
        stats = {"n_ok": 0, "n_fail": len(fail),
                 "dphi_mean": float("nan"), "dphi_p50": float("nan"),
                 "dphi_p90": float("nan"), "dphi_max": float("nan")}
    return stats
