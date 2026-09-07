# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def _rel_l2(a, b, mask=None, silent_frac=1e-6):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is not None:
        m = np.asarray(mask) > 0.5
        a, b = a[m], b[m]
    den = np.linalg.norm(b)
    num = np.linalg.norm(a - b)
    silent = den <= silent_frac * max(np.linalg.norm(b + a), 1.0)
    return {"rel": float(num / den) if den > 0 else float("nan"),
            "abs": float(num), "silent": bool(silent), "den": float(den)}


def first_peak_phase_frac(t, y, period):
    """Peak location as fraction of 4L/a. Not a 70 Hz arrival metric."""
    if y.size < 4:
        return float("nan")
    i = int(np.argmax(np.abs(y)))
    return float((t[i] - t[0]) / period)


def peak_trough_err(a, b):
    return {
        "peak_rel": abs(float(np.max(a) - np.max(b))) / (abs(float(np.max(b))) + 1e-30),
        "trough_rel": abs(float(np.min(a) - np.min(b))) / (abs(float(np.min(b))) + 1e-30),
    }


def psd_err(a, b, dt):
    fa = np.fft.rfft(a - np.mean(a))
    fb = np.fft.rfft(b - np.mean(b))
    Sa, Sb = np.abs(fa) ** 2, np.abs(fb) ** 2
    return float(np.sum(np.abs(Sa - Sb)) / (np.sum(Sb) + 1e-30))


def case_metrics(p_hat, p_true, p0, t, period, dt, mask=None):
    if mask is not None:
        m = mask > 0.5
        p_hat, p_true, t = p_hat[m], p_true[m], t[m]
    pert_h = p_hat - p0
    pert_t = p_true - p0
    return {
        "abs_state_l2": _rel_l2(p_hat, p_true),
        "pert_l2": _rel_l2(pert_h, pert_t),
        "peak_trough": peak_trough_err(pert_h, pert_t),
        "phase_frac_hat": first_peak_phase_frac(t, pert_h, period),
        "phase_frac_true": first_peak_phase_frac(t, pert_t, period),
        "phase_err_over_4La": abs(first_peak_phase_frac(t, pert_h, period) -
                                  first_peak_phase_frac(t, pert_t, period)),
        "psd": psd_err(pert_h, pert_t, dt),
        "finite": bool(np.isfinite(p_hat).all()),
    }


def summarize(rows):
    def gather(path):
        vals = []
        for r in rows:
            x = r
            ok = True
            for k in path:
                if not isinstance(x, dict) or k not in x:
                    ok = False
                    break
                x = x[k]
            if ok and isinstance(x, (int, float)) and np.isfinite(x):
                vals.append(float(x))
        if not vals:
            return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}
        a = np.array(vals)
        return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)),
                "p90": float(np.quantile(a, 0.9)), "max": float(a.max())}

    return {
        "pert_l2": gather(("wellhead", "pert_l2", "rel")),
        "abs_l2": gather(("wellhead", "abs_state_l2", "rel")),
        "phase_4La": gather(("wellhead", "phase_err_over_4La")),
        "psd": gather(("wellhead", "psd")),
        "node_H_pert_l2": gather(("node", "H_pert_l2", "rel")),
        "n_rows": len(rows),
        "n_nonfinite": int(sum(1 for r in rows if not r.get("wellhead", {}).get("finite", True))),
    }
