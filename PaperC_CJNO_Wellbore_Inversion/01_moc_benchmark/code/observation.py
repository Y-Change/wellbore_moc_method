# -*- coding: utf-8 -*-
"""Frozen wellhead observation chain (攻关执行方案_v2 §2.5 / §3.2 / §5.1).

Generation stores the chain definition and a clean (SNR=inf) observed series.
Noise realisations belong to Stage 3 and must be drawn from this spec, not invented later.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def observation_spec(data_cfg: dict, snr_db=float("inf")) -> dict:
    om = data_cfg["observation_model"]
    return {
        "sampling_rate_hz": float(om["sampling_rate_hz"]["default"]),
        "sensor_transfer_function": dict(om["sensor_transfer_function"]),
        "antialias_filter": dict(om["antialias_filter"]),
        "noise_model": {
            "type": om["noise"]["type"],
            "snr_db_applied": float(snr_db),
            "snr_db_grid": list(om["noise"]["snr_db"]),
            "impulsive_outlier_fraction": list(om["noise"]["impulsive_outlier_fraction"]),
        },
        "f_eff_definition": str(om["f_eff_definition"]).strip(),
    }


def apply_observation_chain(t: np.ndarray, p: np.ndarray, spec: dict
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample to fs, apply 2nd-order sensor TF then Butterworth antialias. No noise."""
    from scipy.signal import bilinear, butter, lfilter, sosfilt

    fs = float(spec["sampling_rate_hz"])
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    n = max(8, int(round((t[-1] - t[0]) * fs)) + 1)
    t_u = t[0] + np.arange(n, dtype=float) / fs
    p_u = np.interp(t_u, t, p)

    stf = spec["sensor_transfer_function"]
    wn = 2.0 * np.pi * float(stf["f_n_hz"])
    zeta = float(stf["zeta"])
    b, a = bilinear([wn * wn], [1.0, 2.0 * zeta * wn, wn * wn], fs=fs)
    p_s = lfilter(b, a, p_u)

    aa = spec["antialias_filter"]
    nyq = 0.5 * fs
    cutoff = float(aa["cutoff_fraction_of_nyquist"]) * nyq
    cutoff = min(max(cutoff, 1.0), 0.99 * nyq)
    sos = butter(int(aa["order"]), cutoff, btype="low", fs=fs, output="sos")
    p_f = sosfilt(sos, p_s)
    return t_u.astype(np.float32), p_f.astype(np.float32)


def end_to_end_f_eff_hz(spec: dict) -> float:
    """-3 dB point of sensor × antialias, evaluated on a unit impulse at fs."""
    fs = float(spec["sampling_rate_hz"])
    n = 8192
    t = np.arange(n, dtype=float) / fs
    x = np.zeros(n)
    x[0] = 1.0
    _, y = apply_observation_chain(t, x, spec)
    Y = np.fft.rfft(y)
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(Y)
    mag /= mag[1] + 1e-30
    below = np.where(mag[1:] <= 10 ** (-3.0 / 20.0))[0]
    if below.size == 0:
        return float(freq[-1])
    return float(freq[below[0] + 1])
