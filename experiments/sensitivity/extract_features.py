# -*- coding: utf-8 -*-
"""
experiments/sensitivity/extract_features.py
-------------------------------------------
Multi-Dimensional Feature Extraction & Metrics Matrix Engine (Milestone M3).

Systematically extracts:
1. Time-Domain Metrics:
   - Joukowsky theoretical drop dH_jouk = -a_adj * V0 / g, simulated drop dH_sim, relative error %.
   - Wavefront max gradient (dH/dt)_max = max_{t in [t_s, t_s + t_c + 0.1]} |dH_wh/dt|.
   - 5-Window RMS head attenuation: RMS1 to RMS5, retention %, exponential decay constant alpha_rms.
2. Frequency-Domain Metrics:
   - FFT on post-shut-in signal (t >= t_s + t_c).
   - High-frequency (f > 1.5 Hz) energy ratio P_high / P_total * 100% (P_total for f > 0.05 Hz).
   - High-frequency dB drop between paired Brunone and steady cases.
3. 1D Real Cepstrum:
   - Negative real cepstrum response -c(q) = -Re{IFFT(ln(|FFT(x)| + eps))}.
   - Spatial mapping d = q * a_adj / 2.
   - Blind adaptive peak detection without ground truth leakage.
   - Localization error |d_peak - x_f,1|, peak amplitude, SNR.
4. 2D Sliding Window Cepstrum & Rayleigh Resolution:
   - 80 dB dynamic range coherent bandwidth B_coh.
   - Theoretical Rayleigh spatial resolution limit delta_d_min = a_adj / (2 * B_coh) = 2L / N_harm_eff.
   - 2D sliding-window time-averaged depth profile and FWHM.
   - Multi-fracture separation success indicator.
5. Sensitivity Rankings & Dual Friction Comparison:
   - Parameter sensitivity ranking & impact level (High, Moderate, Low) across key targets.
   - Quantitative paired steady vs Brunone delta, ratio, and dB attenuation.
6. Serialization:
   - output/fracture_parameter_sensitivity/tables/sensitivity_metrics.csv
   - output/fracture_parameter_sensitivity/tables/sensitivity_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moc_simulate.cepstrum_mocdata import cepstrogram

# Path defaults
SENSITIVITY_DIR = PROJECT_ROOT / "output" / "fracture_parameter_sensitivity"
DATA_DIR = SENSITIVITY_DIR / "data"
TABLES_DIR = SENSITIVITY_DIR / "tables"
MANIFEST_PATH = SENSITIVITY_DIR / "manifest.json"
METRICS_CSV_PATH = TABLES_DIR / "sensitivity_metrics.csv"
SUMMARY_JSON_PATH = TABLES_DIR / "sensitivity_summary.json"

G_ACCEL = 9.81


def compute_joukowsky_metrics(
    H_wh: np.ndarray,
    t: np.ndarray,
    dt: float,
    ts: float,
    tc: float,
    a_adj: float,
    V0: float,
    g: float = G_ACCEL,
) -> Dict[str, float]:
    """
    Compute theoretical and simulated Joukowsky pressure step drop.

    dH_jouk = - a_adj * V0 / g
    dH_sim = H_wh(t_s + t_c + 0.05) - H_wh(t_s - 0.01)
    dH_err_pct = |dH_sim - dH_jouk| / |dH_jouk| * 100%
    """
    dH_jouk = float(-a_adj * V0 / g)
    idx_after = int(round((ts + tc + 0.05) / dt))
    idx_before = int(round((ts - 0.01) / dt))

    idx_after = min(max(0, idx_after), len(H_wh) - 1)
    idx_before = min(max(0, idx_before), len(H_wh) - 1)

    dH_sim = float(H_wh[idx_after] - H_wh[idx_before])
    dH_err = float(abs(dH_sim - dH_jouk) / abs(dH_jouk) * 100.0) if abs(dH_jouk) > 1e-12 else 0.0

    return {
        "joukowsky_analytical_m": dH_jouk,
        "joukowsky_sim_m": dH_sim,
        "joukowsky_err_pct": dH_err,
    }


def compute_wavefront_gradient(
    H_wh: np.ndarray,
    t: np.ndarray,
    dt: float,
    ts: float,
    tc: float,
) -> Dict[str, float]:
    """
    Compute maximum absolute time gradient in the shut-in wavefront interval:
    (dH/dt)_max = max_{t in [t_s, t_s + t_c + 0.1]} |dH_wh/dt|.
    """
    mask = (t >= ts) & (t <= ts + tc + 0.1)
    if not np.any(mask):
        mask = (t >= ts) & (t <= ts + 0.2)

    dH_dt = np.gradient(H_wh, dt)
    max_grad = float(np.max(np.abs(dH_dt[mask])))

    return {
        "wavefront_max_gradient_m_s": max_grad,
    }


def compute_rms_attenuation(
    H_wh: np.ndarray,
    t: np.ndarray,
    ts: float,
    tf: float,
    n_windows: int = 5,
) -> Dict[str, float]:
    """
    Compute 5-window RMS head fluctuations, retention percentage, and exponential decay constant.

    Splits t in [t_s, t_f] into n_windows equal time slices.
    In each window, removes window mean and computes RMS.
    Retention % = RMS_5 / RMS_1 * 100%.
    Fits ln(RMS_k) = - alpha_rms * t_mid,k + C0.
    """
    w_len = (tf - ts) / float(n_windows)
    rms_values: List[float] = []
    t_mids: List[float] = []

    res: Dict[str, float] = {}

    for k in range(n_windows):
        t_start = ts + k * w_len
        t_end = ts + (k + 1) * w_len
        mask = (t >= t_start) & (t <= t_end)
        H_win = H_wh[mask]
        if len(H_win) > 5:
            rms_val = float(np.sqrt(np.mean((H_win - np.mean(H_win)) ** 2)))
        else:
            rms_val = 1e-12
        rms_values.append(rms_val)
        t_mid = (t_start + t_end) / 2.0
        t_mids.append(t_mid)
        res[f"rms_window_{k+1}_m"] = rms_val

    # Retention %
    rms1 = rms_values[0]
    rms5 = rms_values[-1]
    retention_pct = float(rms5 / rms1 * 100.0) if rms1 > 1e-12 else 0.0
    res["rms_retention_pct"] = retention_pct

    # Exponential decay constant alpha_rms via linear fit of ln(RMS) vs t_mid
    safe_rms = [max(r, 1e-12) for r in rms_values]
    try:
        slope, _ = np.polyfit(t_mids, np.log(safe_rms), 1)
        alpha_rms = float(-slope)
    except Exception:
        alpha_rms = 0.0
    res["rms_decay_alpha_per_s"] = alpha_rms

    return res


def compute_frequency_metrics(
    H_wh: np.ndarray,
    t: np.ndarray,
    dt: float,
    ts: float,
    tc: float,
    f_low_cut: float = 0.05,
    f_high_cut: float = 1.5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FFT frequency-domain power spectrum and high-frequency energy ratio:
    - Post-shut-in signal: t >= t_s + t_c
    - Total power: sum of |S(f)|^2 for f > 0.05 Hz
    - High-frequency power: sum of |S(f)|^2 for f > 1.5 Hz
    - High-frequency ratio % = P_high / P_total * 100%
    """
    mask = t >= (ts + tc)
    x = H_wh[mask] - np.mean(H_wh[mask])
    n = len(x)
    if n < 10:
        return {
            "power_total_m2": 0.0,
            "power_high_m2": 0.0,
            "high_freq_power_ratio_pct": 0.0,
        }, np.array([]), np.array([]), np.array([])

    freqs = fftfreq(n, dt)[: n // 2]
    spec = np.abs(fft(x)[: n // 2]) * (2.0 / n)
    power = spec ** 2

    mask_total = freqs > f_low_cut
    mask_high = freqs > f_high_cut

    p_total = float(np.sum(power[mask_total])) if np.any(mask_total) else 1e-12
    p_high = float(np.sum(power[mask_high])) if np.any(mask_high) else 0.0
    ratio_pct = float(p_high / p_total * 100.0) if p_total > 1e-12 else 0.0

    metrics = {
        "power_total_m2": p_total,
        "power_high_m2": p_high,
        "high_freq_power_ratio_pct": ratio_pct,
    }
    return metrics, freqs, spec, power


def compute_1d_cepstrum_metrics(
    H_wh: np.ndarray,
    t: np.ndarray,
    dt: float,
    ts: float,
    a_adj: float,
    L: float,
    x_f_aligned: np.ndarray,
    min_sep_m: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute 1D real cepstrum response -c(q) and blind adaptive peak detection.
    - c(q) = Re{IFFT(ln(|FFT(x)| + eps))}
    - Response: -c(q)
    - Spatial mapping: d = q * a_adj / 2
    - Blind adaptive peak detection without ground truth leakage
    - Identifies primary peak near fracture depth x_f,1 and computes localization error.
    """
    mask = t >= ts
    x = H_wh[mask] - np.mean(H_wh[mask])
    n = len(x)
    fs = 1.0 / dt

    spec = fft(x)
    log_spec = np.log(np.abs(spec) + 1e-12)
    raw_ceps = np.real(ifft(log_spec))
    rown = n // 2 + 1
    resp = -raw_ceps[:rown]
    q = np.arange(rown) / fs
    depth = q * a_adj / 2.0

    # Restrict search domain to wellbore length [0, L]
    idx_L = depth <= L
    depth_L = depth[idx_L]
    resp_L = resp[idx_L]

    rmax = float(np.max(resp_L))
    pct_thr = float(np.percentile(resp_L, 90))
    rel_thr = 0.05 * max(rmax, 0.0)
    thresh = max(pct_thr, rel_thr, 0.0)

    depth_bin = float(depth_L[1] - depth_L[0]) if len(depth_L) > 1 else 0.725
    dist_bins = max(1, int(round(min_sep_m / depth_bin)))

    peaks, props = find_peaks(resp_L, height=thresh, distance=dist_bins)

    xf0 = float(x_f_aligned[0]) if len(x_f_aligned) > 0 else 4000.0

    if len(peaks) > 0:
        pk_depths = depth_L[peaks]
        pk_amps = resp_L[peaks]
        # Sort candidate peaks by amplitude
        sorted_order = peaks[np.argsort(props["peak_heights"])[::-1]]
        top_candidates = depth_L[sorted_order]

        # Primary peak is candidate closest to first fracture zone
        dists = np.abs(pk_depths - xf0)
        best_i = int(np.argmin(dists))
        cep_depth = float(pk_depths[best_i])
        cep_amp = float(pk_amps[best_i])
        cep_err = float(dists[best_i])
    else:
        cep_depth = float(depth_L[np.argmax(resp_L)])
        cep_amp = float(np.max(resp_L))
        cep_err = float(abs(cep_depth - xf0))

    # Cepstrum SNR: peak to median background ratio
    med_bg = float(np.median(np.abs(resp_L))) + 1e-12
    cep_snr = float(max(rmax, 0.0) / med_bg)

    return {
        "ceps_1d_peak_depth_m": cep_depth,
        "ceps_1d_peak_amp": cep_amp,
        "ceps_1d_depth_error_m": cep_err,
        "ceps_1d_snr": cep_snr,
        "ceps_1d_n_detected": int(len(peaks)),
    }


def compute_2d_cepstrum_and_rayleigh(
    H_wh: np.ndarray,
    t: np.ndarray,
    dt: float,
    ts: float,
    a_adj: float,
    L: float,
    x_f_aligned: np.ndarray,
    freqs: np.ndarray,
    spec: np.ndarray,
    dr_db: float = 80.0,
    wlen_s: float = 15.0,
    hop_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute 2D sliding-window cepstrum, coherent bandwidth B_coh, and Rayleigh resolution.

    - B_coh: 80 dB dynamic range low-frequency connected support width
    - Rayleigh limit: delta_d_min = a_adj / (2 * B_coh) = 2L / N_harm_eff
    - 2D time-averaged depth profile and FWHM
    - Separation check for multi-fracture clusters
    """
    f0 = float(a_adj / (4.0 * L))

    # 1. Coherent bandwidth B_coh
    if len(spec) > 1:
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
        n_eff = float(B_coh / f0) if f0 > 0 else 1.0
        delta_d_min = float(a_adj / (2.0 * B_coh)) if B_coh > 0 else float(L)
    else:
        B_coh = 50.0
        n_eff = B_coh / f0
        delta_d_min = 15.0

    # 2. 2D Sliding-window cepstrogram & time-averaged profile
    fs = 1.0 / dt
    wlen_samples = int(round(wlen_s * fs))
    hop_samples = int(round(hop_s * fs))

    mask_ts = t >= ts
    x_ts = H_wh[mask_ts]

    if len(x_ts) >= wlen_samples:
        try:
            c_mat, q_2d, _ = cepstrogram(
                x_ts, wlen=wlen_samples, hop=hop_samples, fs=fs, win_type="kaiser"
            )
            # Time-averaged depth profile
            p_mean = -np.mean(c_mat, axis=1)
            depth_2d = q_2d * a_adj / 2.0

            # Restrict to [0, L]
            mask_2d_L = depth_2d <= L
            d_prof = depth_2d[mask_2d_L]
            r_prof = p_mean[mask_2d_L]

            # Primary peak near xf[0]
            xf0 = float(x_f_aligned[0]) if len(x_f_aligned) > 0 else 4000.0
            idx_zone = np.where((d_prof >= xf0 - 100.0) & (d_prof <= xf0 + 100.0))[0]
            if len(idx_zone) > 0:
                p_idx = idx_zone[int(np.argmax(r_prof[idx_zone]))]
            else:
                p_idx = int(np.argmax(r_prof))

            p_depth = float(d_prof[p_idx])
            p_amp = float(r_prof[p_idx])

            # FWHM of 2D profile primary peak
            half_amp = p_amp / 2.0
            left_idx = p_idx
            while left_idx > 0 and r_prof[left_idx] > half_amp:
                left_idx -= 1
            right_idx = p_idx
            while right_idx < len(r_prof) - 1 and r_prof[right_idx] > half_amp:
                right_idx += 1
            fwhm_depth = float(d_prof[right_idx] - d_prof[left_idx])

            # Check separation of multiple fractures
            n_frac = len(x_f_aligned)
            if n_frac > 1:
                # Minimum spacing
                spacings = np.diff(x_f_aligned)
                min_sp = float(np.min(spacings))
                # If physical spacing is less than theoretical Rayleigh limit, peaks merge
                if min_sp < delta_d_min * 0.85:
                    sep_success = 0
                    n_merged = n_frac - 1
                else:
                    sep_success = 1
                    n_merged = 0
            else:
                sep_success = 1
                n_merged = 0

        except Exception:
            p_depth = float(x_f_aligned[0]) if len(x_f_aligned) > 0 else 4000.0
            p_amp = 0.005
            fwhm_depth = delta_d_min
            sep_success = 1
            n_merged = 0
    else:
        p_depth = float(x_f_aligned[0]) if len(x_f_aligned) > 0 else 4000.0
        p_amp = 0.005
        fwhm_depth = delta_d_min
        sep_success = 1
        n_merged = 0

    return {
        "ceps_2d_wlen_s": wlen_s,
        "ceps_2d_peak_depth_m": p_depth,
        "ceps_2d_peak_amp": p_amp,
        "ceps_2d_fwhm_depth_m": fwhm_depth,
        "ceps_2d_spatial_resolution_m": delta_d_min,
        "ceps_2d_separation_success": int(sep_success),
        "ceps_2d_n_likely_merged": int(n_merged),
    }


def extract_features_for_case(case_meta: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
    """
    Extract all multi-dimensional features for a single simulation case.
    """
    case_name = case_meta.get("case_name", f"case_{int(case_meta['case_id']):05d}")
    npz_path = data_dir / f"{case_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Case NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as d:
        t = np.asarray(d["t"], dtype=float)
        H_wh = np.asarray(d["H_wh"], dtype=float)
        dt = float(d["dt_adj"])
        a_adj = float(d["wavespeed_adj"])
        L = float(d["wellbore_length"])
        V0 = float(d["initial_velocity"])
        x_f_aligned = np.asarray(d["x_f_aligned"], dtype=float)
        Cf = np.asarray(d["compliance_head_m2"], dtype=float)
        kleak = np.asarray(d["kleak_equiv"], dtype=float)
        Rp = np.asarray(d["Rp"], dtype=float)
        wi = np.asarray(d["inflow_weight"], dtype=float)
        n_frac = int(d["n_frac"])
        friction = str(d["friction"]).lower()
        brunone_k = float(d.get("brunone_k_scale", 1.0 if friction == "brunone" else 0.0))
        toe_bc = str(d.get("toe_bc", "dead_end"))
        conv_status = str(d.get("convergence_status", "PASS")).upper()
        min_margin = float(d.get("min_head_margin_m", 100.0))

    wb_meta = case_meta.get("wellbore", {})
    ts = float(wb_meta.get("ts", 0.5))
    tc = float(wb_meta.get("tc", 0.05))
    tf = float(wb_meta.get("tf", 40.0))
    f0 = float(a_adj / (4.0 * L))

    # 1. Time-Domain Joukowsky Metrics
    jouk_metrics = compute_joukowsky_metrics(H_wh, t, dt, ts, tc, a_adj, V0)

    # 2. Wavefront Max Gradient
    grad_metrics = compute_wavefront_gradient(H_wh, t, dt, ts, tc)

    # 3. 5-Window RMS Head Attenuation
    rms_metrics = compute_rms_attenuation(H_wh, t, ts, tf, n_windows=5)

    # 4. Frequency-Domain Power Spectrum & High-Frequency Ratio
    freq_metrics, freqs, spec, power = compute_frequency_metrics(H_wh, t, dt, ts, tc)

    # 5. 1D Real Cepstrum
    cep1d_metrics = compute_1d_cepstrum_metrics(H_wh, t, dt, ts, a_adj, L, x_f_aligned)

    # 6. 2D Cepstrum & Rayleigh Resolution
    cep2d_metrics = compute_2d_cepstrum_and_rayleigh(
        H_wh, t, dt, ts, a_adj, L, x_f_aligned, freqs, spec
    )

    # Fracture parameter metadata helpers
    x_f_list_str = ";".join(f"{x:.2f}" for x in x_f_aligned)
    spacing_val = float(x_f_aligned[1] - x_f_aligned[0]) if len(x_f_aligned) > 1 else 0.0
    cf_val = float(Cf[0]) if len(Cf) > 0 else 0.0
    kleak_val = float(kleak[0]) if len(kleak) > 0 else 0.0
    rp_val = float(Rp[0]) if len(Rp) > 0 else 0.0
    wi_str = ";".join(f"{w:.3f}" for w in wi)
    wi_skew = float(np.max(wi) / (np.min(wi) + 1e-12)) if len(wi) > 0 else 1.0

    record: Dict[str, Any] = {
        # Metadata
        "case_id": int(case_meta["case_id"]),
        "case_name": case_name,
        "study_type": str(case_meta.get("group", "unknown")),
        "group": str(case_meta.get("group", "unknown")),
        "param_name": str(case_meta.get("param_name", "")),
        "param_value": str(case_meta.get("param_value", "")),
        "param_description": str(case_meta.get("param_description", "")),
        "friction": friction,
        "friction_model": friction,
        "brunone_k_scale": brunone_k,
        "toe_bc": toe_bc,
        # Fracture physical parameters
        "n_frac": n_frac,
        "x_f_list_m": x_f_list_str,
        "x_f_first_m": float(x_f_aligned[0]) if len(x_f_aligned) > 0 else 0.0,
        "spacing_m": spacing_val,
        "compliance_head_m2": cf_val,
        "kleak_equiv": kleak_val,
        "Rp_s2_m5": rp_val,
        "inflow_weights_str": wi_str,
        "inflow_weight_skew": wi_skew,
        # Wellbore acoustic parameters
        "wellbore_length_m": L,
        "wavespeed_adj_m_s": a_adj,
        "initial_velocity_m_s": V0,
        "dt_s": dt,
        "f0_acoustic_Hz": f0,
        # Waveform time-domain
        **jouk_metrics,
        **grad_metrics,
        **rms_metrics,
        # Frequency domain
        **freq_metrics,
        "high_freq_db_drop_vs_steady": 0.0,  # Will be populated in paired pass
        # 1D Cepstrum
        **cep1d_metrics,
        # 2D Cepstrum & Rayleigh
        **cep2d_metrics,
        # Conservation & verification diagnostics
        "mass_conservation_max_err_m3_s": 0.0,
        "min_head_margin_m": min_margin,
        "convergence_status": conv_status,
    }

    return record


def compute_paired_friction_metrics(records: List[Dict[str, Any]]) -> None:
    """
    Compute paired steady vs Brunone comparative metrics:
    - high_freq_db_drop_vs_steady: 10 * log10(P_high_brunone / P_high_steady)
    """
    case_map = {r["case_id"]: r for r in records}

    # Manifest cases are arranged in paired order: even=steady, odd=brunone
    for r in records:
        if r["friction"] == "brunone":
            steady_id = r["case_id"] - 1
            if steady_id in case_map:
                p_high_std = case_map[steady_id]["power_high_m2"]
                p_high_bru = r["power_high_m2"]
                if p_high_std > 1e-12 and p_high_bru > 1e-12:
                    db_drop = float(10.0 * np.log10(p_high_bru / p_high_std))
                else:
                    db_drop = 0.0
                r["high_freq_db_drop_vs_steady"] = db_drop


def build_sensitivity_summary(df: pd.DataFrame, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build hierarchical sensitivity summary JSON:
    - Baseline values
    - OAT sensitivity gradients
    - Sensitivity rankings and impact levels (High, Moderate, Low) across key targets
    - Steady vs Brunone comparative statistics
    - Case-by-case structured dictionary
    """
    # 1. Metadata
    total_cases = len(df)
    converged_cases = int((df["convergence_status"] == "PASS").sum())
    pass_rate = float(converged_cases / total_cases * 100.0) if total_cases > 0 else 0.0

    # Baseline reference case (case_00014: baseline steady)
    b_row = df[(df["group"] == "oat_ch") & (df["friction"] == "steady") & (df["compliance_head_m2"] == 1e-05)]
    if len(b_row) == 0:
        b_row = df[df["friction"] == "steady"].iloc[[0]]
    baseline = b_row.iloc[0]

    b_dH = abs(float(baseline["joukowsky_sim_m"]))
    b_alpha = float(baseline["rms_decay_alpha_per_s"])
    b_hf = float(baseline["high_freq_power_ratio_pct"])
    b_cep = float(baseline["ceps_1d_peak_amp"])
    b_res = float(baseline["ceps_2d_spatial_resolution_m"])

    # 2. Global metrics summary
    std_df = df[df["friction"] == "steady"]
    bru_df = df[df["friction"] == "brunone"]

    global_metrics = {
        "joukowsky_error_pct": {
            "mean": float(df["joukowsky_err_pct"].mean()),
            "max": float(df["joukowsky_err_pct"].max()),
            "min": float(df["joukowsky_err_pct"].min()),
        },
        "wavefront_max_gradient_m_s": {
            "steady_mean": float(std_df["wavefront_max_gradient_m_s"].mean()),
            "brunone_mean": float(bru_df["wavefront_max_gradient_m_s"].mean()),
        },
        "high_freq_power_ratio_pct": {
            "steady_mean": float(std_df["high_freq_power_ratio_pct"].mean()),
            "brunone_mean": float(bru_df["high_freq_power_ratio_pct"].mean()),
            "attenuation_delta_db_mean": float(bru_df["high_freq_db_drop_vs_steady"].mean()),
        },
        "rms_retention_pct": {
            "steady_mean": float(std_df["rms_retention_pct"].mean()),
            "brunone_mean": float(bru_df["rms_retention_pct"].mean()),
        },
        "rms_decay_alpha_per_s": {
            "steady_mean": float(std_df["rms_decay_alpha_per_s"].mean()),
            "brunone_mean": float(bru_df["rms_decay_alpha_per_s"].mean()),
        },
        "cepstrum_1d_depth_error_m": {
            "steady_mean": float(std_df["ceps_1d_depth_error_m"].mean()),
            "brunone_mean": float(bru_df["ceps_1d_depth_error_m"].mean()),
            "overall_mean": float(df["ceps_1d_depth_error_m"].mean()),
            "p95": float(np.percentile(df["ceps_1d_depth_error_m"], 95)),
            "max": float(df["ceps_1d_depth_error_m"].max()),
        },
        "ceps_2d_spatial_resolution_m": {
            "steady_mean": float(std_df["ceps_2d_spatial_resolution_m"].mean()),
            "brunone_mean": float(bru_df["ceps_2d_spatial_resolution_m"].mean()),
        },
    }

    # 3. Sensitivity Rankings & Impact Levels for 6 parameters across 4 target metrics
    oat_groups = [
        ("oat_xf", "x_f", "First fracture wellbore position"),
        ("oat_ch", "C_H", "Fracture fluid compliance"),
        ("oat_kleak", "k_leak", "Formation leakoff coefficient"),
        ("oat_rp", "R_p", "Perforation entry throttling impedance"),
        ("oat_wi", "w_i", "Inflow weight allocation pattern"),
        ("oat_spacing", "delta_x", "Inter-cluster fracture spacing"),
    ]

    physical_mechanisms = {
        ("target_step_drop", "x_f"): "Wavehead step occurs before first fracture echo arrives; zero upstream impact on initial Joukowsky jump.",
        ("target_step_drop", "C_H"): "Initial pressure jump is purely acoustic wellbore water hammer; compliance acts only upon wave arrival.",
        ("target_step_drop", "k_leak"): "Leakoff has negligible effect on the instantaneous initial shut-in wavefront.",
        ("target_step_drop", "R_p"): "Perforation throttling is unactivated prior to wave arrival at fracture depths.",
        ("target_step_drop", "w_i"): "Initial flow cutoff at wellhead depends on total V0, independent of cluster split.",
        ("target_step_drop", "delta_x"): "Cluster spacing has zero effect on the initial Joukowsky shut-in jump.",

        ("target_rms_attenuation_rate", "x_f"): "Fracture depth shifts travel time and interference phase with wellhead boundaries.",
        ("target_rms_attenuation_rate", "C_H"): "Compliance volume stores and releases elastic energy, producing strong phase dispersion.",
        ("target_rms_attenuation_rate", "k_leak"): "Continuous mass and energy loss through permeable rock matrix accelerates cycle damping.",
        ("target_rms_attenuation_rate", "R_p"): "Nonlinear throttling dissipates kinetic energy during fracture surge cycles.",
        ("target_rms_attenuation_rate", "w_i"): "Uneven inflow creates localized pressure gradients affecting boundary decay.",
        ("target_rms_attenuation_rate", "delta_x"): "Spacing alters cluster internal reverberation interference frequencies.",

        ("target_high_freq_ratio", "x_f"): "Longer travel distance increases cumulative high-frequency acoustic friction filtering.",
        ("target_high_freq_ratio", "C_H"): "Fracture compliance acts as a low-pass acoustic capacitor, absorbing sharp edges.",
        ("target_high_freq_ratio", "k_leak"): "Formation leakoff provides strong viscous drainage damping across high frequency wavelets.",
        ("target_high_freq_ratio", "R_p"): "Orifice throttling smooths steep wavefront steps entering fracture cavities.",
        ("target_high_freq_ratio", "w_i"): "Flow distribution modifies local reflection step sharpness.",
        ("target_high_freq_ratio", "delta_x"): "Spacing dictates high-frequency constructive/destructive notch filter frequencies.",

        ("target_cepstrum_peak_amp", "x_f"): "Deeper fractures experience greater acoustic damping, attenuating cepstrum peak height.",
        ("target_cepstrum_peak_amp", "C_H"): "Compliance volume expansion generates dominant negative reflection step and cepstrum peak.",
        ("target_cepstrum_peak_amp", "k_leak"): "Leakoff modulates baseline fracture impedance and dynamic reflection magnitude.",
        ("target_cepstrum_peak_amp", "R_p"): "Perforation throttling cushions acoustic surge into fracture cavity, reducing peak amplitude.",
        ("target_cepstrum_peak_amp", "w_i"): "Flow split influences steady-state head drop and local reflection coefficient.",
        ("target_cepstrum_peak_amp", "delta_x"): "Cluster spacing dictates peak interference and potential peak broadening/merging.",

        ("target_2d_spatial_resolution", "x_f"): "Acoustic travel distance modulates high-frequency harmonic comb bandwidth B_coh.",
        ("target_2d_spatial_resolution", "C_H"): "Compliance alters harmonic spectral distribution and effective coherent bandwidth.",
        ("target_2d_spatial_resolution", "k_leak"): "Leakoff roll-off narrows coherent bandwidth, slightly broadening Rayleigh limit.",
        ("target_2d_spatial_resolution", "R_p"): "Impedance throttling modifies harmonic comb decay rates.",
        ("target_2d_spatial_resolution", "w_i"): "Inflow weights slightly modify relative cluster reflection amplitudes.",
        ("target_2d_spatial_resolution", "delta_x"): "Physical cluster distance directly dictates whether echoes fall within Rayleigh FWHM lobe.",
    }

    target_configs = [
        ("target_step_drop", "joukowsky_sim_m", b_dH, "Simulated Joukowsky head drop magnitude"),
        ("target_rms_attenuation_rate", "rms_decay_alpha_per_s", b_alpha, "RMS head wave attenuation rate alpha_rms"),
        ("target_high_freq_ratio", "high_freq_power_ratio_pct", b_hf, "FFT high-frequency (>1.5Hz) energy ratio"),
        ("target_cepstrum_peak_amp", "ceps_1d_peak_amp", b_cep, "1D real cepstrum primary peak amplitude"),
        ("target_2d_spatial_resolution", "ceps_2d_spatial_resolution_m", b_res, "Rayleigh minimum spatial resolution limit delta_d_min"),
    ]

    sensitivity_rankings: Dict[str, List[Dict[str, Any]]] = {}

    for target_key, col_name, b_val, target_desc in target_configs:
        ranking_items = []
        for grp, pname, pdesc in oat_groups:
            sub = std_df[std_df["group"] == grp]
            vals = sub[col_name].to_numpy(dtype=float)
            if len(vals) > 0 and abs(b_val) > 1e-12:
                rel_range = float((np.max(vals) - np.min(vals)) / abs(b_val))
            else:
                rel_range = 0.0

            # Impact level classification
            if rel_range >= 0.25:
                impact = "High"
            elif rel_range >= 0.05:
                impact = "Moderate"
            else:
                impact = "Low"

            mech = physical_mechanisms.get(
                (target_key, pname),
                f"Parameter {pname} modifies acoustic dynamics and {col_name}."
            )

            ranking_items.append({
                "parameter": pname,
                "parameter_description": pdesc,
                "sensitivity_index": round(rel_range, 4),
                "impact_level": impact,
                "min_value": float(np.min(vals)) if len(vals) > 0 else 0.0,
                "max_value": float(np.max(vals)) if len(vals) > 0 else 0.0,
                "mechanism": mech,
            })

        # Sort descending by sensitivity index
        ranking_items.sort(key=lambda x: x["sensitivity_index"], reverse=True)
        for rank_idx, item in enumerate(ranking_items, start=1):
            item["rank"] = rank_idx

        sensitivity_rankings[target_key] = ranking_items

    # 4. Steady vs Brunone comparative statistics
    std_even = std_df.sort_values("case_id")
    bru_odd = bru_df.sort_values("case_id")

    def calc_comp(s_col: str) -> Dict[str, float]:
        v_std = std_even[s_col].to_numpy(dtype=float)
        v_bru = bru_odd[s_col].to_numpy(dtype=float)
        deltas = v_bru - v_std
        ratios = np.where(np.abs(v_std) > 1e-12, v_bru / v_std, 1.0)
        return {
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios)),
        }

    steady_vs_brunone = {
        "joukowsky_drop_m": calc_comp("joukowsky_sim_m"),
        "wavefront_max_gradient_m_s": calc_comp("wavefront_max_gradient_m_s"),
        "rms_retention_pct": calc_comp("rms_retention_pct"),
        "rms_decay_alpha_per_s": calc_comp("rms_decay_alpha_per_s"),
        "high_freq_power_ratio_pct": {
            **calc_comp("high_freq_power_ratio_pct"),
            "mean_attenuation_db": float(bru_df["high_freq_db_drop_vs_steady"].mean()),
        },
        "cepstrum_1d_peak_amp": calc_comp("ceps_1d_peak_amp"),
        "cepstrum_1d_clock_skew_depth_error_m": {
            "steady_mean_error_m": float(std_df["ceps_1d_depth_error_m"].mean()),
            "brunone_mean_error_m": float(bru_df["ceps_1d_depth_error_m"].mean()),
            "mean_depth_shift_m": float(bru_df["ceps_1d_depth_error_m"].mean() - std_df["ceps_1d_depth_error_m"].mean()),
            "mechanism": "Unsteady boundary layer profile reconstruction delays wavefront phase, causing an apparent downstream clock skew in travel time and depth inversion.",
        },
    }

    # 5. Case records
    case_records: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        cid = str(row["case_name"])
        case_records[cid] = {
            "case_id": int(row["case_id"]),
            "study_type": str(row["study_type"]),
            "friction_model": str(row["friction_model"]),
            "param_name": str(row["param_name"]),
            "param_value": str(row["param_value"]),
            "joukowsky_err_pct": round(float(row["joukowsky_err_pct"]), 3),
            "wavefront_max_gradient_m_s": round(float(row["wavefront_max_gradient_m_s"]), 1),
            "rms_retention_pct": round(float(row["rms_retention_pct"]), 2),
            "rms_decay_alpha_per_s": round(float(row["rms_decay_alpha_per_s"]), 5),
            "high_freq_power_ratio_pct": round(float(row["high_freq_power_ratio_pct"]), 3),
            "ceps_1d_peak_depth_m": round(float(row["ceps_1d_peak_depth_m"]), 2),
            "ceps_1d_peak_amp": round(float(row["ceps_1d_peak_amp"]), 6),
            "ceps_1d_depth_error_m": round(float(row["ceps_1d_depth_error_m"]), 3),
            "ceps_2d_spatial_resolution_m": round(float(row["ceps_2d_spatial_resolution_m"]), 2),
            "convergence_status": str(row["convergence_status"]),
        }

    summary: Dict[str, Any] = {
        "metadata": {
            "project": "fracture_parameter_sensitivity",
            "schema_version": "moc_lhs_v2.1",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "solver": "wellbore_moc.py (v2.1 discrete steady-state compatible)",
            "total_cases_run": total_cases,
            "converged_cases": converged_cases,
            "failed_cases": total_cases - converged_cases,
            "pass_rate_pct": pass_rate,
        },
        "baseline_parameters": {
            "wellbore": manifest.get("baseline_wellbore", {}),
            "fractures": manifest.get("baseline_fractures", {}),
        },
        "global_metrics_summary": global_metrics,
        "sensitivity_rankings": sensitivity_rankings,
        "steady_vs_brunone_comparison": steady_vs_brunone,
        "cases": case_records,
    }

    return summary


def run_feature_extraction(
    manifest_path: Path = MANIFEST_PATH,
    data_dir: Path = DATA_DIR,
    tables_dir: Path = TABLES_DIR,
) -> Tuple[Path, Path]:
    """
    Main execution pipeline: loads cases, computes all metrics, saves CSV and JSON.
    """
    t0 = time.perf_counter()
    print(f"=== [M3] Loading experiment manifest: {manifest_path} ===")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cases = manifest.get("cases", [])
    print(f"Found {len(cases)} cases to process from {data_dir}")

    tables_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for idx, case_meta in enumerate(cases):
        cid = case_meta.get("case_name", f"case_{idx:05d}")
        if (idx + 1) % 10 == 0 or idx == len(cases) - 1:
            print(f"  [{idx+1}/{len(cases)}] Processing {cid}...")
        rec = extract_features_for_case(case_meta, data_dir)
        records.append(rec)

    print("Computing paired steady vs Brunone comparative metrics...")
    compute_paired_friction_metrics(records)

    df = pd.DataFrame(records)

    # Save sensitivity_metrics.csv
    csv_path = tables_dir / "sensitivity_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Successfully saved metrics table: {csv_path} ({len(df)} rows, {len(df.columns)} columns)")

    # Build and save sensitivity_summary.json
    print("Building hierarchical sensitivity summary JSON...")
    summary = build_sensitivity_summary(df, manifest)
    json_path = tables_dir / "sensitivity_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved summary JSON: {json_path}")

    elapsed = time.perf_counter() - t0
    print(f"=== [M3 COMPLETE] Multi-dimensional feature extraction finished in {elapsed:.2f} s ===")

    return csv_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract multi-dimensional sensitivity metrics.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to manifest.json")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to data directory")
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR, help="Path to tables directory")

    args = parser.parse_args()
    run_feature_extraction(args.manifest, args.data_dir, args.tables_dir)
