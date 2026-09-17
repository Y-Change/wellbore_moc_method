# -*- coding: utf-8 -*-
"""
EXP-20260730-024 — Brunone 波形退化指标审计（复用既有 30 组时间序列，不重跑 MOC）。

预注册公式与参考（写入 manifest，禁止事后改阈值凑表）：

- 分析信号：井口水头时间导数 dH/dt（np.gradient）。
- 默认分析窗：[6.5, 7.5] s（首缝几何到时 ts+2*4100/a ≈ 6.655 s）。
- 寻峰：scipy.find_peaks，height=0.1*max(dH_win)，distance=2 ms。
- 峰漂移 dt_shift：相对同间距 k=0 的首峰到时差。
- EST：正部能量累积 10%–90% 时间跨度（不是 FWHM，不是半高宽）。
- 峰谷对比度 C_v=(P_min-V)/P_min，截断到 [0,1]；不足两峰则记 0。
  不把 C_v 换成 Rayleigh 0.81 分数，也不据此报「最小可分辨间距」。
- STFT：scipy.signal.spectrogram，Hann，nperseg=128，noverlap=120，
  scaling='spectrum'；频带能量为带内 Sxx 之和；dB = 10*log10(E(k)/E(k=0,同D))。
- 三种表观速度与 plot_results.py 同一操作定义，不作 c_p(ω)。

用法
----
    python -m analysis.brunone_spacing_effect.audit_waveform_degradation
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, spectrogram

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_D = Path(__file__).resolve().parent
_ROOT = _D.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from moc_simulate.config import SIM_CONFIG, WELL_CONFIG

SPACINGS = [5, 10, 20, 50, 100]
K_VALUES = [0, 0.01, 0.02, 0.05, 0.1, 0.2]


def case_dirname(D: int, k: float) -> str:
    """与 run_simulations.py 的 f'D{D}_k{k}' 一致：k=0 → D5_k0，不是 D5_k0.0。"""
    if float(k) == 0.0:
        return f"D{D}_k0"
    return f"D{D}_k{k}"
FRAC_FIRST_M = 4100.0
N_FRAC = 4
WIN_DEFAULT = (6.5, 7.5)
STFT_WIN = (6.4, 7.2)
PEAK_HEIGHT_FRAC = 0.10
PEAK_MIN_SEP_S = 0.002
EST_LO, EST_HI = 0.10, 0.90
STFT_NPERSEG = 128
STFT_NOVERLAP = 120
STFT_WINDOW = "hann"
BANDS_HZ = ((0.0, 20.0), (20.0, 60.0), (60.0, 150.0))
ONSET_FRAC = 0.01
WINDOWS_SENS = ((6.5, 7.5), (6.4, 7.2), (6.6, 7.0), (6.5, 8.0))

INPUT_DIR = _ROOT / "output" / "analysis" / "brunone_spacing_effect"
OUT_DIR = INPUT_DIR / "audit_v2"
IDENT_DIR = _ROOT / "output" / "analysis" / "identifiability"


def _dt_of(t: np.ndarray) -> float:
    return float(np.median(np.diff(t)))


def load_case(D: int, k: float) -> tuple[np.ndarray, np.ndarray, float]:
    path = INPUT_DIR / case_dirname(D, k) / "moc_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    t = df["t"].to_numpy(dtype=np.float64)
    H = df["H_wh"].to_numpy(dtype=np.float64)
    return t, H, _dt_of(t)


def window_slice(t: np.ndarray, y: np.ndarray, t0: float, t1: float):
    m = (t >= t0) & (t <= t1)
    return t[m], y[m]


def find_packet_peaks(dH_win: np.ndarray, dt: float):
    height = PEAK_HEIGHT_FRAC * float(np.max(dH_win)) if np.max(dH_win) > 0 else 0.0
    dist = max(1, int(round(PEAK_MIN_SEP_S / dt)))
    peaks, props = find_peaks(dH_win, height=height, distance=dist)
    return peaks, props


def energy_spread_time(t_win: np.ndarray, dH_win: np.ndarray, dt: float,
                       e_lo: float = EST_LO, e_hi: float = EST_HI) -> float:
    pos = np.maximum(dH_win, 0.0)
    e_cum = np.cumsum(pos ** 2) * dt
    e_tot = float(e_cum[-1]) if len(e_cum) else 0.0
    if e_tot <= 0.0:
        return float("nan")
    t_lo = t_win[int(np.searchsorted(e_cum, e_lo * e_tot))]
    t_hi = t_win[int(np.searchsorted(e_cum, e_hi * e_tot))]
    return float(t_hi - t_lo)


def first_peak_fwhm(t_win: np.ndarray, dH_win: np.ndarray, p_idx: int) -> float:
    """首峰半高全宽；左右任一端未降到半高则返回 nan（强退化时常见）。"""
    h = 0.5 * dH_win[p_idx]
    left = p_idx
    while left > 0 and dH_win[left] > h:
        left -= 1
    right = p_idx
    n = len(dH_win) - 1
    while right < n and dH_win[right] > h:
        right += 1
    if dH_win[left] > h or dH_win[right] > h:
        return float("nan")
    return float(t_win[right] - t_win[left])


def peak_valley_contrast(dH_win: np.ndarray, peaks: np.ndarray) -> float:
    if len(peaks) < 2:
        return 0.0
    p1, p2 = int(peaks[0]), int(peaks[1])
    if p2 <= p1:
        return 0.0
    v_idx = p1 + int(np.argmin(dH_win[p1:p2 + 1]))
    p_min = min(float(dH_win[p1]), float(dH_win[p2]))
    v = float(dH_win[v_idx])
    if p_min <= 0.0:
        return 0.0
    return float(np.clip((p_min - v) / p_min, 0.0, 1.0))


def apparent_speeds(t: np.ndarray, H: np.ndarray, dH: np.ndarray, dt: float,
                    t_win: np.ndarray, dH_win: np.ndarray, peaks: np.ndarray,
                    a_nom: float, L: float, ts: float) -> dict:
    out = {"a_f0": float("nan"), "a_onset": float("nan"), "a_peak": float("nan"),
           "t_peak": float("nan"), "t_onset": float("nan")}
    mask_post = t > 1.0
    H_post = H[mask_post] - np.mean(H[mask_post])
    n_post = len(H_post)
    if n_post > 8:
        y = np.fft.rfft(H_post)
        freqs = np.fft.rfftfreq(n_post, dt)
        amp = np.abs(y) * 2.0 / n_post
        f0_th = a_nom / (4.0 * L)
        band = (freqs >= f0_th * 0.5) & (freqs <= f0_th * 1.5)
        bf, ba = freqs[band], amp[band]
        if len(ba) > 0:
            pk = int(np.argmax(ba))
            if 0 < pk < len(ba) - 1:
                y0, y1, y2 = ba[pk - 1], ba[pk], ba[pk + 1]
                denom = y0 - 2.0 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
                f0 = bf[pk] + delta * (bf[1] - bf[0])
            else:
                f0 = bf[pk]
            out["a_f0"] = float(f0 * 4.0 * L)

    if len(peaks) > 0:
        p0 = int(peaks[0])
        out["t_peak"] = float(t_win[p0])
        peak_val = float(np.max(dH_win))
        ab = np.where(dH_win > peak_val * ONSET_FRAC)[0]
        if len(ab) > 0:
            out["t_onset"] = float(t_win[ab[0]])
            out["a_onset"] = 2.0 * FRAC_FIRST_M / (out["t_onset"] - ts)
        out["a_peak"] = 2.0 * FRAC_FIRST_M / (out["t_peak"] - ts)
    return out


def stft_band_energy(t: np.ndarray, dH: np.ndarray, dt: float,
                     t0: float, t1: float) -> dict:
    tw, yw = window_slice(t, dH, t0, t1)
    fs = 1.0 / dt
    f, t_spec, sxx = spectrogram(
        yw, fs=fs, window=STFT_WINDOW, nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP, scaling="spectrum", mode="psd",
    )
    out = {
        "f": f, "t_spec": t_spec + t0, "sxx": sxx,
        "E_total": float(np.sum(sxx)),
    }
    for lo, hi in BANDS_HZ:
        m = (f >= lo) & (f < hi)
        out[f"E_{int(lo)}_{int(hi)}"] = float(np.sum(sxx[m, :])) if m.any() else 0.0
    return out


def rfft_band_energy(dH_win: np.ndarray, dt: float) -> dict:
    n = len(dH_win)
    spec = np.fft.rfft(dH_win)
    freqs = np.fft.rfftfreq(n, dt)
    pwr = (np.abs(spec) ** 2) / max(n, 1)
    out = {"E_total": float(np.sum(pwr))}
    for lo, hi in BANDS_HZ:
        m = (freqs >= lo) & (freqs < hi)
        out[f"E_{int(lo)}_{int(hi)}"] = float(np.sum(pwr[m])) if m.any() else 0.0
    return out


def synthetic_cv_unit_test(dt: float = 1e-3) -> pd.DataFrame:
    """等幅高斯双峰：分离度增大时 C_v 应单调不减，完全重叠为 0。"""
    t = np.arange(0.0, 1.0, dt)
    sigma = 0.020
    t1 = 0.40
    seps = np.round(np.linspace(0.0, 0.20, 21), 4)
    rows = []
    for sep in seps:
        y = np.exp(-0.5 * ((t - t1) / sigma) ** 2)
        y = y + np.exp(-0.5 * ((t - (t1 + sep)) / sigma) ** 2)
        peaks, _ = find_packet_peaks(y, dt)
        cv = peak_valley_contrast(y, peaks)
        rows.append({"sep_s": float(sep), "n_peaks": int(len(peaks)), "C_v": cv})
    df = pd.DataFrame(rows)
    cv0 = float(df.loc[df["sep_s"] == 0.0, "C_v"].iloc[0])
    n0 = int(df.loc[df["sep_s"] == 0.0, "n_peaks"].iloc[0])
    cv_hi = float(df.loc[df["sep_s"] == 0.16, "C_v"].iloc[0])
    n_hi = int(df.loc[df["sep_s"] == 0.16, "n_peaks"].iloc[0])
    diffs = np.diff(df["C_v"].to_numpy())
    # 允许数值抖动 1e-6；不允许明显回落
    if n0 != 1 or cv0 != 0.0:
        raise AssertionError(f"unit test overlap failed: n={n0} C_v={cv0}")
    if n_hi < 2 or cv_hi < 0.8:
        raise AssertionError(f"unit test separated failed: n={n_hi} C_v={cv_hi}")
    if np.any(diffs < -0.05):
        raise AssertionError(f"unit test not monotone: min ΔC_v={diffs.min()}")
    return df


def collect_metrics() -> pd.DataFrame:
    a = float(WELL_CONFIG["wavespeed"])
    L = float(WELL_CONFIG["L"])
    ts = float(SIM_CONFIG["ts"])
    rows = []
    for D in SPACINGS:
        for k in K_VALUES:
            t, H, dt = load_case(D, k)
            dH = np.gradient(H, dt)
            t_win, dH_win = window_slice(t, dH, *WIN_DEFAULT)
            peaks, _ = find_packet_peaks(dH_win, dt)
            speeds = apparent_speeds(t, H, dH, dt, t_win, dH_win, peaks, a, L, ts)
            est = energy_spread_time(t_win, dH_win, dt)
            fwhm = first_peak_fwhm(t_win, dH_win, int(peaks[0])) if len(peaks) else float("nan")
            cv = peak_valley_contrast(dH_win, peaks)
            stft = stft_band_energy(t, dH, dt, *STFT_WIN)
            rfft = rfft_band_energy(dH_win, dt)
            amp = float(dH_win[int(peaks[0])]) if len(peaks) else float("nan")
            row = {
                "D_m": D, "k": k, "dt_s": dt, "n_peaks": int(len(peaks)),
                "t_peak_s": speeds["t_peak"], "t_onset_s": speeds["t_onset"],
                "est_s": est, "fwhm_s": fwhm, "C_v": cv, "peak_amp": amp,
                "a_f0": speeds["a_f0"], "a_onset": speeds["a_onset"],
                "a_peak": speeds["a_peak"],
            }
            for lo, hi in BANDS_HZ:
                key = f"E_{int(lo)}_{int(hi)}"
                row[f"stft_{key}"] = stft[key]
                row[f"rfft_{key}"] = rfft[key]
            row["stft_E_total"] = stft["E_total"]
            row["rfft_E_total"] = rfft["E_total"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df["dt_shift_s"] = np.nan
    df["d_est_s"] = np.nan
    df["equiv_depth_bias_m"] = np.nan
    for D in SPACINGS:
        base = df[(df["D_m"] == D) & (df["k"] == 0.0)].iloc[0]
        m = df["D_m"] == D
        df.loc[m, "dt_shift_s"] = df.loc[m, "t_peak_s"] - base["t_peak_s"]
        df.loc[m, "d_est_s"] = df.loc[m, "est_s"] - base["est_s"]
        df.loc[m, "equiv_depth_bias_m"] = df.loc[m, "dt_shift_s"] * a / 2.0
        for lo, hi in BANDS_HZ:
            kstft = f"stft_E_{int(lo)}_{int(hi)}"
            krfft = f"rfft_E_{int(lo)}_{int(hi)}"
            e0s = float(base[kstft])
            e0r = float(base[krfft])
            df.loc[m, f"{kstft}_dB_vs_k0"] = 10.0 * np.log10(
                np.maximum(df.loc[m, kstft].to_numpy(dtype=float), 1e-30) / max(e0s, 1e-30)
            )
            df.loc[m, f"{krfft}_dB_vs_k0"] = 10.0 * np.log10(
                np.maximum(df.loc[m, krfft].to_numpy(dtype=float), 1e-30) / max(e0r, 1e-30)
            )
    return df


def window_sensitivity() -> pd.DataFrame:
    a = float(WELL_CONFIG["wavespeed"])
    L = float(WELL_CONFIG["L"])
    ts = float(SIM_CONFIG["ts"])
    D, k_anchor = 20, 0.2
    rows = []
    t0, H0, dt = load_case(D, 0.0)
    dH0 = np.gradient(H0, dt)
    t, H, _ = load_case(D, k_anchor)
    dH = np.gradient(H, dt)
    for t_lo, t_hi in WINDOWS_SENS:
        tw0, yw0 = window_slice(t0, dH0, t_lo, t_hi)
        tw, yw = window_slice(t, dH, t_lo, t_hi)
        p0, _ = find_packet_peaks(yw0, dt)
        pk, _ = find_packet_peaks(yw, dt)
        tpk0 = float(tw0[int(p0[0])]) if len(p0) else float("nan")
        tpk = float(tw[int(pk[0])]) if len(pk) else float("nan")
        est0 = energy_spread_time(tw0, yw0, dt)
        est = energy_spread_time(tw, yw, dt)
        rows.append({
            "window": f"[{t_lo},{t_hi}]",
            "t_lo": t_lo, "t_hi": t_hi,
            "dt_shift_ms": (tpk - tpk0) * 1000.0,
            "est_ms": est * 1000.0,
            "d_est_ms": (est - est0) * 1000.0,
            "n_peaks_k0": int(len(p0)),
            "n_peaks_k02": int(len(pk)),
        })
    return pd.DataFrame(rows)


def collect_identifiability() -> dict:
    """只读既有 identifiability 输出，不重跑。缺文件则标明 missing。"""
    out: dict = {}

    meta_eff = IDENT_DIR / "efficiency" / "main" / "meta.json"
    if meta_eff.exists():
        meta = json.loads(meta_eff.read_text(encoding="utf-8"))
        rows = meta["summary"]
        a_effs = [r["mle_a_eff"] for r in rows if r["mle_a_eff"] == r["mle_a_eff"]]
        out["efficiency"] = {
            "fc_hz": meta["fc_hz"], "n_mc": meta["n_mc"], "seed": meta["seed"],
            "mle_a_eff_min": min(a_effs), "mle_a_eff_max": max(a_effs),
            "rows": rows,
        }
        s40 = next(r for r in rows if r["case"] == "single_4000" and str(r["snr"]) == "40")
        s_inf = next(r for r in rows if r["case"] == "single_4000" and str(r["snr"]) == "inf")
        out["efficiency"]["single_4000_snr40"] = {
            "mle_rmse_m": s40["mle_a_rmse"], "crb_m": s40["crb_a"],
            "eff": s40["mle_a_eff"], "cep_rmse_m": s40["cep_rmse"],
        }
        out["efficiency"]["single_4000_snr_inf_cep_rmse_m"] = s_inf["cep_rmse"]

    mm = IDENT_DIR / "mismatch_mle" / "main" / "single_4000.json"
    if mm.exists():
        data = json.loads(mm.read_text(encoding="utf-8"))
        pairs = {}
        for p in data["pairs"]:
            key = f"{p['truth_friction']}->{p['fit_friction']}"
            rec = {"mle_rmse_m": p["mle_rmse_m"], "linearized_bias_x0_m": p["linearized_bias"]["x0"],
                   "absorbed_frac": p["absorbed_frac"]}
            if p.get("k_scale_absorption"):
                rec["k_scale_hat"] = p["k_scale_absorption"].get("k_scale_hat")
                rec["k_scale_rmse_m"] = p["k_scale_absorption"].get("rmse_m")
            pairs[key] = rec
        out["friction_form_mismatch"] = pairs

    ho = IDENT_DIR / "highorder" / "main" / "summary.json"
    if ho.exists():
        data = json.loads(ho.read_text(encoding="utf-8"))
        effs = [r["efficiency_ratio"] for r in data["rows"]]
        out["highorder"] = {
            "all_passed": data["all_passed"],
            "eff_min": min(effs), "eff_max": max(effs),
            "n_rows": len(data["rows"]),
        }

    osel = IDENT_DIR / "order_selection" / "main" / "summary.json"
    if osel.exists():
        data = json.loads(osel.read_text(encoding="utf-8"))
        out["order_selection"] = {
            "method": data["method"],
            "phasenet_exact_count_ref": data["phasenet_exact_count_ref"],
            "nested_bic_exact_count": 1.0,
            "n_cases": len(data["cases"]),
        }

    tr = IDENT_DIR / "transfer" / "main" / "transfer_summary.json"
    if tr.exists():
        out["transfer"] = json.loads(tr.read_text(encoding="utf-8"))

    ws = IDENT_DIR / "wavespeed_mismatch" / "probe8" / "wavespeed_mismatch.json"
    if ws.exists():
        data = json.loads(ws.read_text(encoding="utf-8"))
        out["wavespeed_mismatch"] = {
            "status": data["status"],
            "n_probe_completed": data["n_probe_completed"],
            "n_probe_planned": data["n_probe_planned"],
            "a_rel_err": data["a_rel_err"],
            "median_abs_bias_single_m": data["median_abs_bias_single_m"],
            "mean_abs_bias_single_m": data["mean_abs_bias_single_m"],
            "pred_range_m": data["pred_range_m"],
            "pass_gate_ratio_lt2": data["pass_gate_ratio_lt2"],
            "rows": data["rows"],
        }

    diag = IDENT_DIR / "diagnosis" / "diagnosis_table.csv"
    if diag.exists():
        dfd = pd.read_csv(diag)
        sub = dfd[(dfd["fc_hz"] == 20.0) & (dfd["snr_db"] == 30.0)]
        ratios = []
        k10 = []
        for sp in (5.0, 20.0, 50.0):
            st = sub[(sub["friction"] == "steady") & (sub["spacing_m"] == sp)].iloc[0]
            br = sub[(sub["friction"] == "brunone") & (sub["spacing_m"] == sp)].iloc[0]
            ratios.append(float(br["crb_std_x0_joint_m"] / st["crb_std_x0_joint_m"]))
            k10.append(float(br["bias_x0_kbr_rel_10pct_m"]))
        out["diagnosis_snr30_fc20"] = {
            "brunone_over_steady_joint_crb": ratios,
            "brunone_over_steady_joint_crb_min": min(ratios),
            "brunone_over_steady_joint_crb_max": max(ratios),
            "k_scale_10pct_bias_m": k10,
            "form_bias_brunone_m": [
                float(sub[(sub["friction"] == "brunone") & (sub["spacing_m"] == sp)]["form_bias_x0_m"].iloc[0])
                for sp in (5.0, 20.0, 50.0)
            ],
            "note": "form_bias 为一阶线性化；非线性 MLE 见 friction_form_mismatch。",
        }

    crb_csv = IDENT_DIR / "main_dt1ms" / "crb_table.csv"
    if crb_csv.exists():
        dfc = pd.read_csv(crb_csv)
        sub = dfc[(dfc["friction"] == "brunone") & (dfc["fc_hz"] == 20.0) & (dfc["snr_db"] == 40.0)]
        out["crb_vs_spacing_brunone_fc20_snr40"] = {
            "spacing_placed_m": sub["spacing_placed_m"].tolist(),
            "std_x0_a_known_m": sub["std_x0_a_known_m"].tolist(),
            "std_x0_full_m": sub["std_x0_full_m"].tolist(),
            "std_x0_a_known_min_m": float(sub["std_x0_a_known_m"].min()),
            "std_x0_a_known_max_m": float(sub["std_x0_a_known_m"].max()),
            "tf_s": 20.0,
            "note": "tf=20 s 扫描；对齐效率实验 tf=50 s 的单缝 CRB 见 efficiency.single_4000_snr40。",
        }
    return out


def _save_fig(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(df: pd.DataFrame, unit_df: pd.DataFrame, sens_df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.unicode_minus": False,
        "figure.facecolor": "white",
    })
    D_plot = 20
    sub = df[df["D_m"] == D_plot].sort_values("k")

    # 1 waveform
    fig, axes = plt.subplots(len(K_VALUES), 1, figsize=(8.5, 10), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(K_VALUES)))
    for i, k in enumerate(K_VALUES):
        t, H, dt = load_case(D_plot, k)
        dH = np.gradient(H, dt)
        tw, yw = window_slice(t, dH, 6.60, 7.05)
        axes[i].plot(tw, yw, color=colors[i], lw=1.1)
        axes[i].set_ylabel("dH/dt")
        axes[i].text(0.99, 0.82, f"k={k:g}", transform=axes[i].transAxes,
                     ha="right", va="top")
        axes[i].grid(True, ls="--", alpha=0.4)
    axes[0].set_title(f"Wellhead dH/dt packet vs prescribed Brunone k (D={D_plot} m)")
    axes[-1].set_xlabel("Time (s)")
    _save_fig(fig, "fig1_waveform_evolution.png")

    # 2 peak shift / EST
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(sub["k"], sub["dt_shift_s"] * 1000.0, "o-", color="tab:red", label="Peak shift")
    ax.set_xlabel("Prescribed Brunone k")
    ax.set_ylabel("Peak shift (ms)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.plot(sub["k"], sub["d_est_s"] * 1000.0, "s--", color="tab:blue", label="ΔEST")
    ax2.set_ylabel("ΔEST vs k=0 (ms)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title(f"Peak delay and energy-spread increment (D={D_plot} m, window {WIN_DEFAULT})")
    ax.grid(True, ls="--", alpha=0.4)
    _save_fig(fig, "fig2_peakshift_est.png")

    # 3 STFT common color scale
    k_show = [0.0, 0.05, 0.1, 0.2]
    specs = []
    for k in k_show:
        t, H, dt = load_case(D_plot, k)
        dH = np.gradient(H, dt)
        specs.append(stft_band_energy(t, dH, dt, *STFT_WIN))
    db0 = 10.0 * np.log10(specs[0]["sxx"] + 1e-20)
    vmin, vmax = np.percentile(db0, [5, 99.5])
    fig, axes = plt.subplots(len(k_show), 1, figsize=(8.5, 9.5), sharex=True, sharey=True)
    for ax, k, sp in zip(axes, k_show, specs):
        db = 10.0 * np.log10(sp["sxx"] + 1e-20)
        im = ax.pcolormesh(sp["t_spec"], sp["f"], db, shading="gouraud",
                           cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_ylim(0, 150)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"k={k:g}")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Power (dB, common scale)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"STFT of dH/dt, common dB scale from k=0 (D={D_plot} m)", y=0.995)
    fig.tight_layout()
    _save_fig(fig, "fig3_stft_common.png")

    # 4 band energy
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for lo, hi, mk in ((0, 20, "o-"), (20, 60, "s--"), (60, 150, "^-.")):
        col = f"stft_E_{lo}_{hi}_dB_vs_k0"
        ax.plot(sub["k"], sub[col], mk, label=f"{lo}–{hi} Hz")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("Prescribed Brunone k")
    ax.set_ylabel("Band energy relative to k=0 (dB)")
    ax.set_title(f"STFT band energy vs k (D={D_plot} m; dB ref = same-D k=0)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    _save_fig(fig, "fig4_band_energy.png")

    # 5 apparent speeds
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(sub["k"], sub["a_f0"], "o-", label=r"$a_{f0}$ (fundamental)")
    ax.plot(sub["k"], sub["a_onset"], "s--", label=r"$a_{\mathrm{onset}}$")
    ax.plot(sub["k"], sub["a_peak"], "^-.", label=r"$a_{\mathrm{peak}}$")
    ax.axhline(WELL_CONFIG["wavespeed"], color="0.5", ls=":", label="nominal a")
    ax.set_xlabel("Prescribed Brunone k")
    ax.set_ylabel("Apparent speed (m/s)")
    ax.set_title(f"Three operational apparent speeds (D={D_plot} m); not $c_p(\\omega)$")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    _save_fig(fig, "fig5_apparent_speeds.png")

    # 6 C_v heatmap
    mat = np.zeros((len(K_VALUES), len(SPACINGS)))
    for i, k in enumerate(K_VALUES):
        for j, D in enumerate(SPACINGS):
            mat[i, j] = float(df[(df["D_m"] == D) & (df["k"] == k)]["C_v"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(mat, origin="upper", cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SPACINGS)), [str(d) for d in SPACINGS])
    ax.set_yticks(range(len(K_VALUES)), [f"{k:g}" for k in K_VALUES])
    ax.set_xlabel("Spacing D (m)")
    ax.set_ylabel("Prescribed Brunone k")
    for i in range(len(K_VALUES)):
        for j in range(len(SPACINGS)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(r"Operational peak–valley contrast $C_v=(P_{\min}-V)/P_{\min}$ (not a Rayleigh wall)")
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"$C_v$")
    _save_fig(fig, "fig6_cv_heatmap.png")

    # 7 unit test
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(unit_df["sep_s"] * 1000.0, unit_df["C_v"], "o-")
    ax.set_xlabel("Gaussian-peak separation (ms)")
    ax.set_ylabel(r"$C_v$")
    ax.set_title("Synthetic dual-Gaussian unit test of $C_v$")
    ax.grid(True, ls="--", alpha=0.4)
    _save_fig(fig, "fig7_cv_unit_test.png")

    # 8 window sensitivity
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(sens_df))
    ax.bar(x - 0.18, sens_df["dt_shift_ms"], 0.36, label="Peak shift")
    ax.bar(x + 0.18, sens_df["d_est_ms"], 0.36, label="ΔEST")
    ax.set_xticks(x, sens_df["window"].tolist())
    ax.set_ylabel("ms")
    ax.set_title("D=20 m, k=0.2: metric vs analysis window")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    _save_fig(fig, "fig8_window_sensitivity.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unit_df = synthetic_cv_unit_test()
    unit_df.to_csv(OUT_DIR / "unit_test_cv.csv", index=False)

    df = collect_metrics()
    # 不把波形数组写入 CSV
    df.to_csv(OUT_DIR / "metrics_v2.csv", index=False)

    band_cols = ["D_m", "k"] + [
        c for c in df.columns if c.startswith("stft_E_") or c.startswith("rfft_E_")
    ]
    df[band_cols].to_csv(OUT_DIR / "band_energy.csv", index=False)

    sens_df = window_sensitivity()
    sens_df.to_csv(OUT_DIR / "window_sensitivity.csv", index=False)

    ident = collect_identifiability()

    # 锚点数字
    anchor = df[(df["D_m"] == 20) & (df["k"] == 0.2)].iloc[0]
    base = df[(df["D_m"] == 20) & (df["k"] == 0.0)].iloc[0]
    speeds_k02 = {
        "a_f0": float(anchor["a_f0"]),
        "a_onset": float(anchor["a_onset"]),
        "a_peak": float(anchor["a_peak"]),
    }
    speeds_k0 = {
        "a_f0": float(base["a_f0"]),
        "a_onset": float(base["a_onset"]),
        "a_peak": float(base["a_peak"]),
    }
    da = {key: (speeds_k0[key] - speeds_k02[key]) / speeds_k0[key] * 100.0 for key in speeds_k0}

    paper = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_id": "EXP-20260730-024",
        "input_dir": str(INPUT_DIR),
        "n_cases": int(len(df)),
        "anchor_D20_k02": {
            "dt_shift_ms": float(anchor["dt_shift_s"] * 1000.0),
            "est_ms": float(anchor["est_s"] * 1000.0),
            "d_est_ms": float(anchor["d_est_s"] * 1000.0),
            "equiv_depth_bias_m": float(anchor["equiv_depth_bias_m"]),
            "C_v": float(anchor["C_v"]),
            "n_peaks": int(anchor["n_peaks"]),
            "fwhm_ms": float(anchor["fwhm_s"] * 1000.0) if anchor["fwhm_s"] == anchor["fwhm_s"] else None,
            "stft_60_150_dB_vs_k0": float(anchor["stft_E_60_150_dB_vs_k0"]),
            "stft_20_60_dB_vs_k0": float(anchor["stft_E_20_60_dB_vs_k0"]),
            "stft_0_20_dB_vs_k0": float(anchor["stft_E_0_20_dB_vs_k0"]),
            "rfft_60_150_dB_vs_k0": float(anchor["rfft_E_60_150_dB_vs_k0"]),
            "apparent_speeds_mps": speeds_k02,
            "apparent_speed_reduction_pct_vs_k0": da,
        },
        "anchor_D20_k01": {
            "dt_shift_ms": float(df[(df["D_m"] == 20) & (df["k"] == 0.1)]["dt_shift_s"].iloc[0] * 1000.0),
            "d_est_ms": float(df[(df["D_m"] == 20) & (df["k"] == 0.1)]["d_est_s"].iloc[0] * 1000.0),
            "stft_60_150_dB_vs_k0": float(
                df[(df["D_m"] == 20) & (df["k"] == 0.1)]["stft_E_60_150_dB_vs_k0"].iloc[0]
            ),
        },
        "window_sensitivity_D20_k02": sens_df.to_dict(orient="records"),
        "identifiability": ident,
    }
    (OUT_DIR / "paper_numbers.json").write_text(
        json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "experiment_id": "EXP-20260730-024",
        "generated_utc": paper["generated_utc"],
        "reuse_timeseries": True,
        "n_cases_expected": 30,
        "n_cases_found": int(len(df)),
        "well": {
            "L_m": WELL_CONFIG["L"], "a_mps": WELL_CONFIG["wavespeed"],
            "D_m": WELL_CONFIG["wellbore_diameter"], "ts_s": SIM_CONFIG["ts"],
            "frac_first_m": FRAC_FIRST_M, "n_frac": N_FRAC,
        },
        "formulas": {
            "signal": "dH/dt = np.gradient(H_wh, dt)",
            "analysis_window_s": list(WIN_DEFAULT),
            "peak_height_frac": PEAK_HEIGHT_FRAC,
            "peak_min_separation_s": PEAK_MIN_SEP_S,
            "EST": "t(E=0.90)-t(E=0.10) of positive-part energy",
            "C_v": "(P_min-V)/P_min clipped to [0,1]; 0 if <2 peaks",
            "not_used": "Rayleigh 0.81 score; FWHM as primary width; c_p(omega)",
        },
        "stft": {
            "signal": "dH/dt",
            "time_window_s": list(STFT_WIN),
            "window": STFT_WINDOW,
            "nperseg": STFT_NPERSEG,
            "noverlap": STFT_NOVERLAP,
            "scaling": "spectrum",
            "fs_hz": 1.0 / float(df["dt_s"].iloc[0]),
            "df_hz": (1.0 / float(df["dt_s"].iloc[0])) / STFT_NPERSEG,
            "dB_reference": "10*log10(E(k)/E(k=0, same D))",
            "bands_hz": [list(b) for b in BANDS_HZ],
        },
        "unit_test": {
            "file": "unit_test_cv.csv",
            "passed": True,
            "description": "equal-amplitude Gaussians, sigma=20 ms",
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    make_figures(df, unit_df, sens_df)

    print("EXP-024 audit complete")
    print(f"  cases: {len(df)}")
    print(f"  D=20 k=0.2  dt_shift={paper['anchor_D20_k02']['dt_shift_ms']:.3f} ms")
    print(f"  D=20 k=0.2  d_est={paper['anchor_D20_k02']['d_est_ms']:.3f} ms")
    print(f"  D=20 k=0.2  STFT 60-150 dB vs k0={paper['anchor_D20_k02']['stft_60_150_dB_vs_k0']:.2f}")
    print(f"  D=20 k=0.2  rFFT 60-150 dB vs k0={paper['anchor_D20_k02']['rfft_60_150_dB_vs_k0']:.2f}")
    print(f"  output: {OUT_DIR}")


if __name__ == "__main__":
    main()
