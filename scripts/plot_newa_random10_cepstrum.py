#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 moc_v2_physical_steady_1k_newa 随机抽取 10 口井，
绘制井口水头波形、1D 实倒谱与 2D 滑窗倒谱图。
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moc_simulate.v2.signal.cepstrum_1d import compute_cepstrum_1d
from moc_simulate.v2.signal.cepstrum_2d import compute_cepstrogram_2d

H5_PATH = os.path.join(
    "data",
    "datasets",
    "moc_v2_physical_steady_1k_newa",
    "moc_v2_physical_steady_1k_x4500_4950.h5",
)
OUT_DIR = os.path.join(
    "data",
    "datasets",
    "moc_v2_physical_steady_1k_newa",
    "random10_wave_cepstrum",
)
SEED = 20260916
N_CASES = 10
TS = 1.0
FS = 1000.0
MAX_DISTANCE = 5000.0
CEPS_XMIN = 4000.0
CEPS_XMAX = 5000.0


def _as_text(raw) -> str:
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _type_short(raw) -> str:
    s = _as_text(raw)
    for tag in ("Type V", "Type IV", "Type III", "Type II", "Type I"):
        if s.startswith(tag):
            return tag
    return s[:12]


def load_cases(h5_path: str, ids: np.ndarray) -> dict:
    with h5py.File(h5_path, "r") as f:
        t = np.asarray(f["waveforms/timestamps"][:], dtype=np.float64)
        heads = np.asarray(f["waveforms/wellhead_head"][ids], dtype=np.float64)
        lab = f["labels"]
        n_frac = np.asarray(lab["n_frac"][ids], dtype=np.int64)
        pos = np.asarray(lab["fracture_positions"][ids], dtype=np.float64)
        alpha = np.asarray(lab["fracture_alpha_ss"][ids], dtype=np.float64)
        a = np.asarray(lab["wavespeed"][ids], dtype=np.float64)
        tc = np.asarray(lab["pump_closure_tc"][ids], dtype=np.float64)
        h0 = np.asarray(lab["initial_head_realized"][ids], dtype=np.float64)
        types = lab["fracture_types"][ids]
    return {
        "t": t,
        "heads": heads,
        "n_frac": n_frac,
        "pos": pos,
        "alpha": alpha,
        "a": a,
        "tc": tc,
        "h0": h0,
        "types": types,
    }


def plot_one_case(idx: int, sample_id: int, data: dict, out_dir: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = data["t"]
    h = data["heads"][idx]
    a = float(data["a"][idx])
    nc = int(data["n_frac"][idx])
    xf = data["pos"][idx, :nc]
    alpha = data["alpha"][idx, :nc]
    tc_ms = float(data["tc"][idx]) * 1000.0
    h0 = float(data["h0"][idx])
    raw_types = data["types"][idx]
    type_tags = [_type_short(raw_types[j]) for j in range(nc)]

    ceps1d = compute_cepstrum_1d(
        t, h, wavespeed=a, fs=FS, ts=TS, window="hamming", max_distance=MAX_DISTANCE
    )
    ceps2d = compute_cepstrogram_2d(
        t,
        h,
        wavespeed=a,
        fs=FS,
        ts=TS,
        window_len_s=30.0,
        hop_len_s=1.0,
        window="kaiser",
        max_distance=MAX_DISTANCE,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.2), constrained_layout=True)
    fig.suptitle(
        rf"Case #{sample_id}  $N_c$={nc}  $a$={a:.1f} m/s  $t_c$={tc_ms:.1f} ms  $H_0^*$={h0:.1f} m",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(t, h, color="#1f4e79", lw=1.05)
    ax.axvline(TS, color="#c0392b", ls="--", lw=0.9, label=r"$t_s=1$ s")
    ax.set_xlim(0.0, float(t[-1]))
    ax.set_xlabel("Time $t$ [s]")
    ax.set_ylabel(r"Wellhead head $H_{\mathrm{wh}}$ [m]")
    ax.set_title("(a) Wellhead pressure waveform", loc="left", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, ls="--", alpha=0.35)

    ax = axes[1]
    d1 = ceps1d["distance"]
    c1 = ceps1d["cepstrum"]
    m1 = (d1 >= CEPS_XMIN) & (d1 <= CEPS_XMAX)
    y1 = c1[m1]
    ax.plot(d1[m1], y1, color="#0e6b4a", lw=1.1)
    ax.axhline(0.0, color="#888888", ls=":", lw=0.7)
    for xj in xf:
        ax.axvline(xj, color="#d9534f", ls="--", lw=0.85, alpha=0.85)
    annot = "   ".join(
        f"{tj} @ {xj:.0f} m, α={aj:.2f}" for xj, aj, tj in zip(xf, alpha, type_tags)
    )
    ax.text(
        0.01,
        0.06,
        annot,
        transform=ax.transAxes,
        fontsize=7.0,
        color="#7b241c",
        va="bottom",
        ha="left",
        clip_on=True,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d0d0d0", alpha=0.88),
    )
    ax.set_xlim(CEPS_XMIN, CEPS_XMAX)
    ax.set_xlabel("Two-way reflection depth $x$ [m]")
    ax.set_ylabel("Real cepstrum")
    ax.set_title(
        r"(b) 1D real cepstrum (Hamming, $x=a\tau/2$, zoom 4000–5000 m)",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(True, ls="--", alpha=0.35)

    ax = axes[2]
    dist = ceps2d["distances"]
    tcen = ceps2d["time_centers"]
    mat = np.abs(ceps2d["cepstrogram"])
    m2 = (dist >= CEPS_XMIN) & (dist <= CEPS_XMAX)
    sub = mat[:, m2]
    vmin = float(np.percentile(sub, 2))
    vmax = float(np.percentile(sub, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    mesh = ax.pcolormesh(
        dist[m2],
        tcen,
        sub,
        shading="auto",
        cmap="rainbow",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    for xj in xf:
        ax.axvline(xj, color="white", ls="--", lw=0.95, alpha=0.9)
    ax.set_xlim(CEPS_XMIN, CEPS_XMAX)
    ax.set_ylim(float(tcen[0]), float(tcen[-1]))
    ax.set_xlabel("Two-way reflection depth $x$ [m]")
    ax.set_ylabel("Window center $t$ [s]")
    ax.set_title(
        r"(c) 2D cepstrogram (Kaiser $\beta$=14, 30 s / 1 s hop)",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.035)
    cbar.set_label("Cepstral intensity")

    out_png = os.path.join(out_dir, f"case_{sample_id:04d}_wave_cepstrum.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    return {
        "sample_id": int(sample_id),
        "n_frac": nc,
        "wavespeed_m_s": a,
        "tc_ms": tc_ms,
        "H0_realized_m": h0,
        "fracture_positions_m": [float(v) for v in xf],
        "fracture_alpha_ss": [float(v) for v in alpha],
        "fracture_types": type_tags,
        "figure": os.path.abspath(out_png),
    }


def plot_overview(ids: np.ndarray, data: dict, out_dir: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = data["t"]
    fig, axes = plt.subplots(5, 2, figsize=(12.5, 13.5), sharex=True, constrained_layout=True)
    for k, ax in enumerate(axes.ravel()):
        i = k
        sid = int(ids[i])
        nc = int(data["n_frac"][i])
        ax.plot(t, data["heads"][i], color="#1f4e79", lw=0.9)
        ax.axvline(TS, color="#c0392b", ls="--", lw=0.7)
        ax.set_title(f"#{sid}  $N_c$={nc}", fontsize=9, loc="left")
        ax.grid(True, ls="--", alpha=0.3)
        if k >= 8:
            ax.set_xlabel("Time $t$ [s]")
        if k % 2 == 0:
            ax.set_ylabel(r"$H_{\mathrm{wh}}$ [m]")
    fig.suptitle("Random 10 cases — wellhead head", fontsize=12, fontweight="bold")
    out_png = os.path.join(out_dir, "overview_wellhead_10.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_total = 1000
    ids = np.sort(rng.choice(n_total, size=N_CASES, replace=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[*] HDF5: {H5_PATH}")
    print(f"[*] seed={SEED}, cases={ids.tolist()}")

    data = load_cases(H5_PATH, ids)
    records = []
    for i, sid in enumerate(ids):
        print(f"[-] plotting case {sid} ({i+1}/{N_CASES})")
        records.append(plot_one_case(i, int(sid), data, OUT_DIR))
    overview = plot_overview(ids, data, OUT_DIR)
    summary = {
        "source_h5": os.path.abspath(H5_PATH),
        "seed": SEED,
        "n_cases": N_CASES,
        "sample_ids": [int(v) for v in ids],
        "cepstrum_1d": {"window": "hamming", "ts_s": TS, "fs_hz": FS, "max_distance_m": MAX_DISTANCE},
        "cepstrum_2d": {
            "window": "kaiser",
            "beta": 14.0,
            "window_len_s": 30.0,
            "hop_len_s": 1.0,
            "ts_s": TS,
            "fs_hz": FS,
            "max_distance_m": MAX_DISTANCE,
        },
        "overview": os.path.abspath(overview),
        "cases": records,
    }
    meta_path = os.path.join(OUT_DIR, "random10_summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[+] overview: {overview}")
    print(f"[+] summary: {meta_path}")


if __name__ == "__main__":
    main()
