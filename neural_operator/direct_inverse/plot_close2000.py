# -*- coding: utf-8 -*-
"""Visualize lhs_dataset_2000 physics samples and close2000 P0-A predictions."""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError("project root not found")
    _d = _parent
ROOT = _d

from neural_operator.direct_inverse.config import DataConfig, DetectorConfig, dataclass_from_dict
from neural_operator.direct_inverse.data import load_manifest, spacing_band, time_to_depth_m
from neural_operator.direct_inverse.evaluate import detect_events, match_events

PALETTE = {
    "obs": "#3d3d3d",
    "target": "#2a78d6",
    "pred": "#e34948",
    "true_mark": "#4a3aa7",
    "det_mark": "#c45c26",
    "grid": "#e8e6df",
    "bands": {
        "singleton": "#2a78d6",
        "5-10": "#e34948",
        "10-15": "#c45c26",
        "15-20": "#008300",
    },
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })


def _load_case_pressure(relative_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(ROOT, relative_path.replace("/", os.sep))
    with np.load(path, allow_pickle=False) as npz:
        t = np.asarray(npz["t"], dtype=float)
        h = np.asarray(npz["H_wh"], dtype=float)
    # Match surrogate resampling used in training.
    t_target = np.linspace(0.0, float(t[-1]), 4096)
    h_target = np.interp(t_target, t, h)
    return t_target, h_target


def plot_dataset_overview(manifest: dict, output_dir: str) -> str:
    rows = [row for row in manifest["cases"] if row["split"] != "challenge"]
    spacings = np.asarray([row["min_spacing_m"] for row in rows if row["n_frac"] > 1], dtype=float)
    n_frac = np.asarray([row["n_frac"] for row in rows], dtype=int)
    bands = [row["spacing_band"] for row in rows]
    band_order = ["singleton", "5-10", "10-15", "15-20"]
    band_counts = [bands.count(name) for name in band_order]

    fig = plt.figure(figsize=(12.5, 4.2))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.hist(spacings, bins=np.linspace(5, 20, 16), color="#2a78d6", edgecolor="white", alpha=0.9)
    ax0.axvline(8.85, color="#e34948", linestyle="--", linewidth=1.5, label="grid bin ≈ 8.85 m")
    ax0.set_xlabel("Minimum fracture spacing (m)")
    ax0.set_ylabel("Count")
    ax0.set_title("A. Multi-frac spacing", loc="left", fontweight="semibold")
    ax0.legend(frameon=False)

    counts = [int(np.sum(n_frac == k)) for k in range(1, 7)]
    ax1.bar(range(1, 7), counts, color="#52514e", edgecolor="white")
    ax1.set_xlabel("Number of fractures")
    ax1.set_ylabel("Count")
    ax1.set_title("B. Fracture count", loc="left", fontweight="semibold")

    colors = [PALETTE["bands"][name] for name in band_order]
    ax2.bar(band_order, band_counts, color=colors, edgecolor="white")
    ax2.set_ylabel("Count")
    ax2.set_title("C. Spacing bands", loc="left", fontweight="semibold")
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle("lhs_dataset_2000 overview (close spacing 5–20 m)", fontsize=13, fontweight="semibold")
    path = os.path.join(output_dir, "01_dataset_overview.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_waveform_gallery(manifest: dict, output_dir: str) -> str:
    data_config = dataclass_from_dict(DataConfig, manifest["data_config"])
    by_band = {name: [] for name in ("singleton", "5-10", "10-15", "15-20")}
    for row in manifest["cases"]:
        if row["split"] == "train" and len(by_band[row["spacing_band"]]) < 1:
            # Prefer representative n_frac for multi bands.
            if row["spacing_band"] == "singleton" or row["n_frac"] >= 3:
                by_band[row["spacing_band"]].append(row)
    # Fill remaining bands if empty.
    for row in manifest["cases"]:
        band = row["spacing_band"]
        if row["split"] == "train" and not by_band[band]:
            by_band[band].append(row)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 10.5), sharex=False)
    fig.subplots_adjust(hspace=0.45)
    for ax, band in zip(axes, ("singleton", "5-10", "10-15", "15-20")):
        row = by_band[band][0]
        t, h = _load_case_pressure(row["relative_path"])
        with np.load(os.path.join(ROOT, row["relative_path"].replace("/", os.sep)), allow_pickle=False) as npz:
            x_f = np.asarray(npz["x_f"], dtype=float)
        ax.plot(t, h, color=PALETTE["obs"], linewidth=1.0)
        ymin, ymax = np.percentile(h, [1, 99])
        pad = 0.08 * (ymax - ymin + 1e-6)
        arrivals = data_config.pump_shut_time_s + 2.0 * x_f / data_config.wavespeed_m_s
        for arrival in arrivals:
            ax.axvline(arrival, color=PALETTE["true_mark"], alpha=0.7, linewidth=1.3)
        t0 = max(0.0, float(np.min(arrivals)) - 0.8)
        t1 = min(float(t[-1]), float(np.max(arrivals)) + 2.5)
        ax.set_xlim(t0, t1)
        ax.set_ylim(ymin - pad, ymax + pad)
        try:
            spacing = float(row["min_spacing_m"])
        except (TypeError, ValueError):
            spacing = float("inf")
        spacing_txt = "inf" if not np.isfinite(spacing) else f"{spacing:.1f} m"
        ax.set_title(
            f"{band}  |  {row['case_id']}  |  n={row['n_frac']}  |  minΔx={spacing_txt}",
            loc="left",
            fontweight="semibold",
            color=PALETTE["bands"][band],
        )
        ax.set_ylabel("H_wh (m)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Wellhead pressure with true fracture arrivals (vertical lines)", fontsize=13, fontweight="semibold")
    path = os.path.join(output_dir, "02_waveform_gallery.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _pick_cases(bundle: dict, data_config: DataConfig, per_band: int = 1) -> list[int]:
    chosen: list[int] = []
    seen = {name: 0 for name in ("singleton", "5-10", "10-15", "15-20")}
    for index in range(len(bundle["case_id"])):
        n = int(bundle["n_frac"][index])
        spacing = float(bundle["min_spacing_m"][index])
        band = spacing_band(n, spacing, data_config.spacing_regime)
        if seen[band] < per_band:
            chosen.append(index)
            seen[band] += 1
        if all(value >= per_band for value in seen.values()):
            break
    return chosen


def plot_prediction_cases(
    artifact_path: str,
    manifest: dict,
    detector: DetectorConfig,
    threshold: float,
    output_dir: str,
) -> str:
    bundle = dict(np.load(artifact_path, allow_pickle=False))
    data_config = dataclass_from_dict(DataConfig, manifest["data_config"])
    case_map = {row["case_id"]: row for row in manifest["cases"]}
    indices = _pick_cases(bundle, data_config, per_band=1)

    fig, axes = plt.subplots(len(indices), 2, figsize=(13.5, 2.7 * len(indices)), sharex=False)
    if len(indices) == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(hspace=0.55, wspace=0.22)

    for row_i, index in enumerate(indices):
        case_id = str(bundle["case_id"][index])
        n = int(bundle["n_frac"][index])
        x_f = np.asarray(bundle["x_f"][index, :n], dtype=float)
        spacing = float(bundle["min_spacing_m"][index])
        band = spacing_band(n, spacing, data_config.spacing_regime)
        t = np.asarray(bundle["time_axis"][index], dtype=float).reshape(-1)
        target = np.asarray(bundle["event_target"][index], dtype=float).reshape(-1)
        prob = np.asarray(bundle["probability"][index], dtype=float).reshape(-1)
        mask = np.asarray(bundle["valid_time_mask"][index], dtype=bool).reshape(-1)
        rel = case_map[case_id]["relative_path"]
        _, h = _load_case_pressure(rel)

        detected = detect_events(prob, mask, t, threshold, data_config, detector)
        matched = match_events(detected, x_f, detector.physical_tolerance_m)

        ax0, ax1 = axes[row_i]
        ax0.plot(t, h, color=PALETTE["obs"], linewidth=0.9)
        for depth in x_f:
            arrival = data_config.pump_shut_time_s + 2.0 * depth / data_config.wavespeed_m_s
            ax0.axvline(arrival, color=PALETTE["true_mark"], alpha=0.5, linewidth=1.0)
        ax0.set_xlim(4.5, 8.0)
        ax0.set_ylabel("H_wh")
        spacing_txt = "inf" if not np.isfinite(spacing) else f"{spacing:.1f}m"
        ax0.set_title(
            f"{case_id} | {band} | n={n} | Δx={spacing_txt}",
            loc="left",
            fontweight="semibold",
            color=PALETTE["bands"][band],
        )

        ax1.plot(t, target, color=PALETTE["target"], linewidth=1.6, label="target")
        ax1.plot(t, prob, color=PALETTE["pred"], linewidth=1.4, alpha=0.95, label="prediction")
        ax1.axhline(threshold, color="#888888", linestyle=":", linewidth=1.0, label=f"thr={threshold:.2f}")
        for depth in x_f:
            arrival = data_config.pump_shut_time_s + 2.0 * depth / data_config.wavespeed_m_s
            ax1.axvline(arrival, color=PALETTE["true_mark"], alpha=0.45, linewidth=1.0)
        for event in detected:
            ax1.axvline(event["time_s"], color=PALETTE["det_mark"], linestyle="--", alpha=0.8, linewidth=1.0)
        ax1.set_xlim(4.5, 8.0)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylabel("prob")
        ax1.set_title(
            f"TP/FP/FN={matched['tp']}/{matched['fp']}/{matched['fn']}  pred_n={len(detected)}",
            loc="left",
            fontsize=10,
        )
        if row_i == 0:
            ax1.legend(loc="upper right", frameon=False, ncol=3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(
        "P0-A close2000 subset512 — observation vs event heatmap",
        fontsize=13,
        fontweight="semibold",
        y=0.995,
    )
    path = os.path.join(output_dir, "03_prediction_cases.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metrics_panels(oracle_path: str, gate_path: str, output_dir: str) -> str:
    oracle = json.load(open(oracle_path, encoding="utf-8"))
    gate = json.load(open(gate_path, encoding="utf-8"))
    subset = gate["metrics"]
    band_order = ["singleton", "5-10", "10-15", "15-20"]

    def band_metric(payload: dict, key: str) -> list[float]:
        strata = payload["strata"]["by_spacing_band"]
        values = []
        for band in band_order:
            if band not in strata:
                values.append(0.0)
            elif key == "f1":
                values.append(float(strata[band]["physical"]["f1"]))
            elif key == "exact":
                values.append(float(strata[band]["exact_count_accuracy"]))
            else:
                values.append(0.0)
        return values

    x = np.arange(len(band_order))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for ax, key, title in (
        (axes[0], "f1", "Physical F1 by spacing band"),
        (axes[1], "exact", "Exact-count accuracy by spacing band"),
    ):
        o = band_metric(oracle, key)
        s = band_metric(subset, key)
        ax.bar(x - width / 2, o, width, label="Oracle", color="#2a78d6", edgecolor="white")
        ax.bar(x + width / 2, s, width, label="Subset512", color="#e34948", edgecolor="white")
        ax.set_xticks(x, band_order)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.legend(frameon=False)
        ax.set_ylabel(key.upper() if key == "f1" else "Accuracy")

    overall = (
        f"Oracle F1={oracle['physical']['f1']:.3f}  |  "
        f"Subset512 F1={subset['physical']['f1']:.3f}  |  "
        f"thr={subset['threshold']}"
    )
    fig.suptitle(f"Close-spacing event protocol limits\n{overall}", fontsize=12, fontweight="semibold")
    path = os.path.join(output_dir, "04_metrics_by_band.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    _setup_style()
    manifest_path = os.path.join(
        ROOT, "output", "direct_inverse", "manifests_close2000",
        "direct-inverse-data-v2-close2000-seed42.json",
    )
    artifact_path = os.path.join(
        ROOT, "output", "direct_inverse", "runs_close2000",
        "20260730-143933-151520-p0a-subset512-seed42-p0a-close-subset512",
        "evaluations", "validation_predictions.npz",
    )
    oracle_path = os.path.join(
        ROOT, "output", "direct_inverse", "manifests_close2000", "oracle-val128.json",
    )
    gate_path = os.path.join(
        ROOT, "output", "direct_inverse", "runs_close2000",
        "20260730-143933-151520-p0a-subset512-seed42-p0a-close-subset512",
        "gate_result.json",
    )
    output_dir = os.path.join(ROOT, "output", "lhs_dataset_2000", "figures_p0a")
    os.makedirs(output_dir, exist_ok=True)

    manifest = load_manifest(manifest_path)
    detector = dataclass_from_dict(DetectorConfig, manifest["detector_config"])
    threshold = float(json.load(open(gate_path, encoding="utf-8"))["metrics"]["threshold"])

    paths = [
        plot_dataset_overview(manifest, output_dir),
        plot_waveform_gallery(manifest, output_dir),
        plot_prediction_cases(artifact_path, manifest, detector, threshold, output_dir),
        plot_metrics_panels(oracle_path, gate_path, output_dir),
    ]
    print("Wrote figures:")
    for path in paths:
        print(" ", path)


if __name__ == "__main__":
    main()
