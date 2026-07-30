# -*- coding: utf-8 -*-
"""Plot schema-v2 DCCDM prediction artifacts using their saved time axes."""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


PALETTE = {
    "true": "#2a78d6",
    "raw": "#e34948",
    "post": "#008300",
    "observation": "#52514e",
    "marker": "#4a3aa7",
    "grid": "#e1e0d9",
}


def _series(array: np.ndarray, index: int) -> np.ndarray:
    value = np.asarray(array[index])
    return value.reshape(-1)


def generate_publication_figures(
    artifact_path: str,
    output_dir: str | None = None,
    peak_height: float = 1.0,
    peak_prominence: float = 0.5,
) -> list[str]:
    data = np.load(artifact_path, allow_pickle=False)
    required = {
        "true_map", "raw_prediction", "postprocessed_prediction",
        "clean_observation", "time_axis", "x_f", "n_frac", "case_id",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")

    output_dir = os.path.abspath(output_dir or os.path.join(os.path.dirname(artifact_path), "figures"))
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    paths: list[str] = []
    for index in range(len(data["true_map"])):
        t = _series(data["time_axis"], index)
        y_true = _series(data["true_map"], index)
        y_raw = _series(data["raw_prediction"], index)
        y_post = _series(data["postprocessed_prediction"], index)
        y_obs = _series(data["clean_observation"], index)
        case_id = str(data["case_id"][index])
        count = int(data["n_frac"][index])
        true_arrivals = 1.0 + 2.0 * np.asarray(data["x_f"][index, :count], dtype=float) / 1450.0
        detected, _ = find_peaks(
            np.clip(y_post, 0.0, None),
            height=peak_height,
            prominence=peak_prominence,
        )

        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        fig.subplots_adjust(hspace=0.32)
        fig.suptitle(f"DCCDM fracture-map diagnostic — {case_id}", fontsize=14, fontweight="semibold")

        axes[0].plot(t, y_true, color=PALETTE["true"], linewidth=2.0, label="Ground truth")
        axes[0].fill_between(t, 0, y_true, color=PALETTE["true"], alpha=0.10)
        axes[0].set_title("A. Log-Cf-weighted target map", loc="left", fontweight="semibold")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(t, y_raw, color=PALETTE["raw"], linewidth=2.0, label="Raw DCCDM output")
        axes[1].axhline(0.0, color="#c3c2b7", linewidth=1.0)
        axes[1].set_title("B. Raw decoded prediction (range violations retained)", loc="left", fontweight="semibold")
        axes[1].set_ylabel("Amplitude")

        axes[2].plot(t, y_post, color=PALETTE["post"], linewidth=2.0, label="Postprocessed output")
        if len(detected):
            axes[2].scatter(
                t[detected], y_post[detected], s=42, color=PALETTE["marker"],
                edgecolors="white", linewidths=1.5, label="Blind detected peaks", zorder=3,
            )
        axes[2].set_title("C. Clamped and thresholded diagnostic view", loc="left", fontweight="semibold")
        axes[2].set_ylabel("Amplitude")

        axes[3].plot(t, y_obs, color=PALETTE["observation"], linewidth=1.5, label="Wellhead observation")
        axes[3].set_title("D. Input wellhead pressure trace", loc="left", fontweight="semibold")
        axes[3].set_ylabel("Head (m)")
        axes[3].set_xlabel("Time (s)")

        for axis in axes:
            for arrival in true_arrivals:
                axis.axvline(arrival, color=PALETTE["true"], linestyle=":", linewidth=1.0, alpha=0.65)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
            axis.legend(loc="upper right", frameon=False)
            axis.margins(x=0)
            axis.set_xlim(float(t[0]), float(t[-1]))

        save_path = os.path.join(output_dir, f"dccdm_{case_id}.png")
        fig.savefig(save_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        paths.append(save_path)
        print(f"Generated {save_path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a schema-v2 DCCDM prediction artifact")
    parser.add_argument("artifact", help="predictions.npz or final_predictions.npz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--peak-height", type=float, default=1.0)
    parser.add_argument("--peak-prominence", type=float, default=0.5)
    args = parser.parse_args()
    generate_publication_figures(
        args.artifact,
        output_dir=args.output_dir,
        peak_height=args.peak_height,
        peak_prominence=args.peak_prominence,
    )


if __name__ == "__main__":
    main()
