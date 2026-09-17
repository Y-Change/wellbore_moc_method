"""
Section 5.7: threshold sensitivity of true-neighborhood apparent peaks.
Detection count is n_det(P_th) = sum 1(P_i >= P_th), not a blind-recall rate.
sigma_noise = 0.05 is an assumed reference scale, not injected noise.
No operational zones, no field spacing recommendations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = r"E:\water_hammer_research\wellbore_moc_method"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.75,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.8,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "legend.frameon": False,
    "figure.dpi": 300,
})

COLOR_BLUE_MAIN = "#1B4F72"
COLOR_TEAL = "#117864"
COLOR_ORANGE = "#D35400"
COLOR_RED = "#900C3F"
COLOR_PURPLE = "#6C3483"
COLOR_SLATE = "#2C3E50"

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")
SEC_DIR = os.path.join(BASE_DIR, "文章撰写", "5.7_可辨识度边界与可行域评估")
FIG_DIR = os.path.join(SEC_DIR, "figures")
TAB_DIR = os.path.join(SEC_DIR, "data")

THRESHOLDS = [0.15, 0.25, 0.35, 0.50]
SIGMA_NOISE = 0.05


def _cell_edges(values):
    values = np.asarray(values, dtype=float)
    mid = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mid[0] - values[0])
    last = values[-1] + (values[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _grid_matrix(df, value_col, n_vals, s_vals):
    mat = np.full((len(n_vals), len(s_vals)), np.nan)
    lookup = {
        (int(row["n_total"]), float(row["spacing_m"])): float(row[value_col])
        for _, row in df.iterrows()
    }
    for i, n_val in enumerate(n_vals):
        for j, s_val in enumerate(s_vals):
            mat[i, j] = lookup[(int(n_val), float(s_val))]
    return mat


def extract_identifiability_metrics():
    print("[1/3] Counting true-neighborhood peaks that pass assumed thresholds...")
    csv_table = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
    df = pd.read_csv(csv_table)
    df_sub = df[(df["friction_model"] == "steady") & (df["x1"] == 3000.0) & (df["n_total"] >= 2)].copy()

    records = []
    for (n_val, sp_val), group in df_sub.groupby(["n_total", "spacing_m"]):
        p_vals = group.sort_values("frac_idx")["P_2d"].to_numpy(dtype=float)
        p1 = float(p_vals[0])
        pn = float(p_vals[-1])
        rec = {
            "friction_model": "steady",
            "x1": 3000.0,
            "n_total": int(n_val),
            "spacing_m": float(sp_val),
            "P_1": p1,
            "P_end": pn,
            "R_end": pn / p1,
            "SNR_end_dB": 20.0 * np.log10(max(pn, 1e-4) / SIGMA_NOISE),
            "sigma_noise_ref": SIGMA_NOISE,
        }
        for th in THRESHOLDS:
            rec[f"n_det_th_{int(th * 100):02d}"] = int(np.sum(p_vals >= th))
            rec[f"pass_rate_th_{int(th * 100):02d}"] = float(np.sum(p_vals >= th) / len(p_vals))
        records.append(rec)

    df_env = pd.DataFrame(records)
    os.makedirs(TAB_DIR, exist_ok=True)
    out_csv = os.path.join(TAB_DIR, "identifiability_envelope_metrics.csv")
    df_env.to_csv(out_csv, index=False)
    print(f"[Done] {len(df_env)} grid points saved to {out_csv}")
    return df_env, df_sub


def _draw_discrete_field(ax, df, value_col, cmap, levels_label, vmin=None, vmax=None):
    s_vals = np.array(sorted(df["spacing_m"].unique()))
    n_vals = np.array(sorted(df["n_total"].unique()))
    mat = _grid_matrix(df, value_col, n_vals, s_vals)
    mesh = ax.pcolormesh(
        _cell_edges(s_vals), _cell_edges(n_vals), mat,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
    )
    ax.scatter(df["spacing_m"], df["n_total"], color=COLOR_SLATE, s=6, alpha=0.45, edgecolors="none", zorder=3)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel(r"Fracture Count $n$")
    ax.set_xlim(5, 105)
    ax.set_ylim(1.5, 8.5)
    ax.set_yticks([2, 3, 4, 5, 6, 7, 8])
    ax.set_xticks([10, 20, 40, 60, 80, 100])
    return mesh


def plot_figure_5_7(df_env):
    print("[2/3] Generating Figure 5.7 (threshold sensitivity)...")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)

    ax = axes[0]
    ax.set_box_aspect(4 / 5)
    cmap_pass = LinearSegmentedColormap.from_list("pass", ["#FADBD8", "#FCF3CF", "#D5F5E3", "#52BE80"])
    mesh = _draw_discrete_field(ax, df_env, "pass_rate_th_35", cmap_pass, "pass", vmin=0.35, vmax=1.0)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label(r"Pass rate at $P_{\mathrm{th}}=0.35$", fontsize=6.6, labelpad=3)
    cbar.ax.tick_params(labelsize=6.0, length=2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax = axes[1]
    ax.set_box_aspect(4 / 5)
    sub_n8 = df_env[df_env["n_total"] == 8].sort_values("spacing_m")
    styles = [
        ("n_det_th_15", r"$P_{\mathrm{th}}=0.15$", COLOR_TEAL, "o"),
        ("n_det_th_25", r"$P_{\mathrm{th}}=0.25$", COLOR_ORANGE, "s"),
        ("n_det_th_35", r"$P_{\mathrm{th}}=0.35$", COLOR_RED, "^"),
        ("n_det_th_50", r"$P_{\mathrm{th}}=0.50$", COLOR_PURPLE, "d"),
    ]
    for col_name, lbl, col, mkr in styles:
        ax.plot(sub_n8["spacing_m"], sub_n8[col_name], marker=mkr, color=col, lw=1.0, markersize=3.4, label=lbl)
    ax.axhline(8, color="#BDC3C7", ls=":", lw=0.8)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel(r"$n_{\mathrm{det}}$ ($n=8$)")
    ax.set_xlim(5, 105)
    ax.set_ylim(1.5, 9.2)
    ax.set_yticks([2, 4, 6, 8])
    ax.legend(loc="lower right", fontsize=5.8, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax = axes[2]
    ax.set_box_aspect(4 / 5)
    cmap_snr = LinearSegmentedColormap.from_list("snr", ["#C0392B", "#E67E22", "#F9E79F", "#A9DFBF", "#1E8449"])
    mesh2 = _draw_discrete_field(ax, df_env, "SNR_end_dB", cmap_snr, "snr", vmin=7.5, vmax=28.5)
    cbar2 = fig.colorbar(mesh2, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar2.set_label(r"Apparent $P_n/\sigma_{\mathrm{ref}}$ (dB)", fontsize=6.6, labelpad=3)
    cbar2.ax.tick_params(labelsize=6.0, length=2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    out_prefix = os.path.join(FIG_DIR, "Figure_5_7_Operational_Envelope")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.7: {out_prefix}")


def plot_figure_5_7_supp(df_env):
    print("[3/3] Generating Figure 5.7 supplementary (threshold sweep)...")
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)

    ax = axes[0]
    ax.set_box_aspect(4 / 5)
    cmap_r = LinearSegmentedColormap.from_list("rend", ["#D6EAF8", "#5DADE2", "#F5B041", "#E74C3C"])
    mesh = _draw_discrete_field(ax, df_env, "R_end", cmap_r, "rend", vmin=0.03, vmax=0.48)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label(r"End-to-first ratio $R_{\mathrm{end}}$", fontsize=6.6, labelpad=3)
    cbar.ax.tick_params(labelsize=6.0, length=2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax = axes[1]
    ax.set_box_aspect(4 / 5)
    n_list = [2, 4, 6, 8]
    colors_n = [COLOR_TEAL, COLOR_BLUE_MAIN, COLOR_ORANGE, COLOR_RED]
    for n_val, col in zip(n_list, colors_n):
        sub_n = df_env[df_env["n_total"] == n_val]
        rates = [100.0 * sub_n[f"pass_rate_th_{int(th * 100):02d}"].mean() for th in THRESHOLDS]
        ax.plot(THRESHOLDS, rates, marker="o", color=col, lw=1.1, markersize=3.6, label=f"$n={n_val}$")
    ax.set_xlabel(r"Threshold $P_{\mathrm{th}}$ (a.u.)")
    ax.set_ylabel("Mean pass rate (%)")
    ax.set_xlim(0.10, 0.55)
    ax.set_ylim(35, 110)
    ax.legend(loc="lower left", fontsize=6.0, handlelength=1.1, handletextpad=0.3, borderaxespad=0.25)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax = axes[2]
    ax.set_box_aspect(4 / 5)
    sub_n8 = df_env[df_env["n_total"] == 8].sort_values("spacing_m")
    ax.plot(sub_n8["spacing_m"], sub_n8["P_end"], marker="o", color=COLOR_BLUE_MAIN, lw=1.1, markersize=3.6)
    for th, ls in [(0.15, ":"), (0.35, "--"), (0.50, "-.")]:
        ax.axhline(th, color="#BDC3C7", ls=ls, lw=0.7)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel(r"Tail peak $P_8$ (a.u.)")
    ax.set_xlim(5, 105)
    ax.set_ylim(0.0, 0.65)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    out_prefix = os.path.join(FIG_DIR, "Figure_5_7_Supp_SNR_and_Field_Guidelines")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.7 Supp: {out_prefix}")


if __name__ == "__main__":
    df_env, df_sub = extract_identifiability_metrics()
    plot_figure_5_7(df_env)
    plot_figure_5_7_supp(df_env)
