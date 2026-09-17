"""
Section 5.6: empirical index-exponential envelope within fixed geometries.
Data source: 01_几何网格/峰值表/decay_table.csv (steady only).
No new forward simulations. Spatial and index models are algebraically
equivalent at fixed S and are not treated as competing physical laws.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
PALETTE_SPACING = ["#1B4F72", "#117864", "#D35400", "#900C3F", "#6C3483"]

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")
SEC_DIR = os.path.join(BASE_DIR, "文章撰写", "5.6_空间衰减包络与模型拟合")
FIG_DIR = os.path.join(SEC_DIR, "figures")
TAB_DIR = os.path.join(SEC_DIR, "data")


def exp_decay_model(x, a, gamma):
    return a * np.exp(-gamma * x)


def extract_decay_model_metrics():
    print("[1/3] Fitting P_i = A exp[-gamma (i-1)] on fixed-geometry cases...")
    csv_table = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
    df = pd.read_csv(csv_table)
    df_sub = df[(df["friction_model"] == "steady") & (df["x1"] == 3000.0) & (df["n_total"] >= 2)].copy()

    fit_records = []
    for (n_val, sp_val), group in df_sub.groupby(["n_total", "spacing_m"]):
        group_sorted = group.sort_values("frac_idx")
        idxs = group_sorted["frac_idx"].to_numpy(dtype=float)
        p_vals = group_sorted["P_2d"].to_numpy(dtype=float)
        dx_vals = (idxs - 1.0) * float(sp_val)

        a_fit = np.nan
        gamma = np.nan
        r2 = np.nan
        rmse = np.nan
        beta_x = np.nan
        if len(idxs) >= 2:
            popt, _ = curve_fit(
                exp_decay_model, idxs - 1.0, p_vals, p0=[p_vals[0], 0.35], maxfev=5000
            )
            a_fit, gamma = float(popt[0]), float(popt[1])
            pred = exp_decay_model(idxs - 1.0, a_fit, gamma)
            ss_res = np.sum((p_vals - pred) ** 2)
            ss_tot = np.sum((p_vals - np.mean(p_vals)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
            rmse = float(np.sqrt(np.mean((p_vals - pred) ** 2)))
            # Algebraic reparameterization only: beta_x = gamma / S
            beta_x = gamma / float(sp_val)

        fit_records.append({
            "friction_model": "steady",
            "x1": 3000.0,
            "n_total": int(n_val),
            "spacing_m": float(sp_val),
            "n_frac": int(len(idxs)),
            "A_fit": a_fit,
            "gamma": gamma,
            "R2": r2,
            "RMSE": rmse,
            "P_1": float(p_vals[0]),
            "P_n": float(p_vals[-1]),
            "beta_x": beta_x,
            "gamma_from_beta": beta_x * float(sp_val) if np.isfinite(beta_x) else np.nan,
        })

    df_fit = pd.DataFrame(fit_records)
    os.makedirs(TAB_DIR, exist_ok=True)
    out_csv = os.path.join(TAB_DIR, "decay_model_fitting_metrics.csv")
    df_fit.to_csv(out_csv, index=False)

    df_n38 = df_fit[df_fit["n_total"] >= 3]
    print(
        f"[Done] {len(df_fit)} cases (n=2-8); n=3-8 mean R^2 = {df_n38['R2'].mean():.4f} "
        f"(n={len(df_n38)}). Saved: {out_csv}"
    )
    return df_fit, df_sub


def plot_figure_5_6(df_fit, df_sub):
    print("[2/3] Generating Figure 5.6 (empirical index envelope)...")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)
    df_n8 = df_sub[df_sub["n_total"] == 8]
    sel_spacings = [10, 20, 30, 50, 100]

    ax = axes[0]
    ax.set_box_aspect(4 / 5)
    for sp_val, col in zip(sel_spacings, PALETTE_SPACING):
        sub_sp = df_n8[df_n8["spacing_m"] == sp_val].sort_values("frac_idx")
        ax.plot(
            sub_sp["frac_idx"], sub_sp["P_2d"], marker="o", label=f"$S={sp_val}$ m",
            color=col, lw=1.1, markersize=3.4,
        )
    ax.set_xlabel("Fracture Index $i$")
    ax.set_ylabel("Apparent Peak $P_i$ (a.u.)")
    ax.set_xlim(0.6, 8.4)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_ylim(0.0, 4.8)
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax = axes[1]
    ax.set_box_aspect(4 / 5)
    fit_sel = [10, 30, 70, 100]
    fit_cols = [COLOR_BLUE_MAIN, COLOR_TEAL, COLOR_ORANGE, COLOR_RED]
    i_dense = np.linspace(1, 8, 100)
    for sp_val, col in zip(fit_sel, fit_cols):
        sub_sp = df_n8[df_n8["spacing_m"] == sp_val].sort_values("frac_idx")
        row = df_fit[(df_fit["n_total"] == 8) & (df_fit["spacing_m"] == sp_val)].iloc[0]
        a_fit, gamma, r2 = row["A_fit"], row["gamma"], row["R2"]
        ax.plot(sub_sp["frac_idx"], sub_sp["P_2d"], marker="o", ls="none", color=col, markersize=3.6)
        ax.plot(
            i_dense, a_fit * np.exp(-gamma * (i_dense - 1.0)), color=col, lw=1.1,
            label=f"$S={sp_val}$ m ($R^2={r2:.3f}$)",
        )
    ax.set_xlabel("Fracture Index $i$")
    ax.set_ylabel("Apparent Peak $P_i$ (a.u.)")
    ax.set_xlim(0.6, 8.4)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_ylim(0.0, 5.2)
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax = axes[2]
    ax.set_box_aspect(4 / 5)
    n_list = [3, 4, 5, 6, 7, 8]
    cmap_n = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_list)))
    for n_val, col in zip(n_list, cmap_n):
        sub_n = df_fit[df_fit["n_total"] == n_val].sort_values("spacing_m")
        ax.plot(
            sub_n["spacing_m"], sub_n["R2"], marker="o", label=f"$n={n_val}$",
            color=col, lw=1.0, markersize=3.2,
        )
    ax.axhline(0.95, color="#E74C3C", ls="--", lw=0.8, alpha=0.8)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel("Envelope $R^2$")
    ax.set_xlim(5, 105)
    ax.set_ylim(0.90, 1.005)
    ax.legend(loc="lower left", fontsize=6.0, handlelength=1.1, handletextpad=0.3, borderaxespad=0.25)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    out_prefix = os.path.join(FIG_DIR, "Figure_5_6_Spatial_vs_Topological_Decay")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.6: {out_prefix}")


def plot_figure_5_6_supp(df_fit):
    print("[3/3] Generating Figure 5.6 supplementary (gamma and algebraic identity)...")
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)
    df_n8 = df_fit[df_fit["n_total"] == 8].sort_values("spacing_m")

    ax = axes[0]
    ax.set_box_aspect(4 / 5)
    ax.plot(df_n8["spacing_m"], df_n8["gamma"], marker="o", color=COLOR_BLUE_MAIN, lw=1.1, markersize=3.6)
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel(r"Index Decay Rate $\gamma$")
    ax.set_xlim(5, 105)
    ax.set_ylim(0.15, 0.85)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax = axes[1]
    ax.set_box_aspect(4 / 5)
    ax.plot(
        df_n8["spacing_m"], df_n8["beta_x"], marker="o", color=COLOR_RED,
        lw=1.1, markersize=3.6, label=r"$\beta_x=\gamma/S$",
    )
    s_dense = np.linspace(10, 100, 80)
    ax.plot(s_dense, df_n8["gamma"].mean() / s_dense, ls=":", color=COLOR_SLATE, lw=1.0, label=r"$\bar{\gamma}/S$")
    ax.set_xlabel("Fracture Spacing $S$ (m)")
    ax.set_ylabel(r"Reparameterized $\beta_x$ (m$^{-1}$)")
    ax.set_xlim(5, 105)
    ax.set_ylim(0.0, 0.065)
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.1, handletextpad=0.3, borderaxespad=0.25)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax = axes[2]
    ax.set_box_aspect(4 / 5)
    df_n38 = df_fit[df_fit["n_total"] >= 3]
    ax.plot([0.2, 0.9], [0.2, 0.9], ls=":", color="#BDC3C7", lw=0.9, zorder=0)
    ax.scatter(df_n38["gamma"], df_n38["gamma_from_beta"], s=12, color=COLOR_TEAL, zorder=3)
    ax.set_xlabel(r"Fitted $\gamma$")
    ax.set_ylabel(r"$\beta_x S$")
    ax.set_xlim(0.20, 0.90)
    ax.set_ylim(0.20, 0.90)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    out_prefix = os.path.join(FIG_DIR, "Figure_5_6_Supp_Model_Parameters_and_Goodness")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 5.6 Supp: {out_prefix}")


if __name__ == "__main__":
    df_fit, df_sub = extract_decay_model_metrics()
    plot_figure_5_6(df_fit, df_sub)
    plot_figure_5_6_supp(df_fit)
