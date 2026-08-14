# -*- coding: utf-8 -*-
"""EXP-022 统计段：从已有 decay_table 重算 Pow(idx)/Pow(dx) 与首缝非单调摘要。

不重新跑 MOC。统计单元是工况 (friction, x1, spacing, n_total)，不是裂缝级行。
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError("Cannot find wellbore_moc_method root")
    _d = _parent

from analysis.plotting.paper_plots import apply_paper_rc, save_figure
from moc_simulate.paths import SERIES_DECAY_REGRESSION, output_path

apply_paper_rc()


RNG_SEED = 20260814
N_BOOT = 2000


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_pow_dx(delta_x: np.ndarray, alpha: np.ndarray) -> tuple[float, float]:
    mask = (alpha > 0) & (delta_x > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    dx_m = delta_x[mask]
    a_m = alpha[mask]
    k_dx, _, _, _ = np.linalg.lstsq(np.log(1.0 + dx_m)[:, None], -np.log(a_m), rcond=None)
    k_dx = float(k_dx[0])
    pred = (1.0 + delta_x) ** (-k_dx)
    return k_dx, r_squared(alpha, pred)


def fit_pow_idx(idx: np.ndarray, alpha: np.ndarray) -> tuple[float, float]:
    mask = (alpha > 0) & (idx > 1)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    idx_m = idx[mask].astype(float)
    a_m = alpha[mask]
    k, _, _, _ = np.linalg.lstsq(np.log(idx_m)[:, None], -np.log(a_m), rcond=None)
    k = float(k[0])
    pred = idx.astype(float) ** (-k)
    return k, r_squared(alpha, pred)


def grouped_bootstrap_median(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"n": 0, "median": None, "ci95_lo": None, "ci95_hi": None}
    meds = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        meds[i] = np.median(sample)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {
        "n": int(n),
        "median": float(np.median(values)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "mean": float(np.mean(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def fit_one_case(g: pd.DataFrame, alpha_key: str) -> dict:
    g = g.sort_values("frac_idx")
    idx = g["frac_idx"].to_numpy(dtype=float)
    dx = g["delta_x"].to_numpy(dtype=float)
    alpha = g[alpha_key].to_numpy(dtype=float)
    n_pts = int(len(g))
    k_dx, r2_dx = fit_pow_dx(dx, alpha)
    k_idx, r2_idx = fit_pow_idx(idx, alpha)
    d_r2 = r2_idx - r2_dx if np.isfinite(r2_idx) and np.isfinite(r2_dx) else float("nan")
    return {
        "friction_model": g["friction_model"].iloc[0],
        "x1": float(g["x1"].iloc[0]),
        "spacing_m": float(g["spacing_m"].iloc[0]),
        "n_total": int(g["n_total"].iloc[0]),
        "n_points": n_pts,
        "alpha_key": alpha_key,
        "k_dx": k_dx,
        "r2_dx": r2_dx,
        "k_idx": k_idx,
        "r2_idx": r2_idx,
        "delta_r2": d_r2,
        # n=2 时两模型都过 (idx=1, later-1) 两点，R² 无比较意义。
        "fit_ok": bool(n_pts >= 3 and int(g["n_total"].iloc[0]) >= 3 and np.isfinite(d_r2)),
        "idx_better": bool(np.isfinite(d_r2) and d_r2 > 0),
    }


def first_frac_examples(df: pd.DataFrame) -> dict:
    first = df[df["frac_idx"] == 1].copy()
    examples = {}

    def grab(x1, n, s):
        sub = first[(first["x1"] == x1) & (first["n_total"] == n) & (first["spacing_m"] == s)]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {
            "P_1d": float(row["P_1d"]),
            "P_2d": float(row["P_2d"]),
        }

    depth_series = []
    for x1 in [2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0]:
        rec = grab(x1, 2, 50.0)
        if rec is not None:
            depth_series.append({"x1": x1, **rec})
    examples["n2_s50_vs_x1"] = depth_series

    n_series_s100 = []
    n_series_s50 = []
    for n in range(2, 9):
        rec100 = grab(2000.0, n, 100.0)
        rec50 = grab(2000.0, n, 50.0)
        if rec100 is not None:
            n_series_s100.append({"n_total": n, **rec100})
        if rec50 is not None:
            n_series_s50.append({"n_total": n, **rec50})
    examples["x1_2000_s100_vs_n"] = n_series_s100
    examples["x1_2000_s50_vs_n"] = n_series_s50

    s_series = []
    for s in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        rec = grab(2000.0, 8, s)
        if rec is not None:
            s_series.append({"spacing_m": s, **rec})
    examples["x1_2000_n8_vs_s"] = s_series

    # non-monotonic tallies among complete series
    def series_nonmono(values):
        if len(values) < 3:
            return None
        diffs = np.diff(values)
        return bool(np.any(diffs > 0) and np.any(diffs < 0))

    tallies = {
        "vs_x1_nonmono_frac": None,
        "vs_n_nonmono_frac": None,
        "vs_s_nonmono_frac": None,
        "n_series_x1": 0,
        "n_series_n": 0,
        "n_series_s": 0,
    }
    x1_flags = []
    for (n, s), g in first.groupby(["n_total", "spacing_m"]):
        g = g.sort_values("x1")
        if g["x1"].nunique() < 3:
            continue
        x1_flags.append(series_nonmono(g["P_2d"].to_numpy()))
    n_flags = []
    for (x1, s), g in first.groupby(["x1", "spacing_m"]):
        g = g.sort_values("n_total")
        if g["n_total"].nunique() < 3:
            continue
        n_flags.append(series_nonmono(g["P_2d"].to_numpy()))
    s_flags = []
    for (x1, n), g in first.groupby(["x1", "n_total"]):
        g = g.sort_values("spacing_m")
        if g["spacing_m"].nunique() < 3:
            continue
        s_flags.append(series_nonmono(g["P_2d"].to_numpy()))
    if x1_flags:
        tallies["n_series_x1"] = len(x1_flags)
        tallies["vs_x1_nonmono_frac"] = float(np.mean(x1_flags))
        tallies["vs_x1_nonmono_n"] = int(np.sum(x1_flags))
    if n_flags:
        tallies["n_series_n"] = len(n_flags)
        tallies["vs_n_nonmono_frac"] = float(np.mean(n_flags))
        tallies["vs_n_nonmono_n"] = int(np.sum(n_flags))
    if s_flags:
        tallies["n_series_s"] = len(s_flags)
        tallies["vs_s_nonmono_frac"] = float(np.mean(s_flags))
        tallies["vs_s_nonmono_n"] = int(np.sum(s_flags))
    examples["nonmono_tallies"] = tallies
    return examples


def envelope_spread(df: pd.DataFrame, n_total: int = 8) -> dict:
    """Compare vertical spread of alpha_2d at shared index vs shared physical bins."""
    sub = df[df["n_total"] == n_total].copy()
    out = {}
    idx_spreads = []
    for idx, g in sub.groupby("frac_idx"):
        if len(g) < 3:
            continue
        idx_spreads.append(
            {
                "frac_idx": int(idx),
                "iqr": float(g["alpha_2d"].quantile(0.75) - g["alpha_2d"].quantile(0.25)),
                "std": float(g["alpha_2d"].std(ddof=1)),
                "n": int(len(g)),
            }
        )
    out["by_frac_idx"] = idx_spreads
    # physical-distance bins of 20 m
    sub["dx_bin"] = (sub["delta_x"] / 20.0).round() * 20.0
    dx_spreads = []
    for dx, g in sub.groupby("dx_bin"):
        if len(g) < 3 or dx == 0:
            continue
        dx_spreads.append(
            {
                "delta_x_bin": float(dx),
                "iqr": float(g["alpha_2d"].quantile(0.75) - g["alpha_2d"].quantile(0.25)),
                "std": float(g["alpha_2d"].std(ddof=1)),
                "n": int(len(g)),
            }
        )
    out["by_dx_bin_20m"] = dx_spreads
    if idx_spreads and dx_spreads:
        later_idx = [r["iqr"] for r in idx_spreads if r["frac_idx"] >= 2]
        later_dx = [r["iqr"] for r in dx_spreads]
        out["median_iqr_idx_later"] = float(np.median(later_idx)) if later_idx else None
        out["median_iqr_dx"] = float(np.median(later_dx)) if later_dx else None
    return out


def summarize_fits(fits: pd.DataFrame, rng: np.random.Generator) -> dict:
    ok = fits[fits["fit_ok"]].copy()
    n_all = int(len(fits))
    n_ok = int(len(ok))
    n_idx_better = int(ok["idx_better"].sum()) if n_ok else 0
    summary = {
        "n_cases": n_all,
        "n_fit_ok": n_ok,
        "n_idx_better": n_idx_better,
        "frac_idx_better": float(n_idx_better / n_ok) if n_ok else None,
        "delta_r2": grouped_bootstrap_median(ok["delta_r2"].to_numpy(), rng),
        "r2_idx": grouped_bootstrap_median(ok["r2_idx"].to_numpy(), rng),
        "r2_dx": grouped_bootstrap_median(ok["r2_dx"].to_numpy(), rng),
        "k_idx": grouped_bootstrap_median(ok["k_idx"].to_numpy(), rng),
        "k_dx": grouped_bootstrap_median(ok["k_dx"].to_numpy(), rng),
        "n_negative_r2_idx": int((ok["r2_idx"] < 0).sum()) if n_ok else 0,
        "n_negative_r2_dx": int((ok["r2_dx"] < 0).sum()) if n_ok else 0,
        "n_negative_delta_r2": int((ok["delta_r2"] < 0).sum()) if n_ok else 0,
    }
    by_n = {}
    for n, g in ok.groupby("n_total"):
        by_n[str(int(n))] = {
            "n_cases": int(len(g)),
            "frac_idx_better": float(g["idx_better"].mean()),
            "median_delta_r2": float(g["delta_r2"].median()),
            "median_r2_idx": float(g["r2_idx"].median()),
            "median_r2_dx": float(g["r2_dx"].median()),
        }
    summary["by_n_total"] = by_n
    failures = ok[ok["delta_r2"] <= 0][
        ["x1", "spacing_m", "n_total", "r2_idx", "r2_dx", "delta_r2"]
    ]
    summary["idx_not_better_cases"] = failures.to_dict(orient="records")
    return summary


def audit_inventory(df: pd.DataFrame) -> dict:
    inv = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "friction_models": sorted(df["friction_model"].astype(str).unique().tolist()),
        "x1": sorted(df["x1"].unique().tolist()),
        "spacing_m": sorted(df["spacing_m"].unique().tolist()),
        "n_total": sorted(df["n_total"].unique().tolist()),
    }
    by_fr = {}
    for fr, g in df.groupby("friction_model"):
        cases = g.drop_duplicates(["x1", "spacing_m", "n_total"])
        by_fr[str(fr)] = {
            "n_rows": int(len(g)),
            "n_cases": int(len(cases)),
            "n_x1": int(g["x1"].nunique()),
            "n_spacing": int(g["spacing_m"].nunique()),
            "n_ntotal": int(g["n_total"].nunique()),
        }
    inv["by_friction"] = by_fr
    return inv


def analyze_cf_kleak(path: str) -> dict:
    if not os.path.isfile(path):
        return {"exists": False, "path": path}
    df = pd.read_csv(path)
    out = {
        "exists": True,
        "path": path,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "n_total_values": sorted(df["n_total"].unique().tolist()) if "n_total" in df.columns else None,
        "x1_values": sorted(df["x1"].unique().tolist()) if "x1" in df.columns else None,
        "spacing_values": sorted(df["spacing_m"].unique().tolist()) if "spacing_m" in df.columns else None,
        "friction_models": sorted(df["friction_model"].astype(str).unique().tolist()),
        "n_cf": int(df["Cf"].nunique()) if "Cf" in df.columns else None,
        "n_kleak": int(df["Kleak"].nunique()) if "Kleak" in df.columns else None,
    }
    mixed = bool(df["n_total"].nunique() > 1) if "n_total" in df.columns else True
    out["n_total_mixed"] = mixed
    out["usable_as_unified_n5"] = bool(
        (not mixed) and "n_total" in df.columns and set(df["n_total"].unique()) == {5}
    )
    # 即使全表混 n，也单独报告已有 n=5 子集（不得与 n=3 混拟合）
    if "steady" in set(df["friction_model"].astype(str)) and "n_total" in df.columns:
        steady = df[(df["friction_model"] == "steady") & (df["n_total"] == 5)]
        if steady.empty:
            return out
        case_rows = []
        for (cf, kleak), g in steady.groupby(["Cf", "Kleak"]):
            rec = fit_one_case(g, "alpha_2d")
            rec["Cf"] = float(cf)
            rec["Kleak"] = float(kleak)
            case_rows.append(rec)
        fits = pd.DataFrame(case_rows)
        ok = fits[fits["fit_ok"]]
        out["steady_n_cases"] = int(len(fits))
        out["steady_n_fit_ok"] = int(len(ok))
        if len(ok):
            out["steady_frac_idx_better"] = float(ok["idx_better"].mean())
            out["steady_median_delta_r2"] = float(ok["delta_r2"].median())
            out["steady_median_r2_idx"] = float(ok["r2_idx"].median())
            out["steady_median_r2_dx"] = float(ok["r2_dx"].median())
            out["steady_median_k_idx"] = float(ok["k_idx"].median())
            out["steady_k_idx_min"] = float(ok["k_idx"].min())
            out["steady_k_idx_max"] = float(ok["k_idx"].max())
            out["subset_note"] = "n=5 subset only; 30 cells in the historical table were run at n=3 and are excluded."
            # default-like cell near main-matrix Cf=1e-5, kleak=1e-4
            near = ok[(np.isclose(ok["Cf"], 1e-5)) & (np.isclose(ok["Kleak"], 1e-4))]
            out["default_cell"] = near.to_dict(orient="records")
        fits_path = output_path(SERIES_DECAY_REGRESSION, "paperA_stats", "cf_kleak_case_fits.csv")
        os.makedirs(os.path.dirname(fits_path), exist_ok=True)
        fits.to_csv(fits_path, index=False)
        out["fits_csv"] = fits_path
    return out


def plot_r2_comparison(fits: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    ok = fits[fits["fit_ok"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.scatter(ok["r2_dx"], ok["r2_idx"], s=18, alpha=0.55, c="#0F4D92", edgecolors="none")
    ax.plot([0, 1], [0, 1], ls="--", c="#767676", lw=1)
    ax.set_xlim(0.7, 1.01)
    ax.set_ylim(0.7, 1.01)
    ax.set_xlabel(r"$R^2$ of Pow$(dx)$")
    ax.set_ylabel(r"$R^2$ of Pow$(idx)$")
    ax.set_title("A  Case-wise goodness of fit", loc="left", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.hist(ok["delta_r2"], bins=24, color="#3775BA", edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#B64342", ls="--", lw=1.2)
    ax.axvline(ok["delta_r2"].median(), color="#272727", ls="-", lw=1.2)
    ax.set_xlabel(r"$\Delta R^2 = R^2_{\mathrm{idx}} - R^2_{\mathrm{dx}}$")
    ax.set_ylabel("Number of cases")
    ax.set_title("B  Advantage of topological index", loc="left", fontweight="bold")

    save_figure(fig, out_png)


def main() -> None:
    csv_path = output_path(SERIES_DECAY_REGRESSION, "03_extracted_peaks_csv", "decay_table.csv")
    out_dir = output_path(SERIES_DECAY_REGRESSION, "paperA_stats", "")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    inventory = audit_inventory(df)

    rng = np.random.default_rng(RNG_SEED)
    report = {
        "source_csv": csv_path,
        "seed": RNG_SEED,
        "n_boot": N_BOOT,
        "inventory": inventory,
        "note": "Case is the statistical unit. Fracture-level rows are not bootstrapped independently.",
    }

    for fr in sorted(df["friction_model"].astype(str).unique()):
        sub = df[df["friction_model"] == fr]
        case_rows = []
        for _, g in sub.groupby(["x1", "spacing_m", "n_total"]):
            case_rows.append(fit_one_case(g, "alpha_2d"))
        fits = pd.DataFrame(case_rows)
        fits_csv = os.path.join(out_dir, f"{fr}_case_fits_alpha2d.csv")
        fits.to_csv(fits_csv, index=False)
        fig_png = os.path.join(out_dir, f"{fr}_r2_idx_vs_dx.png")
        plot_r2_comparison(fits, fig_png)
        report[fr] = {
            "fits_csv": fits_csv,
            "r2_figure": fig_png,
            "summary": summarize_fits(fits, rng),
            "first_frac_examples": first_frac_examples(sub),
            "envelope_n8": envelope_spread(sub, n_total=8),
        }

    cf_path = output_path(SERIES_DECAY_REGRESSION, "cf_kleak_study", "cf_kleak_table.csv")
    report["cf_kleak"] = analyze_cf_kleak(cf_path)

    json_path = os.path.join(out_dir, "paperA_recomputed_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json_path)
    print(json.dumps({
        "inventory": inventory["by_friction"],
        "steady_n_cases": report.get("steady", {}).get("summary", {}).get("n_cases"),
        "steady_frac_idx_better": report.get("steady", {}).get("summary", {}).get("frac_idx_better"),
        "steady_delta_r2_median": report.get("steady", {}).get("summary", {}).get("delta_r2", {}).get("median"),
        "cf_kleak_usable": report["cf_kleak"].get("usable_as_unified_n5"),
        "cf_kleak_n_total": report["cf_kleak"].get("n_total_values"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
