"""
Cluster credibility Psi_k = Gamma_k * E_k * I_k on the Paper A steady grid.

- Gamma_k = P_k / sigma_ref, sigma_ref = 0.05 (reference scale, no injected noise)
- E_k from the Sec. 4 index-exponential envelope (local fit; transfer uses
  gamma(S) calibrated at X1=3000 m, n=3-8, applied with local P_1)
- I_k from adjacent-valley depth on the accumulated 2-D cepstral profile
  No eta_phase, T_eff, Grade A/B/C, or Zone labels.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit

REPO_ROOT = r"E:\water_hammer_research\wellbore_moc_method"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum

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
    "axes.spines.top": True,
    "axes.spines.right": True,
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

COLOR_BLUE = "#1B4F72"
COLOR_TEAL = "#117864"
COLOR_ORANGE = "#D35400"
COLOR_RED = "#900C3F"
COLOR_SLATE = "#2C3E50"
# Low min-Psi (weak tail) is dark so it remains visible; not a Zone map.
CMAP = LinearSegmentedColormap.from_list(
    "psi_seq", ["#1B4F72", "#2874A6", "#5DADE2", "#AED6F1", "#EBF5FB"]
)

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")
NPZ_DIR = os.path.join(
    DATA_SRC, [d for d in os.listdir(DATA_SRC) if "npz" in d.lower()][0]
)
DECAY_CSV = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
OUT_DATA = os.path.join(BASE_DIR, "文章撰写", "稿件", "data")
OUT_FIG = os.path.join(BASE_DIR, "文章撰写", "稿件", "图表数据")

SIGMA_REF = 0.05
X1_CAL = 3000.0
X1_GRID = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0)
S_GRID = tuple(range(10, 101, 10))
N_WORKERS = min(8, os.cpu_count() or 4)


def exp_decay_model(x, a, gamma):
    return a * np.exp(-gamma * x)


def paper_table():
    df = pd.read_csv(DECAY_CSV)
    df = df[(df["friction_model"] == "steady") & (df["x1"].isin(X1_GRID))].copy()
    df["x1"] = df["x1"].astype(float)
    df["spacing_m"] = df["spacing_m"].astype(float)
    df["n_total"] = df["n_total"].astype(int)
    df["frac_idx"] = df["frac_idx"].astype(int)
    return df


def fit_local_envelopes(df):
    rows = []
    grouped = df[df["n_total"] >= 2].groupby(["x1", "spacing_m", "n_total"])
    for (x1, S, n), g in grouped:
        g = g.sort_values("frac_idx")
        idxs = g["frac_idx"].to_numpy(dtype=float)
        p_vals = g["P_2d"].to_numpy(dtype=float)
        a_fit = np.nan
        gamma = np.nan
        r2 = np.nan
        if len(idxs) >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        exp_decay_model, idxs - 1.0, p_vals,
                        p0=[max(p_vals[0], 1e-3), 0.35], maxfev=8000,
                    )
                a_fit, gamma = float(popt[0]), float(popt[1])
                pred = exp_decay_model(idxs - 1.0, a_fit, gamma)
                ss_res = np.sum((p_vals - pred) ** 2)
                ss_tot = np.sum((p_vals - np.mean(p_vals)) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
            except Exception:
                a_fit, gamma, r2 = np.nan, np.nan, np.nan
        rows.append({
            "x1": float(x1),
            "spacing_m": float(S),
            "n_total": int(n),
            "A_fit": a_fit,
            "gamma_local": gamma,
            "R2_local": r2,
            "P_1": float(p_vals[0]),
        })
    return pd.DataFrame(rows)


def calibrate_gamma(df_fit):
    sub = df_fit[
        (df_fit["x1"] == X1_CAL)
        & (df_fit["n_total"] >= 3)
        & np.isfinite(df_fit["gamma_local"])
    ]
    cal = sub.groupby("spacing_m", as_index=False).agg(
        gamma_cal=("gamma_local", "mean"),
        gamma_cal_std=("gamma_local", "std"),
        n_fit=("gamma_local", "count"),
    )
    return cal


def npz_name(x1, S, n):
    return f"steady_x1_{int(x1)}_sp_{int(S)}_n_{int(n)}.npz"


def case_key(x1, S, n):
    return (round(float(x1), 1), round(float(S), 1), int(n))


def isolation_from_profile(depth, prof, x_f, S):
    """I_k from adjacent-valley depth on P_2D(x). n=1 -> 1.0."""
    x_f = np.asarray(x_f, dtype=float)
    n = len(x_f)
    I = np.ones(n, dtype=float)
    P_prof = np.full(n, np.nan)
    V = np.full(n, np.nan)
    r = min(15.0, 0.49 * float(S)) if n >= 2 else 15.0
    for k, xk in enumerate(x_f):
        mask = np.abs(depth - xk) <= r
        if np.any(mask):
            P_prof[k] = float(np.max(prof[mask]))
    if n == 1:
        I[0] = 1.0
        return I, P_prof, V

    valleys = []
    for k in range(n - 1):
        between = (depth > x_f[k]) & (depth < x_f[k + 1])
        if np.any(between):
            valleys.append(float(np.min(prof[between])))
        else:
            valleys.append(np.nan)

    for k in range(n):
        adj = []
        if k > 0:
            adj.append(valleys[k - 1])
        if k < n - 1:
            adj.append(valleys[k])
        adj = [v for v in adj if np.isfinite(v)]
        if not adj or not np.isfinite(P_prof[k]) or P_prof[k] <= 0:
            I[k] = np.nan
            continue
        # Negative troughs are cepstral oscillation, not a raised saddle.
        V[k] = float(max(0.0, np.max(adj)))
        I[k] = float(np.clip((P_prof[k] - V[k]) / P_prof[k], 0.0, 1.0))
    return I, P_prof, V


def _isolation_worker(payload):
    x1, S, n, x_f, npz_path = payload
    if not os.path.isfile(npz_path):
        return {
            "x1": x1, "spacing_m": S, "n_total": n, "ok": False,
            "I": None, "P_prof": None, "V": None,
        }
    data = np.load(npz_path)
    t = data["t_sim"]
    H_wh = data["H_wh"]
    v = float(np.atleast_1d(data["v"])[0])
    fs = float(np.atleast_1d(data["fs"])[0])
    ts = float(np.atleast_1d(data["ts"])[0])
    L = float(np.atleast_1d(data["L"])[0])
    out = compute_moc_cepstrum(
        t, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
        wlen_sec=30.0, hop_sec=5.0, win_type="hamming",
    )
    depth = np.asarray(out["depth"], dtype=float)
    prof = -np.sum(out["C"], axis=1)
    I, P_prof, V = isolation_from_profile(depth, prof, x_f, S)
    return {
        "x1": x1, "spacing_m": S, "n_total": n, "ok": True,
        "I": I.tolist(), "P_prof": P_prof.tolist(), "V": V.tolist(),
    }


def compute_isolation_table(df, max_workers=N_WORKERS):
    cases = []
    for (x1, S, n), g in df.groupby(["x1", "spacing_m", "n_total"]):
        g = g.sort_values("frac_idx")
        x_f = g["x_f"].to_numpy(dtype=float)
        if int(n) == 1:
            cases.append((float(x1), float(S), int(n), x_f, None))
        else:
            path = os.path.join(NPZ_DIR, npz_name(x1, S, n))
            cases.append((float(x1), float(S), int(n), x_f, path))

    iso_map = {}
    n1 = [c for c in cases if c[2] == 1]
    n_multi = [c for c in cases if c[2] >= 2]
    for x1, S, n, x_f, _ in n1:
        iso_map[case_key(x1, S, n)] = {
            "I": [1.0], "P_prof": [np.nan], "V": [np.nan], "ok": True,
        }

    payloads = [(c[0], c[1], c[2], c[3].tolist(), c[4]) for c in n_multi]
    print(f"[isolation] {len(payloads)} multi-fracture profiles, workers={max_workers}")
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_isolation_worker, p) for p in payloads]
        for fut in as_completed(futs):
            res = fut.result()
            iso_map[case_key(res["x1"], res["spacing_m"], res["n_total"])] = res
            done += 1
            if done % 40 == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)}")
    return iso_map


def assemble_psi(df, df_fit, df_cal, iso_map):
    cal = {float(s): float(g) for s, g in zip(df_cal["spacing_m"], df_cal["gamma_cal"])}
    fit_map = {
        case_key(r.x1, r.spacing_m, r.n_total): r
        for _, r in df_fit.iterrows()
    }
    rows = []
    for (x1, S, n), g in df.groupby(["x1", "spacing_m", "n_total"]):
        g = g.sort_values("frac_idx")
        key = case_key(x1, S, n)
        iso = iso_map.get(key)
        gamma_cal = cal.get(float(S), np.nan)
        A_fit = np.nan
        gamma_local = np.nan
        r2_local = np.nan
        rec = fit_map.get(key)
        if rec is not None:
            A_fit = float(rec["A_fit"])
            gamma_local = float(rec["gamma_local"])
            r2_local = float(rec["R2_local"])

        I_list = iso["I"] if iso and iso.get("I") is not None else [np.nan] * int(n)
        P_prof = iso["P_prof"] if iso and iso.get("P_prof") is not None else [np.nan] * int(n)
        V_list = iso["V"] if iso and iso.get("V") is not None else [np.nan] * int(n)

        for _, row in g.iterrows():
            k = int(row["frac_idx"])
            Pk = float(row["P_2d"])
            Gamma = Pk / SIGMA_REF
            P_env_local = np.nan
            P_env_xfer = np.nan
            if int(n) == 1:
                E_local = 1.0
                E_xfer = 1.0
            else:
                if np.isfinite(A_fit) and np.isfinite(gamma_local):
                    P_env_local = A_fit * np.exp(-gamma_local * (k - 1))
                    E_local = float(np.clip(Pk / max(P_env_local, 1e-12), 0.0, 1.0))
                else:
                    E_local = np.nan
                if np.isfinite(gamma_cal):
                    P1 = float(g.loc[g["frac_idx"] == 1, "P_2d"].iloc[0])
                    P_env_xfer = P1 * np.exp(-gamma_cal * (k - 1))
                    E_xfer = float(np.clip(Pk / max(P_env_xfer, 1e-12), 0.0, 1.0))
                else:
                    E_xfer = np.nan
            Ik = float(I_list[k - 1]) if k - 1 < len(I_list) else np.nan
            Psi_local = Gamma * E_local * Ik if np.isfinite(E_local) and np.isfinite(Ik) else np.nan
            Psi_xfer = Gamma * E_xfer * Ik if np.isfinite(E_xfer) and np.isfinite(Ik) else np.nan
            rows.append({
                "friction_model": "steady",
                "x1": float(x1),
                "spacing_m": float(S),
                "n_total": int(n),
                "frac_idx": k,
                "x_f": float(row["x_f"]),
                "P_2d": Pk,
                "P_2d_profile": float(P_prof[k - 1]) if k - 1 < len(P_prof) else np.nan,
                "valley": float(V_list[k - 1]) if k - 1 < len(V_list) else np.nan,
                "sigma_ref": SIGMA_REF,
                "Gamma_k": Gamma,
                "A_fit": A_fit,
                "gamma_local": gamma_local,
                "R2_local": r2_local,
                "gamma_cal": gamma_cal,
                "P_env_local": P_env_local,
                "P_env_xfer": P_env_xfer,
                "E_k_local": E_local,
                "E_k_xfer": E_xfer,
                "I_k": Ik,
                "Psi_k_local": Psi_local,
                "Psi_k_xfer": Psi_xfer,
            })
    return pd.DataFrame(rows)


def stage_table(df_psi):
    recs = []
    for (x1, S, n), g in df_psi.groupby(["x1", "spacing_m", "n_total"]):
        recs.append({
            "x1": float(x1),
            "spacing_m": float(S),
            "n_total": int(n),
            "min_Psi_local": float(np.nanmin(g["Psi_k_local"])),
            "min_Psi_xfer": float(np.nanmin(g["Psi_k_xfer"])),
            "median_Psi_local": float(np.nanmedian(g["Psi_k_local"])),
            "min_I": float(np.nanmin(g["I_k"])),
            "min_E_local": float(np.nanmin(g["E_k_local"])),
            "min_E_xfer": float(np.nanmin(g["E_k_xfer"])),
            "min_Gamma": float(np.nanmin(g["Gamma_k"])),
            "P_n": float(g.loc[g["frac_idx"] == int(n), "P_2d"].iloc[0]),
            "gamma_local": float(g["gamma_local"].iloc[0]) if int(n) >= 2 else np.nan,
            "gamma_cal": float(g["gamma_cal"].iloc[0]) if int(n) >= 2 else np.nan,
            "R2_local": float(g["R2_local"].iloc[0]) if int(n) >= 2 else np.nan,
        })
    out = pd.DataFrame(recs)
    out["abs_mismatch"] = np.abs(out["min_Psi_xfer"] - out["min_Psi_local"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rel_mismatch"] = out["abs_mismatch"] / out["min_Psi_local"]
    return out


def fmt(x, nd=3):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def summarize(df_psi, df_stage, df_cal):
    st = df_stage.copy()
    cal3000 = st[(st["x1"] == X1_CAL) & (st["n_total"] >= 2)]
    nge3 = st[st["n_total"] >= 3]
    # transfer stats by depth, n>=3
    by_x1 = []
    for x1, g in nge3.groupby("x1"):
        g = g[np.isfinite(g["min_Psi_local"]) & np.isfinite(g["min_Psi_xfer"])]
        mape = float(np.nanmean(g["rel_mismatch"])) if len(g) else np.nan
        mae = float(np.nanmean(g["abs_mismatch"])) if len(g) else np.nan
        med = float(np.nanmedian(g["rel_mismatch"])) if len(g) else np.nan
        by_x1.append({
            "x1": float(x1),
            "n_cases": int(len(g)),
            "MAE_minPsi": mae,
            "MAPE_minPsi": mape,
            "median_rel_mismatch": med,
            "frac_rel_gt_0.05": float(np.mean(g["rel_mismatch"] > 0.05)),
            "frac_rel_gt_0.10": float(np.mean(g["rel_mismatch"] > 0.10)),
            "max_rel_mismatch": float(np.nanmax(g["rel_mismatch"])),
            "mean_min_Psi_local": float(np.nanmean(g["min_Psi_local"])),
            "mean_min_Psi_xfer": float(np.nanmean(g["min_Psi_xfer"])),
        })
    df_x1 = pd.DataFrame(by_x1)

    off = nge3[nge3["x1"] != X1_CAL]
    off = off[np.isfinite(off["rel_mismatch"])]
    cal_n = nge3[nge3["x1"] == X1_CAL]
    cal_n = cal_n[np.isfinite(cal_n["rel_mismatch"])]

    # archetype cases at X1=3000
    archetypes = [
        ("isolated", 80.0, 3),
        ("coalescence", 10.0, 4),
        ("later_suppression", 30.0, 4),
        ("weak_tail", 20.0, 8),
    ]
    arch_out = []
    for name, S, n in archetypes:
        g = df_psi[(df_psi["x1"] == X1_CAL) & (df_psi["spacing_m"] == S) & (df_psi["n_total"] == n)]
        stg = st[(st["x1"] == X1_CAL) & (st["spacing_m"] == S) & (st["n_total"] == n)]
        arch_out.append({
            "name": name, "S": S, "n": n,
            "min_Psi_local": float(stg["min_Psi_local"].iloc[0]) if len(stg) else np.nan,
            "min_I": float(stg["min_I"].iloc[0]) if len(stg) else np.nan,
            "Psi_k": [round(float(v), 3) for v in g.sort_values("frac_idx")["Psi_k_local"]],
            "I_k": [round(float(v), 3) for v in g.sort_values("frac_idx")["I_k"]],
        })

    # n=8, X1=3000 min_Psi vs S
    n8 = cal3000[cal3000["n_total"] == 8].sort_values("spacing_m")
    n8_psi = {
        int(r.spacing_m): round(float(r.min_Psi_local), 3)
        for _, r in n8.iterrows()
    }

    summary = {
        "sigma_ref": SIGMA_REF,
        "n_cluster_rows": int(len(df_psi)),
        "n_stage_cases": int(len(df_stage)),
        "gamma_cal_by_S": {
            int(r.spacing_m): round(float(r.gamma_cal), 4)
            for _, r in df_cal.iterrows()
        },
        "X1_3000_n>=2_median_min_Psi": float(np.nanmedian(cal3000["min_Psi_local"])),
        "X1_3000_n>=3_median_min_I": float(np.nanmedian(
            st[(st["x1"] == X1_CAL) & (st["n_total"] >= 3)]["min_I"]
        )),
        "calibration_X1_3000_n>=3_MAPE": float(np.nanmean(cal_n["rel_mismatch"])) if len(cal_n) else np.nan,
        "off_calibration_n>=3_MAPE": float(np.nanmean(off["rel_mismatch"])) if len(off) else np.nan,
        "off_calibration_n>=3_median_rel": float(np.nanmedian(off["rel_mismatch"])) if len(off) else np.nan,
        "off_calibration_n>=3_MAE": float(np.nanmean(off["abs_mismatch"])) if len(off) else np.nan,
        "by_x1": by_x1,
        "archetypes_X1_3000": arch_out,
        "n8_X1_3000_min_Psi_by_S": n8_psi,
        "S10_n4_X1_3000_min_I": float(
            st[(st["x1"] == X1_CAL) & (st["spacing_m"] == 10) & (st["n_total"] == 4)]["min_I"].iloc[0]
        ) if len(st[(st["x1"] == X1_CAL) & (st["spacing_m"] == 10) & (st["n_total"] == 4)]) else np.nan,
        "S80_n3_X1_3000_min_I": float(
            st[(st["x1"] == X1_CAL) & (st["spacing_m"] == 80) & (st["n_total"] == 3)]["min_I"].iloc[0]
        ) if len(st[(st["x1"] == X1_CAL) & (st["spacing_m"] == 80) & (st["n_total"] == 3)]) else np.nan,
        "I_k_all_ones": bool(np.allclose(df_psi["I_k"].fillna(-1).to_numpy(dtype=float), 1.0)),
        "off_cal_frac_rel_gt_0.10": float(np.mean(off["rel_mismatch"] > 0.10)) if len(off) else np.nan,
        "off_cal_max_rel": float(np.nanmax(off["rel_mismatch"])) if len(off) else np.nan,
    }
    return summary, df_x1


def plot_psi_figure(df_stage):
    os.makedirs(OUT_FIG, exist_ok=True)
    st = df_stage.copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.45), constrained_layout=True)

    # (a) discrete min Psi at X1=3000, n=3..8 (envelope identified)
    ax = axes[0]
    ax.set_box_aspect(4 / 5)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    S_vals = np.array(S_GRID, dtype=float)
    n_vals = np.arange(3, 9)
    Z = np.full((len(n_vals), len(S_vals)), np.nan)
    sub = st[(st["x1"] == X1_CAL) & (st["n_total"] >= 3)]
    for i, S in enumerate(S_vals):
        for j, n in enumerate(n_vals):
            hit = sub[(sub["spacing_m"] == S) & (sub["n_total"] == n)]
            if len(hit):
                Z[j, i] = float(hit["min_Psi_local"].iloc[0])
    Se = np.concatenate([S_vals - 5.0, [S_vals[-1] + 5.0]])
    ne = np.concatenate([n_vals - 0.5, [n_vals[-1] + 0.5]])
    pcm = ax.pcolormesh(Se, ne, Z, cmap=CMAP, shading="flat")
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.set_yticks(n_vals)
    ax.set_xlabel(r"Fracture spacing $S$ (m)")
    ax.set_ylabel(r"Fracture count $n$")
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\min_k\Psi_k$", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.0, length=2.0)
    ax.plot(sub["spacing_m"], sub["n_total"], "o", color=COLOR_SLATE, ms=1.4, alpha=0.30)

    # (b) local vs transferred, n>=3, colour = X1
    ax = axes[1]
    ax.set_box_aspect(4 / 5)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    nge3 = st[st["n_total"] >= 3]
    from matplotlib.colors import ListedColormap
    x1_cmap = ListedColormap(
        ["#1B4F72", "#117864", "#D35400", "#900C3F", "#6C3483", "#2C3E50"]
    )
    x1_idx = nge3["x1"].map({v: i for i, v in enumerate(X1_GRID)}).to_numpy(dtype=float)
    sc = ax.scatter(
        nge3["min_Psi_local"], nge3["min_Psi_xfer"],
        c=x1_idx, cmap=x1_cmap, s=11, alpha=0.85, linewidths=0, zorder=2,
        vmin=-0.5, vmax=len(X1_GRID) - 0.5,
    )
    lims = nge3[["min_Psi_local", "min_Psi_xfer"]].to_numpy(dtype=float)
    lims = lims[np.isfinite(lims)]
    hi = float(np.nanmax(lims)) * 1.12 if lims.size else 1.0
    ax.plot([0, hi], [0, hi], color="#BDC3C7", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(r"Local $\min_k\Psi_k$")
    ax.set_ylabel(r"Transferred $\min_k\Psi_k$")
    cb2 = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=range(len(X1_GRID)))
    cb2.set_label(r"$X_1$ (m)", fontsize=6.8)
    cb2.set_ticklabels([str(int(x)) for x in X1_GRID])
    cb2.ax.tick_params(labelsize=6.0, length=2.0)

    # (c) MAPE of min Psi vs X1 (n>=3)
    ax = axes[2]
    ax.set_box_aspect(4 / 5)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    mape = []
    for x1 in X1_GRID:
        g = nge3[nge3["x1"] == x1]["rel_mismatch"].to_numpy(dtype=float)
        g = g[np.isfinite(g)]
        mape.append(100.0 * float(np.mean(g)) if len(g) else np.nan)
    xpos = np.arange(len(X1_GRID))
    bar_colors = [COLOR_TEAL if x1 == X1_CAL else COLOR_BLUE for x1 in X1_GRID]
    ax.bar(xpos, mape, width=0.62, color=bar_colors, edgecolor=COLOR_SLATE, lw=0.4, zorder=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(int(x)) for x in X1_GRID], fontsize=6.5)
    ax.set_xlabel(r"Lead-cluster depth $X_1$ (m)")
    ax.set_ylabel(r"MAPE of $\min_k\Psi_k$ (%)")
    ax.set_ylim(0, max(mape) * 1.35 if mape else 1.0)
    for x, v in zip(xpos, mape):
        ax.text(x, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=6.0, color=COLOR_SLATE)

    path = os.path.join(OUT_FIG, "Figure_Psi_k_Calibration_and_Transfer")
    fig.savefig(path + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(path + ".svg", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", path + ".{png,svg,pdf}")
    return path


def main():
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    print("[1/5] load decay_table")
    df = paper_table()
    print("  rows", len(df), "cases", df.groupby(["x1", "spacing_m", "n_total"]).ngroups)

    print("[2/5] local envelopes")
    df_fit = fit_local_envelopes(df)
    df_cal = calibrate_gamma(df_fit)
    df_fit.to_csv(os.path.join(OUT_DATA, "psi_envelope_fits.csv"), index=False)
    df_cal.to_csv(os.path.join(OUT_DATA, "psi_gamma_cal_X1_3000.csv"), index=False)

    print("[3/5] isolation I_k from profiles")
    iso_map = compute_isolation_table(df)

    print("[4/5] assemble Psi_k")
    df_psi = assemble_psi(df, df_fit, df_cal, iso_map)
    df_stage = stage_table(df_psi)
    df_psi.to_csv(os.path.join(OUT_DATA, "psi_k_cluster.csv"), index=False)
    df_stage.to_csv(os.path.join(OUT_DATA, "psi_k_stage.csv"), index=False)

    print("[5/5] summary + figure")
    summary, df_x1 = summarize(df_psi, df_stage, df_cal)
    df_x1.to_csv(os.path.join(OUT_DATA, "psi_k_transfer_by_x1.csv"), index=False)
    with open(os.path.join(OUT_DATA, "psi_k_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_psi_figure(df_stage)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("done")


if __name__ == "__main__":
    main()
