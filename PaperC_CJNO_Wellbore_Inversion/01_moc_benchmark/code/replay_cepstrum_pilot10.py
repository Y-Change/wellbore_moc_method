# -*- coding: utf-8 -*-
"""Replay leakoff_multi cepstrum identification on frozen Pilot 2000 cases.

Same operator as::

    python moc_simulate/leakoff_multi.py --replay

No MOC re-run. Reads ``pilot_2000/cases/*.npz`` wellhead pressure, converts to
head, and calls ``leakoff_multi.run_cepstrum_analysis_and_match`` with the live
``CEPSTRUM_CONFIG`` (Kaiser 2D window, blind peak finder, fixed match tolerance).

Outputs stay under ``PaperC_CJNO_Wellbore_Inversion/01_moc_benchmark/``.
Does not write goldens, pilot NPZ, or formal 30k data.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from paths import (
    CEPSTRUM_PILOT_DIR,
    CODE_DIR,
    MANIFEST_DIR,
    PILOT_DIR,
    REPO_ROOT,
    TABLE_DIR,
    ensure_dirs,
)
from audit import RunAudit, dump_json, sha256_file

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moc_simulate.config import CEPSTRUM_CONFIG  # noqa: E402
from moc_simulate.leakoff_multi import (  # noqa: E402
    cepstrum_block_for_json,
    run_cepstrum_analysis_and_match,
)

PHYSICS_T_S = 1.0
SUBSET_SEED_DEFAULT = 20260907
N_DEFAULT = 10


def _ok_rows(manifest: dict) -> List[dict]:
    rows = []
    for r in manifest.get("cases", []):
        if r.get("status") != "ok":
            continue
        cid = str(r["case_id"])
        path = Path(r["file"]) if r.get("file") else PILOT_DIR / "cases" / f"{cid}.npz"
        if not path.is_file():
            path = PILOT_DIR / "cases" / f"{cid}.npz"
        if path.is_file():
            rec = dict(r)
            rec["file"] = str(path)
            rows.append(rec)
    return rows


def select_cases(manifest: dict, n: int, seed: int,
                 case_ids: Optional[Sequence[str]] = None) -> List[dict]:
    rows = _ok_rows(manifest)
    by_id = {r["case_id"]: r for r in rows}
    if case_ids:
        missing = [c for c in case_ids if c not in by_id]
        if missing:
            raise FileNotFoundError(f"requested case_id not in ok Pilot NPZ: {missing}")
        return [by_id[c] for c in case_ids]
    if n > len(rows):
        raise ValueError(f"asked for {n} cases but only {len(rows)} ok NPZ exist")
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(len(rows), size=n, replace=False)
    return [rows[int(i)] for i in pick]


def load_pilot_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    params = json.loads(str(z["params_json"]))
    meta = json.loads(str(z["meta_json"]))
    t = np.asarray(z["t"], dtype=float)
    p_head = np.asarray(z["p_head"], dtype=float)
    rho_g = float(meta["rho_g"])
    H_wh = p_head / rho_g
    well = params["well"]
    xs = [float(x) for x in params["xs"]]
    gl = [float(cl["eq"]["G_l"]) for cl in params["clusters"]]
    dt = float(meta["dt"])
    if t.size > 1:
        dt = float(np.median(np.diff(t)))
    return {
        "t": t,
        "H_wh": H_wh,
        "Q_head": np.asarray(z["Q_head"], dtype=float),
        "params": params,
        "meta": meta,
        "L": float(well["L"]),
        "a": float(well["a"]),
        "t_s": PHYSICS_T_S,
        "t_c": float(well["t_c"]),
        "dt": dt,
        "xs": xs,
        "N": int(params["N"]),
        "spacing": float(params["spacing"]),
        "mean_Gl": float(np.mean(gl)) if gl else 0.0,
        "friction": str(meta.get("friction_model", "zvb_rec")),
        "rho_g": rho_g,
        "Nx_ref": int(meta.get("Nx_ref", 0)),
        "T_total": float(meta.get("T_total", t[-1] - t[0])),
        "period_4L_a": float(meta.get("period_4L_a", 4.0 * well["L"] / well["a"])),
    }


def replay_one(case_row: dict, out_root: Path, audit: RunAudit) -> dict:
    cid = case_row["case_id"]
    src = Path(case_row["file"])
    case_dir = out_root / "cases" / cid
    case_dir.mkdir(parents=True, exist_ok=True)
    data = load_pilot_npz(src)

    audit.log(
        f"{cid}: N={data['N']} S={data['spacing']:.2f}m L={data['L']:.1f}m "
        f"a={data['a']:.2f}m/s dt={data['dt']:.4g}s T={data['T_total']:.1f}s "
        f"src={src.name}"
    )
    cep_path = str(case_dir / "cepstrum_standard.png")
    zoom_path = str(case_dir / "cepstrum_fracture_zoom.png")
    label = f"{cid} N={data['N']} S={data['spacing']:.1f}m"

    cep_result, cep_1d, cep_2d_avg = run_cepstrum_analysis_and_match(
        data["t"], data["H_wh"],
        a_adj=data["a"],
        L=data["L"],
        ts=data["t_s"],
        dt=data["dt"],
        x_f_list=data["xs"],
        x_f_plot=data["xs"],
        friction=data["friction"],
        label=label,
        kleak=data["mean_Gl"],
        cep_path=cep_path,
        cep_zoom_path=zoom_path,
    )
    plt.close("all")

    cep_block = cepstrum_block_for_json(cep_result, cep_1d, cep_2d_avg)
    rec = {
        "replay": True,
        "no_moc_rerun": True,
        "operator": "moc_simulate.leakoff_multi.run_cepstrum_analysis_and_match",
        "cepstrum_config": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in CEPSTRUM_CONFIG.items()},
        "case_id": cid,
        "source_npz": str(src),
        "source_sha256": sha256_file(src),
        "run_id": audit.run_id,
        "config": {
            "L": data["L"],
            "a": data["a"],
            "V0": float(data["params"]["well"]["Q0"]) / (np.pi * 0.25 * float(data["params"]["well"]["D"]) ** 2),
            "Q0": float(data["params"]["well"]["Q0"]),
            "D": float(data["params"]["well"]["D"]),
            "t_s": data["t_s"],
            "t_c": data["t_c"],
            "dt": data["dt"],
            "T_total": data["T_total"],
            "period_4L_a": data["period_4L_a"],
            "x_f": data["xs"],
            "N": data["N"],
            "spacing": data["spacing"],
            "mean_Gl": data["mean_Gl"],
            "friction": data["friction"],
            "Nx_ref": data["Nx_ref"],
            "rho_g": data["rho_g"],
            "kleak_plot_field_is_mean_Gl": True,
        },
        "cepstrum": cep_block,
        "files": {
            "cepstrum_standard_png": cep_path,
            "cepstrum_fracture_zoom_png": zoom_path,
        },
    }
    dump_json(rec, case_dir / "cepstrum.json")
    return rec


def _row_for_table(rec: dict) -> dict:
    c1 = rec["cepstrum"]["1d_real"]
    c2 = rec["cepstrum"]["2d_time_avg"]
    cfg = rec["config"]
    return {
        "case_id": rec["case_id"],
        "N": cfg["N"],
        "spacing_m": cfg["spacing"],
        "L_m": cfg["L"],
        "a_mps": cfg["a"],
        "t_c_s": cfg["t_c"],
        "1d_n_matched": c1.get("n_matched"),
        "1d_n_fracs": c1.get("n_fracs"),
        "1d_n_detected": c1.get("n_detected"),
        "1d_recall": c1.get("recall"),
        "1d_precision": c1.get("precision"),
        "1d_f1": c1.get("f1"),
        "1d_mean_error_m": c1.get("mean_error_m"),
        "1d_separation_success": c1.get("separation_success"),
        "2d_n_matched": c2.get("n_matched"),
        "2d_n_fracs": c2.get("n_fracs"),
        "2d_n_detected": c2.get("n_detected"),
        "2d_recall": c2.get("recall"),
        "2d_precision": c2.get("precision"),
        "2d_f1": c2.get("f1"),
        "2d_mean_error_m": c2.get("mean_error_m"),
        "2d_separation_success": c2.get("separation_success"),
        "match_tol_m": c1.get("match_tol_m"),
    }


def write_summary_table(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_summary(records: List[dict], path: Path, run_id: str) -> None:
    ids = [r["case_id"].replace("pilot_", "") for r in records]
    n_frac = np.array([r["cepstrum"]["1d_real"]["n_fracs"] for r in records], dtype=float)
    m1 = np.array([r["cepstrum"]["1d_real"]["n_matched"] for r in records], dtype=float)
    m2 = np.array([r["cepstrum"]["2d_time_avg"]["n_matched"] for r in records], dtype=float)
    x = np.arange(len(records))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    w = 0.36
    ax.bar(x - w / 2, m1 / np.maximum(n_frac, 1.0), w, label="1D real", color="#1f4e79")
    ax.bar(x + w / 2, m2 / np.maximum(n_frac, 1.0), w, label="2D time-avg", color="#c45911")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("recall  (matched / N)")
    ax.set_xlabel("pilot case")
    ax.set_title("(a)  Blind match recall  (tol = 10 m)")
    ax.legend(frameon=False, loc="upper right")
    ax.axhline(1.0, color="0.7", lw=0.6, zorder=0)

    ax = axes[1]
    for r in records:
        matches = r["cepstrum"]["1d_real"].get("matches") or []
        tt = [m["true_depth_m"] for m in matches if m.get("matched") and m.get("peak_depth_m") is not None]
        pp = [m["peak_depth_m"] for m in matches if m.get("matched") and m.get("peak_depth_m") is not None]
        if tt:
            ax.scatter(tt, pp, s=22, alpha=0.85)
    all_d = []
    for r in records:
        all_d.extend(r["config"]["x_f"])
        for m in r["cepstrum"]["1d_real"].get("matches") or []:
            if m.get("peak_depth_m") is not None:
                all_d.append(m["peak_depth_m"])
    if all_d:
        lo, hi = min(all_d), max(all_d)
        pad = 0.04 * (hi - lo + 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", lw=0.7, zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("true cluster depth (m)")
    ax.set_ylabel("1D matched peak (m)")
    ax.set_title("(b)  1D matched depths")
    ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"Pilot 2000 cepstrum replay  (leakoff_multi --replay operator)\n{run_id}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay leakoff cepstrum on random Pilot 2000 cases")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--seed", type=int, default=SUBSET_SEED_DEFAULT)
    ap.add_argument("--case-ids", nargs="*", default=None,
                    help="optional explicit case_id list; overrides --n/--seed")
    args = ap.parse_args()

    ensure_dirs()
    man_path = MANIFEST_DIR / "manifest_v1.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    selected = select_cases(manifest, n=args.n, seed=args.seed, case_ids=args.case_ids)

    resolved = {
        "mode": "replay_cepstrum_pilot",
        "n": len(selected),
        "seed": None if args.case_ids else int(args.seed),
        "case_ids": [r["case_id"] for r in selected],
        "pilot_manifest": str(man_path),
        "pilot_generator_run_id": manifest.get("generator", {}).get("run_id"),
        "cepstrum_config": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in CEPSTRUM_CONFIG.items()},
        "analysis_start_s": PHYSICS_T_S,
        "operator": "leakoff_multi.run_cepstrum_analysis_and_match",
        "no_moc_rerun": True,
    }
    audit = RunAudit(
        "s1_cepstrum_pilot10",
        seed=int(args.seed),
        resolved_config=resolved,
        manifest_digest=str(manifest.get("digest") or ""),
    )
    out_root = CEPSTRUM_PILOT_DIR / audit.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    audit.log(f"writing under {out_root}")
    audit.log("cases: " + ", ".join(r["case_id"] for r in selected))

    records = []
    try:
        for row in selected:
            records.append(replay_one(row, out_root, audit))
        table_rows = [_row_for_table(r) for r in records]
        write_summary_table(table_rows, out_root / "summary.csv")
        dump_json({"run_id": audit.run_id, "cases": table_rows, "resolved": resolved},
                  out_root / "summary.json")
        dump_json({"run_id": audit.run_id, "cases": table_rows, "resolved": resolved},
                  TABLE_DIR / "table_cepstrum_pilot10.json")
        plot_summary(records, out_root / "summary_match.png", audit.run_id)
        metrics = {
            "stage": 1,
            "task": "cepstrum_replay_pilot10",
            "n_cases": len(records),
            "seed": None if args.case_ids else int(args.seed),
            "case_ids": [r["case_id"] for r in records],
            "mean_1d_recall": float(np.mean([r["cepstrum"]["1d_real"]["recall"] or 0.0 for r in records])),
            "mean_2d_recall": float(np.mean([r["cepstrum"]["2d_time_avg"]["recall"] or 0.0 for r in records])),
            "n_1d_separation_success": int(sum(bool(r["cepstrum"]["1d_real"].get("separation_success")) for r in records)),
            "n_2d_separation_success": int(sum(bool(r["cepstrum"]["2d_time_avg"].get("separation_success")) for r in records)),
            "output_dir": str(out_root),
            "not_stage1_formal_acceptance": True,
        }
        audit.write_metrics(metrics, also_to=out_root / "metrics.json")
        audit.log("---- summary ----")
        for r in records:
            a = r["cepstrum"]["1d_real"]
            b = r["cepstrum"]["2d_time_avg"]
            audit.log(
                f"  {r['case_id']}: 1D {a['n_matched']}/{a['n_fracs']} "
                f"(F1={a.get('f1')}) | 2Davg {b['n_matched']}/{b['n_fracs']} "
                f"(F1={b.get('f1')})"
            )
        audit.log(f"CSV {out_root / 'summary.csv'}")
        audit.log(f"summary figure {out_root / 'summary_match.png'}")
    finally:
        audit.close()
    return 0


if __name__ == "__main__":
    # Keep Paper C code dir first so `paths` / `audit` resolve locally.
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    raise SystemExit(main())
