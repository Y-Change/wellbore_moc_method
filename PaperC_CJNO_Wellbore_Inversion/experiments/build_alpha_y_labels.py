#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 moc_v2_physical_steady_1k_newa HDF5 构造 CJ-AlphaNet 标签。

写出（不覆盖原始波形 h5）：
  data/datasets/moc_v2_physical_steady_1k_newa/case_labels_alpha_y.csv
  data/datasets/moc_v2_physical_steady_1k_newa/case_labels_alpha_y.npz
  data/datasets/moc_v2_physical_steady_1k_newa/case_labels_alpha_y_meta.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PaperC_CJNO_Wellbore_Inversion.src.label_physics import (
    ALPHA_ACTIVE_THR,
    A_HAT_MAX,
    A_HAT_MIN,
    L_DEFAULT,
    N_GRID_DEFAULT,
    POST_SHUT_DURATION_S,
    SIGMA_M_DEFAULT,
    Y_EQ_FLOOR,
    build_sample_labels,
)

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 h5py") from exc

DEFAULT_H5 = _ROOT / "data" / "datasets" / "moc_v2_physical_steady_1k_newa" / "moc_v2_physical_steady_1k_x4500_4950.h5"
DEFAULT_OUT_DIR = _ROOT / "data" / "datasets" / "moc_v2_physical_steady_1k_newa"
MAX_FRAC = 6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构造 CJ-AlphaNet 进液/活动簇/等效导纳标签")
    p.add_argument("--h5", type=str, default=str(DEFAULT_H5))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--ts", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=POST_SHUT_DURATION_S)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def _pipe(values: Any, n: int) -> str:
    parts = []
    for v in list(values)[:n]:
        parts.append(str(v))
    return "|".join(parts)


def _label_array(grp, preferred: str, fallback: str, n: int, dim: int) -> np.ndarray:
    if preferred in grp:
        return np.asarray(grp[preferred][:n, :dim], dtype=np.float64)
    return np.asarray(grp[fallback][:n, :dim], dtype=np.float64)


def main() -> None:
    args = parse_args()
    h5_path = os.path.abspath(args.h5)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"找不到 HDF5: {h5_path}")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "case_labels_alpha_y.csv")
    npz_path = os.path.join(out_dir, "case_labels_alpha_y.npz")
    meta_path = os.path.join(out_dir, "case_labels_alpha_y_meta.json")

    print(f"[*] 读取 {h5_path}")
    t0 = time.time()
    with h5py.File(h5_path, "r") as h5:
        head = np.asarray(h5["waveforms/wellhead_head"][:], dtype=np.float32)
        timestamps = np.asarray(h5["waveforms/timestamps"][:], dtype=np.float64)
        lab = h5["labels"]
        n_total = int(head.shape[0])
        n_use = n_total if args.max_samples is None else min(n_total, int(args.max_samples))
        n_frac = np.asarray(lab["n_frac"][:n_use], dtype=np.int32)
        pos = np.asarray(lab["fracture_positions"][:n_use], dtype=np.float64)
        cf = np.asarray(lab["fracture_Cf"][:n_use], dtype=np.float64)
        kleak = np.asarray(lab["fracture_kleak"][:n_use], dtype=np.float64)
        kp = np.asarray(lab["fracture_Kp"][:n_use], dtype=np.float64)
        alpha = _label_array(lab, "fracture_alpha_ss", "fracture_weights", n_use, pos.shape[1])
        q_ss = _label_array(lab, "fracture_Q_ss", "fracture_weights", n_use, pos.shape[1])
        if "fracture_Q_ss" not in lab:
            raise KeyError("HDF5 缺少 labels/fracture_Q_ss，无法构造 Y_eq")

    max_dim = int(pos.shape[1])
    m = MAX_FRAC
    positions_pad = np.zeros((n_use, m), dtype=np.float32)
    alpha_pad = np.zeros((n_use, m), dtype=np.float32)
    y_act_pad = np.zeros((n_use, m), dtype=np.int32)
    yeq_pad = np.zeros((n_use, m), dtype=np.float32)
    logy_pad = np.zeros((n_use, m), dtype=np.float32)
    mask_pad = np.zeros((n_use, m), dtype=np.int32)
    q_pad = np.zeros((n_use, m), dtype=np.float32)
    cf_pad = np.zeros((n_use, m), dtype=np.float32)
    k_pad = np.zeros((n_use, m), dtype=np.float32)
    kp_pad = np.zeros((n_use, m), dtype=np.float32)
    x1_arr = np.zeros(n_use, dtype=np.float32)
    t1_arr = np.zeros(n_use, dtype=np.float32)
    ahat_arr = np.zeros(n_use, dtype=np.float32)
    ok_arr = np.zeros(n_use, dtype=np.int32)
    y0_arr = np.zeros(n_use, dtype=np.float32)
    omega_arr = np.zeros(n_use, dtype=np.float32)
    nfrac_arr = np.zeros(n_use, dtype=np.int32)
    m_alpha = np.zeros((n_use, N_GRID_DEFAULT), dtype=np.float32)
    grid_x = None

    fieldnames = [
        "sample_id",
        "n_frac",
        "x1_m",
        "T1_s",
        "a_hat_m_s",
        "a_hat_ok",
        "Y0_m2_s",
        "omega_star_rad_s",
        "fracture_positions_m",
        "mask_design",
        "y_active",
        "alpha_ss",
        "Y_eq_m2_s",
        "log10_Y_over_Y0",
        "fracture_Q_ss_m3_s",
        "fracture_Cf_m2",
        "fracture_kleak_m2p5_s",
        "fracture_Kp_s2_m5",
    ]

    rows: List[Dict[str, Any]] = []
    for i in range(n_use):
        nf = int(min(int(n_frac[i]), m, max_dim))
        nf = max(nf, 0)
        sl = slice(0, nf)
        labels = build_sample_labels(
            timestamps=timestamps,
            wellhead_head=head[i],
            positions=pos[i, sl],
            alpha=alpha[i, sl],
            q_ss=q_ss[i, sl],
            kleak=kleak[i, sl],
            kp=kp[i, sl],
            cf=cf[i, sl],
            ts=float(args.ts),
            duration=float(args.duration),
        )
        if grid_x is None:
            grid_x = labels["grid_x"]
        nfrac_arr[i] = nf
        if nf > 0:
            positions_pad[i, :nf] = labels["positions"]
            alpha_pad[i, :nf] = labels["alpha"]
            y_act_pad[i, :nf] = labels["y_active"]
            yeq_pad[i, :nf] = labels["Y_eq"]
            logy_pad[i, :nf] = labels["log10_Y_over_Y0"]
            mask_pad[i, :nf] = 1
            q_pad[i, :nf] = q_ss[i, sl]
            cf_pad[i, :nf] = cf[i, sl]
            k_pad[i, :nf] = kleak[i, sl]
            kp_pad[i, :nf] = kp[i, sl]
        x1_arr[i] = labels["x1"]
        t1_arr[i] = labels["T1"]
        ahat_arr[i] = labels["a_hat"]
        ok_arr[i] = labels["a_hat_ok"]
        y0_arr[i] = labels["Y0"]
        omega_arr[i] = labels["omega_star"]
        m_alpha[i] = labels["m_alpha_grid"]

        rows.append(
            {
                "sample_id": i,
                "n_frac": nf,
                "x1_m": float(x1_arr[i]),
                "T1_s": float(t1_arr[i]),
                "a_hat_m_s": float(ahat_arr[i]),
                "a_hat_ok": int(ok_arr[i]),
                "Y0_m2_s": float(y0_arr[i]),
                "omega_star_rad_s": float(omega_arr[i]),
                "fracture_positions_m": _pipe(positions_pad[i], nf),
                "mask_design": _pipe(mask_pad[i], nf),
                "y_active": _pipe(y_act_pad[i], nf),
                "alpha_ss": _pipe(alpha_pad[i], nf),
                "Y_eq_m2_s": _pipe(yeq_pad[i], nf),
                "log10_Y_over_Y0": _pipe(logy_pad[i], nf),
                "fracture_Q_ss_m3_s": _pipe(q_pad[i], nf),
                "fracture_Cf_m2": _pipe(cf_pad[i], nf),
                "fracture_kleak_m2p5_s": _pipe(k_pad[i], nf),
                "fracture_Kp_s2_m5": _pipe(kp_pad[i], nf),
            }
        )
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    样本 {i + 1}/{n_use}")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        npz_path,
        sample_id=np.arange(n_use, dtype=np.int32),
        n_frac=nfrac_arr,
        positions=positions_pad,
        mask_design=mask_pad,
        alpha=alpha_pad,
        y_active=y_act_pad,
        Y_eq=yeq_pad,
        log10_Y_over_Y0=logy_pad,
        Q_ss=q_pad,
        Cf=cf_pad,
        kleak=k_pad,
        Kp=kp_pad,
        x1=x1_arr,
        T1=t1_arr,
        a_hat=ahat_arr,
        a_hat_ok=ok_arr,
        Y0=y0_arr,
        omega_star=omega_arr,
        m_alpha_grid=m_alpha,
        grid_x=np.asarray(grid_x, dtype=np.float32),
    )

    n_ok = int(np.sum(ok_arr))
    n_act = int(np.sum(y_act_pad))
    n_design = int(np.sum(mask_pad))
    meta = {
        "source_h5": h5_path,
        "n_samples": n_use,
        "formulas": {
            "T1": "prominent autocorr peak of post-shut dH/dt in [2*x1/1600, 2*x1/1200] s, else |dH/dt| first-echo in the same window",
            "a_hat": "2*x1/T1 clipped to [1200, 1600] m/s; design wavespeed unused",
            "Y_eq": "Re 1/(R_perf + 1/(G_leak + j*omega*Cf)), R=2*Kp*|q|, G=k^2/(2|q|)",
            "omega_star": "2*pi/T1",
            "y_active": f"mask_design AND alpha>= {ALPHA_ACTIVE_THR}",
            "m_alpha": f"Gaussian sigma={SIGMA_M_DEFAULT} m on [0,{L_DEFAULT}] n={N_GRID_DEFAULT}, integral 1",
        },
        "constants": {
            "alpha_active_threshold": ALPHA_ACTIVE_THR,
            "a_hat_min": A_HAT_MIN,
            "a_hat_max": A_HAT_MAX,
            "Y_eq_floor": Y_EQ_FLOOR,
            "post_shut_duration_s": float(args.duration),
            "ts_s": float(args.ts),
        },
        "statistics": {
            "a_hat_ok_count": n_ok,
            "a_hat_ok_fraction": round(n_ok / max(n_use, 1), 4),
            "a_hat_min": float(np.min(ahat_arr)) if n_use else None,
            "a_hat_median": float(np.median(ahat_arr)) if n_use else None,
            "a_hat_max": float(np.max(ahat_arr)) if n_use else None,
            "T1_median_s": float(np.median(t1_arr)) if n_use else None,
            "designed_clusters": n_design,
            "active_clusters": n_act,
            "active_fraction_among_designed": round(n_act / max(n_design, 1), 4),
            "Y_eq_active_median": float(np.median(yeq_pad[y_act_pad == 1])) if n_act else None,
            "Y_eq_inactive_designed_median": float(
                np.median(yeq_pad[(mask_pad == 1) & (y_act_pad == 0)])
            )
            if int(np.sum((mask_pad == 1) & (y_act_pad == 0)))
            else None,
        },
        "outputs": {
            "csv": os.path.abspath(csv_path),
            "npz": os.path.abspath(npz_path),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[+] CSV  {csv_path}")
    print(f"[+] NPZ  {npz_path}")
    print(f"[+] meta {meta_path}")
    print(
        f"[+] n={n_use}  a_hat ok={n_ok}/{n_use}  "
        f"median a_hat={meta['statistics']['a_hat_median']:.1f} m/s  "
        f"active={n_act}/{n_design}"
    )
    print(f"[+] elapsed {meta['elapsed_s']:.1f} s (original h5 unchanged)")


if __name__ == "__main__":
    main()
