# -*- coding: utf-8 -*-
"""为 PaperA 归档补全 leakoff 格式 moc_timeseries.csv（t,H_wh,Q_wh,H_f*,Q_f*）。"""
from __future__ import annotations

import csv
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

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

from moc_simulate.config import FRACTURE_CONFIG, SIM_CONFIG, WELL_CONFIG
from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore

PAPERA = next(
    p
    for p in Path(r"e:\water_hammer_research\wellbore_moc_method\output").iterdir()
    if p.is_dir() and p.name.startswith("PaperA")
)
N_WORKERS = 12
DEFAULT_CF = float(FRACTURE_CONFIG["Cf"])
DEFAULT_KLEAK = float(FRACTURE_CONFIG["kleak"])
H_EXT = float(FRACTURE_CONFIG["H_ext"])
CASE_N = {
    "single": 1,
    "dual": 2,
    "triple": 3,
    "quad": 4,
    "quint": 5,
    "hex": 6,
    "hept": 7,
    "oct": 8,
}


def csv_complete(path: Path, n_frac: int) -> bool:
    if not path.is_file() or path.stat().st_size < 100:
        return False
    with path.open(encoding="utf-8") as f:
        hdr = f.readline().strip().split(",")
    need = ["t", "H_wh", "Q_wh"]
    for i in range(1, n_frac + 1):
        need.extend([f"H_f{i}", f"Q_f{i}"])
    return all(c in hdr for c in need)


def save_timeseries(path: Path, res: dict) -> None:
    t = np.asarray(res["timestamps"])
    H_wh = np.asarray(res["wellhead_head"])
    Q_wh = np.asarray(res["wellhead_velocity"]) * float(res["cfg"].area)
    frac_H = np.asarray(res["fracture_heads"])
    frac_Q = np.asarray(res["fracture_Qs"])
    n_frac = int(frac_H.shape[1])
    cols = [t, H_wh, Q_wh]
    header = ["t", "H_wh", "Q_wh"]
    for k in range(n_frac):
        cols.append(frac_H[:, k])
        cols.append(frac_Q[:, k])
        header.append(f"H_f{k + 1}")
        header.append(f"Q_f{k + 1}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        str(path),
        np.column_stack(cols),
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def build_cfg(friction_model: str, L_required: float) -> MocConfig:
    w, s = WELL_CONFIG, SIM_CONFIG
    L = max(float(w["L"]), float(L_required))
    tf = max(float(s["tf"]), 2.0 * L / float(w["wavespeed"]) + 1.0)
    return MocConfig(
        wellbore_length=L,
        wellbore_diameter=w["wellbore_diameter"],
        fluid_density=w["fluid_density"],
        fluid_viscosity=w["fluid_viscosity"],
        wavespeed=w["wavespeed"],
        roughness_height=w["roughness_height"],
        friction_model=friction_model,
        dt=s["dt"],
        tf=tf,
        wellhead_bc="velocity_step",
        pump_shut_time=s["ts"],
        initial_velocity=w["V0"],
        initial_head=w["H0"],
        theta=w["theta"],
        toe_bc="reservoir",
        toe_head=w["H0"],
    )


def run_job(job: dict) -> dict:
    import moc_simulate.wellbore_moc as wm

    out = Path(job["out"])
    n_frac = len(job["x_f"])
    if csv_complete(out, n_frac):
        return {"out": job["out"], "status": "skip", "sec": 0.0, "err": ""}
    orig_k, orig_kv = wm.brunone_k, wm.brunone_k_vec
    t0 = time.time()
    try:
        k_const = job.get("k_const")
        fric = job["friction"]
        if k_const is not None and float(k_const) == 0.0:
            fric = "steady"
        if k_const is not None and float(k_const) > 0.0:
            k = float(k_const)
            wm.brunone_k = lambda Re, _k=k: _k
            wm.brunone_k_vec = lambda Re, _k=k: np.full_like(
                np.asarray(Re, dtype=np.float64), _k, dtype=np.float64
            )
            fric = "brunone"
        n_frac = len(job["x_f"])
        Cf = job["Cf"]
        kleak = job["kleak"]
        if not isinstance(Cf, list):
            Cf = [float(Cf)] * n_frac
        if not isinstance(kleak, list):
            kleak = [float(kleak)] * n_frac
        cfg = build_cfg(fric, job["L_required"])
        res = simulate_wellbore(
            cfg,
            fracture_positions=list(job["x_f"]),
            fracture_Cf=list(Cf),
            fracture_kleak=list(kleak),
            H_ext=H_EXT,
            store_full_field=False,
        )
        save_timeseries(out, res)
        for extra in job.get("replicate_to") or []:
            dst = Path(extra)
            if dst.resolve() == out.resolve():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, dst)
    except Exception as exc:  # noqa: BLE001
        return {"out": job["out"], "status": "fail", "sec": time.time() - t0, "err": str(exc)}
    finally:
        wm.brunone_k = orig_k
        wm.brunone_k_vec = orig_kv
    return {"out": job["out"], "status": "ok", "sec": time.time() - t0, "err": ""}


def copy_if_needed(src: Path, dst: Path, n_frac: int) -> str:
    if csv_complete(dst, n_frac):
        return "skip"
    if not csv_complete(src, n_frac):
        return "src_incomplete"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def collect_jobs() -> tuple[list[dict], int]:
    jobs: list[dict] = []
    n_copy = 0
    leak = PAPERA / "03_leakoff验证"

    # 03: brunone 无后缀缺 csv → 从 brunone_D50 复制
    for case, n in CASE_N.items():
        src = leak / "brunone_D50" / case / "moc_timeseries.csv"
        dst = leak / "brunone" / case / "moc_timeseries.csv"
        if src.is_file():
            st = copy_if_needed(src, dst, n)
            if st == "copied":
                n_copy += 1

    # 04 energy 几何块：从 leakoff 复制
    ener = PAPERA / "04_能量回归" / "工况"
    for fr in ("steady", "brunone"):
        for D in (5, 10, 20, 50, 100):
            for case, n in list(CASE_N.items())[:5]:
                src = leak / f"{fr}_D{D}" / case / "moc_timeseries.csv"
                dst = ener / fr / f"{case}_D{D}" / "moc_timeseries.csv"
                st = copy_if_needed(src, dst, n)
                if st == "copied":
                    n_copy += 1

    # 04 energy 属性块：重跑
    et = PAPERA / "04_能量回归" / "energy_table.csv"
    with et.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cf, kl = float(r["Cf"]), float(r["kleak"])
            is_default = abs(cf - DEFAULT_CF) < 1e-18 and abs(kl - DEFAULT_KLEAK) < 1e-18
            if is_default:
                continue
            fr = r["friction"]
            n = int(r["n_fracs"])
            D = float(r["spacing_m"])
            x1 = 4100.0
            x_f = [x1 + i * D for i in range(n)]
            out = (
                ener
                / fr
                / f"{r['case']}_cf{cf:.1e}_kl{kl:.1e}"
                / "moc_timeseries.csv"
            )
            if csv_complete(out, n):
                continue
            jobs.append(
                {
                    "out": str(out),
                    "friction": fr,
                    "x_f": x_f,
                    "Cf": cf,
                    "kleak": kl,
                    "L_required": max(x_f) + 500.0,
                    "k_const": None,
                    "replicate_to": [],
                }
            )

    # 05 Brunone 常数 k：缺 H_f 列则重跑
    bru = PAPERA / "05_Brunone常数k" / "工况"
    kpat = re.compile(r"^D(\d+)_k([0-9.]+)$")
    for ddir in sorted(bru.iterdir()):
        m = kpat.match(ddir.name)
        if not m:
            continue
        D = float(m.group(1))
        k = float(m.group(2))
        n = 4
        x_f = [4100.0 + i * D for i in range(n)]
        out = ddir / "moc_timeseries.csv"
        if csv_complete(out, n):
            continue
        jobs.append(
            {
                "out": str(out),
                "friction": "steady" if k == 0.0 else "brunone",
                "x_f": x_f,
                "Cf": DEFAULT_CF,
                "kleak": DEFAULT_KLEAK,
                "L_required": max(x_f) + 500.0,
                "k_const": k,
                "replicate_to": [],
            }
        )

    # 02 Cf×kleak
    cf_csv = PAPERA / "02_裂缝属性_CfKleak" / "cf_kleak_table.csv"
    wave2 = PAPERA / "02_裂缝属性_CfKleak" / "波形"
    seen = set()
    with cf_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["frac_idx"]) != 1:
                continue
            fr = r["friction_model"]
            cf, kl = float(r["Cf"]), float(r["Kleak"])
            n = int(float(r["n_total"]))
            key = (fr, cf, kl, n)
            if key in seen:
                continue
            seen.add(key)
            x1 = float(r["x1"])
            D = float(r["spacing_m"])
            x_f = [x1 + i * D for i in range(n)]
            stem = f"{fr}_cf_{cf:.1e}_kleak_{kl:.1e}"
            out = wave2 / stem / "moc_timeseries.csv"
            if csv_complete(out, n):
                continue
            jobs.append(
                {
                    "out": str(out),
                    "friction": fr,
                    "x_f": x_f,
                    "Cf": cf,
                    "kleak": kl,
                    "L_required": max(x_f) + 500.0,
                    "k_const": None,
                    "replicate_to": [],
                }
            )

    # 01 几何网格：按 npz 清单补 csv；n=1 每 (摩阻,x1) 只跑一次
    geo_npz = PAPERA / "01_几何网格" / "波形_npz"
    geo_csv = PAPERA / "01_几何网格" / "波形"
    pat = re.compile(r"(steady|brunone)_x1_(\d+)_sp_(\d+)_n_(\d+)\.npz")
    n1_groups: dict[tuple, list[tuple[int, Path]]] = {}
    for f in sorted(geo_npz.glob("*.npz")):
        m = pat.match(f.name)
        if not m:
            continue
        fr, x1, D, n = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        stem = f"{fr}_x1_{x1}_sp_{D}_n_{n}"
        out = geo_csv / stem / "moc_timeseries.csv"
        x_f = [float(x1) + i * float(D) for i in range(n)]
        if n == 1:
            n1_groups.setdefault((fr, x1), []).append((D, out))
            continue
        if csv_complete(out, n):
            continue
        jobs.append(
            {
                "out": str(out),
                "friction": fr,
                "x_f": x_f,
                "Cf": DEFAULT_CF,
                "kleak": DEFAULT_KLEAK,
                "L_required": max(x_f) + 500.0,
                "k_const": None,
                "replicate_to": [],
            }
        )

    for (fr, x1), items in n1_groups.items():
        items.sort()
        outs = [p for _, p in items]
        if all(csv_complete(p, 1) for p in outs):
            continue
        canon = next((p for p in outs if csv_complete(p, 1)), None)
        if canon is not None:
            for p in outs:
                if p != canon:
                    st = copy_if_needed(canon, p, 1)
                    if st == "copied":
                        n_copy += 1
            continue
        # 选 D=10 作 canonical
        canon_out = next((p for D, p in items if D == 10), outs[0])
        jobs.append(
            {
                "out": str(canon_out),
                "friction": fr,
                "x_f": [float(x1)],
                "Cf": DEFAULT_CF,
                "kleak": DEFAULT_KLEAK,
                "L_required": float(x1) + 500.0,
                "k_const": None,
                "replicate_to": [str(p) for p in outs if p != canon_out],
            }
        )

    return jobs, n_copy


def main() -> None:
    jobs, n_copy = collect_jobs()
    print(f"PAPERA={PAPERA}")
    print(f"copied={n_copy}  to_simulate={len(jobs)}  workers={N_WORKERS}")
    if not jobs:
        print("nothing to simulate")
        return
    ok = skip = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_job, job): job for job in jobs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            rec = fut.result()
            if rec["status"] == "ok":
                ok += 1
            elif rec["status"] == "skip":
                skip += 1
            else:
                fail += 1
                print(f"FAIL {rec['out']}: {rec['err']}")
            if done % 20 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                print(
                    f"  {done}/{len(jobs)} ok={ok} skip={skip} fail={fail} "
                    f"elapsed={elapsed/60:.1f} min last={rec['sec']:.1f}s"
                )
    print(f"Done. ok={ok} skip={skip} fail={fail} copied_before={n_copy}")


if __name__ == "__main__":
    main()
