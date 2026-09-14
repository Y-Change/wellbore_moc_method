# -*- coding: utf-8 -*-
"""
stratified_bench.py — 分层（间距格子）+ 含噪（后处理 SNR）基准集核心逻辑。

与 ``run_lhs_batch_simulate.generate_lhs_params`` 的连续间距采样不同：
本模块按固定间距格子分层，补齐 20–50 m 空白，并为 EXP-021 / EXP-011 S2
提供统一输入。噪声不进 MOC，后处理加到停泵后 ``H_wh``。
"""
from __future__ import annotations

import csv
import json
import os
import time as time_module
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from moc_simulate.config import FRACTURE_CONFIG, WELL_CONFIG
from moc_simulate.lhs_config import BENCH_STRATIFIED_CONFIG, SIM_CONFIG
from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore

__all__ = [
    "generate_stratified_params",
    "apply_awgn",
    "snr_tag",
    "noise_seed_for",
    "load_clean_case",
    "materialize_noisy_case",
    "simulate_one_case",
    "write_manifest",
    "summarize_cells",
]


def snr_tag(snr_db: Optional[float]) -> str:
    """将 SNR 水平编码为文件名/manifest 友好标签。"""
    if snr_db is None:
        return "inf"
    return f"{float(snr_db):g}"


def noise_seed_for(base_seed: int, case_id: int, snr_db: Optional[float]) -> int:
    """确定性噪声种子：同一 (case, snr) 永远复现同一实现。"""
    tag = 0 if snr_db is None else int(round(float(snr_db) * 10))
    return int(base_seed) + int(case_id) * 1000 + tag


def generate_stratified_params(
    spacing_grid_m: Optional[Sequence[float]] = None,
    n_per_spacing: Optional[int] = None,
    n_frac: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """按固定间距格子生成清洁样本参数（每格 ``n_per_spacing`` 条）。

    相邻缝间距严格等于该格子的间距值；首缝位置在缝网区内均匀随机，
    保证整条缝链落入 ``[frac_zone_start, frac_zone_end]``。
    Cf / kleak 仍对数均匀采样。
    """
    cfg = BENCH_STRATIFIED_CONFIG
    spacing_grid = tuple(
        float(x) for x in (spacing_grid_m or cfg["spacing_grid_m"])
    )
    n_per = int(n_per_spacing if n_per_spacing is not None else cfg["n_per_spacing"])
    n_cl = int(n_frac if n_frac is not None else cfg["n_frac"])
    seed_used = int(seed if seed is not None else cfg["seed"])
    if n_per < 1:
        raise ValueError("n_per_spacing must be >= 1")
    if n_cl < 1:
        raise ValueError("n_frac must be >= 1")

    z_start = float(cfg["frac_zone_start"])
    z_end = float(cfg["frac_zone_end"])
    zone = z_end - z_start
    cf_lmin, cf_lmax = float(cfg["cf_log_min"]), float(cfg["cf_log_max"])
    kl_lmin, kl_lmax = float(cfg["kleak_log_min"]), float(cfg["kleak_log_max"])

    rng = np.random.default_rng(seed_used)
    samples: List[Dict[str, Any]] = []
    case_id = 0

    for spacing in spacing_grid:
        chain_len = float(spacing) * (n_cl - 1)
        if chain_len > zone + 1e-9:
            raise ValueError(
                f"n_frac={n_cl} with spacing={spacing} m cannot fit in "
                f"zone [{z_start}, {z_end}] m (chain_len={chain_len})"
            )
        start_span = zone - chain_len
        for _ in range(n_per):
            u = rng.random(size=1 + 2 * n_cl)
            x0 = float(z_start + float(u[0]) * start_span)
            positions = [x0 + float(spacing) * k for k in range(n_cl)]
            cf_vals = [
                float(10.0 ** (cf_lmin + float(u[1 + k]) * (cf_lmax - cf_lmin)))
                for k in range(n_cl)
            ]
            kleak_vals = [
                float(10.0 ** (kl_lmin + float(u[1 + n_cl + k]) * (kl_lmax - kl_lmin)))
                for k in range(n_cl)
            ]
            samples.append({
                "case_id": case_id,
                "spacing_m": float(spacing),
                "n_frac": n_cl,
                "positions": positions,
                "Cf_list": cf_vals,
                "kleak_list": kleak_vals,
            })
            case_id += 1
    return samples


def apply_awgn(
    signal: np.ndarray,
    snr_db: Optional[float],
    seed: int,
    *,
    ts: float = 1.0,
    dt: float = 1e-3,
    noise_segment: str = "post_shut_in",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """对信号加高斯白噪声；``snr_db=None`` 时原样返回。

    SNR 定义：``10·log10(P_signal / P_noise)``，其中 ``P_signal`` 为选定
    时间段上的方差（去均值）。默认段为停泵后全序列。
    """
    x = np.asarray(signal, dtype=np.float64)
    meta = {
        "snr_db": float("inf") if snr_db is None else float(snr_db),
        "noise_std": 0.0,
        "signal_power": 0.0,
    }
    if snr_db is None:
        return x.copy(), meta

    if noise_segment == "post_shut_in":
        i0 = max(0, int(round(ts / dt)))
        segment = x[i0:]
    elif noise_segment == "full":
        segment = x
    else:
        raise ValueError(f"unknown noise_segment={noise_segment!r}")

    if segment.size < 2:
        raise ValueError("signal segment too short for SNR estimation")

    signal_power = float(np.var(segment - np.mean(segment)))
    if signal_power <= 0.0:
        signal_power = float(np.mean(segment ** 2))
    if signal_power <= 0.0:
        raise ValueError("signal power is zero; cannot define SNR")

    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_std = float(np.sqrt(noise_power))
    rng = np.random.default_rng(int(seed))
    noisy = x + rng.normal(0.0, noise_std, size=x.shape)
    meta["noise_std"] = noise_std
    meta["signal_power"] = signal_power
    return noisy, meta


def load_clean_case(npz_path: str) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def materialize_noisy_case(
    clean_npz: str,
    out_npz: str,
    snr_db: Optional[float],
    seed: int,
    *,
    ts: float = 1.0,
) -> Dict[str, float]:
    """从清洁 npz 写出含噪副本（``H_wh`` 被替换；标签不变）。"""
    clean = load_clean_case(clean_npz)
    dt = float(np.median(np.diff(clean["t"]))) if len(clean["t"]) > 1 else 1e-3
    noisy_h, meta = apply_awgn(
        clean["H_wh"], snr_db, seed, ts=ts, dt=dt,
        noise_segment=BENCH_STRATIFIED_CONFIG["noise_segment"],
    )
    payload = dict(clean)
    payload["H_wh"] = noisy_h.astype(np.float64, copy=False)
    payload["snr_db"] = np.array(meta["snr_db"], dtype=np.float64)
    payload["noise_seed"] = np.array(int(seed), dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **payload)
    return meta


def simulate_one_case(
    sample: Dict[str, Any],
    *,
    friction_model: str,
    tf_cut: float,
    dt_sim: float,
    out_data_dir: str,
) -> Dict[str, Any]:
    """跑单条清洁仿真并落盘 ``case_XXXXX.npz``。"""
    case_id = int(sample["case_id"])
    positions = list(sample["positions"])
    Cf_list = list(sample["Cf_list"])
    kleak_list = list(sample["kleak_list"])
    n_cl = int(sample["n_frac"])
    spacing_m = float(sample["spacing_m"])

    t0 = time_module.time()
    try:
        w = WELL_CONFIG
        s = SIM_CONFIG
        fc = FRACTURE_CONFIG
        cfg = MocConfig(
            wellbore_length=w["L"],
            wellbore_diameter=w["wellbore_diameter"],
            fluid_density=w["fluid_density"],
            fluid_viscosity=w["fluid_viscosity"],
            wavespeed=w["wavespeed"],
            roughness_height=w["roughness_height"],
            friction_model=friction_model,
            dt=dt_sim,
            tf=tf_cut,
            wellhead_bc="velocity_step",
            pump_shut_time=s["ts"],
            initial_velocity=w["V0"],
            initial_head=w["H0"],
            theta=w["theta"],
            toe_bc="reservoir",
            toe_head=w["H0"],
        )
        res = simulate_wellbore(
            cfg,
            fracture_positions=positions,
            fracture_Cf=Cf_list,
            fracture_kleak=kleak_list,
            H_ext=fc["H_ext"],
            store_full_field=False,
        )
        t_arr = res["timestamps"]
        H_wh = res["wellhead_head"]
        V_wh = res["wellhead_velocity"]
        Q_wh = V_wh * cfg.area

        npz_filename = f"case_{case_id:05d}.npz"
        npz_path = os.path.join(out_data_dir, npz_filename)
        np.savez_compressed(
            npz_path,
            t=t_arr,
            H_wh=H_wh,
            Q_wh=Q_wh,
            x_f=np.array(positions, dtype=np.float32),
            Cf=np.array(Cf_list, dtype=np.float32),
            kleak=np.array(kleak_list, dtype=np.float32),
            n_frac=n_cl,
            spacing_m=np.float32(spacing_m),
            friction=str(friction_model),
            tf=float(tf_cut),
            snr_db=np.array(np.inf, dtype=np.float64),
        )
        return {
            "case_id": case_id,
            "status": "PASS",
            "spacing_m": spacing_m,
            "n_frac": n_cl,
            "positions_str": ";".join(f"{x:.1f}" for x in positions),
            "Cf_str": ";".join(f"{c:.2e}" for c in Cf_list),
            "kleak_str": ";".join(f"{k:.2e}" for k in kleak_list),
            "elapsed_s": round(time_module.time() - t0, 2),
            "npz_file": npz_filename,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 — worker 边界必须吞并上报
        return {
            "case_id": case_id,
            "status": "FAIL",
            "spacing_m": spacing_m,
            "n_frac": n_cl,
            "positions_str": "",
            "Cf_str": "",
            "kleak_str": "",
            "elapsed_s": round(time_module.time() - t0, 2),
            "npz_file": "",
            "error": str(exc),
        }


def _worker_entry(args: Tuple) -> Dict[str, Any]:
    sample, friction, tf_cut, dt_sim, out_data_dir = args
    return simulate_one_case(
        sample,
        friction_model=friction,
        tf_cut=tf_cut,
        dt_sim=dt_sim,
        out_data_dir=out_data_dir,
    )


def write_manifest(
    out_root: str,
    *,
    clean_results: Sequence[Dict[str, Any]],
    snr_levels: Sequence[Optional[float]],
    base_seed: int,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """写出清洁索引 + (spacing × snr) 条件格子 manifest。"""
    os.makedirs(out_root, exist_ok=True)
    clean_dir = os.path.join(out_root, "data_clean")

    # 清洁汇总
    clean_csv = os.path.join(out_root, "clean_summary.csv")
    fields = [
        "case_id", "status", "spacing_m", "n_frac", "positions_str",
        "Cf_str", "kleak_str", "elapsed_s", "npz_file", "error",
    ]
    rows = sorted(clean_results, key=lambda r: r["case_id"])
    with open(clean_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 条件格子：每个 PASS 清洁样本 × 每个 SNR
    cell_rows: List[Dict[str, Any]] = []
    for row in rows:
        if row["status"] != "PASS":
            continue
        case_id = int(row["case_id"])
        for snr in snr_levels:
            tag = snr_tag(snr)
            seed = noise_seed_for(base_seed, case_id, snr)
            cell_rows.append({
                "case_id": case_id,
                "spacing_m": float(row["spacing_m"]),
                "snr_db": "" if snr is None else float(snr),
                "snr_tag": tag,
                "n_frac": int(row["n_frac"]),
                "clean_npz": row["npz_file"],
                "clean_path": os.path.join("data_clean", row["npz_file"]).replace("\\", "/"),
                "noise_seed": seed,
                "noisy_npz": (
                    None if snr is None
                    else f"case_{case_id:05d}_snr{tag}.npz"
                ),
            })

    manifest_csv = os.path.join(out_root, "manifest.csv")
    m_fields = [
        "case_id", "spacing_m", "snr_db", "snr_tag", "n_frac",
        "clean_npz", "clean_path", "noise_seed", "noisy_npz",
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=m_fields)
        writer.writeheader()
        writer.writerows(cell_rows)

    cell_summary = summarize_cells(cell_rows)
    meta = {
        "created_time": time_module.strftime("%Y-%m-%d %H:%M:%S"),
        "n_clean_total": len(rows),
        "n_clean_pass": sum(1 for r in rows if r["status"] == "PASS"),
        "n_clean_fail": sum(1 for r in rows if r["status"] != "PASS"),
        "n_manifest_rows": len(cell_rows),
        "spacing_grid_m": sorted({float(r["spacing_m"]) for r in rows}),
        "snr_db_levels": [
            None if s is None else float(s) for s in snr_levels
        ],
        "seed": int(base_seed),
        "clean_data_dir": clean_dir,
        "cell_counts": cell_summary,
        "samples_index": rows,
        "protocol_note": (
            "Noise is applied post-hoc to clean H_wh; re-simulate is not required. "
            "Use noise_seed + apply_awgn for reproducible noisy views."
        ),
    }
    if meta_extra:
        meta.update(meta_extra)

    meta_path = os.path.join(out_root, "bench_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    cell_json = os.path.join(out_root, "cell_counts.json")
    with open(cell_json, "w", encoding="utf-8") as handle:
        json.dump(cell_summary, handle, indent=2, ensure_ascii=False)

    return meta


def summarize_cells(cell_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """统计每个 (spacing, snr) 格子的样本数。"""
    counts: Dict[str, int] = {}
    for row in cell_rows:
        key = f"D{float(row['spacing_m']):g}_SNR{row['snr_tag']}"
        counts[key] = counts.get(key, 0) + 1
    by_spacing: Dict[str, int] = {}
    for row in cell_rows:
        if row["snr_tag"] == "inf":
            key = f"D{float(row['spacing_m']):g}"
            by_spacing[key] = by_spacing.get(key, 0) + 1
    return {
        "by_spacing_snr": dict(sorted(counts.items())),
        "by_spacing_clean": dict(sorted(by_spacing.items())),
        "min_per_cell": min(counts.values()) if counts else 0,
        "max_per_cell": max(counts.values()) if counts else 0,
    }


def run_batch(
    samples: Sequence[Dict[str, Any]],
    *,
    out_root: str,
    friction: str,
    tf: float,
    dt: float,
    workers: int,
) -> List[Dict[str, Any]]:
    """多进程跑清洁仿真。"""
    import multiprocessing as mp

    out_data_dir = os.path.join(out_root, "data_clean")
    os.makedirs(out_data_dir, exist_ok=True)
    worker_args = [
        (samp, friction, tf, dt, out_data_dir) for samp in samples
    ]
    results: List[Dict[str, Any]] = []
    n = len(samples)
    t0 = time_module.time()
    pass_count = fail_count = 0

    if workers <= 1:
        iterator = map(_worker_entry, worker_args)
        pool_ctx = None
    else:
        pool_ctx = mp.Pool(processes=workers)
        iterator = pool_ctx.imap_unordered(_worker_entry, worker_args)

    try:
        for i, res in enumerate(iterator):
            results.append(res)
            if res["status"] == "PASS":
                pass_count += 1
            else:
                fail_count += 1
            if (i + 1) % max(1, n // 20) == 0 or (i + 1) == n:
                elapsed = time_module.time() - t0
                spd = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n - i - 1) / spd if spd > 0 else 0.0
                print(
                    f"  [{(i+1)/n*100:5.1f}%] {i+1}/{n} | "
                    f"PASS={pass_count} FAIL={fail_count} | "
                    f"{spd:.2f} case/s | ETA {eta/60:.1f} min | "
                    f"last case_{res['case_id']:05d} D={res['spacing_m']:g} "
                    f"{res['status']}"
                )
    finally:
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

    return results
