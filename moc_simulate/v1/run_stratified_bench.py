# -*- coding: utf-8 -*-
"""
run_stratified_bench.py — 分层含噪基准集生成入口（EXP-011 S1 / EXP-021 输入）

清洁波形按固定间距格子仿真一次；噪声后处理写入 manifest（可按需 materialize）。

示例
----
    # smoke：每格 2 条，tf=8s，快速验收格子完整性
    python moc_simulate/run_stratified_bench.py --smoke

    # 生产：每格 30 条（默认），tf=30s，brunone
    python moc_simulate/run_stratified_bench.py --n-per-spacing 30 --workers 14

    # 把 SNR 变体物化为独立 npz（可选；默认只写 manifest + 种子）
    python moc_simulate/run_stratified_bench.py --materialize-noise
"""
from __future__ import annotations

import argparse
import os
import sys
import time as time_module

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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

from moc_simulate.lhs_config import BENCH_STRATIFIED_CONFIG, SIM_CONFIG
from moc_simulate.stratified_bench import (
    generate_stratified_params,
    materialize_noisy_case,
    noise_seed_for,
    run_batch,
    snr_tag,
    write_manifest,
)


def _parse_snr_levels(text: str):
    levels = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in ("inf", "none", "clean"):
            levels.append(None)
        else:
            levels.append(float(part))
    if not levels:
        raise ValueError("empty --snr-levels")
    return tuple(levels)


def main() -> None:
    cfg = BENCH_STRATIFIED_CONFIG
    parser = argparse.ArgumentParser(
        description="分层（间距格子）+ 含噪（后处理 SNR）基准集生成器",
    )
    parser.add_argument(
        "--n-per-spacing", type=int, default=cfg["n_per_spacing"],
        help=f"每间距格子清洁样本数（默认 {cfg['n_per_spacing']}）",
    )
    parser.add_argument(
        "--n-frac", type=int, default=cfg["n_frac"],
        help=f"裂缝条数（默认 {cfg['n_frac']}，分辨率基准建议 2）",
    )
    parser.add_argument(
        "--spacing-grid", type=str, default=None,
        help="逗号分隔间距格子 [m]；缺省用 BENCH_STRATIFIED_CONFIG",
    )
    parser.add_argument(
        "--snr-levels", type=str, default=None,
        help="逗号分隔 SNR [dB]，可用 inf 表示清洁；缺省用配置",
    )
    parser.add_argument("--workers", type=int, default=cfg["default_workers"])
    parser.add_argument("--tf", type=float, default=cfg["default_tf"])
    parser.add_argument("--dt", type=float, default=cfg["default_dt"])
    parser.add_argument(
        "--friction", type=str, default=cfg["default_friction"],
        choices=["steady", "brunone"],
    )
    parser.add_argument("--out-dir", type=str, default=cfg["output_dir"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument(
        "--smoke", action="store_true",
        help="快速验收：每格 2 条、tf=8s、workers≤4",
    )
    parser.add_argument(
        "--materialize-noise", action="store_true",
        help="把非 inf 的 SNR 变体写成 data_noisy/*.npz（磁盘更大）",
    )
    parser.add_argument(
        "--skip-sim", action="store_true",
        help="跳过仿真，仅根据已有 data_clean 重建 manifest / 物化噪声",
    )
    args = parser.parse_args()

    if args.smoke:
        args.n_per_spacing = min(args.n_per_spacing, 2)
        args.tf = min(args.tf, 8.0)
        args.workers = min(args.workers, 4)
        if args.out_dir == cfg["output_dir"]:
            args.out_dir = "output/bench_stratified_smoke"

    spacing_grid = (
        tuple(float(x) for x in args.spacing_grid.split(","))
        if args.spacing_grid
        else tuple(cfg["spacing_grid_m"])
    )
    snr_levels = (
        _parse_snr_levels(args.snr_levels)
        if args.snr_levels
        else tuple(cfg["snr_db_levels"])
    )

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    print("=" * 72)
    print("分层含噪基准集生成器（EXP-011 S1）")
    print("=" * 72)
    print(f"  间距格子 : {spacing_grid}")
    print(f"  每格样本 : {args.n_per_spacing}")
    print(f"  裂缝条数 : {args.n_frac}")
    print(f"  SNR 水平 : {[snr_tag(s) for s in snr_levels]}")
    print(f"  摩阻/tf  : {args.friction} / {args.tf}s")
    print(f"  workers  : {args.workers}")
    print(f"  输出路径 : {out_root}")
    print("=" * 72)

    samples = generate_stratified_params(
        spacing_grid_m=spacing_grid,
        n_per_spacing=args.n_per_spacing,
        n_frac=args.n_frac,
        seed=args.seed,
    )
    print(f"\n[Step 1] 参数矩阵: {len(samples)} 条清洁样本")

    if args.skip_sim:
        print("[Step 2] --skip-sim：从已有 data_clean 重建索引")
        clean_dir = os.path.join(out_root, "data_clean")
        results = []
        for samp in samples:
            npz = f"case_{samp['case_id']:05d}.npz"
            path = os.path.join(clean_dir, npz)
            ok = os.path.isfile(path)
            results.append({
                "case_id": samp["case_id"],
                "status": "PASS" if ok else "FAIL",
                "spacing_m": samp["spacing_m"],
                "n_frac": samp["n_frac"],
                "positions_str": ";".join(f"{x:.1f}" for x in samp["positions"]),
                "Cf_str": ";".join(f"{c:.2e}" for c in samp["Cf_list"]),
                "kleak_str": ";".join(f"{k:.2e}" for k in samp["kleak_list"]),
                "elapsed_s": 0.0,
                "npz_file": npz if ok else "",
                "error": "" if ok else "missing npz",
            })
    else:
        print("[Step 2] 多进程清洁仿真...")
        t0 = time_module.time()
        results = run_batch(
            samples,
            out_root=out_root,
            friction=args.friction,
            tf=args.tf,
            dt=args.dt,
            workers=args.workers,
        )
        print(f"  仿真耗时 { (time_module.time()-t0)/60:.2f} min")

    print("[Step 3] 写 manifest / metadata...")
    meta = write_manifest(
        out_root,
        clean_results=results,
        snr_levels=snr_levels,
        base_seed=args.seed,
        meta_extra={
            "tf_s": args.tf,
            "dt_s": args.dt,
            "friction_model": args.friction,
            "workers": args.workers,
            "n_frac": args.n_frac,
            "n_per_spacing": args.n_per_spacing,
            "smoke": bool(args.smoke),
            "param_config": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.items()
            },
            "ts_s": SIM_CONFIG["ts"],
        },
    )

    if args.materialize_noise:
        print("[Step 4] 物化含噪 npz...")
        noisy_dir = os.path.join(out_root, "data_noisy")
        os.makedirs(noisy_dir, exist_ok=True)
        n_written = 0
        for row in results:
            if row["status"] != "PASS":
                continue
            clean_path = os.path.join(out_root, "data_clean", row["npz_file"])
            for snr in snr_levels:
                if snr is None:
                    continue
                tag = snr_tag(snr)
                out_npz = os.path.join(
                    noisy_dir, f"case_{row['case_id']:05d}_snr{tag}.npz",
                )
                materialize_noisy_case(
                    clean_path, out_npz, snr,
                    noise_seed_for(args.seed, row["case_id"], snr),
                    ts=SIM_CONFIG["ts"],
                )
                n_written += 1
        print(f"  写出 {n_written} 个含噪文件 → {noisy_dir}")
        meta["data_noisy_dir"] = noisy_dir
        meta["n_noisy_materialized"] = n_written
        import json
        with open(os.path.join(out_root, "bench_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)

    cells = meta["cell_counts"]
    print("\n" + "=" * 72)
    print("完成")
    print(f"  清洁 PASS/FAIL : {meta['n_clean_pass']}/{meta['n_clean_fail']}")
    print(f"  manifest 行数  : {meta['n_manifest_rows']}")
    print(f"  每格样本数     : min={cells['min_per_cell']} max={cells['max_per_cell']}")
    print(f"  clean_summary  : {os.path.join(out_root, 'clean_summary.csv')}")
    print(f"  manifest       : {os.path.join(out_root, 'manifest.csv')}")
    print(f"  metadata       : {os.path.join(out_root, 'bench_metadata.json')}")
    print("=" * 72)

    if cells["min_per_cell"] < args.n_per_spacing:
        print(
            f"[警告] 某些 (spacing,SNR) 格子样本数 < n_per_spacing="
            f"{args.n_per_spacing}；请检查 FAIL case。"
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
