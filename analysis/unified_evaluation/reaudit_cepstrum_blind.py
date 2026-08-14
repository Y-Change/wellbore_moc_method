# -*- coding: utf-8 -*-
"""
reaudit_cepstrum_blind.py — 用盲协议复算历史 leakoff 倒谱匹配结果。

目的
----
EXP-20260730-018/-025 认定历史倒谱匹配**不是盲结果**：检测器读了真实最小缝距
（设最小峰距）、真实缝数（设 top_n），评分容差也由真实缝距导出且带 80 m 下限。

本脚本不重跑 MOC，直接复用 ``output/leakoff/**/moc_timeseries.csv`` 的时序，
用 :mod:`analysis.unified_evaluation.detection_protocol` 的盲协议重算，
并与 ``moc_leakoff.json`` 中的历史数值逐条对照，量化"摘掉泄漏后差多少"。

历史指标只报 ``n_matched``（近似召回），**完全不惩罚假峰**；本脚本同时给出
precision / F1 / separation_success，因此两者不可直接等同——对照表用于显示
结论强度的变化方向，不是同一指标的前后值。

用法
----
    python -m analysis.unified_evaluation.reaudit_cepstrum_blind
    python -m analysis.unified_evaluation.reaudit_cepstrum_blind --friction steady
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from analysis.unified_evaluation.detection_protocol import (
    DetectorConfig,
    ScoringConfig,
    evaluate_blind,
)
from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum_1d

DEFAULT_LEAKOFF_ROOT = Path("output/leakoff")
DEFAULT_OUT_DIR = Path("output/analysis/blind_protocol_reaudit")


def _load_timeseries(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    t: List[float] = []
    h: List[float] = []
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t.append(float(row["t"]))
            h.append(float(row["H_wh"]))
    return np.asarray(t), np.asarray(h)


def _min_spacing(x_f: List[float]) -> Optional[float]:
    if len(x_f) < 2:
        return None
    return float(np.min(np.diff(sorted(x_f))))


def reaudit_case(
    case_dir: Path,
    min_separation_m: Optional[float],
    search_min_m: float,
    search_margin_m: float,
) -> Optional[Dict]:
    json_path = case_dir / "moc_leakoff.json"
    csv_path = case_dir / "moc_timeseries.csv"
    if not json_path.exists() or not csv_path.exists():
        return None

    with json_path.open(encoding="utf-8") as handle:
        old = json.load(handle)
    cfg = old.get("config") or {}
    x_f = [float(x) for x in (cfg.get("x_f") or [])]
    if not x_f:
        return None

    old_1d = ((old.get("cepstrum") or {}).get("1d_real")) or {}

    t, h = _load_timeseries(csv_path)
    wellbore_length = float(cfg.get("L", 5000.0))
    result = compute_moc_cepstrum_1d(
        t, h,
        v=float(cfg.get("a", 1450.0)),
        dt=float(cfg.get("dt", 1e-3)),
        ts=float(cfg.get("ts", 1.0)),
        wellbore_length=wellbore_length,
    )

    # 深度搜索窗：排除倒频带边缘在 depth≈L 处的固定伪峰。
    # 这是先验井段范围（每个 case 相同），不是逐样本真值，允许使用。
    det_cfg = DetectorConfig(
        min_separation_m=(
            min_separation_m
            if min_separation_m is not None
            else DetectorConfig().min_separation_m
        ),
        search_min_m=search_min_m,
        search_max_m=wellbore_length - search_margin_m,
    )
    blind = evaluate_blind(
        result["depth"], result["response"], x_f,
        detector_config=det_cfg,
        scoring_config=ScoringConfig(),
    )
    primary = blind["primary"]

    return {
        "friction": cfg.get("friction", case_dir.parent.name),
        "case": case_dir.name,
        "n_frac": len(x_f),
        "min_spacing_m": _min_spacing(x_f),
        # ---- 历史（含真值泄漏）----
        "old_n_matched": old_1d.get("n_matched"),
        "old_match_tol_m": old_1d.get("match_tol_m"),
        "old_n_detected": len(old_1d.get("detected_peaks") or []),
        "old_mean_error_m": old_1d.get("mean_error_m"),
        # ---- 盲协议 ----
        "new_n_detected": primary["n_pred"],
        "new_tp": primary["tp"],
        "new_fp": primary["fp"],
        "new_precision": primary["precision"],
        "new_recall": primary["recall"],
        "new_f1": primary["f1"],
        "new_exact_count": primary["exact_count"],
        "new_separation_success": primary["separation_success"],
        "new_n_likely_merged": primary["n_likely_merged"],
        "new_median_error_m": primary["median_error_m"],
        "new_tolerance_m": primary["tolerance_m"],
        "sweep": blind["tolerance_sweep"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leakoff-root", default=str(DEFAULT_LEAKOFF_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--friction", default=None,
        help="只处理名字以此开头的摩阻目录，例如 steady / brunone",
    )
    parser.add_argument(
        "--min-separation-m", type=float, default=None,
        help="盲寻峰最小峰距 [m]；缺省用 CEPSTRUM_CONFIG",
    )
    parser.add_argument(
        "--search-min-m", type=float, default=100.0,
        help="深度搜索窗下界 [m]（与 _kb_core 的 depth_min 约定一致）",
    )
    parser.add_argument(
        "--search-margin-m", type=float, default=100.0,
        help="深度搜索窗上界距井底的余量 [m]，用于排除 depth≈L 的边界伪峰",
    )
    args = parser.parse_args()

    root = Path(args.leakoff_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for friction_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if args.friction and not friction_dir.name.startswith(args.friction):
            continue
        for case_dir in sorted(p for p in friction_dir.iterdir() if p.is_dir()):
            row = reaudit_case(
                case_dir, args.min_separation_m,
                args.search_min_m, args.search_margin_m,
            )
            if row is not None:
                rows.append(row)
                print(
                    f"  {row['friction']:<14s} {row['case']:<8s} "
                    f"n={row['n_frac']} D={row['min_spacing_m']} | "
                    f"old n_matched={row['old_n_matched']}/{row['n_frac']} "
                    f"tol={row['old_match_tol_m']} | "
                    f"blind F1={row['new_f1']:.3f} "
                    f"sep={'Y' if row['new_separation_success'] else 'N'}"
                )

    if not rows:
        print("未找到可复算的 case。")
        return

    csv_path = out_dir / "reaudit_summary.csv"
    fields = [k for k in rows[0] if k != "sweep"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "reaudit_full.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    _write_markdown(rows, out_dir / "REAUDIT.md")
    print(f"\nWrote {csv_path}\nWrote {json_path}\nWrote {out_dir / 'REAUDIT.md'}")


def _write_markdown(rows: List[Dict], path: Path) -> None:
    multi = [r for r in rows if r["n_frac"] > 1]
    lines = [
        "# 盲协议复算：历史 leakoff 倒谱匹配",
        "",
        "由 `analysis/unified_evaluation/reaudit_cepstrum_blind.py` 生成。",
        "",
        "历史列来自 `moc_leakoff.json`（检测器读真值：最小峰距按真实缝距、"
        "容差 `clip(0.45×真实缝距, 80, 250)`）；盲列为摘除泄漏后的重算结果。",
        "",
        "**注意**：历史只报 `n_matched`，不惩罚假峰，与盲列的 F1 不是同一指标，"
        "对照表用于显示结论强度的变化方向。",
        "",
        "## 按最小缝距汇总（仅多缝 case）",
        "",
        "| 最小缝距 [m] | case 数 | 历史 全匹配率 | 盲 分离成功率 | 盲 平均 F1 | 历史 容差 [m] | 盲 容差 [m] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_spacing: Dict[float, List[Dict]] = {}
    for r in multi:
        by_spacing.setdefault(r["min_spacing_m"], []).append(r)
    for spacing in sorted(by_spacing):
        group = by_spacing[spacing]
        old_full = np.mean([
            1.0 if (r["old_n_matched"] or 0) >= r["n_frac"] else 0.0 for r in group
        ])
        sep = np.mean([1.0 if r["new_separation_success"] else 0.0 for r in group])
        f1 = np.mean([r["new_f1"] for r in group])
        old_tol = group[0]["old_match_tol_m"]
        new_tol = group[0]["new_tolerance_m"]
        lines.append(
            f"| {spacing:g} | {len(group)} | {old_full:.0%} | {sep:.0%} | "
            f"{f1:.3f} | {old_tol} | {new_tol:g} |"
        )

    lines += ["", "## 逐 case 明细", "",
              "| 摩阻 | case | n | D [m] | 历史 matched | 历史 峰数 | 盲 峰数 | 盲 P | 盲 R | 盲 F1 | 分离 | 疑似合并 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|"]
    for r in rows:
        lines.append(
            "| {friction} | {case} | {n_frac} | {spacing} | {om}/{n_frac} | {od} | "
            "{nd} | {p:.3f} | {rc:.3f} | {f1:.3f} | {sep} | {mg} |".format(
                friction=r["friction"], case=r["case"], n_frac=r["n_frac"],
                spacing=("-" if r["min_spacing_m"] is None else f"{r['min_spacing_m']:g}"),
                om=r["old_n_matched"], od=r["old_n_detected"], nd=r["new_n_detected"],
                p=r["new_precision"], rc=r["new_recall"], f1=r["new_f1"],
                sep="Y" if r["new_separation_success"] else "N",
                mg=r["new_n_likely_merged"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
