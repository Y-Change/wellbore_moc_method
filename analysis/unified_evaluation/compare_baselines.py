# -*- coding: utf-8 -*-
"""Compare baseline gate metrics under the shared P0-A event protocol."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _load_gate(path: str) -> Dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _row_from_gate(name: str, path: str) -> Dict:
    gate = _load_gate(path)
    metrics = gate.get("metrics") or {}
    physical = metrics.get("physical") or {}
    bands = (metrics.get("strata") or {}).get("by_spacing_band") or {}
    band_5_10 = bands.get("5-10") or {}
    return {
        "name": name,
        "path": os.path.abspath(path),
        "passed": bool(gate.get("passed")),
        "stage": gate.get("stage"),
        "f1": physical.get("f1"),
        "precision": physical.get("precision"),
        "recall": physical.get("recall"),
        "exact_count_accuracy": metrics.get("exact_count_accuracy"),
        "count_mae": metrics.get("count_mae"),
        "median_depth_error_m": metrics.get("median_depth_error_m"),
        "f1_5_10": (band_5_10.get("physical") or {}).get("f1"),
        "exact_5_10": band_5_10.get("exact_count_accuracy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified baseline comparison table")
    parser.add_argument(
        "--entry",
        action="append",
        nargs=2,
        metavar=("NAME", "GATE_JSON"),
        required=True,
        help="Repeatable: --entry phasenet path/to/gate_result.json",
    )
    parser.add_argument("--output", default=None, help="Optional markdown/json output path")
    args = parser.parse_args()

    rows: List[Dict] = [_row_from_gate(name, path) for name, path in args.entry]
    header = (
        "| method | stage | pass | F1 | P | R | exact | MAE | med depth | 5-10 F1 | 5-10 exact |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {name} | {stage} | {passed} | {f1:.3f} | {precision:.3f} | {recall:.3f} | "
            "{exact:.1%} | {mae:.2f} | {med:.2f} | {f1_5:.3f} | {ex_5:.1%} |".format(
                name=row["name"],
                stage=row["stage"],
                passed="PASS" if row["passed"] else "FAIL",
                f1=float(row["f1"] or 0.0),
                precision=float(row["precision"] or 0.0),
                recall=float(row["recall"] or 0.0),
                exact=float(row["exact_count_accuracy"] or 0.0),
                mae=float(row["count_mae"] or 0.0),
                med=float(row["median_depth_error_m"] or 0.0),
                f1_5=float(row["f1_5_10"] or 0.0),
                ex_5=float(row["exact_5_10"] or 0.0),
            )
        )
    table = "\n".join(lines)
    print(table)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        if args.output.endswith(".json"):
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(rows, handle, indent=2, ensure_ascii=False)
        else:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(table + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
