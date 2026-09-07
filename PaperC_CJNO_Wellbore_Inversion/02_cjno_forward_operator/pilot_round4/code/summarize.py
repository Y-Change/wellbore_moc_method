# -*- coding: utf-8 -*-
"""Build Round-4 tables from explicit run_ids only. Never reads latest/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r4_paths import DOCS_DIR, MANIFEST_DIR, RUNS_DIR, ensure_dirs
from common import dump_json


def load_run(run_id: str) -> dict:
    d = RUNS_DIR / run_id
    if not d.is_dir():
        return {"run_id": run_id, "status": "MISSING", "dir": str(d)}
    meta = {}
    metrics = {}
    if (d / "run_meta.json").is_file():
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    if (d / "metrics.json").is_file():
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    else:
        return {"run_id": run_id, "status": "INCOMPLETE", "meta": meta,
                "note": "no metrics.json; must not cite old PASS"}
    if meta.get("status") not in ("completed", None) and meta.get("status") != "completed":
        if meta.get("status") == "running":
            return {"run_id": run_id, "status": "INCOMPLETE", "meta": meta, "metrics": metrics,
                    "note": "process did not close; do not promote to PASS"}
    return {"run_id": run_id, "status": meta.get("status", "unknown"),
            "code_sha256": meta.get("code_sha256"),
            "kernel": (metrics.get("kernel") or metrics.get("kernel_audit")),
            "metrics": metrics, "meta": meta, "dir": str(d)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", required=True, help="comma-separated explicit run_ids")
    args = parser.parse_args()
    ensure_dirs()
    ids = [x.strip() for x in args.run_ids.split(",") if x.strip()]
    rows = [load_run(i) for i in ids]
    out = {
        "protocol": "explicit_run_id_only_no_latest",
        "run_ids": ids,
        "rows": rows,
    }
    dump_json(out, MANIFEST_DIR / "round4_summary_from_run_ids.json")
    dump_json(out, DOCS_DIR / "evidence_registry.json")
    print(json.dumps({"n": len(rows), "ids": ids, "statuses": [r["status"] for r in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
