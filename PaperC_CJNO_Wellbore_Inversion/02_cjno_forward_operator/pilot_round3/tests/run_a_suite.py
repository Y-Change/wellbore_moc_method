# -*- coding: utf-8 -*-
"""Run all A-contract tests and write metrics_regression_tests.json."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "code"))
from common import dump_json
from paths import MANIFEST_DIR, ensure_dirs

TESTS = [
    "test_metrics.py",
    "test_dataset.py",
    "test_checkpoint.py",
    "test_batch_consistency.py",
    "test_node_branches.py",
]


def main():
    ensure_dirs()
    rows = []
    all_ok = True
    for name in TESTS:
        t0 = time.time()
        rec = {"name": name, "status": "FAIL", "seconds": 0.0, "error": None}
        try:
            ns = {"__name__": "__main__", "__file__": str(ROOT / name)}
            code = (ROOT / name).read_text(encoding="utf-8")
            exec(compile(code, str(ROOT / name), "exec"), ns)
            rec["status"] = "PASS"
        except SystemExit as e:
            rec["status"] = "PASS" if int(getattr(e, "code", 0) or 0) == 0 else "FAIL"
            if rec["status"] != "PASS":
                rec["error"] = f"SystemExit {e.code}"
                all_ok = False
        except Exception:
            rec["error"] = traceback.format_exc()
            all_ok = False
        rec["seconds"] = time.time() - t0
        rows.append(rec)
        print(f"{name}: {rec['status']} ({rec['seconds']:.2f}s)")
    out = {
        "suite": "round3_A_metrics_regression",
        "all_pass": all_ok,
        "rows": rows,
    }
    dump_json(out, MANIFEST_DIR / "metrics_regression_tests.json")
    dump_json(out, ROOT.parent / "docs" / "metrics_regression_tests.json")
    if not all_ok:
        raise SystemExit(1)
    print("A suite PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
