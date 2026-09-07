# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT))

from common import dump_json
from paths import MANIFEST_DIR, ensure_dirs


def main():
    ensure_dirs()
    mods = [
        "test_metrics",
        "test_checkpoint",
        "test_dataset",
        "test_batch_consistency",
        "test_node_branches",
    ]
    rows = []
    n_fail = 0
    for name in mods:
        try:
            m = __import__(name)
            if hasattr(m, "test_known_delay"):
                m.test_known_delay()
                m.test_amplitude_change_same_time()
                m.test_cross_period_mismatch_not_counted()
                m.test_no_peak_tagged()
                m.test_silent_is_target_only()
            if hasattr(m, "test_data_version_not_code_digest"):
                m.test_data_version_not_code_digest()
                m.test_same_ckpt_train_val_numbers_are_from_one_state()
            if hasattr(m, "test_test_split_blocked"):
                m.test_test_split_blocked()
                m.test_predict_has_no_future_labels()
                m.test_nested_prefixes()
                m.test_shuffle_targets_does_not_change_predict_arrays()
            if hasattr(m, "test_batch_invariance"):
                m.test_batch_invariance()
            if hasattr(m, "test_stiff_guard_actually_fires"):
                m.test_stiff_guard_actually_fires()
                m.test_reverse_and_near_zero_expectations_registered()
            rows.append({"name": name, "status": "PASS"})
            print("PASS", name)
        except Exception as exc:
            n_fail += 1
            rows.append({"name": name, "status": "FAIL", "error": repr(exc),
                         "tb": traceback.format_exc()})
            print("FAIL", name, exc)
    out = {"status": "PASS" if n_fail == 0 else "FAIL", "n_fail": n_fail, "rows": rows}
    dump_json(out, MANIFEST_DIR / "unit_tests.json")
    print(out["status"])
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
