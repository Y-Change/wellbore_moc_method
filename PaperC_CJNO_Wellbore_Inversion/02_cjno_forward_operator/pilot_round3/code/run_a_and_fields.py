# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from dataset import CONSUMED_BY_MLP, PRESENT_NOT_CONSUMED_BY_MLP
from paths import MANIFEST_DIR, TABLE_DIR, ensure_dirs


def write_field_table():
    ensure_dirs()
    rows = []
    for k in CONSUMED_BY_MLP:
        rows.append({"field": k, "enters_QueryFourierMLP": True, "how": "raw_features / pack_features"})
    for k in PRESENT_NOT_CONSUMED_BY_MLP:
        rows.append({"field": k, "enters_QueryFourierMLP": False,
                     "how": "present on CaseBundle / predict_arrays but not in the 44-d MLP vector"})
    rec = {
        "note": "This is the B-arm consumption table. Completing unused fields is a separate arm, not mixed into B1/B2.",
        "rows": rows,
    }
    dump_json(rec, MANIFEST_DIR / "input_field_consumption.json")
    dump_json(rec, TABLE_DIR / "input_field_consumption.json")
    dump_json(rec, Path(__file__).resolve().parents[1] / "docs" / "input_field_consumption.json")
    return rec


if __name__ == "__main__":
    write_field_table()
    print("wrote input field table")
