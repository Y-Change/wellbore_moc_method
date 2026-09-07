# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PilotDataset, collate_batch, load_research_manifest
from paths import MANIFEST_DIR


def main():
    man = load_research_manifest()
    counts = Counter(c["split"] for c in man["cases"])
    assert man["n_usable"] == len(man["cases"])
    assert not man["leakage_groups"]
    groups = {}
    for c in man["cases"]:
        groups.setdefault(c["group_id"], set()).add(c["split"])
    assert all(len(v) == 1 for v in groups.values()), "group split leakage"
    ds = PilotDataset("train", window_periods=2.0, wellhead_decim=8, node_max_len=64)
    s = ds[0]
    assert s.t.size >= 8 and s.p_pert.size == s.t.size
    assert s.node_H.shape[1] == 12
    assert s.N >= 4
    assert s.nyquist_node < s.nyquist_wh
    norm = ds.compute_train_norm(max_cases=8, seed=1)
    batch = collate_batch([ds[0], ds[1]], norm)
    assert batch["p_pert"].shape[0] == 2
    # val must not recompute norm from val
    dv = PilotDataset("val", window_periods=2.0, wellhead_decim=8, node_max_len=64)
    try:
        dv.compute_train_norm(max_cases=2)
        raise SystemExit("val must not compute train norm")
    except RuntimeError:
        pass
    print(json.dumps({
        "n_usable": man["n_usable"], "splits": dict(counts),
        "sample_T": int(s.t.size), "sample_Tn": int(s.node_t.size),
        "nyquist_wh": s.nyquist_wh, "nyquist_node": s.nyquist_node,
        "status": "PASS",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
