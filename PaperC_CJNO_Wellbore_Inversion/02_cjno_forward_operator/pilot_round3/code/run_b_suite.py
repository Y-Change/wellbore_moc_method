# -*- coding: utf-8 -*-
"""Run B1 then B2; if both fail within budget, run 8 single-case fits."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from dataset import load_round1_manifest, nested_ids, nested_order
from paths import CONFIG_DIR, MANIFEST_DIR, TABLE_DIR, ensure_dirs


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))
    return p.returncode


def latest(prefix: str):
    root = Path(__file__).resolve().parents[1] / "runs"
    cands = sorted([d for d in root.iterdir() if d.name.startswith(prefix) and (d / "metrics.json").exists()],
                   key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main():
    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "train_round3.yaml").read_text(encoding="utf-8"))
    py = sys.executable
    here = Path(__file__).resolve().parent
    extra = []
    if "--allow-other-torch" in sys.argv:
        extra.append("--allow-other-torch")
    rc1 = run([py, str(here / "train_b.py"), "--arm", "B1", "--n", "8"] + extra)
    rc2 = run([py, str(here / "train_b.py"), "--arm", "B2", "--n", "8"] + extra)
    import json
    def loadm(pref):
        d = latest(pref)
        if d is None:
            return None
        return json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    m1, m2 = loadm("r3_B1_n8"), loadm("r3_B2_n8")
    singles = []
    both_fail = (m1 is None or m1["status"] != "PASS") and (m2 is None or m2["status"] != "PASS")
    if both_fail:
        for i in range(8):
            run([py, str(here / "train_b.py"), "--arm", "single", "--n", "8",
                 "--case-index", str(i)] + extra)
            d = latest("r3_single_")
            if d:
                singles.append(json.loads((d / "metrics.json").read_text(encoding="utf-8")))
    extra_seed = None
    if m1 and m1["status"] == "PASS":
        run([py, str(here / "train_b.py"), "--arm", "B1", "--n", "8",
             "--seed", str(cfg["extra_seed"])] + extra)
        extra_seed = loadm("r3_B1_n8")
    elif m2 and m2["status"] == "PASS":
        run([py, str(here / "train_b.py"), "--arm", "B2", "--n", "8",
             "--seed", str(cfg["extra_seed"])] + extra)
        extra_seed = loadm("r3_B2_n8")
    out = {
        "B1": m1, "B2": m2,
        "both_fail_within_budget": both_fail,
        "singles": singles,
        "extra_seed": extra_seed,
        "interpretation_rules": {
            "B1_pass": "short budget/schedule was an important factor",
            "only_B2_pass": "amplitude weighting was an important factor",
            "both_fail": "not passed within this budget; architecture not declared impossible",
        },
    }
    dump_json(out, MANIFEST_DIR / "optimization_diagnosis.json")
    dump_json(out, TABLE_DIR / "optimization_diagnosis.json")
    dump_json(out, Path(__file__).resolve().parents[1] / "docs" / "optimization_diagnosis.json")
    print("B suite written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
