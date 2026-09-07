# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json, sha256_file, sha256_json
from dataset import load_round1_manifest, nested_ids, nested_order
from paths import CONFIG_DIR, MANIFEST_DIR, ROUND1_MANIFEST, ensure_dirs


def main():
    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "train_round2.yaml").read_text(encoding="utf-8"))
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    nests = {}
    for n in cfg["nests"]:
        ids = nested_ids(order, n if n != "all" else "all")
        nests[str(n)] = ids
    out = {
        "subset_seed": cfg["subset_seed"],
        "model_seed_primary": cfg["model_seed_primary"],
        "round1_manifest": str(ROUND1_MANIFEST),
        "round1_manifest_sha256": sha256_file(ROUND1_MANIFEST),
        "n_train_total": len(order),
        "order": order,
        "nests": {k: {"n": len(v), "case_ids": v, "ids_sha256": sha256_json(v)} for k, v in nests.items()},
        "note": "prefixes are nested; model_seed is independent",
    }
    dump_json(out, MANIFEST_DIR / "nested_train_order.json")
    print("wrote", MANIFEST_DIR / "nested_train_order.json", "n_train", len(order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
