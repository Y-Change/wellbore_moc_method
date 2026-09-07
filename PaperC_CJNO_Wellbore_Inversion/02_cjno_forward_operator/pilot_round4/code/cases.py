# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from isolation import SplitGuard, load_round1_manifest
from r4_paths import TRAIN4


def nested_order(manifest: dict, subset_seed: int) -> List[str]:
    train = [c["case_id"] for c in manifest["cases"] if c["split"] == "train"]
    rng = np.random.default_rng(int(subset_seed))
    perm = rng.permutation(len(train))
    return [train[int(k)] for k in perm]


def load_train_cases(n: int = 4, subset_seed: int = 20260907) -> Tuple[List[str], List[dict], List[dict]]:
    man = load_round1_manifest()
    order = nested_order(man, subset_seed)
    ids = list(order[:int(n)])
    if tuple(ids[:4]) != TRAIN4:
        raise RuntimeError(f"nested prefix drifted: {ids[:4]} != {TRAIN4}")
    SplitGuard(man).assert_case_ids(ids, allow_val=False)
    by_id = {c["case_id"]: c for c in man["cases"]}
    params, rows = [], []
    for cid in ids:
        row = by_id[cid]
        z = np.load(row["source_file"], allow_pickle=True)
        p = json.loads(str(z["params_json"]))
        params.append(p)
        rows.append({
            "case_id": cid,
            "source_file": row["source_file"],
            "source_sha256": row["source_sha256"],
            "split": row["split"],
        })
    return ids, params, rows


def joukowsky_scale(params: dict) -> float:
    w = params["well"]
    A = 0.25 * np.pi * float(w["D"]) ** 2
    return float(abs(w["rho"] * w["a"] * w["Q0"] / max(A, 1e-30)))


def classify_silence(target, physical_scale: float, rel_thresh: float = 1e-6) -> bool:
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    scale = float(abs(physical_scale))
    if scale < 1e-30:
        scale = 1.0
    return float(np.linalg.norm(y) / scale) < rel_thresh
