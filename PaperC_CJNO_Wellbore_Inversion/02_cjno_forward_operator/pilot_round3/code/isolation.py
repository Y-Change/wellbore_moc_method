# -*- coding: utf-8 -*-
"""Train-only isolation by actual case_id, not just the split string."""
from __future__ import annotations

from typing import Iterable, List, Optional

from dataset_io import load_round1_manifest


class SplitGuard:
    def __init__(self, manifest: Optional[dict] = None):
        man = manifest or load_round1_manifest()
        self.by_id = {c["case_id"]: c["split"] for c in man["cases"]}
        self.test_ids = {cid for cid, sp in self.by_id.items() if sp == "test"}
        self.train_ids = {cid for cid, sp in self.by_id.items() if sp == "train"}
        self.val_ids = {cid for cid, sp in self.by_id.items() if sp == "val"}

    def assert_split_name(self, split: str) -> None:
        if split == "test":
            raise RuntimeError("old test split is a viewed holdout; blocked")
        if split == "all_usable":
            raise RuntimeError("all_usable is blocked this round (would include viewed test ids)")
        if split not in ("train", "val"):
            raise RuntimeError(f"unknown/blocked split {split}")

    def assert_case_ids(self, case_ids: Iterable[str], allow_val: bool = True) -> None:
        bad = []
        for cid in case_ids:
            if cid in self.test_ids:
                bad.append(cid)
            sp = self.by_id.get(cid)
            if sp is None:
                bad.append(f"{cid}:unknown")
            if (not allow_val) and sp == "val":
                bad.append(f"{cid}:val_blocked_for_this_entry")
        if bad:
            raise RuntimeError(f"blocked case_ids (test/unknown/disallowed): {bad[:8]}")

    def filter_train(self, case_ids: List[str]) -> List[str]:
        self.assert_case_ids(case_ids, allow_val=False)
        out = [c for c in case_ids if c in self.train_ids]
        if len(out) != len(case_ids):
            raise RuntimeError("non-train ids in a train-only request")
        return out
