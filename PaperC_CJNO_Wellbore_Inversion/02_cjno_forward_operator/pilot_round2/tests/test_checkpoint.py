# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from checkpoint import assert_data_version, data_version, load_checkpoint, save_checkpoint
from common import sha256_json


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 1)


def test_data_version_not_code_digest(tmp_path: Path = None):
    root = tmp_path or Path(__file__).resolve().parents[1] / "runs" / "_ckpt_test"
    root.mkdir(parents=True, exist_ok=True)
    m = Tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    cfg = {"query": {"dt_s": 0.02, "n_samples": 400}}
    norm = {"p_pert_scale": 1.0, "feat_mean": [0.0], "feat_std": [1.0]}
    ids = ["a", "b"]
    man_sha = "m" * 64
    code_sha = "c" * 64
    p = root / "t.pt"
    dv = save_checkpoint(p, m, opt, cfg, norm, ids, man_sha, code_sha, 3, 10)
    assert dv["manifest_sha256"] == man_sha
    assert dv["manifest_sha256"] != code_sha
    blob = load_checkpoint(p)
    assert blob["epoch"] == 3
    assert blob["global_step"] == 10
    assert "optimizer" in blob and blob["optimizer"] is not None
    assert_data_version(blob, man_sha, ids)
    try:
        assert_data_version(blob, "wrong", ids)
        raise AssertionError("should fail manifest mismatch")
    except RuntimeError:
        pass
    # forged equality of data_version and code sha is rejected
    blob["data_version"]["manifest_sha256"] = blob["code_sha256"]
    bad = root / "bad.pt"
    torch.save(blob, bad)
    try:
        load_checkpoint(bad)
        raise AssertionError("should reject data_version==code_sha")
    except RuntimeError:
        pass


def test_same_ckpt_train_val_numbers_are_from_one_state():
    dv = data_version("abc", ["c1"], {"dt_s": 0.02})
    assert dv["n_cases"] == 1
    assert dv["case_ids_sha256"] == sha256_json(["c1"])


if __name__ == "__main__":
    test_data_version_not_code_digest()
    test_same_ckpt_train_val_numbers_are_from_one_state()
    print("test_checkpoint PASS")
