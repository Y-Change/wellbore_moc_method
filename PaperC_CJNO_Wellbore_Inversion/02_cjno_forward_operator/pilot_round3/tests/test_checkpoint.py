# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from checkpoint import (
    assert_data_version, data_version, load_checkpoint, restore_train_state, save_checkpoint,
)
from common import sha256_json


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 1)

    def forward(self, x):
        return self.w(x)


def test_data_version_not_code_digest(tmp_path: Path = None):
    root = tmp_path or Path(__file__).resolve().parents[1] / "runs" / "_ckpt_test"
    root.mkdir(parents=True, exist_ok=True)
    m = Tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    cfg = {"query": {"dt_s": 0.02, "n_samples": 400}}
    norm = {"p_pert_scale": 1.0, "feat_mean": [0.0], "feat_std": [1.0], "protocol": "r3"}
    ids = ["a", "b"]
    man_sha = "m" * 64
    code_sha = "c" * 64
    p = root / "t.pt"
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    dv = save_checkpoint(p, m, opt, cfg, norm, ids, man_sha, code_sha, 3, 10, scheduler=sch)
    assert dv["manifest_sha256"] == man_sha
    assert dv["manifest_sha256"] != code_sha
    blob = load_checkpoint(p)
    assert blob["epoch"] == 3
    assert blob["global_step"] == 10
    assert blob["optimizer"] is not None
    assert blob["scheduler"] is not None
    assert blob["rng"]["python"] is not None
    assert blob["rng"]["numpy"] is not None
    assert blob["rng"]["torch"] is not None
    assert_data_version(blob, man_sha, ids)
    try:
        assert_data_version(blob, "wrong", ids)
        raise AssertionError("should fail manifest mismatch")
    except RuntimeError:
        pass
    blob2 = dict(blob)
    blob2["data_version"] = dict(blob["data_version"])
    blob2["data_version"]["manifest_sha256"] = blob2["code_sha256"]
    bad = root / "bad.pt"
    torch.save(blob2, bad)
    try:
        load_checkpoint(bad)
        raise AssertionError("should reject data_version==code_sha")
    except RuntimeError:
        pass


def test_resume_matches_continuous(tmp_path: Path = None):
    root = tmp_path or Path(__file__).resolve().parents[1] / "runs" / "_resume_test"
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)

    def make():
        m = Tiny()
        opt = torch.optim.SGD(m.parameters(), lr=0.2)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        return m, opt, sch

    m1, o1, s1 = make()
    m2, o2, s2 = make()
    m2.load_state_dict(m1.state_dict())
    o2.load_state_dict(o1.state_dict())
    s2.load_state_dict(s1.state_dict())

    def step(m, o, s):
        o.zero_grad()
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        o.step()
        s.step()
        return float(loss)

    step(m1, o1, s1)
    step(m2, o2, s2)
    cfg = {"query": {"dt_s": 0.02}}
    p = root / "mid.pt"
    save_checkpoint(p, m1, o1, cfg, {"protocol": "r3"}, ["c1"], "m" * 64, "c" * 64,
                    1, 1, scheduler=s1)
    # continue without save
    step(m2, o2, s2)
    # resume and continue
    blob = load_checkpoint(p)
    m3, o3, s3 = make()
    restore_train_state(blob, m3, o3, s3)
    step(m3, o3, s3)
    for a, b in zip(m2.parameters(), m3.parameters()):
        assert torch.allclose(a, b, atol=1e-7, rtol=1e-6)
    assert abs(o2.param_groups[0]["lr"] - o3.param_groups[0]["lr"]) < 1e-12


def test_same_ckpt_train_val_numbers_are_from_one_state():
    dv = data_version("abc", ["c1"], {"dt_s": 0.02})
    assert dv["n_cases"] == 1
    assert dv["case_ids_sha256"] == sha256_json(["c1"])


if __name__ == "__main__":
    test_data_version_not_code_digest()
    test_resume_matches_continuous()
    test_same_ckpt_train_val_numbers_are_from_one_state()
    print("test_checkpoint PASS")
