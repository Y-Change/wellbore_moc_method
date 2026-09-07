# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from dataset import (
    N_MAX, Round3Dataset, collate_predict, compute_norm, load_round1_manifest,
    nested_ids, nested_order,
)
from models import QueryFourierMLP
from paths import CONFIG_DIR


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(b)
    num = np.linalg.norm(a - b)
    return float(num / den) if den > 1e-30 else float(num), float(num)


def test_batch_invariance():
    cfg = yaml.safe_load((CONFIG_DIR / "train_round3.yaml").read_text(encoding="utf-8"))
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ds = Round3Dataset("train", cfg, man, case_ids=nested_ids(order, 3))
    norm = compute_norm(ds, max_cases=3)
    torch.manual_seed(0)
    model = QueryFourierMLP(8 + 3 * N_MAX, d=32, n_harmonics=8, n_layers=3)
    torch.nn.init.normal_(model.net[-1].weight, std=0.05)
    torch.nn.init.zeros_(model.net[-1].bias)
    model.eval()
    p_scale = float(norm["p_pert_scale"])

    def pred(bundles, pad=0):
        pb = collate_predict(bundles, norm, pad_extra=pad)
        with torch.no_grad():
            out = model.predict(
                torch.from_numpy(pb["feat"]),
                torch.from_numpy(pb["tau"]),
                p_scale,
            )
        y = out["p_pert"].numpy()
        m = pb["mask"]
        return y, m, pb["case_ids"]

    y1, m1, _ = pred([ds[0]], pad=0)
    T0 = int(m1[0].sum())
    y_partners, m_p, ids_p = pred([ds[0], ds[1]], pad=0)
    y_order, _, ids_o = pred([ds[1], ds[0]], pad=0)
    y_pad, m_pad, _ = pred([ds[0]], pad=17)
    rel_p, abs_p = _rel(y_partners[0, :T0], y1[0, :T0])
    rel_o, abs_o = _rel(y_order[1, :T0], y1[0, :T0])
    rel_pad, abs_pad = _rel(y_pad[0, :T0], y1[0, :T0])
    near_zero = np.linalg.norm(y1[0, :T0]) < 1e-8
    tol = 1e-5
    assert ids_p[0] == ds[0].physical.case_id
    assert ids_o[1] == ds[0].physical.case_id
    if not near_zero:
        assert rel_p <= tol, rel_p
        assert rel_o <= tol, rel_o
        assert rel_pad <= tol, rel_pad
    else:
        assert abs_p <= 1e-8 and abs_o <= 1e-8 and abs_pad <= 1e-8
    pb0 = collate_predict([ds[0]], norm, 0)
    pb17 = collate_predict([ds[0]], norm, 17)
    assert np.allclose(pb0["tau"][0, :T0], pb17["tau"][0, :T0])
    assert float(pb17["mask"][0, T0:].sum()) == 0.0
    print({
        "rel_partners": rel_p, "rel_order": rel_o, "rel_pad": rel_pad,
        "status": "PASS",
    })


if __name__ == "__main__":
    test_batch_invariance()
    print("test_batch_consistency PASS")
