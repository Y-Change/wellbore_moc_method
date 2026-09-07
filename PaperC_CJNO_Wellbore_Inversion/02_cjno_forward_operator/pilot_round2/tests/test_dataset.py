# -*- coding: utf-8 -*-
from __future__ import annotations

import yaml

from dataset import (
    FORBIDDEN_PREDICT_KEYS, Round2Dataset, assert_predict_clean, compute_norm,
    load_round1_manifest, nested_ids, nested_order, predict_arrays,
)
from paths import CONFIG_DIR


def _cfg():
    return yaml.safe_load((CONFIG_DIR / "train_round2.yaml").read_text(encoding="utf-8"))


def test_test_split_blocked():
    cfg = _cfg()
    try:
        Round2Dataset("test", cfg)
        raise AssertionError("test split must be blocked")
    except ValueError:
        pass


def test_predict_has_no_future_labels():
    cfg = _cfg()
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ds = Round2Dataset("train", cfg, man, case_ids=nested_ids(order, 2))
    inp = predict_arrays(ds[0])
    assert_predict_clean(inp)
    for k in FORBIDDEN_PREDICT_KEYS:
        assert k not in inp
    assert "Q_valve" in inp and "tau" in inp and "K" in inp
    assert inp["init_source"] == "t0_observation"
    q = ds[0].query
    assert abs(q.t[0] - q.t_s) < 1e-12
    assert abs(q.dt - cfg["query"]["dt_s"]) < 1e-12
    assert q.t.size == cfg["query"]["n_samples"]
    assert q.nyquist_hz == 0.5 / q.dt
    # valve / free-response flags
    assert q.valve.sum() > 0
    assert q.free_response.sum() > 0


def test_nested_prefixes():
    cfg = _cfg()
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    a, b, c = nested_ids(order, 1), nested_ids(order, 8), nested_ids(order, 64)
    assert a == b[:1] == c[:1]
    assert b == c[:8]
    assert len(set(order)) == len(order)


def test_shuffle_targets_does_not_change_predict_arrays():
    cfg = _cfg()
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ds = Round2Dataset("train", cfg, man, case_ids=nested_ids(order, 1))
    b = ds[0]
    inp1 = predict_arrays(b)
    b.targets.p_head[:] = b.targets.p_head[::-1]
    b.targets.p_pert[:] = b.targets.p_pert[::-1]
    b.targets.node_H[:] = 0.0
    inp2 = predict_arrays(b)
    for k in ("t", "tau", "Q_valve", "p0_wh", "L", "a"):
        import numpy as np
        if isinstance(inp1[k], (list, tuple)):
            continue
        if hasattr(inp1[k], "__len__") and not isinstance(inp1[k], str):
            assert np.allclose(inp1[k], inp2[k])
        else:
            assert inp1[k] == inp2[k]


if __name__ == "__main__":
    test_test_split_blocked()
    test_predict_has_no_future_labels()
    test_nested_prefixes()
    test_shuffle_targets_does_not_change_predict_arrays()
    print("test_dataset PASS")
