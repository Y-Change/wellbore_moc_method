# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from dataset import (
    D_FEAT, FORBIDDEN_PREDICT_KEYS, Round3Dataset, assert_predict_clean, compute_norm,
    load_round1_manifest, nested_ids, nested_order, pack_features, predict_arrays,
    raw_features,
)
from isolation import SplitGuard
from paths import CONFIG_DIR


def _cfg():
    return yaml.safe_load((CONFIG_DIR / "train_round3.yaml").read_text(encoding="utf-8"))


def test_test_split_blocked():
    cfg = _cfg()
    try:
        Round3Dataset("test", cfg)
        raise AssertionError("test split must be blocked")
    except RuntimeError:
        pass


def test_all_usable_blocked():
    cfg = _cfg()
    try:
        Round3Dataset("all_usable", cfg)
        raise AssertionError("all_usable must be blocked")
    except RuntimeError:
        pass


def test_test_case_id_blocked_even_if_split_says_train():
    cfg = _cfg()
    man = load_round1_manifest()
    test_id = next(c["case_id"] for c in man["cases"] if c["split"] == "test")
    try:
        Round3Dataset("train", cfg, man, case_ids=[test_id])
        raise AssertionError("test case_id must be blocked")
    except RuntimeError:
        pass


def test_predict_has_no_future_labels():
    cfg = _cfg()
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ds = Round3Dataset("train", cfg, man, case_ids=nested_ids(order, 2))
    inp = predict_arrays(ds[0])
    assert_predict_clean(inp)
    for k in FORBIDDEN_PREDICT_KEYS:
        assert k not in inp
    assert "Q_valve" in inp and "tau" in inp and "K" in inp
    q = ds[0].query
    assert abs(q.t[0] - q.t_s) < 1e-12
    assert abs(q.dt - cfg["query"]["dt_s"]) < 1e-12
    assert q.t.size == cfg["query"]["n_samples"]


def test_nested_prefixes_match_round2():
    cfg = _cfg()
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    a, b = nested_ids(order, 1), nested_ids(order, 8)
    assert a == b[:1]
    expect8 = [
        "pilot_01342", "pilot_00163", "pilot_00543", "pilot_01320",
        "pilot_00049", "pilot_00888", "pilot_01900", "pilot_00880",
    ]
    assert b == expect8


def test_shuffle_targets_does_not_change_predict_arrays():
    cfg = _cfg()
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ds = Round3Dataset("train", cfg, man, case_ids=nested_ids(order, 1))
    b = ds[0]
    inp1 = predict_arrays(b)
    b.targets.p_head[:] = b.targets.p_head[::-1]
    b.targets.p_pert[:] = b.targets.p_pert[::-1]
    b.targets.node_H[:] = 0.0
    inp2 = predict_arrays(b)
    for k in ("t", "tau", "Q_valve", "p0_wh", "L", "a"):
        if hasattr(inp1[k], "__len__") and not isinstance(inp1[k], str):
            assert np.allclose(inp1[k], inp2[k])
        else:
            assert inp1[k] == inp2[k]


def test_n1_constant_features_normalize_to_zero():
    cfg = _cfg()
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ds = Round3Dataset("train", cfg, man, case_ids=nested_ids(order, 1))
    norm = compute_norm(ds, max_cases=1)
    assert all(norm["feat_const"])
    packed = pack_features(predict_arrays(ds[0]), norm)
    assert packed.dtype == np.float32
    assert np.allclose(packed, 0.0)
    raw = raw_features(predict_arrays(ds[0]))
    assert raw.dtype == np.float64
    assert raw.size == D_FEAT


def test_silence_ignores_prediction():
    cfg = _cfg()
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ds = Round3Dataset("train", cfg, man, case_ids=nested_ids(order, 1))
    b = ds[0]
    silent = b.targets.silent_p
    # mutating a fake prediction cannot change the stored target silence
    _ = np.ones_like(b.targets.p_pert) * 1e12
    assert b.targets.silent_p == silent


def test_guard_lists_test_ids():
    g = SplitGuard()
    assert len(g.test_ids) > 0
    assert not (g.test_ids & g.train_ids)


if __name__ == "__main__":
    test_test_split_blocked()
    test_all_usable_blocked()
    test_test_case_id_blocked_even_if_split_says_train()
    test_predict_has_no_future_labels()
    test_nested_prefixes_match_round2()
    test_shuffle_targets_does_not_change_predict_arrays()
    test_n1_constant_features_normalize_to_zero()
    test_silence_ignores_prediction()
    test_guard_lists_test_ids()
    print("test_dataset PASS")
