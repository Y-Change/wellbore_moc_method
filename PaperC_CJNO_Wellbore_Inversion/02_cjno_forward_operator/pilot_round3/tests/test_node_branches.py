# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from node_scenarios import ALL_NAMES, scenario


def test_stiff_guard_actually_fires():
    s = scenario("stiff_storage")
    ratio = float((s["phys"]["G_l"] * s["dt"] / s["phys"]["C_f"]).min())
    assert ratio > 1.0
    assert s["stiff_guard"] > 0


def test_reverse_and_near_zero_expectations_registered():
    assert scenario("reverse_flow")["expect"]["q_all_negative"]
    assert scenario("near_zero")["near_zero_band"] == 1e-6


def test_all_named_scenarios_build():
    for name in ALL_NAMES:
        s = scenario(name)
        assert s["name"] == name
        assert s["c1"].size == s["K"].size


if __name__ == "__main__":
    test_stiff_guard_actually_fires()
    test_reverse_and_near_zero_expectations_registered()
    test_all_named_scenarios_build()
    print("test_node_branches PASS")
