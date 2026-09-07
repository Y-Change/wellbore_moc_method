# -*- coding: utf-8 -*-
from __future__ import annotations

from node_scenarios import scenario


def test_stiff_guard_actually_fires():
    s = scenario("stiff_storage")
    ratio = float((s["phys"]["G_l"] * s["dt"] / s["phys"]["C_f"]).min())
    assert ratio > 1.0
    assert s["stiff_guard"] > 0


def test_reverse_and_near_zero_expectations_registered():
    assert scenario("reverse_flow")["expect"]["q_all_negative"]
    assert scenario("near_zero")["near_zero_band"] == 1e-6


if __name__ == "__main__":
    test_stiff_guard_actually_fires()
    test_reverse_and_near_zero_expectations_registered()
    print("test_node_branches PASS")
