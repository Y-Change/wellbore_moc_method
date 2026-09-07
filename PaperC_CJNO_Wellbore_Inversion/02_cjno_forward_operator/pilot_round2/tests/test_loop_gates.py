# -*- coding: utf-8 -*-
from __future__ import annotations

from run_loop_gates import (
    gate_causality, gate_hard_counts, gate_newton_failure_blocks,
    gate_shuffle_labels_unused, gate_wellhead_grad,
)


def test_loop_gates_core():
    g = {
        "causality": gate_causality(),
        "hard": gate_hard_counts(),
        "shuffle": gate_shuffle_labels_unused(),
        "grad": gate_wellhead_grad(),
        "block": gate_newton_failure_blocks(),
    }
    failed = [k for k, v in g.items() if not v["pass"]]
    assert not failed, g
    print({k: v["pass"] for k, v in g.items()})


if __name__ == "__main__":
    test_loop_gates_core()
    print("test_loop_gates PASS")
