# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from zvb_recursion import full_kernel_numpy_vs_tensor, unit_one_term_hand


def test_unit():
    rec = unit_one_term_hand()
    assert rec["pass"], rec


def test_full():
    rec = full_kernel_numpy_vs_tensor()
    assert rec["pass"], rec


if __name__ == "__main__":
    test_unit()
    test_full()
    print("zvb recursion tests PASS")
