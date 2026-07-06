# -*- coding: utf-8 -*-
"""[legacy] 兼容 wrapper → validation/test_b/single.py"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.test_b.single import run_validation

if __name__ == '__main__':
    run_validation()
