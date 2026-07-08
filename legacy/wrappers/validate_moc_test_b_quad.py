# -*- coding: utf-8 -*-
"""[legacy] 兼容 wrapper → validation/leakoff_multi.py --friction steady --case quad"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.leakoff_multi import run_case

if __name__ == '__main__':
    run_case('quad', friction='steady')
