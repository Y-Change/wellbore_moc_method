# -*- coding: utf-8 -*-
"""[legacy] 兼容 wrapper → validation/cepstrum/kaiser_bessel_multi.py --case dual"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paths import CASE_DUAL
from validation.cepstrum.kaiser_bessel_multi import run_case

if __name__ == '__main__':
    run_case(CASE_DUAL)
