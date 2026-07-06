# -*- coding: utf-8 -*-
"""自研 MOC 方法目录与分级 output 路径。"""
from __future__ import annotations

import os
import sys
from typing import Optional

METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(METHOD_DIR, "output")
DOCS_DIR = os.path.join(METHOD_DIR, "docs")

# L1 系列名（与 output/ 子目录一致）
SERIES_STEP01_JOUKOWSKY = "step01_joukowsky"
SERIES_STEP03_FRACTURE = "step03_fracture"
SERIES_STEP03B_BRUNONE = "step03b_brunone"
SERIES_STEP04A_DUAL = "step04a_dual_fracture"
SERIES_STEP04A_FRICTION = "step04a_friction_only"
SERIES_TEST_B = "test_b"
SERIES_CEPSTRUM_KB = "cepstrum/kaiser_bessel"
SERIES_CEPSTRUM_WLEN_SWEEP = "cepstrum/wlen_sweep"
SERIES_ANALYSIS_WINDOW = "analysis/window_comparison"

# L2 用例（缝数）
CASE_SINGLE = "single"
CASE_DUAL = "dual"
CASE_TRIPLE = "triple"
CASE_QUAD = "quad"
CASE_QUINT = "quint"


def ensure_method_root_on_path() -> None:
    """确保项目根目录在 sys.path 中（validation 子包脚本使用）。"""
    if METHOD_DIR not in sys.path:
        sys.path.insert(0, METHOD_DIR)


def bootstrap_method_root(caller_file: str) -> str:
    """从任意 validation/ 子目录脚本向上查找并加入 sys.path。"""
    d = os.path.dirname(os.path.abspath(caller_file))
    while True:
        if os.path.isfile(os.path.join(d, "paths.py")) and os.path.isfile(
            os.path.join(d, "wellbore_moc.py")
        ):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"Cannot find wellbore_moc_method root from {caller_file}")
        d = parent


def moc_output_dir() -> str:
    """返回 output 根目录（不存在则创建）。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def moc_output_subdir(series: str, case: Optional[str] = None) -> str:
    """
    返回分级 output 子目录并创建。

    Parameters
    ----------
    series : 验证系列，如 ``test_b``、``cepstrum/kaiser_bessel``、``step01_joukowsky``
    case : 用例名，如 ``single``；Step 验证传 None
    """
    moc_output_dir()
    parts = [OUTPUT_DIR] + series.replace("\\", "/").split("/")
    if case:
        parts.append(case)
    subdir = os.path.join(*parts)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def output_path(series: str, case: Optional[str], filename: str) -> str:
    """返回分级 output 文件的完整路径（父目录自动创建）。"""
    return os.path.join(moc_output_subdir(series, case), filename)
