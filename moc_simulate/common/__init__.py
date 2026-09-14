# -*- coding: utf-8 -*-
"""
moc_simulate.common

共享基础常量与路径工具
"""
from __future__ import annotations

from moc_simulate.common.constants import (
    G,
    G_STANDARD,
    GRAVITY,
    RHO_WATER,
    NU_WATER,
    MU_WATER,
    CASING_ROUGHNESS,
    CASING_OD_5_5,
    WAVESPEED_DEFAULT,
    P_ATM,
    H_ATM,
)
from moc_simulate.common.paths import (
    PROJECT_ROOT,
    PACKAGE_DIR,
    OUTPUT_DIR,
    DOCS_DIR,
    moc_output_dir,
    moc_output_subdir,
    output_path,
    ensure_method_root_on_path,
    bootstrap_method_root,
)

__all__ = [
    "G",
    "G_STANDARD",
    "GRAVITY",
    "RHO_WATER",
    "NU_WATER",
    "MU_WATER",
    "CASING_ROUGHNESS",
    "CASING_OD_5_5",
    "WAVESPEED_DEFAULT",
    "P_ATM",
    "H_ATM",
    "PROJECT_ROOT",
    "PACKAGE_DIR",
    "OUTPUT_DIR",
    "DOCS_DIR",
    "moc_output_dir",
    "moc_output_subdir",
    "output_path",
    "ensure_method_root_on_path",
    "bootstrap_method_root",
]
