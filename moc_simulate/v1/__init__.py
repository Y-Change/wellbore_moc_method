# -*- coding: utf-8 -*-
"""
moc_simulate.v1

原始 V1 MOC 基线版本（完整保留供历史数据对比、基线对标与文献复现）
"""
from __future__ import annotations

from moc_simulate.v1.wellbore_moc import (
    MocConfig,
    simulate_wellbore,
    simulate_case,
    solve_fracture_node,
    compute_steady_state_profile,
    darcy_friction_factor,
    reynolds,
    G,
)
from moc_simulate.v1.config import (
    WELL_CONFIG,
    FRACTURE_CONFIG,
    SIM_CONFIG,
    CEPSTRUM_CONFIG,
    FRICTION_PARAMS,
    CASES,
    build_cases,
    expand_friction_keys,
    friction_cli_choices,
)
from moc_simulate.v1.cepstrum_mocdata import (
    compute_moc_cepstrum,
    compute_moc_cepstrum_1d,
    cepstrogram,
    detect_1d_cepstrum_peaks,
    plot_moc_cepstrum_analysis,
    plot_moc_cepstrum_fracture_zoom,
    evaluate_1d_cepstrum_fracture_match,
    cepstrum_match_summary_for_json,
)

# 别名供通用调用
solve_moc = simulate_wellbore
FRICTION_CONFIGS = FRICTION_PARAMS
FRACTURE_CASES = CASES

__all__ = [
    "MocConfig",
    "simulate_wellbore",
    "simulate_case",
    "solve_moc",
    "solve_fracture_node",
    "compute_steady_state_profile",
    "darcy_friction_factor",
    "reynolds",
    "G",
    "WELL_CONFIG",
    "FRACTURE_CONFIG",
    "SIM_CONFIG",
    "CEPSTRUM_CONFIG",
    "FRICTION_PARAMS",
    "FRICTION_CONFIGS",
    "CASES",
    "FRACTURE_CASES",
    "build_cases",
    "expand_friction_keys",
    "friction_cli_choices",
    "compute_moc_cepstrum",
    "compute_moc_cepstrum_1d",
    "cepstrogram",
    "detect_1d_cepstrum_peaks",
    "plot_moc_cepstrum_analysis",
    "plot_moc_cepstrum_fracture_zoom",
    "evaluate_1d_cepstrum_fracture_match",
    "cepstrum_match_summary_for_json",
]
