# -*- coding: utf-8 -*-
"""
scenarios.py — 与 P0 字典 / lhs_dataset_2000 对齐的效率检验场景。

物理配置取自 analysis/method_migration/build_dictionary.py，使
CRB、全波形 MLE、P0 字典剥离、倒谱落在同一根物理轴上。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from analysis.identifiability.crb_core import ObsModel, Scenario

# ---- 与 build_dictionary.py 对齐的物理常数 ----
L = 5000.0
A_WAVE = 1450.0
DIA = 0.1397
DT = 1.0e-3
TF = 50.0
TS = 1.0
V0 = 1.0
H0 = 300.0
H_EXT = 100.0
ROUGH = 4.5e-5
VISC = 1.0e-6
FRICTION = "brunone"
CF_DEFAULT = 1.0e-6
KLEAK_DEFAULT = 1.0e-4

# 效率检验默认观测带宽（差分收敛域内；dt=1ms → fc≤20 Hz）
FC_DEFAULT = 20.0


@dataclass(frozen=True)
class EfficiencyCase:
    """一次效率检验的真值构型。"""

    name: str
    x_f: Tuple[float, ...]
    Cf: Tuple[float, ...]
    kleak: Tuple[float, ...]
    spacing_m: float  # 0 表示单缝


def make_scenario(
    case: EfficiencyCase,
    *,
    friction: str = FRICTION,
    dt: float = DT,
    tf: float = TF,
    k_scale: float = 1.0,
) -> Scenario:
    return Scenario(
        x_f=case.x_f,
        Cf=case.Cf,
        kleak=case.kleak,
        L=L,
        diameter=DIA,
        a_nominal=A_WAVE,
        dt=dt,
        tf=tf,
        ts=TS,
        V0=V0,
        H0=H0,
        H_ext=H_EXT,
        toe_bc="reservoir",
        friction=friction,
        k_scale=k_scale,
        viscosity=VISC,
        roughness=ROUGH,
    )


def make_obs(fc_hz: float = FC_DEFAULT) -> ObsModel:
    return ObsModel(fc_hz=fc_hz, order=4)


def default_cases() -> List[EfficiencyCase]:
    """计划约定的场景：单缝 / 双缝 / 高阶（n=3/4/5）。"""
    cases = [
        EfficiencyCase(
            name="single_3000",
            x_f=(3000.0,),
            Cf=(CF_DEFAULT,),
            kleak=(KLEAK_DEFAULT,),
            spacing_m=0.0,
        ),
        EfficiencyCase(
            name="single_4000",

            x_f=(4000.0,),
            Cf=(CF_DEFAULT,),
            kleak=(KLEAK_DEFAULT,),
            spacing_m=0.0,
        ),
        EfficiencyCase(
            name="dual_10m",
            x_f=(4000.0, 4010.0),
            Cf=(CF_DEFAULT, CF_DEFAULT),
            kleak=(KLEAK_DEFAULT, KLEAK_DEFAULT),
            spacing_m=10.0,
        ),
        EfficiencyCase(
            name="dual_40m",
            x_f=(4000.0, 4040.0),
            Cf=(CF_DEFAULT, CF_DEFAULT),
            kleak=(KLEAK_DEFAULT, KLEAK_DEFAULT),
            spacing_m=40.0,
        ),
    ]
    # 高阶：等间距簇，锚在 4000 m
    for n, spacing, tag in [
        (3, 10.0, "triple_10m"),
        (3, 40.0, "triple_40m"),
        (4, 10.0, "quad_10m"),
        (4, 40.0, "quad_40m"),
        (5, 10.0, "quint_10m"),
        (5, 40.0, "quint_40m"),
    ]:
        xs = tuple(4000.0 + i * spacing for i in range(n))
        cases.append(
            EfficiencyCase(
                name=tag,
                x_f=xs,
                Cf=tuple(CF_DEFAULT for _ in range(n)),
                kleak=tuple(KLEAK_DEFAULT for _ in range(n)),
                spacing_m=spacing,
            )
        )
    return cases


def highorder_cases() -> List[EfficiencyCase]:
    """仅 n≥3 场景。"""
    return [c for c in default_cases() if len(c.x_f) >= 3]


def case_by_name(name: str) -> EfficiencyCase:
    for c in default_cases():
        if c.name == name:
            return c
    raise KeyError(f"未知场景 {name}；可选 {[c.name for c in default_cases()]}")


def scenario_summary(scn: Scenario) -> Dict[str, object]:
    n = scn.n_cells_nominal()
    return {
        "x_f": list(scn.x_f),
        "Cf": list(scn.Cf),
        "kleak": list(scn.kleak),
        "L": scn.L,
        "a_nominal": scn.a_nominal,
        "a_adj": scn.a_for(n),
        "dx": scn.dx_for(n),
        "n_cells": n,
        "dt": scn.dt,
        "tf": scn.tf,
        "friction": scn.friction,
        "k_scale": scn.k_scale,
    }
