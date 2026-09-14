# -*- coding: utf-8 -*-
"""
moc_simulate.v2.core

MOC_V2 核心力学机理与数值算法引擎
"""
from __future__ import annotations

from moc_simulate.v2.core.moc_mesh import (
    MocGrid,
    compute_riemann_invariants,
    update_internal_nodes,
)
from moc_simulate.v2.core.friction import (
    reynolds,
    darcy_friction_factor,
    friction_term_J,
    brunone_k_vec,
    brunone_friction_Ju,
)
from moc_simulate.v2.core.initial_field import compute_steady_state_field
from moc_simulate.v2.core.fracture_node import solve_fracture_node_v2
from moc_simulate.v2.core.boundary_condition import (
    compute_ramp_velocity,
    compute_ramp_acceleration,
    apply_wellhead_bc,
    apply_toe_bc,
)
from moc_simulate.v2.core.solver import (
    WellboreMocV2Solver,
    simulate_v2,
    simulate_wellbore_v2,
)

__all__ = [
    "MocGrid",
    "compute_riemann_invariants",
    "update_internal_nodes",
    "reynolds",
    "darcy_friction_factor",
    "friction_term_J",
    "brunone_k_vec",
    "brunone_friction_Ju",
    "compute_steady_state_field",
    "solve_fracture_node_v2",
    "compute_ramp_velocity",
    "compute_ramp_acceleration",
    "apply_wellhead_bc",
    "apply_toe_bc",
    "WellboreMocV2Solver",
    "simulate_v2",
    "simulate_wellbore_v2",
]
