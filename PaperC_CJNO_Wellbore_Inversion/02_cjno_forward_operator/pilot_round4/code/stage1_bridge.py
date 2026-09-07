# -*- coding: utf-8 -*-
"""Import Stage-1 MOC without a Round-4 module named paths.

After this import, sys.modules['paths'] is the Stage-1 paths module so that
config_io.CONFIG_DIR stays PaperC/configs. Round-4 code must use r4_paths.
"""
from __future__ import annotations

import importlib
import sys

from r4_paths import STAGE1_CODE

_STAGE1_NAMES = (
    "paths", "config_io", "memory_kernel", "friction", "grid",
    "node_newton", "moc_solver",
)


def _install_stage1() -> dict:
    saved = {name: sys.modules[name] for name in _STAGE1_NAMES if name in sys.modules}
    for name in _STAGE1_NAMES:
        if name in sys.modules:
            del sys.modules[name]
    if str(STAGE1_CODE) not in sys.path:
        sys.path.insert(0, str(STAGE1_CODE))
    import moc_solver as ms  # noqa: F401
    import grid as _grid  # noqa: F401
    import node_newton as _nn  # noqa: F401
    import memory_kernel as _mk  # noqa: F401
    import friction as _fr  # noqa: F401
    import config_io as _cio  # noqa: F401
    import paths as _sp  # noqa: F401
    # Keep Stage-1 paths installed; do not restore a Round-4 paths.
    return saved


_saved_foreign = _install_stage1()

import moc_solver as ms  # noqa: E402
from grid import build_grid  # noqa: E402
from node_newton import (  # noqa: E402
    branch_coefficients, flow_from_characteristic, solve_cluster_node,
)
import memory_kernel as mk  # noqa: E402
from friction import FrictionTable  # noqa: E402
import paths as stage1_paths  # noqa: E402


def assert_stage1_paths_bound() -> None:
    p = sys.modules.get("paths")
    if p is None or getattr(p, "CONFIG_DIR", None) is None:
        raise RuntimeError("Stage-1 paths is not bound; refuse to continue")
    cfg = str(p.CONFIG_DIR).replace("\\", "/").lower()
    if "paperc_cjno_wellbore_inversion/configs" not in cfg and "paperc_cjno_wellbore_inversion\\configs" not in str(p.CONFIG_DIR).lower():
        # Accept either slash style
        if not str(p.CONFIG_DIR).endswith("configs") or "pilot_round" in str(p.CONFIG_DIR):
            raise RuntimeError(f"Stage-1 CONFIG_DIR rebound to {p.CONFIG_DIR}; refuse silent refit")


assert_stage1_paths_bound()


def case_to_well(c: dict):
    w = c["well"]
    cls = [ms.ClusterSpec.from_perf_params(
        x=cl["x"], Cd=cl["Cd"], A_perf=cl["A_perf"], kappa=cl["kappa"], I_f=cl["I_f"],
        R_f=cl["R_f"], C_f=cl["C_f"], G_l=cl["G_l"], p_res=cl["p_res"], rho=w["rho"])
        for cl in c["clusters"]]
    return ms.WellSpec(L=w["L"], D=w["D"], a=w["a"], rho=w["rho"], nu=w["nu"],
                       roughness=w["roughness"], TVD=c["TVD"], clusters=cls,
                       Q0=w["Q0"], t_s=1.0, t_c=w["t_c"], ramp="cosine", toe_bc="dead_end")


F_DEFINITION = {
    "symbol": "F",
    "discrete_def": "F^{n+1} = p_in^{n+1} - p_c^{n+1} = c1^{n->n+1} q^{n+1} + c0^{n->n+1}",
    "units": "Pa",
    "update_time": "after each successful cluster Newton at the new time level",
    "not": "F=0 or copied mean pc",
}

SHARED_PRIMITIVES = [
    "grid.build_grid",
    "moc_solver.solve_steady_state",
    "moc_solver._steady_head_drop",
    "moc_solver._f_lookup / FrictionTable",
    "moc_solver._wellhead_Q",
    "node_newton.branch_coefficients",
    "node_newton.flow_from_characteristic",
    "node_newton.solve_cluster_node",
    "memory_kernel.kernel_fits_from_config / RecursiveKernel / kernel_for_reynolds",
]
