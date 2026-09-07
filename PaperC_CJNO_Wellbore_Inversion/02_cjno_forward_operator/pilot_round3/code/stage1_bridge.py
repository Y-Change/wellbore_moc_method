# -*- coding: utf-8 -*-
"""Import Stage-1 MOC without shadowing Round-3 paths."""
from __future__ import annotations

import sys

from paths import STAGE1_CODE

_saved = sys.modules.get("paths")
if "paths" in sys.modules:
    del sys.modules["paths"]
sys.path.insert(0, str(STAGE1_CODE))
import moc_solver as ms  # noqa: E402
from grid import build_grid  # noqa: E402
from node_newton import branch_coefficients, flow_from_characteristic, solve_cluster_node  # noqa: E402
if _saved is not None:
    sys.modules["paths"] = _saved


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
