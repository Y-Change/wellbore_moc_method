# -*- coding: utf-8 -*-
"""Load Round-1 cluster_newton without a Round-4 module named paths."""
from __future__ import annotations

import importlib.util
import sys
import types

from r4_paths import ROUND1, STAGE1_CODE

_saved_paths = sys.modules.get("paths")
_shim = types.ModuleType("paths")
_shim.STAGE1_CODE = STAGE1_CODE
sys.modules["paths"] = _shim
if str(STAGE1_CODE) not in sys.path:
    sys.path.insert(0, str(STAGE1_CODE))
_spec = importlib.util.spec_from_file_location(
    "r4_round1_interface_newton", ROUND1 / "code" / "interface_newton.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
# Restore Stage-1 paths so config_io cannot bind a Round-4 CONFIG_DIR.
if _saved_paths is not None:
    sys.modules["paths"] = _saved_paths

cluster_newton = _mod.cluster_newton
reduced_residual_and_jac = _mod.reduced_residual_and_jac
COND_LIMIT = _mod.COND_LIMIT
