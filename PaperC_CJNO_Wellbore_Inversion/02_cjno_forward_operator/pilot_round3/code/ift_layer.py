# -*- coding: utf-8 -*-
"""Verified IFT layer from round1, loaded without shadowing round3 paths."""
from __future__ import annotations

import importlib.util
import sys

from paths import ROUND1, STAGE1_CODE

_saved = sys.modules.get("paths")
sys.path.insert(0, str(STAGE1_CODE))
_spec = importlib.util.spec_from_file_location(
    "round1_interface_newton", ROUND1 / "code" / "interface_newton.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
if _saved is not None:
    sys.modules["paths"] = _saved

cluster_newton = _mod.cluster_newton
ImplicitNodeLayer = _mod.ImplicitNodeLayer
reduced_residual_and_jac = _mod.reduced_residual_and_jac
COND_LIMIT = _mod.COND_LIMIT
