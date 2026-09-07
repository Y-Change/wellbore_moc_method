# -*- coding: utf-8 -*-
"""Explicit frozen-kernel load. Missing or mismatched YAML is a hard error.

Never call moc_solver.get_kernel_fits() in Round 4: that path can silently
refit when config_io cannot see PaperC/configs/friction_zielke.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from common import sha256_file, sha256_json
from r4_paths import FROZEN_KERNEL_YAML
from stage1_bridge import mk, stage1_paths


class FrozenKernelError(RuntimeError):
    pass


EXPECTED_M = 12
EXPECTED_TAU = (1.0e-8, 100.0)


@dataclass
class FrozenKernel:
    yaml_path: Path
    yaml_sha256: str
    M: int
    domain_tau: Tuple[float, float]
    fz: Any
    fp: Any
    zielke_m: np.ndarray
    zielke_n: np.ndarray
    powerlaw_m: np.ndarray
    powerlaw_n: np.ndarray
    audit: Dict
    used_online_refit: bool = False

    def as_fits(self):
        return (self.fz, self.fp)


def _close(a, b, rtol=1e-12, atol=1e-15) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                            rtol=rtol, atol=atol))


def load_frozen_kernel(yaml_path: Path = None) -> FrozenKernel:
    path = Path(yaml_path or FROZEN_KERNEL_YAML)
    if not path.is_file():
        raise FrozenKernelError(f"frozen kernel YAML missing: {path}")
    raw = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw)
    sha = sha256_file(path)
    try:
        M = int(cfg["production_path"]["M"])
        dom = cfg["production_path"]["fit"]["domain_tau"]
        tau = (float(dom[0]), float(dom[1]))
        block = cfg["fits_by_M"][M]
        zl = block["zielke_laminar"]
        pl = block["powerlaw_for_vardy_brown"]
    except Exception as exc:
        raise FrozenKernelError(f"frozen kernel YAML unreadable: {exc}") from exc
    if M != EXPECTED_M:
        raise FrozenKernelError(f"production M={M}, expected {EXPECTED_M}")
    if abs(tau[0] - EXPECTED_TAU[0]) > 1e-30 or abs(tau[1] - EXPECTED_TAU[1]) > 1e-12:
        raise FrozenKernelError(f"domain_tau={tau}, expected {EXPECTED_TAU}")
    if int(zl["M"]) != M or int(pl["M"]) != M:
        raise FrozenKernelError("fit M does not match production M")
    # Stage-1 CONFIG_DIR must still point at PaperC/configs
    cfg_dir = Path(stage1_paths.CONFIG_DIR)
    if cfg_dir.resolve() != path.parent.resolve():
        raise FrozenKernelError(
            f"Stage-1 CONFIG_DIR={cfg_dir} is not the frozen YAML parent {path.parent}"
        )
    fz, fp = mk.kernel_fits_from_config(cfg, M)
    if not _close(fz.m, zl["m"]) or not _close(fz.n, zl["n"]):
        raise FrozenKernelError("Zielke coefficients do not match YAML; refuse refit")
    if not _close(fp.m, pl["m"]) or not _close(fp.n, pl["n"]):
        raise FrozenKernelError("power-law coefficients do not match YAML; refuse refit")
    audit = {
        "yaml_path": str(path),
        "yaml_sha256": sha,
        "M": M,
        "domain_tau": list(tau),
        "powerlaw_domain_tau": list(cfg["production_path"]["fit"]["powerlaw_domain_tau"]),
        "zielke_max_rel_err": float(zl["max_rel_err"]),
        "powerlaw_max_rel_err": float(pl["max_rel_err"]),
        "zielke_m_sha256": sha256_json(np.asarray(zl["m"], dtype=np.float64).tolist()),
        "zielke_n_sha256": sha256_json(np.asarray(zl["n"], dtype=np.float64).tolist()),
        "powerlaw_m_sha256": sha256_json(np.asarray(pl["m"], dtype=np.float64).tolist()),
        "powerlaw_n_sha256": sha256_json(np.asarray(pl["n"], dtype=np.float64).tolist()),
        "used_online_refit": False,
        "stage1_CONFIG_DIR": str(cfg_dir),
        "load_method": "yaml.safe_load + memory_kernel.kernel_fits_from_config",
        "not_used": "moc_solver.get_kernel_fits",
    }
    return FrozenKernel(
        yaml_path=path, yaml_sha256=sha, M=M, domain_tau=tau, fz=fz, fp=fp,
        zielke_m=np.asarray(zl["m"], dtype=np.float64),
        zielke_n=np.asarray(zl["n"], dtype=np.float64),
        powerlaw_m=np.asarray(pl["m"], dtype=np.float64),
        powerlaw_n=np.asarray(pl["n"], dtype=np.float64),
        audit=audit, used_online_refit=False,
    )
