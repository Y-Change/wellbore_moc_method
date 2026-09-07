# -*- coding: utf-8 -*-
"""Canonical paths for the Stage-1 archive tree (攻关执行方案_v2 §1).

Everything produced by Stage 1 must live below ``PaperC_CJNO_Wellbore_Inversion/``.
"""
from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
STAGE_DIR = CODE_DIR.parent                      # 01_moc_benchmark
PAPER_ROOT = STAGE_DIR.parent                    # PaperC_CJNO_Wellbore_Inversion
REPO_ROOT = PAPER_ROOT.parent                    # wellbore_moc_method

CONFIG_DIR = PAPER_ROOT / "configs"
SCHEMA_DIR = CONFIG_DIR / "schemas"

DATA_DIR = STAGE_DIR / "data"
GOLDEN_DIR = DATA_DIR / "goldens"
PILOT_DIR = DATA_DIR / "pilot_2000"
SOBOL_DIR = DATA_DIR / "sobol_30000"
OOD_DIR = DATA_DIR / "ood"
MANIFEST_DIR = DATA_DIR / "manifests"
RUNS_DIR = DATA_DIR / "runs"
CEPSTRUM_PILOT_DIR = DATA_DIR / "cepstrum_pilot10"

FIGURE_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, SCHEMA_DIR, DATA_DIR, GOLDEN_DIR, PILOT_DIR, SOBOL_DIR,
              OOD_DIR, MANIFEST_DIR, RUNS_DIR, CEPSTRUM_PILOT_DIR, FIGURE_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
