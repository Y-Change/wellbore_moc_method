# -*- coding: utf-8 -*-
"""Paths for Stage-2 pilot preliminary only.

New outputs stay under 02_cjno_forward_operator/pilot_preliminary/.
Stage-1 goldens / manifests / runs are read-only.
"""
from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PILOT_PRELIM = CODE_DIR.parent
STAGE2_DIR = PILOT_PRELIM.parent
PAPER_ROOT = STAGE2_DIR.parent
STAGE1_DIR = PAPER_ROOT / "01_moc_benchmark"
STAGE1_CODE = STAGE1_DIR / "code"
STAGE1_DATA = STAGE1_DIR / "data"
STAGE1_MANIFEST = STAGE1_DATA / "manifests" / "manifest_v1.json"
STAGE1_PILOT_CASES = STAGE1_DATA / "pilot_2000" / "cases"
STAGE1_GOLDENS = STAGE1_DATA / "goldens"
CONFIGS_ROOT = PAPER_ROOT / "configs"

CONFIG_DIR = PILOT_PRELIM / "configs"
DATA_AUDIT_DIR = PILOT_PRELIM / "data_audit"
MANIFEST_DIR = PILOT_PRELIM / "manifests"
RUNS_DIR = PILOT_PRELIM / "runs"
CKPT_DIR = PILOT_PRELIM / "checkpoints"
FIG_DIR = PILOT_PRELIM / "figures"
TABLE_DIR = PILOT_PRELIM / "tables"
DOCS_DIR = PILOT_PRELIM / "docs"
CACHE_DIR = PILOT_PRELIM / "cache"


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, DATA_AUDIT_DIR, MANIFEST_DIR, RUNS_DIR, CKPT_DIR,
              FIG_DIR, TABLE_DIR, DOCS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
