# -*- coding: utf-8 -*-
"""Round-2 paths. New outputs stay under pilot_round2/. Round-1 is read-only."""
from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROUND2 = CODE_DIR.parent
STAGE2_DIR = ROUND2.parent
PAPER_ROOT = STAGE2_DIR.parent
STAGE1_DIR = PAPER_ROOT / "01_moc_benchmark"
STAGE1_CODE = STAGE1_DIR / "code"
STAGE1_DATA = STAGE1_DIR / "data"
STAGE1_MANIFEST = STAGE1_DATA / "manifests" / "manifest_v1.json"
STAGE1_PILOT_CASES = STAGE1_DATA / "pilot_2000" / "cases"
ROUND1 = STAGE2_DIR / "pilot_preliminary"
ROUND1_MANIFEST = ROUND1 / "manifests" / "pilot_manifest.json"
ROUND1_CONCLUSION = ROUND1 / "pilot_stage_conclusion.md"
CONFIGS_ROOT = PAPER_ROOT / "configs"

CONFIG_DIR = ROUND2 / "configs"
DOCS_DIR = ROUND2 / "docs"
TESTS_DIR = ROUND2 / "tests"
MANIFEST_DIR = ROUND2 / "manifests"
RUNS_DIR = ROUND2 / "runs"
CKPT_DIR = ROUND2 / "checkpoints"
FIG_DIR = ROUND2 / "figures"
TABLE_DIR = ROUND2 / "tables"
REPLAY_DIR = ROUND2 / "replay"


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, DOCS_DIR, TESTS_DIR, MANIFEST_DIR, RUNS_DIR, CKPT_DIR,
              FIG_DIR, TABLE_DIR, REPLAY_DIR, CODE_DIR):
        d.mkdir(parents=True, exist_ok=True)
