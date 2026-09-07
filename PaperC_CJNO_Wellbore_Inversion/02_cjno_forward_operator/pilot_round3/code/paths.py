# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROUND3 = CODE_DIR.parent
STAGE2_DIR = ROUND3.parent
PAPER_ROOT = STAGE2_DIR.parent
STAGE1_DIR = PAPER_ROOT / "01_moc_benchmark"
STAGE1_CODE = STAGE1_DIR / "code"
STAGE1_DATA = STAGE1_DIR / "data"
ROUND1 = STAGE2_DIR / "pilot_preliminary"
ROUND2 = STAGE2_DIR / "pilot_round2"
ROUND1_MANIFEST = ROUND1 / "manifests" / "pilot_manifest.json"
ROUND2_NESTED = ROUND2 / "manifests" / "nested_train_order.json"
ROUND2_REPLAY = ROUND2 / "replay"
CONFIGS_ROOT = PAPER_ROOT / "configs"
CONFIG_DIR = ROUND3 / "configs"
DOCS_DIR = ROUND3 / "docs"
TESTS_DIR = ROUND3 / "tests"
MANIFEST_DIR = ROUND3 / "manifests"
RUNS_DIR = ROUND3 / "runs"
CKPT_DIR = ROUND3 / "checkpoints"
FIG_DIR = ROUND3 / "figures"
TABLE_DIR = ROUND3 / "tables"
REPLAY_DIR = ROUND3 / "replay"


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, DOCS_DIR, TESTS_DIR, MANIFEST_DIR, RUNS_DIR, CKPT_DIR,
              FIG_DIR, TABLE_DIR, REPLAY_DIR, CODE_DIR):
        d.mkdir(parents=True, exist_ok=True)
