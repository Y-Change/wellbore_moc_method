# -*- coding: utf-8 -*-
"""Round-4 paths. This file is intentionally not named paths.py.

Stage-1 config_io / moc_solver import `paths`. Naming this module paths.py
would redirect CONFIG_DIR away from PaperC configs and can trigger a silent
online kernel refit. Round-4 code must `import r4_paths`.
"""
from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROUND4 = CODE_DIR.parent
STAGE2_DIR = ROUND4.parent
PAPER_ROOT = STAGE2_DIR.parent
REPO_ROOT = PAPER_ROOT.parent
STAGE1_DIR = PAPER_ROOT / "01_moc_benchmark"
STAGE1_CODE = STAGE1_DIR / "code"
STAGE1_DATA = STAGE1_DIR / "data"
ROUND1 = STAGE2_DIR / "pilot_preliminary"
ROUND2 = STAGE2_DIR / "pilot_round2"
ROUND3 = STAGE2_DIR / "pilot_round3"
ROUND1_MANIFEST = ROUND1 / "manifests" / "pilot_manifest.json"
ROUND2_NESTED = ROUND2 / "manifests" / "nested_train_order.json"
PAPER_CONFIGS = PAPER_ROOT / "configs"
FROZEN_KERNEL_YAML = PAPER_CONFIGS / "friction_zielke.yaml"
CONFIG_DIR = ROUND4 / "configs"
DOCS_DIR = ROUND4 / "docs"
TESTS_DIR = ROUND4 / "tests"
MANIFEST_DIR = ROUND4 / "manifests"
RUNS_DIR = ROUND4 / "runs"
CKPT_DIR = ROUND4 / "checkpoints"
FIG_DIR = ROUND4 / "figures"
TABLE_DIR = ROUND4 / "tables"
REPLAY_DIR = ROUND4 / "replay"
PYTHON_EXE = Path(r"D:\Anaconda\envs\torch24\python.exe")

TRAIN4 = ("pilot_01342", "pilot_00163", "pilot_00543", "pilot_01320")
TRAIN8 = TRAIN4 + ("pilot_00049", "pilot_00888", "pilot_01900", "pilot_00880")


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, DOCS_DIR, TESTS_DIR, MANIFEST_DIR, RUNS_DIR, CKPT_DIR,
              FIG_DIR, TABLE_DIR, REPLAY_DIR, CODE_DIR):
        d.mkdir(parents=True, exist_ok=True)
