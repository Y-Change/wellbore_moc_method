# -*- coding: utf-8 -*-
"""Reproducible entry point (攻关执行方案_v2 §1 / §8).

    python run_all_reproducible.py --stage 1 --seed 42 --device cpu --smoke
    python run_all_reproducible.py --stage 1 --seed 42 --device cpu --full
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent / "01_moc_benchmark" / "code"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage != 1:
        print(f"Stage {args.stage} is not implemented yet (Stage 1 only).")
        sys.exit(2)
    smoke = args.smoke or not args.full
    cmd = [sys.executable, str(CODE / "validate.py")]
    if smoke:
        cmd.append("--smoke")
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(CODE))
    if rc != 0:
        sys.exit(rc)
    if args.full:
        cmd2 = [sys.executable, str(CODE / "generate_dataset.py"), "--pilot",
                "--workers", "1"]
        print(" ".join(cmd2), flush=True)
        rc = subprocess.call(cmd2, cwd=str(CODE))
        sys.exit(rc)


if __name__ == "__main__":
    main()
