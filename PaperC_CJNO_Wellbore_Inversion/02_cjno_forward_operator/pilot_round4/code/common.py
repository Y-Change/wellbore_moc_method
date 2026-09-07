# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from r4_paths import CODE_DIR, PAPER_ROOT, RUNS_DIR, STAGE1_CODE, ensure_dirs


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def code_digest(root: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o.as_posix())
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PAPER_ROOT), text=True).strip()
    except Exception:
        return "unknown"


def env_record() -> Dict:
    rec = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}
    try:
        import torch
        rec["torch"] = torch.__version__
        rec["cuda"] = bool(torch.cuda.is_available())
        rec["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        rec["torch"] = f"unavailable: {exc}"
        rec["cuda"] = False
        rec["device"] = "cpu"
    return rec


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class RunLogger:
    """One process writes only into its own run directory. No latest/ pointer."""

    def __init__(self, prefix: str, resolved_config: Dict, extra: Optional[Dict] = None):
        ensure_dirs()
        self.run_id = new_run_id(prefix)
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.completed = False
        self.meta = {
            "run_id": self.run_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "code_sha256": code_digest(CODE_DIR),
            "stage1_code_sha256": code_digest(STAGE1_CODE),
            "resolved_config": resolved_config,
            "env": env_record(),
            "protocol": "r4_run_dir_only_no_latest",
            "status": "running",
        }
        if extra:
            self.meta.update(extra)
        dump_json(self.meta, self.dir / "run_meta.json")
        dump_json(resolved_config, self.dir / "resolved_config.json")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.dir / "stdout.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_metrics(self, metrics: Dict) -> None:
        metrics = dict(metrics)
        metrics["run_id"] = self.run_id
        metrics["wall_clock_s"] = time.time() - self.t0
        dump_json(metrics, self.dir / "metrics.json")

    def close(self, status: str = "completed") -> None:
        self.completed = status == "completed"
        self.meta["wall_clock_s"] = time.time() - self.t0
        self.meta["status"] = status
        self.meta["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        dump_json(self.meta, self.dir / "run_meta.json")


def pert_l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    da = a[:n] - a[0]
    db = b[:n] - b[0]
    den = float(np.linalg.norm(db))
    if den < 1e-30:
        return float("nan")
    return float(np.linalg.norm(da - db) / den)


def rel_l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    den = float(np.linalg.norm(b[:n]))
    if den < 1e-30:
        return float("nan")
    return float(np.linalg.norm(a[:n] - b[:n]) / den)


def first_diff_index(a, b, atol=1e-8, rtol=1e-6):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    d = np.abs(a[:n] - b[:n])
    thr = atol + rtol * np.maximum(np.abs(a[:n]), np.abs(b[:n]))
    hit = np.nonzero(d > thr)[0]
    return int(hit[0]) if hit.size else None
