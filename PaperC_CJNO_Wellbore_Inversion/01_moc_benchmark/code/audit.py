# -*- coding: utf-8 -*-
"""Run audit utilities (攻关执行方案_v2 §1 运行审计).

Every run gets a unique ``run_id`` and a directory ``data/runs/<run_id>/`` holding
``resolved_config.yaml``, ``run_meta.json`` (git commit, code digest, config digest,
manifest digest, RNG state, environment, wall-clock), ``stdout.log`` and ``metrics.json``.
Figures must carry ``run_id`` + the first 8 hex digits of the manifest digest as a margin note.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import yaml

from paths import CODE_DIR, CONFIG_DIR, REPO_ROOT, RUNS_DIR


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def digest_of_files(paths: Iterable[Path]) -> str:
    """Order-independent digest of a set of files (sorted by relative name)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda q: str(q)):
        h.update(str(p.name).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def code_digest() -> str:
    return digest_of_files(sorted(CODE_DIR.glob("*.py")))


def config_digest() -> str:
    files = list(CONFIG_DIR.glob("*.yaml")) + list((CONFIG_DIR / "schemas").glob("*.json"))
    return digest_of_files(files) if files else ""


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "PaperC_CJNO_Wellbore_Inversion"],
                             capture_output=True, text=True, timeout=20)
        return bool(out.stdout.strip())
    except Exception:
        return True


def environment_info() -> Dict:
    import numba
    import scipy
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    return info


def new_run_id(prefix: str = "s1") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class RunAudit:
    """Context object writing the audit trail of one run."""

    def __init__(self, name: str, seed: int = 42, run_id: Optional[str] = None, resolved_config: Optional[Dict] = None,
                 manifest_digest: str = ""):
        self.name = name
        self.seed = seed
        self.run_id = run_id or new_run_id(name)
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        self.rng = np.random.default_rng(seed)
        self.manifest_digest = manifest_digest
        self.meta = {
            "run_id": self.run_id,
            "name": name,
            "seed": seed,
            "started": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "git_dirty_paperc": git_dirty(),
            "code_digest": code_digest(),
            "config_digest": config_digest(),
            "manifest_digest": manifest_digest,
            "rng_bit_generator": "PCG64",
            "rng_state": json.loads(json.dumps(self.rng.bit_generator.state, default=str)),
            "environment": environment_info(),
            "argv": sys.argv,
        }
        if resolved_config is not None:
            with open(self.dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(resolved_config, f, allow_unicode=True, sort_keys=False)
        self._log = open(self.dir / "stdout.log", "a", encoding="utf-8")
        self.write_meta()

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def write_meta(self) -> None:
        self.meta["wall_clock_s"] = time.perf_counter() - self.t0
        with open(self.dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False, default=str)

    def write_metrics(self, metrics: Dict, also_to: Optional[Path] = None) -> None:
        metrics = dict(metrics)
        metrics["run_id"] = self.run_id
        metrics["code_digest"] = self.meta["code_digest"]
        metrics["config_digest"] = self.meta["config_digest"]
        metrics["manifest_digest"] = self.manifest_digest
        metrics["git_commit"] = self.meta["git_commit"]
        with open(self.dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)
        if also_to is not None:
            also_to.parent.mkdir(parents=True, exist_ok=True)
            with open(also_to, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)
        self.write_meta()

    def provenance_tag(self) -> str:
        d = self.manifest_digest[:8] if self.manifest_digest else self.meta["code_digest"][:8]
        return f"run_id={self.run_id} | digest={d}"

    def close(self) -> None:
        self.meta["finished"] = datetime.now().isoformat(timespec="seconds")
        self.write_meta()
        self._log.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
