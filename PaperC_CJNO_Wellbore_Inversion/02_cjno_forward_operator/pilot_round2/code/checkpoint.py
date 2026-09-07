# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from common import sha256_json


def data_version(manifest_sha: str, case_ids: List[str], query_protocol: dict) -> dict:
    return {
        "manifest_sha256": manifest_sha,
        "case_ids_sha256": sha256_json(list(case_ids)),
        "n_cases": len(case_ids),
        "query_protocol": query_protocol,
    }


def rng_blob():
    blob = {
        "numpy": np.random.get_state(),
        "python": None,
    }
    blob["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        blob["cuda"] = torch.cuda.get_rng_state_all()
    return blob


def save_checkpoint(path: Path, model, optimizer, cfg, norm, case_ids,
                    manifest_sha, code_sha, epoch, global_step, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "cfg": copy.deepcopy(cfg),
        "norm": copy.deepcopy(norm),
        "case_ids": list(case_ids),
        "data_version": data_version(manifest_sha, case_ids, cfg["query"]),
        "code_sha256": code_sha,
        "rng": rng_blob(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "torch": torch.__version__,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return payload["data_version"]


def load_checkpoint(path: Path, map_location="cpu"):
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if "data_version" not in blob or "manifest_sha256" not in blob["data_version"]:
        raise RuntimeError("checkpoint missing data_version contract")
    if blob["data_version"].get("manifest_sha256") == blob.get("code_sha256"):
        raise RuntimeError("data_version must not be the code digest")
    return blob


def assert_data_version(blob, manifest_sha: str, case_ids: List[str]):
    dv = blob["data_version"]
    expect = data_version(manifest_sha, case_ids, blob["cfg"]["query"])
    if dv["manifest_sha256"] != expect["manifest_sha256"]:
        raise RuntimeError("manifest SHA mismatch")
    if dv["case_ids_sha256"] != expect["case_ids_sha256"]:
        raise RuntimeError("case_ids SHA mismatch")
