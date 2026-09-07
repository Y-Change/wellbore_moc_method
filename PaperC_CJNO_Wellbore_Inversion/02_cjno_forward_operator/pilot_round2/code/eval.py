# -*- coding: utf-8 -*-
"""Evaluate one checkpoint on train and/or val. Never opens the viewed test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import assert_data_version, load_checkpoint
from common import dump_json, sha256_file
from dataset import N_MAX, Round2Dataset, load_round1_manifest
from models import QueryFourierMLP
from paths import CONFIG_DIR, ROUND1_MANIFEST, RUNS_DIR
from train import evaluate_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if "test" in args.splits.split(","):
        raise SystemExit("old test is a viewed holdout; blocked this round")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blob = load_checkpoint(Path(args.ckpt), map_location=device)
    cfg = blob["cfg"]
    norm = blob["norm"]
    ids = blob["case_ids"]
    man = load_round1_manifest()
    man_sha = sha256_file(ROUND1_MANIFEST)
    assert_data_version(blob, man_sha, ids)
    model = QueryFourierMLP(
        8 + 3 * N_MAX, d=int(cfg["model"]["d"]),
        n_harmonics=int(cfg["model"]["n_harmonics"]),
        n_layers=int(cfg["model"]["n_layers"]),
    ).to(device)
    model.load_state_dict(blob["model"])
    p_scale = float(norm["p_pert_scale"])
    out = {
        "ckpt": args.ckpt,
        "epoch": blob["epoch"],
        "global_step": blob["global_step"],
        "kind": blob.get("kind"),
        "data_version": blob["data_version"],
    }
    for sp in args.splits.split(","):
        sp = sp.strip()
        if sp == "train":
            ds = Round2Dataset("train", cfg, man, case_ids=ids)
        elif sp == "val":
            ds = Round2Dataset("val", cfg, man)
        else:
            raise SystemExit(f"blocked split {sp}")
        out[sp] = evaluate_split(model, ds, norm, device, p_scale)
    dest = RUNS_DIR / "eval_from_ckpt"
    dest.mkdir(parents=True, exist_ok=True)
    dump_json(out, dest / (Path(args.ckpt).stem + "_eval.json"))
    print({k: (v.get("pert_l2_mean") if isinstance(v, dict) and "pert_l2_mean" in v else v)
           for k, v in out.items() if k in ("train", "val", "epoch", "kind")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
