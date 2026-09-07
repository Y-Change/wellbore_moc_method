# -*- coding: utf-8 -*-
"""Evaluate a frozen checkpoint on val or test. Test only after tuning is sealed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjno import PilotPrototype
from common import dump_json
from dataset import PilotDataset, collate_batch
from metrics_eval import case_metrics, summarize
from paths import CONFIG_DIR, TABLE_DIR
from train import batch_to_torch, require_torch24


@torch.no_grad()
def eval_split(ckpt_path: Path, split: str, max_cases: int = 0):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load((CONFIG_DIR / "train_forward_pilot.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PilotPrototype(mode=blob.get("mode", "hard"), d=cfg["model"]["d"],
                           fno_d=cfg["model"]["fno_d"], fno_modes=cfg["model"]["fno_modes"])
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    ds = PilotDataset(split, window_periods=cfg["data"]["window_periods"],
                      wellhead_decim=cfg["data"]["wellhead_decim"],
                      node_max_len=cfg["data"]["node_max_len"])
    n = len(ds) if max_cases <= 0 else min(len(ds), max_cases)
    rows = []
    by_n = {}
    for i in range(n):
        s = ds[i]
        batch = batch_to_torch(collate_batch([s], blob["norm"]), device)
        out = model(batch, apply_hard_times=0)
        p_hat = out["p_pert_hat"][0].cpu().numpy() + float(s.p0)
        met = case_metrics(p_hat, batch["p_head"][0].cpu().numpy(), s.p0,
                           s.t, s.period, s.dt, mask=batch["wh_mask"][0].cpu().numpy())
        rec = {"case_id": s.case_id, "N": s.N, "wellhead": met,
               "nyquist_wh_train_grid_hz": s.nyquist_wh,
               "nyquist_node_hz": s.nyquist_node}
        if out.get("node") is not None:
            nmask = (batch["mask"][0].cpu().numpy()[None, :] *
                     batch["node_mask_t"][0].cpu().numpy()[:, None])
            Ht = (batch["node_H"][0].cpu().numpy() - float(s.H0)) * nmask
            Hh = out["node"]["H_pert"][0].cpu().numpy() * nmask
            from metrics_eval import _rel_l2
            rec["node"] = {"H_pert_l2": _rel_l2(Hh, Ht, mask=nmask)}
        rows.append(rec)
        by_n.setdefault(s.N, []).append(rec)
    out = {
        "ckpt": str(ckpt_path.as_posix()),
        "split": split,
        "n": n,
        "summary": summarize(rows),
        "by_N": {str(k): summarize(v) for k, v in sorted(by_n.items())},
        "worst": sorted(rows, key=lambda r: r["wellhead"]["pert_l2"]["rel"], reverse=True)[:10],
        "test_is_not_formal_ood": split == "test",
        "cannot_certify": cfg["eval"]["cannot_certify"],
        "seed": blob.get("seed"),
        "mode": blob.get("mode"),
    }
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--allow-other-torch", action="store_true")
    args = ap.parse_args()
    require_torch24(args.allow_other_torch)
    if args.split == "test":
        print("WARNING: test is a held-out ID set on the truncated pilot prior, not formal OOD.", flush=True)
    rec, rows = eval_split(Path(args.ckpt), args.split, args.max_cases)
    dest = TABLE_DIR / f"eval_{args.split}_{Path(args.ckpt).parent.name}.json"
    dump_json(rec, dest)
    dump_json(rows, dest.with_name(dest.stem + "_rows.json"))
    print(json.dumps({"split": args.split, "summary": rec["summary"], "out": str(dest)}, indent=2))


if __name__ == "__main__":
    main()
