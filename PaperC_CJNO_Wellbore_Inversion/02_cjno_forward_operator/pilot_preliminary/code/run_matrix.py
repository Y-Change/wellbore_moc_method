# -*- coding: utf-8 -*-
"""Controlled A/B (and optional FNO) matrix. Does not claim official HPO."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from paths import CONFIG_DIR, TABLE_DIR, ensure_dirs
from train import require_torch24, run_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with-fno", action="store_true")
    ap.add_argument("--open-test", action="store_true")
    ap.add_argument("--allow-other-torch", action="store_true")
    args = ap.parse_args()
    require_torch24(args.allow_other_torch)
    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "train_forward_pilot.yaml").read_text(encoding="utf-8"))
    jobs = []
    if args.smoke:
        jobs = [("hard", 1, 42, "smoke_hard_n1"), ("soft", 1, 42, "smoke_soft_n1"),
                ("hard", 8, 42, "smoke_hard_n8")]
    else:
        jobs = [
            ("hard", 1, 42, "hard_n1"),
            ("soft", 1, 42, "soft_n1"),
            ("hard", 8, 42, "hard_n8"),
            ("soft", 8, 42, "soft_n8"),
            ("hard", 64, 42, "hard_n64"),
            ("soft", 64, 42, "soft_n64"),
            ("hard", 256, 42, "hard_n256"),
            ("soft", 256, 42, "soft_n256"),
        ]
        if args.with_fno:
            jobs.append(("fno", 256, 42, "fno_n256"))
    results = []
    incomplete = []
    for mode, n, seed, tag in jobs:
        try:
            res, rid = run_stage(mode, n, seed, cfg, tag, smoke=args.smoke, eval_test=False)
            results.append({"tag": tag, "run_id": rid, "mode": mode, "n": n, "seed": seed,
                            "val_pert": res["best"].get("val_pert_l2"),
                            "train_pert": res["train_overfit"]["pert_l2"]["mean"],
                            "n_params": res["n_params"], "wall_s": res["wall_clock_s"]})
        except Exception as exc:
            incomplete.append({"tag": tag, "error": repr(exc)})
            print("FAILED", tag, exc)
    # extra seeds only if 256 hard/soft succeeded and not smoke
    if not args.smoke and all(r["tag"] in ("hard_n256", "soft_n256") or True for r in results):
        done256 = {r["tag"] for r in results}
        if "hard_n256" in done256 and "soft_n256" in done256:
            for seed in (43, 44):
                for mode in ("hard", "soft"):
                    tag = f"{mode}_n256_s{seed}"
                    try:
                        res, rid = run_stage(mode, 256, seed, cfg, tag, smoke=False, eval_test=False)
                        results.append({"tag": tag, "run_id": rid, "mode": mode, "n": 256, "seed": seed,
                                        "val_pert": res["best"].get("val_pert_l2"),
                                        "train_pert": res["train_overfit"]["pert_l2"]["mean"],
                                        "n_params": res["n_params"], "wall_s": res["wall_clock_s"]})
                    except Exception as exc:
                        incomplete.append({"tag": tag, "error": repr(exc)})
    if args.open_test:
        for mode in ("hard", "soft"):
            tag = f"{mode}_n256_test"
            try:
                res, rid = run_stage(mode, 256, 42, cfg, tag, smoke=False, eval_test=True)
                results.append({"tag": tag, "run_id": rid, "test": res.get("test")})
            except Exception as exc:
                incomplete.append({"tag": tag, "error": repr(exc)})
    summary = {"results": results, "incomplete": incomplete,
               "not_official_5seed_hpo": True}
    dump_json(summary, TABLE_DIR / ("matrix_smoke.json" if args.smoke else "matrix_pilot.json"))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
