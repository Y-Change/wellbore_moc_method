# -*- coding: utf-8 -*-
"""C1 overfit: train-selected checkpoint, no val early-stop. Predict never sees targets."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import assert_data_version, load_checkpoint, save_checkpoint
from common import RunLogger, code_digest, dump_json, sha256_file
from dataset import (
    N_MAX, Round2Dataset, collate_predict, collate_targets, compute_norm,
    load_round1_manifest, nested_ids, nested_order,
)
from metrics import first_arrival_error, rel_l2, summarize_first_arrival
from models import QueryFourierMLP, loss_head
from paths import CKPT_DIR, CODE_DIR, CONFIG_DIR, ROUND1_MANIFEST, TABLE_DIR, ensure_dirs

TORCH24_HINT = r"D:\Anaconda\envs\torch24\python.exe"


def require_torch24(allow_other: bool) -> None:
    ver = torch.__version__
    print(f"python={sys.executable}", flush=True)
    print(f"torch={ver} cuda={torch.cuda.is_available()}", flush=True)
    if not ver.startswith("2.4.") and not allow_other:
        raise SystemExit(f"Need torch 2.4.x. Rerun: {TORCH24_HINT} train.py ...")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass


def evaluate_split(model, ds, norm, device, p_scale):
    model.eval()
    rows = []
    phases = []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            pred_np = collate_predict([b], norm)
            feat = torch.from_numpy(pred_np["feat"]).to(device)
            tau = torch.from_numpy(pred_np["tau"]).to(device)
            mask = pred_np["mask"][0]
            out = model.predict(feat, tau, p_scale)
            y = out["p_pert"][0].detach().cpu().numpy()
            tgt = b.targets.p_pert
            T0 = tgt.size
            m = rel_l2(y[:T0], tgt, mask[:T0])
            ph = first_arrival_error(
                b.query.t, y[:T0], tgt, b.query.t_valve_end, b.physical.period,
            )
            m["phase"] = ph
            m["case_id"] = b.physical.case_id
            m["silent_p"] = b.targets.silent_p
            rows.append(m)
            phases.append(ph)
    rels = np.array([r["rel"] for r in rows if not r["silent"]], dtype=float)
    silent_n = int(sum(1 for r in rows if r["silent"]))
    summ = {
        "n": len(rows),
        "n_silent": silent_n,
        "pert_l2_mean": float(np.mean(rels)) if rels.size else float("nan"),
        "pert_l2_p50": float(np.median(rels)) if rels.size else float("nan"),
        "pert_l2_p90": float(np.quantile(rels, 0.9)) if rels.size else float("nan"),
        "pert_l2_max": float(np.max(rels)) if rels.size else float("nan"),
        "first_arrival": summarize_first_arrival(phases),
        "n_hard_calls": 0.0,
        "rows": rows,
    }
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", required=True, help="nested prefix length or all")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--allow-other-torch", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    require_torch24(args.allow_other_torch)
    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "train_round2.yaml").read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else cfg["model_seed_primary"])
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    n_arg = args.n
    ids = nested_ids(order, n_arg if n_arg != "all" else "all")
    n_tag = n_arg
    ds = Round2Dataset("train", cfg, man, case_ids=ids)
    norm = compute_norm(ds, max_cases=len(ds))
    norm["norm_scope"] = f"overfit_subset_n={n_tag}"
    d_feat = 8 + 3 * N_MAX
    model = QueryFourierMLP(
        d_feat, d=int(cfg["model"]["d"]),
        n_harmonics=int(cfg["model"]["n_harmonics"]),
        n_layers=int(cfg["model"]["n_layers"]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    p_scale = float(norm["p_pert_scale"])
    man_sha = sha256_file(ROUND1_MANIFEST)
    code_sha = code_digest(CODE_DIR)
    n_key = int(n_tag) if str(n_tag).isdigit() else 64
    max_ep = int(cfg["overfit"]["max_epochs"].get(n_key, cfg["overfit"]["max_epochs"].get(64, 80)))
    tlim = float(cfg["overfit"]["time_limit_s"].get(n_key, 600))
    target = float(cfg["overfit"]["target_mean"].get(n_key, 0.02))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep, eta_min=1e-5)
    bs = 1 if len(ds) == 1 else (4 if len(ds) <= 8 else 8)
    logger = RunLogger(f"r2_overfit_n{n_tag}", cfg, extra={
        "n": n_tag, "case_ids": ids, "subset_seed": cfg["subset_seed"],
        "model_seed": seed, "device": str(device),
        "n_active_params": model.n_active_params(),
        "n_registered_params": model.n_active_params(),
        "manifest_sha256": man_sha,
    })
    logger.log(f"n={n_tag} cases={len(ds)} params={model.n_active_params()} device={device}")
    best = {"pert": float("inf"), "epoch": -1}
    history = []
    t0 = time.time()
    step = 0
    last_path = CKPT_DIR / f"{logger.run_id}_last.pt"
    best_path = CKPT_DIR / f"{logger.run_id}_best.pt"
    for ep in range(1, max_ep + 1):
        model.train()
        order_i = np.random.permutation(len(ds))
        ep_loss = []
        gnorms = []
        for s in range(0, len(ds), bs):
            sl = order_i[s:s + bs]
            bundles = [ds[int(i)] for i in sl]
            pb = collate_predict(bundles, norm)
            tb = collate_targets(bundles)
            feat = torch.from_numpy(pb["feat"]).to(device)
            tau = torch.from_numpy(pb["tau"]).to(device)
            mask = torch.from_numpy(pb["mask"]).to(device)
            tgt = torch.from_numpy(tb["p_pert"]).to(device=device, dtype=torch.float32)
            pred = model.predict(feat, tau, p_scale)
            if not torch.isfinite(pred["p_pert"]).all():
                raise RuntimeError("non-finite prediction; step blocked")
            loss = loss_head(pred, tgt, mask, p_scale)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss; step blocked")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            gnorms.append(gn)
            opt.step()
            ep_loss.append(float(loss.detach()))
            step += 1
        sched.step()
        train_m = evaluate_split(model, ds, norm, device, p_scale)
        rec = {
            "epoch": ep, "train_loss": float(np.mean(ep_loss)),
            "train_pert_mean": train_m["pert_l2_mean"],
            "train_pert_p90": train_m["pert_l2_p90"],
            "train_pert_max": train_m["pert_l2_max"],
            "grad_norm_mean": float(np.mean(gnorms)) if gnorms else 0.0,
            "lr": float(opt.param_groups[0]["lr"]),
            "wall_s": time.time() - t0,
        }
        history.append(rec)
        logger.log(
            f"ep={ep} loss={rec['train_loss']:.4e} pert={rec['train_pert_mean']:.4f} "
            f"p90={rec['train_pert_p90']:.4f} max={rec['train_pert_max']:.4f} gn={rec['grad_norm_mean']:.3f}"
        )
        extra = {"selection": "train_pert_l2", "metrics_at_save": rec, "kind": "last"}
        save_checkpoint(last_path, model, opt, cfg, norm, ids, man_sha, code_sha, ep, step, extra)
        if train_m["pert_l2_mean"] < best["pert"]:
            best = {"pert": train_m["pert_l2_mean"], "epoch": ep, "metrics": train_m}
            extra_b = {"selection": "train_pert_l2", "metrics_at_save": rec, "kind": "best"}
            save_checkpoint(best_path, model, opt, cfg, norm, ids, man_sha, code_sha, ep, step, extra_b)
        if time.time() - t0 > tlim:
            logger.log(f"time limit {tlim}s reached")
            break
        if train_m["pert_l2_mean"] <= target and train_m["pert_l2_max"] <= max(4 * target, 0.08):
            logger.log("registered overfit target reached")
            break
    # reload best and last; report both from their own weights
    blob_best = load_checkpoint(best_path, map_location=device)
    assert_data_version(blob_best, man_sha, ids)
    model.load_state_dict(blob_best["model"])
    best_eval = evaluate_split(model, ds, norm, device, p_scale)
    blob_last = load_checkpoint(last_path, map_location=device)
    model.load_state_dict(blob_last["model"])
    last_eval = evaluate_split(model, ds, norm, device, p_scale)
    # val is reported from BEST only if n>=64 diagnostic; never used for selection
    val_eval = None
    if len(ds) >= 8:
        ds_val = Round2Dataset("val", cfg, man)
        # same checkpoint (best), do not mix
        model.load_state_dict(blob_best["model"])
        # evaluate at most 16 val cases for cost; tagged as not selection
        # keep a shallow view without opening test
        class _Head:
            def __init__(self, base, k):
                self.base = base
                self.k = min(k, len(base))
            def __len__(self):
                return self.k
            def __getitem__(self, i):
                return self.base[i]
        val_eval = evaluate_split(model, _Head(ds_val, 16), norm, device, p_scale)
        val_eval["note"] = "same best ckpt; not used for selection; val16 only"
    metrics = {
        "status": "PASS" if best_eval["pert_l2_mean"] <= target else "FAIL",
        "target_mean": target,
        "n": n_tag,
        "model_seed": seed,
        "subset_seed": cfg["subset_seed"],
        "case_ids": ids,
        "n_active_params": model.n_active_params(),
        "best": best_eval,
        "last": last_eval,
        "best_epoch": blob_best["epoch"],
        "last_epoch": blob_last["epoch"],
        "val16_from_best": val_eval,
        "history": history,
        "ckpt_best": str(best_path),
        "ckpt_last": str(last_path),
        "data_version": blob_best["data_version"],
        "code_sha256": code_sha,
        "selection": "train_pert_l2",
        "node_loss_used": False,
        "note": "node-vs-wellhead gradient angle not applicable (w_node=0)",
    }
    logger.write_metrics(metrics)
    dump_json(metrics, TABLE_DIR / f"overfit_n{n_tag}_{logger.run_id}.json")
    logger.close()
    print(json_brief(metrics), flush=True)
    return 0 if metrics["status"] == "PASS" else 2


def json_brief(m):
    return (
        f"run={m.get('run_id', '')} n={m['n']} status={m['status']} "
        f"best_mean={m['best']['pert_l2_mean']:.4f} last_mean={m['last']['pert_l2_mean']:.4f} "
        f"best_ep={m['best_epoch']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
