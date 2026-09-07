# -*- coding: utf-8 -*-
"""B-arm: same 8 cases, same net, budget 5000 steps or 120 s. B1 vs B2."""
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
from checkpoint import load_checkpoint, save_checkpoint
from common import RunLogger, code_digest, dump_json, sha256_file
from dataset import (
    D_FEAT, PRESENT_NOT_CONSUMED_BY_MLP, CONSUMED_BY_MLP, Round3Dataset,
    collate_predict, collate_targets, compute_norm, load_round1_manifest,
    nested_ids, nested_order,
)
from metrics import first_arrival_pair, rel_l2, summarize_phase_errors
from models import QueryFourierMLP, loss_b1_shared_mse, loss_b2_energy_equal
from paths import CKPT_DIR, CODE_DIR, CONFIG_DIR, ROUND1_MANIFEST, TABLE_DIR, ensure_dirs

TORCH24 = r"D:\Anaconda\envs\torch24\python.exe"


def require_torch24(allow_other: bool) -> None:
    print(f"python={sys.executable}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if not torch.__version__.startswith("2.4.") and not allow_other:
        raise SystemExit(f"Need torch 2.4.x. Rerun: {TORCH24} train_b.py ...")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batch_plan(n_cases: int, batch_size: int, max_steps: int, seed: int):
    rng = np.random.RandomState(int(seed))
    plan = []
    while len(plan) < max_steps:
        order = rng.permutation(n_cases)
        for s in range(0, n_cases, batch_size):
            sl = order[s:s + batch_size]
            if sl.size:
                plan.append(sl.tolist())
            if len(plan) >= max_steps:
                break
    return plan[:max_steps]


def evaluate(model, ds, norm, device, p_scale):
    model.eval()
    rows, phases = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            pb = collate_predict([b], norm)
            feat = torch.from_numpy(pb["feat"]).to(device)
            tau = torch.from_numpy(pb["tau"]).to(device)
            y = model.predict(feat, tau, p_scale)["p_pert"][0].detach().cpu().numpy()
            tgt = b.targets.p_pert
            T0 = tgt.size
            r = rel_l2(y[:T0], tgt, silent=b.targets.silent_p)
            ph = first_arrival_pair(b.query.t, y[:T0], tgt, b.query.t_valve_end, b.physical.period)
            rows.append({
                "case_id": b.physical.case_id,
                "rel_l2": r,
                "silent_p": bool(b.targets.silent_p),
                "target_l2": float(np.linalg.norm(tgt)),
                "pred_l2": float(np.linalg.norm(y[:T0])),
                "phase": ph,
            })
            phases.append(ph)
    active = np.array([r["rel_l2"] for r in rows if not r["silent_p"]], dtype=float)
    return {
        "n": len(rows),
        "n_silent": int(sum(1 for r in rows if r["silent_p"])),
        "pert_l2_mean": float(np.mean(active)) if active.size else float("nan"),
        "pert_l2_p90": float(np.quantile(active, 0.9)) if active.size else float("nan"),
        "pert_l2_max": float(np.max(active)) if active.size else float("nan"),
        "first_arrival": summarize_phase_errors(phases),
        "rows": rows,
    }


def resolved(cfg, arm, ids, seed, extra=None):
    opt = cfg["optim"]
    rec = {
        "arm": arm,
        "case_ids": list(ids),
        "subset_seed": int(cfg["subset_seed"]),
        "model_seed": int(seed),
        "batch_plan_seed": int(seed),
        "query": dict(cfg["query"]),
        "model": dict(cfg["model"]),
        "optim": dict(opt),
        "norm_protocol": cfg["norm"]["protocol"],
        "loss": arm,
        "thresholds": dict(cfg["thresholds"]),
        "hardcoded_now_in_config": [
            "lr", "eta_min", "clip", "batch_size", "max_steps", "time_limit_s",
            "scheduler", "d", "n_harmonics", "n_layers", "dt_s", "n_samples",
        ],
        "do_not_reuse_round2_ckpt": True,
        "val_early_stop": False,
        "select": "train_pert_l2",
    }
    if extra:
        rec.update(extra)
    return rec


def train_arm(arm: str, cfg: dict, ids, seed: int, device, allow_other: bool,
              prefix: str, n_tag: str, target: float):
    set_seed(seed)
    man = load_round1_manifest()
    ds = Round3Dataset("train", cfg, man, case_ids=ids)
    norm = compute_norm(ds, max_cases=len(ds))
    norm["norm_scope"] = f"round3_{arm}_n={n_tag}"
    model = QueryFourierMLP(
        D_FEAT, d=int(cfg["model"]["d"]),
        n_harmonics=int(cfg["model"]["n_harmonics"]),
        n_layers=int(cfg["model"]["n_layers"]),
    ).to(device)
    opt_cfg = cfg["optim"]
    opt = torch.optim.Adam(model.parameters(), lr=float(opt_cfg["lr"]))
    max_steps = int(opt_cfg["max_steps"])
    tlim = float(opt_cfg["time_limit_s"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_steps, eta_min=float(opt_cfg["eta_min"])
    )
    plan = batch_plan(len(ds), int(opt_cfg["batch_size"]), max_steps, seed)
    p_scale = float(norm["p_pert_scale"])
    man_sha = sha256_file(ROUND1_MANIFEST)
    code_sha = code_digest(CODE_DIR)
    rc = resolved(cfg, arm, ids, seed, extra={
        "n_active_params": model.n_active_params(),
        "n_registered_params": model.n_active_params(),
        "batch_plan_head": plan[:8],
        "consumed_fields": list(CONSUMED_BY_MLP),
        "present_not_consumed": list(PRESENT_NOT_CONSUMED_BY_MLP),
        "data_digest": {
            "manifest_sha256": man_sha,
            "case_ids": ids,
            "source_sha256": [ds[i].source_sha256 for i in range(len(ds))],
        },
    })
    logger = RunLogger(prefix, rc, extra={"n": n_tag, "arm": arm, "case_ids": ids})
    logger.log(f"{arm} n={n_tag} cases={ids} params={model.n_active_params()} device={device}")
    best = {"pert": float("inf"), "step": -1}
    history = []
    t0 = time.time()
    stop_reason = "not_started"
    last_path = CKPT_DIR / f"{logger.run_id}_last.pt"
    best_path = CKPT_DIR / f"{logger.run_id}_best.pt"
    clip = float(opt_cfg["clip"])
    eval_every = 50
    clip_hits = 0
    gnorms = []
    step = 0
    prev_best = None
    for step_i, sl in enumerate(plan, start=1):
        model.train()
        bundles = [ds[int(i)] for i in sl]
        pb = collate_predict(bundles, norm)
        tb = collate_targets(bundles)
        feat = torch.from_numpy(pb["feat"]).to(device)
        tau = torch.from_numpy(pb["tau"]).to(device)
        mask = torch.from_numpy(pb["mask"]).to(device)
        tgt = torch.from_numpy(tb["p_pert"]).to(device=device, dtype=torch.float32)
        silent = torch.from_numpy(tb["silent_p"]).to(device)
        t_l2 = torch.from_numpy(tb["target_l2"]).to(device=device, dtype=torch.float32)
        pred = model.predict(feat, tau, p_scale)
        if not torch.isfinite(pred["p_pert"]).all():
            raise RuntimeError("non-finite prediction; step blocked")
        if arm == "B1":
            loss = loss_b1_shared_mse(pred, tgt, mask, p_scale)
            silent_abs = torch.zeros((), device=device)
        elif arm == "B2":
            loss, silent_abs, _ = loss_b2_energy_equal(pred, tgt, mask, silent, t_l2)
        else:
            raise ValueError(arm)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss; step blocked")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        gnorms.append(gn)
        if gn >= clip:
            clip_hits += 1
        opt.step()
        sched.step()
        step = step_i
        do_eval = (step % eval_every == 0) or step == 1 or step == max_steps
        if do_eval:
            ev = evaluate(model, ds, norm, device, p_scale)
            rec = {
                "step": step,
                "loss": float(loss.detach()),
                "silent_abs": float(silent_abs.detach()),
                "train_pert_mean": ev["pert_l2_mean"],
                "train_pert_p90": ev["pert_l2_p90"],
                "train_pert_max": ev["pert_l2_max"],
                "grad_norm": gn,
                "grad_norm_mean": float(np.mean(gnorms[-eval_every:])),
                "clip_frac": clip_hits / step,
                "lr": float(opt.param_groups[0]["lr"]),
                "wall_s": time.time() - t0,
                "improve_rate": None if prev_best is None else float(prev_best - ev["pert_l2_mean"]) / max(eval_every, 1),
            }
            history.append(rec)
            logger.log(
                f"{arm} step={step} loss={rec['loss']:.3e} pert={rec['train_pert_mean']:.4f} "
                f"p90={rec['train_pert_p90']:.4f} max={rec['train_pert_max']:.4f} "
                f"gn={gn:.3f} lr={rec['lr']:.2e} t={rec['wall_s']:.1f}s"
            )
            extra = {"kind": "last", "metrics_at_save": rec, "arm": arm}
            save_checkpoint(last_path, model, opt, cfg, norm, ids, man_sha, code_sha,
                            step, step, scheduler=sched, extra=extra)
            if ev["pert_l2_mean"] < best["pert"]:
                prev_best = best["pert"] if best["pert"] < 1e20 else ev["pert_l2_mean"]
                best = {"pert": ev["pert_l2_mean"], "step": step, "metrics": ev}
                save_checkpoint(best_path, model, opt, cfg, norm, ids, man_sha, code_sha,
                                step, step, scheduler=sched,
                                extra={"kind": "best", "metrics_at_save": rec, "arm": arm})
            else:
                prev_best = best["pert"]
        if time.time() - t0 >= tlim:
            stop_reason = "time_limit"
            logger.log(f"time limit {tlim}s at step={step}")
            break
        if step >= max_steps:
            stop_reason = "step_budget"
            break
    else:
        stop_reason = "step_budget"
    if not last_path.exists():
        save_checkpoint(last_path, model, opt, cfg, norm, ids, man_sha, code_sha,
                        step, step, scheduler=sched, extra={"kind": "last", "arm": arm})
        ev = evaluate(model, ds, norm, device, p_scale)
        best = {"pert": ev["pert_l2_mean"], "step": step, "metrics": ev}
        save_checkpoint(best_path, model, opt, cfg, norm, ids, man_sha, code_sha,
                        step, step, scheduler=sched, extra={"kind": "best", "arm": arm})
    blob_best = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(blob_best["model"])
    best_eval = evaluate(model, ds, norm, device, p_scale)
    blob_last = load_checkpoint(last_path, map_location=device)
    model.load_state_dict(blob_last["model"])
    last_eval = evaluate(model, ds, norm, device, p_scale)
    status = "PASS" if best_eval["pert_l2_mean"] <= target else "FAIL_WITHIN_BUDGET"
    metrics = {
        "status": status,
        "stop_reason": stop_reason,
        "arm": arm,
        "target_mean": target,
        "n": n_tag,
        "model_seed": seed,
        "case_ids": ids,
        "n_active_params": model.n_active_params(),
        "optimizer_steps_completed": step,
        "wall_clock_s": time.time() - t0,
        "clip_frac": clip_hits / max(step, 1),
        "best": best_eval,
        "last": last_eval,
        "best_step": blob_best["global_step"],
        "last_step": blob_last["global_step"],
        "history": history,
        "ckpt_best": str(best_path),
        "ckpt_last": str(last_path),
        "selection": "train_pert_l2",
        "note": "budget exhaustion is not convergence",
        "run_id": logger.run_id,
    }
    logger.write_metrics(metrics)
    dump_json(metrics, TABLE_DIR / f"{logger.run_id}.json")
    logger.close()
    print(f"run={logger.run_id} {arm} {status} best={best_eval['pert_l2_mean']:.4f} "
          f"steps={step} stop={stop_reason}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["B1", "B2", "single"])
    ap.add_argument("--n", default="8")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--case-index", type=int, default=None, help="for single-case arm")
    ap.add_argument("--allow-other-torch", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    require_torch24(args.allow_other_torch)
    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "train_round3.yaml").read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else cfg["model_seed_primary"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ids = nested_ids(order, int(args.n) if args.n != "all" else "all")
    if args.arm == "single":
        if args.case_index is None:
            raise SystemExit("--case-index required for single")
        ids = [ids[int(args.case_index)]]
        target = float(cfg["thresholds"]["n1_train_mean"])
        prefix = f"r3_single_{ids[0]}"
        n_tag = "1"
        arm = "B1"
    else:
        target = float(cfg["thresholds"]["n8_train_mean"] if len(ids) > 1 else cfg["thresholds"]["n1_train_mean"])
        prefix = f"r3_{args.arm}_n{len(ids)}"
        n_tag = str(len(ids))
        arm = args.arm
    train_arm(arm, cfg, ids, seed, device, args.allow_other_torch, prefix, n_tag, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
