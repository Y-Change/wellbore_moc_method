# -*- coding: utf-8 -*-
"""Pilot training: 1 → 8 → 256 overfit → optional full train. Val-only selection."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjno import PilotPrototype, count_params
from common import RunLogger, dump_json
from dataset import PilotDataset, collate_batch
from metrics_eval import case_metrics, summarize
from paths import CKPT_DIR, CONFIG_DIR, FIG_DIR, TABLE_DIR, ensure_dirs


TORCH24_HINT = r"D:\Anaconda\envs\torch24\python.exe"


def require_torch24(allow_other: bool) -> None:
    ver = torch.__version__
    exe = sys.executable
    print(f"python={exe}", flush=True)
    print(f"torch={ver} cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    ok = ver.startswith("2.4.")
    if not ok and not allow_other:
        raise SystemExit(
            f"Training must use the torch24 env (torch 2.4.x). Got {ver} from {exe}.\n"
            f"Rerun as: {TORCH24_HINT} train.py ..."
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass


def batch_to_torch(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v).to(device)
        else:
            out[k] = v
    return out


def compute_loss(model, batch, cfg):
    w = cfg["loss"]
    out = model(batch, apply_hard_times=int(cfg["train"]["apply_hard_times"]) if model.mode == "hard" else 0)
    mask = batch["wh_mask"]
    err = (out["p_pert_hat"] - batch["p_pert"]) * mask
    head = (err ** 2).sum() / mask.sum().clamp_min(1.0) / (batch["p_pert_scale"] ** 2 + 1e-12)
    loss = w["w_head"] * head
    extra = {"L_head": float(head.detach())}
    if out.get("node") is not None:
        nmask = batch["mask"][:, None, :] * batch["node_mask_t"][:, :, None]
        H_pert_t = (batch["node_H"] - batch["H0"][:, None, None]) * nmask
        H_pert_h = out["node"]["H_pert"] * nmask
        node = (H_pert_h - H_pert_t).pow(2).sum() / nmask.sum().clamp_min(1.0)
        node = node / (batch["H_pert_scale"] ** 2 + 1e-12)
        loss = loss + w["w_node_soft"] * node
        extra["L_node"] = float(node.detach())
    if out.get("hard_samples"):
        hs = 0.0
        rsum = 0.0
        for rec in out["hard_samples"]:
            hs = hs + (rec["H"] - rec["H_true"]).abs() / 4000.0
            rsum = rsum + rec["resid"]
        hs = hs / max(len(out["hard_samples"]), 1)
        loss = loss + w["w_hard"] * hs
        extra["L_hard"] = float(hs.detach())
        extra["R_jump"] = float(rsum.detach() / max(len(out["hard_samples"]), 1))
    extra["L"] = float(loss.detach())
    return loss, extra, out


@torch.no_grad()
def eval_loader(model, ds, norm, device, cfg, max_cases=64):
    model.eval()
    rows = []
    n = min(len(ds), max_cases)
    for i in range(n):
        s = ds[i]
        batch = batch_to_torch(collate_batch([s], norm), device)
        out = model(batch, apply_hard_times=0)
        p_hat = out["p_pert_hat"][0].cpu().numpy() + float(batch["p0"][0])
        p_true = batch["p_head"][0].cpu().numpy()
        met = case_metrics(p_hat, p_true, float(batch["p0"][0]), s.t,
                           float(s.period), float(s.dt),
                           mask=batch["wh_mask"][0].cpu().numpy())
        node = None
        if out.get("node") is not None:
            nmask = (batch["mask"][0].cpu().numpy()[None, :] *
                     batch["node_mask_t"][0].cpu().numpy()[:, None])
            Ht = (batch["node_H"][0].cpu().numpy() - float(batch["H0"][0])) * nmask
            Hh = out["node"]["H_pert"][0].cpu().numpy() * nmask
            from metrics_eval import _rel_l2
            node = {"H_pert_l2": _rel_l2(Hh, Ht, mask=nmask)}
        rows.append({"case_id": ds.rows[i]["case_id"], "N": int(ds[i].N),
                     "wellhead": met, "node": node})
    return rows, summarize(rows)


def save_ckpt(path, model, opt, epoch, seed, norm, digest, best):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "norm": norm,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
        "data_version": digest,
        "best": best,
        "mode": model.mode,
    }, path)


def run_stage(mode, n_train, seed, cfg, tag, smoke=False, eval_test=False):
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    data_cfg = cfg["data"]
    train_ds = PilotDataset("train", window_periods=data_cfg["window_periods"],
                            wellhead_decim=data_cfg["wellhead_decim"],
                            node_max_len=data_cfg["node_max_len"])
    val_ds = PilotDataset("val", window_periods=data_cfg["window_periods"],
                          wellhead_decim=data_cfg["wellhead_decim"],
                          node_max_len=data_cfg["node_max_len"])
    if n_train is not None:
        train_ds = train_ds.subset(n_train, seed=seed)
    norm = PilotDataset("train", window_periods=data_cfg["window_periods"],
                        wellhead_decim=data_cfg["wellhead_decim"],
                        node_max_len=data_cfg["node_max_len"]).compute_train_norm(max_cases=256)
    model = PilotPrototype(mode=mode, d=cfg["model"]["d"],
                           fno_d=cfg["model"]["fno_d"],
                           fno_modes=cfg["model"]["fno_modes"]).to(device)
    npar = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    max_ep = cfg["train"]["max_epochs_smoke"] if smoke else (
        cfg["train"]["max_epochs_overfit256"] if n_train and n_train <= 256
        else cfg["train"]["max_epochs_full_train"])
    tlim = 180 if smoke else (cfg["train"]["time_limit_s_overfit256"] if n_train and n_train <= 256
                              else cfg["train"]["time_limit_s_full"])
    logger = RunLogger(f"s2_{tag}", resolved_config={"mode": mode, "n_train": n_train, "seed": seed,
                                                     "cfg": cfg, "smoke": smoke, "device": str(device)},
                       extra={"n_params": npar})
    logger.log(f"mode={mode} n_train={len(train_ds)} n_val={len(val_ds)} params={npar} device={device}")
    t0 = time.time()
    best = {"val_pert_l2": 1e9, "epoch": -1}
    hist = []
    bs = 1 if len(train_ds) <= 2 else int(cfg["train"]["batch_size"])
    for ep in range(max_ep):
        model.train()
        order = np.arange(len(train_ds))
        np.random.shuffle(order)
        tr_loss = []
        for i0 in range(0, len(order), bs):
            sl = [train_ds[int(j)] for j in order[i0:i0 + bs]]
            batch = batch_to_torch(collate_batch(sl, norm), device)
            opt.zero_grad(set_to_none=True)
            loss, extra, _ = compute_loss(model, batch, cfg)
            if not torch.isfinite(loss):
                logger.log(f"nonfinite loss at epoch {ep}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            tr_loss.append(extra)
        _, val_sum = eval_loader(model, val_ds, norm, device, cfg, max_cases=min(16, len(val_ds)))
        vp = val_sum["pert_l2"]["mean"]
        rec = {"epoch": ep, "train_L": float(np.mean([x["L"] for x in tr_loss])) if tr_loss else float("nan"),
               "val_pert_l2": vp, "val_abs_l2": val_sum["abs_l2"]["mean"],
               "wall_s": time.time() - t0}
        hist.append(rec)
        logger.log(f"ep {ep:03d} trainL={rec['train_L']:.4e} val_pert={vp:.4f}")
        if np.isfinite(vp) and vp < best["val_pert_l2"]:
            best = {"val_pert_l2": vp, "epoch": ep, "val_sum": val_sum}
            save_ckpt(CKPT_DIR / tag / f"seed{seed}_best.pt", model, opt, ep, seed, norm,
                      logger.meta["train_code_digest"], best)
        if time.time() - t0 > tlim:
            logger.log("time limit")
            break
        if ep - best["epoch"] >= cfg["train"]["early_stopping_patience"] and ep > 5:
            logger.log("early stop")
            break
    # train-set overfit diagnostics (not test)
    tr_rows, tr_sum = eval_loader(model, train_ds, norm, device, cfg, max_cases=min(len(train_ds), 256))
    # batch=1 latency
    s0 = train_ds[0]
    b1 = batch_to_torch(collate_batch([s0], norm), device)
    t1 = time.time()
    with torch.no_grad():
        model.eval()
        model(b1, apply_hard_times=0)
    infer_s = time.time() - t1
    result = {
        "tag": tag, "mode": mode, "seed": seed, "n_train": len(train_ds),
        "n_params": npar, "device": str(device), "epochs_ran": len(hist),
        "best": best, "history": hist, "train_overfit": tr_sum,
        "infer_batch1_s": infer_s, "wall_clock_s": time.time() - t0,
        "test_sealed": not eval_test,
        "run_id": logger.run_id,
        "val_selection_max_cases": min(16, len(val_ds)),
        "val_selection_is_subset": len(val_ds) > 16,
    }
    if eval_test:
        te = PilotDataset("test", window_periods=data_cfg["window_periods"],
                          wellhead_decim=data_cfg["wellhead_decim"],
                          node_max_len=data_cfg["node_max_len"])
        te_rows, te_sum = eval_loader(model, te, norm, device, cfg, max_cases=len(te))
        result["test"] = te_sum
        result["test_is_not_formal_ood"] = True
        dump_json(te_rows, logger.dir / "test_rows.json")
    dump_json(result, logger.dir / "metrics.json")
    dump_json(result, TABLE_DIR / f"{tag}_seed{seed}.json")
    logger.write_metrics(result)
    logger.close()
    return result, logger.run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hard", "soft", "fno"], default="hard")
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--stage1", action="store_true", help="1-case smoke")
    ap.add_argument("--allow-other-torch", action="store_true")
    args = ap.parse_args()
    require_torch24(args.allow_other_torch)
    cfg = yaml.safe_load((CONFIG_DIR / "train_forward_pilot.yaml").read_text(encoding="utf-8"))
    n = 1 if args.stage1 or args.smoke else args.n_train
    if args.smoke and not args.stage1:
        n = min(8, args.n_train)
    tag = args.tag or f"{args.mode}_n{n}"
    res, rid = run_stage(args.mode, n, args.seed, cfg, tag, smoke=args.smoke, eval_test=args.eval_test)
    print(json.dumps({"run_id": rid, "best_val_pert": res["best"].get("val_pert_l2"),
                      "train_pert": res["train_overfit"]["pert_l2"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
