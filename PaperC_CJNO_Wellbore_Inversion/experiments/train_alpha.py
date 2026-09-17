#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 moc_v2_physical_steady_1k_newa 上训练 CJ-AlphaNet 与基线（不覆盖旧 Pilot 权重）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from PaperC_CJNO_Wellbore_Inversion.src.alpha_dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_NPZ_PATH,
    AlphaInversionDataset,
)
from PaperC_CJNO_Wellbore_Inversion.src.alpha_losses import AlphaCompositeLoss
from PaperC_CJNO_Wellbore_Inversion.src.alpha_metrics import compute_alpha_metrics, selection_score
from PaperC_CJNO_Wellbore_Inversion.src.models.alpha_baselines import (
    AlphaCJCepNet,
    AlphaFNO1D,
    AlphaResNet1D,
    TrainMeanBaseline,
    UniformFloorBaseline,
)
from PaperC_CJNO_Wellbore_Inversion.src.models.cj_alphanet import CJAlphaNet

DEFAULT_OUT = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "alpha_newa")

MODEL_BUILDERS = {
    "cj_alphanet": lambda: CJAlphaNet(use_cep2d=True),
    "cj_alphanet_no2d": lambda: CJAlphaNet(use_cep2d=False),
    "resnet": lambda: AlphaResNet1D(),
    "fno": lambda: AlphaFNO1D(),
    "cjcep": lambda: AlphaCJCepNet(),
}


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
    return out


def _cat_pred(store: Dict[str, List[torch.Tensor]], out: Dict[str, torch.Tensor]) -> None:
    keys = [
        "pred_alpha",
        "pred_active",
        "pred_active_logits",
        "pred_logY",
        "pred_m_alpha",
        "alpha_field",
    ]
    for k in keys:
        if k in out:
            store.setdefault(k, []).append(out[k].detach().cpu())


def _cat_tgt(store: Dict[str, List[torch.Tensor]], batch: Dict[str, torch.Tensor]) -> None:
    for k in [
        "mask_design",
        "alpha",
        "y_act",
        "logY",
        "m_alpha_grid",
        "positions",
        "Y0",
        "a_hat",
        "sample_id",
        "n_frac",
    ]:
        if k in batch:
            store.setdefault(k, []).append(batch[k].detach().cpu())


def _stack(store: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat(v, dim=0) for k, v in store.items()}


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device, max_batches: Optional[int] = None) -> Dict[str, Any]:
    model.eval()
    pred_store: Dict[str, List[torch.Tensor]] = {}
    tgt_store: Dict[str, List[torch.Tensor]] = {}
    losses = []
    criterion = AlphaCompositeLoss().to(device)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch_d = _to_device(batch, device)
        out = model(batch_d)
        ld = criterion(out, batch_d)
        losses.append(float(ld["loss"].item()))
        _cat_pred(pred_store, out)
        _cat_tgt(tgt_store, batch)
    pred = _stack(pred_store)
    tgt = _stack(tgt_store)
    metrics = compute_alpha_metrics(pred, tgt)
    metrics["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["select_score"] = selection_score(metrics)
    return {"metrics": metrics, "pred": pred, "target": tgt}


def train_one_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    output_dir: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_train_batches: Optional[int] = None,
    eval_every: int = 1,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    weight_path = os.path.join(output_dir, f"{model_name}_best.pt")
    last_path = os.path.join(output_dir, f"{model_name}_last.pt")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[*] 训练 {model_name}  params={n_params:,}  epochs={epochs}  device={device}")

    criterion = AlphaCompositeLoss().to(device)
    trainable = list(model.parameters())
    if any(p.requires_grad for p in trainable):
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
        do_opt = True
    else:
        optimizer = None
        scheduler = None
        do_opt = False

    history: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_metrics: Dict[str, Any] = {}
    t0 = time.time()

    if not do_opt:
        ev = evaluate_loader(model, val_loader, device)
        m = ev["metrics"]
        best_metrics = m
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "metrics": {k: v for k, v in m.items() if np.isscalar(v)},
            "model_name": model_name,
            "n_params": n_params,
        }
        torch.save(payload, weight_path)
        elapsed = time.time() - t0
        hist_obj = {
            "model_name": model_name,
            "n_params": n_params,
            "train_time_sec": elapsed,
            "best_metrics": {k: v for k, v in best_metrics.items() if np.isscalar(v)},
            "history": [{"epoch": 0, **{k: m[k] for k in ("val_loss", "m_alpha_w1_m", "active_f1") if k in m}}],
            "weight_path": weight_path,
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(hist_obj, f, indent=2, ensure_ascii=False)
        print(
            f"[OK] {model_name} 无参基线  W1={m['m_alpha_w1_m']:.1f}m  F1={m['active_f1']:.3f}  "
            f"{elapsed:.1f}s -> {weight_path}"
        )
        return hist_obj

    for epoch in range(1, epochs + 1):
        model.train(do_opt)
        train_losses = []
        for bi, batch in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            batch_d = _to_device(batch, device)
            out = model(batch_d)
            loss_d = criterion(out, batch_d)
            loss = loss_d["loss"]
            if do_opt:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            train_losses.append(float(loss.item()))
        if scheduler is not None:
            scheduler.step()

        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
        }
        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            ev = evaluate_loader(model, val_loader, device)
            m = ev["metrics"]
            record.update(
                {
                    "val_loss": m["val_loss"],
                    "m_alpha_w1_m": m["m_alpha_w1_m"],
                    "m_alpha_corr": m["m_alpha_corr"],
                    "active_f1": m["active_f1"],
                    "active_logY_mae": m["active_logY_mae"],
                    "active_alpha_mae": m["active_alpha_mae"],
                    "simplex_max_dev": m["simplex_max_dev"],
                    "select_score": m["select_score"],
                }
            )
            is_best = m["select_score"] < best_score
            if is_best:
                best_score = m["select_score"]
                best_metrics = m
                payload = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": {k: v for k, v in m.items() if np.isscalar(v)},
                    "model_name": model_name,
                    "n_params": n_params,
                }
                torch.save(payload, weight_path)
            tag = " [BEST]" if is_best else ""
            print(
                f"  epoch {epoch:03d}/{epochs:03d}  train={record['train_loss']:.4f}  "
                f"W1={m['m_alpha_w1_m']:.1f}m  corr={m['m_alpha_corr']:.3f}  "
                f"F1={m['active_f1']:.3f}  logY={m['active_logY_mae']:.3f}  "
                f"aMAE={m['active_alpha_mae']:.4f}{tag}"
            )
        history.append(record)

    torch.save({"model_state_dict": model.state_dict(), "epoch": epochs, "model_name": model_name}, last_path)
    elapsed = time.time() - t0
    hist_obj = {
        "model_name": model_name,
        "n_params": n_params,
        "train_time_sec": elapsed,
        "best_metrics": {k: v for k, v in best_metrics.items() if np.isscalar(v)},
        "history": history,
        "weight_path": weight_path,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(hist_obj, f, indent=2, ensure_ascii=False)
    print(f"[OK] {model_name}  {elapsed:.1f}s  best -> {weight_path}")
    return hist_obj


def build_model(name: str, train_ds: Optional[AlphaInversionDataset] = None) -> nn.Module:
    if name == "uniform":
        return UniformFloorBaseline()
    if name == "train_mean":
        if train_ds is None:
            raise ValueError("train_mean 需要训练集统计")
        return TrainMeanBaseline.from_dataset(train_ds)
    if name not in MODEL_BUILDERS:
        raise KeyError(f"未知模型 {name}")
    return MODEL_BUILDERS[name]()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CJ-AlphaNet / 基线训练")
    p.add_argument("--h5", type=str, default=DEFAULT_H5_PATH)
    p.add_argument("--npz", type=str, default=DEFAULT_NPZ_PATH)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--models", type=str, default="cj_alphanet", help="逗号分隔: cj_alphanet,cj_alphanet_no2d,resnet,fno,cjcep,uniform,train_mean")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--epochs-baseline", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batch-size-baseline", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--force-recompute-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"[*] device={device}  threads={torch.get_num_threads()}")

    print("[*] 加载 AlphaInversionDataset (800/100/100, seed=42)...")
    train_ds = AlphaInversionDataset(
        h5_path=args.h5,
        npz_path=args.npz,
        split="train",
        seed=args.seed,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute_cache,
    )
    val_ds = AlphaInversionDataset(
        h5_path=args.h5,
        npz_path=args.npz,
        split="val",
        seed=args.seed,
        cache_dir=args.cache_dir,
        force_recompute=False,
    )
    print(f"[*] train={len(train_ds)} val={len(val_ds)}  T={train_ds.n_time}  cache={train_ds.cache_path}")

    names = [s.strip() for s in args.models.split(",") if s.strip()]
    if args.smoke:
        args.epochs = 1
        args.epochs_baseline = 1
        if args.max_train_batches is None:
            args.max_train_batches = 2
        print("[*] SMOKE 模式: 1 epoch, 有限 batch")

    weights_dir = os.path.join(args.out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    summary = {}
    for name in names:
        model = build_model(name, train_ds=train_ds)
        is_nn = name not in ("uniform", "train_mean")
        is_alpha = name.startswith("cj_alphanet")
        bs = args.batch_size if is_alpha else args.batch_size_baseline
        drop_last = is_nn and bs > 1
        train_loader = train_ds.get_dataloader(batch_size=bs, shuffle=True, num_workers=args.num_workers, drop_last=drop_last)
        val_loader = val_ds.get_dataloader(batch_size=max(bs, 4), shuffle=False, num_workers=args.num_workers)
        ep = args.epochs if is_alpha else args.epochs_baseline
        if not is_nn:
            ep = 1
        res = train_one_model(
            model_name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=weights_dir,
            epochs=ep,
            lr=args.lr,
            weight_decay=1e-4,
            max_train_batches=args.max_train_batches,
            eval_every=1 if args.smoke or ep <= 15 else 2,
        )
        summary[name] = {
            "weight_path": res["weight_path"],
            "train_time_sec": res["train_time_sec"],
            "best_metrics": res["best_metrics"],
            "n_params": res["n_params"],
        }

    summary_path = os.path.join(args.out_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[+] 训练摘要 {summary_path}")


if __name__ == "__main__":
    main()
