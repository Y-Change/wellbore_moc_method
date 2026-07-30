# -*- coding: utf-8 -*-
"""Stage-gated training CLI for the raw-waveform PhaseNet baseline."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError("Cannot find project root")
    _d = _parent
PROJECT_ROOT = _d

from neural_operator.dccdm_pipeline import create_run_directory, git_state, seed_everything, write_json
from .config import (
    DataConfig,
    DetectorConfig,
    GateConfig,
    LossConfig,
    ModelConfig,
    TargetConfig,
    TrainingConfig,
)
from .data import DirectInverseDataset, load_manifest
from .evaluate import calibrate_threshold, evaluate_bundle
from .phasenet_1d import PhaseNet1D, parameter_count
from .pipeline import (
    collect_predictions,
    composite_loss,
    manifest_digest,
    normalize_observation,
    save_checkpoint,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def stage_defaults(stage: str) -> Dict:
    values = {
        "smoke": {"train": "overfit1", "validation": "overfit1", "epochs": 1, "batch": 1},
        "overfit1": {"train": "overfit1", "validation": "overfit1", "epochs": 500, "batch": 1},
        "overfit16": {"train": "overfit16", "validation": "overfit16", "epochs": 500, "batch": 16},
        "subset512": {"train": "train512", "validation": "val128", "epochs": 150, "batch": 16},
        "full": {"train": "train", "validation": "validation", "epochs": 150, "batch": 16},
    }
    return values[stage]


def make_loader(dataset: DirectInverseDataset, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_loss(model, loader, device, loss_config) -> Dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "focal": 0.0, "dice": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            observation = normalize_observation(batch["observation"].to(device))
            logits = model(observation)
            _, parts = composite_loss(
                logits,
                batch["event_target"].to(device),
                batch["valid_time_mask"].to(device),
                loss_config,
            )
            batch_count = observation.shape[0]
            for key in totals:
                totals[key] += float(parts[key]) * batch_count
            count += batch_count
    return {key: value / max(count, 1) for key, value in totals.items()}


def create_scheduler(optimizer, epochs: int, steps_per_epoch: int, config: TrainingConfig):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(total_steps * config.warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        minimum_ratio = config.eta_min / config.learning_rate
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def overfit_gate(
    stage: str,
    metrics: Dict,
    initial_loss: float,
    best_loss: float,
    gate: GateConfig,
    best_dice_loss: float | None = None,
) -> Dict:
    reduction = (initial_loss - best_loss) / max(initial_loss, 1.0e-12)
    physical = metrics["physical"]
    soft_dice = 1.0 - best_dice_loss if best_dice_loss is not None else None
    if stage == "overfit1":
        criteria = {
            "precision_eq_1": physical["precision"] == 1.0,
            "recall_eq_1": physical["recall"] == 1.0,
            "f1_eq_1": physical["f1"] == gate.overfit1_f1,
            "exact_count": metrics["exact_count_cases"] == metrics["n_cases"],
            "median_depth_error_le_5_31m": metrics["median_depth_error_m"] is not None and metrics["median_depth_error_m"] <= 5.31,
        }
    elif stage == "overfit16":
        criteria = {
            "precision_ge_0_98": physical["precision"] >= gate.overfit16_precision,
            "recall_ge_0_98": physical["recall"] >= gate.overfit16_recall,
            "f1_ge_0_98": physical["f1"] >= gate.overfit16_f1,
            "exact_count_ge_15": metrics["exact_count_cases"] >= gate.overfit16_exact_count_cases,
            "median_depth_error_le_5_31m": metrics["median_depth_error_m"] is not None and metrics["median_depth_error_m"] <= gate.overfit16_median_depth_error_m,
            "p95_depth_error_le_10_62m": metrics["p95_depth_error_m"] is not None and metrics["p95_depth_error_m"] <= gate.overfit16_p95_depth_error_m,
        }
    else:
        criteria = {"completed": True}
    diagnostics = {
        "composite_loss_reduction": reduction,
        "soft_dice": soft_dice,
        "legacy_loss_reduction_ge_90pct": reduction >= gate.overfit16_loss_reduction,
        "legacy_soft_dice_ge_0_98": soft_dice is not None and soft_dice >= gate.overfit1_dice,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "diagnostics": diagnostics,
        "gate_schema": "direct-inverse-event-primary-v2",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the P0-A raw-waveform direct inverse baseline")
    parser.add_argument("--stage", choices=("smoke", "overfit1", "overfit16", "subset512", "full"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="output/direct_inverse/runs")
    parser.add_argument("--run-name", default="p0a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()

    defaults = stage_defaults(args.stage)
    epochs = args.epochs or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch"]
    seed_everything(args.seed)
    device = resolve_device(args.device)

    manifest = load_manifest(args.manifest)
    data_config = DataConfig(**manifest["data_config"])
    target_config = TargetConfig(**{
        key: value for key, value in manifest["target_config"].items()
        if key in TargetConfig.__dataclass_fields__
    })
    model_config = ModelConfig()
    loss_config = LossConfig()
    detector_config = DetectorConfig()
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        maximum_epochs=epochs,
        num_workers=args.workers,
    )
    gate_config = GateConfig()

    train_dataset = DirectInverseDataset(args.manifest, defaults["train"])
    validation_dataset = DirectInverseDataset(args.manifest, defaults["validation"])
    train_loader = make_loader(train_dataset, batch_size, True, args.seed, args.workers)
    validation_loader = make_loader(validation_dataset, batch_size, False, args.seed + 1, 0)

    model = PhaseNet1D(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    scheduler = create_scheduler(optimizer, epochs, len(train_loader), training_config)

    run_dir = create_run_directory(args.output_root, f"p0a-{args.stage}", args.seed, args.run_name)
    run_config = {
        "stage": args.stage,
        "seed": args.seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "manifest": os.path.abspath(args.manifest),
        "manifest_digest": manifest_digest(args.manifest),
        "model_parameters": parameter_count(model),
        "git_state": git_state(PROJECT_ROOT),
        "input_channels": 1,
    }
    write_json(os.path.join(run_dir, "manifest.json"), {
        "run_config": run_config,
        "model_config": asdict(model_config),
        "data_config": asdict(data_config),
        "target_config": asdict(target_config),
        "loss_config": asdict(loss_config),
        "detector_config": asdict(detector_config),
        "training_config": asdict(training_config),
    })

    initial_validation = evaluate_loss(model, validation_loader, device, loss_config)
    best_loss = float("inf")
    best_validation_metrics = None
    best_epoch = 0
    global_step = 0
    no_improvement = 0
    log_fields = ["epoch", "train_loss", "train_focal", "train_dice", "validation_loss", "validation_focal", "validation_dice", "learning_rate", "elapsed_s"]
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=log_fields).writeheader()

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        sums = {"total": 0.0, "focal": 0.0, "dice": 0.0}
        seen = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            observation = normalize_observation(batch["observation"].to(device))
            if observation.shape[1] != 1:
                raise RuntimeError("P0-A leakage guard failed")
            target = batch["event_target"].to(device)
            mask = batch["valid_time_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(observation)
            loss, parts = composite_loss(logits, target, mask, loss_config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            n = observation.shape[0]
            for key in sums:
                sums[key] += float(parts[key].detach()) * n
            seen += n
            progress.set_postfix(loss=f"{float(loss):.5f}")

        train_metrics = {key: value / max(seen, 1) for key, value in sums.items()}
        validation_metrics = evaluate_loss(model, validation_loader, device, loss_config)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["total"],
            "train_focal": train_metrics["focal"],
            "train_dice": train_metrics["dice"],
            "validation_loss": validation_metrics["total"],
            "validation_focal": validation_metrics["focal"],
            "validation_dice": validation_metrics["dice"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - started,
        }
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=log_fields).writerow(row)
        save_checkpoint(
            os.path.join(run_dir, "checkpoints", "last.pt"),
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            global_step=global_step, best_validation_loss=min(best_loss, validation_metrics["total"]),
            model_config=model_config, data_config=data_config, target_config=target_config,
            loss_config=loss_config, detector_config=detector_config,
            training_config=training_config, data_manifest_digest=manifest_digest(args.manifest),
            run_config=run_config,
        )
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
            no_improvement = 0
            save_checkpoint(
                os.path.join(run_dir, "checkpoints", "best.pt"),
                model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                global_step=global_step, best_validation_loss=best_loss,
                model_config=model_config, data_config=data_config, target_config=target_config,
                loss_config=loss_config, detector_config=detector_config,
                training_config=training_config, data_manifest_digest=manifest_digest(args.manifest),
                run_config=run_config,
            )
        else:
            no_improvement += 1
        print(f"Epoch {epoch}: train={train_metrics['total']:.6f} val={validation_metrics['total']:.6f}")
        if args.stage in ("subset512", "full") and no_improvement >= training_config.early_stopping_patience:
            break

    checkpoint, best_model = __import__(
        "neural_operator.direct_inverse.pipeline", fromlist=["load_checkpoint"]
    ).load_checkpoint(os.path.join(run_dir, "checkpoints", "best.pt"), device)
    bundle = collect_predictions(best_model, validation_loader, device)
    artifact_path = os.path.join(run_dir, "evaluations", "validation_predictions.npz")
    np.savez_compressed(artifact_path, **bundle)

    if args.stage in ("overfit1", "overfit16", "smoke"):
        threshold = detector_config.overfit_threshold
        metrics = evaluate_bundle(bundle, threshold, data_config, detector_config)
    else:
        calibration = calibrate_threshold(bundle, data_config, detector_config)
        threshold = calibration["threshold"]
        metrics = calibration["validation_metrics"]
        write_json(os.path.join(run_dir, "threshold.json"), {
            "threshold": threshold,
            "validation_manifest_digest": manifest_digest(args.manifest),
            "calibration": calibration,
        })
    write_json(os.path.join(run_dir, "evaluations", "metrics_summary.json"), {
        key: value for key, value in metrics.items() if key != "per_case"
    })
    gate = overfit_gate(
        args.stage,
        metrics,
        initial_validation["total"],
        best_loss,
        gate_config,
        best_dice_loss=best_validation_metrics["dice"] if best_validation_metrics else None,
    )
    gate.update({
        "stage": args.stage,
        "best_epoch": best_epoch,
        "initial_validation_loss": initial_validation["total"],
        "best_validation_loss": best_loss,
        "metrics": {key: value for key, value in metrics.items() if key != "per_case"},
    })
    write_json(os.path.join(run_dir, "gate_result.json"), gate)
    print(f"Run written to {run_dir}")
    print(f"Gate: {'PASS' if gate['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
