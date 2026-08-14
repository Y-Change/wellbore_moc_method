# -*- coding: utf-8 -*-
"""Train / evaluate fixed-kernel LISTA against the P0-A event protocol."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
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
from neural_operator.direct_inverse.config import (
    DataConfig,
    DetectorConfig,
    GateConfig,
    LossConfig,
    TargetConfig,
    TrainingConfig,
    close2000_gate_config,
    dataclass_from_dict,
    resampled_bin_depth_m,
)
from neural_operator.direct_inverse.data import DirectInverseDataset, load_manifest
from neural_operator.direct_inverse.evaluate import calibrate_threshold, evaluate_bundle
from neural_operator.direct_inverse.lista import (
    LISTA1D,
    ListaConfig,
    build_or_load_kernel,
    demean_observation,
    lista_parameter_count,
)
from neural_operator.direct_inverse.pipeline import composite_loss, manifest_digest
from neural_operator.direct_inverse.train import (
    create_scheduler,
    overfit_gate,
    resolve_device,
    stage_defaults,
)


class ListaEventModel(nn.Module):
    """LISTA reflectivity + 1x1 readout to event logits."""

    def __init__(self, kernel: torch.Tensor, lista_config: ListaConfig):
        super().__init__()
        self.lista = LISTA1D(kernel, lista_config)
        self.readout = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[1] != 1:
            observation = observation[:, :1]
        y = demean_observation(observation)
        reflectivity = self.lista(y)
        return self.readout(torch.abs(reflectivity))


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
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
            observation = batch["observation"].to(device)
            if observation.shape[1] != 1:
                observation = observation[:, :1]
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


def collect_lista_predictions(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    storage = {
        "case_id": [],
        "logits": [],
        "probability": [],
        "event_target": [],
        "valid_time_mask": [],
        "time_axis": [],
        "x_f": [],
        "Cf": [],
        "kleak": [],
        "n_frac": [],
        "min_spacing_m": [],
        "predicted_count": [],
        "count_probability": [],
    }
    with torch.no_grad():
        for batch in loader:
            observation = batch["observation"].to(device)
            if observation.shape[1] != 1:
                observation = observation[:, :1]
            logits = model(observation)
            batch_size = observation.shape[0]
            storage["case_id"].extend([str(value) for value in batch["case_id"]])
            storage["logits"].append(logits.cpu().numpy())
            storage["probability"].append(torch.sigmoid(logits).cpu().numpy())
            for key in ("event_target", "valid_time_mask", "time_axis", "x_f_m", "Cf", "kleak", "n_frac", "min_spacing_m"):
                output_key = "x_f" if key == "x_f_m" else key
                storage[output_key].append(batch[key].cpu().numpy())
            storage["predicted_count"].append(np.full(batch_size, -1, dtype=np.int64))
            storage["count_probability"].append(np.full((batch_size, 1), np.nan, dtype=np.float32))
    return {
        "case_id": np.asarray(storage["case_id"]),
        **{key: np.concatenate(value, axis=0) for key, value in storage.items() if key != "case_id"},
    }


def save_lista_checkpoint(path: str, model: ListaEventModel, optimizer, scheduler, **meta) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            **meta,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fixed-kernel LISTA event baseline")
    parser.add_argument("--stage", choices=("smoke", "overfit1", "overfit16", "subset512", "full"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="output/direct_inverse/runs_lista")
    parser.add_argument("--run-name", default="lista")
    parser.add_argument("--kernel-cache", default="output/direct_inverse/lista_kernel/brunone_n8192.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--friction", default="brunone")
    args = parser.parse_args()

    defaults = stage_defaults(args.stage)
    epochs = args.epochs or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch"]
    seed_everything(args.seed)
    device = resolve_device(args.device)

    manifest = load_manifest(args.manifest)
    data_config = dataclass_from_dict(DataConfig, manifest["data_config"])
    target_config = dataclass_from_dict(TargetConfig, manifest["target_config"])
    loss_config = LossConfig()
    detector_config = (
        dataclass_from_dict(DetectorConfig, manifest["detector_config"])
        if manifest.get("detector_config")
        else DetectorConfig()
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        maximum_epochs=epochs,
        num_workers=args.workers,
    )
    gate_config = (
        close2000_gate_config(
            resampled_bin_depth_m(
                data_config.seq_length,
                data_config.expected_tf_s,
                data_config.wavespeed_m_s,
            )
        )
        if data_config.spacing_regime == "close5_20"
        else GateConfig()
    )

    kernel_np, kernel_meta = build_or_load_kernel(
        args.kernel_cache,
        seq_length=data_config.seq_length,
        tf_s=data_config.expected_tf_s,
        wavespeed_m_s=data_config.wavespeed_m_s,
        pump_shut_time_s=data_config.pump_shut_time_s,
        friction=args.friction,
    )
    lista_config = ListaConfig(n_layers=args.n_layers)
    model = ListaEventModel(torch.from_numpy(kernel_np), lista_config).to(device)

    train_dataset = DirectInverseDataset(args.manifest, defaults["train"])
    validation_dataset = DirectInverseDataset(args.manifest, defaults["validation"])
    train_loader = make_loader(train_dataset, batch_size, True, args.seed, args.workers)
    validation_loader = make_loader(validation_dataset, batch_size, False, args.seed + 1, 0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    scheduler = create_scheduler(optimizer, epochs, len(train_loader), training_config)

    run_dir = create_run_directory(args.output_root, f"lista-{args.stage}", args.seed, args.run_name)
    run_config = {
        "stage": args.stage,
        "seed": args.seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "manifest": os.path.abspath(args.manifest),
        "manifest_digest": manifest_digest(args.manifest),
        "kernel_cache": os.path.abspath(args.kernel_cache),
        "kernel_meta": kernel_meta,
        "lista_config": asdict(lista_config),
        "model_parameters": lista_parameter_count(model.lista),
        "git_state": git_state(PROJECT_ROOT),
        "method": "lista_fixed_kernel",
    }
    write_json(
        os.path.join(run_dir, "manifest.json"),
        {
            "run_config": run_config,
            "data_config": asdict(data_config),
            "target_config": asdict(target_config),
            "loss_config": asdict(loss_config),
            "detector_config": asdict(detector_config),
            "training_config": asdict(training_config),
        },
    )

    initial_validation = evaluate_loss(model, validation_loader, device, loss_config)
    best_loss = float("inf")
    best_validation_metrics = None
    best_epoch = 0
    no_improvement = 0
    log_fields = [
        "epoch",
        "train_loss",
        "train_focal",
        "train_dice",
        "validation_loss",
        "validation_focal",
        "validation_dice",
        "learning_rate",
        "elapsed_s",
    ]
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=log_fields).writeheader()

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        sums = {"total": 0.0, "focal": 0.0, "dice": 0.0}
        seen = 0
        progress = tqdm(train_loader, desc=f"LISTA Epoch {epoch}/{epochs}")
        for batch in progress:
            observation = batch["observation"].to(device)
            if observation.shape[1] != 1:
                observation = observation[:, :1]
            target = batch["event_target"].to(device)
            mask = batch["valid_time_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(observation)
            loss, parts = composite_loss(logits, target, mask, loss_config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
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
        save_lista_checkpoint(
            os.path.join(run_dir, "checkpoints", "last.pt"),
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_validation_loss=min(best_loss, validation_metrics["total"]),
            lista_config=asdict(lista_config),
            kernel=kernel_np,
            run_config=run_config,
        )
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
            no_improvement = 0
            save_lista_checkpoint(
                os.path.join(run_dir, "checkpoints", "best.pt"),
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_validation_loss=best_loss,
                lista_config=asdict(lista_config),
                kernel=kernel_np,
                run_config=run_config,
            )
        else:
            no_improvement += 1
        print(f"Epoch {epoch}: train={train_metrics['total']:.6f} val={validation_metrics['total']:.6f}")
        if args.stage in ("subset512", "full") and no_improvement >= training_config.early_stopping_patience:
            break

    checkpoint = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    bundle = collect_lista_predictions(model, validation_loader, device)
    np.savez_compressed(os.path.join(run_dir, "evaluations", "validation_predictions.npz"), **bundle)

    if args.stage in ("overfit1", "overfit16", "smoke"):
        threshold = detector_config.overfit_threshold
        metrics = evaluate_bundle(bundle, threshold, data_config, detector_config)
    else:
        calibration = calibrate_threshold(bundle, data_config, detector_config)
        threshold = calibration["threshold"]
        metrics = calibration["validation_metrics"]
        write_json(
            os.path.join(run_dir, "threshold.json"),
            {
                "threshold": threshold,
                "validation_manifest_digest": manifest_digest(args.manifest),
                "calibration": calibration,
            },
        )
    write_json(
        os.path.join(run_dir, "evaluations", "metrics_summary.json"),
        {key: value for key, value in metrics.items() if key != "per_case"},
    )
    gate = overfit_gate(
        args.stage,
        metrics,
        initial_validation["total"],
        best_loss,
        gate_config,
        best_dice_loss=best_validation_metrics["dice"] if best_validation_metrics else None,
    )
    gate.update(
        {
            "stage": args.stage,
            "best_epoch": best_epoch,
            "initial_validation_loss": initial_validation["total"],
            "best_validation_loss": best_loss,
            "metrics": {key: value for key, value in metrics.items() if key != "per_case"},
            "method": "lista_fixed_kernel",
        }
    )
    write_json(os.path.join(run_dir, "gate_result.json"), gate)
    print(f"Run written to {run_dir}")
    print(f"Gate: {'PASS' if gate['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
