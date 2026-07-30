# -*- coding: utf-8 -*-
"""Shared training, loss, checkpoint, and prediction helpers for P0-A."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from neural_operator.dccdm_pipeline import normalize_time_series
from .config import (
    CHECKPOINT_SCHEMA,
    DataConfig,
    DetectorConfig,
    LossConfig,
    ModelConfig,
    TargetConfig,
    TrainingConfig,
)
from .phasenet_1d import PhaseNet1D


def masked_focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
) -> torch.Tensor:
    target = target.to(logits.dtype)
    mask = mask.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha = config.focal_alpha_positive * target + (1.0 - config.focal_alpha_positive) * (1.0 - target)
    focal = alpha * (1.0 - p_t).pow(config.focal_gamma) * bce
    return (focal * mask).sum() / mask.sum().clamp_min(1.0)


def masked_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits) * mask
    target = target * mask
    numerator = 2.0 * (probability * target).sum(dim=(-1, -2)) + epsilon
    denominator = probability.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) + epsilon
    return (1.0 - numerator / denominator).mean()


def composite_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    focal = masked_focal_bce(logits, target, mask, config)
    dice = masked_soft_dice_loss(logits, target, mask, config.epsilon)
    total = focal + config.dice_weight * dice
    return total, {"focal": focal, "dice": dice, "total": total}


def normalize_observation(observation: torch.Tensor) -> torch.Tensor:
    normalized, _, _ = normalize_time_series(observation, 1.0e-6)
    return normalized


def save_checkpoint(
    path: str,
    *,
    model: PhaseNet1D,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    model_config: ModelConfig,
    data_config: DataConfig,
    target_config: TargetConfig,
    loss_config: LossConfig,
    detector_config: DetectorConfig,
    training_config: TrainingConfig,
    data_manifest_digest: str,
    run_config: Dict,
) -> None:
    payload = {
        "schema_name": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "model_id": model_config.model_id,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "model_config": asdict(model_config),
        "data_config": asdict(data_config),
        "target_config": asdict(target_config),
        "loss_config": asdict(loss_config),
        "detector_config": asdict(detector_config),
        "training_config": asdict(training_config),
        "run_config": run_config,
        "data_manifest_digest": data_manifest_digest,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device | str) -> Tuple[Dict, PhaseNet1D]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_name") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint: {path}")
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PhaseNet1D(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint, model


def manifest_digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def collect_predictions(
    model: PhaseNet1D,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    storage: Dict[str, list] = {
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
    }
    with torch.no_grad():
        for batch in loader:
            observation = normalize_observation(batch["observation"].to(device))
            if observation.shape[1] != 1:
                raise RuntimeError("P0-A leakage guard: model input must have exactly one channel")
            logits = model(observation)
            storage["case_id"].extend([str(value) for value in batch["case_id"]])
            storage["logits"].append(logits.cpu().numpy())
            storage["probability"].append(torch.sigmoid(logits).cpu().numpy())
            for key in ("event_target", "valid_time_mask", "time_axis", "x_f_m", "Cf", "kleak", "n_frac", "min_spacing_m"):
                output_key = "x_f" if key == "x_f_m" else key
                storage[output_key].append(batch[key].cpu().numpy())
    return {
        "case_id": np.asarray(storage["case_id"]),
        **{
            key: np.concatenate(value, axis=0)
            for key, value in storage.items()
            if key != "case_id"
        },
    }
