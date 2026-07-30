# -*- coding: utf-8 -*-
"""Shared preprocessing, reproducibility, and checkpoint helpers for DCCDM."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 2
PREPROCESSING_SCHEMA = "dccdm-context-v2"


@dataclass(frozen=True)
class DCCDMPreprocessingConfig:
    observation_eps: float = 1.0e-6
    cepstrum_eps: float = 1.0e-6
    target_scale: float = 25.0
    physical_threshold: float = 1.0
    keep_positive_quefrency_only: bool = True


@dataclass(frozen=True)
class DCCDMModelConfig:
    seq_length: int = 4096
    timesteps: int = 1000
    base_dim: int = 64
    context_dim: int = 2


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch without silently changing model inputs."""
    if deterministic:
        # Required by deterministic CuBLAS kernels on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    device = torch.device(device)
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def normalize_time_series(x: torch.Tensor, eps: float = 1.0e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample temporal z-score using population standard deviation."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, correction=0).clamp_min(eps)
    return (x - mean) / std, mean, std


def add_observation_noise(
    observation: torch.Tensor,
    fraction: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fraction < 0:
        raise ValueError("observation noise fraction must be non-negative")
    if fraction == 0:
        return observation, torch.zeros_like(observation)
    scale = observation.std(dim=-1, keepdim=True, correction=0).clamp_min(1.0e-6) * fraction
    noise = torch.randn(
        observation.shape,
        device=observation.device,
        dtype=observation.dtype,
        generator=generator,
    ) * scale
    return observation + noise, noise


def build_dccdm_context(
    observation: torch.Tensor,
    dedisp: torch.nn.Module,
    cepstrum: torch.nn.Module,
    config: DCCDMPreprocessingConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build normalized cepstrum plus a quefrency coordinate channel."""
    observation_norm, observation_mean, observation_std = normalize_time_series(
        observation, config.observation_eps
    )
    sharpened = dedisp(observation_norm)
    full_cepstrum = cepstrum(sharpened)

    if config.keep_positive_quefrency_only:
        n_time = full_cepstrum.shape[-1]
        cepstrum_feature = full_cepstrum[..., 1 : n_time // 2 + 1]
        coordinate = torch.arange(
            1,
            n_time // 2 + 1,
            device=full_cepstrum.device,
            dtype=full_cepstrum.dtype,
        ) / float(n_time // 2)
    else:
        cepstrum_feature = full_cepstrum[..., 1:]
        coordinate = torch.arange(
            1,
            full_cepstrum.shape[-1],
            device=full_cepstrum.device,
            dtype=full_cepstrum.dtype,
        ) / float(full_cepstrum.shape[-1] - 1)

    cepstrum_norm, cepstrum_mean, cepstrum_std = normalize_time_series(
        cepstrum_feature, config.cepstrum_eps
    )
    coordinate = coordinate.view(1, 1, -1).expand(observation.shape[0], 1, -1)
    context = torch.cat([cepstrum_norm, coordinate], dim=1)

    if not torch.isfinite(context).all():
        raise FloatingPointError("non-finite values in DCCDM context")

    diagnostics = {
        "observation_normalized": observation_norm,
        "observation_mean": observation_mean,
        "observation_std": observation_std,
        "sharpened": sharpened,
        "cepstrum_full": full_cepstrum,
        "cepstrum_normalized": cepstrum_norm,
        "cepstrum_mean": cepstrum_mean,
        "cepstrum_std": cepstrum_std,
        "quefrency_coordinate": coordinate,
    }
    return context, diagnostics


def encode_target(raw_target: torch.Tensor, target_scale: float = 25.0) -> torch.Tensor:
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")
    if not torch.isfinite(raw_target).all():
        raise ValueError("target contains NaN or Inf")
    target_min = float(raw_target.detach().min())
    target_max = float(raw_target.detach().max())
    tolerance = max(1.0e-6, target_scale * 1.0e-6)
    if target_min < -tolerance or target_max > target_scale + tolerance:
        raise ValueError(
            f"target range [{target_min:.6g}, {target_max:.6g}] is outside [0, {target_scale}]"
        )
    return raw_target / target_scale * 2.0 - 1.0


def decode_target(normalized_target: torch.Tensor, target_scale: float = 25.0) -> torch.Tensor:
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")
    return (normalized_target + 1.0) * 0.5 * target_scale


def postprocess_prediction(
    raw_physical_prediction: torch.Tensor,
    target_scale: float = 25.0,
    threshold: float = 1.0,
) -> torch.Tensor:
    prediction = raw_physical_prediction.clamp(0.0, target_scale)
    if threshold > 0:
        prediction = torch.where(prediction >= threshold, prediction, torch.zeros_like(prediction))
    return prediction


def create_run_directory(output_root: str, mode: str, seed: int, run_name: Optional[str] = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    suffix = run_name or "run"
    run_id = f"{stamp}-{mode}-seed{seed}-{suffix}"
    run_dir = os.path.abspath(os.path.join(output_root, run_id))
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=False)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=False)
    os.makedirs(os.path.join(run_dir, "evaluations"), exist_ok=False)
    return run_dir


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().numpy())
    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def file_manifest_digest(paths: list[str]) -> str:
    normalized = "\n".join(os.path.abspath(path) for path in paths)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def git_state(project_root: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
        result["dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def save_checkpoint(
    path: str,
    *,
    dedisp: torch.nn.Module,
    unet: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    global_step: int,
    best_metric: float,
    model_config: DCCDMModelConfig,
    preprocessing_config: DCCDMPreprocessingConfig,
    training_config: Dict[str, Any],
    data_manifest: Dict[str, Any],
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "preprocessing_schema": PREPROCESSING_SCHEMA,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "model_config": asdict(model_config),
        "preprocessing_config": asdict(preprocessing_config),
        "training_config": training_config,
        "data_manifest": data_manifest,
        "dedisp_state_dict": dedisp.state_dict(),
        "unet_state_dict": unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device | str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"{path} is not a DCCDM schema-{CHECKPOINT_SCHEMA_VERSION} checkpoint; "
            "evaluate legacy weights with the legacy workflow"
        )
    return checkpoint
