# -*- coding: utf-8 -*-
"""Versioned configuration contracts for deterministic direct inversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


DATA_SCHEMA = "direct-inverse-data-v1"
CHECKPOINT_SCHEMA = "direct-inverse-checkpoint-v1"
TARGET_SCHEMA = "fracture-event-heatmap-v1"
MODEL_ID = "p0a_raw_unet"


@dataclass(frozen=True)
class DataConfig:
    seq_length: int = 4096
    wavespeed_m_s: float = 1450.0
    pump_shut_time_s: float = 1.0
    fracture_zone_m: Tuple[float, float] = (3500.0, 4800.0)
    nominal_min_spacing_m: float = 50.0
    split_seed: int = 42
    train_count: int = 2896
    validation_count: int = 620
    test_count: int = 620
    expected_total_count: int = 6000
    expected_nominal_count: int = 4136
    expected_challenge_count: int = 1864
    max_fractures: int = 6


@dataclass(frozen=True)
class TargetConfig:
    schema: str = TARGET_SCHEMA
    fwhm_depth_m: float = 38.36
    composition: str = "max"
    amplitude_weighted: bool = False

    @property
    def fwhm_time_s(self) -> float:
        return 2.0 * self.fwhm_depth_m / 1450.0

    @property
    def sigma_time_s(self) -> float:
        return self.fwhm_time_s / 2.3548200450309493


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = MODEL_ID
    in_channels: int = 1
    out_channels: int = 1
    channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 256)
    kernel_size: int = 7
    bottleneck_dropout: float = 0.1
    groups: int = 8


@dataclass(frozen=True)
class LossConfig:
    focal_alpha_positive: float = 0.75
    focal_gamma: float = 2.0
    dice_weight: float = 0.5
    epsilon: float = 1.0e-6


@dataclass(frozen=True)
class DetectorConfig:
    overfit_threshold: float = 0.5
    prominence: float = 0.10
    minimum_separation_m: float = 38.36
    physical_tolerance_m: float = 38.36
    grid_tolerance_bins: int = 1
    threshold_grid: Tuple[float, ...] = tuple(i / 100.0 for i in range(10, 91, 5))


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    warmup_fraction: float = 0.05
    eta_min: float = 1.0e-6
    effective_batch_size: int = 32
    maximum_epochs: int = 150
    early_stopping_patience: int = 30
    num_workers: int = 0


@dataclass(frozen=True)
class GateConfig:
    overfit1_f1: float = 1.0
    overfit1_dice: float = 0.98
    overfit16_precision: float = 0.98
    overfit16_recall: float = 0.98
    overfit16_f1: float = 0.98
    overfit16_exact_count_cases: int = 15
    overfit16_median_depth_error_m: float = 5.31
    overfit16_p95_depth_error_m: float = 10.62
    overfit16_loss_reduction: float = 0.90
