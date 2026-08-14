# -*- coding: utf-8 -*-
"""Versioned configuration contracts for deterministic direct inversion."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Tuple


DATA_SCHEMA = "direct-inverse-data-v1"
DATA_SCHEMA_CLOSE2000 = "direct-inverse-data-v2-close2000"
DATA_SCHEMA_CLOSE2000_N8192 = "direct-inverse-data-v2-close2000-n8192"
DATA_SCHEMA_CLOSE2000_N16384 = "direct-inverse-data-v2-close2000-n16384"
DATA_SCHEMA_CLOSE2000_N16384_WIDEFWHM = "direct-inverse-data-v2-close2000-n16384-widefwhm"
DATA_SCHEMA_CLOSE2000_N8192_PHYS3 = "direct-inverse-data-v2-close2000-n8192-phys3"
CHECKPOINT_SCHEMA = "direct-inverse-checkpoint-v1"
TARGET_SCHEMA = "fracture-event-heatmap-v1"
FEATURE_SCHEMA_RAW = "direct-inverse-feature-raw-v1"
FEATURE_SCHEMA_PHYS3 = "direct-inverse-feature-phys3-v1"
MODEL_ID = "p0a_raw_unet"
MODEL_ID_PHYS3 = "p1_phys3_unet"

# Resampled grid depth for tf=50 s, N=4096, c=1450 m/s:
# dt = 50/4095, bin_depth = c*dt/2 ≈ 8.8525 m.
CLOSE2000_BIN_DEPTH_M = 8.85


def resampled_bin_depth_m(
    seq_length: int,
    tf_s: float = 50.0,
    wavespeed_m_s: float = 1450.0,
) -> float:
    """Depth per resampled time sample under two-way travel."""
    if seq_length < 2:
        raise ValueError("seq_length must be >= 2")
    dt = tf_s / float(seq_length - 1)
    return wavespeed_m_s * dt / 2.0


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
    expected_tf_s: float = 30.0
    spacing_regime: str = "nominal50"
    schema: str = DATA_SCHEMA


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
class FeatureConfig:
    """Observation-channel recipe for the inverse dataset."""

    schema: str = FEATURE_SCHEMA_RAW
    n_channels: int = 1
    include_envelope: bool = False
    include_cepstrum: bool = False
    cepstrum_as_response: bool = True  # use -C so fracture peaks tend positive


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = MODEL_ID
    in_channels: int = 1
    out_channels: int = 1
    channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 256)
    kernel_size: int = 7
    bottleneck_dropout: float = 0.1
    groups: int = 8
    enable_count_head: bool = False
    count_classes: int = 6  # n_frac in {1..6} -> class index n_frac-1


@dataclass(frozen=True)
class LossConfig:
    focal_alpha_positive: float = 0.75
    focal_gamma: float = 2.0
    dice_weight: float = 0.5
    epsilon: float = 1.0e-6
    count_weight: float = 0.25


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
    # Event-primary hard gates for held-out pilots / full baseline.
    # Dense-map Dice and composite-loss reduction remain diagnostics only.
    subset512_precision: float = 0.90
    subset512_f1: float = 0.85
    subset512_exact_count_accuracy: float = 0.65
    subset512_count_mae: float = 0.55
    subset512_median_depth_error_m: float = 5.31
    subset512_p95_depth_error_m: float = 10.62
    full_precision: float = 0.90
    full_f1: float = 0.88
    full_exact_count_accuracy: float = 0.70
    full_count_mae: float = 0.50
    full_median_depth_error_m: float = 5.31
    full_p95_depth_error_m: float = 10.62


def close2000_data_config(split_seed: int = 42, seq_length: int = 4096, schema: str = DATA_SCHEMA_CLOSE2000) -> DataConfig:
    """Study pool = all 2000 cases with cluster spacing in [5, 20] m."""
    return DataConfig(
        schema=schema,
        seq_length=seq_length,
        split_seed=split_seed,
        nominal_min_spacing_m=5.0,
        train_count=1400,
        validation_count=300,
        test_count=300,
        expected_total_count=2000,
        expected_nominal_count=2000,
        expected_challenge_count=0,
        expected_tf_s=50.0,
        spacing_regime="close5_20",
    )


def close2000_target_config(bin_depth_m: float | None = None) -> TargetConfig:
    return TargetConfig(fwhm_depth_m=CLOSE2000_BIN_DEPTH_M if bin_depth_m is None else float(bin_depth_m))


def close2000_detector_config(bin_depth_m: float | None = None) -> DetectorConfig:
    bin_m = CLOSE2000_BIN_DEPTH_M if bin_depth_m is None else float(bin_depth_m)
    return DetectorConfig(
        minimum_separation_m=bin_m,
        physical_tolerance_m=bin_m,
        prominence=0.01,
    )


def close2000_gate_config(bin_depth_m: float | None = None) -> GateConfig:
    bin_m = CLOSE2000_BIN_DEPTH_M if bin_depth_m is None else float(bin_depth_m)
    return GateConfig(
        overfit16_median_depth_error_m=bin_m,
        overfit16_p95_depth_error_m=2.0 * bin_m,
        # Close-spacing held-out gates are intentionally softer than the >=50 m regime.
        subset512_precision=0.80,
        subset512_f1=0.70,
        subset512_exact_count_accuracy=0.45,
        subset512_count_mae=1.00,
        subset512_median_depth_error_m=bin_m,
        subset512_p95_depth_error_m=2.0 * bin_m,
        full_precision=0.80,
        full_f1=0.75,
        full_exact_count_accuracy=0.50,
        full_count_mae=0.85,
        full_median_depth_error_m=bin_m,
        full_p95_depth_error_m=2.0 * bin_m,
    )


def phys3_feature_config() -> FeatureConfig:
    return FeatureConfig(
        schema=FEATURE_SCHEMA_PHYS3,
        n_channels=3,
        include_envelope=True,
        include_cepstrum=True,
        cepstrum_as_response=True,
    )


def profile_configs(profile: str, split_seed: int = 42) -> Dict[str, object]:
    if profile == "nominal50":
        return {
            "data": DataConfig(split_seed=split_seed),
            "target": TargetConfig(),
            "detector": DetectorConfig(),
            "gate": GateConfig(),
            "feature": FeatureConfig(),
        }
    if profile == "close2000":
        data = close2000_data_config(split_seed)
        bin_m = resampled_bin_depth_m(data.seq_length, data.expected_tf_s, data.wavespeed_m_s)
        return {
            "data": data,
            "target": close2000_target_config(bin_m),
            "detector": close2000_detector_config(bin_m),
            "gate": close2000_gate_config(bin_m),
            "feature": FeatureConfig(),
        }
    if profile == "close2000_n8192":
        data = close2000_data_config(
            split_seed,
            seq_length=8192,
            schema=DATA_SCHEMA_CLOSE2000_N8192,
        )
        bin_m = resampled_bin_depth_m(data.seq_length, data.expected_tf_s, data.wavespeed_m_s)
        return {
            "data": data,
            "target": close2000_target_config(bin_m),
            "detector": close2000_detector_config(bin_m),
            "gate": close2000_gate_config(bin_m),
            "feature": FeatureConfig(),
        }
    if profile == "close2000_n8192_phys3":
        data = close2000_data_config(
            split_seed,
            seq_length=8192,
            schema=DATA_SCHEMA_CLOSE2000_N8192_PHYS3,
        )
        bin_m = resampled_bin_depth_m(data.seq_length, data.expected_tf_s, data.wavespeed_m_s)
        return {
            "data": data,
            "target": close2000_target_config(bin_m),
            "detector": close2000_detector_config(bin_m),
            "gate": close2000_gate_config(bin_m),
            "feature": phys3_feature_config(),
        }
    if profile == "close2000_n16384":
        data = close2000_data_config(
            split_seed,
            seq_length=16384,
            schema=DATA_SCHEMA_CLOSE2000_N16384,
        )
        bin_m = resampled_bin_depth_m(data.seq_length, data.expected_tf_s, data.wavespeed_m_s)
        return {
            "data": data,
            "target": close2000_target_config(bin_m),
            "detector": close2000_detector_config(bin_m),
            "gate": close2000_gate_config(bin_m),
            "feature": FeatureConfig(),
        }
    if profile == "close2000_n16384_widefwhm":
        # Keep N=16384 grid (bin≈2.21 m) for peak separability, but use the
        # N=8192 soft-label width (FWHM≈4.43 m) so training targets are less sharp.
        data = close2000_data_config(
            split_seed,
            seq_length=16384,
            schema=DATA_SCHEMA_CLOSE2000_N16384_WIDEFWHM,
        )
        bin_m = resampled_bin_depth_m(data.seq_length, data.expected_tf_s, data.wavespeed_m_s)
        fwhm_m = resampled_bin_depth_m(8192, data.expected_tf_s, data.wavespeed_m_s)
        return {
            "data": data,
            "target": close2000_target_config(fwhm_m),
            "detector": close2000_detector_config(bin_m),
            "gate": close2000_gate_config(bin_m),
            "feature": FeatureConfig(),
        }
    raise ValueError(f"unknown profile: {profile}")


def dataclass_from_dict(cls, payload: Dict):
    fields = set(cls.__dataclass_fields__)
    return cls(**{key: value for key, value in payload.items() if key in fields})
