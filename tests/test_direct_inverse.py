# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os

import numpy as np
import torch

from neural_operator.direct_inverse.config import DataConfig, LossConfig, ModelConfig, TargetConfig
from neural_operator.direct_inverse.data import (
    construct_event_target,
    construct_search_mask,
    event_time_s,
    minimum_spacing_m,
    spacing_band,
)
from neural_operator.direct_inverse.evaluate import detect_events, match_events
from neural_operator.direct_inverse.phasenet_1d import PhaseNet1D
from neural_operator.direct_inverse.pipeline import composite_loss, normalize_observation


def test_narrow_target_has_correct_center_and_range():
    data = DataConfig(seq_length=4096)
    target = TargetConfig()
    t = np.linspace(0.0, 30.0, data.seq_length, dtype=np.float32)
    depth = np.asarray([4100.0], dtype=np.float32)
    y = construct_event_target(t, depth, data, target)
    expected = float(event_time_s(depth, data)[0])
    assert 0.0 <= float(y.min()) <= float(y.max()) <= 1.0
    assert abs(float(t[int(np.argmax(y))]) - expected) <= 0.5 * float(t[1] - t[0])


def test_target_uses_max_not_sum():
    data = DataConfig(seq_length=4096)
    target = TargetConfig()
    t = np.linspace(0.0, 30.0, data.seq_length, dtype=np.float32)
    single = construct_event_target(t, np.asarray([4100.0]), data, target)
    duplicate = construct_event_target(t, np.asarray([4100.0, 4100.0]), data, target)
    assert np.array_equal(single, duplicate)
    assert float(duplicate.max()) <= 1.0


def test_search_mask_matches_fracture_zone():
    data = DataConfig(seq_length=4096)
    t = np.linspace(0.0, 30.0, data.seq_length, dtype=np.float32)
    mask = construct_search_mask(t, data)
    depths = (t[mask] - data.pump_shut_time_s) * data.wavespeed_m_s / 2.0
    assert depths.min() >= data.fracture_zone_m[0] - 6.0
    assert depths.max() <= data.fracture_zone_m[1] + 6.0


def test_spacing_helpers():
    assert np.isinf(minimum_spacing_m(np.asarray([4100.0])))
    assert minimum_spacing_m(np.asarray([4200.0, 4100.0])) == 100.0
    assert spacing_band(1, float("inf")) == "singleton"
    assert spacing_band(2, 60.0) == "50-75"


def test_model_shape_and_input_leakage_guard():
    model = PhaseNet1D(ModelConfig(channels=(8, 16, 32), groups=8))
    x = torch.randn(2, 1, 128)
    output = model(x)
    assert output.shape == x.shape
    try:
        model(torch.randn(2, 4, 128))
    except ValueError:
        pass
    else:
        raise AssertionError("four-channel surrogate input was accepted")


def test_loss_is_masked_and_finite():
    config = LossConfig()
    logits = torch.zeros(2, 1, 64, requires_grad=True)
    target = torch.zeros_like(logits)
    target[..., 31:34] = 1.0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[..., 16:48] = True
    loss, parts = composite_loss(logits, target, mask, config)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    changed = logits.detach().clone()
    changed[..., :16] = 100.0
    changed[..., 48:] = -100.0
    changed_loss, _ = composite_loss(changed, target, mask, config)
    assert torch.allclose(loss.detach(), changed_loss)


def test_normalization_one_channel():
    x = torch.randn(4, 1, 128) * 20 + 300
    normalized = normalize_observation(x)
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(4, 1), atol=1e-5)
    assert torch.allclose(normalized.std(dim=-1, correction=0), torch.ones(4, 1), atol=1e-5)


def test_detector_ignores_outside_mask_and_matching_is_one_to_one():
    data = DataConfig(seq_length=128)
    detector = __import__(
        "neural_operator.direct_inverse.config", fromlist=["DetectorConfig"]
    ).DetectorConfig(prominence=0.1)
    t = np.linspace(0.0, 30.0, 128)
    mask = construct_search_mask(t, data)
    probability = np.zeros(128)
    valid_indices = np.where(mask)[0]
    probability[valid_indices[len(valid_indices) // 2]] = 0.9
    probability[2] = 1.0
    events = detect_events(probability, mask, t, 0.5, data, detector)
    assert len(events) == 1
    metrics = match_events(events, np.asarray([events[0]["depth_m"], events[0]["depth_m"] + 5]), 38.36)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
