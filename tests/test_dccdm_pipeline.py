# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import torch

from neural_operator.cepstrum import DifferentiableCepstrum
from neural_operator.dccdm_pipeline import (
    DCCDMPreprocessingConfig,
    build_dccdm_context,
    decode_target,
    encode_target,
    make_generator,
    normalize_time_series,
    postprocess_prediction,
)
from neural_operator.diffusion_1d import ConditionalUNet1D, GaussianDiffusion1D
from neural_operator.evaluate_diffusion import detect_positive_peaks, match_peaks_one_to_one


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def test_target_roundtrip():
    values = torch.tensor([0.0, 1.0, 12.5, 25.0]).view(1, 1, -1)
    encoded = encode_target(values, 25.0)
    decoded = decode_target(encoded, 25.0)
    assert torch.allclose(values, decoded)


def test_target_rejects_invalid_values():
    for values in (torch.tensor([[[26.0]]]), torch.tensor([[[float("nan")]]])):
        try:
            encode_target(values, 25.0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid target was accepted")


def test_normalization_is_finite_for_constant_input():
    normalized, mean, std = normalize_time_series(torch.ones(2, 1, 32))
    assert torch.isfinite(normalized).all()
    assert torch.isfinite(std).all()
    assert torch.allclose(normalized, torch.zeros_like(normalized))


def test_context_removes_q0_and_adds_coordinate():
    observation = torch.randn(2, 1, 64)
    context, diagnostics = build_dccdm_context(
        observation,
        Identity(),
        DifferentiableCepstrum(),
        DCCDMPreprocessingConfig(),
    )
    assert context.shape == (2, 2, 32)
    assert torch.isfinite(context).all()
    coordinate = diagnostics["quefrency_coordinate"]
    assert torch.all(coordinate[..., 1:] > coordinate[..., :-1])
    assert float(coordinate.min()) > 0.0
    assert float(coordinate.max()) == 1.0


def test_timestep_changes_unet_output():
    torch.manual_seed(2)
    model = ConditionalUNet1D(context_dim=2, base_dim=8)
    x = torch.randn(1, 1, 32)
    context = torch.randn(1, 2, 16)
    output_1 = model(x, torch.tensor([1]), context)
    output_2 = model(x, torch.tensor([9]), context)
    assert output_1.shape == x.shape
    assert not torch.allclose(output_1, output_2)


def test_sampling_is_deterministic_for_same_seed():
    torch.manual_seed(3)
    model = ConditionalUNet1D(context_dim=2, base_dim=8)
    diffusion = GaussianDiffusion1D(model, seq_length=32, timesteps=4)
    context = torch.randn(1, 2, 16)
    first = diffusion.sample(context, generator=make_generator("cpu", 77))
    second = diffusion.sample(context, generator=make_generator("cpu", 77))
    third = diffusion.sample(context, generator=make_generator("cpu", 78))
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_peak_detector_ignores_negative_peaks():
    response = np.array([0.0, -8.0, 0.0, 2.0, 0.0])
    time = np.arange(len(response), dtype=float)
    peaks = detect_positive_peaks(response, time, height=1.0, prominence=0.5)
    assert len(peaks) == 1
    assert peaks[0]["index"] == 3


def test_matching_does_not_reuse_prediction():
    detected = [{"time_s": 1.1, "amplitude": 2.0}]
    metrics = match_peaks_one_to_one(
        detected,
        true_times=np.array([1.0, 1.2]),
        true_depths=np.array([0.0, 145.0]),
        true_amplitudes=np.array([2.0, 2.0]),
        tolerance_s=0.2,
        wavespeed=1450.0,
    )
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1


def test_postprocessing_preserves_raw_tensor():
    raw = torch.tensor([[[-2.0, 0.5, 2.0, 30.0]]])
    before = raw.clone()
    processed = postprocess_prediction(raw, 25.0, 1.0)
    assert torch.equal(raw, before)
    assert torch.equal(processed, torch.tensor([[[0.0, 0.0, 2.0, 25.0]]]))
