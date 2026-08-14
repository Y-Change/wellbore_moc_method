# -*- coding: utf-8 -*-
"""Fixed-kernel LISTA unrolling for sparse reflectivity recovery."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.sparse_deconv.reference_signal import extract_wavelet, generate_reference_signal


@dataclass(frozen=True)
class ListaConfig:
    n_layers: int = 12
    init_lambda: float = 0.05
    learnable: bool = True
    model_id: str = "lista_fixed_kernel_v1"


def soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Elementwise soft-threshold; threshold broadcasts as scalar or [B,1,1]."""
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)


class LISTA1D(nn.Module):
    """ISTA unrolling with fixed FFT convolution kernel and per-layer η, θ."""

    def __init__(self, kernel: torch.Tensor, config: ListaConfig = ListaConfig()):
        super().__init__()
        if kernel.ndim != 1:
            raise ValueError("kernel must be 1-D")
        self.config = config
        # Register fixed physical wavelet (circular convolution via FFT).
        self.register_buffer("kernel", kernel.detach().float().clone())
        lipschitz = float(torch.max(torch.abs(torch.fft.fft(self.kernel))) ** 2)
        lipschitz = max(lipschitz, 1.0e-8)
        init_step = 1.0 / lipschitz
        init_thresh = float(config.init_lambda) * init_step
        if config.learnable:
            self.steps = nn.Parameter(torch.full((config.n_layers,), init_step))
            self.thresholds = nn.Parameter(torch.full((config.n_layers,), init_thresh))
        else:
            self.register_buffer("steps", torch.full((config.n_layers,), init_step))
            self.register_buffer("thresholds", torch.full((config.n_layers,), init_thresh))

    def _conv(self, signal: torch.Tensor) -> torch.Tensor:
        """Circular conv with fixed kernel: signal [B,1,T] -> [B,1,T]."""
        length = signal.shape[-1]
        kernel = self.kernel
        if kernel.numel() < length:
            pad = torch.zeros(length - kernel.numel(), device=kernel.device, dtype=kernel.dtype)
            kernel = torch.cat([kernel, pad], dim=0)
        elif kernel.numel() > length:
            kernel = kernel[:length]
        kernel_b = kernel.view(1, 1, -1)
        signal_f = torch.fft.rfft(signal, n=length, dim=-1)
        kernel_f = torch.fft.rfft(kernel_b, n=length, dim=-1)
        return torch.fft.irfft(signal_f * kernel_f, n=length, dim=-1)

    def _corr(self, signal: torch.Tensor) -> torch.Tensor:
        """Circular correlation H^T x via conj(FFT(h))."""
        length = signal.shape[-1]
        kernel = self.kernel
        if kernel.numel() < length:
            pad = torch.zeros(length - kernel.numel(), device=kernel.device, dtype=kernel.dtype)
            kernel = torch.cat([kernel, pad], dim=0)
        elif kernel.numel() > length:
            kernel = kernel[:length]
        kernel_b = kernel.view(1, 1, -1)
        signal_f = torch.fft.rfft(signal, n=length, dim=-1)
        kernel_f = torch.fft.rfft(kernel_b, n=length, dim=-1)
        return torch.fft.irfft(signal_f * torch.conj(kernel_f), n=length, dim=-1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        observation : [B,1,T] demeaned pressure / difference signal

        Returns
        -------
        reflectivity : [B,1,T]
        """
        if observation.ndim != 3 or observation.shape[1] != 1:
            raise ValueError(f"expected [B,1,T], got {tuple(observation.shape)}")
        y = observation
        r = torch.zeros_like(y)
        for layer in range(self.config.n_layers):
            step = self.steps[layer].abs().clamp_min(1.0e-8)
            thresh = self.thresholds[layer].abs()
            residual = self._conv(r) - y
            grad = self._corr(residual)
            r = soft_threshold(r - step * grad, thresh)
        return r


def reflectivity_to_event_map(reflectivity: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Map sparse r to a non-negative peak map in [0,1] for event detection."""
    magnitude = torch.abs(reflectivity)
    peak = magnitude.amax(dim=-1, keepdim=True).clamp_min(eps)
    return (magnitude / peak).clamp(0.0, 1.0)


def demean_observation(observation: torch.Tensor) -> torch.Tensor:
    """Per-trace mean removal; keeps channel dim."""
    return observation - observation.mean(dim=-1, keepdim=True)


def build_or_load_kernel(
    cache_path: str,
    *,
    seq_length: int,
    tf_s: float,
    wavespeed_m_s: float,
    pump_shut_time_s: float,
    well_length_m: float = 5000.0,
    friction: str = "brunone",
) -> Tuple[np.ndarray, Dict]:
    """Estimate a fixed wavelet from no-fracture MOC and cache it."""
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    meta = {
        "seq_length": seq_length,
        "tf_s": tf_s,
        "wavespeed_m_s": wavespeed_m_s,
        "pump_shut_time_s": pump_shut_time_s,
        "well_length_m": well_length_m,
        "friction": friction,
    }
    if os.path.isfile(cache_path):
        payload = np.load(cache_path, allow_pickle=False)
        kernel = np.asarray(payload["kernel"], dtype=np.float32)
        return kernel, meta

    reference = generate_reference_signal(friction=friction)
    t_ref = np.asarray(reference["t"], dtype=float)
    h_ref = np.asarray(reference["H_wh"], dtype=float)
    fs_ref = float(reference["fs"])
    a_adj = float(reference["a_adj"])
    kernel_native = extract_wavelet(
        h_ref, fs_ref, a_adj, well_length_m, pump_shut_time_s, t_ref
    ).astype(np.float32)

    # Resample wavelet onto the direct-inverse grid spacing.
    dt_target = tf_s / float(seq_length - 1)
    duration = (len(kernel_native) - 1) / fs_ref
    n_target = max(8, int(round(duration / dt_target)) + 1)
    t_native = np.arange(len(kernel_native), dtype=np.float64) / fs_ref
    t_target = np.arange(n_target, dtype=np.float64) * dt_target
    kernel = np.interp(t_target, t_native, kernel_native).astype(np.float32)
    # Unit-energy normalize for stable LISTA Lipschitz init.
    energy = float(np.linalg.norm(kernel))
    if energy > 0:
        kernel = kernel / energy
    np.savez_compressed(cache_path, kernel=kernel, **{key: np.asarray(value) for key, value in meta.items()})
    return kernel, meta


def lista_parameter_count(model: LISTA1D) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "config": asdict(model.config),
    }
