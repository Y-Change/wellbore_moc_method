# -*- coding: utf-8 -*-
"""Predict-only models. Same QueryFourierMLP as Round2; losses are Round3 arms."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QueryFourierMLP(nn.Module):
    def __init__(self, d_feat: int, d=128, n_harmonics=32, n_layers=4):
        super().__init__()
        self.n_harmonics = int(n_harmonics)
        d_in = d_feat + 1 + 2 * self.n_harmonics
        layers = [nn.Linear(d_in, d), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d, d), nn.GELU()]
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def time_feat(self, tau: torch.Tensor) -> torch.Tensor:
        parts = [tau.unsqueeze(-1)]
        for k in range(1, self.n_harmonics + 1):
            parts.append(torch.sin(2 * np.pi * k * tau).unsqueeze(-1))
            parts.append(torch.cos(2 * np.pi * k * tau).unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def predict(self, feat: torch.Tensor, tau: torch.Tensor, p_scale: float) -> dict:
        B, T = tau.shape
        tf = self.time_feat(tau)
        f = feat[:, None, :].expand(B, T, feat.shape[-1])
        x = torch.cat([f, tf], dim=-1)
        y = self.net(x).squeeze(-1) * p_scale
        return {"p_pert": y}

    def n_active_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def loss_b1_shared_mse(pred, p_pert, mask, scale):
    """Shared-scale absolute MSE (Round2 contract)."""
    err = (pred["p_pert"] - p_pert) * mask
    return (err ** 2).sum() / mask.sum().clamp_min(1.0) / (scale ** 2 + 1e-12)


def loss_b2_energy_equal(pred, p_pert, mask, silent, target_l2):
    """Per-case target-energy normalized SE, then equal-weight mean over cases.

    Denominator is ||target|| only. Silent cases use absolute error and are
    excluded from the equal-weight energy mean; they are returned separately.
    """
    err = (pred["p_pert"] - p_pert) * mask
    se = (err ** 2).sum(dim=1)
    den = target_l2.clamp_min(1e-30) ** 2
    per = se / den
    active = (~silent) & (mask.sum(dim=1) > 0)
    if active.any():
        main = per[active].mean()
    else:
        main = se.new_zeros(())
    silent_abs = se[silent].mean() if silent.any() else se.new_zeros(())
    return main, silent_abs, per
