# -*- coding: utf-8 -*-
"""Predict-only models. No future labels in the forward path."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

class QueryFourierMLP(nn.Module):
    """Wellhead baseline: physical features + Fourier time → p_pert.

    Not CJ-NO. No node path. Used for C1 overfit diagnosis.
    """

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
        # tau: (B,T)
        parts = [tau.unsqueeze(-1)]
        for k in range(1, self.n_harmonics + 1):
            parts.append(torch.sin(2 * np.pi * k * tau).unsqueeze(-1))
            parts.append(torch.cos(2 * np.pi * k * tau).unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def predict(self, feat: torch.Tensor, tau: torch.Tensor, p_scale: float) -> dict:
        """feat (B,F) physical only; tau (B,T) = (t-t_s)/(4L/a)."""
        B, T = tau.shape
        tf = self.time_feat(tau)
        f = feat[:, None, :].expand(B, T, feat.shape[-1])
        x = torch.cat([f, tf], dim=-1)
        y = self.net(x).squeeze(-1) * p_scale
        return {"p_pert": y, "n_hard_calls": torch.zeros((), device=tau.device),
                "hard_resid": torch.zeros((), device=tau.device)}

    def n_active_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def loss_head(pred, p_pert, mask, scale):
    err = (pred["p_pert"] - p_pert) * mask
    return (err ** 2).sum() / mask.sum().clamp_min(1.0) / (scale ** 2 + 1e-12)
