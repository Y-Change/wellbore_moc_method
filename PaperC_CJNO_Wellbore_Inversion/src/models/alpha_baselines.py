# -*- coding: utf-8 -*-
"""
同一 newa 标签/切分上的基线：
- 均匀 α + 全部活动 + Y_eq 地板
- 训练集按簇数统计的均值 α / 活动频率 / 活动簇 logY
- ResNet1D / FNO1D / CJ-Cep 骨干改编（4096 点波形，可选 1024 倒谱）
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.models.fno1d import FNOBlock1D
from PaperC_CJNO_Wellbore_Inversion.src.models.resnet1d import ResBlock1D
from PaperC_CJNO_Wellbore_Inversion.src.models.cj_cep_deeponet import (
    CepstrumEncoder,
    ClusterSetEncoder,
    FourierPositionalEncoding,
    WaveEncoder,
)


def _gauss_density(pred_alpha: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor, L: float = 5000.0, n_grid: int = 500, sigma: float = 20.0) -> torch.Tensor:
    B, M = pred_alpha.shape
    grid = torch.linspace(0.0, L, n_grid, device=pred_alpha.device, dtype=pred_alpha.dtype)
    diff = grid.view(1, 1, n_grid) - positions.unsqueeze(-1)
    gauss = torch.exp(-(diff ** 2) / (2.0 * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)
    w = torch.where(mask, pred_alpha, torch.zeros_like(pred_alpha))
    field = (w.unsqueeze(-1) * gauss).sum(dim=1)
    dx = L / max(n_grid - 1, 1)
    integ = (field.sum(dim=-1, keepdim=True) * dx).clamp(min=1e-8)
    return field / integ


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    a = F.softmax(logits.masked_fill(~mask, -1e9), dim=-1)
    a = torch.where(mask, a, torch.zeros_like(a))
    s = a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    a = a / s
    return torch.where(mask, a, torch.zeros_like(a))


class UniformFloorBaseline(nn.Module):
    """均匀 α、设计槽全部活动、Y_eq 地板。"""

    def __init__(self, y_floor: float = 1e-10, L: float = 5000.0, n_grid: int = 500):
        super().__init__()
        self.y_floor = float(y_floor)
        self.L = L
        self.n_grid = n_grid

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = batch["mask_design"] if "mask_design" in batch else batch["mask"]
        pos = batch["positions"]
        y0 = batch["Y0"].view(-1, 1).clamp(min=1e-16)
        n = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        pred_alpha = torch.where(mask, 1.0 / n, torch.zeros_like(n).expand_as(mask.float()))
        pred_active = mask.float()
        pred_logY = torch.log10((torch.full_like(pred_alpha, self.y_floor) + self.y_floor) / y0)
        pred_logY = torch.where(mask, pred_logY, torch.zeros_like(pred_logY))
        pred_m = _gauss_density(pred_alpha, pos, mask, L=self.L, n_grid=self.n_grid)
        y_hat = y0 * torch.pow(10.0, pred_logY)
        gamma = -y_hat / (2.0 * y0 + y_hat + 1e-16)
        return {
            "pred_alpha": pred_alpha,
            "alpha": pred_alpha,
            "pred_active": pred_active,
            "pred_active_logits": torch.where(mask, torch.full_like(pred_alpha, 8.0), torch.zeros_like(pred_alpha)),
            "pred_logY": pred_logY,
            "pred_m_alpha": pred_m,
            "m_alpha_grid": pred_m,
            "alpha_field": pred_alpha,
            "Y_hat": torch.where(mask, y_hat, torch.zeros_like(y_hat)),
            "gamma": torch.where(mask, gamma, torch.zeros_like(gamma)),
        }


class TrainMeanBaseline(nn.Module):
    """按训练集聚类数 n_frac 的均值 α、活动频率、活动簇 logY。"""

    def __init__(self, stats: Dict[int, Dict[str, np.ndarray]], y_floor: float = 1e-10, L: float = 5000.0, n_grid: int = 500, act_thr: float = 0.5):
        super().__init__()
        self.stats = stats
        self.y_floor = float(y_floor)
        self.L = L
        self.n_grid = n_grid
        self.act_thr = float(act_thr)

    @staticmethod
    def from_dataset(dataset) -> "TrainMeanBaseline":
        stats: Dict[int, Dict[str, np.ndarray]] = {}
        n_frac = dataset.n_frac.numpy()
        alpha = dataset.alpha.numpy()
        y_act = dataset.y_act.numpy()
        logY = dataset.logY.numpy()
        mask = dataset.mask_design.numpy()
        for nf in range(1, 7):
            sel = n_frac == nf
            if not np.any(sel):
                continue
            a = alpha[sel]
            m = mask[sel]
            ya = y_act[sel]
            ly = logY[sel]
            mean_a = np.zeros(6, dtype=np.float64)
            p_act = np.zeros(6, dtype=np.float64)
            mean_ly = np.zeros(6, dtype=np.float64)
            for j in range(nf):
                col_m = m[:, j]
                if col_m.any():
                    mean_a[j] = float(np.mean(a[col_m, j]))
                    p_act[j] = float(np.mean(ya[col_m, j]))
                    act_col = col_m & (ya[:, j] > 0.5)
                    if np.any(act_col):
                        mean_ly[j] = float(np.mean(ly[act_col, j]))
                    else:
                        mean_ly[j] = -6.0
            s = mean_a[:nf].sum()
            if s > 1e-8:
                mean_a[:nf] = mean_a[:nf] / s
            else:
                mean_a[:nf] = 1.0 / nf
            stats[nf] = {"alpha": mean_a, "p_act": p_act, "logY": mean_ly}
        return TrainMeanBaseline(stats)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = batch["mask_design"] if "mask_design" in batch else batch["mask"]
        pos = batch["positions"]
        n_frac = batch["n_frac"]
        y0 = batch["Y0"].view(-1, 1).clamp(min=1e-16)
        B, M = mask.shape
        pred_alpha = torch.zeros(B, M, device=mask.device, dtype=torch.float32)
        pred_p = torch.zeros(B, M, device=mask.device, dtype=torch.float32)
        pred_logY = torch.zeros(B, M, device=mask.device, dtype=torch.float32)
        for b in range(B):
            nf = int(n_frac[b].item())
            nf = max(min(nf, 6), 1)
            st = self.stats.get(nf)
            if st is None:
                n = int(mask[b].sum().item()) or 1
                pred_alpha[b, :n] = 1.0 / n
                pred_p[b, :n] = 1.0
                pred_logY[b, :n] = float(np.log10((self.y_floor + self.y_floor) / max(float(y0[b, 0]), 1e-16)))
            else:
                pred_alpha[b] = torch.as_tensor(st["alpha"], device=mask.device, dtype=torch.float32)
                pred_p[b] = torch.as_tensor(st["p_act"], device=mask.device, dtype=torch.float32)
                pred_logY[b] = torch.as_tensor(st["logY"], device=mask.device, dtype=torch.float32)
            pred_alpha[b] = torch.where(mask[b], pred_alpha[b], torch.zeros_like(pred_alpha[b]))
            s = pred_alpha[b].sum().clamp(min=1e-8)
            pred_alpha[b] = torch.where(mask[b], pred_alpha[b] / s, torch.zeros_like(pred_alpha[b]))
            pred_p[b] = torch.where(mask[b], pred_p[b], torch.zeros_like(pred_p[b]))
            pred_logY[b] = torch.where(mask[b], pred_logY[b], torch.zeros_like(pred_logY[b]))
        pred_m = _gauss_density(pred_alpha, pos, mask, L=self.L, n_grid=self.n_grid)
        logits = torch.where(pred_p >= self.act_thr, torch.full_like(pred_p, 4.0), torch.full_like(pred_p, -4.0))
        y_hat = y0 * torch.pow(10.0, pred_logY)
        gamma = -y_hat / (2.0 * y0 + y_hat + 1e-16)
        return {
            "pred_alpha": pred_alpha,
            "alpha": pred_alpha,
            "pred_active": pred_p,
            "pred_active_logits": logits,
            "pred_logY": pred_logY,
            "pred_m_alpha": pred_m,
            "m_alpha_grid": pred_m,
            "alpha_field": pred_alpha,
            "Y_hat": torch.where(mask, y_hat, torch.zeros_like(y_hat)),
            "gamma": torch.where(mask, gamma, torch.zeros_like(gamma)),
        }


class _AlphaClusterHeads(nn.Module):
    def __init__(self, in_dim: int, L: float = 5000.0, n_grid: int = 500):
        super().__init__()
        self.L = L
        self.n_grid = n_grid
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, cluster_in: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor, y0: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.mlp(cluster_in)
        alpha_logits = out[..., 0]
        logY = out[..., 1]
        act_logits = out[..., 2]
        pred_alpha = _masked_softmax(alpha_logits, mask)
        pred_active = torch.sigmoid(act_logits)
        pred_active = torch.where(mask, pred_active, torch.zeros_like(pred_active))
        pred_logY = torch.where(mask, logY, torch.zeros_like(logY))
        pred_m = _gauss_density(pred_alpha, positions, mask, L=self.L, n_grid=self.n_grid)
        y0e = y0.view(-1, 1).clamp(min=1e-16)
        y_hat = y0e * torch.pow(10.0, pred_logY)
        gamma = -y_hat / (2.0 * y0e + y_hat + 1e-16)
        return {
            "pred_alpha": pred_alpha,
            "alpha": pred_alpha,
            "pred_active": pred_active,
            "pred_active_logits": torch.where(mask, act_logits, torch.zeros_like(act_logits)),
            "pred_logY": pred_logY,
            "pred_m_alpha": pred_m,
            "m_alpha_grid": pred_m,
            "alpha_field": pred_alpha,
            "Y_hat": torch.where(mask, y_hat, torch.zeros_like(y_hat)),
            "gamma": torch.where(mask, gamma, torch.zeros_like(gamma)),
        }


def _pos_features(norm_pos: torch.Tensor) -> torch.Tensor:
    s = norm_pos.unsqueeze(-1)
    two_pi = 2.0 * torch.pi
    return torch.cat(
        [
            s,
            torch.sin(two_pi * s),
            torch.cos(two_pi * s),
            torch.sin(2.0 * two_pi * s),
            torch.cos(2.0 * two_pi * s),
        ],
        dim=-1,
    )


class AlphaResNet1D(nn.Module):
    """ResNet1D 骨干吃 4096 点波形，同一套 α/活动/logY/密度头。"""

    def __init__(self, cond_dim: int = 1, base_channels: int = 32, latent_dim: int = 128, L: float = 5000.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.layer1 = nn.Sequential(
            ResBlock1D(c1),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(c2),
            nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            ResBlock1D(c3),
            nn.Conv1d(c3, latent_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 32), nn.ReLU(inplace=True))
        self.heads = _AlphaClusterHeads(latent_dim + 32 + 5, L=L)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wave = batch["wave_4096"] if "wave_4096" in batch else batch["wave"]
        mask = batch["mask_design"] if "mask_design" in batch else batch["mask"]
        norm_pos = batch["norm_positions"]
        pos = batch["positions"]
        cond = batch["cond"]
        B, M = norm_pos.shape
        x = self.layer3(self.layer2(self.layer1(self.stem(wave)))).squeeze(-1)
        z = torch.cat([x, self.cond_mlp(cond)], dim=-1).unsqueeze(1).expand(-1, M, -1)
        cluster_in = torch.cat([z, _pos_features(norm_pos)], dim=-1)
        return self.heads(cluster_in, mask, pos, batch["Y0"])


class AlphaFNO1D(nn.Module):
    def __init__(self, cond_dim: int = 1, width: int = 64, modes: int = 32, L: float = 5000.0):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv1d(2, width, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.fno1 = FNOBlock1D(width, modes=modes)
        self.fno2 = FNOBlock1D(width, modes=modes)
        self.fno3 = FNOBlock1D(width, modes=modes)
        self.fno4 = FNOBlock1D(width, modes=modes)
        self.proj = nn.Sequential(nn.Conv1d(width, 128, kernel_size=1), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.heads = _AlphaClusterHeads(128 + 32 + 5, L=L)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wave = batch["wave_4096"] if "wave_4096" in batch else batch["wave"]
        mask = batch["mask_design"] if "mask_design" in batch else batch["mask"]
        norm_pos = batch["norm_positions"]
        pos = batch["positions"]
        cond = batch["cond"]
        B, M = norm_pos.shape
        x = self.lift(wave)
        x = self.fno4(self.fno3(self.fno2(self.fno1(x))))
        z = torch.cat([self.proj(x).squeeze(-1), self.cond_mlp(cond)], dim=-1).unsqueeze(1).expand(-1, M, -1)
        cluster_in = torch.cat([z, _pos_features(norm_pos)], dim=-1)
        return self.heads(cluster_in, mask, pos, batch["Y0"])


class AlphaCJCepNet(nn.Module):
    """旧 CJ-Cep 骨干：4096 波形 + 插值到 1024 的 1D 倒谱（仅基线预处理）。"""

    def __init__(self, cond_dim: int = 1, L: float = 5000.0, n_grid: int = 500, p: int = 64):
        super().__init__()
        self.L = L
        self.n_grid = n_grid
        self.p = p
        self.wave_encoder = WaveEncoder(in_channels=2, out_dim=64)
        self.cep_encoder = CepstrumEncoder(in_channels=1, out_dim=64)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.set_encoder = ClusterSetEncoder(num_freqs=8, out_dim=32)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.branch_alpha = nn.Linear(128, p + 1)
        self.trunk_embed = FourierPositionalEncoding(num_freqs=10)
        self.trunk_mlp = nn.Sequential(
            nn.Linear(2 * 10 + 1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, p),
        )
        self.register_buffer("grid_norm", torch.linspace(0.0, 1.0, n_grid))
        self.cluster_from_well = nn.Sequential(nn.Linear(128 + 5, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wave = batch["wave_4096"] if "wave_4096" in batch else batch["wave"]
        ceps = batch["cepstrum"]
        mask = batch["mask_design"] if "mask_design" in batch else batch["mask"]
        norm_pos = batch["norm_positions"]
        pos = batch["positions"]
        cond = batch["cond"]
        B, M = mask.shape
        z = self.fusion(
            torch.cat(
                [
                    self.wave_encoder(wave),
                    self.cep_encoder(ceps),
                    self.cond_mlp(cond),
                    self.set_encoder(norm_pos, mask),
                ],
                dim=-1,
            )
        )
        ba = self.branch_alpha(z)
        b, bias = ba[:, : self.p], ba[:, self.p :]
        t_grid = self.trunk_mlp(self.trunk_embed(self.grid_norm.unsqueeze(-1)))
        m = F.softplus(torch.matmul(b, t_grid.t()) + bias)
        dx = self.L / max(self.n_grid - 1, 1)
        m = m / (m.sum(dim=-1, keepdim=True) * dx).clamp(min=1e-8)
        cluster_in = torch.cat([z.unsqueeze(1).expand(-1, M, -1), _pos_features(norm_pos)], dim=-1)
        out = self.cluster_from_well(cluster_in)
        pred_alpha = _masked_softmax(out[..., 0], mask)
        pred_logY = torch.where(mask, out[..., 1], torch.zeros_like(out[..., 1]))
        act_logits = out[..., 2]
        pred_active = torch.where(mask, torch.sigmoid(act_logits), torch.zeros_like(act_logits))
        y0e = batch["Y0"].view(-1, 1).clamp(min=1e-16)
        y_hat = y0e * torch.pow(10.0, pred_logY)
        gamma = -y_hat / (2.0 * y0e + y_hat + 1e-16)
        return {
            "pred_alpha": pred_alpha,
            "alpha": pred_alpha,
            "pred_active": pred_active,
            "pred_active_logits": torch.where(mask, act_logits, torch.zeros_like(act_logits)),
            "pred_logY": pred_logY,
            "pred_m_alpha": m,
            "m_alpha_grid": m,
            "alpha_field": pred_alpha,
            "Y_hat": torch.where(mask, y_hat, torch.zeros_like(y_hat)),
            "gamma": torch.where(mask, gamma, torch.zeros_like(gamma)),
        }
