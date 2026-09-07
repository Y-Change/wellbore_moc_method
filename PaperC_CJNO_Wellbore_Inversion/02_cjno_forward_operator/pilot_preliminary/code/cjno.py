# -*- coding: utf-8 -*-
"""Pilot prototype — not the full CJ-NO of §4.1.

Has: cluster-set encoder, temporal wellhead trunk, optional hard/soft node head.
Lacks: characteristic-coordinate Fourier segment propagator, full-field trunk,
       memory-channel vs MOC z_l(x,t) pointwise (no interior snapshots).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import F_CLUSTER, N_MAX
from interface_newton import ImplicitNodeLayer


class ClusterSetEncoder(nn.Module):
    def __init__(self, d_in=F_CLUSTER, d_well=8, d=64, n_heads=4):
        super().__init__()
        self.d = d
        self.cl = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.well = nn.Sequential(nn.Linear(d_well, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, cluster_n, mask, well_n):
        # cluster_n: (B,N,F)  mask: (B,N)
        h = self.cl(cluster_n)
        w = self.well(well_n).unsqueeze(1)
        key_pad = mask < 0.5
        tok, _ = self.attn(h, h, h, key_padding_mask=key_pad)
        tok = tok * mask.unsqueeze(-1)
        pooled = tok.sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        cond = self.out(pooled + w.squeeze(1))
        return tok, cond


class TimeFNO1d(nn.Module):
    """Minimal 1D Fourier layer stack on a regular wellhead grid."""

    def __init__(self, d=32, modes=16, n_layers=3):
        super().__init__()
        self.lift = nn.Linear(1, d)
        self.modes = modes
        self.d = d
        self.n_layers = n_layers
        self.weight = nn.ParameterList(
            [nn.Parameter(torch.randn(d, d, modes, dtype=torch.cfloat) * 0.02)
             for _ in range(n_layers)]
        )
        self.w = nn.ModuleList([nn.Conv1d(d, d, 1) for _ in range(n_layers)])
        self.proj = nn.Linear(d, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _spec(self, x, k):
        # x: (B,d,T)
        xf = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xf.shape[-1])
        out = torch.zeros_like(xf)
        w = self.weight[k][..., :m]
        out[..., :m] = torch.einsum("bdm,dom->bom", xf[..., :m], w)
        return torch.fft.irfft(out, n=x.shape[-1], dim=-1)

    def forward(self, p_scale_cond, T, cond):
        # cond (B,d_cond) broadcast as bias via 1x1 after lift of zeros/query
        B = cond.shape[0]
        grid = torch.linspace(0, 1, T, device=cond.device, dtype=cond.dtype).view(1, T, 1).expand(B, T, 1)
        x = self.lift(grid)  # (B,T,d)
        # add cond
        x = x + cond[:, None, :x.shape[-1]] if cond.shape[-1] == x.shape[-1] else x + F.pad(
            cond, (0, x.shape[-1] - cond.shape[-1]))[:, None, :]
        x = x.transpose(1, 2)
        for k in range(self.n_layers):
            x = F.gelu(self._spec(x, k) + self.w[k](x))
        y = self.proj(x.transpose(1, 2)).squeeze(-1)
        return y


class SoftNodeHead(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4)
        )

    def forward(self, tok, t_n, mask):
        # tok (B,N,d), t_n (B,Tn)
        B, Tn = t_n.shape
        N = tok.shape[1]
        tt = t_n.unsqueeze(2).expand(B, Tn, N).unsqueeze(-1)
        # also sin time
        feat = torch.cat([tok.unsqueeze(1).expand(B, Tn, N, -1), tt, torch.sin(2 * 3.1416 * tt)], -1)
        y = self.net(feat)
        return {
            "H_pert": y[..., 0] * mask[:, None, :],
            "Qm_pert": y[..., 1] * mask[:, None, :],
            "Qp_pert": y[..., 2] * mask[:, None, :],
            "sq": y[..., 3] * mask[:, None, :],
        }


class HardIncomingHead(nn.Module):
    """Predicts incoming C_P, C_M (head units) at node times from tokens."""

    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, tok, t_n, H0, mask):
        B, Tn = t_n.shape
        N = tok.shape[1]
        tt = t_n.unsqueeze(2).expand(B, Tn, N).unsqueeze(-1)
        feat = torch.cat([tok.unsqueeze(1).expand(B, Tn, N, -1), tt, torch.sin(2 * 3.1416 * tt)], -1)
        dc = self.net(feat)
        CP = H0[:, None, None] + dc[..., 0]
        CM = H0[:, None, None] + dc[..., 1]
        return CP * mask[:, None, :], CM * mask[:, None, :]


class PilotPrototype(nn.Module):
    """mode: hard | soft | fno  (fno shares the encoder)."""

    def __init__(self, mode="hard", d=64, fno_d=64, fno_modes=16):
        super().__init__()
        self.mode = mode
        self.enc = ClusterSetEncoder(d=d)
        self.cond_proj = nn.Linear(d, fno_d)
        self.fno = TimeFNO1d(d=fno_d, modes=fno_modes, n_layers=3)
        self.soft = SoftNodeHead(d=d)
        self.incoming = HardIncomingHead(d=d)
        self.node = ImplicitNodeLayer()
        # physical B placeholder used only when applying Newton on a few times
        self.default_B = 80.0

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def wellhead(self, batch, cond) -> torch.Tensor:
        T = batch["p_pert"].shape[1]
        c = self.cond_proj(cond)
        return self.fno(None, T, c) * batch["p_pert_scale"]

    def forward(self, batch: Dict, apply_hard_times: int = 8) -> Dict:
        cluster = batch["cluster_n"]
        mask = batch["mask"]
        well = batch["well_n"]
        tok, cond = self.enc(cluster, mask, well)
        p_hat = self.wellhead(batch, cond)
        out = {"p_pert_hat": p_hat, "cond": cond, "tok": tok}
        if self.mode == "soft":
            out["node"] = self.soft(tok, batch["node_t"], mask)
        elif self.mode == "hard":
            # teacher-forced previous state is NOT used as future waveform input;
            # incoming CP/CM are predicted from params+time only.
            CP, CM = self.incoming(tok, batch["node_t"], batch["H0"], mask)
            out["CP"], out["CM"] = CP, CM
            out["node"] = self.soft(tok, batch["node_t"], mask)  # cheap readout
            out["hard_samples"] = self._hard_subsample(batch, CP, CM, apply_hard_times)
        else:
            out["node"] = None
        return out

    def _hard_subsample(self, batch, CP, CM, n_times: int):
        """Apply implicit layer on a few node times (CPU budget)."""
        B, Tn, N = CP.shape
        device = CP.device
        n_use = min(n_times, Tn)
        idx = torch.linspace(0, Tn - 1, n_use, device=device).long()
        rec = []
        layer = self.node
        for b in range(B):
            rho_g = float(batch["rho_g"][b])
            D = float(batch["well"][b, 1])
            a = float(batch["well"][b, 2])
            A = 3.141592653589793 * D * D / 4.0
            Bphys = a / (9.80665 * A)
            H_scale = float(batch.get("H_scale", [4000.0])[b] if "H_scale" in batch else 4000.0)
            Q_scale = float(max(abs(float(batch["Q0"][b])), 1e-3))
            p_scale = rho_g * H_scale
            Nb = int(batch["N"][b])
            for j in range(Nb):
                for it in idx.tolist():
                    if float(batch["node_mask_t"][b, it]) < 0.5:
                        continue
                    pack = {
                        "CP": CP[b, it, j].to(torch.float64),
                        "CM": CM[b, it, j].to(torch.float64),
                        "BL": torch.tensor(Bphys, dtype=torch.float64, device=device),
                        "BR": torch.tensor(Bphys, dtype=torch.float64, device=device),
                        "RL": torch.tensor(0.0, dtype=torch.float64, device=device),
                        "RR": torch.tensor(0.0, dtype=torch.float64, device=device),
                        "z_j": torch.tensor(0.0, dtype=torch.float64, device=device),
                        "rho_g": torch.tensor(rho_g, dtype=torch.float64, device=device),
                        "K": torch.tensor([2e10], dtype=torch.float64, device=device),
                        "kappa": torch.tensor([0.0], dtype=torch.float64, device=device),
                        "eps_q": 1e-8,
                        "c1": torch.tensor([1e7], dtype=torch.float64, device=device),
                        "c0": torch.tensor([0.0], dtype=torch.float64, device=device),
                        "a1": torch.tensor([0.0], dtype=torch.float64, device=device),
                        "a0": torch.tensor([float(batch["node_pc"][b, it, j])], dtype=torch.float64, device=device),
                        "H_init": batch["node_H"][b, it, j].to(torch.float64),
                        "q_init": torch.tensor([float(batch["node_sq"][b, it, j])], dtype=torch.float64, device=device),
                        "H_scale": H_scale, "Q_scale": Q_scale, "p_scale": p_scale,
                    }
                    sol = layer.forward_one(pack)
                    rec.append({
                        "b": b, "j": j, "t": it,
                        "H": sol["H"], "Qm": sol["Qm"], "Qp": sol["Qp"],
                        "H_true": batch["node_H"][b, it, j].to(torch.float64),
                        "Qm_true": batch["node_Qm"][b, it, j].to(torch.float64),
                        "Qp_true": batch["node_Qp"][b, it, j].to(torch.float64),
                        "resid": sol["stat"][2],
                    })
        return rec


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
