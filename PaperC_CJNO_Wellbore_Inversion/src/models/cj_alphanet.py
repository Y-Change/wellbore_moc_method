# -*- coding: utf-8 -*-
"""
CJ-AlphaNet：停泵后全速率波形 + 簇位点 1D/2D 倒谱取样 + 声学距离偏置 Transformer。

输出：pred_m_alpha, pred_active, pred_logY, pred_alpha；
Ŷ = Y0 * 10^{ŷ}；Γ 由闭式 Γ = -Ŷ/(2 Y0 + Ŷ) 给出，无 Cf 头。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.modules.acoustic_transformer import (
    AcousticBiasedTransformer,
    FourierPositionalEncoding,
)
from PaperC_CJNO_Wellbore_Inversion.src.modules.dual_track_heads import ContinuousTrunkHead
from PaperC_CJNO_Wellbore_Inversion.src.modules.time_gating import RelativeWindowGating


class LargeStrideWaveEncoder(nn.Module):
    """大步长 1D CNN，把 ~60000 点压到全局向量 z_wave。"""

    def __init__(self, in_channels: int = 2, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=8, padding=7),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, out_dim),
            nn.GELU(),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(wave))


class CJAlphaNet(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_nc: int = 6,
        L: float = 5000.0,
        n_grid: int = 500,
        n_cep2d_frames: int = 41,
        t_total: float = 60.0,
        use_cep2d: bool = True,
        use_a_hat_cond: bool = True,
        p: int = 64,
        d_well: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_nc = max_nc
        self.L = L
        self.n_grid = n_grid
        self.n_cep2d_frames = n_cep2d_frames
        self.t_total = t_total
        self.use_cep2d = bool(use_cep2d)
        self.use_a_hat_cond = bool(use_a_hat_cond)
        cond_dim = 1 if use_a_hat_cond else 0
        self.cond_dim = max(cond_dim, 1)

        self.wave_encoder = LargeStrideWaveEncoder(in_channels=2, out_dim=d_model)
        self.time_gate = RelativeWindowGating(
            in_channels=2,
            d_model=d_model,
            n_win_pts=128,
            t_pre=0.05,
            t_post=0.25,
            t_total=t_total,
        )
        self.cep1d_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.cep2d_mlp = nn.Sequential(
            nn.Linear(n_cep2d_frames, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )
        token_in = d_model + 16 + 32 + self.cond_dim
        self.token_proj = nn.Sequential(
            nn.Linear(token_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.transformer = AcousticBiasedTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=128,
            cond_dim=self.cond_dim,
            num_pos_freqs=8,
            dropout=0.0,
            use_acoustic_bias=True,
            init_gamma=10.0,
        )
        self.well_fusion = nn.Sequential(
            nn.Linear(d_model + d_model + self.cond_dim, d_well),
            nn.LayerNorm(d_well),
            nn.GELU(),
            nn.Linear(d_well, d_well),
            nn.GELU(),
        )
        self.continuous_head = ContinuousTrunkHead(
            d_well=d_well,
            p=p,
            num_freqs=10,
            n_grid=n_grid,
            L=L,
            cf_ref=0.01,
        )
        self.active_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.logY_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.alpha_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.alpha_scale = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        self.pos_embed = FourierPositionalEncoding(num_freqs=8)

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(~mask, -1e9)
        alpha = F.softmax(masked, dim=-1)
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
        s = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha = alpha / s
        return torch.where(mask, alpha, torch.zeros_like(alpha))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wave = batch["wave"]
        positions = batch["positions"]
        mask = batch["mask_design"] if "mask_design" in batch else batch["mask"]
        a_hat = batch["a_hat"].float().view(-1)
        Y0 = batch["Y0"].float().view(-1)
        B, M = positions.shape
        if "norm_positions" in batch:
            norm_pos = batch["norm_positions"]
        else:
            norm_pos = positions / self.L
        cond = batch["cond"] if "cond" in batch else ((a_hat - 1400.0) / 200.0).unsqueeze(-1)
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)

        z_wave = self.wave_encoder(wave)
        tau = 2.0 * positions / a_hat.view(B, 1).clamp(min=500.0)
        wave_tok = self.time_gate(wave=wave, tau=tau, mask=mask)

        c1 = batch["cepstrum_1d_at_xj"].unsqueeze(-1)
        tok_1d = self.cep1d_mlp(c1)
        c2 = batch["cepstrum_2d_at_xj"]
        # 数据集为 (B, n_frames, M)；簇 token 需要 (B, M, n_frames)
        if c2.dim() == 3 and c2.shape[-1] == M and c2.shape[1] != M:
            c2 = c2.transpose(1, 2)
        n_fr = c2.shape[-1]
        if n_fr < self.n_cep2d_frames:
            c2 = F.pad(c2, (0, self.n_cep2d_frames - n_fr))
        elif n_fr > self.n_cep2d_frames:
            c2 = c2[..., : self.n_cep2d_frames]
        if self.use_cep2d:
            tok_2d = self.cep2d_mlp(c2)
        else:
            tok_2d = torch.zeros(B, M, 32, device=wave.device, dtype=wave.dtype)

        a_tok = cond.unsqueeze(1).expand(-1, M, -1)
        raw = torch.cat([wave_tok, tok_1d, tok_2d, a_tok], dim=-1)
        tokens = self.token_proj(raw)
        tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        h, attn = self.transformer(
            cluster_tokens=tokens,
            positions=positions,
            norm_positions=norm_pos,
            cond=cond,
            mask=mask,
            wavespeed=a_hat,
        )

        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        z_well = self.well_fusion(torch.cat([pooled, z_wave, cond], dim=-1))
        m_alpha, _c_grid, alpha_field, _cf_f, _logcf_f = self.continuous_head(
            z_well=z_well,
            positions=positions,
            mask=mask,
        )
        # 连续场只保留进液密度；顺应性场丢弃
        alpha_field = torch.where(mask, alpha_field, torch.zeros_like(alpha_field))
        af_sum = alpha_field.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha_field_n = torch.where(mask, alpha_field / af_sum, torch.zeros_like(alpha_field))

        act_logits = self.active_mlp(h).squeeze(-1)
        act_logits = torch.where(mask, act_logits, torch.zeros_like(act_logits))
        pred_active = torch.sigmoid(act_logits)
        pred_active = torch.where(mask, pred_active, torch.zeros_like(pred_active))

        pred_logY = self.logY_mlp(h).squeeze(-1)
        pred_logY = torch.where(mask, pred_logY, torch.zeros_like(pred_logY))

        alpha_logits = self.alpha_mlp(h).squeeze(-1) * F.softplus(self.alpha_scale)
        pred_alpha = self._masked_softmax(alpha_logits, mask)

        Y_hat = Y0.view(B, 1) * torch.pow(10.0, pred_logY)
        Y_hat = torch.where(mask, Y_hat, torch.zeros_like(Y_hat))
        y0e = Y0.view(B, 1).clamp(min=1e-16)
        gamma = -Y_hat / (2.0 * y0e + Y_hat + 1e-16)
        gamma = torch.where(mask, gamma, torch.zeros_like(gamma))

        return {
            "pred_m_alpha": m_alpha,
            "m_alpha_grid": m_alpha,
            "pred_active": pred_active,
            "pred_active_logits": act_logits,
            "pred_logY": pred_logY,
            "pred_alpha": pred_alpha,
            "alpha": pred_alpha,
            "alpha_field": alpha_field_n,
            "Y_hat": Y_hat,
            "gamma": gamma,
            "tau": tau,
            "attn_weights": attn,
            "z_well": z_well,
            "z_wave": z_wave,
        }
