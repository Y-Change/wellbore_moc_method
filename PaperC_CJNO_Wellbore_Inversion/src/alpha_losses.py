# -*- coding: utf-8 -*-
"""
CJ-AlphaNet 多任务损失（不监督 Cf）。

L = λ_m L_mα + λ_act L_act + λ_Y L_Y + λ_α L_α + λ_cons L_cons
默认 λ_m=1, λ_act=0.5, λ_Y=0.4, λ_α=0.1, λ_cons=0.1
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.losses import (
    DualTrackConsistencyLoss,
    SimplexKLDivergenceLoss,
    Wasserstein1DLoss,
)


class AlphaCompositeLoss(nn.Module):
    def __init__(
        self,
        lambda_m: float = 1.0,
        lambda_act: float = 0.5,
        lambda_Y: float = 0.4,
        lambda_alpha: float = 0.1,
        lambda_cons: float = 0.1,
        L: float = 5000.0,
        w1_scale_m: float = 20.0,
        field_mae_scale: float = 200.0,
    ):
        super().__init__()
        self.lambda_m = float(lambda_m)
        self.lambda_act = float(lambda_act)
        self.lambda_Y = float(lambda_Y)
        self.lambda_alpha = float(lambda_alpha)
        self.lambda_cons = float(lambda_cons)
        self.w1_scale_m = float(w1_scale_m)
        self.field_mae_scale = float(field_mae_scale)
        self.loss_alpha = SimplexKLDivergenceLoss()
        self.loss_w1 = Wasserstein1DLoss(L=L, normalize_by_L=False)
        self.loss_cons = DualTrackConsistencyLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2, reduction="none")

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask = target_dict["mask_design"] if "mask_design" in target_dict else target_dict["mask"]
        y_act = target_dict["y_act"].float()
        pred_alpha = pred_dict["pred_alpha"] if "pred_alpha" in pred_dict else pred_dict["alpha"]
        pred_m = pred_dict["pred_m_alpha"] if "pred_m_alpha" in pred_dict else pred_dict["m_alpha_grid"]
        target_m = target_dict["m_alpha_grid"]

        # 密度：W1(m) 与场 MAE，缩放到 O(1)
        l_w1 = self.loss_w1(pred_m, target_m)
        l_field = torch.mean(torch.abs(pred_m - target_m))
        l_m = l_w1 / max(self.w1_scale_m, 1e-6) + self.field_mae_scale * l_field

        # 活动簇：仅 design 槽
        if "pred_active_logits" in pred_dict:
            act_logits = pred_dict["pred_active_logits"]
            bce = self.bce(act_logits, y_act)
        else:
            p = torch.clamp(pred_dict["pred_active"], 1e-6, 1.0 - 1e-6)
            bce = -(y_act * torch.log(p) + (1.0 - y_act) * torch.log(1.0 - p))
        bce = torch.where(mask, bce, torch.zeros_like(bce))
        n_des = mask.sum().clamp(min=1).float()
        l_act = bce.sum() / n_des

        # 导纳：仅活动簇
        pred_logY = pred_dict["pred_logY"] if "pred_logY" in pred_dict else pred_dict["logY"]
        target_logY = target_dict["logY"]
        act_mask = mask & (y_act > 0.5)
        l_y_mat = self.smooth_l1(pred_logY, target_logY)
        l_y_mat = torch.where(act_mask, l_y_mat, torch.zeros_like(l_y_mat))
        n_act = act_mask.sum().clamp(min=1).float()
        l_Y = l_y_mat.sum() / n_act

        # 比例：全部 design 槽（含砂堵）
        l_alpha = self.loss_alpha(pred_alpha, target_dict["alpha"], mask)

        if "alpha_field" in pred_dict:
            l_cons = self.loss_cons(pred_alpha, pred_dict["alpha_field"], mask)
        else:
            l_cons = pred_alpha.new_zeros(())

        total = (
            self.lambda_m * l_m
            + self.lambda_act * l_act
            + self.lambda_Y * l_Y
            + self.lambda_alpha * l_alpha
            + self.lambda_cons * l_cons
        )
        return {
            "loss": total,
            "loss_m": l_m,
            "loss_w1_m": l_w1,
            "loss_act": l_act,
            "loss_Y": l_Y,
            "loss_alpha": l_alpha,
            "loss_cons": l_cons,
        }
