# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.losses

水击波物理反演复合损失函数库：
1. SimplexKLDivergenceLoss: 概率单纯形上基于 KL 散度的流量份额 alpha 监督；
2. LogHuberComplianceLoss: 对数尺度跨量级水力顺应性 Cf 的 Log-Huber 鲁棒回归；
3. Wasserstein1DLoss: 连续进液密度场 m_alpha(x) 与离散测度的可微 1D Wasserstein-1 距离；
4. CompositeInversionLoss: 物理加权复合多目标反演损失。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplexKLDivergenceLoss(nn.Module):
    """
    针对流量份额向量 alpha in Simplex (sum=1, alpha >= 0) 的 KL 散度与 L1 复合损失。
    """

    def __init__(self, eps: float = 1e-8, l1_weight: float = 0.5, mse_weight: float = 2.0):
        super().__init__()
        self.eps = eps
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(
        self,
        pred_alpha: torch.Tensor,   # (B, M)
        target_alpha: torch.Tensor, # (B, M)
        mask: torch.Tensor,         # (B, M) bool
    ) -> torch.Tensor:
        # 裁剪保证预测概率数值安全
        pred_clamped = torch.clamp(pred_alpha, min=self.eps, max=1.0)

        # 概率单纯形上的交叉熵损失: - sum_j target_j * log(pred_j)
        ce = - target_alpha * torch.log(pred_clamped)
        ce_masked = torch.where(mask, ce, torch.zeros_like(ce))
        loss_ce = ce_masked.sum(dim=-1).mean()

        # 辅助 L1 损失 (MAE)
        l1 = torch.abs(pred_alpha - target_alpha)
        l1_masked = torch.where(mask, l1, torch.zeros_like(l1))
        num_valid = mask.sum(dim=-1).clamp(min=1.0).float()
        loss_l1 = (l1_masked.sum(dim=-1) / num_valid).mean()

        # 辅助 MSE 损失 (直接驱动 R^2 决定系数分子收敛)
        if self.mse_weight > 0.0:
            mse = (pred_alpha - target_alpha) ** 2
            mse_masked = torch.where(mask, mse, torch.zeros_like(mse))
            loss_mse = (mse_masked.sum(dim=-1) / num_valid).mean()
            return loss_ce + self.l1_weight * loss_l1 + self.mse_weight * loss_mse

        return loss_ce + self.l1_weight * loss_l1


class LogHuberComplianceLoss(nn.Module):
    """
    对数尺度水力顺应性 Log-Huber 损失函数。
    输入为 log(Cf / Cf0) 或 Cf，在对数空间执行 Smooth L1 回归。
    """

    def __init__(self, delta: float = 0.2):
        super().__init__()
        self.delta = delta
        self.smooth_l1 = nn.SmoothL1Loss(beta=delta, reduction="none")

    def forward(
        self,
        pred_log_cf: torch.Tensor,   # (B, M)
        target_log_cf: torch.Tensor, # (B, M)
        mask: torch.Tensor,          # (B, M) bool
    ) -> torch.Tensor:
        loss_matrix = self.smooth_l1(pred_log_cf, target_log_cf)
        masked_loss = torch.where(mask, loss_matrix, torch.zeros_like(loss_matrix))
        total_valid = mask.sum().clamp(min=1).float()
        return masked_loss.sum() / total_valid


class Wasserstein1DLoss(nn.Module):
    """
    可微 1D Wasserstein-1 距离算子 (Earth Mover's Distance)：
    W1(p, q) = int_0^L |F_p(x) - F_q(x)| dx
    可作用于连续密度场网格，亦支持离散簇坐标下的解析 CDF 差分积分。
    """

    def __init__(self, L: float = 5000.0, normalize_by_L: bool = True):
        super().__init__()
        self.L = L
        self.normalize_by_L = normalize_by_L

    def forward(
        self,
        pred_field: torch.Tensor,   # (B, N_grid) 或 pred_alpha (B, M)
        target_field: torch.Tensor, # (B, N_grid) 或 target_alpha (B, M)
        positions: torch.Tensor = None,  # (B, M) 仅在离散模式下需要
        mask: torch.Tensor = None,       # (B, M)
    ) -> torch.Tensor:
        """
        若输入是 (B, N_grid) 连续网格：
        """
        if pred_field.ndim == 2 and pred_field.shape[-1] > 10:
            n_grid = pred_field.shape[-1]
            dx = self.L / (n_grid - 1)

            # 计算经验累积分布函数 CDF: F(x) = cumsum(m(x)) * dx
            cdf_pred = torch.cumsum(pred_field, dim=-1) * dx
            cdf_target = torch.cumsum(target_field, dim=-1) * dx

            # 归一化至 [0, 1] 严格单调累积测度
            total_pred = cdf_pred[..., -1:].clamp(min=1e-6)
            total_target = cdf_target[..., -1:].clamp(min=1e-6)
            cdf_pred = cdf_pred / total_pred
            cdf_target = cdf_target / total_target

            # 积分 CDF 差绝对值: int |F_pred(x) - F_target(x)| dx
            w1 = torch.sum(torch.abs(cdf_pred - cdf_target), dim=-1) * dx
            if self.normalize_by_L:
                w1 = w1 / self.L
            return w1.mean()

        # 若输入是离散簇参数 (B, M) 与有序坐标 positions (B, M)
        if positions is not None and mask is not None:
            B, M = pred_field.shape
            w1_list = []
            for b in range(B):
                m_b = mask[b]
                nc = int(m_b.sum().item())
                if nc <= 1:
                    w1_list.append(torch.tensor(0.0, device=pred_field.device))
                    continue
                pos_b = positions[b, :nc]
                p_b = pred_field[b, :nc]
                t_b = target_field[b, :nc]

                # 累积质量
                cdf_p = torch.cumsum(p_b, dim=0)
                cdf_t = torch.cumsum(t_b, dim=0)
                dx_b = pos_b[1:] - pos_b[:-1]  # (nc-1,)

                # 差值积分
                diff_cdf = torch.abs(cdf_p[:-1] - cdf_t[:-1])
                w1_val = torch.sum(diff_cdf * dx_b)
                if self.normalize_by_L:
                    w1_val = w1_val / self.L
                w1_list.append(w1_val)
            return torch.stack(w1_list).mean()

        raise ValueError("输入维度不支持或缺少必要坐标参数")


class DualTrackConsistencyLoss(nn.Module):
    """
    双轨协同一致性损失 (Dual-Track Consistency Loss):
    L_cons = sum_j |hat{alpha}_j - int_{Omega_j} hat{m}_alpha(x) dx|
    约束离散精准头 (轨 1) 与连续场 Voronoi 守恒空间积分 (轨 2) 达成物理自洽。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_alpha: torch.Tensor,       # (B, M) 轨 1 离散预测
        alpha_field: torch.Tensor,      # (B, M) 轨 2 连续场 Voronoi 守恒积分
        mask: torch.Tensor,             # (B, M) bool
    ) -> torch.Tensor:
        diff = torch.abs(pred_alpha - alpha_field)
        masked_diff = torch.where(mask, diff, torch.zeros_like(diff))
        num_valid = mask.sum(dim=-1).clamp(min=1.0).float()
        return (masked_diff.sum(dim=-1) / num_valid).mean()


class CompositeInversionLoss(nn.Module):
    """
    PaperC 物理反演全局多目标复合损失函数。
    L = lambda_alpha * L_alpha + lambda_c * L_c + lambda_w * L_w + lambda_cons * L_cons
    各分量损失经物理尺度归一化平衡在 [0.05, 0.5] 区间。
    """

    def __init__(
        self,
        lambda_alpha: float = 1.0,
        lambda_c: float = 1.0,
        lambda_w: float = 0.01,
        lambda_cons: float = 0.5,
        lambda_exist: float = 0.0,
        lambda_pos: float = 0.0,
        huber_delta: float = 0.2,
        L: float = 5000.0,
    ):
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.lambda_c = lambda_c
        self.lambda_w = lambda_w
        self.lambda_cons = lambda_cons
        self.lambda_exist = lambda_exist
        self.lambda_pos = lambda_pos

        self.loss_alpha = SimplexKLDivergenceLoss()
        self.loss_c = LogHuberComplianceLoss(delta=huber_delta)
        # normalize_by_L=False 输出以米为单位的 W1 距离
        self.loss_w1 = Wasserstein1DLoss(L=L, normalize_by_L=False)
        self.loss_cons = DualTrackConsistencyLoss()

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask = target_dict["mask"]
        pred_alpha = pred_dict["alpha"]
        target_alpha = target_dict["alpha"]
        pred_log_cf = pred_dict["log_cf"]
        target_log_cf = target_dict["log_cf"]

        # 1. 流量分配损失
        l_alpha = self.loss_alpha(pred_alpha, target_alpha, mask)

        # 2. 顺应性损失
        l_c = self.loss_c(pred_log_cf, target_log_cf, mask)

        # 3. 空间 Wasserstein-1 距离损失
        if "m_alpha_grid" in pred_dict and "m_alpha_grid" in target_dict:
            l_w = self.loss_w1(pred_dict["m_alpha_grid"], target_dict["m_alpha_grid"])
        else:
            # 使用离散簇位置做解析 Wasserstein
            l_w = self.loss_w1(
                pred_alpha,
                target_alpha,
                positions=target_dict["positions"],
                mask=mask,
            )

        # 4. 双轨协同一致性损失
        if "alpha_field" in pred_dict:
            l_cons = self.loss_cons(pred_alpha, pred_dict["alpha_field"], mask)
        else:
            l_cons = torch.tensor(0.0, device=pred_alpha.device)

        # 5. 存在性分类损失 (若模型输出 p_exist)
        if "p_exist" in pred_dict and self.lambda_exist > 0:
            l_exist = F.binary_cross_entropy(pred_dict["p_exist"], mask.float())
        else:
            l_exist = torch.tensor(0.0, device=pred_alpha.device)

        # 6. 位置微调损失 (若模型输出 delta_x 且目标包含 true_positions)
        if "delta_x" in pred_dict and "true_positions" in target_dict and self.lambda_pos > 0:
            pred_pos = target_dict["positions"] + pred_dict["delta_x"]
            diff_pos = torch.abs(pred_pos - target_dict["true_positions"])
            l_pos = torch.where(mask, diff_pos, torch.zeros_like(diff_pos)).sum() / mask.sum().clamp(min=1).float()
        else:
            l_pos = torch.tensor(0.0, device=pred_alpha.device)

        total_loss = (
            self.lambda_alpha * l_alpha
            + self.lambda_c * l_c
            + self.lambda_w * l_w
            + self.lambda_cons * l_cons
            + self.lambda_exist * l_exist
            + self.lambda_pos * l_pos
        )

        out_dict = {
            "loss": total_loss,
            "loss_alpha": l_alpha,
            "loss_c": l_c,
            "loss_w1": l_w,
            "loss_cons": l_cons,
        }
        if "p_exist" in pred_dict and self.lambda_exist > 0:
            out_dict["loss_exist"] = l_exist
        if "delta_x" in pred_dict and "true_positions" in target_dict and self.lambda_pos > 0:
            out_dict["loss_pos"] = l_pos
        return out_dict

