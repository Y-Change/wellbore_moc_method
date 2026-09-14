# -*- coding: utf-8 -*-
r"""
PaperC_CJNO_Wellbore_Inversion.src.modules.layer_stripping

波动方程可微逆散射层剥离算子 (Differentiable Layer-Stripping Operator, DIS-Op):
基于一维瞬变流声学传递矩阵与 Schur/Bruckstein 逆散射原理，
严格沿井筒由跟端到趾端 (x_1 -> x_{N_c}) 逐级消除上游裂缝的累积透射功率损耗与层间混响多径串扰。

核心物理公式:
1. 特征阻抗与导纳:
   Z_0 = a / (g * A) ~ 9647.4 s/m^2 (标称)
   Y_0 = 1 / Z_0 ~ 1.03655e-4 m^2/s
2. 反射系数与支路导纳映射:
   Gamma_j = - (Z_0 * Y_{b,j}) / (2 + Z_0 * Y_{b,j}) in (-1, 0]
   Y_{b,j} = - (2 * Y_0 * Gamma_j) / (1 + Gamma_j) >= 0
3. 上游双程累积透射损耗补偿:
   T_{1:j-1} = \prod_{k=1}^{j-1} (1 + Gamma_k)^2
   Gamma_j = Gamma_{raw, j} / (T_{1:j-1} + eps)
4. 解混重构与物理特征注入:
   stripped_features = OutProj([h_{dechoked}, Gamma, log(Y_b + eps), T_{cum}])
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerStrippingOutput(tuple):
    """
    可微逆散射层剥离算子前向输出结构:
    支持 4-元组解包 (h_stripped, gamma, admittance, attenuation_factors)、
    属性访问 (.gamma) 以及字典键访问 (['gamma'])。
    """

    def __new__(
        cls,
        h_stripped: torch.Tensor,
        gamma: torch.Tensor,
        admittance: torch.Tensor,
        attenuation_factors: torch.Tensor,
    ):
        return super().__new__(cls, (h_stripped, gamma, admittance, attenuation_factors))

    @property
    def h_stripped(self) -> torch.Tensor:
        return self[0]

    @property
    def gamma(self) -> torch.Tensor:
        return self[1]

    @property
    def admittance(self) -> torch.Tensor:
        return self[2]

    @property
    def attenuation_factors(self) -> torch.Tensor:
        return self[3]

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "h_stripped":
                return self[0]
            elif item == "gamma":
                return self[1]
            elif item == "admittance":
                return self[2]
            elif item == "attenuation_factors":
                return self[3]
            raise KeyError(f"Key '{item}' not found in LayerStrippingOutput")
        return super().__getitem__(item)


class DifferentiableLayerStripping(nn.Module):
    """
    波动方程可微逆散射层剥离算子 (Differentiable Layer-Stripping Layer, DIS Layer):
    沿井筒由跟端至趾端 (x_1 -> x_{N_c}) 递归解耦上游裂缝累积透射扼流与多程混响。
    显式输出物理反射率 Gamma_j in [-1, 0] 与支路导纳 Y_{b,j} >= 0。
    """

    def __init__(
        self,
        in_dim: int = 64,
        d_model: Optional[int] = None,
        max_nc: int = 6,
        g: float = 9.80665,
        A_pipe: float = 0.015328,  # D = 0.1397 m -> A = pi * D^2 / 4 = 0.015328 m^2
        a_ref: float = 1450.0,
        gamma_max: float = 0.95,
        gamma_min: float = 1e-4,
        eps_stab: float = 1e-4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model if d_model is not None else in_dim
        self.max_nc = max_nc
        self.g = g
        self.A_pipe = A_pipe
        self.a_ref = a_ref
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.eps_stab = eps_stab

        # 初始反射率估计器 (从局部到时窗特征提取未归一化视反射强度)
        self.refl_mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # 健康物理初始化: 初始视反射强度基线在 ~0.05，避免初始化时累积透射骤降导致钳位饱和
        nn.init.xavier_uniform_(self.refl_mlp[0].weight, gain=0.5)
        nn.init.constant_(self.refl_mlp[0].bias, 0.0)
        nn.init.xavier_uniform_(self.refl_mlp[2].weight, gain=0.1)
        nn.init.constant_(self.refl_mlp[2].bias, -2.5)

        # 多程混响预测与残差层剥离网络 (Schur 逆散射时空混响模型)
        # 输入: [前序混响状态 (in_dim), 前序反射率 (1)] -> 预测当前簇虚假多程回波分量
        self.reverb_mlp = nn.Sequential(
            nn.Linear(in_dim + 1, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )
        nn.init.xavier_uniform_(self.reverb_mlp[2].weight, gain=0.01)
        nn.init.constant_(self.reverb_mlp[2].bias, 0.0)

        # 剥离后物理特征升维融合投影层:
        # 输入: [解混波形特征 h_{dechoked} (in_dim), Gamma_j (1), log(Y_b) (1), T_{cum} (1)]
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim + 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )

    def forward(
        self,
        h_patches: torch.Tensor,                      # (B, M, in_dim) 局部波前特征
        positions: Optional[torch.Tensor] = None,      # (B, M) 簇位置 [m]
        wavespeed: Optional[torch.Tensor] = None,      # (B, 1) 或 (B,) 声速 [m/s] 或 cond (B, 3)
        mask: Optional[torch.Tensor] = None,           # (B, M) bool
    ) -> LayerStrippingOutput:
        """
        前向逐级层剥离递归计算:
        参数:
            h_patches: (B, M, in_dim) 到时对齐波前特征
            positions: (B, M) 簇绝对位置 [m]，可选
            wavespeed: (B, 1) 或 (B,) 或 (B, 3) 瞬变流声速或工况张量
            mask: (B, M) 真实裂缝掩码 (True 为有效激活簇)
        返回:
            LayerStrippingOutput 包含:
                - h_stripped: (B, M, d_model) 解耦重构增强表征
                - gamma: (B, M) 物理反射率序列 Gamma_j in [-1.0, 0.0]
                - admittance: (B, M) 支路导纳序列 Y_{b,j} >= 0.0
                - attenuation_factors: (B, M) 上游双程累积透射损耗因子 T_{1:j-1}
        """
        B, M, D = h_patches.shape
        device = h_patches.device

        # 解析声速并计算特征阻抗 Z_0 与特征导纳 Y_0
        if wavespeed is None:
            a = torch.full((B, 1), self.a_ref, device=device)
        elif wavespeed.ndim == 2 and wavespeed.shape[1] == 3:
            # 传入了 cond 字典 (B, 3)，cond[:, 1] 为归一化声速扰动
            a = wavespeed[:, 1:2] * 20.0 + self.a_ref
        elif wavespeed.ndim == 2:
            a = wavespeed[:, :1]
        elif wavespeed.ndim == 1:
            a = wavespeed.unsqueeze(1)
        else:
            a = torch.full((B, 1), self.a_ref, device=device)
        a = torch.clamp(a, min=1000.0, max=2000.0)

        # Z_0 = a / (g * A), Y_0 = 1 / Z_0
        Z_0 = a / (self.g * self.A_pipe)  # (B, 1)
        Y_0 = 1.0 / Z_0                   # (B, 1)

        if mask is None:
            mask = torch.ones((B, M), dtype=torch.bool, device=device)

        raw_refl_logits = self.refl_mlp(h_patches).squeeze(-1)  # (B, M)

        gamma_list = []
        yb_list = []
        t_cum_list = []
        h_dechoked_list = []

        # 累积透射功率初始化为 1.0 (第 1 簇未受上游透射扼流)
        T_cum_prev = torch.ones(B, device=device)  # (B,)
        # 混响记忆状态初始化为全 0 (第 1 簇无上游多程震荡)
        reverb_state = torch.zeros((B, self.in_dim), device=device)

        for j in range(M):
            m_j = mask[:, j].float()  # (B,)

            # 1. 混响多径预测与剥离 (Layer-Stripping De-reverberation)
            if j == 0:
                echo_j = torch.zeros((B, self.in_dim), device=device)
            else:
                prev_gamma = gamma_list[-1].unsqueeze(-1)  # (B, 1)
                echo_input = torch.cat([reverb_state, prev_gamma], dim=-1)
                echo_j = self.reverb_mlp(echo_input)

            # 从原始波形中减去上游伪周期混响分量
            h_j_clean = h_patches[:, j] - echo_j

            # 2. 基础单层反射强度估计 in [gamma_min, gamma_max]
            base_refl = torch.sigmoid(raw_refl_logits[:, j]) * (self.gamma_max - self.gamma_min) + self.gamma_min

            # 3. 逆散射层剥离：补偿上游所有裂缝的双程累积透射损耗 T_{1:j-1}
            # Gamma_j = - (base_refl / (T_{1:j-1} + eps))
            t_comp = T_cum_prev + self.eps_stab
            gamma_j_val = - (base_refl / t_comp)
            gamma_j = - torch.clamp(-gamma_j_val, min=self.gamma_min, max=self.gamma_max)  # 严格在 [-gamma_max, -gamma_min]

            # 4. 波前解扼流幅值归一化 (De-choked Representation)
            h_j_dechoked = h_j_clean / (torch.sqrt(t_comp).unsqueeze(-1))

            # 5. 单层透射率 T_j = 1 + Gamma_j in [1 - gamma_max, 1 - gamma_min]
            T_j = 1.0 + gamma_j  # 严格 > 0

            # 6. 物理支路导纳: Y_{b,j} = - 2 * Y_0 * Gamma_j / (1 + Gamma_j)
            Y_b_j = - (2.0 * Y_0.squeeze(-1) * gamma_j) / (T_j + 1e-8)

            # 7. 更新下游声波双程累积透射损耗因子 (若未激活簇则不施加衰减)
            effective_T_sq = (T_j ** 2) * m_j + 1.0 * (1.0 - m_j)
            T_cum_next = T_cum_prev * effective_T_sq

            # 8. 更新混响记忆状态
            reverb_state = (reverb_state + h_j_clean * gamma_j.unsqueeze(-1)) * m_j.unsqueeze(-1)

            gamma_list.append(gamma_j)
            yb_list.append(Y_b_j)
            t_cum_list.append(T_cum_prev)
            h_dechoked_list.append(h_j_dechoked)

            T_cum_prev = T_cum_next

        gamma = torch.stack(gamma_list, dim=1)                    # (B, M)
        yb = torch.stack(yb_list, dim=1)                          # (B, M)
        t_cum = torch.stack(t_cum_list, dim=1)                    # (B, M)
        h_dechoked = torch.stack(h_dechoked_list, dim=1)          # (B, M, in_dim)

        # 掩码清零: 未激活簇严格置 0
        gamma = torch.where(mask, gamma, torch.zeros_like(gamma))
        yb = torch.where(mask, yb, torch.zeros_like(yb))

        # 物理特征拼接与升维投影
        log_yb = torch.log(torch.clamp(yb, min=1e-8))
        log_yb = torch.where(mask, log_yb, torch.zeros_like(log_yb))

        phys_feat = torch.stack([gamma, log_yb, t_cum], dim=-1)   # (B, M, 3)
        combined = torch.cat([h_dechoked, phys_feat], dim=-1)     # (B, M, in_dim + 3)
        stripped_tokens = self.out_proj(combined)                 # (B, M, d_model)
        stripped_tokens = torch.where(mask.unsqueeze(-1), stripped_tokens, torch.zeros_like(stripped_tokens))

        return LayerStrippingOutput(
            h_stripped=stripped_tokens,
            gamma=gamma,
            admittance=yb,
            attenuation_factors=t_cum,
        )
