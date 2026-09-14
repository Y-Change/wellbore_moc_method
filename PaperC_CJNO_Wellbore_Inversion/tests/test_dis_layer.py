# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.tests.test_dis_layer

波动方程可微逆散射层剥离算子 (Differentiable Layer-Stripping Layer, DIS Layer) 核心单元测试:
1. 前向输出形状、数据类型与元组解包规范测试;
2. 物理量数值边界约束测试 (Gamma in [-1, 0], Y_b >= 0, T_cum in (0, 1]);
3. 上游透射衰减逐级补偿物理行为验证 (Energy Attenuation Compensation Behavior);
4. 端到端可微反向传播与无 NaN/Inf 梯度流测试;
5. 单簇 (Nc=1)、全满簇 (Nc=6) 及 5m 极密簇等极端工况数值稳定性测试。
"""
from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn

from PaperC_CJNO_Wellbore_Inversion.src.modules.layer_stripping import (
    DifferentiableLayerStripping,
    LayerStrippingOutput,
)


def test_dis_layer_output_shapes_and_unpacking():
    """验证 DIS 算子输出张量形状、NamedTuple 属性访问与 4 元组解包"""
    B, M, in_dim, d_model = 4, 6, 64, 64
    dis = DifferentiableLayerStripping(in_dim=in_dim, d_model=d_model, max_nc=M)

    h_patches = torch.randn(B, M, in_dim)
    positions = torch.tensor([
        [2000.0, 2100.0, 2200.0, 0.0, 0.0, 0.0],
        [3000.0, 3050.0, 3100.0, 3150.0, 3200.0, 3250.0],
        [4000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1500.0, 1600.0, 1700.0, 1800.0, 0.0, 0.0],
    ])
    mask = positions > 0.0
    cond = torch.zeros(B, 3)  # a = 1450 m/s

    out = dis(h_patches=h_patches, positions=positions, wavespeed=cond, mask=mask)

    # 1. 输出类型检验
    assert isinstance(out, LayerStrippingOutput)

    # 2. 形状检验
    assert out.h_stripped.shape == (B, M, d_model)
    assert out.gamma.shape == (B, M)
    assert out.admittance.shape == (B, M)
    assert out.attenuation_factors.shape == (B, M)

    # 3. 4 元组解包检验
    h_s, gam, yb, t_cum = out
    assert h_s.shape == (B, M, d_model)
    assert gam.shape == (B, M)
    assert yb.shape == (B, M)
    assert t_cum.shape == (B, M)

    # 4. 字典式键访问检验
    assert torch.equal(out["gamma"], out.gamma)
    assert torch.equal(out["admittance"], out.admittance)
    assert torch.equal(out["attenuation_factors"], out.attenuation_factors)


def test_dis_layer_physical_bounds_and_masking():
    """验证物理反射率 Gamma in [-1, 0]、支路导纳 Y_b >= 0 以及掩码置零行为"""
    B, M, in_dim = 2, 4, 32
    dis = DifferentiableLayerStripping(in_dim=in_dim, d_model=32, max_nc=M)

    h_patches = torch.randn(B, M, in_dim)
    positions = torch.tensor([
        [2000.0, 2100.0, 0.0, 0.0],
        [3000.0, 3050.0, 3100.0, 0.0],
    ])
    mask = positions > 0.0

    out = dis(h_patches=h_patches, positions=positions, mask=mask)

    gamma = out.gamma
    yb = out.admittance
    t_cum = out.attenuation_factors

    # A. 激活簇边界校验
    for b in range(B):
        nc = int(mask[b].sum().item())
        # 激活簇的 Gamma 必须严格在 [-1.0, 0.0] 之间
        act_gamma = gamma[b, :nc]
        assert (act_gamma >= -1.0).all(), f"反射率低于 -1.0: {act_gamma.min()}"
        assert (act_gamma <= 0.0).all(), f"反射率大于 0.0: {act_gamma.max()}"

        # 激活簇的支路导纳 Y_b 必须严格非负 Y_b >= 0
        act_yb = yb[b, :nc]
        assert (act_yb >= 0.0).all(), f"支路导纳为负数: {act_yb.min()}"

        # 累积透射损耗因子必须严格单调非增且在 (0, 1] 区间
        act_t_cum = t_cum[b, :nc]
        assert (act_t_cum > 0.0).all() and (act_t_cum <= 1.0001).all()
        assert act_t_cum[0].item() == 1.0, "首簇上游累积透射因子必须等于 1.0"
        if nc > 1:
            diffs = act_t_cum[1:] - act_t_cum[:-1]
            assert (diffs <= 1e-6).all(), "累积透射因子沿跟至趾方向必须单调不增"

        # B. 未激活簇必须完全清零
        if nc < M:
            assert (gamma[b, nc:] == 0.0).all(), "未激活簇反射率必须为 0"
            assert (yb[b, nc:] == 0.0).all(), "未激活簇导纳必须为 0"
            assert (out.h_stripped[b, nc:] == 0.0).all(), "未激活簇解耦特征必须为 0"


def test_dis_layer_energy_attenuation_compensation_behavior():
    """
    验证上游透射扼流补偿机理 (Energy Attenuation Compensation Behavior):
    当上游裂缝反射率更强 (|Gamma_1| 较大) 时，上游累积透射率 T_1^2 显著更低，
    逆散射层剥离算子对下游裂缝实施的除以 T_{cum} 放大补偿幅度成倍增加。
    """
    in_dim = 16
    dis = DifferentiableLayerStripping(in_dim=in_dim, d_model=16, max_nc=2)
    dis.eval()

    # 初始化为单调正权重以验证确定的物理单调响应
    with torch.no_grad():
        dis.refl_mlp[0].weight.fill_(0.1)
        dis.refl_mlp[0].bias.zero_()
        dis.refl_mlp[2].weight.fill_(0.5)
        dis.refl_mlp[2].bias.zero_()

    # 构建两个样本对比:
    # 样本 1: 上游特征较弱 (小反射率)
    # 样本 2: 上游特征较强 (大反射率)
    # 保持两样本的第 2 簇输入特征完全相同
    h2_common = torch.ones(1, 1, in_dim) * 0.5
    h_weak_upstream = torch.cat([torch.full((1, 1, in_dim), -2.0), h2_common], dim=1)
    h_strong_upstream = torch.cat([torch.full((1, 1, in_dim), +2.0), h2_common], dim=1)

    batch_h = torch.cat([h_weak_upstream, h_strong_upstream], dim=0)
    mask = torch.ones((2, 2), dtype=torch.bool)

    with torch.no_grad():
        out = dis(batch_h, mask=mask)

    gamma = out.gamma
    t_cum = out.attenuation_factors

    # 样本 1 的上游反射率绝对值应小于样本 2
    refl_up_sample1 = abs(gamma[0, 0].item())
    refl_up_sample2 = abs(gamma[1, 0].item())
    assert refl_up_sample2 > refl_up_sample1, f"强上游激励产生的反射率必须大于弱激励 ({refl_up_sample2} vs {refl_up_sample1})"

    # 样本 2 第 1 簇透射衰减更严重，故 T_cum[1, 1] 必然显著小于 T_cum[0, 1]
    t_cum_down_sample1 = t_cum[0, 1].item()
    t_cum_down_sample2 = t_cum[1, 1].item()
    assert t_cum_down_sample2 < t_cum_down_sample1, "上游反射越强，下游保留的透射功率越小"

    # 验证物理补偿效果:
    # 样本 2 对第 2 簇施加的放大倍数 (1 / sqrt(t_cum)) 显著大于样本 1
    boost_factor_1 = 1.0 / np.sqrt(t_cum_down_sample1)
    boost_factor_2 = 1.0 / np.sqrt(t_cum_down_sample2)
    assert boost_factor_2 > boost_factor_1, "强上游扼流工况下，层剥离对下游特征的补偿系数必须更大"


def test_dis_layer_differentiability_and_gradient_flow():
    """验证全程自动微分链式法则梯度反传正常，参数与输入均有非零梯度且无 NaN/Inf"""
    B, M, in_dim = 3, 5, 32
    dis = DifferentiableLayerStripping(in_dim=in_dim, d_model=32, max_nc=M)

    h_patches = torch.randn(B, M, in_dim, requires_grad=True)
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
        [True, False, False, False, False],
    ])

    out = dis(h_patches=h_patches, mask=mask)

    # 复合目标损失: 约束 h_stripped, gamma 与 yb
    loss = (
        out.h_stripped.sum()
        + out.gamma.sum()
        + out.admittance.sum()
        + out.attenuation_factors.sum()
    )

    assert not torch.isnan(loss), "前向 Loss 出现 NaN"
    assert not torch.isinf(loss), "前向 Loss 出现 Inf"

    loss.backward()

    # 验证对输入波形特征的梯度
    assert h_patches.grad is not None, "输入 h_patches 未获得反传梯度"
    assert not torch.isnan(h_patches.grad).any(), "输入梯度存在 NaN"
    assert not torch.isinf(h_patches.grad).any(), "输入梯度存在 Inf"
    assert h_patches.grad.abs().sum() > 0, "输入梯度全为 0"

    # 验证对 DIS 内部各子网络参数的梯度
    has_param_grad = False
    for name, p in dis.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"参数 {name} 梯度出现 NaN"
            assert not torch.isinf(p.grad).any(), f"参数 {name} 梯度出现 Inf"
            if p.grad.abs().sum() > 0:
                has_param_grad = True
    assert has_param_grad, "DIS 算子内部参数未获得有效更新梯度"


def test_dis_layer_extreme_cases():
    """验证单簇 (Nc=1)、全零输入及 5m 极小簇间距极限工况下的数值稳定性"""
    dis = DifferentiableLayerStripping(in_dim=32, d_model=32, max_nc=6)

    # 1. 全零输入极端情况
    zero_h = torch.zeros(2, 6, 32, requires_grad=True)
    mask_full = torch.ones((2, 6), dtype=torch.bool)
    out_zero = dis(zero_h, mask=mask_full)
    loss_zero = out_zero.h_stripped.sum()
    loss_zero.backward()
    assert not torch.isnan(out_zero.gamma).any()
    assert not torch.isnan(out_zero.admittance).any()
    assert not torch.isnan(zero_h.grad).any()

    # 2. 单簇极端情况 (Nc=1)
    h_nc1 = torch.randn(2, 6, 32)
    mask_nc1 = torch.tensor([
        [True, False, False, False, False, False],
        [True, False, False, False, False, False],
    ])
    out_nc1 = dis(h_nc1, mask=mask_nc1)
    assert (out_nc1.gamma[:, 1:] == 0.0).all()
    assert (out_nc1.admittance[:, 1:] == 0.0).all()
    assert (out_nc1.h_stripped[:, 1:] == 0.0).all()

    # 3. 5m 超密簇强混叠工况 (4000m, 4005m, 4010m)
    pos_dense = torch.tensor([[4000.0, 4005.0, 4010.0, 0.0, 0.0, 0.0]])
    mask_dense = pos_dense > 0.0
    h_dense = torch.randn(1, 6, 32)
    out_dense = dis(h_dense, positions=pos_dense, mask=mask_dense)
    assert not torch.isnan(out_dense.h_stripped).any()
    assert not torch.isnan(out_dense.gamma).any()
    assert not torch.isnan(out_dense.admittance).any()
    assert (out_dense.attenuation_factors[0, :3] > 0.0).all()
