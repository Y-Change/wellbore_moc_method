# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.tests.test_tg_dis_model

TG-DIS-DeepONet 第三阶段旗舰网络系统级单元测试:
1. 全流程前向传播张量形状与关键字段完整性测试;
2. 物理单纯形守恒严格性测试 (max |sum(alpha) - 1.0| < 1e-6);
3. 裂缝起裂存在性分类头 (p_exist in [0, 1]) 与亚米级位置细化头 (delta_x in [-10, 10] m) 行为测试;
4. 复合物理损失反向传播与全网络梯度健康度测试;
5. 二分图最近邻匹配 F1-score 指标准确性测试 (容差 +/- 10m);
6. 单簇 (Nc=1) 与 5m 极密多簇极端物理工况测试。
"""
from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import TGDISDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.losses import CompositeInversionLoss
from PaperC_CJNO_Wellbore_Inversion.src.metrics import (
    compute_detection_f1_score,
    compute_inversion_metrics,
    DetectionF1Result,
)


def test_tg_dis_deeponet_forward_output_dict():
    """验证 TG-DIS-DeepONet 前向计算输出字典包含全部规范要求字段且维度正确"""
    B, M = 3, 6
    model = TGDISDeepONet(
        window_mode="relative_window",
        use_acoustic_bias=True,
        d_model=64,
        n_heads=4,
        n_layers=2,
        p=64,
        max_nc=M,
        n_grid=500,
        L=5000.0,
    )
    model.eval()

    positions = torch.tensor([
        [2000.0, 2100.0, 2200.0, 0.0, 0.0, 0.0],
        [3000.0, 3050.0, 3100.0, 3150.0, 3200.0, 3250.0],
        [4000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    mask = positions > 0.0
    norm_positions = positions / 5000.0

    batch = {
        "wave": torch.randn(B, 2, 4096),
        "cepstrum": torch.randn(B, 1, 1024),
        "cond": torch.zeros(B, 3),
        "positions": positions,
        "norm_positions": norm_positions,
        "mask": mask,
    }

    with torch.no_grad():
        out = model(batch)

    # 验证全部核心验收字段存在
    required_keys = [
        "alpha", "cf", "log_cf", "log10_cf",
        "p_exist", "delta_x", "gamma", "admittance",
        "attenuation_factors", "m_alpha_grid", "c_grid",
        "alpha_field", "tau", "attn_weights", "z_well"
    ]
    for key in required_keys:
        assert key in out, f"输出字典缺失必要字段: '{key}'"

    # 验证各字段形状
    assert out["alpha"].shape == (B, M)
    assert out["cf"].shape == (B, M)
    assert out["log_cf"].shape == (B, M)
    assert out["p_exist"].shape == (B, M)
    assert out["delta_x"].shape == (B, M)
    assert out["gamma"].shape == (B, M)
    assert out["admittance"].shape == (B, M)
    assert out["attenuation_factors"].shape == (B, M)
    assert out["m_alpha_grid"].shape == (B, 500)
    assert out["alpha_field"].shape == (B, M)
    assert out["attn_weights"].shape == (B, 2, 4, M, M)  # (B, n_layers, n_heads, M, M)


def test_tg_dis_deeponet_simplex_conservation():
    """验证流量分配 alpha 严格满足单纯形物理守恒: max |sum(alpha) - 1.0| < 1e-6"""
    B, M = 4, 6
    model = TGDISDeepONet(d_model=64, max_nc=M)
    model.eval()

    positions = torch.tensor([
        [1500.0, 1600.0, 1700.0, 1800.0, 0.0, 0.0],
        [2500.0, 2550.0, 0.0, 0.0, 0.0, 0.0],
        [3500.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4100.0, 4120.0, 4140.0, 4160.0, 4180.0, 4200.0],
    ])
    mask = positions > 0.0

    batch = {
        "wave": torch.randn(B, 2, 4096),
        "cepstrum": torch.randn(B, 1, 1024),
        "cond": torch.zeros(B, 3),
        "positions": positions,
        "mask": mask,
    }

    with torch.no_grad():
        out = model(batch)

    pred_alpha = out["alpha"]

    for b in range(B):
        nc = int(mask[b].sum().item())
        sum_val = pred_alpha[b].sum().item()
        dev = abs(sum_val - 1.0)
        assert dev < 1e-6, f"样本 {b} (Nc={nc}) 违背单纯形物理守恒: 偏差={dev:.2e} >= 1e-6"
        assert (pred_alpha[b, :nc] >= 0.0).all(), "激活簇流量分配份额必须非负"
        if nc < M:
            assert (pred_alpha[b, nc:] == 0.0).all(), "未激活簇流量分配份额必须严格为 0"


def test_tg_dis_deeponet_heads_bounds_and_masking():
    """验证各预测头的物理合法值域与掩码置零行为"""
    B, M = 3, 6
    model = TGDISDeepONet(d_model=64, max_nc=M, delta_x_max=10.0)
    model.eval()

    positions = torch.tensor([
        [2000.0, 2050.0, 0.0, 0.0, 0.0, 0.0],
        [3000.0, 3100.0, 3200.0, 0.0, 0.0, 0.0],
        [4000.0, 4010.0, 4020.0, 4030.0, 0.0, 0.0],
    ])
    mask = positions > 0.0

    batch = {
        "wave": torch.randn(B, 2, 4096),
        "cepstrum": torch.randn(B, 1, 1024),
        "cond": torch.zeros(B, 3),
        "positions": positions,
        "mask": mask,
    }

    with torch.no_grad():
        out = model(batch)

    p_exist = out["p_exist"]
    delta_x = out["delta_x"]
    gamma = out["gamma"]
    admittance = out["admittance"]

    for b in range(B):
        nc = int(mask[b].sum().item())

        # 1. 存在性概率 in [0, 1]
        act_exist = p_exist[b, :nc]
        assert (act_exist >= 0.0).all() and (act_exist <= 1.0).all()

        # 2. 亚米级位置偏差 in [-10, 10] m
        act_dx = delta_x[b, :nc]
        assert (act_dx >= -10.0).all() and (act_dx <= 10.0).all()

        # 3. 反射率 in [-1, 0] 与导纳 >= 0
        act_gamma = gamma[b, :nc]
        act_adm = admittance[b, :nc]
        assert (act_gamma >= -1.0).all() and (act_gamma <= 0.0).all()
        assert (act_adm >= 0.0).all()

        # 4. 未激活簇严格全为 0
        if nc < M:
            assert (p_exist[b, nc:] == 0.0).all()
            assert (delta_x[b, nc:] == 0.0).all()
            assert (gamma[b, nc:] == 0.0).all()
            assert (admittance[b, nc:] == 0.0).all()


def test_tg_dis_deeponet_backward_loss_and_gradients():
    """验证结合分类损失的复合物理损失前向计算与全网络端到端反向传播梯度流"""
    B, M = 2, 4
    model = TGDISDeepONet(d_model=64, max_nc=M)
    model.train()

    criterion = CompositeInversionLoss(
        lambda_alpha=1.0,
        lambda_c=1.0,
        lambda_w=0.01,
        lambda_cons=0.5,
        lambda_exist=0.5,
        lambda_pos=0.5,
    )

    positions = torch.tensor([
        [2000.0, 2100.0, 2200.0, 0.0],
        [3000.0, 3050.0, 0.0, 0.0],
    ])
    true_positions = torch.tensor([
        [2002.0, 2098.0, 2201.0, 0.0],
        [3001.0, 3048.0, 0.0, 0.0],
    ])
    mask = positions > 0.0

    batch = {
        "wave": torch.randn(B, 2, 4096),
        "cepstrum": torch.randn(B, 1, 1024),
        "cond": torch.zeros(B, 3),
        "positions": positions,
        "true_positions": true_positions,
        "mask": mask,
        "alpha": torch.tensor([
            [0.3, 0.4, 0.3, 0.0],
            [0.6, 0.4, 0.0, 0.0],
        ]),
        "log_cf": torch.zeros(B, M),
    }

    out = model(batch)
    loss_dict = criterion(out, batch)

    loss = loss_dict["loss"]
    assert not torch.isnan(loss), "复合损失为 NaN"
    assert not torch.isinf(loss), "复合损失为 Inf"
    assert "loss_exist" in loss_dict, "损失字典缺失分类损失 loss_exist"
    assert "loss_pos" in loss_dict, "损失字典缺失位置损失 loss_pos"

    loss.backward()

    # 验证关键子模块均获得梯度
    assert model.time_gating.extractor.conv[0].weight.grad is not None
    assert model.layer_stripping.refl_mlp[0].weight.grad is not None
    assert model.transformer.layers[0].attn.w_out.weight.grad is not None
    assert model.alpha_mlp[0].weight.grad is not None
    assert model.cf_mlp[0].weight.grad is not None
    assert model.exist_mlp[0].weight.grad is not None
    assert model.pos_mlp[0].weight.grad is not None


def test_tg_dis_deeponet_forward_with_kwargs():
    """验证支持直接以关键字参数调用 forward(wave=..., cepstrum=...)"""
    model = TGDISDeepONet(d_model=32, max_nc=4)
    model.eval()

    positions = torch.tensor([[2000.0, 2100.0, 0.0, 0.0]])
    mask = positions > 0.0

    with torch.no_grad():
        out = model(
            wave=torch.randn(1, 2, 4096),
            cepstrum=torch.randn(1, 1, 1024),
            cond=torch.zeros(1, 3),
            positions=positions,
            mask=mask,
        )

    assert "alpha" in out
    assert abs(out["alpha"][0, :2].sum().item() - 1.0) < 1e-6


def test_detection_f1_score_bipartite_matching():
    """验证二分图贪心距离匹配 F1-Score 计算的精确度与容差边界"""
    # 场景 1: 完美匹配 (2 簇，偏移均在 10m 以内)
    true_pos = np.array([[2000.0, 2100.0, 0.0]])
    true_mask = np.array([[True, True, False]])
    pred_pos = np.array([[2005.0, 2095.0, 0.0]])  # 偏移 +5m, -5m <= 10m
    pred_exist = np.array([[0.9, 0.8, 0.1]])

    res1 = compute_detection_f1_score(pred_pos, pred_exist, true_pos, true_mask, tolerance_m=10.0)
    assert isinstance(res1, DetectionF1Result)
    assert res1.f1_score == 1.0
    assert res1.precision == 1.0
    assert res1.recall == 1.0
    assert res1.tp == 2
    assert res1.fp == 0
    assert res1.fn == 0

    # 验证三元组解包
    p, r, f1 = res1
    assert p == 1.0 and r == 1.0 and f1 == 1.0

    # 场景 2: 偏移超出 10m 容差 (偏移 15m -> 不匹配)
    pred_pos_far = np.array([[2015.0, 2115.0, 0.0]])
    res2 = compute_detection_f1_score(pred_pos_far, pred_exist, true_pos, true_mask, tolerance_m=10.0)
    assert res2.f1_score == 0.0
    assert res2.tp == 0
    assert res2.fp == 2
    assert res2.fn == 2

    # 场景 3: 包含虚警与漏报 (1 个 TP, 1 个 FP, 1 个 FN)
    # 真实: 2000m, 3000m; 预测: 2005m (匹配2000m), 4000m (虚警); 3000m 漏报
    t_pos3 = np.array([[2000.0, 3000.0]])
    t_mask3 = np.array([[True, True]])
    p_pos3 = np.array([[2005.0, 4000.0]])
    p_ext3 = np.array([[0.95, 0.90]])

    res3 = compute_detection_f1_score(p_pos3, p_ext3, t_pos3, t_mask3, tolerance_m=10.0)
    assert res3.tp == 1
    assert res3.fp == 1
    assert res3.fn == 1
    assert abs(res3.precision - 0.5) < 1e-5
    assert abs(res3.recall - 0.5) < 1e-5
    assert abs(res3.f1_score - 0.5) < 1e-5


def test_edge_cases_single_and_dense_5m():
    """验证 Nc=1 单簇边界与 5m 极小簇间距极端混叠工况"""
    model = TGDISDeepONet(d_model=32, max_nc=6)
    model.eval()

    # 1. 单簇 Nc=1
    batch_nc1 = {
        "wave": torch.randn(1, 2, 4096),
        "cepstrum": torch.randn(1, 1, 1024),
        "cond": torch.zeros(1, 3),
        "positions": torch.tensor([[3500.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "mask": torch.tensor([[True, False, False, False, False, False]]),
    }
    with torch.no_grad():
        out_nc1 = model(batch_nc1)
    assert abs(out_nc1["alpha"][0, 0].item() - 1.0) < 1e-6
    assert (out_nc1["alpha"][0, 1:] == 0.0).all()

    # 2. 5m 极密多簇 (4500m, 4505m, 4510m)
    batch_dense = {
        "wave": torch.randn(1, 2, 4096),
        "cepstrum": torch.randn(1, 1, 1024),
        "cond": torch.zeros(1, 3),
        "positions": torch.tensor([[4500.0, 4505.0, 4510.0, 0.0, 0.0, 0.0]]),
        "mask": torch.tensor([[True, True, True, False, False, False]]),
    }
    with torch.no_grad():
        out_dense = model(batch_dense)
    assert abs(out_dense["alpha"][0, :3].sum().item() - 1.0) < 1e-6
    assert (out_dense["alpha"][0, :3] > 0.0).all()
    assert (out_dense["alpha"][0, 3:] == 0.0).all()
    assert (out_dense["attenuation_factors"][0, :3] > 0.0).all()
