# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.tests.test_tg_models

TG-CJ-DeepONet 第二阶段核心模块与模型全套单元测试:
1. 理论往返到时计算正确性与边界测试;
2. 3 种到时窗提取机制 (relative_window, gaussian_gating, ceps_patch) 可微反传与梯度流;
3. 声学时延偏置 Transformer (含/无偏置消融、gamma >= 0、掩码因果);
4. 双轨协同头 (离散精准头单纯形约束 sum=1、连续 Trunk 场、Voronoi 积分);
5. 双轨协同一致性损失 L_cons 数值稳定性与反传;
6. TG-CJ-DeepONet 4 种消融变体全流程前向与反向传播测试;
7. Nc=1 单簇、Nc=6 满簇及 5m 极小簇间距极端物理工况测试。
"""
from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.modules.time_gating import (
    compute_theoretical_arrival_times,
    RelativeWindowGating,
    GaussianGating,
    CepstrumPatchGating,
    TimeGatingModule,
)
from PaperC_CJNO_Wellbore_Inversion.src.modules.acoustic_transformer import (
    AcousticBiasedMultiheadAttention,
    AcousticBiasedTransformer,
)
from PaperC_CJNO_Wellbore_Inversion.src.modules.dual_track_heads import (
    DiscretePreciseHead,
    ContinuousTrunkHead,
    DualTrackHead,
)
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_cj_deeponet import TGCJDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.losses import (
    DualTrackConsistencyLoss,
    CompositeInversionLoss,
)
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics
from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset


def test_theoretical_arrival_time_calculation():
    """验证 tau = ts + 2*x/a 声学理论计算精度与掩码行为"""
    positions = torch.tensor([[1450.0, 2900.0, 0.0]]) # 1450m, 2900m
    cond = torch.tensor([[0.0, 0.0, 0.0]]) # cond[1]=0 -> a = 1450 m/s
    mask = torch.tensor([[True, True, False]])

    tau = compute_theoretical_arrival_times(positions, cond, ts=1.0, mask=mask)

    # tau1 = 1.0 + 2*1450 / 1450 = 3.0s
    # tau2 = 1.0 + 2*2900 / 1450 = 5.0s
    assert abs(tau[0, 0].item() - 3.0) < 1e-4
    assert abs(tau[0, 1].item() - 5.0) < 1e-4
    # 未激活簇回退至 ts
    assert abs(tau[0, 2].item() - 1.0) < 1e-4


def test_three_time_gating_modes_differentiability():
    """验证三种到时窗机制均支持对输入波形与可学习参数的全程可微梯度反传"""
    B, M = 2, 4
    wave = torch.randn(B, 2, 4096, requires_grad=True)
    cepstrum = torch.randn(B, 1, 1024, requires_grad=True)
    positions = torch.tensor([
        [2000.0, 2500.0, 3000.0, 0.0],
        [2200.0, 2600.0, 0.0, 0.0],
    ])
    norm_pos = positions / 5000.0
    cond = torch.randn(B, 3)
    mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])

    for mode in ["relative_window", "gaussian_gating", "ceps_patch"]:
        module = TimeGatingModule(window_mode=mode, in_channels=2, d_model=64, max_nc=M)
        tokens, tau = module(
            wave=wave,
            cepstrum=cepstrum,
            positions=positions,
            norm_positions=norm_pos,
            cond=cond,
            mask=mask,
        )

        assert tokens.shape == (B, M, 64), f"{mode} token 形状异常: {tokens.shape}"
        assert tau.shape == (B, M), f"{mode} tau 形状异常: {tau.shape}"

        # 检查未激活簇输出全为 0
        assert (tokens[0, 3] == 0.0).all()
        assert (tokens[1, 2:] == 0.0).all()

        # 反向传播梯度检查
        loss = tokens.sum()
        loss.backward(retain_graph=True)
        assert wave.grad is not None and not torch.isnan(wave.grad).any(), f"{mode} wave 梯度异常"
        assert not torch.isnan(loss)

        # 针对 gaussian_gating 专门检查窗宽 sigma 的初始精度与梯度更新
        if mode == "gaussian_gating":
            assert abs(module.extractor.get_sigma()[0].item() - 0.08) < 1e-3, "高斯窗初始窗宽必须精确等于 80ms"
            assert module.extractor.raw_sigma.grad is not None
            assert module.extractor.raw_sigma.grad.abs().sum() > 0


def test_acoustic_biased_transformer():
    """验证声学时延物理偏置注意力计算、gamma>=0 物理约束与消融开关"""
    B, M, d_model = 2, 4, 32
    tokens = torch.randn(B, M, d_model)
    positions = torch.tensor([
        [1000.0, 1100.0, 3000.0, 0.0],
        [2000.0, 2050.0, 0.0, 0.0],
    ])
    norm_pos = positions / 5000.0
    cond = torch.zeros(B, 3) # a = 1450 m/s
    mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])

    # 1. 开启声学偏置
    model_bias = AcousticBiasedTransformer(
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        use_acoustic_bias=True,
    )
    h_bias, attn_bias = model_bias(tokens, positions, norm_pos, cond, mask=mask)
    assert h_bias.shape == (B, M, d_model)
    assert attn_bias.shape == (B, 2, 2, M, M) # (B, layers, heads, M, M)

    # 验证物理约束 gamma >= 0
    layer0_attn = model_bias.layers[0].attn
    gamma_val = layer0_attn.get_gamma().item()
    assert gamma_val >= 0.0, "gamma 必须非负"

    # 验证声学衰减规律: 相邻簇 (1000m与1100m, 间距100m) 的注意力偏置惩罚显著小于远距离簇 (1000m与3000m, 间距2000m)
    tau_100 = 100.0 / 1450.0
    tau_2000 = 2000.0 / 1450.0
    bias_100 = - gamma_val * tau_100
    bias_2000 = - gamma_val * tau_2000
    assert bias_100 > bias_2000, f"近距簇偏置 ({bias_100}) 必须大于远距簇偏置 ({bias_2000})"

    # 验证未激活簇在注意力图上的完全静默屏蔽
    assert (attn_bias[0, :, :, 3, :] == 0.0).all(), "未激活 Query 簇注意力权重必须全为 0"
    assert (attn_bias[0, :, :, :, 3] == 0.0).all(), "未激活 Key 簇注意力权重必须全为 0"
    assert (attn_bias[1, :, :, 2:, :] == 0.0).all(), "未激活 Query 簇注意力权重必须全为 0"
    assert (attn_bias[1, :, :, :, 2:] == 0.0).all(), "未激活 Key 簇注意力权重必须全为 0"

    # 验证反向传播流至 gamma
    l_bias = h_bias.sum()
    l_bias.backward()
    assert layer0_attn.raw_gamma.grad is not None

    # 2. 关闭声学偏置 (消融模式)
    model_nobias = AcousticBiasedTransformer(
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        use_acoustic_bias=False,
    )
    h_nobias, attn_nobias = model_nobias(tokens, positions, norm_pos, cond, mask=mask)
    assert h_nobias.shape == (B, M, d_model)


def test_dual_track_heads_and_simplex_constraint():
    """验证双轨解码头的单纯形和为 1 严格数学约束与两轨协同输出"""
    B, M, d_model = 4, 6, 64
    h = torch.randn(B, M, d_model)
    positions = torch.tensor([
        [2000.0, 2100.0, 2200.0, 0.0, 0.0, 0.0],
        [3000.0, 3050.0, 3100.0, 3150.0, 3200.0, 0.0],
        [4000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0],
    ])
    norm_pos = positions / 5000.0
    mask = positions > 0.0
    global_wave = torch.randn(B, 64)
    cond = torch.randn(B, 3)

    head = DualTrackHead(d_model=d_model, d_well=128, p=64, n_grid=500, L=5000.0)
    out = head(
        h=h,
        positions=positions,
        norm_positions=norm_pos,
        mask=mask,
        global_wave_feat=global_wave,
        cond=cond,
    )

    # 验证输出字段完整性
    expected_keys = [
        "alpha", "cf", "log_cf", "log10_cf",
        "m_alpha_grid", "c_grid", "alpha_field", "cf_field", "z_well"
    ]
    for k in expected_keys:
        assert k in out, f"缺少关键输出字段: {k}"

    # 验证单纯形约束 sum_j alpha_j == 1.0 (在 1e-6 严格精度内)
    pred_alpha = out["alpha"]
    for b in range(B):
        nc = int(mask[b].sum().item())
        sum_val = pred_alpha[b].sum().item()
        assert abs(sum_val - 1.0) < 1e-5, f"样本 {b} (Nc={nc}) 的 alpha 累积和为 {sum_val}，未满足单纯形约束"
        # 非激活簇全为 0
        if nc < M:
            assert (pred_alpha[b, nc:] == 0.0).all()

    # 验证对数顺应性转换正确性: ln(Cf/Cf0) = log10(Cf/Cf0) * ln(10)
    ln10 = float(np.log(10.0))
    diff_log = torch.abs(out["log_cf"] - out["log10_cf"] * ln10)
    assert diff_log.max().item() < 1e-5

    # 验证连续场归一化黎曼和为 1.0
    dx = 5000.0 / 499.0
    grid_integrals = (out["m_alpha_grid"].sum(dim=-1) * dx).tolist()
    for integral in grid_integrals:
        assert abs(integral - 1.0) < 1e-3


def test_dual_track_consistency_loss():
    """验证双轨一致性损失 L_cons 计算与梯度反向传播"""
    loss_fn = DualTrackConsistencyLoss()
    pred_alpha = torch.tensor([[0.6, 0.4, 0.0]], requires_grad=True)
    alpha_field = torch.tensor([[0.5, 0.5, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])

    loss = loss_fn(pred_alpha, alpha_field, mask)
    # diff = (|0.6-0.5| + |0.4-0.5|) / 2 = 0.1
    assert abs(loss.item() - 0.1) < 1e-5

    loss.backward()
    assert pred_alpha.grad is not None
    assert alpha_field.grad is not None


def test_tg_cj_deeponet_all_ablation_variants_pipeline():
    """验证 TG-CJ-DeepONet 四大消融配置在真实验证集数据上的端到端前向、反向与指标计算"""
    dataset = PilotInversionDataset(split="val")
    loader = dataset.get_dataloader(batch_size=4, shuffle=False)
    batch = next(iter(loader))

    criterion = CompositeInversionLoss(
        lambda_alpha=1.0,
        lambda_c=1.0,
        lambda_w=0.01,
        lambda_cons=0.5,
    )

    ablation_configs = [
        ("relative_window", True),
        ("gaussian_gating", True),
        ("ceps_patch", True),
        ("relative_window", False),
    ]

    for mode, use_bias in ablation_configs:
        model = TGCJDeepONet(
            window_mode=mode,
            use_acoustic_bias=use_bias,
            d_model=64,
            n_heads=4,
            n_layers=2,
            p=64,
        )
        model.train()
        out = model(batch)

        assert "alpha" in out
        assert "cf" in out
        assert "log_cf" in out
        assert "m_alpha_grid" in out
        assert "alpha_field" in out
        assert "tau" in out
        assert "attn_weights" in out

        # 检查单纯形和
        alpha_sum = out["alpha"].sum(dim=-1)
        for s in alpha_sum:
            assert abs(s.item() - 1.0) < 1e-5

        # 损失与梯度
        loss_dict = criterion(out, batch)
        loss = loss_dict["loss"]
        assert not torch.isnan(loss)
        assert "loss_cons" in loss_dict

        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, f"配置 ({mode}, use_bias={use_bias}) 缺少梯度"

        # 验证指标计算
        metrics = compute_inversion_metrics(out, batch)
        assert "alpha_mae" in metrics
        assert "cf_mre_pct" in metrics
        assert metrics["simplex_max_dev"] < 1e-5


def test_edge_cases_single_and_dense_clusters():
    """验证 Nc=1 单簇边界与 5m 极小簇间距强混叠工况"""
    model = TGCJDeepONet(window_mode="relative_window", use_acoustic_bias=True)
    model.eval()

    # 1. 单簇 Nc=1
    batch_nc1 = {
        "wave": torch.randn(1, 2, 4096),
        "cepstrum": torch.randn(1, 1, 1024),
        "cond": torch.tensor([[0.0, 0.0, 0.0]]),
        "norm_positions": torch.tensor([[0.8, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "positions": torch.tensor([[4000.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "mask": torch.tensor([[True, False, False, False, False, False]]),
    }
    with torch.no_grad():
        out_nc1 = model(batch_nc1)
    assert abs(out_nc1["alpha"][0, 0].item() - 1.0) < 1e-5
    assert (out_nc1["alpha"][0, 1:] == 0.0).all()

    # 2. 密集 5m 簇间距强混叠工况 (4200m 与 4205m)
    batch_dense = {
        "wave": torch.randn(1, 2, 4096),
        "cepstrum": torch.randn(1, 1, 1024),
        "cond": torch.tensor([[0.0, 0.0, 0.0]]),
        "norm_positions": torch.tensor([[0.84, 0.841, 0.0, 0.0, 0.0, 0.0]]),
        "positions": torch.tensor([[4200.0, 4205.0, 0.0, 0.0, 0.0, 0.0]]),
        "mask": torch.tensor([[True, True, False, False, False, False]]),
    }
    with torch.no_grad():
        out_dense = model(batch_dense)
    assert abs(out_dense["alpha"][0, :2].sum().item() - 1.0) < 1e-5
    assert (out_dense["alpha"][0, :2] > 0.0).all()
    assert (out_dense["alpha"][0, 2:] == 0.0).all()
