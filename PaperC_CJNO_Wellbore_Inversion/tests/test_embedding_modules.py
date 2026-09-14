# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.tests.test_embedding_modules

单元测试套件: 连续时空 Embedding 架构模块与 TG-DIS 模型变体
1. extract_raw_wave_packet 可微波包截取与坐标归一化测试;
2. PhysicsQueryEmbedding (方案一) 前向、声学偏置注意力与反向梯度测试;
3. FourierFeatureEmbedding (方案二) 多尺度频带、特征融合与掩码测试;
4. SincNetAcousticFilterbank (方案三) 奇数次驻波谐频滤波核、物理包络与数值稳定性测试;
5. TGDISEmbeddingModel 四大变体统一前向传播、物理单纯形守恒与多目标联合反向测试;
6. 边界异常处理测试 (非法 embedding_type, 全 0 掩码等)。
"""
from __future__ import annotations

import pytest
import math
import torch
import torch.nn.functional as F

from PaperC_CJNO_Wellbore_Inversion.src.modules.embedding_modules import (
    extract_raw_wave_packet,
    PhysicsQueryEmbedding,
    FourierFeatureEmbedding,
    SincNetAcousticFilterbank,
    BaselineResampleEmbedding,
)
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_embedding_variants import (
    TGDISEmbeddingModel,
)
from PaperC_CJNO_Wellbore_Inversion.src.losses import CompositeInversionLoss


def test_extract_raw_wave_packet_dimensions_and_continuity():
    """验证 extract_raw_wave_packet 抽取毫秒级局部波包的维度与坐标正确性"""
    B, C, N_raw, M = 3, 2, 60001, 4
    raw_wave = torch.randn(B, C, N_raw)
    tau = torch.tensor([
        [2.0, 3.5, 4.2, 5.8],
        [1.8, 2.9, 4.0, 5.1],
        [2.5, 3.2, 4.8, 6.0],
    ])

    packets, rel_offsets = extract_raw_wave_packet(
        raw_wave=raw_wave,
        tau=tau,
        t_pre=0.05,
        t_post=0.25,
        n_pts=301,
        t_total=60.0,
    )

    assert packets.shape == (B, M, C, 301)
    assert rel_offsets.shape == (301,)
    # 验证相对时延严格覆盖 [-50ms, +250ms]
    assert pytest.approx(rel_offsets[0].item(), abs=1e-4) == -0.05
    assert pytest.approx(rel_offsets[-1].item(), abs=1e-4) == 0.25
    # 验证中心索引 (第 50 点) 接近 0 时差
    assert pytest.approx(rel_offsets[50].item(), abs=1e-3) == 0.0
    assert not torch.isnan(packets).any()


def test_physics_query_embedding_forward_and_gradients():
    """验证 PhysicsQueryEmbedding 前向输出、声学衰减偏置及反向梯度"""
    B, M = 2, 6
    module = PhysicsQueryEmbedding(in_channels=2, d_model=64, max_nc=M)

    wave = torch.randn(B, 2, 4096)
    raw_wave = torch.randn(B, 2, 60001)
    positions = torch.tensor([
        [1000.0, 2000.0, 3000.0, 4000.0, 0.0, 0.0],
        [1500.0, 2500.0, 0.0, 0.0, 0.0, 0.0],
    ])
    mask = positions > 0
    cond = torch.tensor([[0.0, 0.5, 0.0], [0.1, -0.2, 0.3]])

    tokens, tau = module(
        wave=wave,
        raw_wave=raw_wave,
        positions=positions,
        cond=cond,
        mask=mask,
    )

    assert tokens.shape == (B, M, 64)
    assert tau.shape == (B, M)
    assert not torch.isnan(tokens).any()

    # 验证掩码位置全 0
    assert torch.all(tokens[0, 4:] == 0.0)
    assert torch.all(tokens[1, 2:] == 0.0)

    # 验证梯度反向传播
    loss = tokens.sum()
    loss.backward()
    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"参数 {name} 未收到梯度"
            assert not torch.isnan(param.grad).any(), f"参数 {name} 梯度含 NaN"


def test_fourier_feature_embedding_frequency_and_forward():
    """验证 FourierFeatureEmbedding 多尺度连续傅里叶字典与前向"""
    B, M = 2, 6
    module = FourierFeatureEmbedding(
        in_channels=2,
        d_model=64,
        max_nc=M,
        num_freqs=16,
        f_min=0.0725,
        f_max=500.0,
    )

    # 验证注册 buffer omegas 覆盖物理基频与高频
    assert module.omegas.shape == (16,)
    min_f = (module.omegas[0] / (2.0 * math.pi)).item()
    max_f = (module.omegas[-1] / (2.0 * math.pi)).item()
    assert pytest.approx(min_f, abs=1e-3) == 0.0725
    assert pytest.approx(max_f, abs=1e-2) == 500.0

    raw_wave = torch.randn(B, 2, 60001)
    positions = torch.tensor([
        [1200.0, 1800.0, 2400.0, 3000.0, 3600.0, 4200.0],
        [2000.0, 3000.0, 0.0, 0.0, 0.0, 0.0],
    ])
    mask = positions > 0
    cond = torch.zeros(B, 3)

    tokens, tau = module(
        wave=raw_wave,
        raw_wave=raw_wave,
        positions=positions,
        cond=cond,
        mask=mask,
    )

    assert tokens.shape == (B, M, 64)
    assert not torch.isnan(tokens).any()
    assert torch.all(tokens[1, 2:] == 0.0)

    # 反向传播测试
    loss = tokens.pow(2).sum()
    loss.backward()
    assert not torch.isnan(module.fourier_proj[0].weight.grad).any()


def test_sinc_acoustic_filterbank_physics_and_stability():
    """验证 SincNetAcousticFilterbank 滤波器核物理初始化与数值稳定性 (无 NaN/Inf)"""
    B, M = 2, 6
    module = SincNetAcousticFilterbank(
        in_channels=2,
        d_model=64,
        max_nc=M,
        n_filters=32,
        kernel_size=65,
        fs=1000.0,
    )

    # 1. 验证动态滤波器核生成正常无 NaN
    filters = module.get_filter_kernels()
    assert filters.shape == (32, 1, 65)
    assert not torch.isnan(filters).any()
    assert not torch.isinf(filters).any()

    # 2. 验证截止频率严格满足 0 < f_low < f_high < 500
    f_low = F.softplus(module.raw_f_low) + 0.01
    band = F.softplus(module.raw_band) + 0.5
    f_high = f_low + band
    assert torch.all(f_low > 0.0)
    assert torch.all(f_high > f_low)
    assert torch.all(f_high <= 500.0)

    # 3. 前向与梯度
    raw_wave = torch.randn(B, 2, 60001)
    positions = torch.tensor([
        [1000.0, 2000.0, 3000.0, 0.0, 0.0, 0.0],
        [2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3000.0],
    ])
    mask = positions > 0
    cond = torch.zeros(B, 3)

    tokens, tau = module(
        wave=raw_wave,
        raw_wave=raw_wave,
        positions=positions,
        cond=cond,
        mask=mask,
    )

    assert tokens.shape == (B, M, 64)
    assert not torch.isnan(tokens).any()

    loss = tokens.sum()
    loss.backward()
    assert module.raw_f_low.grad is not None
    assert not torch.isnan(module.raw_f_low.grad).any()


@pytest.mark.parametrize("embedding_type", [
    "baseline_resample",
    "physics_query",
    "fourier_feature",
    "sinc_filterbank",
])
def test_tg_dis_embedding_model_full_workflow(embedding_type: str):
    """测试 TGDISEmbeddingModel 四大变体的全套端到端前向、单纯形守恒与多目标反向"""
    B, M = 3, 6
    model = TGDISEmbeddingModel(
        embedding_type=embedding_type,
        d_model=64,
        max_nc=M,
        n_grid=500,
        L=5000.0,
    )
    model.eval()

    positions = torch.tensor([
        [2000.0, 2050.0, 2100.0, 2150.0, 0.0, 0.0],
        [3000.0, 3020.0, 3040.0, 3060.0, 3080.0, 3100.0],
        [4000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    mask = positions > 0
    norm_positions = positions / 5000.0

    batch = {
        "wave": torch.randn(B, 2, 4096),
        "raw_wave": torch.randn(B, 2, 60001),
        "cepstrum": torch.randn(B, 1, 1024),
        "cond": torch.zeros(B, 3),
        "positions": positions,
        "norm_positions": norm_positions,
        "mask": mask,
        "alpha": torch.tensor([
            [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
            [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        "cf": torch.ones(B, M) * 0.01,
        "log_cf": torch.zeros(B, M),
        "m_alpha_grid": torch.zeros(B, 500),
    }

    with torch.no_grad():
        out = model(batch)

    # 1. 验证字典键
    for k in ["alpha", "cf", "log_cf", "p_exist", "delta_x", "gamma", "admittance", "m_alpha_grid", "tau"]:
        assert k in out, f"模型输出缺失关键字段 {k}"

    # 2. 验证单纯形物理守恒: max |sum(alpha) - 1.0| < 1e-6
    alpha_sum = out["alpha"].sum(dim=-1)
    for b in range(B):
        assert pytest.approx(alpha_sum[b].item(), abs=1e-5) == 1.0

    # 3. 验证掩码位置全 0
    assert torch.all(out["alpha"][0, 4:] == 0.0)
    assert torch.all(out["alpha"][2, 1:] == 0.0)

    # 4. 验证多目标复合损失梯度反向传播正常
    model.train()
    criterion = CompositeInversionLoss()
    train_out = model(batch)
    loss_dict = criterion(train_out, batch)
    loss = loss_dict["loss"]
    loss.backward()

    # 验证无 NaN 梯度
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"参数 {name} 在 {embedding_type} 下反向产生 NaN 梯度"


def test_tg_dis_embedding_model_invalid_type():
    """测试非法 embedding_type 抛出 ValueError"""
    with pytest.raises(ValueError, match="不支持的 embedding_type"):
        _ = TGDISEmbeddingModel(embedding_type="invalid_type")


@pytest.mark.parametrize("module_cls", [
    PhysicsQueryEmbedding,
    FourierFeatureEmbedding,
    SincNetAcousticFilterbank,
    BaselineResampleEmbedding,
])
def test_embedding_modules_cond_none_robustness(module_cls):
    """测试四大 Embedding 模块在 cond=None 时的鲁棒性 (不崩溃且默认基准工况)"""
    B, M = 2, 6
    module = module_cls(d_model=64, max_nc=M)
    wave = torch.randn(B, 2, 4096)
    raw_wave = torch.randn(B, 2, 60001)
    positions = torch.tensor([
        [1000.0, 2000.0, 3000.0, 0.0, 0.0, 0.0],
        [1500.0, 2500.0, 3500.0, 4500.0, 0.0, 0.0],
    ])
    mask = positions > 0

    tokens, tau = module(
        wave=wave,
        raw_wave=raw_wave,
        positions=positions,
        cond=None,  # 显式不提供 cond
        mask=mask,
    )
    assert tokens.shape == (B, M, 64)
    assert tau.shape == (B, M)
    assert not torch.isnan(tokens).any()


def test_tg_dis_model_raw_wave_alone():
    """测试 TGDISEmbeddingModel 在仅传入 raw_wave 未传入 wave 时的前向正常"""
    B, M = 2, 6
    model = TGDISEmbeddingModel(embedding_type="physics_query", d_model=64, max_nc=M)
    raw_wave = torch.randn(B, 2, 60001)
    cepstrum = torch.randn(B, 1, 1024)
    positions = torch.tensor([
        [1200.0, 2400.0, 3600.0, 0.0, 0.0, 0.0],
        [1800.0, 2800.0, 0.0, 0.0, 0.0, 0.0],
    ])
    cond = torch.zeros(B, 3)

    out = model(
        raw_wave=raw_wave,
        wave=None,  # 未显式提供 4096 点 wave
        cepstrum=cepstrum,
        cond=cond,
        positions=positions,
    )
    assert "alpha" in out
    assert out["alpha"].shape == (B, M)
    assert pytest.approx(out["alpha"].sum(dim=-1)[0].item(), abs=1e-5) == 1.0


def test_extract_raw_wave_packet_2d_and_dtype():
    """测试 extract_raw_wave_packet 处理 2D 信号与 float64 精度的一致性"""
    B, N_raw, M = 2, 60001, 3
    # 1. 2D 输入 (B, N_raw)
    raw_2d = torch.randn(B, N_raw, dtype=torch.float32)
    tau = torch.tensor([[2.0, 3.0, 4.0], [1.5, 2.5, 3.5]])
    sampled_2d, offsets = extract_raw_wave_packet(raw_2d, tau, n_pts=301)
    assert sampled_2d.shape == (B, M, 1, 301)

    # 2. float64 输入
    raw_64 = torch.randn(B, 2, N_raw, dtype=torch.float64)
    tau_64 = torch.tensor([[2.0, 3.0, 4.0], [1.5, 2.5, 3.5]], dtype=torch.float64)
    sampled_64, offsets_64 = extract_raw_wave_packet(raw_64, tau_64, n_pts=301)
    assert sampled_64.dtype == torch.float64
    assert offsets_64.dtype == torch.float64


def test_extreme_cluster_spacing_resolution():
    """测试极限近距离多簇 (间距 2m, 时差 2.76ms) 下波包特征与声学注意力可分性"""
    module = PhysicsQueryEmbedding(d_model=64, max_nc=4)
    raw_wave = torch.randn(1, 2, 60001)
    # 间距 2m (时差 2.76ms), 5m (时差 6.90ms)
    positions = torch.tensor([[2000.0, 2002.0, 2007.0, 2015.0]])
    cond = torch.zeros(1, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)

    tokens, tau = module(
        wave=raw_wave,
        raw_wave=raw_wave,
        positions=positions,
        cond=cond,
        mask=mask,
    )
    assert tokens.shape == (1, 4, 64)
    # 验证相邻 2m 簇的特征向量不发生坍缩为同一向量
    tok_diff = torch.norm(tokens[0, 0] - tokens[0, 1]).item()
    assert tok_diff > 1e-4, f"近距离簇特征坍缩: {tok_diff}"

    # 验证反向梯度正常
    loss = tokens.sum()
    loss.backward()
    assert not torch.isnan(module.raw_lambda.grad)


def test_missing_positions_and_norm_positions_error():
    """测试 positions 与 norm_positions 均缺失时抛出 ValueError"""
    module = PhysicsQueryEmbedding()
    raw_wave = torch.randn(1, 2, 60001)
    with pytest.raises(ValueError, match="必须提供 positions 或 norm_positions"):
        _ = module(wave=raw_wave, raw_wave=raw_wave, positions=None, norm_positions=None)

