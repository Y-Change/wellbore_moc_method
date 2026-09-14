# -*- coding: utf-8 -*-
"""
Unit tests for PaperC models, loss functions, metrics, and pipeline.
"""
import numpy as np
import pytest
import torch
from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.src.models.resnet1d import ResNet1D
from PaperC_CJNO_Wellbore_Inversion.src.models.fno1d import FNO1D
from PaperC_CJNO_Wellbore_Inversion.src.models.deeponet import VanillaDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.cj_cep_deeponet import CJCepDeepONet, VoronoiPooling1D
from PaperC_CJNO_Wellbore_Inversion.src.losses import (
    CompositeInversionLoss,
    SimplexKLDivergenceLoss,
    LogHuberComplianceLoss,
    Wasserstein1DLoss,
)
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics


def test_dataset_and_models_pipeline():
    dataset = PilotInversionDataset(split="val")
    assert len(dataset) == 100

    loader = dataset.get_dataloader(batch_size=8, shuffle=False)
    batch = next(iter(loader))

    assert batch["wave"].shape == (8, 2, 4096)
    assert batch["cepstrum"].shape == (8, 1, 1024)
    assert batch["cond"].shape == (8, 3)
    assert batch["norm_positions"].shape == (8, 6)
    assert batch["mask"].shape == (8, 6)
    assert batch["alpha"].shape == (8, 6)
    assert batch["cf"].shape == (8, 6)
    assert batch["log_cf"].shape == (8, 6)
    assert batch["m_alpha_grid"].shape == (8, 500)

    criterion = CompositeInversionLoss(lambda_w=0.01)

    models = {
        "resnet": ResNet1D(),
        "fno": FNO1D(),
        "deeponet": VanillaDeepONet(),
        "cj_cep_deeponet": CJCepDeepONet(),
    }

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(batch)

        assert "alpha" in out
        assert "cf" in out
        assert "log_cf" in out
        assert "m_alpha_grid" in out

        assert out["alpha"].shape == (8, 6)
        assert out["cf"].shape == (8, 6)
        assert out["log_cf"].shape == (8, 6)
        assert out["m_alpha_grid"].shape == (8, 500)

        # 检查单纯形和为 1
        alpha_sum = out["alpha"].sum(dim=-1)
        for i in range(8):
            nc = batch["mask"][i].sum().item()
            if nc > 0:
                assert abs(alpha_sum[i].item() - 1.0) < 1e-4

        # 训练前向与梯度测试
        model.train()
        out_train = model(batch)
        loss_dict = criterion(out_train, batch)
        loss = loss_dict["loss"]
        assert not torch.isnan(loss)
        assert loss.item() > 0

        loss.backward()
        # 验证梯度存在
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, f"Model {name} has no gradient"

        # 验证评估指标
        metrics = compute_inversion_metrics(out, batch)
        assert "alpha_mae" in metrics
        assert "cf_mre_pct" in metrics
        assert "w1_mean_m" in metrics
        assert "simplex_max_dev" in metrics
        assert metrics["simplex_max_dev"] < 1e-4


def test_single_sample_and_1d_metrics():
    """验证 batch_size=1 单样本及 1D 数组输入时指标函数不抛出 AxisError"""
    pred_1d = {
        "alpha": np.array([0.4, 0.6, 0.0, 0.0, 0.0, 0.0]),
        "cf": np.array([0.01, 0.02, 0.0, 0.0, 0.0, 0.0]),
        "m_alpha_grid": np.ones(500) / 500.0,
    }
    targ_1d = {
        "alpha": np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        "cf": np.array([0.012, 0.018, 0.0, 0.0, 0.0, 0.0]),
        "mask": np.array([True, True, False, False, False, False]),
        "positions": np.array([4200.0, 4220.0, 0.0, 0.0, 0.0, 0.0]),
        "m_alpha_grid": np.ones(500) / 500.0,
    }
    res = compute_inversion_metrics(pred_1d, targ_1d)
    assert abs(res["alpha_mae"] - 0.1) < 1e-5
    assert res["simplex_max_dev"] < 1e-6
    assert "w1_field_mean_m" in res


def test_edge_cases_single_cluster_and_max_clusters():
    """验证 Nc=1 单簇边界与 Nc=6 满簇边界条件下的守恒性与指标"""
    model = CJCepDeepONet()
    model.eval()

    # Case A: Nc = 1 (单簇)
    batch_nc1 = {
        "wave": torch.randn(1, 2, 4096),
        "cepstrum": torch.randn(1, 1, 1024),
        "cond": torch.randn(1, 3),
        "norm_positions": torch.tensor([[0.84, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "positions": torch.tensor([[4200.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "mask": torch.tensor([[True, False, False, False, False, False]]),
        "alpha": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "log_cf": torch.zeros(1, 6),
        "m_alpha_grid": torch.ones(1, 500) / 500.0,
    }
    with torch.no_grad():
        out_nc1 = model(batch_nc1)
    assert abs(out_nc1["alpha"][0, 0].item() - 1.0) < 1e-5
    assert (out_nc1["alpha"][0, 1:] == 0.0).all()

    # Case B: Nc = 6 (满簇)
    batch_nc6 = {
        "wave": torch.randn(1, 2, 4096),
        "cepstrum": torch.randn(1, 1, 1024),
        "cond": torch.randn(1, 3),
        "norm_positions": torch.tensor([[0.80, 0.82, 0.84, 0.86, 0.88, 0.90]]),
        "positions": torch.tensor([[4000.0, 4100.0, 4200.0, 4300.0, 4400.0, 4500.0]]),
        "mask": torch.tensor([[True, True, True, True, True, True]]),
        "alpha": torch.full((1, 6), 1.0 / 6.0),
        "log_cf": torch.zeros(1, 6),
        "m_alpha_grid": torch.ones(1, 500) / 500.0,
    }
    with torch.no_grad():
        out_nc6 = model(batch_nc6)
    assert abs(out_nc6["alpha"].sum().item() - 1.0) < 1e-5
    assert (out_nc6["alpha"] > 0.0).all()


def test_voronoi_pooling_closely_spaced_clusters():
    """验证 5m 极小簇间距下 VoronoiPooling1D 不发生空网格丢失或单簇偏置"""
    pool = VoronoiPooling1D(L=5000.0, n_grid=500, n_quad=5)
    # 2 簇间距仅 5.0m
    pos_m = torch.tensor([[4200.0, 4205.0, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False, False, False]])
    m_alpha = torch.ones(1, 500)
    c_grid = torch.full((1, 500), 0.01)

    pa, pcf, plcf = pool(m_alpha, c_grid, pos_m, mask)

    # 验证非零且和为 1
    assert pa[0, 0].item() > 0.0
    assert pa[0, 1].item() > 0.0
    assert abs(pa[0].sum().item() - 1.0) < 1e-5
    # 验证对称性: 均匀密度下相邻 5m 两簇在自身区间对称，分配比为 1:1
    assert abs(pa[0, 0].item() - pa[0, 1].item()) < 0.05
    # 顺应性正常
    assert abs(pcf[0, 0].item() - 0.01) < 1e-4
    assert abs(pcf[0, 1].item() - 0.01) < 1e-4


def test_losses_edge_cases():
    """验证损失函数的数值稳定性与梯度反传"""
    # 1. Simplex cross entropy with zero target
    loss_alpha = SimplexKLDivergenceLoss()
    pa = torch.tensor([[0.5, 0.5, 0.0]], requires_grad=True)
    ta = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False]])
    l_a = loss_alpha(pa, ta, mask)
    l_a.backward()
    assert not torch.isnan(l_a)
    assert pa.grad is not None and not torch.isnan(pa.grad).any()

    # 2. LogHuber loss
    loss_c = LogHuberComplianceLoss(delta=0.2)
    p_log = torch.tensor([[0.0, 2.0]], requires_grad=True)
    t_log = torch.tensor([[0.0, 0.0]])
    mask_c = torch.tensor([[True, True]])
    l_c = loss_c(p_log, t_log, mask_c)
    l_c.backward()
    assert not torch.isnan(l_c)

    # 3. Wasserstein 1D
    loss_w = Wasserstein1DLoss(L=5000.0, normalize_by_L=False)
    p_field = torch.ones(2, 500, requires_grad=True)
    t_field = torch.ones(2, 500)
    l_w = loss_w(p_field, t_field)
    l_w.backward()
    assert not torch.isnan(l_w)
    assert abs(l_w.item()) < 1e-3
