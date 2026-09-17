# -*- coding: utf-8 -*-
"""AlphaInversionDataset / CJ-AlphaNet 单测。"""
from __future__ import annotations

import inspect

import h5py
import numpy as np
import torch

from PaperC_CJNO_Wellbore_Inversion.src.alpha_dataset import (
    DEFAULT_H5_PATH,
    DEFAULT_NPZ_PATH,
    AlphaInversionDataset,
    USES_DESIGN_WAVESPEED,
    normalize_a_hat,
)
from PaperC_CJNO_Wellbore_Inversion.src.alpha_losses import AlphaCompositeLoss
from PaperC_CJNO_Wellbore_Inversion.src.models.cj_alphanet import CJAlphaNet
from moc_simulate.v2.batch.torch_dataset import split_dataset_indices


def test_y_act_matches_alpha_threshold_npz():
    z = np.load(DEFAULT_NPZ_PATH)
    mask = z["mask_design"] > 0
    y = z["y_active"].astype(bool)
    pred = (z["alpha"] >= 0.04) & mask
    assert np.array_equal(y, pred)
    assert np.all(z["y_active"][~mask] == 0)


def test_dataset_does_not_use_design_wavespeed():
    assert USES_DESIGN_WAVESPEED is False
    src = inspect.getsource(AlphaInversionDataset)
    assert "labels/wavespeed" not in src
    assert "a_hat" in src


def test_dataset_batch_shapes_and_cond_is_a_hat():
    ds = AlphaInversionDataset(split="test", seed=42)
    assert len(ds) == 100
    loader = ds.get_dataloader(batch_size=4, shuffle=False)
    batch = next(iter(loader))
    B = 4
    T = batch["wave"].shape[-1]
    assert batch["wave"].shape[0] == B and batch["wave"].shape[1] == 2
    assert T >= 50000
    assert batch["cepstrum_1d_at_xj"].shape == (B, 6)
    assert batch["cepstrum_2d_at_xj"].shape[0] == B
    assert batch["cepstrum_2d_at_xj"].shape[-1] == 6
    assert batch["cepstrum_2d_at_xj"].shape[1] >= 20
    assert batch["mask_design"].shape == (B, 6)
    assert batch["alpha"].shape == (B, 6)
    assert batch["y_act"].shape == (B, 6)
    assert batch["logY"].shape == (B, 6)
    assert batch["m_alpha_grid"].shape == (B, 500)
    assert batch["cond"].shape == (B, 1)
    assert batch["a_hat"].shape == (B,)
    # cond 必须来自 â 而非设计波速
    cond = batch["cond"].numpy().reshape(-1)
    a_hat = batch["a_hat"].numpy().reshape(-1)
    expect = normalize_a_hat(a_hat).reshape(-1)
    assert np.allclose(cond, expect, atol=1e-5)
    sids = batch["sample_id"].numpy().astype(int)
    with h5py.File(DEFAULT_H5_PATH, "r") as f:
        ws = np.asarray(f["labels/wavespeed"][:])[sids]
    # 不允许 cond 等于设计波速编码
    ws_enc = (ws - 1400.0) / 200.0
    assert np.mean(np.abs(cond - ws_enc)) > 1e-4 or np.mean(np.abs(a_hat - ws)) > 1.0


def test_split_800_100_100():
    n = 1000
    sp = split_dataset_indices(n, (0.8, 0.1, 0.1), 42)
    assert len(sp["train"]) == 800
    assert len(sp["val"]) == 100
    assert len(sp["test"]) == 100


def test_alphanet_softmax_and_no_cf_head():
    torch.manual_seed(0)
    B, M, T = 2, 6, 4096
    mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    pos = torch.tensor(
        [[4510.0, 4550.0, 4600.0, 0.0, 0.0, 0.0], [4700.0, 4780.0, 0.0, 0.0, 0.0, 0.0]]
    )
    a_hat = torch.tensor([1430.0, 1460.0])
    y0 = torch.tensor([1.05e-4, 1.02e-4])
    batch = {
        "wave": torch.randn(B, 2, T),
        "cepstrum_1d_at_xj": torch.randn(B, M),
        "cepstrum_2d_at_xj": torch.randn(B, 41, M),
        "positions": pos,
        "norm_positions": pos / 5000.0,
        "mask_design": mask,
        "a_hat": a_hat,
        "Y0": y0,
        "cond": ((a_hat - 1400.0) / 200.0).unsqueeze(-1),
        "alpha": torch.tensor([[0.5, 0.45, 0.05, 0, 0, 0], [0.7, 0.3, 0, 0, 0, 0]]),
        "y_act": torch.tensor([[1.0, 1.0, 1.0, 0, 0, 0], [1.0, 1.0, 0, 0, 0, 0]]),
        "logY": torch.zeros(B, M),
        "m_alpha_grid": torch.rand(B, 500),
    }
    model = CJAlphaNet(use_cep2d=True)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert "cf" not in out and "pred_cf" not in out
    assert "pred_m_alpha" in out and "pred_active" in out and "pred_logY" in out
    assert "gamma" in out
    for i in range(B):
        s = float(out["pred_alpha"][i, mask[i]].sum())
        assert abs(s - 1.0) < 1e-5
        assert torch.all(out["pred_alpha"][i, ~mask[i]] == 0)
    Yh = out["Y_hat"]
    g = out["gamma"]
    g_closed = -Yh / (2.0 * y0.view(B, 1) + Yh + 1e-16)
    assert torch.allclose(g[mask], g_closed[mask], atol=1e-5)
    model.train()
    out2 = model(batch)
    loss2 = AlphaCompositeLoss()(out2, batch)
    assert torch.isfinite(loss2["loss"])
    loss2["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_ablation_no2d_forward():
    B, M, T = 1, 6, 2048
    mask = torch.zeros(B, M, dtype=torch.bool)
    mask[0, :2] = True
    pos = torch.zeros(B, M)
    pos[0, 0] = 4500.0
    pos[0, 1] = 4600.0
    a_hat = torch.tensor([1450.0])
    batch = {
        "wave": torch.randn(B, 2, T),
        "cepstrum_1d_at_xj": torch.zeros(B, M),
        "cepstrum_2d_at_xj": torch.zeros(B, 41, M),
        "positions": pos,
        "norm_positions": pos / 5000.0,
        "mask_design": mask,
        "a_hat": a_hat,
        "Y0": torch.tensor([1e-4]),
        "cond": torch.zeros(B, 1),
    }
    m = CJAlphaNet(use_cep2d=False)
    m.eval()
    with torch.no_grad():
        out = m(batch)
    assert out["pred_alpha"].shape == (1, 6)
    assert abs(float(out["pred_alpha"][0, mask[0]].sum()) - 1.0) < 1e-5
