# -*- coding: utf-8 -*-
"""1-case forward/backward + checkpoint restore."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjno import PilotPrototype
from dataset import PilotDataset, collate_batch
from paths import CKPT_DIR, CONFIG_DIR
from train import batch_to_torch, set_seed


def main():
    cfg = yaml.safe_load((CONFIG_DIR / "train_forward_pilot.yaml").read_text(encoding="utf-8"))
    device = torch.device("cpu")
    set_seed(0)
    ds = PilotDataset("train", window_periods=2.0, wellhead_decim=8, node_max_len=32).subset(1, seed=0)
    norm = PilotDataset("train", window_periods=2.0, wellhead_decim=8, node_max_len=32).compute_train_norm(8)
    model = PilotPrototype(mode="soft", d=32, fno_d=32, fno_modes=8).to(device)
    batch = batch_to_torch(collate_batch([ds[0]], norm), device)
    out = model(batch, apply_hard_times=0)
    # wellhead proj may be zero-initialized; pull a gradient through the encoder cond
    loss = ((out["p_pert_hat"] - batch["p_pert"]) * batch["wh_mask"]).pow(2).mean()
    loss = loss + out["cond"].abs().mean()
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert out["p_pert_hat"].shape == batch["p_pert"].shape
    assert sum(grads) > 0
    ckpt = CKPT_DIR / "_restore_test.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "norm": norm, "mode": "soft"}, ckpt)
    m2 = PilotPrototype(mode="soft", d=32, fno_d=32, fno_modes=8)
    m2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    m2.eval()
    with torch.no_grad():
        y2 = m2(batch, apply_hard_times=0)["p_pert_hat"]
    assert torch.allclose(out["p_pert_hat"].detach(), y2, atol=1e-5)
    ckpt.unlink(missing_ok=True)
    print("PASS checkpoint restore + grad flow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
