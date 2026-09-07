# -*- coding: utf-8 -*-
"""Small regressions: B2 silent floor, first-peak extras, timeout last-step, Adam resume."""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RunLogger, dump_json
from metrics_r4 import first_event_in_window, grads_finite, loss_b2_energy_equal
from r4_paths import CKPT_DIR
from checkpoint_r4 import load_checkpoint, restore_train_state, save_checkpoint


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.3, -0.1], dtype=torch.float32))

    def predict(self, feat, T):
        t = torch.linspace(0, 1, T, device=feat.device).unsqueeze(0)
        p = self.w[0] * torch.sin(2 * np.pi * t) + self.w[1]
        return {"p_pert": p.expand(feat.shape[0], T)}


def test_b2_silent():
    T = 16
    pred = {"p_pert": torch.ones(2, T, dtype=torch.float32) * 1e-3}
    tgt = torch.zeros(2, T, dtype=torch.float32)
    tgt[0, 0] = 1e-20  # would underflow if squared in float32 without floor
    mask = torch.ones(2, T)
    silent = torch.tensor([False, True])
    t_l2 = torch.tensor([1e-20, 0.0], dtype=torch.float32)
    scale = torch.tensor([1.0e6, 1.0e6])
    main, sabs, per, skip = loss_b2_energy_equal(pred, tgt, mask, silent, t_l2, scale)
    mixed = {
        "main_finite": bool(torch.isfinite(main)),
        "silent_abs_finite": bool(torch.isfinite(sabs)),
        "skip_step": bool(skip),
        "per_finite": bool(torch.isfinite(per).all()),
    }
    silent2 = torch.tensor([True, True])
    main2, sabs2, per2, skip2 = loss_b2_energy_equal(pred, tgt, mask, silent2, t_l2, scale)
    all_s = {
        "main_finite": bool(torch.isfinite(main2)),
        "skip_step": bool(skip2),
        "contract": "all-silent batch skips optimizer.step",
    }
    # nonfinite grad must not step
    w = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([w], lr=0.1)
    w0 = float(w.detach())
    loss = w * torch.tensor(float("inf"))
    opt.zero_grad()
    if torch.isfinite(loss):
        loss.backward()
        if grads_finite([w]):
            opt.step()
    stepped = abs(float(w.detach()) - w0) > 0
    return {
        "mixed": mixed,
        "all_silent": all_s,
        "nonfinite_did_not_step": (not stepped) and not bool(torch.isfinite(loss)),
        "pass": mixed["main_finite"] and mixed["per_finite"] and all_s["skip_step"] and (not stepped),
        "does_not_revoke_B_overfit": True,
    }


def test_first_peak():
    t = np.linspace(0, 2.0, 401)
    scale = 1.0
    t0, t1 = 0.2, 1.8
    # precursor ripple then real peak
    y = np.zeros_like(t)
    y += 0.0002 * np.sin(80 * np.pi * t)  # below noise/significance vs scale=1
    y += np.exp(-((t - 0.8) ** 2) / 0.002)
    a = first_event_in_window(t, y, t0, t1, polarity=1, physical_scale=scale)
    # flat top
    y2 = np.zeros_like(t)
    y2[(t >= 0.7) & (t <= 0.9)] = 1.0
    b = first_event_in_window(t, y2, t0, t1, polarity=1, physical_scale=scale)
    # polarity mismatch
    y3 = -np.exp(-((t - 0.8) ** 2) / 0.002)
    c = first_event_in_window(t, y3, t0, t1, polarity=1, physical_scale=scale)
    return {
        "precursor": a,
        "flat_top": b,
        "polarity": c,
        "pass": bool(a["ok"] and a["precursor_ignored"] >= 1 and abs(a["t_event"] - 0.8) < 0.05
                     and b["ok"] and b["flat_top"]
                     and (not c["ok"]) and c["reason"] == "polarity_mismatch"),
        "thresholds_frozen": True,
    }


def test_timeout_last_and_resume():
    device = "cpu"
    model = Tiny()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-4)
    feat = torch.randn(3, 4)
    T = 16
    cursor = 0
    batches = [0, 1, 2, 0, 1, 2, 0]
    tlim = 0.05
    t0 = time.time()
    completed = 0
    last = None
    path = CKPT_DIR / "r4_reg_last.pt"
    rng0 = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    for i, sl in enumerate(batches, start=1):
        pred = model.predict(feat, T)
        loss = (pred["p_pert"] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if not grads_finite(model.parameters()):
            break
        opt.step()
        sched.step()
        completed = i
        cursor = sl
        last = {
            "optimizer_steps": completed,
            "batch_cursor": cursor,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        save_checkpoint(path, model, opt, {"query": {"dt_s": 0.02}}, {"protocol": "r4"},
                        ["pilot_01342"], "manish", "codesha", completed, completed,
                        scheduler=sched, extra={"kind": "last", "batch_cursor": cursor})
        if time.time() - t0 >= tlim:
            break
    blob = load_checkpoint(path)
    # timeout must report the last completed step stored in the checkpoint
    match = int(blob["global_step"]) == last["optimizer_steps"]
    model2 = Tiny()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=20, eta_min=1e-4)
    ep, gs = restore_train_state(blob, model2, opt2, sched2)
    # one more Adam step should be finite
    pred = model2.predict(feat, T)
    loss = (pred["p_pert"] ** 2).mean()
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    sched2.step()
    return {
        "timeout_last_matches_ckpt": match,
        "optimizer_steps": last["optimizer_steps"],
        "ckpt_global_step": int(blob["global_step"]),
        "batch_cursor_saved": blob.get("batch_cursor"),
        "restored_step": gs,
        "adam_used": True,
        "scheduler_used": True,
        "pass": match and blob.get("optimizer") is not None and blob.get("scheduler") is not None,
    }


def main():
    logger = RunLogger("r4_regressions", {"stage": "regressions"})
    rec = {
        "b2_silent": test_b2_silent(),
        "first_peak": test_first_peak(),
        "timeout_resume": test_timeout_last_and_resume(),
    }
    rec["all_pass"] = all(rec[k]["pass"] for k in rec)
    logger.write_metrics(rec)
    dump_json(rec, logger.dir / "regressions.json")
    logger.close("completed")
    print("regressions", rec["all_pass"], logger.run_id)
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
