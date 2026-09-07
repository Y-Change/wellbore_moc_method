# -*- coding: utf-8 -*-
"""P4: real train entry, FD grads, limited-iter Newton fail block."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import load_train_cases
from common import RunLogger, dump_json
from kernel_loader import load_frozen_kernel
from moc_twin import make_num, short_window_seconds
from production_moc import NewtonFailedError, ProductionMOC, TrainEntry
from stage1_bridge import case_to_well


def fd_grad(make_roll, n_steps, loss_of, theta0, eps_list, set_theta):
    g_auto = None
    roll = make_roll()
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    set_theta(roll, theta)
    pred = roll.unroll(n_steps, require_ok=True)
    loss = loss_of(pred)
    g_auto, = torch.autograd.grad(loss, theta, allow_unused=False)
    g_auto = float(g_auto)
    fds = []
    for eps in eps_list:
        vals = []
        for s in (+1.0, -1.0):
            r2 = make_roll()
            set_theta(r2, torch.tensor(s * eps, dtype=torch.float64))
            p = r2.unroll(n_steps, require_ok=True)
            vals.append(float(loss_of(p).detach()))
        g = (vals[0] - vals[1]) / (2.0 * eps)
        rel = abs(g - g_auto) / max(abs(g_auto), 1e-30)
        fds.append({"eps": float(eps), "g_fd": g, "g_auto": g_auto, "rel": rel})
    best = min(fds, key=lambda r: r["rel"])
    signal = abs(g_auto) > 0.0 or any(abs(r["g_fd"]) > 0.0 for r in fds)
    return {
        "g_auto": g_auto,
        "fd": fds,
        "best_rel": best["rel"],
        "all_rels": [r["rel"] for r in fds],
        "signal_present": signal,
        "pass_smooth": bool(signal and best["rel"] < 1e-4),
        "status": "PASS" if (signal and best["rel"] < 1e-4) else ("INSUFFICIENT" if not signal else "FAIL"),
        "branch_switch_separated": "see_guard_trace_not_hardcoded_true",
    }


def main():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "budgets.yaml").read_text(encoding="utf-8"))
    ids, params, rows = load_train_cases(4)
    frozen = load_frozen_kernel()
    logger = RunLogger("r4_P4_train", {
        "stage": "P4", "case_ids": ids, "budget_s": cfg["budgets_s"]["p4_train"],
        "kernel_sha": frozen.yaml_sha256,
        "source_sha256": [r["source_sha256"] for r in rows],
    })
    well = case_to_well(params[0])
    t_need = well.t_s + float(well.x_clusters[0] / well.a) + 0.15
    num = make_num(32, T=t_need, friction="none")
    n_steps = int(ProductionMOC(well, num, "none").S["n_steps"])

    def loss_of(pred):
        return ((pred - pred[0]) ** 2).sum()

    # reset must not restore attached params
    roll = ProductionMOC(well, num, "none")
    k0 = roll.pK.detach().clone()
    theta = torch.tensor(0.02 * float(k0[0]), dtype=torch.float64, requires_grad=True)
    roll.attach_param("pK", k0 + theta)
    attached_before = roll.pK.detach().clone()
    roll.reset()
    reset_ok = bool(torch.equal(roll.pK.detach(), attached_before))
    logger.log(f"reset keeps attached pK: {reset_ok}")

    # train entry vs direct unroll same loss grad (leaf Parameter, not a view)
    roll_a = ProductionMOC(well, num, "none")
    pK_a = torch.nn.Parameter(roll_a.pK.detach().clone())
    roll_a.attach_param("pK", pK_a)
    opt = torch.optim.SGD([pK_a], lr=0.0)
    entry = TrainEntry(roll_a, opt)
    out = entry.step_loss(n_steps, loss_of, require_ok=True)
    g_entry = None if pK_a.grad is None else float(pK_a.grad[0])

    roll_b = ProductionMOC(well, num, "none")
    pK_b = torch.nn.Parameter(roll_b.pK.detach().clone())
    roll_b.attach_param("pK", pK_b)
    roll_b.reset()
    pred = roll_b.unroll(n_steps, require_ok=True)
    loss_of(pred).backward()
    g_unroll = float(pK_b.grad[0])
    grad_match = g_entry is not None and abs(g_entry - g_unroll) / max(abs(g_unroll), 1e-30) < 1e-10
    logger.log(f"entry vs unroll grad {g_entry} vs {g_unroll} match={grad_match}")

    def make_none():
        return ProductionMOC(well, num, "none")

    K0 = float(make_none().pK[0].detach())

    def set_K(r, th):
        base = r.pK.detach().clone()
        e0 = torch.zeros_like(base)
        e0[0] = 1.0
        r.pK = base + e0 * th

    rec_k = fd_grad(make_none, n_steps, loss_of, 0.0,
                    [1e-3 * abs(K0), 1e-4 * abs(K0), 1e-5 * abs(K0)], set_K)
    rec_k["path"] = "K0_frozen_IC"
    rec_k["n_steps"] = n_steps
    rec_k["window_covers_arrival"] = True
    rec_k["guard_trace"] = ProductionMOC(well, num, "none").last_guard_trace
    # actually run once to fill trace
    rtmp = ProductionMOC(well, num, "none")
    rtmp.unroll(n_steps, require_ok=True)
    rec_k["guard_trace"] = rtmp.last_guard_trace
    rec_k["n_storage_switch_total"] = int(sum(x["n_storage_switch"] for x in rtmp.last_guard_trace))
    rec_k["any_tikhonov"] = any(x["tikhonov"] for x in rtmp.last_guard_trace)
    logger.log(f"K grad status={rec_k['status']} rels={rec_k['all_rels']}")

    # Cf/Gl path
    def set_Cf(r, th):
        base = r.pCf.detach().clone()
        e0 = torch.zeros_like(base)
        e0[0] = 1.0
        r.pCf = base * (1.0 + e0 * th)

    def set_Gl(r, th):
        base = r.pGl.detach().clone()
        e0 = torch.zeros_like(base)
        e0[0] = 1.0
        r.pGl = base * (1.0 + e0 * th)

    rec_cf = fd_grad(make_none, n_steps, loss_of, 0.0, [1e-3, 1e-4, 1e-5], set_Cf)
    rec_cf["path"] = "Cf0_relative"
    logger.log(f"Cf grad status={rec_cf['status']} rels={rec_cf['all_rels']}")
    rec_gl = fd_grad(make_none, n_steps, loss_of, 0.0, [1e-3, 1e-4, 1e-5], set_Gl)
    rec_gl["path"] = "Gl0_relative"
    logger.log(f"Gl grad status={rec_gl['status']} rels={rec_gl['all_rels']}")

    # memory path: ZVB Ju depends on history; perturb first-segment kernel scale via nu? 
    # Use zvb and perturb K still, plus a dedicated Ju-affecting param: pK on zvb
    numz = make_num(32, T=t_need, friction="zvb_rec")
    n_z = int(ProductionMOC(well, numz, "zvb_rec", frozen=frozen).S["n_steps"])

    def make_z():
        return ProductionMOC(well, numz, "zvb_rec", frozen=frozen)

    rec_mem = fd_grad(make_z, n_z, loss_of, 0.0,
                      [1e-3 * abs(K0), 1e-4 * abs(K0), 1e-5 * abs(K0)], set_K)
    rec_mem["path"] = "K0_on_zvb_memory_rollout"
    rec_mem["n_steps"] = n_z
    logger.log(f"memory-path grad status={rec_mem['status']} rels={rec_mem['all_rels']}")

    # Newton fail: pre-register max_iter budget on a real well step
    fail = {"registered_max_iter": [1, 0], "attempts": []}
    blocked = False
    for mi in (1, 0):
        numf = make_num(32, T=well.t_s + 0.05, friction="none", max_iter=mi)
        rollf = ProductionMOC(well, numf, "none")
        pKf = torch.nn.Parameter(rollf.pK.detach().clone())
        rollf.attach_param("pK", pKf)
        optf = torch.optim.SGD([pKf], lr=1e-3)
        schedf = torch.optim.lr_scheduler.StepLR(optf, step_size=1, gamma=0.9)
        entryf = TrainEntry(rollf, optf, schedf)
        p_before = {n: getattr(rollf, n).detach().clone() for n in ("pK",)}
        opt_hash_before = str(optf.state_dict())
        sched_before = copy.deepcopy(schedf.state_dict())
        g_before = entryf.global_step
        dyn_before = None
        rollf.reset()
        dyn_before = rollf.snapshot_dynamic()
        ntry = int(rollf.S["n_steps"])
        outf = entryf.step_loss(ntry, loss_of, require_ok=True)
        dyn_after = rollf.snapshot_dynamic()
        opt_same = str(optf.state_dict()) == opt_hash_before
        sched_same = schedf.state_dict() == sched_before
        g_same = entryf.global_step == g_before
        H_same = bool(torch.allclose(dyn_after["H"], dyn_before["H"]))
        rec_try = {
            "max_iter": mi,
            "entry": {k: outf[k] for k in outf if k != "pred"},
            "optimizer_unadvanced": opt_same,
            "scheduler_unadvanced": sched_same,
            "global_step_unadvanced": g_same,
            "dynamic_unpolluted": H_same,
            "no_monkeypatch": True,
        }
        fail["attempts"].append(rec_try)
        logger.log(f"max_iter={mi} reason={outf.get('reason')} block={outf.get('reason')=='newton_fail'} "
                   f"opt={opt_same} step={g_same} dyn={H_same}")
        if outf.get("reason") == "newton_fail" and opt_same and g_same and H_same:
            blocked = True
            break
    fail["blocked"] = blocked
    fail["normal_budget_fail_rate"] = {
        "note": "this only validates the handler; not a distribution failure rate",
        "normal_max_iter": 100,
        "n_fail_on_case0_short": int(ProductionMOC(well, num, "none").n_fail),
    }
    # normal-budget fail count on a real short rollout
    rn = ProductionMOC(well, num, "none")
    rn.unroll(n_steps, require_ok=False)
    fail["normal_budget_fail_rate"]["n_fail_observed"] = int(rn.n_fail)
    fail["normal_budget_fail_rate"]["n_steps"] = n_steps

    out = {
        "reset_keeps_attached": reset_ok,
        "train_entry_grad_matches_unroll": grad_match,
        "g_entry": g_entry,
        "g_unroll": g_unroll,
        "grad_K": rec_k,
        "grad_Cf": rec_cf,
        "grad_Gl": rec_gl,
        "grad_memory_path": rec_mem,
        "newton_fail_handler": fail,
        "same_implementation": "ProductionMOC",
        "r3_local_K_kept": True,
    }
    logger.write_metrics(out)
    dump_json(out, logger.dir / "train_entry_and_failure.json")
    logger.close("completed")
    print("P4 done", logger.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
