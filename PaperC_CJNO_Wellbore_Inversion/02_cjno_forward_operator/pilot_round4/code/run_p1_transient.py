# -*- coding: utf-8 -*-
"""P1: non-silent transient parity. No fixed 15/20-step endpoint."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import classify_silence, joukowsky_scale, load_train_cases
from common import RunLogger, dump_json, pert_l2, rel_l2
from kernel_loader import load_frozen_kernel
from moc_twin import grid_inventory, long_window_seconds, make_num, run_twin, short_window_seconds
from physics import event_window_ok, simulate_ref
from production_moc import ProductionMOC
from stage1_bridge import SHARED_PRIMITIVES, case_to_well


def _prod_hist(roll: ProductionMOC, n_steps: int):
    q, pc, F = [roll.q.detach().cpu().numpy().copy()], [roll.pc.detach().cpu().numpy().copy()], [roll.F.detach().cpu().numpy().copy()]
    Hl, Hr, Ql, Qr = [], [], [], []
    grid = roll.grid
    ncl = roll.S["ncl"]

    def snap_nodes():
        hl, hr, ql, qr = [], [], [], []
        for jc in range(ncl):
            iL = grid.cluster_left_index(jc)
            iR = grid.cluster_right_index(jc)
            hl.append(float(roll.H[iL].detach()))
            hr.append(float(roll.H[iR].detach()))
            ql.append(float(roll.Q[iL].detach()))
            qr.append(float(roll.Q[iR].detach()))
        Hl.append(hl); Hr.append(hr); Ql.append(ql); Qr.append(qr)

    p = [float(roll.p_head().detach())]
    snap_nodes()
    for _ in range(n_steps):
        roll.step(require_ok=True)
        p.append(float(roll.p_head().detach()))
        q.append(roll.q.detach().cpu().numpy().copy())
        pc.append(roll.pc.detach().cpu().numpy().copy())
        F.append(roll.F.detach().cpu().numpy().copy())
        snap_nodes()
    return {
        "p_head": np.asarray(p, dtype=np.float64),
        "q_hist": np.asarray(q),
        "pc_hist": np.asarray(pc),
        "F_hist": np.asarray(F),
        "H_left": np.asarray(Hl),
        "H_right": np.asarray(Hr),
        "Q_left": np.asarray(Ql),
        "Q_right": np.asarray(Qr),
        "newton_calls": roll.newton_calls,
        "n_fail": roll.n_fail,
        "resid": list(roll.last_resid),
        "guard_trace": list(roll.last_guard_trace),
        "state": roll.export_numpy_state(),
    }


def one_case(logger, cid, params, frozen, friction, nx, T, kind, budget_left):
    well = case_to_well(params)
    inv = grid_inventory(well, nx)
    scale = joukowsky_scale(params)
    num = make_num(nx, T=T, friction=friction)
    t0 = time.perf_counter()
    ref, wref = simulate_ref(well, num, frozen=frozen if friction == "zvb_rec" else None, friction=friction)
    twin = run_twin(well, num, friction, frozen=frozen if friction == "zvb_rec" else None)
    ev = event_window_ok(well, twin.t, twin.p_head, scale, kind)
    if ev["silent"] and kind != "steady":
        return {"case_id": cid, "status": "STEADY_ONLY_SILENT", "event": ev,
                "note": "silent window cannot receive a transient PASS"}
    if not ev["event_occurred"] or not ev["complete"]:
        return {"case_id": cid, "status": "INSUFFICIENT", "event": ev,
                "note": "event/window incomplete; not converted to physical FAIL"}
    if not np.allclose(twin.t, ref.t):
        return {"case_id": cid, "status": "FAIL", "reason": "time_axis_mismatch"}
    n_steps = int(twin.t.size - 1)
    t1 = time.perf_counter()
    with torch.no_grad():
        prod = ProductionMOC(well, num, friction, device="cpu", frozen=frozen if friction == "zvb_rec" else None)
        ph = _prod_hist(prod, n_steps)
    t_prod = time.perf_counter() - t1
    if ph["newton_calls"] <= 0:
        return {"case_id": cid, "status": "FAIL", "reason": "newton_not_called"}
    row = {
        "case_id": cid,
        "friction": friction,
        "nx": nx,
        "kind": kind,
        "T": T,
        "grid": inv,
        "event": ev,
        "shared_primitives": list(SHARED_PRIMITIVES),
        "consistency_not_independent_physics": True,
        "wh_pert_l2": {
            "twin_vs_ref": pert_l2(twin.p_head, ref.p_head),
            "prod_vs_twin": pert_l2(ph["p_head"], twin.p_head),
            "prod_vs_ref": pert_l2(ph["p_head"], ref.p_head),
        },
        "cluster": {
            "q": rel_l2(ph["q_hist"], twin.q_hist),
            "pc": rel_l2(ph["pc_hist"], twin.pc_hist),
            "F": rel_l2(ph["F_hist"], twin.F_hist),
            "H_left": rel_l2(ph["H_left"], twin.H_left),
            "H_right": rel_l2(ph["H_right"], twin.H_right),
            "Q_left": rel_l2(ph["Q_left"], twin.Q_left),
            "Q_right": rel_l2(ph["Q_right"], twin.Q_right),
            "n_perf": int(twin.perf_off[-1]),
            "n_clusters": int(twin.H_left.shape[1]),
        },
        "newton_calls": {"twin": twin.newton_calls, "prod": ph["newton_calls"]},
        "n_fail": {"ref": int(ref.n_fail), "twin": twin.n_fail, "prod": ph["n_fail"]},
        "resid_max_twin": twin.resid_max,
        "full_state_exported": True,
        "not_only_cluster0": True,
        "timings_s": {"ref": wref, "twin_and_ref": time.perf_counter() - t0, "prod": t_prod},
        "kernel_audit": twin.kernel_audit,
        "used_online_refit": False,
    }
    disc = max(abs(row["wh_pert_l2"]["twin_vs_ref"]), abs(row["wh_pert_l2"]["prod_vs_twin"]))
    cl = max(v for v in row["cluster"].values() if isinstance(v, float))
    row["same_discrete_pass"] = bool(disc < 1e-12 and cl < 1e-10 and ph["newton_calls"] > 0 and ev["event_occurred"])
    row["status"] = "PASS" if row["same_discrete_pass"] else "FAIL"
    arch = logger.dir / f"{cid}_{friction}_{kind}_nx{nx}_state.npz"
    np.savez_compressed(
        arch,
        case_id=cid, t=twin.t, p_head_ref=ref.p_head, p_head_twin=twin.p_head, p_head_prod=ph["p_head"],
        q_hist=twin.q_hist, pc_hist=twin.pc_hist, F_hist=twin.F_hist, perf_off=twin.perf_off,
        H_left=twin.H_left, H_right=twin.H_right, Q_left=twin.Q_left, Q_right=twin.Q_right,
        H0=twin.H0, Q0=twin.Q0, q0=twin.q0, pc0=twin.pc0, F0=twin.F0,
        H_final=twin.H_final, Q_final=twin.Q_final, z_final=twin.z_final, Ju_final=twin.Ju_final,
        prod_q=ph["q_hist"], prod_pc=ph["pc_hist"], prod_F=ph["F_hist"], prod_z=ph["state"]["z"],
    )
    row["archive"] = str(arch)
    logger.log(f"{cid} {friction} {kind} nx={nx} twin-ref={row['wh_pert_l2']['twin_vs_ref']:.3e} "
               f"prod-twin={row['wh_pert_l2']['prod_vs_twin']:.3e} {row['status']}")
    return row


def main():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "budgets.yaml").read_text(encoding="utf-8"))
    budget = float(cfg["budgets_s"]["p1_transient"])
    ids, params, rows = load_train_cases(4)
    frozen = load_frozen_kernel()
    logger = RunLogger("r4_P1_transient", {
        "stage": "P1", "case_ids": ids, "budget_s": budget,
        "windows": cfg["windows"], "kernel_sha": frozen.yaml_sha256,
        "source_sha256": [r["source_sha256"] for r in rows],
    })
    t0 = time.perf_counter()
    # choose nx: verify Cr0=1 at 512 on all 4, but start short at 64 then 512 if budget
    inventories = []
    nx_pref = 64
    cr1 = True
    for p in params:
        w = case_to_well(p)
        g512 = grid_inventory(w, 512)
        inventories.append({"case": None, "g512": g512, "g64": grid_inventory(w, 64)})
        cr1 = cr1 and g512["Cr0_is_one"]
    for i, cid in enumerate(ids):
        inventories[i]["case"] = cid
    logger.log(f"Nx512 Cr0==1 on all4? {cr1}")
    results = {"inventories": inventories, "rows": []}
    # none short @64 first (timed), then darcy short @64, then none short @512 if budget
    plan = [
        ("none", 64, "short"),
        ("darcy", 64, "short"),
    ]
    if cr1:
        plan.append(("none", 512, "short"))
        plan.append(("darcy", 512, "short"))
    plan.append(("none", 64, "long"))
    plan.append(("darcy", 64, "long"))
    for fric, nx, kind in plan:
        if time.perf_counter() - t0 > budget:
            results["rows"].append({"status": "TIME_BUDGET", "plan": [fric, nx, kind]})
            logger.log(f"TIME_BUDGET before {fric} {kind} nx={nx}")
            break
        for i, cid in enumerate(ids):
            if time.perf_counter() - t0 > budget:
                results["rows"].append({"case_id": cid, "status": "TIME_BUDGET", "friction": fric, "kind": kind, "nx": nx})
                break
            well = case_to_well(params[i])
            T = short_window_seconds(well) if kind == "short" else long_window_seconds(well)
            row = one_case(logger, cid, params[i], frozen, fric, nx, T, kind, budget - (time.perf_counter() - t0))
            results["rows"].append(row)
            dump_json(row, logger.dir / f"row_{cid}_{fric}_{kind}_nx{nx}.json")
    results["elapsed_s"] = time.perf_counter() - t0
    results["budget_s"] = budget
    transient_pass = [r for r in results["rows"] if r.get("status") == "PASS" and r.get("event", {}).get("event_occurred")]
    results["summary"] = {
        "n_pass": len(transient_pass),
        "n_rows": len(results["rows"]),
        "same_discrete_zero_is_consistency_only": True,
    }
    logger.write_metrics(results)
    dump_json(results, logger.dir / "transient_parity.json")
    logger.close("completed")
    print("P1 done", logger.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
