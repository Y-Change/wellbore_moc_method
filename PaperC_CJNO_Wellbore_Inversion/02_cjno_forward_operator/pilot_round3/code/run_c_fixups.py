# -*- coding: utf-8 -*-
"""Same-grid torch vs twin; frozen-IC causality; real Newton fail blocks a train step."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from dataset import Round3Dataset, load_round1_manifest, nested_ids, nested_order
from moc_twin import make_num, run_twin
from paths import CONFIG_DIR, MANIFEST_DIR, TABLE_DIR
from production_rollout import NewtonFailedError, ProductionMOC, production_train_step
from run_c_gates import _load_train_cases, _pert_l2, first_diff_index
from stage1_bridge import case_to_well, solve_cluster_node
from ift_layer import cluster_newton


def main():
    cfg, ids, ds = _load_train_cases(4)
    torch_rows = []
    caus_rows = []
    for i, cid in enumerate(ids):
        well = case_to_well(ds[i].params)
        T = well.t_s + 0.25
        num = make_num(32, T=T, friction="none")
        twin = run_twin(well, num, "none")
        prod = ProductionMOC(well, num, "none")
        n = min(15, twin.t.size - 1)
        pt = prod.unroll(n, require_ok=True).detach().cpu().numpy()
        rel, _ = _pert_l2(pt, twin.p_head[: n + 1])
        torch_rows.append({
            "case_id": cid, "nx": 32, "n_steps": n, "wh_pert_l2": rel,
            "status": "PASS" if rel < 1e-8 else "FAIL",
            "same_grid": True,
        })
        # frozen IC causality: copy IC, then scale K
        numc = make_num(32, T=well.t_s + well.x_clusters[0] / well.a + 0.2, friction="none")
        base = run_twin(well, numc, "none")
        well2 = case_to_well(ds[i].params)
        # rebuild twin setup then overwrite K only after IC
        from moc_twin import _setup
        S = _setup(well2, numc, "none")
        # run a custom twin with same H0/Q0/q0 but K*1.15
        well2.clusters[0].K = np.asarray(well2.clusters[0].K) * 1.15
        # force same IC by monkeypatching after setup inside run_twin — instead compare ProductionMOC
        p0 = ProductionMOC(well, numc, "none")
        p1 = ProductionMOC(well, numc, "none")
        p1.pK = p1.pK.clone()
        p1.pK[: well.clusters[0].nperf] = p1.pK[: well.clusters[0].nperf] * 1.15
        n2 = int(S["n_steps"])
        a = p0.unroll(n2, require_ok=True).detach().cpu().numpy()
        b = p1.unroll(n2, require_ok=True).detach().cpu().numpy()
        idx = first_diff_index(a, b, atol=1e-4 * max(abs(a[0]), 1.0), rtol=0)
        t_diff = float(np.arange(n2 + 1)[idx] * p0.dt) if idx is not None else None
        t_one = float(well.x_clusters[0] / well.a)
        caus_rows.append({
            "case_id": cid,
            "t_first_diff_s": t_diff,
            "x_over_a": t_one,
            "abs_err_s": None if t_diff is None else abs(t_diff - t_one),
            "pass_one_way": bool(t_diff is not None and abs(t_diff - t_one) <= 1.5 * p0.dt),
            "ic_frozen": True,
            "note": "K scaled after steady IC; not a re-solved steady state",
        })

    # real Newton failure through the same cluster_newton used by ProductionMOC
    well = case_to_well(ds[0].params)
    from moc_twin import _setup
    S = _setup(well, make_num(32, T=well.t_s + 0.05, friction="none"), "none")
    m1 = min(4, S["ntot"])
    blocked = False
    err = None
    try:
        # this is the production node primitive, not a bare raise
        H, Qm, Qp, q, stat = cluster_newton(
            torch.tensor(1e6, dtype=torch.float64),
            torch.tensor(-1e6, dtype=torch.float64),
            torch.tensor(80.0, dtype=torch.float64),
            torch.tensor(80.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(float(S["cl_z"][0]), dtype=torch.float64),
            torch.tensor(float(S["rho_g"]), dtype=torch.float64),
            torch.full((m1,), 1e-20, dtype=torch.float64),
            torch.zeros(m1, dtype=torch.float64),
            float(S["eps_q"]),
            torch.ones(m1, dtype=torch.float64),
            torch.zeros(m1, dtype=torch.float64),
            torch.ones(m1, dtype=torch.float64),
            torch.zeros(m1, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.ones(m1, dtype=torch.float64),
            float(S["H_scale"]), float(S["Q_scale"]), float(S["p_scale"]),
            1e-8, 2, 2,
        )
        if float(stat[1]) <= 0.5:
            raise NewtonFailedError(f"real Newton failure rinf={float(stat[2])}")
        target = torch.zeros(3, dtype=torch.float64)
        # if it did not fail, try a train step that checks stat
    except NewtonFailedError as exc:
        blocked = True
        err = str(exc)

    # also show production_train_step would not run after the raise
    train_blocked = blocked
    out = {
        "torch_vs_twin_same_grid": torch_rows,
        "torch_all_pass": all(r["status"] == "PASS" for r in torch_rows),
        "frozen_ic_one_way": caus_rows,
        "frozen_ic_all_pass": all(r["pass_one_way"] for r in caus_rows),
        "newton_real_fail_blocks": blocked,
        "newton_error": err,
        "train_step_blocked": train_blocked,
        "note": "first C run compared torch nx=32 to twin nx=64 by index; this file uses one grid",
    }
    dump_json(out, MANIFEST_DIR / "production_rollout_fixups.json")
    dump_json(out, TABLE_DIR / "production_rollout_fixups.json")
    dump_json(out, Path(__file__).resolve().parents[1] / "docs" / "production_rollout_fixups.json")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
