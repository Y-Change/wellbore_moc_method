# -*- coding: utf-8 -*-
"""Production vs sufficiently refined reference. Independent of the causality table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import load_train_cases
from common import RunLogger, dump_json, pert_l2
from kernel_loader import load_frozen_kernel
from moc_twin import grid_inventory, make_num, run_twin, short_window_seconds
from physics import align_to_query, simulate_ref
from production_moc import ProductionMOC
from stage1_bridge import case_to_well


def main():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "budgets.yaml").read_text(encoding="utf-8"))
    p2_id = None
    # explicit predecessor from this session; also accept CLI
    if len(sys.argv) > 1:
        p2_id = sys.argv[1]
    ids, params, rows = load_train_cases(4)
    frozen = load_frozen_kernel()
    logger = RunLogger("r4_P2_align", {
        "stage": "P2b",
        "predecessor_run_id": p2_id,
        "case_ids": ids,
        "kernel_sha": frozen.yaml_sha256,
        "source_sha256": [r["source_sha256"] for r in rows],
        "gate": 0.005,
        "precondition": 0.002,
    })
    pred = None
    if p2_id:
        p = Path(__file__).resolve().parents[1] / "runs" / p2_id / "causality_and_convergence.json"
        if p.is_file():
            pred = json.loads(p.read_text(encoding="utf-8"))
    out_rows = []
    for i, cid in enumerate(ids):
        well = case_to_well(params[i])
        T = short_window_seconds(well)
        inv256 = grid_inventory(well, 256)
        inv512 = grid_inventory(well, 512)
        inv1024 = grid_inventory(well, 1024)
        r256, _ = simulate_ref(well, make_num(256, T=T, friction="darcy"), friction="darcy")
        r512, _ = simulate_ref(well, make_num(512, T=T, friction="darcy"), friction="darcy")
        r1024, _ = simulate_ref(well, make_num(1024, T=T, friction="darcy"), friction="darcy")
        t_q = r1024.t
        p256 = align_to_query(r256.t, r256.p_head, t_q)
        p512 = align_to_query(r512.t, r512.p_head, t_q)
        refine_256_512 = pert_l2(p256, align_to_query(r512.t, r512.p_head, t_q))
        refine_512_1024 = pert_l2(p512, r1024.p_head)
        pre = refine_512_1024 < 0.002
        # production at Cr=1 operating grid (512) vs finest ref
        num_op = make_num(512, T=T, friction="darcy")
        twin = run_twin(well, num_op, "darcy")
        with torch.no_grad():
            prod = ProductionMOC(well, num_op, "darcy")
            pp = prod.unroll(int(twin.t.size - 1), require_ok=True).detach().cpu().numpy()
        p_prod = align_to_query(twin.t, pp, t_q)
        p_twin = align_to_query(twin.t, twin.p_head, t_q)
        a_prod = pert_l2(p_prod, r1024.p_head)
        a_twin = pert_l2(p_twin, r1024.p_head)
        a_disc = pert_l2(pp, twin.p_head)
        row = {
            "case_id": cid,
            "T": T,
            "inventories": {"256": inv256, "512": inv512, "1024": inv1024},
            "same_discrete_prod_vs_twin_512": a_disc,
            "reference_refine": {
                "256_vs_512": refine_256_512,
                "512_vs_1024": refine_512_1024,
                "precondition_finest_lt_0p2pct": pre,
                "three_levels": True,
                "not_continuum_bound": True,
            },
            "production_vs_refined_ref": {
                "operating_nx": 512,
                "ref_nx": 1024,
                "aligned_physical_time": True,
                "prod_vs_ref1024": a_prod,
                "twin_vs_ref1024": a_twin,
                "gate": 0.005,
                "gate_applicable": pre,
                "pass": bool(pre and a_prod <= 0.005 and a_twin <= 0.005),
                "status": ("PASS" if (pre and a_prod <= 0.005) else
                           ("NOT_APPLICABLE" if not pre else "FAIL")),
            },
        }
        out_rows.append(row)
        logger.log(f"{cid} refine512-1024={refine_512_1024:.4e} prod-vs-1024={a_prod:.4e} "
                   f"applicable={pre} {row['production_vs_refined_ref']['status']}")
    rec = {
        "rows": out_rows,
        "predecessor_run_id": p2_id,
        "tables_independent": True,
        "all_applicable_pass": all(r["production_vs_refined_ref"]["status"] == "PASS" for r in out_rows),
    }
    logger.write_metrics(rec)
    dump_json(rec, logger.dir / "alignment_0p5pct.json")
    logger.close("completed")
    print("P2b done", logger.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
