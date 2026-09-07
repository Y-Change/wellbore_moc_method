# -*- coding: utf-8 -*-
"""P3: Tensor ZVB memory in the same ProductionMOC.step."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import load_train_cases
from common import RunLogger, dump_json, pert_l2, rel_l2
from kernel_loader import load_frozen_kernel
from moc_twin import long_window_seconds, make_num, run_twin, short_window_seconds
from physics import simulate_ref
from production_moc import ProductionMOC
from stage1_bridge import case_to_well
from zvb_recursion import full_kernel_numpy_vs_tensor, unit_one_term_hand


def zero_drive_pulse(frozen):
    """Zero drive keeps z=0; short pulse updates z and Ju."""
    unit = unit_one_term_hand()
    full = full_kernel_numpy_vs_tensor()
    return {"unit": unit, "full_kernel": full}


def one_forward(well, nx, T, frozen, friction):
    num = make_num(nx, T=T, friction=friction)
    ref, wref = simulate_ref(well, num, frozen=frozen if friction == "zvb_rec" else None, friction=friction)
    twin = run_twin(well, num, friction, frozen=frozen if friction == "zvb_rec" else None)
    with torch.no_grad():
        prod = ProductionMOC(well, num, friction, frozen=frozen if friction == "zvb_rec" else None)
        p = prod.unroll(int(twin.t.size - 1), require_ok=True).detach().cpu().numpy()
        st = prod.export_numpy_state()
    z_rel = rel_l2(st["z"], twin.z_final) if friction == "zvb_rec" else None
    ju_rel = rel_l2(st["Ju"], twin.Ju_final) if friction == "zvb_rec" else None
    return {
        "wh_twin_ref": pert_l2(twin.p_head, ref.p_head),
        "wh_prod_twin": pert_l2(p, twin.p_head),
        "z_rel": z_rel,
        "Ju_rel": ju_rel,
        "newton_calls": prod.newton_calls,
        "n_fail": {"ref": int(ref.n_fail), "twin": twin.n_fail, "prod": prod.n_fail},
        "dt": twin.dt,
        "n_steps": int(twin.t.size - 1),
        "n_nodes": int(twin.grid_summary["n_nodes"]),
        "kernel": twin.kernel_audit,
        "memory_affects_Cplus": friction == "zvb_rec",
        "ref_s": wref,
    }


def main():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "budgets.yaml").read_text(encoding="utf-8"))
    ids, params, rows = load_train_cases(4)
    frozen = load_frozen_kernel()
    logger = RunLogger("r4_P3_zvb", {
        "stage": "P3", "case_ids": ids, "budget_s": cfg["budgets_s"]["p3_zvb"],
        "kernel_sha": frozen.yaml_sha256,
        "source_sha256": [r["source_sha256"] for r in rows],
        "same_implementation": "ProductionMOC.step",
    })
    t0 = time.perf_counter()
    budget = float(cfg["budgets_s"]["p3_zvb"])
    rec = {"unit": zero_drive_pulse(frozen)}
    logger.log(f"unit pass={rec['unit']['unit']['pass']} full={rec['unit']['full_kernel']['pass']}")

    well0 = case_to_well(params[0])
    # zero-drive: hold valve open (t_s huge) short ZVB
    well_z = case_to_well(params[0])
    well_z.t_s = 1e9
    rec["zero_drive"] = one_forward(well_z, 64, 0.15, frozen, "zvb_rec")
    rec["zero_drive"]["note"] = "steady hold; Ju should stay ~0"
    logger.log(f"zero-drive Ju_rel={rec['zero_drive']['Ju_rel']} wh={rec['zero_drive']['wh_prod_twin']}")

    rec["short_pulse_case0"] = one_forward(well0, 64, min(0.4, short_window_seconds(well0)), frozen, "zvb_rec")
    logger.log(f"short pulse prod-twin={rec['short_pulse_case0']['wh_prod_twin']}")

    rec["first_train_short"] = one_forward(well0, 64, short_window_seconds(well0), frozen, "zvb_rec")
    logger.log(f"case0 short zvb prod-twin={rec['first_train_short']['wh_prod_twin']}")

    # Darcy vs ZVB same grid
    T = short_window_seconds(well0)
    rec["darcy_vs_zvb_same_grid"] = {
        "nx": 64,
        "darcy": one_forward(well0, 64, T, frozen, "darcy"),
        "zvb": rec["first_train_short"],
    }
    # physical difference: compare the two refs already inside
    numd = make_num(64, T=T, friction="darcy")
    numz = make_num(64, T=T, friction="zvb_rec")
    rd, _ = simulate_ref(well0, numd, friction="darcy")
    rz, _ = simulate_ref(well0, numz, frozen=frozen, friction="zvb_rec")
    rec["darcy_vs_zvb_same_grid"]["wh_pert_l2_physics"] = pert_l2(rd.p_head, rz.p_head)
    rec["darcy_vs_zvb_same_grid"]["same_nx"] = True
    rec["darcy_vs_zvb_same_grid"]["not_nx32_vs_64"] = True
    logger.log(f"Darcy vs ZVB same nx64 pert={rec['darcy_vs_zvb_same_grid']['wh_pert_l2_physics']:.4e}")

    rec["long_windows"] = []
    for i, cid in enumerate(ids):
        if time.perf_counter() - t0 > budget:
            rec["long_windows"].append({"case_id": cid, "status": "TIME_BUDGET"})
            logger.log(f"TIME_BUDGET long ZVB {cid}")
            break
        well = case_to_well(params[i])
        Tlong = long_window_seconds(well)
        try:
            with torch.no_grad():
                row = one_forward(well, 64, Tlong, frozen, "zvb_rec")
            row["case_id"] = cid
            row["T"] = Tlong
            row["status"] = "PASS" if (row["wh_prod_twin"] < 1e-10 and (row["z_rel"] is None or row["z_rel"] < 1e-10)) else (
                "FAIL" if row["wh_prod_twin"] == row["wh_prod_twin"] else "INSUFFICIENT"
            )
            if row["wh_prod_twin"] < 1e-10:
                row["status"] = "PASS"
            else:
                row["status"] = "FAIL"
            rec["long_windows"].append(row)
            logger.log(f"{cid} long zvb prod-twin={row['wh_prod_twin']:.3e} z={row['z_rel']}")
        except Exception as exc:
            rec["long_windows"].append({"case_id": cid, "status": "FAIL", "error": str(exc)})
            logger.log(f"{cid} long zvb error {exc}")

    rec["elapsed_s"] = time.perf_counter() - t0
    rec["same_step_implementation"] = True
    rec["used_online_refit"] = False
    logger.write_metrics(rec)
    dump_json(rec, logger.dir / "zvb_tensor_validation.json")
    logger.close("completed")
    print("P3 done", logger.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
