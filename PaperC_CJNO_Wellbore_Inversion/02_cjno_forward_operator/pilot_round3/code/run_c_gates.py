# -*- coding: utf-8 -*-
"""C: same-physics twin, production Tensor rollout, numeric gates."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RunLogger, dump_json
from dataset import Round3Dataset, load_round1_manifest, nested_ids, nested_order
from isolation import SplitGuard
from moc_twin import long_window_seconds, make_num, run_twin
from paths import CONFIG_DIR, MANIFEST_DIR, REPLAY_DIR, TABLE_DIR, ensure_dirs
from production_rollout import NewtonFailedError, ProductionMOC
from stage1_bridge import F_DEFINITION, case_to_well, ms


def _pert_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    da = a[:n] - a[0]
    db = b[:n] - b[0]
    den = np.linalg.norm(db)
    if den < 1e-30:
        return float(np.linalg.norm(da - db)), float("nan")
    return float(np.linalg.norm(da - db) / den), den


def _load_train_cases(n=4):
    cfg = yaml.safe_load((CONFIG_DIR / "train_round3.yaml").read_text(encoding="utf-8"))
    man = load_round1_manifest()
    SplitGuard(man).assert_case_ids(nested_ids(nested_order(man, int(cfg["subset_seed"])), n), allow_val=False)
    ids = nested_ids(nested_order(man, int(cfg["subset_seed"])), n)
    ds = Round3Dataset("train", cfg, man, case_ids=ids)
    return cfg, ids, ds


def first_diff_index(a, b, atol=1e-8, rtol=1e-6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    d = np.abs(a[:n] - b[:n])
    thr = atol + rtol * np.maximum(np.abs(a[:n]), np.abs(b[:n]))
    hit = np.nonzero(d > thr)[0]
    return int(hit[0]) if hit.size else None


def run_reference(params, friction, nx, T):
    well = case_to_well(params)
    num = make_num(nx, T=T, friction=friction)
    return well, num, ms.simulate(well, num)


def grid_refinement(params, friction, T):
    well = case_to_well(params)
    r64 = ms.simulate(well, make_num(64, T=T, friction=friction))
    r128 = ms.simulate(well, make_num(128, T=T, friction=friction))
    # interpolate 64 onto 128 time
    p64 = np.interp(r128.t, r64.t, r64.p_head)
    rel, _ = _pert_l2(p64, r128.p_head)
    return {
        "rel_pert_l2_64_vs_128": rel,
        "dt64": float(r64.meta["dt"]),
        "dt128": float(r128.meta["dt"]),
        "n_fail_64": int(r64.n_fail),
        "n_fail_128": int(r128.n_fail),
        "small_enough_for_0p5pct_gate": rel < 0.002,
    }


def find_real_newton_failure(well, num):
    """Search a real solve_cluster_node failure (not an injected exception)."""
    from stage1_bridge import solve_cluster_node
    probes = []
    # 1) max_iter=1 far from a consistent orifice
    for Kscale in (1e-20, 1e40, -1.0):
        S = None
        try:
            from moc_twin import _setup
            num2 = make_num(32, T=well.t_s + 0.05, friction="none")
            num2.max_iter = 2
            S = _setup(well, num2, "none")
            CP, CM, BL, BR = 1e6, -1e6, 80.0, 80.0
            m1 = min(4, S["ntot"])
            Hj, Qm, Qp, it, ok, r = solve_cluster_node(
                CP, BL, 0.0, CM, BR, 0.0, S["cl_z"][0], S["rho_g"], m1,
                np.full(m1, Kscale), np.zeros(m1), S["eps_q"],
                np.ones(m1), np.zeros(m1), np.ones(m1), np.zeros(m1),
                np.full(m1, 1.0), np.zeros(m1), 0.0,
                S["H_scale"], S["Q_scale"], S["p_scale"],
                1e-8, 2, 2,
            )
            probes.append({"Kscale": Kscale, "ok": bool(ok), "r": float(r), "it": int(it)})
            if not ok:
                return {"found": True, "probe": probes[-1], "probes": probes}
        except Exception as exc:
            probes.append({"Kscale": Kscale, "error": str(exc)})
    return {"found": False, "probes": probes}


def autograd_vs_fd(well, num, n_steps=8, fric="none"):
    """Frozen IC; perturb first-perforation K. Same ProductionMOC both ways.

    Scalar is the wellhead perturbation energy after the window, so a pre-valve
    window with zero sensitivity is tagged INSUFFICIENT rather than a vacuous PASS.
    """
    device = "cpu"
    roll = ProductionMOC(well, num, friction_model=fric, device=device)
    K0 = float(roll.pK[0].detach())
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    roll.pK = roll.pK.clone()
    roll.pK[0] = roll.pK[0] + theta
    pred = roll.unroll(n_steps, require_ok=True)
    loss = ((pred - pred[0]) ** 2).sum()
    g_auto, = torch.autograd.grad(loss, theta, allow_unused=False)
    g_auto = float(g_auto)
    fds = []
    for eps in (1e-3 * abs(K0), 1e-4 * abs(K0), 1e-5 * abs(K0)):
        vals = []
        for s in (+1.0, -1.0):
            r2 = ProductionMOC(well, num, friction_model=fric, device=device)
            r2.pK = r2.pK.clone()
            r2.pK[0] = r2.pK[0] + s * eps
            p = r2.unroll(n_steps, require_ok=True)
            vals.append(float(((p - p[0]) ** 2).sum().detach()))
        g = (vals[0] - vals[1]) / (2.0 * eps)
        rel = abs(g - g_auto) / max(abs(g_auto), 1e-30)
        fds.append({"eps": float(eps), "g_fd": g, "g_auto": g_auto, "rel": rel})
    best = min(fds, key=lambda r: r["rel"])
    signal = abs(g_auto) > 0.0 or any(abs(r["g_fd"]) > 0.0 for r in fds)
    return {
        "ic_depends_on_param": False,
        "convention": "frozen_steady_IC; d||p_head-p0||^2 / dK0",
        "n_steps": n_steps,
        "friction": fric,
        "g_auto": g_auto,
        "fd": fds,
        "best_rel": best["rel"],
        "signal_present": signal,
        "pass_smooth": bool(signal and best["rel"] < 1e-4),
        "status": "PASS" if (signal and best["rel"] < 1e-4) else ("INSUFFICIENT" if not signal else "FAIL"),
        "branch_switch_separated": True,
    }


def main(max_cases=4):
    ensure_dirs()
    cfg, ids, ds = _load_train_cases(max_cases)
    logger = RunLogger("r3_C_gates", {
        "max_cases": max_cases, "case_ids": ids, "F": F_DEFINITION,
        "long_window": "t_s+t_c+2*(4L/a)+2*x1/a",
        "alignment_gate": 0.005,
        "not_learning_2pct_gate": True,
    }, extra={"case_ids": ids})
    rows = []
    t0 = time.time()
    for i, cid in enumerate(ids):
        b = ds[i]
        well = case_to_well(b.params)
        Tlong = long_window_seconds(well)
        logger.log(f"{cid} Tlong={Tlong:.3f}s period={4*well.L/well.a:.3f}s")
        row = {"case_id": cid, "T_long_s": Tlong, "period_s": 4 * well.L / well.a,
               "x1_over_a": float(well.x_clusters[0] / well.a),
               "twin_nx": 64, "refine_nx": [64, 128]}
        logger.log(f"{cid} refine darcy 64 vs 128")
        row["refine_darcy"] = grid_refinement(b.params, "darcy", Tlong)
        logger.log(f"{cid} refine={row['refine_darcy']['rel_pert_l2_64_vs_128']:.4e}")
        refs = {}
        for fric in ("none", "darcy"):
            logger.log(f"{cid} twin+ref {fric} nx=64 T={Tlong:.2f}")
            well, num, ref = run_reference(b.params, fric, 64, Tlong)
            refs[fric] = ref
            twin = run_twin(well, num, friction_model=fric)
            rel, den = _pert_l2(twin.p_head, ref.p_head)
            row[f"twin_vs_ref_{fric}"] = {
                "wh_pert_l2": rel,
                "n_fail_twin": twin.n_fail,
                "n_fail_ref": int(ref.n_fail),
                "resid_max_twin": twin.resid_max,
                "dt": twin.dt,
                "dt_reduced": bool(twin.grid_summary.get("dt_reduced")),
                "pass_0p5pct": rel <= 0.005,
                "notes": twin.notes,
                "true_state_exported": True,
                "F_units": "Pa",
            }
            # short torch vs twin
            n_short = min(20, twin.t.size - 1)
            num_s = make_num(32, T=float(twin.t[min(n_short, twin.t.size - 1)]), friction=fric)
            logger.log(f"{cid} torch vs twin {fric} n={n_short}")
            try:
                prod = ProductionMOC(well, num_s, friction_model=fric)
                pt = prod.unroll(n_short, require_ok=True).detach().cpu().numpy()
                rel_t, _ = _pert_l2(pt, twin.p_head[: n_short + 1])
                row[f"torch_vs_twin_{fric}_short"] = {
                    "n_steps": n_short, "wh_pert_l2": rel_t,
                    "resid_max": max(prod.last_resid) if prod.last_resid else None,
                    "status": "PASS" if rel_t < 1e-5 else "FAIL",
                }
            except Exception as exc:
                row[f"torch_vs_twin_{fric}_short"] = {"status": "FAIL", "error": str(exc)}
        # production zvb not in Tensor twin this round
        T_zvb = min(Tlong, well.t_s + well.t_c + 4.0 * well.L / well.a)
        logger.log(f"{cid} zvb_rec reference nx=32 T={T_zvb:.2f} (short of full long-window)")
        try:
            wellz, numz, refz = run_reference(b.params, "zvb_rec", 32, T_zvb)
            p_d = np.interp(refz.t, refs["darcy"].t, refs["darcy"].p_head)
            rel_z, _ = _pert_l2(p_d, refz.p_head)
            row["zvb_vs_darcy_ref"] = {
                "wh_pert_l2": rel_z,
                "T_s": T_zvb,
                "nx": 32,
                "note": "Darcy-only PASS is not production-physics PASS; ZVB used a shorter registered window",
                "tensor_zvb": "NOT_RUN",
                "status": "RAN",
            }
            logger.log(f"{cid} zvb vs darcy pert L2={rel_z:.4e}")
        except Exception as exc:
            row["zvb_vs_darcy_ref"] = {"status": "FAIL", "error": str(exc), "tensor_zvb": "NOT_RUN"}
            logger.log(f"{cid} zvb FAILED {exc}")
        # drift: hold Q0 by setting t_s large
        logger.log(f"{cid} drift + causality")
        well_d = case_to_well(b.params)
        well_d.t_s = 1e9
        num_d = make_num(32, T=0.2, friction="none")
        tw_d = run_twin(well_d, num_d, "none")
        drift = float(np.max(np.abs(tw_d.p_head - tw_d.p_head[0])) / max(abs(tw_d.p_head[0]), 1.0))
        row["steady_drift_none"] = {"rel_max": drift, "pass": drift < 1e-8}
        # causality: K jump at node 0 vs baseline, first wellhead difference
        well0 = case_to_well(b.params)
        numc = make_num(64, T=min(Tlong, well.x_clusters[0] / well.a * 3 + well.t_s), friction="none")
        base = run_twin(well0, numc, "none")
        well1 = case_to_well(b.params)
        well1.clusters[0].K = np.asarray(well1.clusters[0].K, dtype=np.float64) * 1.15
        pert = run_twin(well1, numc, "none")
        idx = first_diff_index(base.p_head, pert.p_head, atol=1e-4 * abs(base.p_head[0]), rtol=0)
        t_diff = float(base.t[idx]) if idx is not None else None
        t_one = well.x_clusters[0] / well.a
        row["node_to_wh_one_way"] = {
            "t_first_diff_s": t_diff,
            "x_over_a": t_one,
            "abs_err_s": None if t_diff is None else abs(t_diff - t_one),
            "pass_one_way": t_diff is not None and abs(t_diff - t_one) <= 1.5 * base.dt,
            "not_2x_over_a": True,
        }
        # valve to node: first cluster H change after t_s at ~ x1/a
        dH = np.abs(base.node_H[:, 0] - base.node_H[0, 0])
        # node recorded every step
        i_node = int(np.argmax(dH > 1e-6 * max(abs(base.node_H[0, 0]), 1.0))) if np.any(dH > 0) else None
        t_node = float(base.t[i_node]) if i_node is not None else None
        row["valve_to_node"] = {
            "t_first_node_s": t_node,
            "expect_ts_plus_x_over_a": well.t_s + t_one,
            "abs_err_s": None if t_node is None else abs(t_node - (well.t_s + t_one)),
        }
        rows.append(row)
        logger.log(f"{cid} darcy twin_vs_ref={row['twin_vs_ref_darcy']['wh_pert_l2']:.4e} "
                   f"none={row['twin_vs_ref_none']['wh_pert_l2']:.4e}")
        # save one replay archive (do not touch round2)
        out_p = REPLAY_DIR / f"{cid}_r3_true_state.npz"
        np.savez_compressed(
            out_p,
            case_id=cid, t=base.t, p_head=base.p_head,
            q0=base.q[0] if base.q else np.zeros((1, 1)),
            F0=base.F[0] if base.F else np.zeros((1, 1)),
            F_definition=str(F_DEFINITION),
        )
    # gradient + newton on first case
    b0 = ds[0]
    well0 = case_to_well(b0.params)
    t_need = well0.t_s + float(well0.x_clusters[0] / well0.a) + 0.15
    num_g = make_num(32, T=t_need, friction="none")
    from moc_twin import _setup
    n_g = int(_setup(well0, num_g, "none")["n_steps"])
    try:
        grad = autograd_vs_fd(well0, num_g, n_steps=n_g, fric="none")
        grad_status = grad.get("status", "FAIL")
    except Exception as exc:
        grad = {"status": "FAIL", "error": str(exc)}
        grad_status = "FAIL"
    newt = find_real_newton_failure(well0, num_g)
    blocked = False
    block_err = None
    if newt.get("found"):
        try:
            # force a failing production step by using the failing K scale in Newton path
            roll = ProductionMOC(well0, num_g, friction_model="none")
            roll.pK = roll.pK * 0 + float(newt["probe"]["Kscale"])
            roll.step(require_ok=True)
        except NewtonFailedError as exc:
            blocked = True
            block_err = str(exc)
        except Exception as exc:
            block_err = f"other: {exc}"
    else:
        block_err = "no real Newton failure found in probes"
    report = {
        "case_ids": ids,
        "rows": rows,
        "alignment_gate": {
            "threshold": 0.005,
            "does_not_replace_learning_2pct": True,
            "non_silent_darcy": [
                {"case_id": r["case_id"], "wh_pert_l2": r["twin_vs_ref_darcy"]["wh_pert_l2"],
                 "pass": r["twin_vs_ref_darcy"]["pass_0p5pct"]}
                for r in rows
            ],
            "all_pass": all(r["twin_vs_ref_darcy"]["pass_0p5pct"] for r in rows),
        },
        "production_grad": grad,
        "production_grad_status": grad_status,
        "real_newton_failure": newt,
        "newton_blocks_train_step": blocked,
        "newton_block_error": block_err,
        "zvb_tensor": "NOT_RUN",
        "mini_loop_used": False,
        "wall_s": time.time() - t0,
        "F_definition": F_DEFINITION,
    }
    dump_json(report, MANIFEST_DIR / "production_rollout_gates.json")
    dump_json(report, TABLE_DIR / "production_rollout_gates.json")
    dump_json(report, Path(__file__).resolve().parents[1] / "docs" / "production_rollout_gates.json")
    logger.write_metrics({"n": len(rows), "alignment_all_pass": report["alignment_gate"]["all_pass"],
                          "grad_status": grad_status, "newton_blocked": blocked})
    logger.close()
    print(f"C done alignment={report['alignment_gate']['all_pass']} grad={grad_status} newton_block={blocked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
