# -*- coding: utf-8 -*-
"""Diagnostic replay of ≤16 TRAIN cases. Does not overwrite original NPZ. No val/test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json, sha256_file
from dataset import load_round1_manifest, nested_ids, nested_order
from loop import CharacteristicLoop, loop_from_bundle
from paths import CONFIG_DIR, MANIFEST_DIR, REPLAY_DIR, STAGE1_CODE, ensure_dirs
from dataset import Round2Dataset

_saved = sys.modules.get("paths")
if "paths" in sys.modules:
    del sys.modules["paths"]
sys.path.insert(0, str(STAGE1_CODE))
import moc_solver as ms  # noqa: E402
if _saved is not None:
    sys.modules["paths"] = _saved


def case_to_well(c: dict):
    w = c["well"]
    cls = [ms.ClusterSpec.from_perf_params(
        x=cl["x"], Cd=cl["Cd"], A_perf=cl["A_perf"], kappa=cl["kappa"], I_f=cl["I_f"],
        R_f=cl["R_f"], C_f=cl["C_f"], G_l=cl["G_l"], p_res=cl["p_res"], rho=w["rho"])
        for cl in c["clusters"]]
    return ms.WellSpec(L=w["L"], D=w["D"], a=w["a"], rho=w["rho"], nu=w["nu"],
                       roughness=w["roughness"], TVD=c["TVD"], clusters=cls,
                       Q0=w["Q0"], t_s=1.0, t_c=w["t_c"], ramp="cosine", toe_bc="dead_end")


def replay_moc(params: dict, n_periods=0.5, nx=128, cr=1.0):
    well = case_to_well(params)
    num = ms.NumericsSpec(
        Nx_ref=int(nx), Cr_target=float(cr), friction_model="darcy",
        n_periods=float(n_periods), node_decim=1, do_energy=False, snap_every=0,
    )
    return ms.simulate(well, num)


def main(max_cases=4):
    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "train_round2.yaml").read_text(encoding="utf-8"))
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ids = nested_ids(order, max_cases)
    ds = Round2Dataset("train", cfg, man, case_ids=ids)
    rows = []
    for i, cid in enumerate(ids):
        b = ds[i]
        assert b.physical.case_id == cid
        params = b.params
        src = Path(b.source_file)
        out_p = REPLAY_DIR / f"{cid}_replay_short.npz"
        res = replay_moc(params, n_periods=0.5, nx=128)
        # 1-cell loop on the same query window (not claimed equal to high-res MOC)
        n_loop = min(80, b.query.t.size)
        loop_res = loop_from_bundle(b, n_steps=n_loop, require_ok=True)
        payload = {
            "case_id": cid,
            "source_file": str(src),
            "source_sha256": b.source_sha256,
            "t": res.t, "p_head": res.p_head, "Q_head": res.Q_head,
            "node_t": res.node_t, "node_H": res.node_H, "node_Qm": res.node_Qm,
            "node_Qp": res.node_Qp, "node_sq": res.node_sq, "node_pc": res.node_pc,
            "loop_p_wh": loop_res.p_wh, "loop_H_nodes": loop_res.H_nodes,
            "loop_Qm": loop_res.Qm, "loop_Qp": loop_res.Qp, "loop_sq": loop_res.sq,
            "loop_q_perf0": loop_res.q_perf[0],
            "gen": json.dumps({
                "Nx_ref": 128, "Cr_target": 1.0, "friction": "darcy",
                "n_periods": 0.5, "node_decim": 1,
                "loop": "CharacteristicLoop_1cell",
                "note": "per-perf time series only from 1-cell loop; high-res MOC stores cluster mean/sum",
            }),
        }
        np.savez_compressed(out_p, **payload)
        # compare loop wellhead vs resampled high-res on the loop query (honest error)
        t_q = b.query.t[:n_loop]
        p_ref = np.interp(t_q, res.t + 0.0, res.p_head)
        # high-res t starts at 0; query starts at t_s. align on absolute t
        p_ref = np.interp(t_q, res.t, res.p_head)
        rel = float(np.linalg.norm(loop_res.p_wh - p_ref) / (np.linalg.norm(p_ref - p_ref[0]) + 1e-30))
        row = {
            "case_id": cid,
            "source_sha256": b.source_sha256,
            "replay_file": str(out_p),
            "replay_sha256": sha256_file(out_p),
            "moc_dt": float(res.meta["dt"]),
            "moc_n_fail": int(res.n_fail),
            "loop_n_hard": loop_res.n_hard,
            "loop_resid_max": float(np.max(loop_res.resid)),
            "loop_vs_moc_pert_rel": rel,
            "diff_vs_original": {
                "original_node_decim": "16 (pilot NPZ)",
                "replay_node_decim": 1,
                "original_friction": "zvb_rec production",
                "replay_friction": "darcy short-window diagnostic",
                "loop_discretization": "1 cell / segment",
            },
        }
        rows.append(row)
        print(f"{cid} moc_fail={res.n_fail} loop_hard={loop_res.n_hard} loop_vs_moc={rel:.3f}")
    man_out = {
        "max_cases": max_cases,
        "split": "train_only",
        "n": len(rows),
        "rows": rows,
        "note": "does not overwrite stage-1/round1 NPZ; val/test not replayed",
    }
    dump_json(man_out, MANIFEST_DIR / "replay_manifest.json")
    dump_json(man_out, REPLAY_DIR / "replay_manifest.json")
    print("wrote replay manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
