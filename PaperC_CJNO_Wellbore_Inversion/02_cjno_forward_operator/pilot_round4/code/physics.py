# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cases import classify_silence, joukowsky_scale
from common import pert_l2, rel_l2
from kernel_loader import FrozenKernel
from moc_twin import TwinResult, long_window_seconds, short_window_seconds
from stage1_bridge import case_to_well, ms


def simulate_ref(well, num, frozen: Optional[FrozenKernel] = None, friction: Optional[str] = None):
    if friction is not None:
        num.friction_model = friction
    fits = None
    used_explicit = False
    if num.friction_model in ("zvb_rec", "zvb_direct"):
        if frozen is None:
            raise RuntimeError("ZVB reference requires an explicit frozen kernel")
        fits = frozen.as_fits()
        used_explicit = True
    t0 = time.perf_counter()
    res = ms.simulate(well, num, kernel_fits=fits)
    wall = time.perf_counter() - t0
    res.meta["kernel_fits_explicit"] = used_explicit
    res.meta["used_online_refit"] = False
    if used_explicit:
        res.meta["kernel_yaml_sha256"] = frozen.yaml_sha256
        if res.meta.get("seg_kernel") is None:
            raise RuntimeError("ZVB reference missing seg_kernel; refuse fit-kernel silence")
    return res, wall


def event_window_ok(well, t, p, physical_scale, kind: str) -> dict:
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    pert = p - p[0]
    silent = classify_silence(pert, physical_scale)
    if kind == "short":
        t_need = short_window_seconds(well)
    else:
        t_need = long_window_seconds(well)
    complete = bool(t.size and t[-1] + 1e-12 >= t_need)
    t_valve_end = well.t_s + well.t_c
    t_refl = well.t_s + well.t_c + 2.0 * float(well.x_clusters[0]) / well.a
    after_valve = bool(t[-1] >= t_valve_end)
    after_refl = bool(t[-1] >= t_refl)
    event = (not silent) and after_valve and after_refl
    return {
        "silent": silent,
        "window_kind": kind,
        "t_need": t_need,
        "t_end": float(t[-1]) if t.size else None,
        "complete": complete,
        "after_valve": after_valve,
        "after_first_reflection": after_refl,
        "event_occurred": event,
        "physical_scale": physical_scale,
        "pert_l2_norm": float(np.linalg.norm(pert)),
    }


def compare_three(ref, twin: TwinResult, prod_p, prod_state, params, kind: str) -> dict:
    scale = joukowsky_scale(params)
    well = case_to_well(params)
    ev = event_window_ok(well, twin.t, twin.p_head, scale, kind)
    t = twin.t
    if not np.allclose(t, ref.t):
        raise RuntimeError("time axes differ between reference and twin")
    if prod_p.size != t.size:
        raise RuntimeError("production time length differs")
    wh = {
        "ref_vs_twin": pert_l2(twin.p_head, ref.p_head),
        "prod_vs_twin": pert_l2(prod_p, twin.p_head),
        "prod_vs_ref": pert_l2(prod_p, ref.p_head),
    }
    node = {
        "H_left_rel": rel_l2(prod_state.get("H_left"), twin.H_left) if "H_left" in prod_state else None,
        "q_rel": rel_l2(prod_state.get("q_hist"), twin.q_hist) if "q_hist" in prod_state else None,
        "pc_rel": rel_l2(prod_state.get("pc_hist"), twin.pc_hist) if "pc_hist" in prod_state else None,
        "F_rel": rel_l2(prod_state.get("F_hist"), twin.F_hist) if "F_hist" in prod_state else None,
        "Q_left_rel": rel_l2(prod_state.get("Q_left"), twin.Q_left) if "Q_left" in prod_state else None,
        "Q_right_rel": rel_l2(prod_state.get("Q_right"), twin.Q_right) if "Q_right" in prod_state else None,
    }
    return {
        "event": ev,
        "wellhead_pert_l2": wh,
        "cluster_rels": node,
        "newton_calls_twin": twin.newton_calls,
        "newton_called": twin.newton_calls > 0,
        "n_fail_ref": int(ref.n_fail),
        "n_fail_twin": twin.n_fail,
        "resid_max_twin": twin.resid_max,
        "full_state_exported": bool(twin.full_state_exported and twin.q_hist.shape[1] == twin.perf_off[-1]),
        "n_perf_archived": int(twin.perf_off[-1]),
        "n_clusters": int(twin.perf_off.size - 1),
    }


def align_to_query(t_src, y_src, t_q):
    return np.interp(np.asarray(t_q, dtype=np.float64),
                     np.asarray(t_src, dtype=np.float64),
                     np.asarray(y_src, dtype=np.float64))
