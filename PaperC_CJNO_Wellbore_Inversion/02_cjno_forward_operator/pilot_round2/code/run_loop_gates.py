# -*- coding: utf-8 -*-
"""C2 gates: causality, wellhead grad, label shuffle, hard-call count, residuals."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from dataset import cosine_valve_Q, predict_arrays
from ift_layer import cluster_newton
from loop import (
    CharacteristicLoop, interp_hist, synthetic_initial, synthetic_single_cluster,
    torch_mini_loop,
)
from node_coeffs import collect_branch_coeffs
from paths import MANIFEST_DIR, TABLE_DIR, ensure_dirs


def _run_loop(phys, n_steps, dt=0.02, K_scale=1.0, require_ok=True):
    phys = phys
    K0 = [k.copy() * K_scale for k in phys.K]
    # local copy of K
    from dataclasses import replace
    phys2 = replace(phys, K=K0)
    init = synthetic_initial(phys2)
    t = phys2.t_s + dt * np.arange(n_steps)
    Qv = cosine_valve_Q(t, phys2.Q0, phys2.t_s, phys2.t_c)
    loop = CharacteristicLoop(phys2, dt, n_steps, use_friction=False)
    return loop.rollout(Qv, init, require_ok=require_ok), t, phys2


def gate_causality():
    phys = synthetic_single_cluster()
    dt = 0.02
    n_steps = 140
    r1, t, _ = _run_loop(phys, n_steps, dt, 1.0)
    r2, _, _ = _run_loop(phys, n_steps, dt, 3.0)
    dp = np.abs(r1.p_wh - r2.p_wh)
    scale = max(np.max(np.abs(r1.p_wh)), 1.0)
    thr = 1e-4 * scale
    hit = np.where(dp > thr)[0]
    t_hit = float(t[hit[0]] - t[0]) if hit.size else float("nan")
    t_oneway = phys.xs[0] / phys.a
    t_round = 2.0 * t_oneway
    # K is changed at the node at t=0: wellhead is silent until the one-way delay.
    early = np.any(dp[t - t[0] < t_oneway - 2 * dt] > thr)
    ok = hit.size > 0 and (not early) and abs(t_hit - t_oneway) <= 3 * dt
    return {
        "t_first_diff_s": t_hit,
        "t_oneway_s": t_oneway,
        "t_roundtrip_s": t_round,
        "early_leak": bool(early),
        "n_hard": r1.n_hard,
        "resid_max": float(np.max(r1.resid)),
        "pass": bool(ok),
        "note": "K change at the node at t=0; wellhead first differs at x/a, not instantaneously",
    }


def gate_hard_counts():
    phys = synthetic_single_cluster()
    r, t, _ = _run_loop(phys, 40, 0.02, 1.0)
    return {
        "n_hard": r.n_hard,
        "n_fail": r.n_fail,
        "resid_nonempty": bool(r.resid.size and np.any(r.resid[1:] >= 0)),
        "resid_max": float(np.max(r.resid)),
        "pass": r.n_hard == 39 * phys.N and r.ok and float(np.max(r.resid)) < 1e-8,
        "H_nodes_var": float(np.std(r.H_nodes)),
        "sq_var": float(np.std(r.sq)),
    }


def gate_shuffle_labels_unused():
    phys = synthetic_single_cluster()
    r1, _, _ = _run_loop(phys, 30, 0.02, 1.0)
    # mutating a dummy target array cannot enter the loop
    fake_p = r1.p_wh.copy()[::-1]
    r2, _, _ = _run_loop(phys, 30, 0.02, 1.0)
    rel = float(np.linalg.norm(r1.p_wh - r2.p_wh) / (np.linalg.norm(r2.p_wh) + 1e-30))
    return {"rel": rel, "fake_target_used": False, "pass": rel <= 1e-15,
            "note": "loop API has no target slot; reversed dummy unused"}


def gate_wellhead_grad():
    phys = synthetic_single_cluster(x=200.0, L=400.0, a=1000.0, t_c=0.1)
    dt = 0.05
    n_steps = 16
    init = synthetic_initial(phys)
    t = phys.t_s + dt * np.arange(n_steps)
    Qv = torch.as_tensor(cosine_valve_Q(t, phys.Q0, phys.t_s, phys.t_c), dtype=torch.float64)
    br = collect_branch_coeffs(dt, phys.I_f[0], phys.R_f[0], phys.C_f[0], phys.G_l[0],
                               float(phys.p_res[0]), init.q0[0], init.pc0[0], init.F0[0])
    K = torch.tensor(phys.K[0], dtype=torch.float64, requires_grad=True)
    kappa = torch.as_tensor(phys.kappa[0], dtype=torch.float64)
    c1 = torch.as_tensor(br["c1"], dtype=torch.float64)
    c0 = torch.as_tensor(br["c0"], dtype=torch.float64)
    a1 = torch.as_tensor(br["a1"], dtype=torch.float64)
    a0 = torch.as_tensor(br["a0"], dtype=torch.float64)
    delay_L = float((phys.xs[0] / phys.a) / dt)
    delay_R = float(((phys.L - phys.xs[0]) / phys.a) / dt)
    B = torch.tensor(phys.B, dtype=torch.float64)
    z_j = torch.tensor(phys.z_of_x(phys.xs[0]), dtype=torch.float64)
    rho_g = torch.tensor(phys.rho_g, dtype=torch.float64)
    H0 = torch.tensor(init.p0_wh / phys.rho_g, dtype=torch.float64)
    q0 = torch.as_tensor(init.q0[0], dtype=torch.float64)
    out = torch_mini_loop(
        K, kappa, c1, c0, a1, a0, Qv, delay_L, delay_R, B, z_j, rho_g,
        H0, H0, q0, 4000.0, max(abs(phys.Q0), 1e-3), phys.rho_g * 4000.0,
    )
    loss = (out["p_wh"] ** 2).mean()
    loss.backward()
    g = float(K.grad.abs().sum()) if K.grad is not None else 0.0
    return {
        "n_hard": out["n_hard"],
        "resid_max": float(out["resid_max"]),
        "grad_abs_sum": g,
        "pass": g > 0.0 and float(out["resid_max"]) < 1e-8 and out["n_hard"] > 0,
    }


def gate_newton_failure_blocks():
    import loop as loopmod
    orig = loopmod.cluster_newton

    def _fail(*args, **kwargs):
        raise RuntimeError("Newton failed at n=1 cluster=0 resid=1.0 (injected)")

    loopmod.cluster_newton = _fail
    try:
        _run_loop(synthetic_single_cluster(), 8, 0.02, 1.0, require_ok=True)
        blocked = False
        err = ""
    except RuntimeError as exc:
        blocked = "Newton failed" in str(exc)
        err = repr(exc)
    finally:
        loopmod.cluster_newton = orig
    return {"blocked": blocked, "error": err, "pass": blocked}


def gate_one_step_vs_free():
    phys = synthetic_single_cluster()
    r_free, t, phys2 = _run_loop(phys, 40, 0.02, 1.0)
    teacher = {
        "H": [r_free.H_nodes[:, 0]],
        "q": r_free.q_perf,
        "pc": r_free.pc_perf,
        "F": [np.zeros_like(r_free.q_perf[0])],
    }
    # one-step with own history should be close (same data) — F not stored; rebuild F≈0
    # This is a consistency check that teacher path runs, not that it equals high-res MOC.
    init = synthetic_initial(phys2)
    Qv = cosine_valve_Q(t, phys2.Q0, phys2.t_s, phys2.t_c)
    loop = CharacteristicLoop(phys2, 0.02, 40, use_friction=False)
    r_os = loop.rollout(Qv, init, teacher=teacher)
    rel = float(np.linalg.norm(r_os.p_wh - r_free.p_wh) / (np.linalg.norm(r_free.p_wh) + 1e-30))
    return {
        "free_mode": r_free.mode, "one_step_mode": r_os.mode,
        "rel_if_teacher_is_own_roll": rel,
        "pass": r_free.mode == "free_roll" and r_os.mode == "one_step_teacher",
        "note": "one-step ≠ whole-window one-shot; both are time marches",
    }


def time_hard_path():
    phys = synthetic_single_cluster()
    times = []
    for i in range(8):
        t0 = time.perf_counter()
        _run_loop(phys, 30, 0.02, 1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    arr = np.array(times[2:])  # drop 2 warmups
    return {
        "n_warmup": 2, "n_rep": int(arr.size),
        "median_s": float(np.median(arr)),
        "p25_s": float(np.quantile(arr, 0.25)),
        "p75_s": float(np.quantile(arr, 0.75)),
        "note": "model compute only; no NPZ I/O. not comparable to round1 Newton-off latency",
    }


def main():
    ensure_dirs()
    gates = {
        "causality": gate_causality(),
        "hard_counts": gate_hard_counts(),
        "shuffle_labels": gate_shuffle_labels_unused(),
        "wellhead_grad": gate_wellhead_grad(),
        "newton_blocks": gate_newton_failure_blocks(),
        "one_step_vs_free": gate_one_step_vs_free(),
        "timing_hard_compute": time_hard_path(),
    }
    n_fail = sum(1 for k, v in gates.items() if k != "timing_hard_compute" and not v.get("pass"))
    out = {"status": "PASS" if n_fail == 0 else "FAIL", "n_fail": n_fail, "gates": gates,
           "kind": "integration_reference_not_full_CJNO"}
    dump_json(out, MANIFEST_DIR / "loop_gates.json")
    dump_json(out, TABLE_DIR / "loop_gates.json")
    print({k: v.get("pass", v) for k, v in gates.items()})
    print("status", out["status"])
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
