# -*- coding: utf-8 -*-
"""P0: Round-3 C evidence freeze, frozen kernel load, bounded warmup."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import load_train_cases
from common import RunLogger, dump_json, sha256_file
from kernel_loader import load_frozen_kernel
from moc_twin import grid_inventory, make_num, short_window_seconds
from physics import simulate_ref
from r4_paths import FROZEN_KERNEL_YAML, ROUND3, TRAIN4
from stage1_bridge import case_to_well, stage1_paths


def audit_round3_c() -> dict:
    runs = [
        {
            "run_id": "r3_C_gates_20260906_223058_4eb48517",
            "code_sha256_prefix": "358e4a6a",
            "created_local": "2026-09-06T22:30:58",
            "terminal_id": "538392",
            "terminal_status": "failed",
            "terminal_exit": 4294967295,
            "terminal_elapsed_ms": 259492,
            "terminal_ended_utc": "2026-09-06T14:35:13.564Z",
            "run_meta_wall_s": 1291.396,
            "stdout_last_stamp": "22:52:03",
            "stdout_content": "twin+ref none/darcy for 4 cases; no refine/C-done line",
            "metrics_present": True,
            "metrics_fields": ["n", "alignment_all_pass", "grad_status", "newton_blocked", "run_id", "wall_clock_s"],
            "stop_classification": "CONFLICTING_EVIDENCE",
            "stop_note": (
                "观察时终端记为 killed（259 s）。run 目录 stdout 时间戳延续到 22:52，"
                "metrics.wall=1291 s。不能唯一判定‘当时已死’还是‘kill 未生效后继续写完 twin’。"
                "不得把该 metrics 的 alignment_all_pass 当作完整 C 门完成。"
            ),
        },
        {
            "run_id": "r3_C_gates_20260906_223531_c5f1025e",
            "code_sha256_prefix": "82890a82",
            "created_local": "2026-09-06T22:35:31",
            "terminal_status": "observed_killed_then_files_continued",
            "run_meta_wall_s": 1010.163,
            "stdout_content": "refine + twin + zvb_rec nx=64 started; 16 min gap on first ZVB then remaining cases",
            "metrics_present": True,
            "metrics_fields": ["n", "alignment_all_pass", "grad_status", "newton_blocked"],
            "stop_classification": "TIMEOUT_OR_KILLED_THEN_PARTIAL_CONTINUE",
            "stop_note": (
                "观察时仍在首案长窗 ZVB nx=64。终端曾记 killed。"
                "stdout 显示 22:35:35 启动 zvb_rec，22:51:43 才打出 twin 汇总。"
                "metrics 无 cases 明细。不得引用为完整 PASS。"
            ),
        },
        {
            "run_id": "r3_C_gates_20260906_223918_f538d6f9",
            "code_sha256_prefix": "542f26d4",
            "created_local": "2026-09-06T22:39:18",
            "terminal_status": "succeeded",
            "terminal_elapsed_ms": 225961,
            "run_meta_wall_s": 224.575,
            "stdout_content": "C done; skipped full long-window Tensor ZVB",
            "metrics_present": True,
            "stop_classification": "COMPLETED",
            "stop_note": "唯一正常结束的 C-gates 进程。",
        },
    ]
    shared = ROUND3 / "manifests" / "production_rollout_gates.json"
    shared_info = {
        "path": str(shared),
        "exists": shared.is_file(),
        "source": "UNCERTAIN",
        "note": (
            "该 JSON 被多个 C 进程与事后手工补丁写过。"
            "same_grid_discrete_pass / continuum_0p5pct_gate 等字段来源不确定，Round4 不猜测补全，"
            "不以 latest 或该共享文件作为过门依据。"
        ),
    }
    if shared.is_file():
        shared_info["sha256"] = sha256_file(shared)
        blob = json.loads(shared.read_text(encoding="utf-8"))
        shared_info["keys"] = sorted(blob.keys())
        shared_info["case_ids"] = blob.get("case_ids")
    fixup = ROUND3 / "manifests" / "production_rollout_fixups.json"
    return {
        "c_runs": runs,
        "shared_manifest": shared_info,
        "fixups_manifest": {
            "path": str(fixup),
            "exists": fixup.is_file(),
            "sha256": sha256_file(fixup) if fixup.is_file() else None,
            "note": "frozen-IC causality FAIL on nx=32 kept; 15-step torch compare is pre-event and not a transient PASS",
        },
        "protocol": "do_not_cite_incomplete_or_shared_JSON_as_PASS",
    }


def main():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "budgets.yaml").read_text(encoding="utf-8"))
    logger = RunLogger("r4_P0_kernel", {
        "stage": "P0",
        "budgets_s": cfg["budgets_s"],
        "case_ids": list(TRAIN4),
        "frozen_yaml": str(FROZEN_KERNEL_YAML),
    })
    t_all = time.perf_counter()
    ev = audit_round3_c()
    dump_json(ev, logger.dir / "round3_c_audit.json")
    logger.log("Round3 C audit written")

    t0 = time.perf_counter()
    frozen = load_frozen_kernel()
    t_load = time.perf_counter() - t0
    logger.log(f"kernel load {t_load:.3f}s sha={frozen.yaml_sha256[:12]} M={frozen.M} refit={frozen.used_online_refit}")
    if frozen.used_online_refit:
        raise RuntimeError("online refit")
    if "pilot_round" in str(stage1_paths.CONFIG_DIR):
        raise RuntimeError("Stage-1 CONFIG_DIR shadowed")

    ids, params, rows = load_train_cases(4)
    well0 = case_to_well(params[0])
    Tshort = short_window_seconds(well0)
    inv = {nx: grid_inventory(well0, nx) for nx in (64, 128, 256, 512)}
    dump_json({"case_id": ids[0], "grids": inv}, logger.dir / "grid_probe_case0.json")
    logger.log(f"case0 Cr0 @512={inv[512]['Cr0']} dt_reduced={inv[512]['dt_reduced']}")

    # JIT / first simulate warmup: none then short ZVB, confirm not fit-kernel
    times = {"kernel_load_s": t_load}
    num_n = make_num(64, T=min(0.25, Tshort), friction="none")
    t0 = time.perf_counter()
    simulate_ref(well0, num_n)
    times["warmup_none_s"] = time.perf_counter() - t0
    logger.log(f"warmup none {times['warmup_none_s']:.3f}s")

    t0 = time.perf_counter()
    ss_num = make_num(64, T=0.05, friction="zvb_rec")
    from moc_twin import setup
    S = setup(well0, ss_num, "zvb_rec", frozen=frozen)
    times["steady_zvb_setup_s"] = time.perf_counter() - t0
    logger.log(f"steady+kernel setup {times['steady_zvb_setup_s']:.3f}s kernels={S['kernel_audit']['seg_kernel']}")

    t0 = time.perf_counter()
    num_z = make_num(64, T=min(0.4, Tshort), friction="zvb_rec")
    refz, wallz = simulate_ref(well0, num_z, frozen=frozen)
    times["short_zvb_simulate_s"] = wallz
    times["short_zvb_n_steps"] = int(refz.meta["n_steps"])
    times["short_zvb_dt"] = float(refz.meta["dt"])
    times["short_zvb_per_step_s"] = wallz / max(int(refz.meta["n_steps"]), 1)
    if refz.meta.get("friction_model") != "zvb_rec":
        raise RuntimeError("not on zvb_rec")
    if not refz.meta.get("kernel_fits_explicit"):
        raise RuntimeError("simulate did not receive explicit kernel_fits")
    logger.log(f"short ZVB {wallz:.3f}s steps={refz.meta['n_steps']} per_step={times['short_zvb_per_step_s']:.4f}s")

    # estimate long-window cost, do not 10x yet
    Tlong = 25.87
    n_est = int(Tlong / max(float(refz.meta["dt"]), 1e-9))
    est_long_64 = n_est * times["short_zvb_per_step_s"]
    times["est_long_zvb_nx64_s"] = est_long_64
    times["do_not_multiply_timeout_by_10"] = True

    metrics = {
        "status": "PASS",
        "kernel": frozen.audit,
        "timings": times,
        "stage1_CONFIG_DIR": str(stage1_paths.CONFIG_DIR),
        "used_online_refit": False,
        "fit_kernel_branch": False,
        "case_ids": ids,
        "source_sha256": [r["source_sha256"] for r in rows],
        "round3_c_audit": ev,
        "grid_probe_case0": inv,
        "wall_s": time.perf_counter() - t_all,
    }
    logger.write_metrics(metrics)
    dump_json(metrics, logger.dir / "kernel_loading_audit.json")
    logger.close("completed")
    print("P0 done", logger.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
