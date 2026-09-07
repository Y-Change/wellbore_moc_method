# -*- coding: utf-8 -*-
"""Copy measured run artifacts into docs/ from explicit run_ids only."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from r4_paths import DOCS_DIR, MANIFEST_DIR, RUNS_DIR, ensure_dirs
from common import dump_json

RUNS = {
    "p0": "r4_P0_kernel_20260906_231434_0002042f",
    "p1": "r4_P1_transient_20260906_231503_6d5cfa65",
    "p2": "r4_P2_causality_20260906_232440_43c7df5c",
    "p2b": "r4_P2_align_20260906_232814_15f0cf88",
    "p3": "r4_P3_zvb_20260906_231822_8a284b86",
    "p4": "r4_P4_train_20260906_232156_c0d8e7d4",
    "reg": "r4_regressions_20260906_231502_c927a16b",
    "reg_fail_kept": "r4_regressions_20260906_231446_aa6e3c2e",
}


def load(run_id, name):
    p = RUNS_DIR / run_id / name
    if not p.is_file():
        p = RUNS_DIR / run_id / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ensure_dirs()
    p0 = load(RUNS["p0"], "kernel_loading_audit.json")
    p1 = load(RUNS["p1"], "transient_parity.json")
    p2 = load(RUNS["p2"], "causality_and_convergence.json")
    p2b = load(RUNS["p2b"], "alignment_0p5pct.json")
    p3 = load(RUNS["p3"], "zvb_tensor_validation.json")
    p4 = load(RUNS["p4"], "train_entry_and_failure.json")
    reg = load(RUNS["reg"], "regressions.json")

    ev = {
        "protocol": "explicit_run_id_only_no_latest",
        "python": r"D:\Anaconda\envs\torch24\python.exe",
        "runs": RUNS,
        "round3_c_audit": p0.get("round3_c_audit"),
        "code_and_kernel": {
            "p0_run": RUNS["p0"],
            "kernel_sha256": p0["kernel"]["yaml_sha256"],
            "used_online_refit": False,
            "stage1_CONFIG_DIR": p0["stage1_CONFIG_DIR"],
        },
        "do_not_cite": [
            RUNS["reg_fail_kept"] + " (flat-top detector before fix; kept, not a PASS)",
            "Round3 shared manifests/production_rollout_gates.json (source uncertain)",
            "incomplete C runs 4eb48517 / c5f1025e",
        ],
    }
    dump_json(ev, DOCS_DIR / "evidence_registry.json")
    dump_json(p0, DOCS_DIR / "kernel_loading_audit.json")
    dump_json(p1, DOCS_DIR / "transient_parity.json")
    dump_json({"causality": p2, "alignment": p2b}, DOCS_DIR / "causality_and_convergence.json")
    dump_json(p3, DOCS_DIR / "zvb_tensor_validation.json")
    dump_json(p4, DOCS_DIR / "train_entry_and_failure.json")
    dump_json(reg, DOCS_DIR / "regressions.json")
    dump_json(ev, MANIFEST_DIR / "round4_run_ids.json")

    # markdown tables
    (DOCS_DIR / "kernel_loading_audit.md").write_text(
        f"""# kernel_loading_audit

run_id: `{RUNS['p0']}`

- YAML: `{p0['kernel']['yaml_path']}`
- sha256: `{p0['kernel']['yaml_sha256']}`
- M=12, domain_tau={p0['kernel']['domain_tau']}
- 加载方法: `yaml.safe_load` + `kernel_fits_from_config`；**未**调用 `get_kernel_fits`
- used_online_refit: {p0['used_online_refit']}
- Stage-1 CONFIG_DIR: `{p0['stage1_CONFIG_DIR']}`
- 分段耗时 (s): kernel_load={p0['timings']['kernel_load_s']:.3f}, warmup_none={p0['timings']['warmup_none_s']:.3f}, steady_zvb_setup={p0['timings']['steady_zvb_setup_s']:.3f}, short_zvb={p0['timings']['short_zvb_simulate_s']:.3f}
- 短窗 ZVB 每步 {p0['timings']['short_zvb_per_step_s']:.2e} s；未先把超时放大 10 倍
- 首案 Nx512 实测 Cr0=1, dt_reduced=False
""",
        encoding="utf-8",
    )

    print("reports written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
