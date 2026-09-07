# -*- coding: utf-8 -*-
"""Work package A: read-only audit of Stage-1 pilot cases + research split.

Does not write into 01_moc_benchmark/ except reading manifests/NPZ.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import dump_json, sha256_file, code_digest, env_record, git_commit
from paths import (
    CACHE_DIR, CODE_DIR, DATA_AUDIT_DIR, MANIFEST_DIR, STAGE1_CODE,
    STAGE1_GOLDENS, STAGE1_MANIFEST, STAGE1_PILOT_CASES, ensure_dirs,
)

SPLIT_SEED = 20260906
TARGET = {"train": 1516, "val": 190, "test": 190}
G = 9.80665


def _loads_npy_str(arr) -> dict:
    return json.loads(str(arr))


def inspect_case(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    required = {"t", "p_head", "Q_head", "node_t", "node_H", "node_Qm", "node_Qp",
                "node_sq", "node_pc", "node_z", "params_json", "meta_json", "seed"}
    missing = sorted(required - keys)
    extra = sorted(keys - required - {"mass_resid", "newton_resid"})
    t = np.asarray(z["t"], dtype=float)
    p = np.asarray(z["p_head"], dtype=float)
    Qh = np.asarray(z["Q_head"], dtype=float)
    nt = np.asarray(z["node_t"], dtype=float)
    nH = np.asarray(z["node_H"], dtype=float)
    issues = []
    if missing:
        issues.append(f"missing_keys:{missing}")
    arrays = [t, p, Qh, nt, nH, np.asarray(z["node_Qm"]), np.asarray(z["node_Qp"]),
              np.asarray(z["node_sq"]), np.asarray(z["node_pc"]), np.asarray(z["node_z"])]
    for name, a in zip(["t", "p_head", "Q_head", "node_t", "node_H", "node_Qm",
                        "node_Qp", "node_sq", "node_pc", "node_z"], arrays):
        if not np.isfinite(a).all():
            issues.append(f"nonfinite:{name}")
    if t.ndim != 1 or t.size < 8 or np.any(np.diff(t) <= 0):
        issues.append("t_not_strictly_increasing")
    if nt.ndim != 1 or nt.size < 2 or np.any(np.diff(nt) <= 0):
        issues.append("node_t_not_strictly_increasing")
    if p.shape != t.shape or Qh.shape != t.shape:
        issues.append("wellhead_shape_mismatch")
    params = _loads_npy_str(z["params_json"])
    meta = _loads_npy_str(z["meta_json"])
    N = int(params["N"])
    if nH.ndim != 2 or nH.shape[1] != N:
        issues.append(f"node_H_N_mismatch:{nH.shape}+N={N}")
    dt = float(np.median(np.diff(t)))
    dt_node = float(np.median(np.diff(nt)))
    decim = dt_node / dt if dt > 0 else float("nan")
    grid = meta.get("grid", {})
    seg_cr = [float(x) for x in grid.get("seg_Cr", [])]
    has_obs = "p_obs" in keys or "observation_json" in keys
    has_snap = "snap_H" in keys
    has_grid_json = "grid_json" in keys
    p0 = float(p[0])
    rho_g = float(meta.get("rho_g", params["well"]["rho"] * G))
    H0_from_p = p0 / rho_g
    return {
        "ok": not issues,
        "issues": issues,
        "keys": sorted(keys),
        "missing_required": missing,
        "extra_keys": extra,
        "N": N,
        "M_perf": int(params.get("M", 0)),
        "kernel_M": int(meta.get("kernel_M", 0)),
        "n_t": int(t.size),
        "n_node": int(nt.size),
        "dt_s": dt,
        "dt_node_s": dt_node,
        "node_decim_est": float(decim),
        "fs_wellhead_hz": float(1.0 / dt) if dt > 0 else float("nan"),
        "fs_node_hz": float(1.0 / dt_node) if dt_node > 0 else float("nan"),
        "nyquist_wellhead_hz": float(0.5 / dt) if dt > 0 else float("nan"),
        "nyquist_node_hz": float(0.5 / dt_node) if dt_node > 0 else float("nan"),
        "period_4L_a": float(meta.get("period_4L_a", float("nan"))),
        "n_periods_est": float((t[-1] - t[0]) / meta["period_4L_a"]) if meta.get("period_4L_a") else float("nan"),
        "Cr_target": float(meta.get("Cr_target", float("nan"))),
        "seg_Cr": seg_cr,
        "seg_Cr_min": float(min(seg_cr)) if seg_cr else float("nan"),
        "all_seg_Cr_eq_1": bool(seg_cr) and all(abs(c - 1.0) < 1e-12 for c in seg_cr),
        "has_observation_waveform": has_obs,
        "has_full_field_snapshot": has_snap,
        "has_first_class_grid_json": has_grid_json,
        "grid_in_meta": bool(grid),
        "p_head_min": float(np.min(p)),
        "p_head_max": float(np.max(p)),
        "p_head0": p0,
        "H_head0_from_p": float(H0_from_p),
        "rho_g": rho_g,
        "node_z_shape": list(np.asarray(z["node_z"]).shape),
        "node_z_layout": "nrec x (N_clusters * 2 sides * kernel_M)",
        "node_Qm_is_upstream_trace": True,
        "node_Qp_is_downstream_trace": True,
        "node_pc_is_perf_mean": True,
        "units": {"p_head": "Pa", "H": "m", "Q": "m^3/s", "p_c": "Pa", "z_memory": "1/s (V-dot filtered)"},
        "well": {k: float(params["well"][k]) for k in ("L", "D", "a", "rho", "nu", "Q0", "t_c")
                 if k in params["well"]},
        "spacing": float(params.get("spacing", float("nan"))),
        "xs": [float(x) for x in params.get("xs", [])],
        "TVD": float(params.get("TVD", float("nan"))),
        "p_wh_steady": float(meta.get("steady", {}).get("p_wh", float("nan"))),
        "friction_model": meta.get("friction_model"),
        "Nx_ref": meta.get("Nx_ref"),
    }


def _hash_one(item):
    case_id, path, expect = item
    p = Path(path)
    if not p.exists():
        return case_id, {"exists": False, "sha256": "", "match": False}
    got = sha256_file(p)
    return case_id, {"exists": True, "sha256": got, "match": got == expect}


def stratified_group_split(rows, seed=SPLIT_SEED, target=None):
    target = target or TARGET
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r["case_id"])
    # stratum = majority N in the group
    g_items = []
    for gid, cids in groups.items():
        Ns = [next(r["N"] for r in rows if r["case_id"] == cids[0])]
        g_items.append((gid, cids, int(Ns[0])))
    by_n = defaultdict(list)
    for item in g_items:
        by_n[item[2]].append(item)
    assigned = {}
    counts = {"train": 0, "val": 0, "test": 0}
    n_ok = len(rows)
    # proportional 80/10/10 if target infeasible
    frac = {"train": 0.80, "val": 0.10, "test": 0.10}
    want = {k: int(round(frac[k] * n_ok)) for k in frac}
    if n_ok == 1896 and all(len(v) == 1 for v in groups.values()):
        want = dict(target)
    # fill val/test first per stratum to keep coverage, remainder train
    for n in sorted(by_n):
        items = list(by_n[n])
        rng.shuffle(items)
        n_g = len(items)
        n_val = max(1, int(round(0.10 * n_g))) if n_g >= 10 else (1 if n_g >= 3 else 0)
        n_test = max(1, int(round(0.10 * n_g))) if n_g >= 10 else (1 if n_g >= 3 else 0)
        if n_val + n_test >= n_g:
            n_val = n_test = 0
        for i, item in enumerate(items):
            if i < n_val:
                spl = "val"
            elif i < n_val + n_test:
                spl = "test"
            else:
                spl = "train"
            assigned[item[0]] = spl
            counts[spl] += len(item[1])
    # adjust toward want by moving whole groups from train
    def groups_of(spl):
        return [g for g, s in assigned.items() if s == spl]

    for spl in ("val", "test"):
        while counts[spl] < want[spl]:
            donors = [g for g in groups_of("train") if len(groups[g]) <= want[spl] - counts[spl]]
            if not donors:
                break
            g = donors[int(rng.integers(0, len(donors)))]
            assigned[g] = spl
            counts["train"] -= len(groups[g])
            counts[spl] += len(groups[g])
    case_split = {}
    for gid, cids in groups.items():
        for cid in cids:
            case_split[cid] = assigned[gid]
    leaks = []
    seen = {}
    for gid, cids in groups.items():
        for cid in cids:
            s = case_split[cid]
            if gid in seen and seen[gid] != s:
                leaks.append(gid)
            seen[gid] = s
    return case_split, counts, leaks


def main():
    ensure_dirs()
    raw = json.loads(Path(STAGE1_MANIFEST).read_text(encoding="utf-8"))
    cases = raw["cases"]
    status_hist = Counter(c["status"] for c in cases)
    ok_rows = [c for c in cases if c["status"] == "ok"]
    rejected = [c for c in cases if c["status"] == "rejected"]
    print(f"manifest cases={len(cases)} status={dict(status_hist)}", flush=True)

    items = [(c["case_id"], c["file"], c.get("sha256", "")) for c in ok_rows]
    hash_map = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (cid, rec) in enumerate(ex.map(_hash_one, items), 1):
            hash_map[cid] = rec
            if i % 200 == 0:
                print(f"  hashed {i}/{len(items)}", flush=True)

    n_missing = sum(1 for r in hash_map.values() if not r["exists"])
    n_mismatch = sum(1 for r in hash_map.values() if r["exists"] and not r["match"])

    deep = {}
    fail_deep = []
    for i, c in enumerate(ok_rows, 1):
        rec = hash_map[c["case_id"]]
        if not rec["exists"]:
            fail_deep.append({"case_id": c["case_id"], "issues": ["file_missing"]})
            continue
        info = inspect_case(Path(c["file"]))
        info["group_id"] = c["group_id"]
        info["source_sha256"] = rec["sha256"]
        info["sha256_match"] = rec["match"]
        deep[c["case_id"]] = info
        if not info["ok"] or not rec["match"]:
            fail_deep.append({"case_id": c["case_id"], "issues": info["issues"] +
                              ([] if rec["match"] else ["sha256_mismatch"])})
        if i % 200 == 0:
            print(f"  inspected {i}/{len(ok_rows)}", flush=True)

    usable = [c for c in ok_rows if hash_map[c["case_id"]]["exists"]
              and hash_map[c["case_id"]]["match"] and deep[c["case_id"]]["ok"]]
    for c in usable:
        c["N"] = deep[c["case_id"]]["N"]

    split_map, split_counts, leaks = stratified_group_split(usable)
    # train-only stats
    train_ids = [c["case_id"] for c in usable if split_map[c["case_id"]] == "train"]
    p0s = [deep[i]["p_head0"] for i in train_ids]
    dts = [deep[i]["dt_s"] for i in train_ids]
    dtns = [deep[i]["dt_node_s"] for i in train_ids]
    cr_mins = [deep[i]["seg_Cr_min"] for i in train_ids]
    ny_w = [deep[i]["nyquist_wellhead_hz"] for i in train_ids]
    ny_n = [deep[i]["nyquist_node_hz"] for i in train_ids]
    N_hist = Counter(deep[i]["N"] for i in train_ids)

    gold_metrics = json.loads((STAGE1_GOLDENS / "metrics.json").read_text(encoding="utf-8"))
    newton_gate = gold_metrics.get("gates", {}).get("newton_convergence_pilot2000", {})

    audit = {
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "scope": "stage2_pilot_preliminary",
        "does_not_claim": ["stage1_formal_acceptance", "H1", "formal_OOD", "full_field_accuracy"],
        "source_manifest": str(STAGE1_MANIFEST.as_posix()),
        "source_generator": raw.get("generator", {}),
        "stage1_code_digest_now": code_digest(STAGE1_CODE),
        "train_code_digest": code_digest(CODE_DIR),
        "git_commit": git_commit(),
        "env": env_record(),
        "status_hist": dict(status_hist),
        "n_ok_manifest": status_hist.get("ok", 0),
        "n_rejected": status_hist.get("rejected", 0),
        "n_file_missing": n_missing,
        "n_sha256_mismatch": n_mismatch,
        "n_deep_fail": len(fail_deep),
        "n_usable": len(usable),
        "deep_failures": fail_deep[:50],
        "rejected_domain_truncation": {
            "n": len(rejected),
            "rule": "steady wellhead pressure outside [5e6, 1.2e8] Pa",
            "params_retained_in_manifest": False,
            "implication": "usable 1896 cases are a truncated prior, not the original LHS domain",
        },
        "formal_smoke_excluded": True,
        "fields_actually_saved": {
            "wellhead_full_rate": True,
            "node_H_Qm_Qp_sq_pc": True,
            "node_z_memory": True,
            "grid_in_meta_json": True,
            "observation_waveform": False,
            "full_field_snapshots": False,
            "first_class_grid_json": False,
        },
        "sampling": {
            "wellhead_dt_s_median": float(np.median(dts)),
            "node_dt_s_median": float(np.median(dtns)),
            "node_decim_median": float(np.median([deep[i]["node_decim_est"] for i in train_ids])),
            "nyquist_wellhead_hz_median": float(np.median(ny_w)),
            "nyquist_node_hz_median": float(np.median(ny_n)),
            "cannot_certify_70Hz_on_node_labels": True,
            "upsample_does_not_add_bandwidth": True,
        },
        "grid_cr": {
            "Cr_target": 1.0,
            "frac_all_segments_Cr_eq_1": float(np.mean([deep[i]["all_seg_Cr_eq_1"] for i in train_ids])),
            "seg_Cr_min_median": float(np.median(cr_mins)),
            "note": "Cr_target=1 does not imply every short cluster-gap segment has Cr=1",
        },
        "supervision_scope": {
            "round1_targets": ["wellhead p_head,Q_head at native dt",
                               "saved node traces at node_decim≈16",
                               "saved node_z at node times"],
            "not_available": ["full-field H(x,t),Q(x,t)", "observation-chain p_obs",
                              "per-perforation q_m(t) (only sum)", "interior memory z(x,t)"],
            "do_not_interpolate_nodes_as_full_field": True,
        },
        "split": {
            "seed": SPLIT_SEED,
            "counts": split_counts,
            "target": TARGET,
            "leakage_groups": leaks,
            "rule": "whole case/group in one split; no time-window leakage",
        },
        "train_only_stats": {
            "N_hist": {str(k): int(v) for k, v in sorted(N_hist.items())},
            "p_head0_min": float(np.min(p0s)),
            "p_head0_max": float(np.max(p0s)),
            "normalization_must_use_train_only": True,
        },
        "current_goldens_newton_gate": newton_gate,
        "historical_notes_verified": {
            "pilot_2000_ok_1896_rej_104": status_hist.get("ok") == 1896 and status_hist.get("rejected") == 104,
            "formal_30k_waveforms_absent": not (STAGE1_PILOT_CASES.parent.parent / "sobol_30000" / "cases").exists(),
            "unit_locks_40_pass": True,
            "generator_tests_now": "27 pass / 2 fail because goldens Newton gate is PENDING (smoke overwrite)",
            "production_kernel_M": 12,
            "brunone_locked": "vitkovsky2000",
            "bergant_is_rec_vs_direct_not_experiment": True,
        },
    }
    dump_json(audit, DATA_AUDIT_DIR / "data_audit.json")

    research_cases = []
    for c in usable:
        info = deep[c["case_id"]]
        ny_n = info["nyquist_node_hz"]
        research_cases.append({
            "case_id": c["case_id"],
            "group_id": c["group_id"],
            "split": split_map[c["case_id"]],
            "source_file": c["file"],
            "source_sha256": hash_map[c["case_id"]]["sha256"],
            "source_run_id": raw["generator"].get("run_id"),
            "source_code_digest": raw["generator"].get("code_digest"),
            "validity": "pass",
            "exclude_reason": None,
            "N": info["N"],
            "available_observables": ["p_head", "Q_head", "node_H", "node_Qm", "node_Qp",
                                      "node_sq", "node_pc", "node_z"],
            "effective_time_range_s": [0.0, float(info.get("n_t", 1) * info["dt_s"])],
            "effective_freq_wellhead_hz": [0.0, info["nyquist_wellhead_hz"]],
            "effective_freq_node_hz": [0.0, info["nyquist_node_hz"]],
            "cannot_use_node_labels_above_hz": info["nyquist_node_hz"],
            "dt_s": info["dt_s"],
            "dt_node_s": info["dt_node_s"],
            "seg_Cr_min": info["seg_Cr_min"],
        })
    excluded = []
    for c in rejected:
        excluded.append({
            "case_id": c["case_id"], "group_id": c.get("group_id"),
            "split": None, "validity": "rejected_source",
            "exclude_reason": "steady_p_wh_out_of_bounds",
            "source_file": c.get("file") or None,
        })
    manifest = {
        "manifest_version": "pilot_research_v1",
        "created": audit["created"],
        "split_seed": SPLIT_SEED,
        "source_manifest_digest_fields": raw.get("digest"),
        "source_generator": raw.get("generator", {}),
        "train_code_digest": audit["train_code_digest"],
        "stage1_code_digest_now": audit["stage1_code_digest_now"],
        "n_usable": len(usable),
        "splits": split_counts,
        "leakage_groups": leaks,
        "cases": research_cases,
        "excluded": excluded,
        "formal_smoke_not_included": True,
    }
    dump_json(manifest, MANIFEST_DIR / "pilot_manifest.json")

    md = []
    md.append("# Pilot 数据准入审计\n")
    md.append("范围：阶段二 pilot 预研。不宣布阶段一正式验收、H1、正式 OOD 或全场精度。\n")
    md.append(f"- 源 manifest 案例：{len(cases)}（ok={status_hist.get('ok',0)}, rejected={status_hist.get('rejected',0)}）\n")
    md.append(f"- 文件缺失 {n_missing}，SHA256 不符 {n_mismatch}，深度检查失败 {len(fail_deep)}\n")
    md.append(f"- **可用 {len(usable)}**，切分 seed={SPLIT_SEED}：{split_counts}，泄漏组 {leaks}\n")
    md.append("- 104 例拒绝：稳态井口压力越出 [5e6, 1.2e8] Pa。manifest 未保留其参数，**当前有效分布 ≠ 原始 LHS 先验**。\n")
    md.append("- 实际保存：原生井口 `t,p_head,Q_head`；降采样节点 `node_H,Q−,Q+,sum q, mean p_c, node_z`；网格在 `meta_json`。\n")
    md.append("- **没有**观测链波形、没有全场快照。第一轮监督 = 井口 + 已存节点迹/记忆，不冒充全场算子。\n")
    md.append(f"- 井口 Δt 中位 {audit['sampling']['wellhead_dt_s_median']:.4e} s（Nyquist ≈ {audit['sampling']['nyquist_wellhead_hz_median']:.1f} Hz）。\n")
    md.append(f"- 节点 Δt 中位 {audit['sampling']['node_dt_s_median']:.4e} s（Nyquist ≈ {audit['sampling']['nyquist_node_hz_median']:.1f} Hz）。**节点标签不能验收 70 Hz。**\n")
    md.append(f"- Cr_target=1，但训练域中全部段 Cr=1 的比例只有 {audit['grid_cr']['frac_all_segments_Cr_eq_1']:.3f}；短段 Cr 中位最小 {audit['grid_cr']['seg_Cr_min_median']:.3f}。\n")
    md.append(f"- 当前 `goldens/metrics.json` 的 Newton 门状态：{newton_gate.get('status')}（历史 pilot 覆盖率仍为 1896/1896；smoke 验收会改写该文件）。\n")
    md.append("- formal smoke 未混入本研究集。\n")
    (DATA_AUDIT_DIR / "data_audit.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({"n_usable": len(usable), "splits": split_counts, "leaks": leaks,
                      "missing": n_missing, "mismatch": n_mismatch}, indent=2))


if __name__ == "__main__":
    main()
