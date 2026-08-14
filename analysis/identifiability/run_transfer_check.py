# -*- coding: utf-8 -*-
"""
run_transfer_check.py — 配对摩阻失配可迁移性（P3a 剪刀差检验）。

数据：lhs_dataset_2000（brunone）↔ lhs_dataset_2000_steady（同 case_id 同真值）。
方法：倒谱 / P0 字典剥离 / PhaseNet（可选）。

预测剪刀差：
  - P0、PhaseNet（brunone 隐含模型）在 steady 上变差；
  - 倒谱（纯回声隐含模型）在 steady 上变好。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.method_migration.run_peeling import peel_signal  # noqa: E402
from analysis.unified_evaluation.detection_protocol import (  # noqa: E402
    DetectorConfig,
    ScoringConfig,
    detect_peaks,
    score_detections,
)
from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum_1d  # noqa: E402

DEFAULT_OUT = os.path.join("output", "analysis", "identifiability", "transfer")
BRUNONE_ROOT = Path("output/lhs_dataset_2000")
STEADY_ROOT = Path("output/lhs_dataset_2000_steady")
DICT_PATH = Path("output/analysis/method_migration/dictionary/nominal/dictionary.npz")
A_WAVE = 1450.0
DET_CFG = DetectorConfig(
    min_separation_m=5.0,
    height_pct=85.0,
    height_rel=0.03,
    max_peaks=10,
    search_min_m=3400.0,
    search_max_m=4900.0,
)
SCORE_CFG = ScoringConfig(tolerance_m=10.0)


def _load_case(root: Path, case_id: int) -> Dict:
    z = np.load(root / "data" / f"case_{case_id:05d}.npz")
    return {
        "t": z["t"],
        "H_wh": z["H_wh"],
        "x_f": np.asarray(z["x_f"], dtype=float),
        "n_frac": int(z["n_frac"]),
    }


def _verify_pairing(n_check: int = 50) -> Dict:
    """抽样核对 brunone/steady 真值是否逐 case 一致。"""
    mismatches = []
    for i in range(n_check):
        a = _load_case(BRUNONE_ROOT, i)
        b = _load_case(STEADY_ROOT, i)
        if a["n_frac"] != b["n_frac"] or not np.allclose(a["x_f"], b["x_f"], atol=1e-6):
            mismatches.append(i)
    return {
        "n_check": n_check,
        "n_mismatch": len(mismatches),
        "paired": len(mismatches) == 0,
        "mismatch_ids": mismatches[:20],
    }


def _cepstrum_predict(t: np.ndarray, H: np.ndarray) -> List[float]:
    cep = compute_moc_cepstrum_1d(
        t, H, v=A_WAVE, ts=1.0, wellbore_length=5000.0, depth_min=3400.0,
    )
    depth = np.asarray(cep["depth"], dtype=np.float64)
    resp = np.asarray(cep["response"], dtype=np.float64)
    peaks = detect_peaks(depth, resp, DET_CFG)
    return [float(p["depth_m"]) for p in peaks]


def _p0_predict(H: np.ndarray, dic: Dict[str, np.ndarray]) -> List[float]:
    det = peel_signal(
        H, dic["intact"], dic["templates"], dic["depth_grid"], dic["cf_grid"],
    )
    return [float(d["depth_m"]) for d in det]


def _median_depth_error(pred: Sequence[float], true: Sequence[float]) -> float:
    if len(true) == 0:
        return float("nan")
    true = np.sort(np.asarray(true, dtype=float))
    pred = np.sort(np.asarray(pred, dtype=float))
    # 贪心一对一到 min(len)
    m = min(len(pred), len(true))
    if m == 0:
        return float("nan")
    # 对每个真值找最近预测
    errs = []
    used = set()
    for xt in true:
        if len(pred) == 0:
            break
        d = np.abs(pred - xt)
        j = int(np.argmin(d))
        if j in used and len(used) < len(pred):
            # 找未用最近
            order = np.argsort(d)
            j = next((int(k) for k in order if int(k) not in used), j)
        used.add(j)
        errs.append(abs(float(pred[j]) - float(xt)))
    return float(np.median(errs)) if errs else float("nan")


def run_classical(
    case_ids: Sequence[int],
    *,
    dic: Dict[str, np.ndarray],
) -> Dict[str, object]:
    methods = ("cepstrum", "p0")
    roots = {"brunone": BRUNONE_ROOT, "steady": STEADY_ROOT}
    per_case = []
    for cid in case_ids:
        row = {"case_id": int(cid)}
        truth = None
        for fric, root in roots.items():
            case = _load_case(root, cid)
            if truth is None:
                truth = case["x_f"]
                row["n_frac"] = case["n_frac"]
                row["x_f"] = case["x_f"].tolist()
            for method in methods:
                if method == "cepstrum":
                    pred = _cepstrum_predict(case["t"], case["H_wh"])
                else:
                    pred = _p0_predict(case["H_wh"], dic)
                sc = score_detections(pred, truth, SCORE_CFG.tolerance_m)
                row[f"{method}_{fric}_f1"] = sc["f1"]
                row[f"{method}_{fric}_precision"] = sc["precision"]
                row[f"{method}_{fric}_recall"] = sc["recall"]
                row[f"{method}_{fric}_med_err"] = _median_depth_error(pred, truth)
                row[f"{method}_{fric}_n_pred"] = len(pred)
        per_case.append(row)
        if (len(per_case) % 50) == 0:
            print(f"  classical progress {len(per_case)}/{len(case_ids)}")

    def agg(method: str, fric: str) -> Dict[str, float]:
        f1s = [r[f"{method}_{fric}_f1"] for r in per_case]
        errs = [r[f"{method}_{fric}_med_err"] for r in per_case]
        errs = [e for e in errs if np.isfinite(e)]
        return {
            "f1_mean": float(np.mean(f1s)),
            "f1_median": float(np.median(f1s)),
            "med_err_median": float(np.median(errs)) if errs else float("nan"),
            "n": len(per_case),
        }

    summary = {}
    for method in methods:
        summary[method] = {
            "brunone": agg(method, "brunone"),
            "steady": agg(method, "steady"),
        }
        # 剪刀差方向：F1_steady - F1_brunone；倒谱期望 >0，P0 期望 <0
        summary[method]["delta_f1"] = (
            summary[method]["steady"]["f1_mean"] - summary[method]["brunone"]["f1_mean"]
        )
        summary[method]["delta_med_err"] = (
            summary[method]["steady"]["med_err_median"]
            - summary[method]["brunone"]["med_err_median"]
        )

    scissors = {
        "cepstrum_improves_on_steady": summary["cepstrum"]["delta_f1"] > 0,
        "p0_degrades_on_steady": summary["p0"]["delta_f1"] < 0,
        "scissors_hold": (
            summary["cepstrum"]["delta_f1"] > 0 and summary["p0"]["delta_f1"] < 0
        ),
    }
    return {"per_case": per_case, "summary": summary, "scissors": scissors}


def run_phasenet(
    out_dir: Path,
    *,
    checkpoint: str,
    max_cases_note: str = "full test split",
) -> Optional[Dict[str, object]]:
    """对 steady 集生成 manifest、推理、阈值重校准并对照 brunone 指标。"""
    try:
        import torch
        from torch.utils.data import DataLoader
        from neural_operator.direct_inverse.data import (
            DirectInverseDataset,
            load_manifest,
        )
        from neural_operator.direct_inverse.pipeline import (
            collect_predictions,
            load_checkpoint,
        )
        from neural_operator.direct_inverse.evaluate import (
            calibrate_threshold,
            evaluate_bundle,
        )
        from neural_operator.direct_inverse.config import dataclass_from_dict, DataConfig, DetectorConfig as DIDet
    except Exception as e:
        print(f"  PhaseNet skipped (import): {e}")
        return {"skipped": True, "reason": str(e)}

    # 1) 生成 steady manifest（若尚无）
    man_root = Path("output/direct_inverse/manifests_close2000_steady")
    man_root.mkdir(parents=True, exist_ok=True)
    man_path = man_root / "direct-inverse-data-v2-close2000-n8192-seed42.json"
    if not man_path.exists():
        print("  generating steady manifest ...")
        import subprocess
        cmd = [
            sys.executable, "-m", "neural_operator.direct_inverse.data",
            "--data-dir", str(STEADY_ROOT / "data"),
            "--profile", "close2000_n8192",
            "--output-root", str(man_root),
            "--split-seed", "42",
        ]
        subprocess.check_call(cmd)
    # 找实际写入的 manifest
    cands = list(man_root.glob("*seed42.json"))
    if not cands:
        return {"skipped": True, "reason": "manifest not created"}
    man_path = cands[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model = load_checkpoint(checkpoint, device)
    manifest = load_manifest(str(man_path))
    data_cfg = dataclass_from_dict(DataConfig, manifest["data_config"])
    det_cfg = dataclass_from_dict(DIDet, manifest["detector_config"])

    def _infer(selection: str) -> Dict[str, np.ndarray]:
        ds = DirectInverseDataset(str(man_path), selection)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        return collect_predictions(model, loader, device)

    print("  PhaseNet infer validation (recalibrate) ...")
    val_bundle = _infer("validation")
    calib = calibrate_threshold(val_bundle, data_cfg, det_cfg)
    thr_new = float(calib["threshold"])
    thr_old = 0.35

    print("  PhaseNet infer test ...")
    test_bundle = _infer("test")
    art_path = out_dir / "phasenet_steady_test.npz"
    np.savez_compressed(art_path, **test_bundle)

    metrics_old = evaluate_bundle(test_bundle, thr_old, data_cfg, det_cfg)
    metrics_new = evaluate_bundle(test_bundle, thr_new, data_cfg, det_cfg)

    # brunone 基线：读已有 metrics_summary
    brunone_f1 = None
    base_sum = Path(
        "output/direct_inverse/runs_close2000_n8192/"
        "20260730-151610-921822-p0a-subset512-seed42-p0a-n8192-subset512/"
        "evaluations/metrics_summary.json"
    )
    if base_sum.exists():
        with base_sum.open(encoding="utf-8") as fh:
            brunone_f1 = json.load(fh).get("physical", {}).get("f1")

    result = {
        "manifest": str(man_path),
        "checkpoint": checkpoint,
        "threshold_old": thr_old,
        "threshold_recalibrated": thr_new,
        "steady_test_f1_old_thr": metrics_old["physical"]["f1"],
        "steady_test_f1_new_thr": metrics_new["physical"]["f1"],
        "steady_test_precision_new": metrics_new["physical"]["precision"],
        "steady_test_recall_new": metrics_new["physical"]["recall"],
        "brunone_ref_f1": brunone_f1,
        "delta_f1_vs_brunone_ref": (
            None if brunone_f1 is None
            else metrics_new["physical"]["f1"] - brunone_f1
        ),
        "degrades_on_steady": (
            None if brunone_f1 is None
            else metrics_new["physical"]["f1"] < brunone_f1
        ),
        "note": max_cases_note,
        "artifact": str(art_path),
    }
    (out_dir / "phasenet_steady_metrics.json").write_text(
        json.dumps({
            "old_thr": metrics_old,
            "new_thr": metrics_new,
            "calibration": {
                "threshold": thr_new,
                "val_f1": calib["validation_metrics"]["physical"]["f1"],
            },
        }, indent=2, default=str),
        encoding="utf-8",
    )
    return result


def main():
    p = argparse.ArgumentParser(description="配对失配可迁移性检验")
    p.add_argument("--tag", default="main")
    p.add_argument("--max-cases", type=int, default=300,
                   help="古典方法评测的 case 数（按 id 顺序，配对）")
    p.add_argument("--skip-phasenet", action="store_true")
    p.add_argument(
        "--phasenet-ckpt",
        default=(
            "output/direct_inverse/runs_close2000_n8192/"
            "20260730-151610-921822-p0a-subset512-seed42-p0a-n8192-subset512/"
            "checkpoints/best.pt"
        ),
    )
    args = p.parse_args()

    out_dir = Path(DEFAULT_OUT) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== pairing check ===")
    pairing = _verify_pairing(50)
    print(f"  paired={pairing['paired']} mismatches={pairing['n_mismatch']}")
    if not pairing["paired"]:
        print("  WARNING: 几何未完全配对，结果解释需谨慎")

    print("=== load P0 dictionary ===")
    if not DICT_PATH.exists():
        raise FileNotFoundError(f"missing dictionary: {DICT_PATH}")
    with np.load(DICT_PATH) as z:
        dic = {
            "depth_grid": z["depth_grid"],
            "cf_grid": z["cf_grid"],
            "intact": z["intact"],
            "templates": z["templates"],
        }

    # 取 PASS case
    with (BRUNONE_ROOT / "lhs_metadata.json").open(encoding="utf-8") as fh:
        idx = json.load(fh)["samples_index"]
    case_ids = [r["case_id"] for r in idx if r.get("status", "PASS") == "PASS"]
    case_ids = case_ids[: args.max_cases]
    print(f"=== classical methods on {len(case_ids)} paired cases ===")
    t0 = time.time()
    classical = run_classical(case_ids, dic=dic)
    print(f"  done in {time.time() - t0:.1f}s")
    print("  cepstrum ΔF1 (steady-brunone) =", classical["summary"]["cepstrum"]["delta_f1"])
    print("  P0       ΔF1 (steady-brunone) =", classical["summary"]["p0"]["delta_f1"])
    print("  scissors_hold =", classical["scissors"]["scissors_hold"])

    phasenet = None
    if not args.skip_phasenet:
        print("=== PhaseNet on steady ===")
        if not Path(args.phasenet_ckpt).exists():
            phasenet = {"skipped": True, "reason": f"missing ckpt {args.phasenet_ckpt}"}
            print("  skipped:", phasenet["reason"])
        else:
            phasenet = run_phasenet(out_dir, checkpoint=args.phasenet_ckpt)

    # 综合剪刀差（含 PhaseNet 若可用）
    scissors = dict(classical["scissors"])
    if phasenet and not phasenet.get("skipped") and phasenet.get("degrades_on_steady") is not None:
        scissors["phasenet_degrades_on_steady"] = phasenet["degrades_on_steady"]
        scissors["full_scissors_hold"] = (
            scissors["scissors_hold"] and phasenet["degrades_on_steady"]
        )

    result = {
        "tag": args.tag,
        "pairing": pairing,
        "n_cases_classical": len(case_ids),
        "classical_summary": classical["summary"],
        "scissors": scissors,
        "phasenet": {
            k: v for k, v in (phasenet or {"skipped": True}).items()
            if k != "per_case"
        },
    }
    (out_dir / "transfer_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    # 精简 per-case（不含过长）
    slim_cases = [
        {k: v for k, v in r.items() if k != "x_f"}
        for r in classical["per_case"]
    ]
    (out_dir / "classical_per_case.json").write_text(
        json.dumps(slim_cases, indent=2), encoding="utf-8"
    )
    print(f"\n写入 {out_dir}")
    print("scissors:", json.dumps(scissors, indent=2))


if __name__ == "__main__":
    main()
