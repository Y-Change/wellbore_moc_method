# -*- coding: utf-8 -*-
"""Re-evaluate completed runs under the current event-primary gate without retraining."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from neural_operator.dccdm_pipeline import write_json
from .config import DataConfig, DetectorConfig, GateConfig
from .evaluate import evaluate_bundle
from .train import overfit_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Amend a completed gate without retraining")
    parser.add_argument("run_dir")
    parser.add_argument(
        "--stage",
        choices=("overfit1", "overfit16", "subset512", "full"),
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override detector threshold; default uses run threshold.json or 0.5 for overfit",
    )
    parser.add_argument(
        "--output-name",
        default="gate_result_v3.json",
        help="Filename written under the run directory",
    )
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    data_config = DataConfig(**manifest["data_config"])
    artifact = np.load(
        os.path.join(run_dir, "evaluations", "validation_predictions.npz"),
        allow_pickle=False,
    )
    bundle = {key: artifact[key] for key in artifact.files}

    if args.threshold is not None:
        threshold = float(args.threshold)
    elif args.stage in ("overfit1", "overfit16"):
        threshold = 0.5
    else:
        threshold_path = os.path.join(run_dir, "threshold.json")
        with open(threshold_path, encoding="utf-8") as handle:
            threshold = float(json.load(handle)["threshold"])

    metrics = evaluate_bundle(bundle, threshold, data_config, DetectorConfig())
    with open(os.path.join(run_dir, "train_log.csv"), newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best_row = min(rows, key=lambda row: float(row["validation_loss"]))
    gate = overfit_gate(
        args.stage,
        metrics,
        initial_loss=float(rows[0]["validation_loss"]),
        best_loss=float(best_row["validation_loss"]),
        gate=GateConfig(),
        best_dice_loss=float(best_row["validation_dice"]),
    )
    gate.update({
        "stage": args.stage,
        "amendment": {
            "reason": (
                "Re-evaluate without retraining under event-primary hard gates and stratified "
                "event reports. Dense-map Dice / composite-loss reduction remain diagnostics."
            ),
            "original_gate": "gate_result.json",
            "training_reused": True,
            "prediction_artifact_reused": True,
            "threshold_reused": args.threshold is None,
            "threshold": threshold,
        },
        "best_epoch": int(best_row["epoch"]),
        "initial_loss_for_amendment": float(rows[0]["validation_loss"]),
        "best_validation_loss": float(best_row["validation_loss"]),
        "metrics": {key: value for key, value in metrics.items() if key != "per_case"},
    })
    output = os.path.join(run_dir, args.output_name)
    write_json(output, gate)
    strata_path = os.path.join(run_dir, "evaluations", "metrics_summary_stratified.json")
    write_json(strata_path, {key: value for key, value in metrics.items() if key != "per_case"})
    print(json.dumps({key: value for key, value in gate.items() if key != "metrics"}, ensure_ascii=False, indent=2))
    print(f"Written {output}")
    print(f"Written {strata_path}")
    print(f"Gate: {'PASS' if gate['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
