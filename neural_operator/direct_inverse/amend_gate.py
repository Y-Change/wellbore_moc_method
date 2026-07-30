# -*- coding: utf-8 -*-
"""Re-evaluate completed memorization runs under the preregistered event-primary v2 gate."""
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
    parser = argparse.ArgumentParser(description="Amend a completed overfit gate without retraining")
    parser.add_argument("run_dir")
    parser.add_argument("--stage", choices=("overfit1", "overfit16"), required=True)
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
    metrics = evaluate_bundle(bundle, 0.5, data_config, DetectorConfig())
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
                "The v1 dense-loss gate was miscalibrated for soft Gaussian labels: "
                "a probability-mass Dice is not expected to approach 0.98, while blind event "
                "precision/recall/count/localization are the preregistered task outcomes."
            ),
            "original_gate": "gate_result.json",
            "training_reused": True,
            "prediction_artifact_reused": True,
        },
        "best_epoch": int(best_row["epoch"]),
        "initial_loss_for_amendment": float(rows[0]["validation_loss"]),
        "best_validation_loss": float(best_row["validation_loss"]),
        "metrics": {key: value for key, value in metrics.items() if key != "per_case"},
    })
    output = os.path.join(run_dir, "gate_result_v2.json")
    write_json(output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"Written {output}")


if __name__ == "__main__":
    main()
