# -*- coding: utf-8 -*-
"""Reproducible DCCDM inference for schema-v2 checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from datetime import datetime

import numpy as np
import torch

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError("Cannot find wellbore_moc_method root")
    _d = _parent

from neural_operator.cepstrum import DifferentiableCepstrum
from neural_operator.dataset_surrogate import FracturingMOCSurrogateDataset
from neural_operator.dccdm_pipeline import (
    DCCDMModelConfig,
    DCCDMPreprocessingConfig,
    load_checkpoint,
    seed_everything,
    write_json,
)
from neural_operator.dedispersion import DeDispersionFrontEnd
from neural_operator.diffusion_1d import ConditionalUNet1D, GaussianDiffusion1D
from neural_operator.evaluate_diffusion import evaluate_prediction_bundle
from neural_operator.train_diffusion import collect_predictions, resolve_device


def _dataclass_from_dict(cls, values: dict):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible DCCDM schema-v2 inference")
    parser.add_argument("checkpoint", help="schema-v2 best.pt or last.pt")
    parser.add_argument("--data-dir", default="output/lhs_dataset/data")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = _dataclass_from_dict(DCCDMModelConfig, checkpoint["model_config"])
    preprocessing = _dataclass_from_dict(
        DCCDMPreprocessingConfig, checkpoint["preprocessing_config"]
    )

    dedisp = DeDispersionFrontEnd().to(device)
    cepstrum = DifferentiableCepstrum(eps_threshold=preprocessing.cepstrum_eps).to(device)
    unet = ConditionalUNet1D(
        in_channels=1,
        out_channels=1,
        context_dim=model_config.context_dim,
        base_dim=model_config.base_dim,
    ).to(device)
    diffusion = GaussianDiffusion1D(
        unet, seq_length=model_config.seq_length, timesteps=model_config.timesteps
    ).to(device)
    dedisp.load_state_dict(checkpoint["dedisp_state_dict"])
    unet.load_state_dict(checkpoint["unet_state_dict"])

    dataset = FracturingMOCSurrogateDataset(
        data_dir=args.data_dir,
        n_time_target=model_config.seq_length,
        split=args.split,
        seed=int(checkpoint["training_config"].get("seed", 42)),
    )
    count = min(args.num_samples, len(dataset))
    indices = list(range(count))
    predictions = collect_predictions(
        diffusion,
        dedisp,
        cepstrum,
        dataset,
        indices,
        preprocessing,
        device,
        args.seed,
        args.batch_size,
    )
    evaluation = evaluate_prediction_bundle(
        true_maps=predictions["true_map"],
        raw_predictions=predictions["raw_prediction"],
        postprocessed_predictions=predictions["postprocessed_prediction"],
        time_axes=predictions["time_axis"],
        x_f_padded=predictions["x_f"],
        Cf_padded=predictions["Cf"],
        n_frac=predictions["n_frac"],
        case_ids=predictions["case_id"],
        target_scale=preprocessing.target_scale,
        peak_height=preprocessing.physical_threshold,
    )

    checkpoint_run = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=False)
    else:
        evaluation_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-seed{args.seed}"
        output_dir = os.path.join(checkpoint_run, "evaluations", evaluation_id)
        os.makedirs(output_dir, exist_ok=False)

    prediction_path = os.path.join(output_dir, "predictions.npz")
    np.savez_compressed(prediction_path, **predictions)
    write_json(os.path.join(output_dir, "evaluation_manifest.json"), {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_schema": checkpoint["schema_version"],
        "seed": args.seed,
        "split": args.split,
        "n_samples": count,
        "prediction_artifact": prediction_path,
        "evaluation_config": evaluation["config"],
    })
    write_json(os.path.join(output_dir, "metrics_summary.json"), {
        "config": evaluation["config"],
        "aggregate": evaluation["aggregate"],
    })
    print(f"Inference and evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
