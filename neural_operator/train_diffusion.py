# -*- coding: utf-8 -*-
"""Train DCCDM with reproducible run artifacts and an ARIS overfit sanity gate."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time as time_module
from dataclasses import asdict
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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
PROJECT_ROOT = _d

from neural_operator.cepstrum import DifferentiableCepstrum
from neural_operator.dataset_surrogate import FracturingMOCSurrogateDataset
from neural_operator.dccdm_pipeline import (
    DCCDMModelConfig,
    DCCDMPreprocessingConfig,
    add_observation_noise,
    build_dccdm_context,
    create_run_directory,
    decode_target,
    encode_target,
    file_manifest_digest,
    git_state,
    make_generator,
    postprocess_prediction,
    save_checkpoint,
    seed_everything,
    write_json,
)
from neural_operator.dedispersion import DeDispersionFrontEnd
from neural_operator.diffusion_1d import ConditionalUNet1D, GaussianDiffusion1D
from neural_operator.evaluate_diffusion import evaluate_prediction_bundle


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_data_manifest(dataset: FracturingMOCSurrogateDataset, indices: Sequence[int], split: str) -> Dict:
    cases = []
    for index in indices:
        metadata = dataset.get_case_metadata(index)
        cases.append({
            "dataset_index": int(index),
            "case_id": metadata["case_id"],
            "source_file": metadata["source_file"],
            "tf": metadata["tf"],
            "n_frac": metadata["n_frac"],
        })
    files = [case["source_file"] for case in cases]
    return {
        "split": split,
        "n_cases": len(cases),
        "cases": cases,
        "file_manifest_digest": file_manifest_digest(files),
    }


def fixed_diagnostic_loss(
    diffusion: GaussianDiffusion1D,
    dedisp: torch.nn.Module,
    cepstrum: torch.nn.Module,
    loader: DataLoader,
    preprocessing: DCCDMPreprocessingConfig,
    device: torch.device,
    seed: int,
) -> float:
    dedisp.eval()
    diffusion.model.eval()
    total = 0.0
    count = 0
    generator = make_generator(device, seed)
    with torch.no_grad():
        for batch_input, batch_H in loader:
            batch_input = batch_input.to(device)
            batch_H = batch_H.to(device)
            context, _ = build_dccdm_context(batch_H, dedisp, cepstrum, preprocessing)
            target = encode_target(batch_input[:, 2:3, :], preprocessing.target_scale)
            batch_size = batch_H.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device, generator=generator)
            noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
            loss = diffusion.p_losses(target, t, context=context, noise=noise)
            total += float(loss) * batch_size
            count += batch_size
    return total / max(count, 1)


def collect_predictions(
    diffusion: GaussianDiffusion1D,
    dedisp: torch.nn.Module,
    cepstrum: torch.nn.Module,
    dataset: FracturingMOCSurrogateDataset,
    indices: Sequence[int],
    preprocessing: DCCDMPreprocessingConfig,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False, num_workers=0)
    dedisp.eval()
    diffusion.model.eval()

    true_maps: List[np.ndarray] = []
    raw_predictions: List[np.ndarray] = []
    postprocessed: List[np.ndarray] = []
    observations: List[np.ndarray] = []
    time_axes: List[np.ndarray] = []
    metadata_rows = [dataset.get_case_metadata(index) for index in indices]
    offset = 0
    with torch.no_grad():
        for batch_input, batch_H in loader:
            batch_input = batch_input.to(device)
            batch_H = batch_H.to(device)
            context, _ = build_dccdm_context(batch_H, dedisp, cepstrum, preprocessing)
            generator = make_generator(device, seed + offset)
            raw_normalized = diffusion.sample(context=context, generator=generator)
            raw_physical = decode_target(raw_normalized, preprocessing.target_scale)
            post = postprocess_prediction(
                raw_physical,
                target_scale=preprocessing.target_scale,
                threshold=preprocessing.physical_threshold,
            )
            true_maps.append(batch_input[:, 2:3, :].cpu().numpy())
            raw_predictions.append(raw_physical.cpu().numpy())
            postprocessed.append(post.cpu().numpy())
            observations.append(batch_H.cpu().numpy())
            time_axes.extend(row["time_axis"] for row in metadata_rows[offset : offset + len(batch_H)])
            offset += len(batch_H)

    max_fracs = max(row["n_frac"] for row in metadata_rows)
    x_f = np.full((len(metadata_rows), max_fracs), np.nan, dtype=np.float32)
    Cf = np.full_like(x_f, np.nan)
    kleak = np.full_like(x_f, np.nan)
    n_frac = np.zeros(len(metadata_rows), dtype=np.int64)
    for index, row in enumerate(metadata_rows):
        count = row["n_frac"]
        n_frac[index] = count
        x_f[index, :count] = row["x_f"]
        Cf[index, :count] = row["Cf"]
        kleak[index, :count] = row["kleak"]

    return {
        "true_map": np.concatenate(true_maps),
        "raw_prediction": np.concatenate(raw_predictions),
        "postprocessed_prediction": np.concatenate(postprocessed),
        "clean_observation": np.concatenate(observations),
        "time_axis": np.stack(time_axes),
        "x_f": x_f,
        "Cf": Cf,
        "kleak": kleak,
        "n_frac": n_frac,
        "case_id": np.asarray([row["case_id"] for row in metadata_rows]),
    }


def reconstruct_target(metadata: Dict, n_time: int) -> np.ndarray:
    t = np.linspace(0.0, metadata["tf"], n_time, dtype=np.float32)
    result = np.zeros(n_time, dtype=np.float32)
    for xf, cf in zip(metadata["x_f"], metadata["Cf"]):
        arrival = metadata["pump_shut_time"] + 2.0 * float(xf) / metadata["wavespeed"]
        weight = np.log10(max(float(cf), 1.0e-12)) + 12.0
        result += weight * np.exp(-((t - arrival) ** 2) / (2.0 * metadata["sigma_impulse_s"] ** 2))
    return result


def write_training_plot(run_dir: str, rows: List[Dict]) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in rows], label="Training noise loss", linewidth=2)
    ax.plot(epochs, [row["diagnostic_loss"] for row in rows], label="Fixed diagnostic loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("DCCDM training diagnostics")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plots", "training_loss.png"), dpi=200)
    plt.close(fig)


def append_experiment_log(run_dir: str, args: argparse.Namespace, gate: Dict) -> None:
    log_path = os.path.abspath(os.path.join(args.output_root, "..", "experiment_log.csv"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fields = [
        "run_id", "mode", "seed", "status", "best_diagnostic_loss",
        "raw_fixed_f1", "raw_strict_f1", "artifact_dir",
    ]
    exists = os.path.isfile(log_path)
    raw = gate["raw_metrics"]
    row = {
        "run_id": os.path.basename(run_dir),
        "mode": args.mode,
        "seed": args.seed,
        "status": "pass" if gate["passed"] else ("fail" if gate["passed"] is False else "completed"),
        "best_diagnostic_loss": gate["best_diagnostic_loss"],
        "raw_fixed_f1": raw["fixed_tolerance"]["micro_f1"],
        "raw_strict_f1": raw["strict_tolerance"]["micro_f1"],
        "artifact_dir": run_dir,
    }
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_full_gate(path: str, model: DCCDMModelConfig, preprocessing: DCCDMPreprocessingConfig) -> None:
    if not path:
        raise ValueError("full mode requires --sanity-run")
    gate_path = os.path.join(os.path.abspath(path), "gate_result.json")
    if not os.path.isfile(gate_path):
        raise FileNotFoundError(f"missing sanity gate: {gate_path}")
    with open(gate_path, encoding="utf-8") as handle:
        gate = json.load(handle)
    if not gate.get("passed"):
        raise RuntimeError(f"sanity gate did not pass: {gate_path}")
    if gate.get("model_config") != asdict(model) or gate.get("preprocessing_config") != asdict(preprocessing):
        raise RuntimeError("sanity gate configuration is incompatible with this full run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCCDM with an ARIS overfit sanity gate")
    parser.add_argument("--mode", choices=("overfit16", "full"), default="overfit16")
    parser.add_argument("--data-dir", default="output/lhs_dataset/data")
    parser.add_argument("--output-root", default="output/dccdm/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--sanity-run", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--base-dim", type=int, default=64)
    parser.add_argument("--learning-rate-unet", type=float, default=2.0e-4)
    parser.add_argument("--learning-rate-dedisp", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--observation-noise-fraction", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-samples", type=int, default=16)
    args = parser.parse_args()

    if args.seq_length % 4:
        raise ValueError("seq-length must be divisible by 4")
    if args.mode == "overfit16" and args.observation_noise_fraction != 0.0:
        raise ValueError("overfit16 requires --observation-noise-fraction 0")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation-steps must be >= 1")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    preprocessing = DCCDMPreprocessingConfig()
    model_config = DCCDMModelConfig(
        seq_length=args.seq_length,
        timesteps=args.timesteps,
        base_dim=args.base_dim,
        context_dim=2,
    )
    if args.mode == "full":
        validate_full_gate(args.sanity_run, model_config, preprocessing)

    dataset = FracturingMOCSurrogateDataset(
        data_dir=args.data_dir, n_time_target=args.seq_length, split="train", seed=args.seed
    )
    if args.mode == "overfit16":
        if len(dataset) < 16:
            raise RuntimeError("overfit16 requires at least 16 training cases")
        train_indices = list(range(16))
        diagnostic_indices = list(train_indices)
        diagnostic_dataset = dataset
    else:
        train_indices = list(range(len(dataset)))
        diagnostic_dataset = FracturingMOCSurrogateDataset(
            data_dir=args.data_dir, n_time_target=args.seq_length, split="val", seed=args.seed
        )
        diagnostic_indices = list(range(len(diagnostic_dataset)))

    run_dir = create_run_directory(args.output_root, args.mode, args.seed, args.run_name)
    data_manifest = {
        "train": build_data_manifest(dataset, train_indices, "train"),
        "diagnostic": build_data_manifest(
            diagnostic_dataset, diagnostic_indices,
            "overfit-diagnostic" if args.mode == "overfit16" else "validation",
        ),
    }
    write_json(os.path.join(run_dir, "data_manifest.json"), data_manifest)

    training_config = vars(args).copy()
    training_config.update({"resolved_device": str(device), "project_git": git_state(PROJECT_ROOT)})
    write_json(os.path.join(run_dir, "manifest.json"), {
        "run_id": os.path.basename(run_dir),
        "model_config": asdict(model_config),
        "preprocessing_config": asdict(preprocessing),
        "training_config": training_config,
    })

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=loader_generator, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    diagnostic_loader = DataLoader(
        Subset(diagnostic_dataset, diagnostic_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    dedisp = DeDispersionFrontEnd().to(device)
    cepstrum = DifferentiableCepstrum(eps_threshold=preprocessing.cepstrum_eps).to(device)
    unet = ConditionalUNet1D(
        in_channels=1, out_channels=1, context_dim=model_config.context_dim,
        base_dim=model_config.base_dim,
    ).to(device)
    diffusion = GaussianDiffusion1D(
        unet, seq_length=model_config.seq_length, timesteps=model_config.timesteps
    ).to(device)
    optimizer = optim.AdamW([
        {"params": dedisp.parameters(), "lr": args.learning_rate_dedisp},
        {"params": unet.parameters(), "lr": args.learning_rate_unet},
    ], weight_decay=args.weight_decay)

    initial_diagnostic = fixed_diagnostic_loss(
        diffusion, dedisp, cepstrum, diagnostic_loader, preprocessing, device, args.seed + 10_000
    )
    log_path = os.path.join(run_dir, "train_log.csv")
    fields = ["epoch", "train_loss", "diagnostic_loss", "lr_unet", "lr_dedisp", "grad_norm_unet", "grad_norm_dedisp", "elapsed_s"]
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    rows: List[Dict] = []
    best_metric = float("inf")
    global_step = 0
    training_generator = make_generator(device, args.seed + 20_000)
    for epoch in range(1, args.epochs + 1):
        started = time_module.time()
        dedisp.train()
        unet.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_samples = 0
        last_grad_unet = 0.0
        last_grad_dedisp = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (batch_input, batch_H) in enumerate(progress):
            batch_input = batch_input.to(device, non_blocking=True)
            batch_H = batch_H.to(device, non_blocking=True)
            noisy_observation, _ = add_observation_noise(
                batch_H, args.observation_noise_fraction, generator=training_generator
            )
            context, _ = build_dccdm_context(noisy_observation, dedisp, cepstrum, preprocessing)
            target = encode_target(batch_input[:, 2:3, :], preprocessing.target_scale)
            t = torch.randint(
                0, diffusion.timesteps, (batch_H.shape[0],), device=device, generator=training_generator
            )
            noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=training_generator)
            raw_loss = diffusion.p_losses(target, t, context=context, noise=noise)
            (raw_loss / args.accumulation_steps).backward()

            should_step = (step + 1) % args.accumulation_steps == 0 or step + 1 == len(train_loader)
            if should_step:
                last_grad_dedisp = float(torch.nn.utils.clip_grad_norm_(dedisp.parameters(), 1.0))
                last_grad_unet = float(torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            batch_size = batch_H.shape[0]
            total_loss += float(raw_loss.detach()) * batch_size
            n_samples += batch_size
            progress.set_postfix(loss=f"{float(raw_loss):.4f}")

        train_loss = total_loss / max(n_samples, 1)
        diagnostic_loss = fixed_diagnostic_loss(
            diffusion, dedisp, cepstrum, diagnostic_loader, preprocessing, device, args.seed + 10_000
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "diagnostic_loss": diagnostic_loss,
            "lr_unet": optimizer.param_groups[1]["lr"],
            "lr_dedisp": optimizer.param_groups[0]["lr"],
            "grad_norm_unet": last_grad_unet,
            "grad_norm_dedisp": last_grad_dedisp,
            "elapsed_s": time_module.time() - started,
        }
        rows.append(row)
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)

        save_checkpoint(
            os.path.join(run_dir, "checkpoints", "last.pt"), dedisp=dedisp, unet=unet,
            optimizer=optimizer, epoch=epoch, global_step=global_step,
            best_metric=min(best_metric, diagnostic_loss), model_config=model_config,
            preprocessing_config=preprocessing, training_config=training_config,
            data_manifest=data_manifest,
        )
        if diagnostic_loss < best_metric:
            best_metric = diagnostic_loss
            save_checkpoint(
                os.path.join(run_dir, "checkpoints", "best.pt"), dedisp=dedisp, unet=unet,
                optimizer=optimizer, epoch=epoch, global_step=global_step,
                best_metric=best_metric, model_config=model_config,
                preprocessing_config=preprocessing, training_config=training_config,
                data_manifest=data_manifest,
            )
        print(f"Epoch {epoch}: train={train_loss:.6f}, diagnostic={diagnostic_loss:.6f}")

    write_training_plot(run_dir, rows)

    evaluation_indices = diagnostic_indices[: min(args.eval_samples, len(diagnostic_indices))]
    predictions = collect_predictions(
        diffusion, dedisp, cepstrum, diagnostic_dataset, evaluation_indices,
        preprocessing, device, args.seed + 30_000, args.batch_size,
    )
    np.savez_compressed(os.path.join(run_dir, "evaluations", "final_predictions.npz"), **predictions)
    evaluation = evaluate_prediction_bundle(
        true_maps=predictions["true_map"], raw_predictions=predictions["raw_prediction"],
        postprocessed_predictions=predictions["postprocessed_prediction"],
        time_axes=predictions["time_axis"], x_f_padded=predictions["x_f"],
        Cf_padded=predictions["Cf"], n_frac=predictions["n_frac"],
        case_ids=predictions["case_id"], target_scale=preprocessing.target_scale,
        peak_height=preprocessing.physical_threshold,
    )
    write_json(os.path.join(run_dir, "evaluations", "metrics_summary.json"), {
        "config": evaluation["config"], "aggregate": evaluation["aggregate"],
    })

    reproducibility_predictions = collect_predictions(
        diffusion, dedisp, cepstrum, diagnostic_dataset, evaluation_indices,
        preprocessing, device, args.seed + 30_000, args.batch_size,
    )
    same_seed_reproducible = np.array_equal(
        predictions["raw_prediction"], reproducibility_predictions["raw_prediction"]
    )

    label_checks = []
    for dataset_index in evaluation_indices:
        model_input, _ = diagnostic_dataset[dataset_index]
        metadata = diagnostic_dataset.get_case_metadata(dataset_index)
        reconstructed = reconstruct_target(metadata, args.seq_length)
        label_checks.append(bool(np.allclose(model_input[2].numpy(), reconstructed, rtol=1.0e-5, atol=1.0e-5)))

    raw = evaluation["aggregate"]["raw"]
    zero = evaluation["aggregate"]["zero_baseline"]
    fixed = raw["fixed_tolerance"]
    strict = raw["strict_tolerance"]
    loss_reduction = (initial_diagnostic - best_metric) / max(initial_diagnostic, 1.0e-12)
    criteria = {
        "all_finite": raw["physical_validity"]["finite_fraction"] == 1.0,
        "diagnostic_loss_reduction_ge_80pct": loss_reduction >= 0.80,
        "diagnostic_loss_le_0_10": best_metric <= 0.10,
        "raw_map_mae_better_than_half_zero_baseline": raw["map"]["mae"] < 0.5 * zero["map"]["mae"],
        "fixed_f1_ge_0_90": fixed["micro_f1"] >= 0.90,
        "strict_f1_ge_0_75": strict["micro_f1"] >= 0.75,
        "strict_median_depth_error_le_36_25m": strict["median_depth_error_m"] is not None and strict["median_depth_error_m"] <= 36.25,
        "all_labels_reconstructed": all(label_checks),
        "same_seed_inference_reproducible": same_seed_reproducible,
    }
    gate = {
        "passed": all(criteria.values()) if args.mode == "overfit16" else None,
        "mode": args.mode,
        "criteria": criteria,
        "initial_diagnostic_loss": initial_diagnostic,
        "best_diagnostic_loss": best_metric,
        "diagnostic_loss_reduction": loss_reduction,
        "model_config": asdict(model_config),
        "preprocessing_config": asdict(preprocessing),
        "raw_metrics": raw,
        "postprocessed_metrics": evaluation["aggregate"]["postprocessed"],
        "zero_baseline_metrics": zero,
    }
    write_json(os.path.join(run_dir, "gate_result.json"), gate)
    append_experiment_log(run_dir, args, gate)
    print(f"Run written to {run_dir}")
    if args.mode == "overfit16":
        print(f"ARIS sanity gate: {'PASS' if gate['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
