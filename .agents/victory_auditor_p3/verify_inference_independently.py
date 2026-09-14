import os
import sys
import json
from pathlib import Path
import torch
import numpy as np

repo_root = Path("e:/water_hammer_research/wellbore_moc_method")
paper_c_dir = repo_root / "PaperC_CJNO_Wellbore_Inversion"

sys.path.insert(0, str(repo_root))

from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import TGDISDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics

print("=" * 80)
print("INDEPENDENT INFERENCE AUDIT FROM PYTORCH CHECKPOINT")
print("=" * 80)

device = torch.device("cpu")
ckpt_path = paper_c_dir / "checkpoints" / "tg_dis_deeponet_best.pt"
assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

# 1. Instantiate fresh model and load state_dict
model = TGDISDeepONet(
    window_mode="relative_window",
    use_acoustic_bias=True,
    d_model=64,
    n_heads=4,
    n_layers=2,
    p=64,
    max_nc=6,
    d_well=128,
    delta_x_max=10.0,
).to(device)

ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {ckpt_path.name} | Parameters: {sum(p.numel() for p in model.parameters())}")

# 2. Load independent test dataset
test_ds = PilotInversionDataset(split="test")
test_loader = test_ds.get_dataloader(batch_size=100, shuffle=False)
test_batch = next(iter(test_loader))
print(f"Loaded test dataset: {len(test_ds)} samples")

# 3. Independent forward pass
test_batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}
with torch.no_grad():
    out = model(test_batch_dev)

pred_alpha = out["alpha"].cpu()
pred_cf = out["cf"].cpu() if "cf" in out else torch.zeros_like(pred_alpha)
pred_field = out.get("m_alpha_grid", torch.zeros(len(pred_alpha), 500)).cpu()
pred_exist = out.get("p_exist", (pred_alpha > 0.05).float()).cpu()
pred_delta_x = out.get("delta_x", torch.zeros_like(pred_alpha)).cpu()

pred_dict = {
    "alpha": pred_alpha,
    "cf": pred_cf,
    "m_alpha_grid": pred_field,
    "p_exist": pred_exist,
    "delta_x": pred_delta_x,
}

targ_dict = {
    "alpha": test_batch["alpha"],
    "cf": test_batch["cf"],
    "m_alpha_grid": test_batch["m_alpha_grid"],
    "mask": test_batch["mask"],
    "positions": test_batch["positions"],
}

computed_metrics = compute_inversion_metrics(pred_dict, targ_dict, tolerance_m=10.0)

# 4. Compare with saved JSON
bench_json_p = paper_c_dir / "output" / "phase3_benchmark_metrics.json"
with open(bench_json_p, "r", encoding="utf-8") as f:
    saved_json = json.load(f)["tg_dis_deeponet"]

print("\n--- INDEPENDENT METRICS VS SAVED BENCHMARK JSON ---")
keys_to_compare = [
    ("alpha_mae", "Alpha MAE (Full)"),
    ("alpha_r2", "Alpha R2 (Full)"),
    ("alpha_r2_dense", "Alpha R2 (Dense Nc>=4)"),
    ("alpha_mae_dense", "Alpha MAE (Dense)"),
    ("alpha_mae_sparse", "Alpha MAE (Sparse)"),
    ("cf_mre_pct", "Cf MRE %"),
    ("cf_log10_mae", "Cf Log10 MAE"),
    ("w1_mean_m", "W1 Mean (m)"),
    ("f1_score", "Detection F1-Score"),
    ("simplex_max_dev", "Simplex Max Dev"),
]

all_matched = True
for key, label in keys_to_compare:
    val_computed = computed_metrics[key]
    val_saved = saved_json[key]
    diff = abs(val_computed - val_saved)
    match = diff < 1e-5
    if not match:
        all_matched = False
    print(f"  {label:24s} | Computed: {val_computed:10.6f} | Saved: {val_saved:10.6f} | Diff: {diff:.2e} | {'MATCH' if match else 'DISCREPANCY'}")

print(f"\nOverall Match Verdict: {'PERFECT MATCH' if all_matched else 'DISCREPANCY FOUND'}")
assert all_matched, "Computed metrics do not match saved benchmark JSON!"
