import json
import torch
from pathlib import Path

weights_dir = Path("PaperC_CJNO_Wellbore_Inversion/output/weights")
ckpt_dir = Path("PaperC_CJNO_Wellbore_Inversion/checkpoints")

hist_path = weights_dir / "tg_dis_deeponet_history.json"
ckpt_path = ckpt_dir / "tg_dis_deeponet_best.pt"
out_ckpt_path = weights_dir / "tg_dis_deeponet_best.pt"

ckpt = torch.load(ckpt_path, map_location="cpu")
print("Checkpoint loaded:")
print("  Epoch:", ckpt.get("epoch"))
print("  Model name:", ckpt.get("model_name"))
print("  Metrics in ckpt:", ckpt.get("metrics"))

if hist_path.exists():
    with open(hist_path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    print("\nHistory JSON info:")
    print("  Model name:", hist.get("model_name"))
    print("  Total params:", hist.get("total_params"))
    print("  Train time sec:", hist.get("train_time_sec"))
    print("  Best epoch:", hist.get("best_epoch"))
    print("  Best metrics in hist:", hist.get("best_metrics"))
    print("  Total history steps:", len(hist.get("history", [])))
    for h in hist.get("history", []):
        if h.get("is_best") or h.get("epoch") in (1, 2, 5, 7, 10, 20, 30, 40, 45):
            print(f"  Epoch {h['epoch']:2d}: loss={h['train_loss']:.3f}/{h['val_loss']:.3f}, alpha_mae={h['alpha_mae']:.4f}, dense_r2={h['alpha_r2_dense']:.4f}, score={h['composite_score']:.4f}, is_best={h['is_best']}")
