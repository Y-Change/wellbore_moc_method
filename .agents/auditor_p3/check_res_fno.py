import torch
import numpy as np
import sys
sys.path.insert(0, ".")

from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.experiments.evaluate_benchmark import load_model, evaluate_single_model

device = torch.device("cpu")
test_ds = PilotInversionDataset(split="test")
test_loader = test_ds.get_dataloader(batch_size=100, shuffle=False)
test_batch = next(iter(test_loader))

weights_dir = "PaperC_CJNO_Wellbore_Inversion/output/weights"
ckpt_dir = "PaperC_CJNO_Wellbore_Inversion/checkpoints"

res_model = load_model("resnet", weights_dir, ckpt_dir, device)
fno_model = load_model("fno", weights_dir, ckpt_dir, device)

m_res, out_res = evaluate_single_model(res_model, "resnet", test_batch, device)
m_fno, out_fno = evaluate_single_model(fno_model, "fno", test_batch, device)

print("ResNet alpha shape:", out_res["alpha"].shape)
print("FNO alpha shape:   ", out_fno["alpha"].shape)

diff_alpha = (out_res["alpha"] - out_fno["alpha"]).abs().max().item()
print("Max abs difference between ResNet and FNO alpha predictions:", diff_alpha)

print("ResNet alpha mae:", m_res["alpha_mae"])
print("FNO alpha mae:   ", m_fno["alpha_mae"])

print("Sample 0 ResNet alpha:", out_res["alpha"][0].numpy())
print("Sample 0 FNO alpha:   ", out_fno["alpha"][0].numpy())
print("Sample 0 Ground truth alpha:", test_batch["alpha"][0].numpy())
print("Sample 0 Mask:              ", test_batch["mask"][0].numpy())
