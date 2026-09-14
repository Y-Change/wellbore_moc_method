import torch
import numpy as np
import sys
sys.path.insert(0, ".")

from PaperC_CJNO_Wellbore_Inversion.src.modules.layer_stripping import DifferentiableLayerStripping
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import TGDISDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_detection_f1_score, compute_inversion_metrics

print("=" * 80)
print("AUDITOR P3: ADVERSARIAL STRESS TESTING")
print("=" * 80)

# ---------------------------------------------------------------------------
# TEST 1: DIS Layer Extreme Inputs & Gradient Flow
# ---------------------------------------------------------------------------
print("\n[Test 1] DIS Layer Extreme Inputs & Gradients:")
dis_layer = DifferentiableLayerStripping(in_dim=64, d_model=64, max_nc=6)

h_extreme = (torch.randn(8, 6, 64) * 1e4).detach().requires_grad_()
wavespeed_extreme = torch.tensor([[900.0], [2500.0], [1450.0], [1000.0], [2000.0], [1200.0], [1800.0], [1450.0]])
mask_extreme = torch.ones(8, 6, dtype=torch.bool)
mask_extreme[0, 3:] = False  # Partial mask
mask_extreme[1, :] = False   # All false mask!

out = dis_layer(h_extreme, wavespeed=wavespeed_extreme, mask=mask_extreme)
loss = out.h_stripped.sum() + out.gamma.sum() + out.admittance.sum()
loss.backward()

assert not torch.isnan(out.h_stripped).any(), "NaN in h_stripped!"
assert not torch.isinf(out.h_stripped).any(), "Inf in h_stripped!"
assert not torch.isnan(out.gamma).any(), "NaN in gamma!"
assert not torch.isnan(out.admittance).any(), "NaN in admittance!"
assert not torch.isnan(h_extreme.grad).any(), "NaN in input gradients!"
print("  [PASS] DIS Layer survived extreme 10^4 amplitude, extreme sound speeds, and partial/empty masks with zero NaN/Inf!")

# Case B: Check reflection coefficient bounds
assert (out.gamma <= 0.0).all(), "Gamma must be <= 0.0!"
assert (out.gamma >= -0.95).all(), "Gamma must be >= -0.95!"
assert (out.admittance >= 0.0).all(), "Admittance must be >= 0.0!"
print("  [PASS] Physical bounds strictly verified: Gamma in [-0.95, 0.0], Admittance >= 0.0!")

# ---------------------------------------------------------------------------
# TEST 2: Simplex Conservation Under Extreme Inputs
# ---------------------------------------------------------------------------
print("\n[Test 2] Simplex Conservation Under Adversarial Model Inputs:")
model = TGDISDeepONet()
model.eval()

B = 10
M = 6
# Random waves with noise and extreme amplitudes
dummy_wave = torch.randn(B, 2, 4096) * 50.0
dummy_ceps = torch.randn(B, 1, 1024) * 20.0
dummy_cond = torch.randn(B, 3) * 5.0
dummy_pos = torch.sort(torch.rand(B, M) * 4000.0 + 500.0, dim=-1)[0]
dummy_mask = torch.ones(B, M, dtype=torch.bool)
# Set various mask patterns (1 cluster to 6 clusters)
for b in range(B):
    n_act = (b % 6) + 1
    dummy_mask[b, n_act:] = False

with torch.no_grad():
    out_model = model(
        wave=dummy_wave,
        cepstrum=dummy_ceps,
        cond=dummy_cond,
        positions=dummy_pos,
        mask=dummy_mask,
    )

alpha = out_model["alpha"]
active_sums = torch.where(dummy_mask, alpha, torch.zeros_like(alpha)).sum(dim=-1)
max_dev = (active_sums - 1.0).abs().max().item()

print(f"  Max simplex deviation across {B} adversarial batches: {max_dev:.4e}")
assert max_dev < 1e-6, f"Simplex violation! Max dev {max_dev} >= 1e-6"
print("  [PASS] Simplex conservation strictly holds under adversarial inputs (max dev < 1e-6)!")

# ---------------------------------------------------------------------------
# TEST 3: Bipartite Greedy F1-Score Edge Cases
# ---------------------------------------------------------------------------
print("\n[Test 3] Detection F1-Score Bipartite Matching Edge Cases:")
# Case 3A: Exact match at boundary
p_pos = np.array([[4500.0, 4510.0, 4520.0]])
t_pos = np.array([[4500.0, 4510.0, 4530.0]]) # 3rd cluster distance = 10.0m (exactly tolerance)
p_ext = np.array([[0.9, 0.9, 0.9]])
t_msk = np.array([[True, True, True]])

res = compute_detection_f1_score(p_pos, p_ext, t_pos, t_msk, tolerance_m=10.0)
print(f"  Boundary match (dist=10.0m == tol): precision={res.precision:.2f}, recall={res.recall:.2f}, f1={res.f1:.2f}, tp={res.tp}")
assert res.tp == 3 and res.f1 == 1.0, "Boundary distance of 10.0m should match!"

# Case 3B: Just beyond boundary (10.001m)
t_pos_beyond = np.array([[4500.0, 4510.0, 4530.001]])
res_beyond = compute_detection_f1_score(p_pos, p_ext, t_pos_beyond, t_msk, tolerance_m=10.0)
print(f"  Beyond boundary (dist=10.001m > tol): precision={res_beyond.precision:.2f}, recall={res_beyond.recall:.2f}, f1={res_beyond.f1:.2f}, tp={res_beyond.tp}")
assert res_beyond.tp == 2, "Beyond boundary should not match!"

# Case 3C: All inactive / 0 clusters
p_ext_zero = np.array([[0.1, 0.1, 0.1]])
t_msk_zero = np.array([[False, False, False]])
res_zero = compute_detection_f1_score(p_pos, p_ext_zero, t_pos, t_msk_zero, tolerance_m=10.0)
print(f"  Zero active clusters: precision={res_zero.precision:.2f}, recall={res_zero.recall:.2f}, f1={res_zero.f1:.2f}")
assert res_zero.tp == 0 and res_zero.fp == 0 and res_zero.fn == 0, "Zero active should have 0 TP/FP/FN!"

print("  [PASS] Detection F1 bipartite matching passes all edge cases!")
print("\nAll adversarial tests PASSED successfully!")
