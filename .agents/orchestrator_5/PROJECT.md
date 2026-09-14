# Project: PaperC Phase 3 — 面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究

## Architecture
- **Acoustic Physics Layer**: 1D wave equation acoustic transfer matrix, layer-stripping recursion from heel to toe ($x_1 \to x_{N_c}$), explicit reflection coefficient $\Gamma_j$ and branch admittance $Y_{b,j}$.
- **Neural Operator**: TG-DIS-DeepONet with relative-window time gating, acoustic delay-bias self-attention, and differentiable layer-stripping operator.
- **Dual-Track Decoding**: Discrete precise heads ($\alpha_j$ with strict simplex conservation, existence probability $e_j$, position adjustment $\Delta x_j$, compliance $C_{f,j}$) and Continuous Trunk head ($m_\alpha(x)$ field with Voronoi conservation).
- **Benchmark & Audit Pipeline**: 1,000 cases physical dataset benchmark across 5 models, comprehensive ablation, noise stress test (30dB, 20dB, 10dB white/pink noise, $\pm 1\%$ sound speed), and automated metric assertions.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Differentiable Layer-Stripping Layer | Implements upstream transmission compensation $\prod (1+\Gamma_k)^2$, multi-path reflection stripping, and explicit outputs of $\Gamma_j$ & $Y_{b,j}$ | M1 | R1 |
| 2 | Layer-Stripping Unit Tests | Verifies gradient flow, energy attenuation compensation, and numeric stability of DIS layer | M1 | R1 |
| 3 | TG-DIS-DeepONet Architecture | Integrates DIS layer, relative time-gating, acoustic delay-bias attention, and dual-track heads | M2 | R2 |
| 4 | Existence & Position Refinement Heads | Predicts fracture initiation probability $\hat{e}_j$ and sub-meter position correction $\Delta \hat{x}_j$ | M2 | R2 |
| 5 | Detection F1-Score Metric ($\pm 10\mathrm{m}$) | Implements bipartite matching F1-score with $\pm 10\mathrm{m}$ spatial tolerance in `metrics.py` | M2 | R2 |
| 6 | Simplex Conservation ($\max\|\sum \alpha - 1\| < 10^{-6}$) | Enforces strict physical simplex conservation via Masked Softmax + normalization | M2 | R2 |
| 7 | Multi-Model Benchmark Training | Trains and benchmarks 1D-ResNet, 1D-FNO, DeepONet, TG-DeepONet, and TG-DIS-DeepONet on 1,000 cases | M3 | R3 |
| 8 | Systematic Ablation Study | Evaluates contributions of Time-Gating, Delay-Bias, DIS-Op, and Dual-Track loss | M3 | R3 |
| 9 | Two-Stage Robustness Audit | Evaluates models under 30dB, 20dB, 10dB white/pink noise and $\pm 1\%$ sound speed perturbation | M3 | R3 |
| 10 | Publication-Grade Figures | Generates Figures 1-5 (300 DPI PNG + SVG) meeting Nature/Q2 standards | M4 | R3/R4 |
| 11 | Comprehensive Research Report | Final report `PaperC_CJNO_Wellbore_Inversion/phase3_inverse_scattering_report.md` | M4 | R4 |
| 12 | Independent Multi-Agent Verification | Dual review, adversarial challenge, and forensic integrity audit (CLEAN verdict) | M5 | Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Differentiable Layer-Stripping Layer | Implement `src/modules/layer_stripping.py` and `tests/test_dis_layer.py` | Survey | DONE |
| M2 | TG-DIS-DeepONet Inversion Model | Implement `src/models/tg_dis_deeponet.py`, update `src/metrics.py` (F1-score), update heads | M1 | DONE |
| M3 | Benchmark & Robustness Audit | Train TG-DIS-DeepONet, run full benchmark across 5 models, run 30/20/10dB noise & speed audits | M2 | DONE |
| M4 | Report & Publication Figures | Generate Figures 1-5, compile `PaperC_CJNO_Wellbore_Inversion/phase3_inverse_scattering_report.md` | M3 | DONE |
| M5 | Review, Challenge & Forensic Audit | 2 Reviewers, 2 Challengers, 1 Forensic Auditor verifying all acceptance criteria | M4 | DONE |
| M6 | Sentinel Delivery | Final synthesis and formal completion report to Sentinel | M5 | DONE |

## Interface Contracts
### `src/modules/layer_stripping.py`
- Input: `h_patches` of shape `(batch, n_clusters, patch_len)`
- Outputs:
  - `h_stripped`: reconstructed / de-choked patch features `(batch, n_clusters, patch_len)`
  - `gamma`: explicit physical reflection coefficients `(batch, n_clusters)` in `[-1.0, 0.0]`
  - `admittance`: physical branch admittance $Y_{b,j}$ `(batch, n_clusters)`
  - `attenuation_factors`: cumulative transmission product `(batch, n_clusters)`

### `src/models/tg_dis_deeponet.py`
- Input: `waveforms` `(B, 2, 4096)`, `cepstrums` `(B, 1, 1024)`, `conds` `(B, 3)`, `positions` `(B, 6)`, `masks` `(B, 6)`
- Outputs dict:
  - `alpha`: `(B, 6)`, sum=1.0 on valid clusters, simplex error < 1e-6
  - `cf`: `(B, 6)`
  - `log_cf`: `(B, 6)`
  - `p_exist`: `(B, 6)`, existence probability in `[0, 1]`
  - `delta_x`: `(B, 6)`, position offset in `[-10, 10]` m
  - `gamma`: `(B, 6)`, reflection coefficients
  - `admittance`: `(B, 6)`, branch admittance
  - `m_alpha_grid`: `(B, 500)`, continuous flow field
  - `attn_weights`: attention maps from transformer

### `src/metrics.py`
- `compute_detection_f1_score(pred_pos, pred_exist, true_pos, true_mask, tolerance_m=10.0, prob_threshold=0.5)` -> `(precision, recall, f1)`
- `evaluate_metrics(preds, targets)` -> dict containing $R^2$ (overall, dense $N_c \ge 4$, sparse $N_c \le 3$), MAE ($\alpha$), $W_1$, F1, Simplex Max Error

## Code Layout
- `PaperC_CJNO_Wellbore_Inversion/src/modules/layer_stripping.py`: DIS operator
- `PaperC_CJNO_Wellbore_Inversion/src/models/tg_dis_deeponet.py`: Main inversion network
- `PaperC_CJNO_Wellbore_Inversion/src/metrics.py`: Metrics including F1-score
- `PaperC_CJNO_Wellbore_Inversion/experiments/train_dis.py`: Training pipeline
- `PaperC_CJNO_Wellbore_Inversion/experiments/evaluate_benchmark.py`: 5-model benchmark
- `PaperC_CJNO_Wellbore_Inversion/experiments/audit_noise_robustness.py`: Noise & speed stress tests
- `PaperC_CJNO_Wellbore_Inversion/experiments/plot_phase3_figures.py`: Publication figure generator
- `PaperC_CJNO_Wellbore_Inversion/phase3_inverse_scattering_report.md`: Final deliverable
- `PaperC_CJNO_Wellbore_Inversion/tests/test_dis_layer.py`: DIS layer unit tests
- `PaperC_CJNO_Wellbore_Inversion/tests/test_tg_dis_model.py`: Model integration unit tests
