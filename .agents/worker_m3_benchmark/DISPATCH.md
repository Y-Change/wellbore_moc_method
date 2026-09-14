## 2026-09-13T14:50:48Z

You are the expert benchmark and training worker for PaperC Phase 3: '面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究'.
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\worker_m3_benchmark\
Create your working directory if needed. Write your progress.md and handoff.md there.

MANDATORY FIRST STEP:
Read e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md (especially the section at ## 2026-09-13T14:28:38Z) and e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5\PROJECT.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

CONTEXT & EXISTING ASSETS:
- `PaperC_CJNO_Wellbore_Inversion/src/modules/layer_stripping.py` and `src/models/tg_dis_deeponet.py` have been implemented and verified with 24/24 passing unit tests!
- Dataset: `PaperC_CJNO_Wellbore_Inversion/data/cache_1k_t4096_c1024_g500.npz` (1,000 cases, 800 train, 100 val, 100 test).
- Pretrained baseline checkpoints exist in `PaperC_CJNO_Wellbore_Inversion/checkpoints/` (or can be evaluated via existing scripts).
- Phase 2 benchmark evaluation script: `experiments/evaluate_ablation.py`.

WORK OBJECTIVES (Milestone M3):
1. Create and execute training pipeline `PaperC_CJNO_Wellbore_Inversion/experiments/train_dis.py`:
   - Train `TGDISDeepONet` on the 1,000 cases dataset.
   - Use `CompositeInversionLoss` with appropriate weights, AdamW optimizer, cosine annealing schedule (~60-100 epochs, CPU completes in ~2 minutes).
   - Supervise existence probability \hat{e}_j and position refinement \Delta \hat{x}_j if advantageous for F1 score.
   - Save the best checkpoint to `checkpoints/tg_dis_deeponet_best.pt`.
2. Create and execute comprehensive benchmark script `PaperC_CJNO_Wellbore_Inversion/experiments/evaluate_benchmark.py`:
   - Benchmark all 5 models on the 100-case test set:
     1. 1D-ResNet
     2. 1D-FNO
     3. Vanilla DeepONet
     4. TG-DeepONet (Phase 2 best: `tg_relative_bias`)
     5. TG-DIS-DeepONet (Phase 3 proposed)
   - Evaluate all core metrics:
     * Overall \alpha R^2 and MAE
     * Dense multi-cluster (N_c \ge 4) \alpha R^2 (TARGET: R^2 > 0.75)
     * Sparse (N_c \le 3) and single-cluster (N_c=1) \alpha MAE (TARGET: < 0.08 overall, < 0.03 single/sparse)
     * 1D Wasserstein distance W_1 (TARGET: < 5.0 m)
     * Detection F1-score with \pm 10m tolerance (TARGET: > 0.88)
     * Simplex conservation max|\sum \alpha - 1| (TARGET: < 10^{-6})
   - Run systematic ablation (Vanilla DeepONet vs TG no bias vs TG with delay bias vs TG-DIS-DeepONet).
   - Export results to `output/phase3_benchmark_metrics.json` and `output/phase3_ablation_metrics.json`.
3. Create and execute noise robustness audit `PaperC_CJNO_Wellbore_Inversion/experiments/audit_noise_robustness.py`:
   - Stress test TG-DIS-DeepONet vs TG-DeepONet under:
     * Clean baseline
     * White Gaussian Noise (AWGN): SNR 30dB, 20dB, 10dB
     * Pink Noise (Colored 1/f noise): SNR 30dB, 20dB, 10dB
     * Sound speed perturbations: \pm 1%
   - Quantify relative performance degradation at 20dB strong noise (TARGET: degradation < 15%).
   - Export results to `output/phase3_noise_robustness_metrics.json`.
4. Validate that all Acceptance Criteria thresholds are met and documented.

DELIVERABLE:
Write a complete `handoff.md` in `e:\water_hammer_research\wellbore_moc_method\.agents\worker_m3_benchmark\`. Include exact metric tables, validation logs, command lines, and check against all Acceptance Criteria. When done, notify orchestrator via send_message.
