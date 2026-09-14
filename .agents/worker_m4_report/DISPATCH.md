## 2026-09-09T07:12:05Z

You are teamwork_preview_worker for Milestone M4 (Publication Figure Plates & Academic Research Report).
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\worker_m4_report
Project root is: e:\water_hammer_research\wellbore_moc_method
Path to ORIGINAL_REQUEST.md: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md (MANDATORY: Read it first!)
Scope document: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator\PROJECT.md
Survey reports:
- e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_features\survey_features_report.md
E2E test suite: e:\water_hammer_research\wellbore_moc_method\tests\test_fracture_sensitivity_e2e.py
Data & tables:
- e:\water_hammer_research\wellbore_moc_method\output\fracture_parameter_sensitivity\data\
- e:\water_hammer_research\wellbore_moc_method\output\fracture_parameter_sensitivity\tables\sensitivity_metrics.csv
- e:\water_hammer_research\wellbore_moc_method\output\fracture_parameter_sensitivity\tables\sensitivity_summary.json
- e:\water_hammer_research\wellbore_moc_method\output\fracture_parameter_sensitivity\manifest.json

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Tasks:
1. Read ORIGINAL_REQUEST.md, PROJECT.md, survey_features_report.md, and tests/test_fracture_sensitivity_e2e.py.
2. Implement experiments/sensitivity/plot_figures.py:
   - Nature-style publication graphics: Arial/sans-serif typography, clean axes without top/right spines, clear physical units.
   - Semantic color scheme: Blue (#0F4D92) for Steady Darcy, Red (#B64342) for Brunone unsteady friction, distinct palette for parameter tiers.
   - Export both 300 dpi PNG (>=200 dpi) and vector SVG.
   - Generate Plate 1: output/fracture_parameter_sensitivity/figures/fig1_wavefront_step_gradient.png & .svg (Joukowsky drop, wavefront arrival, gradient dH/dt across C_H, R_p, x_f).
   - Generate Plate 2: output/fracture_parameter_sensitivity/figures/fig2_envelope_rms_decay.png & .svg (5-window RMS attenuation and damping decay across k_leak and C_H, comparing steady vs Brunone).
   - Generate Plate 3: output/fracture_parameter_sensitivity/figures/fig3_frequency_spectral_dissipation.png & .svg (FFT power spectra, harmonic frequency roll-off, f > 1.5 Hz dissipation).
   - Generate Plate 4: output/fracture_parameter_sensitivity/figures/fig4_cepstrum_rayleigh_resolution.png & .svg (1D real cepstrum peak detection vs depth, and 2D sliding-window cepstrogram Rayleigh resolution limit against spacing delta_x).
3. Author an exhaustive, publication-grade academic research report:
   output/fracture_parameter_sensitivity/README.md
   - File size >= 5000 bytes (aim for 15KB-30KB of rigorous academic analysis).
   - Must contain the 8 specified chapters:
     1. Executive Summary & Problem Formulation
     2. Governing Equations, Discrete MOC Schemes & Dual-Friction Models
     3. Parametric Sensitivity Matrix & Numerical Setup
     4. Waveform Step & Wavefront Gradient Sensitivity Analysis
     5. Dual Friction Head Attenuation & Unsteady Damping Confusion
     6. Frequency-Domain Logarithmic Dissipation & High-Frequency Energy Ratios
     7. 1D/2D Real Cepstrum Peak Response & Rayleigh Spatial Resolution Limits
     8. Sensitivity Rankings & Implications for Deep Neural Operator Inversion
   - Ensure logical closure, exact numerical references from sensitivity_metrics.csv and sensitivity_summary.json, clear LaTeX formulations, embedded figure references, and quantitative sensitivity hierarchy.
4. Run full E2E test verification:
   pytest tests/test_fracture_sensitivity_e2e.py -v
   Verify all 18 test cases PASS 100%!
5. Write handoff.md in your working directory e:\water_hammer_research\wellbore_moc_method\.agents\worker_m4_report\ and send a message back to parent when done.

## 2026-09-13T15:23:48Z

You are the expert scientific visualizer and academic report author for PaperC Phase 3: '面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究'.
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\worker_m4_report\
Create your working directory if needed. Write your progress.md and handoff.md there.

MANDATORY FIRST STEP:
Read e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md (especially the section at ## 2026-09-13T14:28:38Z) and e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5\PROJECT.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

DATA & METRIC ASSETS FOR PLOTTING & REPORTING:
- Benchmark metrics: `PaperC_CJNO_Wellbore_Inversion/output/phase3_benchmark_metrics.json`
- Ablation metrics: `PaperC_CJNO_Wellbore_Inversion/output/phase3_ablation_metrics.json`
- Noise robustness metrics: `PaperC_CJNO_Wellbore_Inversion/output/phase3_noise_robustness_metrics.json`
- Historical weights & histories: `PaperC_CJNO_Wellbore_Inversion/output/weights/`
- Test dataset: `PaperC_CJNO_Wellbore_Inversion/data/cache_1k_t4096_c1024_g500.npz`
- Survey theory report: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_p3_theory\report.md`
- Survey codebase report: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_p3_codebase\report.md`
- Milestone M3 handoff report: `e:\water_hammer_research\wellbore_moc_method\.agents\worker_m3_benchmark\handoff.md`

WORK OBJECTIVES (Milestone M4):
1. Create and execute plotting pipeline `PaperC_CJNO_Wellbore_Inversion/experiments/plot_phase3_figures.py`:
   - Generate 5 publication-grade composite figures in `PaperC_CJNO_Wellbore_Inversion/output/figures/`:
     * `fig1_layer_stripping_mechanism`: Acoustic wave propagation, multi-path reverberation choke, layer-stripping pulse restoration, explicit \Gamma_j and Y_{b,j} extraction.
     * `fig2_tg_dis_architecture`: TG-DIS-DeepONet dual-track architecture, acoustic delay-bias matrix, and attention heatmap attribution aligning with acoustic travel time.
     * `fig3_benchmark_and_ablation`: 5-model benchmark comparison bar charts, dense multi-cluster true vs pred scatter plot, and 4-step ablation progression.
     * `fig4_noise_and_speed_robustness`: 30dB, 20dB, 10dB AWGN/Pink noise degradation curves, \pm 1% sound speed perturbation spatial error profiles.
     * `fig5_typical_cases_inversion`: Inversion profiles across 5 representative fracturing cases (uniform, heel-dominant, saddle, toe-dominant, sand screen-out dead cluster).
   - Export BOTH 300 DPI PNG and vector SVG formats for all 5 figures (10 files total).
   - Ensure clean typography, sans-serif fonts, professional color palettes, and clear physical annotations.

2. Author and compile the complete, publication-grade academic research technical report:
   `PaperC_CJNO_Wellbore_Inversion/phase3_inverse_scattering_report.md`
   - Structure of the report:
     * Abstract & Executive Summary (Highlighting core breakthroughs, jump in R^2 and robustness, and physical conservation).
     * Chapter 1: Introduction & Research Background (Horizontal well fracturing multi-cluster diagnostic challenges, water hammer transient flow, why traditional data-driven models suffer from the "equalization trap").
     * Chapter 2: Mathematical Foundation of 1D Wave Equation Inverse Scattering & Differentiable Layer Stripping (Full mathematical derivation of transfer matrix, characteristic equations, Schur/Bruckstein recursive de-choking, reflection coefficients \Gamma_j \in [-1, 0], branch admittance Y_{b,j} = -2 Y_0 \Gamma_j / (1 + \Gamma_j), and decoupling of linear acoustics from nonlinear friction/perforation dissipation).
     * Chapter 3: Explainable TG-DIS-DeepONet Neural Operator Architecture (Time-gating module, arrival-time alignment, acoustic delay-bias attention, dual-track discrete/continuous heads, strict simplex conservation proof, and loss functions).
     * Chapter 4: 1,000-Case Physical Dataset & Benchmark Methodology (Dataset schema, 5 baseline models, 4-step ablation ladder, evaluation metrics: R^2, MAE, W1, F1-score, simplex error).
     * Chapter 5: Benchmark Results & Comprehensive Ablation Analysis (In-depth discussion of 5-model comparison, dense cluster performance leap from -0.0294 to 0.1825, embedding Figure 1, 2, 3).
     * Chapter 6: Two-Stage Noise & Sound Speed Robustness Audit (30dB, 20dB, 10dB AWGN and Pink noise, \pm 1% sound speed perturbation resilience, embedding Figure 4, 5).
     * Chapter 7: Deep Physical Analysis of the Single-Channel Resolution Limit & Theoretical Caveats (Rigorous academic discussion of the fundamental time-domain sampling constraint: \Delta t = 14.65 ms vs round-trip travel time \Delta \tau = 13.79 ms for 10-20m clusters; single-channel ill-posedness; homomorphic cepstrum phase loss; and future roadmap toward downhole DAS and differentiable MOC closed-loop optimization).
     * Chapter 8: Conclusions, Engineering Recommendations & Future Work.
   - Embed all generated figures with rich markdown syntax, tables, formulas in LaTeX math format ($...$ and $$...$$).

3. Verify that all 7 Acceptance Criteria are thoroughly addressed and documented in the report.

DELIVERABLE:
Ensure all 10 figure files are generated in `PaperC_CJNO_Wellbore_Inversion/output/figures/` and `phase3_inverse_scattering_report.md` is complete and verified. Write a complete 5-component `handoff.md` in `e:\water_hammer_research\wellbore_moc_method\.agents\worker_m4_report\`. When done, notify orchestrator via send_message.
