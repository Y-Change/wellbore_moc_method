## 2026-09-09T07:05:35Z
You are teamwork_preview_worker for Milestone M3 (Multi-Dimensional Feature Extraction & Metrics Matrix).
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\worker_m3_features
Project root is: e:\water_hammer_research\wellbore_moc_method
Path to ORIGINAL_REQUEST.md: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md (MANDATORY: Read it first!)
Scope document: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator\PROJECT.md
Survey report: e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_features\survey_features_report.md
E2E test suite: e:\water_hammer_research\wellbore_moc_method\tests\test_fracture_sensitivity_e2e.py
Data directory: e:\water_hammer_research\wellbore_moc_method\output\fracture_parameter_sensitivity\data\
Manifest file: e:\water_hammer_research\wellbore_moc_method\output\fracture_parameter_sensitivity\manifest.json

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Tasks:
1. Read ORIGINAL_REQUEST.md, PROJECT.md, survey_features_report.md, and tests/test_fracture_sensitivity_e2e.py.
2. Implement experiments/sensitivity/extract_features.py:
   - Load each simulated NPZ case from output/fracture_parameter_sensitivity/data/case_*.npz and manifest.json.
   - Time-Domain Metrics:
     * Joukowsky theoretical drop dH_jouk = -a_adj * V0 / g, simulated drop dH_sim = H_wh(t_s + t_c + 0.05) - H_wh(t_s - 0.01), and relative error %.
     * Wavefront max gradient (dH/dt)_max = max_{t in [t_s, t_s + t_c + 0.1]} |dH_wh/dt|.
     * 5-Window RMS head attenuation: split post-shut-in into 5 equal time windows (e.g. 5 windows covering t in [t_s, t_f]), compute RMS1 to RMS5, retention % = RMS5 / RMS1 * 100%, and exponential decay constant alpha_rms via linear fit of ln(RMS) vs window center time.
   - Frequency-Domain Metrics:
     * FFT on post-shut-in signal (t >= t_s + t_c).
     * High-frequency (f > 1.5 Hz) energy ratio P_high / P_total * 100% (where P_total is power for f > 0.05 Hz).
   - 1D Real Cepstrum:
     * Compute negative real cepstrum response -c(q) = -Re{IFFT(ln(|FFT(x)| + eps))}.
     * Spatial mapping d = q * a_adj / 2.
     * Blind adaptive peak detection: detect peak depths and amplitudes, identify primary peak near fracture depth, compute localization error |d_peak - x_f,1|.
   - 2D Sliding Window Cepstrum & Rayleigh Resolution:
     * Dynamic range threshold (80 dB) on log-spectrum, compute coherent bandwidth B_coh.
     * Theoretical Rayleigh spatial resolution limit delta_d_min approx a_adj / (2 * B_coh) = 2L / N_harm_eff.
     * FWHM of primary cepstrum peak in depth domain.
   - Sensitivity Rankings & Impact Levels:
     * Compute parameter sensitivity ranking and impact level (High, Moderate, Low) for each fracture parameter (x_f, C_H, k_leak, R_p, w_i, delta_x) across step drop, RMS wave attenuation, FFT high-frequency ratio, and cepstrum peak amplitude.
     * Dual friction comparison: compute delta/ratio between steady Darcy and Brunone unsteady friction for all metrics to quantify unsteady friction contribution.
3. Save results to:
   - output/fracture_parameter_sensitivity/tables/sensitivity_metrics.csv (all cases and all metric columns).
   - output/fracture_parameter_sensitivity/tables/sensitivity_summary.json (hierarchical summary with baseline values, OAT sensitivity gradients, sensitivity rankings, impact levels, and steady vs Brunone comparative statistics).
4. Run tests/test_fracture_sensitivity_e2e.py specifically for Tier 3:
   pytest tests/test_fracture_sensitivity_e2e.py -m "tier3" -v
   Ensure all Tier 3 tests PASS.
5. Write handoff.md in your working directory e:\water_hammer_research\wellbore_moc_method\.agents\worker_m3_features\ and send a message back to parent when done.
