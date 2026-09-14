# Test Writer Dispatch

You are teamwork_preview_test_writer.
Working directory: e:\water_hammer_research\wellbore_moc_method\.agents\test_writer
Project root: e:\water_hammer_research\wellbore_moc_method
Original request: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md
Scope document: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator\PROJECT.md

## 2026-09-09T06:52:07Z
You are teamwork_preview_test_writer.
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\test_writer
Project root is: e:\water_hammer_research\wellbore_moc_method
Path to ORIGINAL_REQUEST.md: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md (MANDATORY: Read it first!)
Scope document: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator\PROJECT.md

Tasks:
1. Read ORIGINAL_REQUEST.md and PROJECT.md.
2. Implement a comprehensive E2E pytest test suite in tests/test_fracture_sensitivity_e2e.py.
3. The test suite must independently and strictly verify all acceptance criteria:
   - Tier 1: Existence and convergence of output simulation files in output/fracture_parameter_sensitivity/data/case_*.npz and timeseries_csv/case_*.csv. Test 100% convergence, zero NaN/Inf, valid shapes and non-trivial amplitudes.
   - Tier 2: Schema moc_lhs_v2.1 compatibility (all 41 metadata/array keys, schema_version=='moc_lhs_v2.1', friction in ['steady', 'brunone'], positive margin min(Hf - H_ext) > 0).
   - Tier 3: Metrics table and summary JSON completeness (output/fracture_parameter_sensitivity/tables/sensitivity_metrics.csv containing Joukowsky drop, max gradient, 5-window RMS head attenuation rate, FFT high frequency >1.5Hz energy ratio, 1D real cepstrum peak depth and amplitude, 2D sliding window cepstrum spatial resolution; output/fracture_parameter_sensitivity/tables/sensitivity_summary.json containing sensitivity rankings and impact levels).
   - Tier 4: Publication figures verification (at least 4 figures in output/fracture_parameter_sensitivity/figures/: fig1_wavefront_step_gradient.png, fig2_envelope_rms_decay.png, fig3_frequency_spectral_dissipation.png, fig4_cepstrum_rayleigh_resolution.png, verify DPI >= 200 via PIL/image headers, verify companion SVG files exist).
   - Tier 5: Academic report output/fracture_parameter_sensitivity/README.md completeness (file size >= 5000 bytes, contains all 8 required chapter headings: Executive Summary, Physical Models & Discretization, Experimental Design, Waveform Step & Gradient Sensitivity, Dual Friction Attenuation & Damping Confusion, Frequency Dissipation & Energy Ratios, Cepstrum Peak Response & Rayleigh Resolution Limit, and Implications for Deep Neural Operator Inversion).
4. Create TEST_READY.md at project root e:\water_hammer_research\wellbore_moc_method\TEST_READY.md and in your working directory with the test runner command and coverage checklist.
5. Run pytest on the test suite (it will fail on uncompleted downstream items initially, or use conditional skip/markers where appropriate, or verify the test suite syntax and readiness).
6. Write handoff.md and send a message back to parent when done.
