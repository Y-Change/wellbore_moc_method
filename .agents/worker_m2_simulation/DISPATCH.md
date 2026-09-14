## 2026-09-11T11:55:35Z

You are Worker M2 (7 Sensitivity Studies Simulation Pipeline & Figure Generator).
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\worker_m2_simulation
Your parent is orchestrator_3 (id: 378ebea8-5954-4263-ba6d-94834c1aab6c).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Input Context:
- User Request: e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md (You MUST read this first!)
- Project Plan: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_3\PROJECT.md
- Survey Findings from Explorer 1: e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_1_o3\survey_moc_solver.md and handoff.md (Contains exact parameters, Base Case, Topic 1-7, and Topic 7's 5 cases)
- Survey Findings from Explorer 3: e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_3_o3\survey_vis_and_tests.md and handoff.md (Contains figure formatting, Nature styling, Rainbow colormap, rasterized SVG advice, Hamming window for 2D cepstrogram)

File Write Ownership:
You exclusively own and may edit:
- docs/moc_v2_technical_report/run_sensitivity_study.py
- docs/moc_v2_technical_report/sensitivity_figures/ (and its contents: fig1 through fig7 png and svg)
Do NOT touch output/fracture_parameter_sensitivity/ or tests/ or build_report.py!

Task Description (Milestone M2 & M3 / R2 & R3):
1. Read ORIGINAL_REQUEST.md and the survey reports.
2. Implement `docs/moc_v2_technical_report/run_sensitivity_study.py`:
   - Base Case: L=5000m, D=0.1397m, a=1450m/s, V0=1.0m/s, H0=300m, Hext=100m, 3 fractures at [4500, 4510, 4520]m, Cf=0.01m^2, kleak=1.0e-4 m^2.5/s, Kp=5.43e5 s^2/m^5, tc=1.0s (cosine ramp, Brunone friction).
   - Topic 1: Nc in [1, 2, 3, 4, 5, 6, 7, 8], start at 4500m, 10m spacing, w=1/Nc (8 runs).
   - Topic 2: 3 fractures, start at 4500m, spacing d in [5, 10, 15, 20, 25, 30, 50, 80]m (8 runs).
   - Topic 3: Cf in [0.002, 0.005, 0.010, 0.020, 0.030]m^2 (5 runs).
   - Topic 4: kleak in [0.2, 0.6, 1.0, 3.0, 10.0] * 1e-4 m^2.5/s (5 runs).
   - Topic 5: Kp in [1.5, 3.5, 5.43, 10.0, 25.0] * 1e5 s^2/m^5 (16, 8, 6, 4, 2 holes) (5 runs).
   - Topic 6: tc in [0.0, 0.5, 1.0, 1.5, 2.0]s (5 runs).
   - Topic 7: 5 intake combinations (use exact values from Section 3.3 of survey_moc_solver.md):
     * Case 7.1 【中，中，中】: w=[1/3, 1/3, 1/3], Cf=[0.01, 0.01, 0.01], kleak=[1.0, 1.0, 1.0]*1e-4, Kp=[5.43, 5.43, 5.43]*1e5
     * Case 7.2 【高，中，中】: w=[0.50, 0.25, 0.25], Cf=[0.02, 0.01, 0.01], kleak=[2.0, 1.0, 1.0]*1e-4, Kp=[2.5, 5.43, 5.43]*1e5
     * Case 7.3 【高，中，高】: w=[0.45, 0.10, 0.45], Cf=[0.02, 0.004, 0.02], kleak=[2.0, 0.4, 2.0]*1e-4, Kp=[2.5, 12.0, 2.5]*1e5
     * Case 7.4 【中，中，高】: w=[0.25, 0.25, 0.50], Cf=[0.01, 0.01, 0.02], kleak=[1.0, 1.0, 2.0]*1e-4, Kp=[5.43, 5.43, 2.5]*1e5
     * Case 7.5 【死，中，高】: w=[0.01, 0.35, 0.64], Cf=[0.0005, 0.01, 0.02], kleak=[0.05, 1.0, 2.2]*1e-4, Kp=[8.0e7, 5.43e5, 2.0e5]
   - Parallel Execution: Use ProcessPoolExecutor with 10-12 workers so that all 41 simulations run in ~4-5 minutes. Ensure store_full_field=False to keep memory usage minimal.
   - Signal Processing:
     * 1D real cepstrum via `compute_cepstrum_1d`.
     * 2D continuous cepstrogram via `compute_cepstrogram_2d(..., window='hamming')` (CRITICAL: must use window='hamming' to avoid SciPy Kaiser tuple crash).
   - Figure Plotting (Nature-Grade):
     * Create directory `docs/moc_v2_technical_report/sensitivity_figures/` if not exists.
     * Figures: `fig1_fracture_count_sensitivity`, `fig2_fracture_spacing_sensitivity`, `fig3_compliance_sensitivity`, `fig4_leakoff_sensitivity`, `fig5_perforation_sensitivity`, `fig6_ramp_closure_sensitivity`, `fig7_intake_capacity_combinations`.
     * Each figure exported as both 300 DPI PNG and vector SVG (14 files total).
     * Panel a: 100s full time waveform and early valve closure zoom.
     * Panel b: 1D real cepstrum curves with vertical dashed lines marking true fracture positions.
     * Panel c/d: 2D continuous cepstrograms with Rainbow colormap (`cmap='rainbow'`), true fracture vertical depth lines, and STRICTLY NO text detection criteria.
     * For SVG exports, set `rasterized=True` on pcolormesh to ensure SVG files remain lightweight (< 300 KB).
3. Execute `python docs/moc_v2_technical_report/run_sensitivity_study.py`.
4. Verify all 41 simulations finish with NO NaN/Inf and all 14 figure files exist and have non-zero size.
5. Run `pytest` to confirm 100% green pass on all 99+ tests.
6. Write your handoff report to:
   e:\water_hammer_research\wellbore_moc_method\.agents\worker_m2_simulation\handoff.md
   documenting all observations, verification commands, simulation outputs, and figure file paths.
7. Send completion message to parent (378ebea8-5954-4263-ba6d-94834c1aab6c).
