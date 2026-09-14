## 2026-09-09T06:52:07Z

You are teamwork_preview_worker for Milestones M1 & M2 (Simulation Matrix & Dual Friction Batch Execution).
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\worker_m1_m2
Project root is: e:\water_hammer_research\wellbore_moc_method
Path to ORIGINAL_REQUEST.md: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md (MANDATORY: Read it first!)
Scope document: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator\PROJECT.md
Survey reports to reference:
- e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_moc\survey_moc_report.md
- e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_dataset\survey_dataset_report.md
- e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_features\survey_features_report.md

Tasks:
1. Read ORIGINAL_REQUEST.md, PROJECT.md, and the survey reports.
2. Implement experiments/sensitivity/generate_matrix.py:
   - Define baseline wellbore: L=5000.0 m, D=0.1397 m, rho=1000.0 kg/m3, nu=1.0e-6 m2/s, a=1450.0 m/s, roughness=4.5e-5 m, V0=1.0 m/s, H0=300.0 m, H_ext=100.0 m, theta=0.0, toe_bc='dead_end', tf=40.0 s, tc=0.05 s, ts=0.5 s.
   - Baseline 3-fracture system: x_f=[4000.0, 4020.0, 4040.0] m, C_H=[1e-5, 1e-5, 1e-5] m2, k_leak=[1e-4, 1e-4, 1e-4] m^(5/2)/s, R_p=[0.0, 0.0, 0.0] s2/m5, w_i=[1/3, 1/3, 1/3].
   - Generate OAT (One-At-a-Time) single-variable ablation matrix across 6 physical parameters:
     * x_f location scan: [3500.0, 3800.0, 4100.0, 4400.0, 4700.0]
     * C_H head compliance: [1e-7, 1e-6, 1e-5, 3e-5, 1e-4]
     * k_leak leakoff: [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
     * R_p perforation resistance: [0.0, 100.0, 500.0, 2000.0, 10000.0]
     * w_i flow split: uniform [1/3, 1/3, 1/3], toe-dominant [0.1, 0.2, 0.7], heel-dominant [0.7, 0.2, 0.1], middle-dominant [0.1, 0.8, 0.1]
     * delta_x cluster spacing: [5.0, 10.0, 20.0, 35.0, 50.0]
   - Generate Orthogonal cross-parameter matrix (e.g. C_H in [1e-6, 1e-5, 1e-4] x delta_x in [10.0, 35.0] x k_leak in [1e-5, 1e-4]) to investigate damping confusion and interaction effects.
   - Save case manifest to output/fracture_parameter_sensitivity/manifest.json.
3. Implement experiments/sensitivity/run_simulation.py:
   - For EVERY condition in OAT and orthogonal matrix, run PAIRED simulations: one with Darcy-Weisbach steady friction (friction='steady'), one with Brunone unsteady friction (friction='brunone').
   - Use moc_simulate.wellbore_moc.solve_moc and MocConfig with exact CFL=1.0.
   - Guarantee 100% numerical convergence, exact mass conservation, zero NaN/Inf, and verified positive head margin min(Hf - H_ext) > 0.
   - Save standard schema moc_lhs_v2.1 compatible NPZ files into output/fracture_parameter_sensitivity/data/case_*.npz with all 41 metadata and timeseries keys.
   - Save high-precision CSV files into output/fracture_parameter_sensitivity/timeseries_csv/case_*.csv with columns t, H_wh, Q_wh, H_f1, Q_f1, ...
4. Run experiments/sensitivity/run_simulation.py to generate all dataset files. Ensure all cases PASS.
5. Verify output files on disk (NPZ and CSV files exist, check non-empty, check schema).
6. Write handoff.md in your working directory e:\water_hammer_research\wellbore_moc_method\.agents\worker_m1_m2\ and send a message back to parent when done.
