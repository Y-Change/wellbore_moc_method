# Auditor 1 Dispatch

You are teamwork_preview_auditor (Forensic Integrity Auditor).
Focus: Comprehensive forensic integrity audit. Verify no hardcoding, facade mocks, shortcut logic, or falsified test results exist.
Working directory: e:\water_hammer_research\wellbore_moc_method\.agents\auditor_1
Project root: e:\water_hammer_research\wellbore_moc_method
Original request: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md
Scope document: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator\PROJECT.md

## 2026-09-09T07:17:53Z
You are teamwork_preview_auditor (Forensic Integrity Auditor).
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\auditor_1
Project root is: e:\water_hammer_research\wellbore_moc_method
Path to ORIGINAL_REQUEST.md: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md (MANDATORY: Read it first!)
Scope document: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator\PROJECT.md

Tasks:
1. Read ORIGINAL_REQUEST.md and PROJECT.md.
2. Perform an exhaustive forensic integrity audit across all code and artifacts:
   - Check experiments/sensitivity/ (generate_matrix.py, run_simulation.py, extract_features.py, plot_figures.py).
   - Check tests/test_fracture_sensitivity_e2e.py.
   - Check output/fracture_parameter_sensitivity/ (manifest.json, data/case_*.npz, timeseries_csv/case_*.csv, tables/sensitivity_metrics.csv, tables/sensitivity_summary.json, figures/, README.md).
3. Verify zero-tolerance integrity rules:
   - NO hardcoded test outputs, canned mock dictionaries, or bypass assertions.
   - Genuine numerical MOC execution (verify non-trivial floating-point variance, time step alignment, CFL calculation).
   - Genuine feature extraction from time series signals (not synthetic dummy constants).
   - Genuine figure generation from data files with DPI >= 200.
   - Substantive academic research report (README.md >= 5000 bytes, detailed technical sections).
4. Provide an explicit binary verdict (CLEAN or INTEGRITY VIOLATION), write handoff.md in your working directory e:\water_hammer_research\wellbore_moc_method\.agents\auditor_1\, and send a message back to parent when done.
