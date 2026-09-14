## 2026-09-11T11:39:02Z

You are the Project Orchestrator for the MOC_V2 report refactoring and sensitivity study project.

Workspace Root: e:\water_hammer_research\wellbore_moc_method
Your Working Directory: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_2
Authoritative User Request: e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md (Refer to the latest section under '## Follow-up — 2026-09-11T11:37:49Z')

### Core Objectives:
1. R1. 理论主报告精炼重构:
   - Modify `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md`:
     - Delete Chapters 6 and 7.
     - Extract/split Chapters 3, 4, and 5 (archive their contents/figures to make room for independent experimental report).
     - Retain and refine Chapter 1 (V1 defects, original erroneous formula, Figure 0) and Chapter 2 (complete MOC governing equations, friction, compliance decoupling, perforation choke, ramp boundary, self-consistent initial field, Newton iteration convergence proof, and 2.6.4 five typical fracture types).
   - Update `docs/moc_v2_technical_report/build_report.py` to ensure the theoretical report builds and all assertions pass.

2. R2. 现场级多裂缝参数敏感性正演流水线:
   - Implement `docs/moc_v2_technical_report/run_sensitivity_study.py` using `moc_simulate.v2` production kernel to run all forward simulations:
     - Base Case: L=5000m, 3 clusters at [4500, 4510, 4520]m (start 4500m, spacing 10m, toe dead zone 480m), Cf=0.01 m^2, k_leak=1.0e-4 m^2.5/s, Kp=5.43e5 s^2/m^5, tc=1.0s.
     - Topic 1 (Fracture count): Nc in [1, 2, 3, 4, 5, 6, 7, 8], start 4500m, spacing 10m.
     - Topic 2 (Cluster spacing): 3 clusters, start 4500m, spacing d in [5, 10, 15, 20, 25, 30, 50, 80]m.
     - Topic 3 (Compliance storage): Cf in [0.002, 0.005, 0.010, 0.020, 0.030] m^2.
     - Topic 4 (Quasi-Darcy leakoff): k_leak in [0.2, 0.6, 1.0, 3.0, 10.0]e-4 m^2.5/s (including Type V fault strong leakoff).
     - Topic 5 (Perforation choke impedance): Kp in [1.5, 3.5, 5.43, 10.0, 25.0]e5 s^2/m^5 (perforation holes 16, 8, 6, 4, 2).
     - Topic 6 (Shut-in ramp duration): tc in [0.0, 0.5, 1.0, 1.5, 2.0]s.
     - Topic 7 (Multi-cluster flow intake combinations): 5 typical cases ([med,med,med] uniform; [high,med,med] heel breakthrough; [high,med,high] saddle; [med,med,high] toe reverse dominance; [dead,med,high] heel sandout).
   - Total 38+ simulation cases, all solved successfully without NaN/Inf.

3. R3. Nature-Grade 7 Independent Figure Plates (Rainbow 2D Cepstrogram):
   - Generate Figure 1 to Figure 7 in `docs/moc_v2_technical_report/sensitivity_figures/` in both 300 DPI PNG and vector SVG formats (14 files total).
   - Each plate contains:
     - Panel a: 100s full-timeseries and shut-in initial head waveform comparison.
     - Panel b: 1D real cepstrum curves with vertical markers for true fracture positions.
     - Panel c/d: Rainbow colormap (`cmap='rainbow'`) 2D continuous cepstrogram with fracture depth lines, WITHOUT text detection rate statistics overlay.
   - Strictly follow Nature figure standards (sans-serif font, clear typography, professional layout).

4. R4. Independent Academic Analysis Report:
   - Write `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md` with complete introduction, Base Case definition, in-depth physical mechanisms for all 7 topics, embedded HD figures with bilingual captions, summary tables, mechanics and acoustic discussions (wave attenuation, choke, micro-spacing reverberation, rebound storage release, fault leakoff dewatering, intake non-uniformity), and full SI units/definitions.

5. Regression and Quality:
   - Ensure all existing pytest tests (99 tests) continue to pass 100%.
