# Project: MOC_V2 现场工况正演与参数敏感性分析

## Architecture
- **Solver Engine**: `moc_simulate/v2` production kernel (physical decoupling, Darcy continuity, Newton-Raphson coupled fracture nodes, Brunone unsteady friction, cosine ramp closure).
- **Simulation Pipeline**: `docs/moc_v2_technical_report/run_sensitivity_study.py` orchestrating Base Case and 7 sensitivity topics (41 simulation runs) using multi-process parallel execution.
- **Visualization Assets**: `docs/moc_v2_technical_report/sensitivity_figures/` containing Figure 1-7 in both 300 DPI PNG and vector SVG format (14 files total).
- **Technical Reports**:
  - Theory Report: `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md` (streamlined to focus purely on Ch 1 & 2 theory, validated by `build_report.py`).
  - Simulation Sensitivity Report: `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md` (new 12-chapter comprehensive academic report).
- **Regression Suite**: `pytest` full test suite (99+ tests) maintained at 100% pass rate.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Theory Report Streamlining | Prune Ch 3-7 from `MOC_V2_Physics_Upgrade_Report.md`, retain & polish Ch 1 (V1 defects, equations, Fig 0) & Ch 2 (governing equations, friction, compliance, perforation, ramp, initial field, Newton convergence, 5 fracture types) | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Report Build Validation | Update `build_report.py` (single source of truth for theory report) so that all 62 keywords, balanced LaTeX delimiters, and character length checks PASS 100% | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Base Case Field Simulation | $L=5000\mathrm{m}, D=0.1397\mathrm{m}, a=1450\mathrm{m/s}, V_0=1.0\mathrm{m/s}, H_0=300\mathrm{m}, H_{ext}=100\mathrm{m}$, 3 fractures at $[4500, 4510, 4520]\mathrm{m}, C_f=0.01\mathrm{m^2}, k_{leak}=1.0\times 10^{-4}\mathrm{m^{2.5}/s}, K_p=5.43\times 10^5\mathrm{s^2/m^5}, t_c=1.0\mathrm{s}$ | M2 | ORIGINAL_REQUEST §R2 |
| 4 | Topic 1 Simulation (Fracture Count) | $N_c \in [1, 2, 3, 4, 5, 6, 7, 8]$, start at 4500m, spacing 10m, equal weight $w=1/N_c$ (8 runs) | M2 | ORIGINAL_REQUEST §R2 |
| 5 | Topic 2 Simulation (Fracture Spacing) | 3 fractures, start at 4500m, spacing $d \in [5, 10, 15, 20, 25, 30, 50, 80]\mathrm{m}$ (8 runs) | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Topic 3 Simulation (Compliance) | $C_f \in [0.002, 0.005, 0.010, 0.020, 0.030]\mathrm{m^2}$ (5 runs) | M2 | ORIGINAL_REQUEST §R2 |
| 7 | Topic 4 Simulation (Leak-off) | $k_{leak} \in [0.2, 0.6, 1.0, 3.0, 10.0]\times 10^{-4}\mathrm{m^{2.5}/s}$ including Type V fault leak-off (5 runs) | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Topic 5 Simulation (Perforation Resistance) | $K_p \in [1.5, 3.5, 5.43, 10.0, 25.0]\times 10^5\mathrm{s^2/m^5}$ (16, 8, 6, 4, 2 holes) (5 runs) | M2 | ORIGINAL_REQUEST §R2 |
| 9 | Topic 6 Simulation (Valve Closure Ramp) | $t_c \in [0.0, 0.5, 1.0, 1.5, 2.0]\mathrm{s}$ with smooth cosine closure (5 runs) | M2 | ORIGINAL_REQUEST §R2 |
| 10 | Topic 7 Simulation (Intake Combinations) | 5 typical cases: [中,中,中], [高,中,中], [高,中,高], [中,中,高], [死,中,高] with physical $w, C_f, k_{leak}, K_p$ (5 runs) | M2 | ORIGINAL_REQUEST §R2 |
| 11 | One-click Pipeline Script | Implement `docs/moc_v2_technical_report/run_sensitivity_study.py` executing all 41 cases in parallel via ProcessPoolExecutor under 5 minutes | M2 | ORIGINAL_REQUEST §R2 |
| 12 | Signal Feature Extraction | Compute 1D real cepstrum and 2D continuous sliding-window cepstrogram with `window="hamming"` | M2 | ORIGINAL_REQUEST §R3 |
| 13 | Nature-grade Figures 1-7 (14 files) | Generate Figure 1-7 in `docs/moc_v2_technical_report/sensitivity_figures/` as both 300 DPI PNG and vector SVG | M3 | ORIGINAL_REQUEST §R3 |
| 14 | Rainbow 2D Cepstrograms | Strict `cmap='rainbow'` for 2D cepstrograms, true fracture vertical depth lines, and STRICTLY NO text detection criteria | M3 | ORIGINAL_REQUEST §R3 |
| 15 | Simulation Sensitivity Report Drafting | Comprehensive 12-chapter academic report `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md` with in-depth physics, equations, SI units, and figure embeddings | M4 | ORIGINAL_REQUEST §R4 |
| 16 | Full Test Suite Regression Pass | Full pytest regression pass: 99+ tests 100% green | M5 | ORIGINAL_REQUEST §Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Theory Report Streamlining & Build Validation (R1) | Streamline `MOC_V2_Physics_Upgrade_Report.md` via `build_report.py`, retain Ch 1 & 2, verify all 62 keywords & LaTeX delimiters PASS | none | PLANNED |
| M2 | 7 Sensitivity Studies Simulation Pipeline (R2) | Implement `run_sensitivity_study.py`, execute Base Case and 7 topics (41 cases) with zero NaN/Inf | none | PLANNED |
| M3 | Nature-grade Figure 1-7 Generation (R3) | Output 14 files (PNG+SVG) in `sensitivity_figures/`, Rainbow 2D cepstrograms, true depth lines, no detection text | M2 | PLANNED |
| M4 | MOC_V2 Simulation Sensitivity Report (R4) | Author complete academic report `MOC_V2_Simulation_Sensitivity_Report.md` embedding Figures 1-7 and deep physics analysis | M1, M3 | PLANNED |
| M5 | Full Regression Verification (R5) | Run pytest across all 99+ tests to guarantee 100% pass rate with zero regressions | M1, M2, M3, M4 | PLANNED |

## Interface Contracts
### `build_report.py` ↔ `MOC_V2_Physics_Upgrade_Report.md`
- `build_report.py` writes `REPORT_CONTENT` to `MOC_V2_Physics_Upgrade_Report.md`.
- Content retains Chapter 1 & Chapter 2 (MOC governing equations, friction, compliance decoupling, perforation throttling, ramp boundary, initial state, Newton iteration convergence, 5 fracture types).
- All 62 required keywords must be present in `REPORT_CONTENT`.
- Delimiters `$$` and `$` must be strictly balanced.
- Length threshold adjusted to `>= 45000` characters.

### `run_sensitivity_study.py` ↔ `sensitivity_figures/`
- `run_sensitivity_study.py` produces or saves simulation data (time, wellhead head, fracture heads, 1D cepstrum, 2D cepstrogram) and generates Figure 1 to Figure 7.
- Output directory: `docs/moc_v2_technical_report/sensitivity_figures/`.
- File naming: `fig1_fracture_count_sensitivity.{png,svg}` to `fig7_intake_capacity_combinations.{png,svg}`.
- Colors: Panel c/d 2D cepstrogram uses `cmap='rainbow'`.
- Annotations: True fracture positions as dashed vertical lines, NO detection text or match percentages.

### Figures ↔ `MOC_V2_Simulation_Sensitivity_Report.md`
- Figures referenced as `./sensitivity_figures/fig{N}_*.png`.
- Detailed bilingual captions explaining Panel a, b, c/d for each figure.

## Code Layout
- `docs/moc_v2_technical_report/build_report.py` — Theory report builder
- `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md` — Theory report
- `docs/moc_v2_technical_report/run_sensitivity_study.py` — Simulation pipeline & plotting
- `docs/moc_v2_technical_report/sensitivity_figures/` — 14 figure files
- `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md` — Simulation sensitivity technical report
- `output/fracture_parameter_sensitivity/` — UNTOUCHED (protected for regression tests)
