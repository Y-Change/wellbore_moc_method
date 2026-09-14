# Project: Fracture Parameter Sensitivity Ablation Experiment

## Architecture
- **MOC Physics Core**: `moc_simulate/wellbore_moc.py` with exact $CFL \equiv 1.0$ Courant adjustment, non-dissipative characteristic lines, double-velocity fracture node formulation, Newton-Raphson orifice drop ($R_p$), backward Euler compliance storage ($C_H$), steady-state dead-end initialization, and dual friction formulations (Darcy-Weisbach steady Zigrand-Swami vs. Brunone unsteady with Vardy decay and localized $\tanh$ smoothing).
- **Ablation & Simulation Pipeline**: Scripted generation of One-At-a-Time (OAT) single-variable scans ($x_f, C_H, k_{leak}, R_p, w_i, \Delta x$) and orthogonal interaction matrix, executing paired simulations for steady vs. Brunone friction, serializing to `output/fracture_parameter_sensitivity/data/case_*.npz` (schema `moc_lhs_v2.1` with 41 keys) and `output/fracture_parameter_sensitivity/timeseries_csv/case_*.csv`.
- **Time-Frequency & Cepstrum Feature Engine**: Automated extraction of Joukowsky theoretical and simulated step, wavefront max gradient $(\partial H/\partial t)_{max}$, 5-window RMS attenuation and exponential decay rate $\alpha_{RMS}$, FFT high-frequency ($f > 1.5\,\mathrm{Hz}$) energy ratio, 1D negative real cepstrum $-c(q)$ with anti-leakage blind peak detection, and 2D sliding-window Kaiser-Bessel cepstrum Rayleigh spatial resolution $\Delta d_{min} \approx a / (2 B_{coh})$. Outputs: `tables/sensitivity_metrics.csv` and `tables/sensitivity_summary.json`.
- **Publication Graphics & Dissemination Engine**: High-impact multi-panel figures ($\ge 300\,\mathrm{dpi}$, Arial font, semantic palette `#0F4D92` steady / `#B64342` Brunone, SVG+PNG) and comprehensive academic research report (`README.md`) detailing physics mechanisms, damping confusion, and neural inversion boundaries.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F1: Baseline Configuration & Parameter Bounds | Wellbore ($L=5000, D=0.1397, a=1450, V_0=1.0, H_0=300, H_{ext}=100$) and 3-fracture baseline definition | M1 | survey |
| 2 | F2: Single-Variable (OAT) Ablation Matrix | 6 physical axes ($x_f, C_H, k_{leak}, R_p, w_i, \Delta x$) with 5~7 levels each | M1 | survey |
| 3 | F3: Orthogonal Multi-Parameter Matrix | Factorial cross-grid for coupling effects ($C_H \times \Delta x \times k_{leak}$) | M1 | survey |
| 4 | F4: Dual Friction MOC Batch Simulation Engine | Parallel MOC runner supporting Darcy steady and Brunone unsteady friction with 100% convergence and zero initial perturbation | M2 | survey |
| 5 | F5: Data Serialization to Standard NPZ/CSV | Output schema `moc_lhs_v2.1` (41 keys) NPZ and high-precision CSV time series | M2 | survey |
| 6 | F6: Time-Domain Metrics Extraction | Joukowsky drop, wavefront $(\partial H/\partial t)_{max}$, 5-window RMS attenuation and $\alpha_{RMS}$ | M3 | survey |
| 7 | F7: Frequency-Domain Spectral & Roll-off Extraction | FFT energy ratio ($f > 1.5\,\mathrm{Hz}$), spectral dissipation indices | M3 | survey |
| 8 | F8: 1D Real Cepstrum Feature Extraction | Real cepstrum $-c(q)$, spatial mapping $d=qa/2$, blind adaptive peak detection | M3 | survey |
| 9 | F9: 2D Cepstrum & Rayleigh Resolution Limit | Kaiser-Bessel sliding window cepstrogram, coherent bandwidth $B_{coh}$, Rayleigh resolution $\Delta d_{min}$ | M3 | survey |
| 10 | F10: Sensitivity Ranking & Quantitative Tables | `tables/sensitivity_metrics.csv` (42+ columns) and hierarchical JSON summary with sensitivity rankings | M3 | survey |
| 11 | F11: Publication Figures ($\ge 200\,\mathrm{dpi}$) | 4 multi-panel plates (wavefront step, RMS envelope decay, FFT dissipation, 1D/2D cepstrum resolution) in 300 dpi PNG + SVG | M4 | survey |
| 12 | F12: Academic Research Report (`README.md`) | 8-chapter academic monograph detailing physical mechanisms, damping confusion, and neural network inversion boundaries | M4 | survey |
| 13 | F13: Comprehensive Verification & Forensic Audit | 100% convergence check, schema compliance, figure resolution check, and binary zero-tolerance forensic integrity audit | M5 | survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Experiment Design & Matrix Generation | F1, F2, F3: Generate complete OAT & orthogonal parameter matrix definitions and validation scripts | none | DONE |
| M2 | Dual-Friction Simulation & Dataset Generation | F4, F5: Execute paired MOC batch runs (Darcy vs Brunone), serialize 100% convergent `moc_lhs_v2.1` NPZ and CSV files | M1 | DONE |
| M3 | Multi-Dimensional Feature Extraction & Metrics | F6, F7, F8, F9, F10: Compute time-domain, spectral, 1D/2D cepstrum metrics, generate `sensitivity_metrics.csv` & JSON | M2 | DONE |
| M4 | Academic Report & Publication Figure Plates | F11, F12: Render 4 high-impact figure plates (>=200 dpi), write comprehensive academic `README.md` | M3 | DONE |
| M5 | E2E Verification & Forensic Integrity Audit | F13: Rigorous verification of acceptance criteria, test execution, and forensic integrity audit | M4 | DONE |

## Interface Contracts
### Matrix Generator ↔ Simulation Runner (M1 ↔ M2)
- Input: `experiments/sensitivity/experiment_manifest.json` specifying `cases: [{case_id, name, group, friction, wellbore_params, fracture_params, is_baseline}]`.
- Output format: Dictionary list consumed by MOC runner.

### Simulation Runner ↔ Feature Extractor (M2 ↔ M3)
- Path: `output/fracture_parameter_sensitivity/data/case_*.npz` (schema `moc_lhs_v2.1`) and `timeseries_csv/case_*.csv`.
- NPZ keys: `t`, `H_wh`, `Q_wh`, `x_f_aligned`, `compliance_head_m2`, `kleak_equiv`, `Rp`, `inflow_weight`, `Hw_ss`, `Hf_ss`, `Qf_ss`, `schema_version='moc_lhs_v2.1'`, `friction`, `wavespeed_adj`, etc.
- CSV format: `t, H_wh, Q_wh, H_f1, Q_f1, H_f2, Q_f2, ...`.

### Feature Extractor ↔ Figure & Report Generator (M3 ↔ M4)
- Path: `output/fracture_parameter_sensitivity/tables/sensitivity_metrics.csv` and `tables/sensitivity_summary.json`.
- Columns: `case_id, group, param_name, param_value, friction, dH_jouk_sim, dH_err_pct, max_gradient, rms_win1..5, rms_retention_pct, alpha_rms, high_freq_ratio_pct, cep_peak1_depth, cep_peak1_amp, rayleigh_limit_m, ...`.

## Code Layout
- `experiments/sensitivity/`:
  - `generate_matrix.py`: Parametric space generator & manifest builder.
  - `run_simulation.py`: Dual-friction batch simulation executor.
  - `extract_features.py`: Time-frequency, spectral & 1D/2D cepstrum feature extractor.
  - `plot_figures.py`: Publication-grade multi-panel figure generator.
- `output/fracture_parameter_sensitivity/`:
  - `data/`: `case_00000.npz` ... `case_*.npz` (`moc_lhs_v2.1`).
  - `timeseries_csv/`: `case_00000.csv` ... `case_*.csv`.
  - `tables/`: `sensitivity_metrics.csv`, `sensitivity_summary.json`.
  - `figures/`: `fig1_wavefront_step_gradient.png/svg`, `fig2_envelope_rms_decay.png/svg`, `fig3_frequency_spectral_dissipation.png/svg`, `fig4_cepstrum_rayleigh_resolution.png/svg`.
  - `README.md`: Comprehensive academic research report.
- `tests/`:
  - `test_fracture_sensitivity_e2e.py`: E2E test suite for schema, convergence, figures, metrics completeness.
