# TEST_READY — E2E Acceptance Test Suite

## Executive Overview
The End-to-End (E2E) acceptance test suite for the **Fracture Parameter Sensitivity Ablation Experiment** is authored, syntactically verified, and registered in the test infrastructure at:
- **Test File Path**: `tests/test_fracture_sensitivity_e2e.py`
- **Scope & Contract Reference**: `.agents/orchestrator/PROJECT.md` & `ORIGINAL_REQUEST.md`
- **Schema Specification**: `docs/MOC仿真原理与数据生成规范_v2.1.md` (`moc_lhs_v2.1`, 41 keys)

The suite strictly verifies all five acceptance criteria tiers independently and without facades.

---

## Authoritative Sources of Expected Values
1. **Korteweg-Joukowsky Relationship**:
   $$\Delta H_{\text{jouk}} = -\frac{a_{\text{adj}} V_0}{g}$$
   Evaluated against simulated initial head step $\Delta H_{\text{sim}} = H_{\text{wh}}(t_s + 10\Delta t) - H_{\text{wh}}(t_s - 10\Delta t)$.
2. **Rayleigh Spatial Resolution Limit**:
   $$\Delta d_{\text{min}} = \frac{a_{\text{adj}}}{2 B_{\text{coh}}} = \frac{2L}{N_{\text{harm,eff}}} \approx \text{FWHM}_d$$
   Determined from coherent effective bandwidth $B_{\text{coh}}$ on log magnitude spectrum with 80 dB dynamic range.
3. **Discrete Mass Conservation & Head Margin**:
   $$\min_{t,i} [H_{f,i}(t) - H_{\text{ext}}] > 0, \quad \sum_{i=1}^n w_i = 1.0$$
4. **Publication Standards**:
   Nature Figure design language: $\ge 200\,\mathrm{dpi}$ (target 300 dpi), companion vector SVG with `<text>` nodes, semantic dual palette (`#0F4D92` steady blue, `#B64342` Brunone red).

---

## Test Execution Commands

### 1. Strict Mode (Default — Final Milestone / CI Audit)
Runs the entire acceptance suite. Fails with actionable diagnostics if any artifact or acceptance criteria is unsatisfied:
```powershell
pytest tests/test_fracture_sensitivity_e2e.py -v
```

### 2. Progressive Mode (Active Development / Pre-M4 Verification)
Allows passing during active upstream development while skipping pending deliverables with informative notices:
```powershell
$env:E2E_ALLOW_SKIP="1"; pytest tests/test_fracture_sensitivity_e2e.py -v
```

### 3. Tier-by-Tier Granular Execution
Execute tests specific to a single milestone:
```powershell
# Tier 1: Numerical Simulation & Convergence (Milestone M2)
pytest tests/test_fracture_sensitivity_e2e.py -m tier1 -v

# Tier 2: Schema moc_lhs_v2.1 & Physical Invariants (Milestone M2)
pytest tests/test_fracture_sensitivity_e2e.py -m tier2 -v

# Tier 3: Metrics Table & Summary JSON Completeness (Milestone M3)
pytest tests/test_fracture_sensitivity_e2e.py -m tier3 -v

# Tier 4: Publication Figure Plates & DPI (Milestone M4)
pytest tests/test_fracture_sensitivity_e2e.py -m tier4 -v

# Tier 5: Academic Research Report Completeness (Milestone M4)
pytest tests/test_fracture_sensitivity_e2e.py -m tier5 -v
```

### 4. Running Repo Unit & Physics Regression Tests
```powershell
pytest -m "not e2e"
```

---

## Acceptance Criteria Coverage Checklist

| Tier | Requirement / Verification Target | Test Method | Criteria & Invariants | Status |
| :--- | :--- | :--- | :--- | :---: |
| **0** | **Test Contract & Schema Spec** | `test_e2e_contract_and_metadata_spec` | All 41 keys, 4 figures, 8 chapters, valid paths | **PASS** |
| **1** | **Simulation Output Existence** | `test_tier1_output_directories_exist`<br>`test_tier1_case_files_exist_and_counts_match` | `data/case_*.npz` and `timeseries_csv/case_*.csv` exist and match | READY |
| **1** | **Numerical Convergence** | `test_tier1_npz_files_convergence_and_finite` | 100% convergence (`PASS`), zero NaN/Inf, $\Delta t$ monotonic, $\text{ptp}(H_{wh}) > 1.0\,\mathrm{m}$ | READY |
| **1** | **CSV Time Series Completeness** | `test_tier1_csv_timeseries_validity` | Non-empty ($>100\,\mathrm{B}$), `t, H_wh, Q_wh`, $\ge 100$ rows, zero NaN/Inf | READY |
| **2** | **41-Key Schema Compliance** | `test_tier2_all_41_keys_present_in_every_npz` | All 41 keys present in every NPZ case file | READY |
| **2** | **Schema Version & Friction Models** | `test_tier2_schema_version_and_friction_models` | `schema_version == 'moc_lhs_v2.1'`, covers both `steady` and `brunone` | READY |
| **2** | **Physical Pressure Head Margin** | `test_tier2_positive_head_margin` | $\min(H_f - H_{\text{ext}}) > 0$ across all cases (prevents cavitation) | READY |
| **2** | **Alias Consistency & Physical Feasibility** | `test_tier2_aliases_and_physical_constraints` | $x_f \equiv x_{f,\text{aligned}}$, $Cf \equiv C_H \ge 0$, $k_{\text{leak}} \ge 0$, $\sum w_i = 1.0$ | READY |
| **3** | **Metrics CSV Schema & Coverage** | `test_tier3_metrics_csv_existence_and_schema` | Contains Joukowsky, max gradient, 5-win RMS, FFT $>1.5\,\mathrm{Hz}$, 1D ceps, 2D resolution | READY |
| **3** | **Metrics Data Validity** | `test_tier3_metrics_csv_data_validity` | Finite numeric data, zero NaN/Inf, covers both `steady` and `brunone` | READY |
| **3** | **Summary JSON Rankings** | `test_tier3_summary_json_rankings_and_impact_levels` | Valid JSON with sensitivity rankings, impact levels, mechanisms | READY |
| **4** | **Publication Figures Existence** | `test_tier4_all_four_png_figures_exist_and_non_empty` | `fig1_wavefront_step_gradient.png`<br>`fig2_envelope_rms_decay.png`<br>`fig3_frequency_spectral_dissipation.png`<br>`fig4_cepstrum_rayleigh_resolution.png` ($\ge 5\,\mathrm{KB}$) | READY |
| **4** | **Figure Resolution $\ge 200\,\mathrm{DPI}$** | `test_tier4_figure_dpi_greater_than_200` | PIL Image header $\ge 200\,\mathrm{dpi}$ and dimensions $\ge 1200\times 800$ | READY |
| **4** | **Companion Vector SVG Figures** | `test_tier4_companion_svg_files_exist_and_valid` | Companion `.svg` present, size $\ge 1000\,\mathrm{B}$, valid `<svg>` root tags | READY |
| **5** | **Academic Report File & Size** | `test_tier5_readme_file_exists_and_size_ge_5000` | `output/fracture_parameter_sensitivity/README.md` $\ge 5000\,\mathrm{bytes}$ | READY |
| **5** | **All 8 Required Chapters** | `test_tier5_all_eight_chapter_headings_present` | All 8 chapters present: Executive Summary, Physical Models, Design, Waveform Step, Dual Friction, Frequency Dissipation, Cepstrum Response, Neural Inversion | READY |
| **5** | **Substantive Chapter Content** | `test_tier5_substantive_chapter_content` | Substantive content between headers ($\ge 150$ chars per section, no empty headers) | READY |

---

## Initial Verification Results
- **Contract Invariant Test**: `test_e2e_contract_and_metadata_spec` PASSED.
- **Strict Mode**: 1 PASSED, 17 FAILED (Expected pre-M1/M2 behavior: accurately diagnoses absence of downstream simulation data, metrics, figures, and report).
- **Progressive Mode (`E2E_ALLOW_SKIP=1`)**: 1 PASSED, 17 SKIPPED (Non-blocking for intermediate pipeline runs).
- **Regression Suite (`pytest -m "not e2e"`)**: 26 PASSED in 5.70s.
