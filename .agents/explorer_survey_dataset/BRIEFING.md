# BRIEFING — 2026-09-09T06:50:30Z

## Mission
Investigate dataset schemas (especially moc_lhs_v2.1 compatible NPZ and CSV time series), parameter spaces, sampling distributions, ablation/orthogonal designs, and output serialization standards across the repository.

## 🔒 My Identity
- Archetype: explorer
- Roles: survey dataset schemas, parameter spaces, and existing experiment scripts
- Working directory: e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_dataset
- Original parent: 0e5df0b9-cf56-4e73-90ab-a6315bcdc031
- Milestone: dataset schema and parameter space survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Write reports only to working directory .agents\explorer_survey_dataset
- Deliver survey_dataset_report.md and handoff.md

## Current Parent
- Conversation ID: 0e5df0b9-cf56-4e73-90ab-a6315bcdc031
- Updated: 2026-09-09T06:46:29Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`
  - `docs/MOC仿真原理与数据生成规范_v2.1.md`
  - `moc_simulate/` (`wellbore_moc.py`, `config.py`, `lhs_config.py`, `run_lhs_batch_simulate.py`, `build_reservoir_dataset.py`, `run_no_fracture_benchmark.py`, `stratified_bench.py`, `leakoff_multi.py`, `paths.py`)
  - `experiments/` (`exp_shutdown_friction_2d_cepstrum/`, `exp_window_sweep_comparison/`)
  - `analysis/` (`decay_analysis/decay_regression_cf_kleak.py`, `brunone_spacing_effect/`, `cepstrum/wlen_hop_sweep.py`, `unified_evaluation/`)
  - `PaperA井口多裂缝水击响应/`, `PaperB_考虑brunone的倒谱识别/`, `PaperC_CJNO_Wellbore_Inversion/`
  - `tests/test_final_acceptance.py`
  - `output/acceptance_10_test/data/case_00000.npz` (and metadata json, summary csv)
  - `neural_operator/dataset_surrogate.py`
- **Key findings**:
  - Schema `moc_lhs_v2.1` verified: 41 fields in `.npz`, strictly typed, preserving backward aliases and physical bijectivity.
  - Standard time series format: `moc_timeseries.csv` (`t, H_wh, Q_wh, H_f1, Q_f1, ...`).
  - Baseline and sampling distributions identified for $x_f, \Delta x, C_H, k_\text{leak}, w_i, R_p$, wellbore, and valve shutdown parameters.
  - Implementation mechanisms for OAT single-variable ablation, Cartesian orthogonal arrays, and Dirichlet-LHS mapped out.
  - Directory structure and file serialization standards defined for `output/fracture_parameter_sensitivity/`.
- **Unexplored areas**: None within the survey scope.

## Key Decisions Made
- Fully documented all 41 fields of `moc_lhs_v2.1` schema.
- Synthesized quantitative parameter ranges and recommended OAT/orthogonal grids.
- Completed comprehensive survey report `survey_dataset_report.md`.

## Artifact Index
- DISPATCH.md — incoming instructions
- BRIEFING.md — persistent working memory
- progress.md — liveness heartbeat
- survey_dataset_report.md — detailed survey report (completed)
- handoff.md — 5-component handoff report (drafting)
