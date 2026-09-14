# BRIEFING — 2026-09-11T11:56:00Z

## Mission
Implement and execute the 7 sensitivity studies simulation pipeline (`docs/moc_v2_technical_report/run_sensitivity_study.py`) and generate 14 Nature-grade figures (7 PNG + 7 SVG) in `docs/moc_v2_technical_report/sensitivity_figures/`.

## 🔒 My Identity
- Archetype: Worker M2 (Simulation Pipeline & Figure Generator)
- Roles: implementer, qa, specialist
- Working directory: e:\water_hammer_research\wellbore_moc_method\.agents\worker_m2_simulation
- Original parent: 378ebea8-5954-4263-ba6d-94834c1aab6c (orchestrator_3)
- Milestone: Milestone M2 & M3 (Tasks R2 & R3)

## 🔒 Key Constraints
- Exclusively own and edit: `docs/moc_v2_technical_report/run_sensitivity_study.py` and `docs/moc_v2_technical_report/sensitivity_figures/`.
- Do NOT touch `output/fracture_parameter_sensitivity/`, `tests/`, or `build_report.py`.
- DO NOT CHEAT: No dummy/facade implementations, genuine simulations via MOC solver.
- 41 simulations in total (Topics 1 to 7).
- Parallel execution via ProcessPoolExecutor with 10-12 workers, `store_full_field=False`.
- Windowing: `compute_cepstrogram_2d(..., window='hamming')` (CRITICAL: SciPy Kaiser tuple fix).
- Nature styling: Rainbow colormap (`cmap='rainbow'`), true fracture vertical depth lines, STRICTLY NO text detection criteria.
- SVG exports: `rasterized=True` on pcolormesh for SVG files < 300 KB.
- All 14 figure files (7 PNG, 7 SVG) must exist with non-zero size.
- Pytest must pass 100% green.

## Current Parent
- Conversation ID: 378ebea8-5954-4263-ba6d-94834c1aab6c
- Updated: 2026-09-11T11:56:00Z

## Task Summary
- **What to build**: `run_sensitivity_study.py` running 41 genuine MOC simulations across 7 sensitivity topics, computing 1D cepstrum and 2D continuous cepstrograms, and plotting 7 publication-quality multi-panel figures saved as PNG and SVG.
- **Success criteria**: 41 runs completed without NaN/Inf, 14 figure files generated, pytest 100% green pass.
- **Interface contracts**: `docs/moc_v2_technical_report/`
- **Code layout**: Python script in `docs/moc_v2_technical_report/`, figures in `docs/moc_v2_technical_report/sensitivity_figures/`.

## Key Decisions Made
- Use ProcessPoolExecutor with multiprocessing for parallel execution.
- Window 'hamming' for continuous cepstrograms.
- Ensure pcolormesh rasterized=True for vector SVG output.

## Change Tracker
- **Files modified**: None yet
- **Build status**: Pending
- **Pending issues**: None

## Quality Status
- **Build/test result**: Pending pytest check
- **Lint status**: Clean
- **Tests added/modified**: None (no edits to tests allowed)

## Loaded Skills
- **Source**: C:\Users\Change\.gemini\config\skills\nature-figure\SKILL.md
- **Local copy**: Pending
- **Core methodology**: Nature/high-impact journal multi-panel scientific plotting, typography, rasterized SVG for heavy mesh data.

## Artifact Index
- `docs/moc_v2_technical_report/run_sensitivity_study.py` — Main simulation and plotting script
- `docs/moc_v2_technical_report/sensitivity_figures/*.png, *.svg` — Generated figures
