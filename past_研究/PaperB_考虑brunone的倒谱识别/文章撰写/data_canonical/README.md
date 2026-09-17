# Paper B 主图 canonical 数据清单

主文证据认 `01`–`05` 的 `data/`，不认沙箱 `analysis/brunone_spacing_effect/` 或未入仓的 `output/analysis/`。

井口时程已归档于 `01_.../data/timeseries/*.csv.gz`（SHA-256 见该目录 `SHA256.csv`）。作图脚本不得再硬编码 `output/analysis/...`。

完整校验表见 `MANIFEST.csv`。路径相对于 `PaperB_考虑brunone的倒谱识别/`。

## 使用规则

- 作图：`文章撰写/code/generate_spej_figures.py`、`plot_fig5a_cepstrogram.py`
- 时程生成：`01_.../code/archive_wellhead_timeseries.py`
- **禁止**把 `04_.../data/p7_pvr_q_metrics.csv` 写入主图或 SI（表未修复）。

## 主图绑定

| 正文图 | 主数据 |
| :--- | :--- |
| 图 1 | 示意图，无 CSV |
| 图 2(a,c) | `01/.../data/timeseries/n1_k*.csv.gz` |
| 图 2(b) | `02/.../metrics_v2.csv`（列 `est_s`，Matrix A，`D=20`） |
| 图 3(a,b) | `02/.../four_clocks_n1_vs_n4.csv`（`n=1`） |
| 图 3(c) | `02/.../metrics_v2.csv`（`a_onset`/`a_peak`/`a_f0`，Matrix A，`n=4, D=20`） |
| 图 4 | `01/.../p5_model_comparison_clocks.csv` |
| 图 5 | `02/.../Tc_sweep_n1_k0.01.csv` |
| 图 6(a) | `01/.../data/timeseries/n1_k0.csv.gz` 与 `n1_k0.02.csv.gz` |
| 图 6(b) | `02/.../metrics_v2.csv`（Matrix A 频带） |
| 图 6(c) | schematic，无 CSV |
| 图 7 | `01/.../data/timeseries/`（n1 与 n4_D20/D5） |
| 图 8 | `02/.../metrics_v2.csv`（$C_v$ 与 FWHM） |
| 图 9 | `05/.../k_kleak_damping_surface.csv`（`H_ext=100 m`） |
| 表 3 | `02/.../p6_grid_sensitivity.csv` |
| 表 4 | `02/.../four_clocks_n1_vs_n4.csv` |
| 表 5 中 `k(Re)` | `02/.../onset_correction_verdict.csv` |
| 表 7 | `05/.../realised_k_n1_snapshots.csv` |
