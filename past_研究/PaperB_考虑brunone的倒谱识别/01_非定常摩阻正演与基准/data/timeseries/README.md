# 井口时程（主图可复现）

由 `../../code/archive_wellhead_timeseries.py` 生成。列为 `t, H_wh, Q_wh`，gzip CSV。

| 文件 | 工况 |
| :--- | :--- |
| `n1_k0.csv.gz` | 单缝，$k=0$（Darcy） |
| `n1_k0.01.csv.gz` | 单缝，常数 IAB $k=0.01$ |
| `n1_k0.02.csv.gz` | 单缝，常数 IAB $k=0.02$ |
| `n1_k0.05.csv.gz` | 单缝，常数 IAB $k=0.05$ |
| `n4_D20_k0.01.csv.gz` | 四缝，$D=20\,\mathrm{m}$，$k=0.01$ |
| `n4_D5_k0.01.csv.gz` | 四缝，$D=5\,\mathrm{m}$，$k=0.01$ |

SHA-256 见 `SHA256.csv`。图 2(a,c)、图 6(a)、图 7 只读本目录，不读 `output/analysis/`。
