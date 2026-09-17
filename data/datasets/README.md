# data/datasets

当前物理稳态反演正演数据目录。各子目录互不覆盖。

| 目录 | 内容 | 用途 |
|---|---|---|
| `archive_legacy_nonsteady_v1/` | 旧 1k HDF5、inspection 图、meta | 非当前 `physical_flow_control` 方案，仅归档 |
| `archive_lhs10_full_curves/` | 早期 10 例全曲线 HDF5 | 归档，勿当新反演训练集 |
| `moc_v2_physical_steady_1k/` | 既有 1k 目录（保留不覆盖） | 旧窗口/旧采样，勿混用 |
| `moc_v2_physical_steady_10/` | 10 例试跑：h5、inspection、meta、csv | 冒烟集 |
| `moc_v2_physical_steady_1k_x4500_4950/` | 1000 例：h5、`dataset_1k_inspection.png`、meta、`case_parameters.csv` | 当前 LHS（`coupling_mode=physical`，首簇 4500–4900 m，末簇 ≤4950 m，\(t_c\) 1–100 ms，\(t_f=61\) s） |
| `moc_v2_physical_independent_steady_1k/` | 1000 例：h5、inspection、meta、csv | 独立簇扰动（`coupling_mode=independent`，无 Dirichlet \(r_j\)，各簇基态 \(\times 0.8\sim 1.2\)，几何窗口同当前 LHS） |
