# claim_evidence_table

未运行 / 超时 / 证据不足单独标记，不与 FAIL/PASS 混写。

| 主张 | 状态 | 代码路径 | 数值断言 | 适用范围 |
|---|---|---|---|---|
| 指标反例 + 原回归 | PASS | `tests/test_metrics.py` 等；`manifests/metrics_regression_tests.json` | A 套件 5/5 PASS | 合成信号与 train 隔离；不是现场首达 |
| 按 case_id 隔离旧 test | PASS | `isolation.py`；`tests/test_dataset.py` | test split / all_usable / test case_id 均 raise | 仅登记 train |
| n=1 常数特征 → 0 | PASS | `dataset.compute_norm` / `pack_features` | 单案全部 `feat_const`，packed=0 | Round3 协议；旧 ckpt 不可重评 |
| 检查点 RNG+opt+sched 恢复 | PASS | `tests/test_checkpoint.py` | 连续 vs 恢复后下一步权重一致 | Tiny SGD |
| batch/pad/标签隔离 | PASS | `tests/test_batch_consistency.py` `test_dataset.py` | 伙伴/顺序/pad rel≤1e-5；打乱目标不改 predict | QueryFourierMLP |
| B1 8 案 ≤2%（本预算） | PASS | `runs/r3_B1_n8_20260906_223059_cc991c43` | mean 0.00449；5000 step；`step_budget` | 8s 查询、共享 MSE、无节点损失 |
| B2 8 案 ≤2%（本预算） | PASS | `runs/r3_B2_n8_20260906_223214_b708b92f` | mean 0.00609 | 能量等权；分母只用目标 |
| B1 extra seed 43 | PASS | `runs/r3_B1_n8_20260906_223334_622777f2` | mean 0.00410 | 稳定性抽查，不是调参 |
| 关阀后第一峰（B 窗） | 指标拒收 | `metrics.first_arrival_pair` | 8/8 `ref_incomplete_window`，coverage=0 | 8s < 4L/a；不是节点首达 |
| 同网格孪生 = 参考 MOC | PASS | `moc_twin.run_twin` vs `ms.simulate` | 4 案 none/Darcy 井口 pert L2 = 0 | nx=64 长窗；不是连续极限 |
| 网格加密差足够小 | FAIL | `run_c_gates.grid_refinement` | 64 vs 128 = 0.45–2.49% | 新 0.5% 连续对齐门的前提不成立 |
| 新 0.5% 连续对齐门 | 不成立 | 同上 | 前提失败，不把同网格 0 写成该门 PASS | 不替换学习 2% 门 |
| Tensor ZVB | NOT_RUN | — | — | Darcy 过 ≠ 生产过 |
| 长窗 ZVB 参考 | TIMEOUT | 第一次 nx=64 T≈26 s 中止 | — | 缩短窗 nx=32 已跑，L2 0.9–3% vs Darcy |
| 同网格 Tensor = 孪生 | PASS | `run_c_fixups.py` | 4 案 15 步 rel=0 | 短窗 none nx=32 |
| 生产 autograd vs FD | PASS | `ProductionMOC` n=302 | 最佳相对误差 1.59e-5 < 1e-4 | 冻结初态；光滑点；none |
| 真实 Newton 失败阻断 | PASS | `cluster_newton` → `NewtonFailedError` | 病理 K，`ok=False`，rinf≈1.03e-4 | 构造 pack；完整井滚动失败 INSUFFICIENT |
| 冻结初态 x/a 单程 | FAIL | `run_c_fixups.py` | 偏差 0.19–0.58 s > 1.5 dt | nx=32；量级对、门未过 |
| 段传播学习可启动 | 否 | — | 见审计第 4 问 | 本轮不训新大模型 |
