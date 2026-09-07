# Round-3 A：指标 / 隔离 / 检查点回归

环境：`D:\Anaconda\envs\torch24\python.exe`，torch 2.4.0 + CUDA。  
命令：`python tests/run_a_suite.py`（在 `pilot_round3/`）。  
机器结果：`manifests/metrics_regression_tests.json`。

| 测试 | 状态 | 覆盖 |
|---|---|---|
| `test_metrics.py` | PASS | 负谷、多正峰取最早、多负谷、错误极性、窗外强峰、截断窗、真静默、大预测误差不改静默、正负相位抵消不作为门、歧义匹配不进成功均值、原延时/跨周期回归 |
| `test_dataset.py` | PASS | `split=test` 阻断；`all_usable` 阻断；**test 的 case_id 即使 split 写成 train 也阻断**；predict 无未来标签；嵌套 8 案与 Round2 相同；打乱目标不改 predict；n=1 常数特征归一化为 0 |
| `test_checkpoint.py` | PASS | data_version ≠ 代码 digest；保存 Python/NumPy/Torch RNG + optimizer + scheduler；连续训练 vs 保存后恢复一致 |
| `test_batch_consistency.py` | PASS | 单独 / 换伙伴 / 换顺序 / pad+17，float32 相对差 ~8e-8 |
| `test_node_branches.py` | PASS | stiff_guard 实际触发；反向流 / 近零根场景可构建 |

**A 通过后才启动 B 学习对照。** Round2 检查点未在新归一化协议下重评。
