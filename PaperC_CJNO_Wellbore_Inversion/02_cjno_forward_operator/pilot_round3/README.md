# PaperC Stage-2 Round 3

指标修复、短训失败归因、同物理可微闭环。  
**不**证明 CJ-NO 已成功。历史证据在 `../pilot_preliminary/` 与 `../pilot_round2/`，本目录不覆盖。

环境：`D:\Anaconda\envs\torch24\python.exe`（Python 3.9, torch 2.4.0 + CUDA）。

## 已实际跑过的复现命令

在 `02_cjno_forward_operator/pilot_round3` 下：

```text
D:\Anaconda\envs\torch24\python.exe tests\run_a_suite.py
D:\Anaconda\envs\torch24\python.exe code\run_a_and_fields.py
D:\Anaconda\envs\torch24\python.exe code\run_b_suite.py
D:\Anaconda\envs\torch24\python.exe code\run_c_gates.py
D:\Anaconda\envs\torch24\python.exe code\run_c_fixups.py
```

单臂：

```text
D:\Anaconda\envs\torch24\python.exe code\train_b.py --arm B1 --n 8
D:\Anaconda\envs\torch24\python.exe code\train_b.py --arm B2 --n 8
```

A 未通过时不要跑 B。旧 test 按 case_id 阻断。

## 交付

- `docs/round3_audit.md`
- `docs/metrics_regression_tests.md`（`manifests/metrics_regression_tests.json`）
- `docs/optimization_diagnosis.md`
- `docs/reference_parity.md`
- `docs/production_rollout_gates.md`
- `docs/claim_evidence_table.md`
- `docs/input_field_consumption.json`

## 状态摘要

- A：PASS
- B1 / B2 / extra seed 43：PASS（8 案 train mean pert L2 0.0045 / 0.0061 / 0.0041；stop=`step_budget`）
- 同网格孪生 vs MOC：PASS（L2=0）
- 0.5% 连续对齐门：前提失败（加密差 0.45–2.49%）
- Tensor vs 孪生（同网格短窗）：PASS
- 生产梯度：PASS（1.59e-5）
- Tensor ZVB：NOT_RUN
- 长窗 ZVB 参考：TIMEOUT
- x/a 因果：FAIL
- 段传播学习：不启动
