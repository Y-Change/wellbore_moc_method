# PaperC Round 4

冻结参考、真实瞬态对齐、Tensor ZVB 记忆、训练入口。  
新结果只在 `pilot_round4/`。不覆盖 Round1–Round3。

Python：`D:\Anaconda\envs\torch24\python.exe`

## 实际跑过的命令

在 `02_cjno_forward_operator/pilot_round4` 下：

```text
D:\Anaconda\envs\torch24\python.exe code\run_p0_kernel.py
D:\Anaconda\envs\torch24\python.exe tests\test_zvb_recursion.py
D:\Anaconda\envs\torch24\python.exe code\run_regressions.py
D:\Anaconda\envs\torch24\python.exe code\run_p1_transient.py
D:\Anaconda\envs\torch24\python.exe code\run_p3_zvb.py
D:\Anaconda\envs\torch24\python.exe code\run_p4_train_entry.py
D:\Anaconda\envs\torch24\python.exe code\run_p2_causality.py
D:\Anaconda\envs\torch24\python.exe code\run_p2_alignment.py r4_P2_causality_20260906_232440_43c7df5c
D:\Anaconda\envs\torch24\python.exe code\summarize.py --run-ids r4_P0_kernel_20260906_231434_0002042f,r4_P1_transient_20260906_231503_6d5cfa65,r4_P2_causality_20260906_232440_43c7df5c,r4_P2_align_20260906_232814_15f0cf88,r4_P3_zvb_20260906_231822_8a284b86,r4_P4_train_20260906_232156_c0d8e7d4,r4_regressions_20260906_231502_c927a16b
D:\Anaconda\envs\torch24\python.exe code\write_round4_reports.py
```

禁止用 `latest/` 取结果。汇总必须带显式 `run_id`。

## 本轮 run_id

| 阶段 | run_id | 停止 |
|---|---|---|
| P0 | `r4_P0_kernel_20260906_231434_0002042f` | completed |
| 回归 | `r4_regressions_20260906_231502_c927a16b` | completed PASS |
| 回归（修复前，保留） | `r4_regressions_20260906_231446_aa6e3c2e` | completed 未过门 |
| P1 | `r4_P1_transient_20260906_231503_6d5cfa65` | completed |
| P3 | `r4_P3_zvb_20260906_231822_8a284b86` | completed |
| P4 | `r4_P4_train_20260906_232156_c0d8e7d4` | completed |
| P2 时序/加密 | `r4_P2_causality_20260906_232440_43c7df5c` | completed |
| P2 0.5% 对齐 | `r4_P2_align_20260906_232814_15f0cf88` | completed |

## 交付

- `docs/evidence_registry.json`
- `docs/kernel_loading_audit.json` / `.md`
- `docs/transient_parity.json`
- `docs/causality_and_convergence.json`
- `docs/zvb_tensor_validation.json`
- `docs/train_entry_and_failure.json`
- `docs/round4_decision.md`

## 禁止事项（保持）

不重训 B1/B2，不训 64/256，不开 D，不改 2% / 0.5% / 1.5dt 门，不静默 refit 核，不打开旧 test。
