# PaperC 阶段二预研 Round 2

目录：`PaperC_CJNO_Wellbore_Inversion/02_cjno_forward_operator/pilot_round2/`

本轮只修实验有效性，并建立「真实节点写入井口」的小规模闭环。  
**不**启动 30k、正式 FIM、反演；**不**打开已查看的旧 test；**不**覆盖 `pilot_preliminary/`。

训练环境：`D:\Anaconda\envs\torch24\python.exe`（torch 2.4.0 + CUDA）。

## 复现命令

```bat
cd /d E:\water_hammer_research\wellbore_moc_method\PaperC_CJNO_Wellbore_Inversion\02_cjno_forward_operator\pilot_round2\code

D:\Anaconda\envs\torch24\python.exe write_nested_manifest.py
D:\Anaconda\envs\torch24\python.exe ..\tests\run_all_tests.py
D:\Anaconda\envs\torch24\python.exe run_node_validation.py
D:\Anaconda\envs\torch24\python.exe run_loop_gates.py
D:\Anaconda\envs\torch24\python.exe replay_short.py

D:\Anaconda\envs\torch24\python.exe train.py --n 1 --seed 42
D:\Anaconda\envs\torch24\python.exe train.py --n 8 --seed 42
D:\Anaconda\envs\torch24\python.exe train.py --n 64 --seed 42

D:\Anaconda\envs\torch24\python.exe plot_overfit.py
```

评估同一检查点的 train/val（禁止 `--splits test`）：

```bat
D:\Anaconda\envs\torch24\python.exe eval.py --ckpt ..\checkpoints\<run>_best.pt --splits train,val
```

## 关键结果指针

| 项 | 路径 |
|---|---|
| 结论 | `round2_conclusion.md` |
| Round1 审核 | `round1_findings.md` |
| 数据合同 | `data_contract.md` |
| 嵌套子集 | `manifests/nested_train_order.json` |
| 单元回归 | `manifests/unit_tests.json` |
| 节点分支/梯度 | `manifests/branch_coverage.json`, `manifests/node_gradient_validation.json` |
| 闭环门 | `manifests/loop_gates.json` |
| 诊断重放 | `replay/replay_manifest.json` |
| 过拟合汇总 | `tables/overfit_summary.json` |

## 名称边界

- `QueryFourierMLP`：井口基线，**不是** CJ-NO。
- `CharacteristicLoop`：1 单元/段的**集成参考**，**不是**完整特征坐标段传播，也不是全场算子。
