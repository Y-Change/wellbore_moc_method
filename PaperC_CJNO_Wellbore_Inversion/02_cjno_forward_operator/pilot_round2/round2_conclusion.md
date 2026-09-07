# Round-2 结论（有效性修复 + 小规模物理闭环）

日期：2026-09-06。环境：torch 2.4.0 + CUDA。旧 test **未打开**。未改方案阈值。失败结果全部保留。

本文件回答交付要求的五个问题。每个数字绑 `run_id` / 检查点 / 指标文件 / 数据版本。

---

## 1. Round1 哪些数字仍可描述，哪些主张必须撤回？

依据 `round1_findings.md`（对照 `pilot_preliminary/` 源码与落盘指标，未改 round1 数字）。

**仍可作描述性结果**

- 可用 1896 例、切分 1514/191/191、组泄漏空、SHA 全匹配。
- IFT 在**名义光滑点**上残差 \(<10^{-8}\)、梯度相对误差 \(\sim10^{-10}\)（不推广到未触发的 stiff / 负流 / 近零根）。
- 求解器侧记忆一致性（递推 vs 同核卷积；Joukowsky 重建 \(z\)）。
- 共享 1D FNO 井口头在 256 例上扰动 L2 约 0.12–0.26，须分开读 last 与 16 例选模；且当时时间轴依赖 batch 最大长度。
- 旧 test 已查看，只当 **已查看的 pilot 保留集**。

**必须撤回**

- hard 相对 soft 的井口差来自隐式节点层 / A1b 已隔离硬层。
- 与省略 Newton 的推理延迟比较。
- teacher-free 硬层训练（`a0=node_pc[t]` 等当前时刻标签）。
- 占位 \(K,c_1\) 下的节点误差归因于「真实物理难」。
- 嵌套学习曲线；同一检查点的 train/val；全窗 argmax「首波」\(\Delta\phi_1\)。
- 256 例未过 2% 已证明「只是数据量不足」。
- 70 Hz、全场、网络记忆逐点、H1、正式 OOD。

---

## 2. batch 时间轴、标签依赖、分支覆盖是否被回归测试锁住？

**是，本轮已锁。** 单元报告：`manifests/unit_tests.json`（PASS）。

| 问题 | 锁 | 证据 |
|---|---|---|
| 时间轴依赖 batch 最大长度 | 固定查询网格 \(t=t_s+i\Delta t\)，\(\Delta t=0.02\,\mathrm{s}\)，\(T=400\)；padding 只加在有效区间之后并 mask | `data_contract.md`；`tests/test_dataset.py`；`tests/test_batch_consistency.py`：同案单独 / 换伙伴 / 换顺序 / pad+17，float32 相对差 \(\le 9.6\times10^{-8}<10^{-5}\) |
| 推理吃未来标签 | `predict_arrays` 禁止 `p_head/node_H/node_pc/node_sq`；`split=test` 抛错；打乱目标不改 predict 数组 | `tests/test_dataset.py`；闭环 `loop_gates.shuffle_labels.pass=true`（`manifests/loop_gates.json`） |
| `data_version` 误存代码 digest | 检查点 `data_version={manifest_sha, case_ids_sha, query}`；相等即拒载 | `tests/test_checkpoint.py` |
| 64/256 不嵌套 | `subset_seed=20260907` 前缀嵌套 | `manifests/nested_train_order.json` |
| 首波全窗 argmax | 物理窗 + 同极性局部峰；合成延时/变幅/跨周期错峰单测 | `tests/test_metrics.py` |
| stiff/reverse/near_zero 未发生 | 场景断言实际状态 | `run_node_validation.py` → `manifests/branch_coverage.json` **PASS** |

节点覆盖（均已发生，不是只改初值）：

- `stiff_storage`：`G_l dt/C_f>1`，`stiff_guard=4`
- `reverse_flow`：求解 \(q\) 全负（均值 \(-2.78\times10^{-2}\)）
- `near_zero`：根 \(q=0\) 落入 \(|q|\le10^{-6}\)
- 异质多孔、非零摩阻、指数储容支路、Tikhonov 条件数分支均覆盖
- 光滑点：前向残差 \(<10^{-8}\)，CP/CM/K 梯度相对误差 \(<10^{-4}\)；\(C_d\) 尺度经 \(K\propto C_d^{-2}\) 传到节点输出
- \(q|q|\) 在 0 处为 **C1**（一阶导 \(2|q|\) 连续）；正则迂曲 \(C^\infty\)；`stiff_guard` 切换是离散的，不写成处处 C1

检查点评估：同一权重分别报 best / last；n=8/64 的 val16 来自 **best**，不参与选模。

---

## 3. 1 / 8 / 64 / 256 例真实过拟合分别到什么水平？

选模规则：`train_pert_l2`，**无 val 早停**。模型：`QueryFourierMLP`（47233 可训练参数，全部参与计算）。  
子集：`nested_train_order.json` 前 n 例。norm 只用该训练子集。

| n | 状态 | run_id | best 扰动 L2 mean / p90 / max | 目标 | last（同 ckpt 家族） | val16（best，不选模） |
|---|---|---|---|---|---|---|
| 1 | **PASS** | `r2_overfit_n1_20260906_174918_934f991e` | 0.0100 / 0.0100 / 0.0100 | ≤0.01 | 0.0100（ep 296） | — |
| 1（保留失败） | FAIL | `r2_overfit_n1_20260906_174839_708825a3` | 0.0120 | ≤0.01 | 0.0634 | — |
| 8 | **FAIL** | `r2_overfit_n8_20260906_174935_f6f7a808` | 0.135 / 0.274 / 0.430 | ≤0.02 | 0.135（ep 200） | 0.571 |
| 64 | **FAIL** | `r2_overfit_n64_20260906_174959_f8c81858` | 0.196 / 0.414 / 0.633 | ≤0.02 | 0.196（ep 80） | 0.225 |
| 256 | **未跑** | — | — | ≤0.02（方案） | — | C1 门未过，禁止扩大 |

检查点：`checkpoints/<run_id>_best.pt` 与 `_last.pt`。  
`data_version` 含 round1 manifest SHA 与该子集 `case_ids` SHA（见各表 JSON，不是代码 digest）。

首波 \(\Delta\phi_1\)：本轮查询窗 8 s，多数案例 \(4L/a\sim 9\,\mathrm{s}\)，物理窗 `(t_valve_end, t_valve_end+period)` 超出查询 → 全部 `ref_no_peak`，**均值不报**。这是指标按合同拒收，不是「相位已过门」。

**达不到 8/64 的 2% 时，不归因为「差这点数据」：**

- 1 例能到 1%，说明时间轴、抗混叠、loss/predict 分离、优化器在单波形上够用。
- 8 例在 200 epoch 平台于 13.5%，64 例 80 epoch 平台于 19.6%，比 1 例差一个数量级。
- 共享网络只吃簇级汇总特征（\(\sum K,\sum C_f,x_j/L\)），**不能**表达异质逐孔正演；不同井的 \(t_c/(4L/a)\) 被压到同一 \(\tau\) 轴。
- 本轮 `w_node=0`，**没有**「节点损失压制井口」的梯度夹角证据，不得使用该归因。
- 图：`figures/Fig_r2_overfit_curves.png`，`figures/Fig_r2_n1_waveform.png`。

---

## 4. hard 节点是否真正参与井口预测？代价和收益？

**在集成参考闭环里：是。在过拟合井口网络里：否。**

闭环（`CharacteristicLoop`，1 单元/段，**不是**完整 CJ-NO）：

- 路径：关阀 \(Q(t)\) → 延迟线 C+/C− → **真实** \(K,\kappa,c_1,c_0,a_1,a_0\)（上一时刻状态，不读当前 `pc`）→ Newton → 反射/透射 → 井口 \(H=C_M+BQ\)。
- 门：`manifests/loop_gates.json` **PASS**
  - 改节点 \(K\)：井口在 \(x/a=0.80\,\mathrm{s}\) 才分叉（实测 0.82 s），此前无泄漏。
  - \(\partial p_{wh}/\partial K\) 绝对值之和 267 ≠ 0；残差 \(4\times10^{-11}\)。
  - 打乱未来标签不进入接口。
  - 30 步合成：`n_hard=39`，残差非空且 \(<10^{-8}\)。
  - Newton 失败 `require_ok=True` 阻断，不静默继续。
  - 单步 teacher 与自由滚动是两条时间推进，不是整窗一次预测。
- 计时（仅模型，预热 2 次，无 NPZ）：30 步合成中位 **6.2 ms**（p25 6.0 / p75 6.3）。**禁止**与 round1 关闭 Newton 的 ~3 ms 比速度。

诊断重放（只 train，最多 4 例，不覆盖原 NPZ）：`replay/replay_manifest.json`

| case | loop Newton 次数 | 节点残差 max | 1-cell 井口 vs 短窗 Darcy-MOC 扰动相对 L2 |
|---|---:|---:|---:|
| pilot_01342 | 395 | 1.3e-10 | 0.257 |
| pilot_00163 | 395 | 9.4e-11 | 0.565 |
| pilot_00543 | 316 | 9.2e-11 | 0.329 |
| pilot_01320 | 316 | 9.7e-11 | 0.523 |

**收益：** 硬节点现在按传播时延改井口，并提供有效梯度；残差门可训练阻断。  
**代价：** 每步每簇一次 CPU Newton；1-cell 段传播相对高分辨 MOC 井口误差 26–56%，**不能**当精度基线，只能当结构闭环。  
**QueryFourierMLP 过拟合路径没有 hard 调用**（`n_hard_calls=0`）。因此 C1 数字不是「硬节点井口精度」。

工作包 D（hard / 同骨架 soft / 编码器基线，64→256，多 seed）**未启动**：C1 的 8/64 门未过。

---

## 5. 继续扩大样本，还是调整路线？

**不支持「先把 n 加到 256/全量再看」。** 证据是：嵌套前缀上误差随 n **变差**（1% → 13% → 20%），不是单调变好。扩大样本不会修复「共享 Fourier MLP + 簇汇总特征」对异质多井的不可表示性，也不会自动变成 CJ-NO。

**支持继续的路线（按优先级）：**

1. 保持本轮合同（predict/loss 分离、嵌套子集、train 选模、真实系数、失败阻断）。
2. 把可学习对象改成**段传播**（先加密 1-cell 参考，再接入神经 \(\mathcal{P}_j\)），hard 节点已经能接到井口。
3. 用重放短窗的逐孔状态做节点 H/Q/分流对照；不要用簇均值反推异质孔。
4. 加长查询窗或单独的首波窗，使 \(\Delta\phi_1\) 有定义后再报相位。
5. 基础门通过后再做 D 的 hard/soft/encoder 受控对照；推理计时必须含 Newton。

**不支持：** 改 2% 门、重开旧 test、宣称 H1、把 `CharacteristicLoop` 叫做完整 CJ-NO、用 round1 延迟对比。

---

## 门控总表

| 包 | 状态 | 依据 |
|---|---|---|
| A 数据/评价合同 | PASS | 单元测试 + 检查点合同 |
| B 真实节点 + 覆盖 | PASS | `branch_coverage.json` / `node_gradient_validation.json` |
| C1 真过拟合 1/8/64 | **部分**：1 PASS，8/64 FAIL | `tables/overfit_summary.json` |
| C2 物理闭环 | PASS（集成参考） | `loop_gates.json` |
| D 64/256 对照 | 未跑 | C1 未全过 |
| 256 例 / seed 43–44 | 未跑 | 同上 |
