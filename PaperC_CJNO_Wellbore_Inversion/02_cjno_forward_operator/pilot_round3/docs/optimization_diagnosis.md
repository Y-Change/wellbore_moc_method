# Round-3 B：8 案有限预算对照

冻结：嵌套前缀 8 案、`dt=0.02 s`、`T=400`、`QueryFourierMLP`（47233 参数）、Round3 修正归一化、同一 batch 计划种子。  
**不**加长查询窗、不换网络、不加节点损失。选模只看 train mean 扰动 L2。门槛 ≤2%。  
预算耗尽写成 `stop_reason=step_budget`，**不是收敛**。

8 案：`pilot_01342, 00163, 00543, 01320, 00049, 00888, 01900, 00880`。

## B1 共享尺度绝对 MSE

| 项 | 值 |
|---|---|
| run_id | `r3_B1_n8_20260906_223059_cc991c43` |
| seed | 42 |
| 状态 | **PASS**（本预算内过 2% 门） |
| stop_reason | `step_budget`（5000 步，70.2 s，未触 120 s） |
| best train pert L2 mean / p90 / max | **0.00449 / 0.00775 / 0.01110** |
| last | 与 best 同量级（5000 步末） |
| clip_frac | 0.0218 |
| 静默案 | 0 |

逐案 best rel L2：01342 0.00632；00163 0.00404；00543 0.00283；01320 0.00339；00049 0.00479；00888 0.00162；01900 0.00182；00880 0.01110。

与 Round2 同协议短训对照：Round2 n=8 约 200 epoch（~400 step）停在 **0.135**。本臂 step 200 时 mean≈0.141，与 Round2 同量级；step 700 已到 0.0197（过门）；step 5000 到 0.0045。

## B2 逐案目标能量归一 + 等权

| 项 | 值 |
|---|---|
| run_id | `r3_B2_n8_20260906_223214_b708b92f` |
| seed | 42（同一 batch 计划） |
| 状态 | **PASS** |
| stop_reason | `step_budget`（5000 步，74.5 s） |
| best train pert L2 mean / p90 / max | **0.00609 / 0.0085 / 0.00944** |
| 静默案 | 0；分母只用 `\|\|target\|\|` |

两臂都完成 5000 步，**无需再比共同步数**。

## 预先登记 extra seed（B1 过门后）

| 项 | 值 |
|---|---|
| run_id | `r3_B1_n8_20260906_223334_622777f2` |
| seed | 43 |
| 状态 | **PASS** |
| mean / max | 0.00410 / 0.01060 |

## 解释（不改门槛）

- B1 在延长预算后过门：**支持「原短预算/调度是重要因素」**。
- B2 也过门：幅值权重**不是**本预算内过门的必要条件；不能把 Round2 失败主要归给共享尺度 MSE。
- 两臂均过门，**未做** 8 个单例拟合（该分支仅在两臂都未过门时启动）。
- 未宣布架构已解决 64/256。64 案 Round2 数字仍保留为失败证据。

## 实际消费的输入字段

见 `docs/input_field_consumption.json`。QueryFourierMLP 只用井筒标量 + `x_j/L` + log mean K + log mean C_f。  
**未进入网络**：`I_f, R_f, G_l, kappa, p_res, roughness, TVD, 逐孔异质性, 初始 q/pc/F`。  
输入补全未混进 B1/B2。

## 首峰指标（B 的 8 s 窗）

8 案 `4L/a ≈ 9–14 s`，关阀后一个周期的窗超出 8 s 查询 → 全部 `ref_incomplete_window`，coverage=0。  
这是**指标拒收**，不是相位已过门。与「节点反射首达」不是同一定义。

JSON：`manifests/optimization_diagnosis.json`；检查点在 `checkpoints/`。
