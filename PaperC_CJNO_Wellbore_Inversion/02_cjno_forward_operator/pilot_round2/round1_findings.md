# Round-1 实验有效性审核（逐项核实）

本文件只描述 `pilot_preliminary/` 已落盘事实。不改 round1 数字，不把旧 test 重新标成盲测。

| # | 审核项 | 核实结果 | 对 round1 主张的影响 |
|---|---|---|---|
| 1 | hard 节点未接入井口预测 | `cjno.py` 井口头只吃 encoder cond；`eval`/`train.eval_loader` 设 `apply_hard_times=0` | **撤回**「hard 与 soft 的井口差来自隐式节点层」。val/test 井口数字只描述共享 FNO |
| 2 | 评估时关闭 Newton | 同上；推理延迟 ~3 ms 不含 Newton | **撤回**与省略 Newton 的延迟比较。`R_jump` 未进入正式 eval |
| 3 | `node_pc` 进入硬层 `a0`；H/分流作初值 | `_hard_subsample`：`a0=node_pc[t]`，`H_init=node_H[t]`，`q_init=node_sq[t]` | 硬层训练偷看了**当前时刻**标签。不能称 teacher-free |
| 4 | `K/c1` 占位 | `K=[2e10]`，`c1=[1e7]`，`z_j=0`，`RL=RR=0` | 硬层不是 §2.3 节点。节点 H L2≈1 不能归因于「真实物理难」 |
| 5 | 时间查询依赖 batch 最大长度 | `TimeFNO1d` 用 `linspace(0,1,T)`，`T=batch max` | **撤回**跨 batch 可复现的频谱/相位解释。训练环 `phase_4La` 一度用采样下标，已作废 |
| 6 | 64/256 子集不嵌套 | `subset(n, seed)` 每次独立 `choice` | 嵌套学习曲线主张不成立 |
| 7 | 过拟合按 val 早停；train/val 来自不同状态 | `save_ckpt` 只存 best-val；结束时 `eval_loader(train)` 用 **last** 权重 | train pert 与 val pert **不是同一检查点**。2% 过拟合门的 train 数字不能当 best 模型诊断 |
| 8 | stiff/reverse/near_zero 未断言状态发生 | 场景只改输入猜测；`stiff_guard` 未要求 >0；`q_mean` 仍为正；近零只改 `q_n` 初值 | 分支覆盖 **未锁死**。IFT 残差/梯度数字仍可作为光滑名义点的描述性结果 |
| 9 | 「首波相位」是全窗 \|y\| argmax | `first_peak_phase_frac` = 全窗绝对最大峰 | **撤回** §4.3 首波 \(\Delta\phi_1\) 主张。eval 的 phase p50 只能当「最大峰时间」 |
| 10 | `node_max_len=128` | 节点再抽稀，Nyquist 低于已存 ~11 Hz | 节点监督带宽比 NPZ 更窄。不能验收 70 Hz |
| 11 | `data_version` 存成代码 digest | `save_ckpt(..., logger.meta["train_code_digest"], ...)` | 检查点不能校验数据版本 |

## 仍可作描述性结果（收窄后）

- 可用 1896 例、切分 1514/191/191、组泄漏空、SHA 全匹配。
- 节点 IFT 在**名义光滑点**上残差 \(<10^{-8}\)、梯度相对误差 \(\sim10^{-10}\)（不推广到未触发的 stiff/负流/近零根）。
- 求解器侧记忆：递推 vs 同核卷积、Joukowsky 重建 \(z\)。
- 共享 FNO 井口头在 256 例上 train/val 扰动 L2 约 0.12–0.26（last 与 16 例选模口径须分开读）。
- 旧 test 已查看，今后只当 **已查看的 pilot 保留集**。

## 必须撤回的主张

- hard 节点提升井口精度 / A1b 已隔离硬层。
- 256 例过拟合失败已证明「数据量不足」（train 与 val 不同状态；子集不嵌套；时间轴扭曲）。
- 正式首波相位门、70 Hz、全场、网络记忆逐点、H1、正式 OOD。
