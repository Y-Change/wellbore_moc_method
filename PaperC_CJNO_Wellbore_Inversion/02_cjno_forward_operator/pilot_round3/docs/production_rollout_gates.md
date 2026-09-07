# Round-3 C：生产 Tensor rollout 门

门控对象是 `production_rollout.py` 的 `ProductionMOC`，**不是** Round2 `mini_loop`。

每步用自身 `q/pc/F` 经 Tensor `branch_coefficients_t` 更新系数；hard `cluster_newton` 写出的 H/Q 进入后续传播与井口。需要求导的路径上没有 `detach` / NumPy / `float()` 切断（Cr 网格量是 Python float，不是待微参数）。

## 前向：同一网格上 Tensor vs 孪生

首次全量跑曾把 nx=32 的 Tensor 与 nx=64 的孪生按**下标**比较，01342 出现虚假 rel=1。该比较作废。

补跑 `run_c_fixups.py`（同一 `NumericsSpec` / nx=32 / 15 步）：

| case | torch vs twin pert L2 | 状态 |
|---|---|---|
| 01342 | 0 | PASS |
| 00163 | 0 | PASS |
| 00543 | 0 | PASS |
| 01320 | 0 | PASS |

范围：短窗、无损、nx=32。不是长窗全场、不是 ZVB。

## 梯度：同一 `ProductionMOC`，autograd vs 中心差分

- 冻结稳态初态（`ic_depends_on_param=false`）
- 标量：`||p_head - p0||^2` 对第一孔 K
- n_steps=302（覆盖 `t_s + x_1/a`），friction=none
- g_auto = −16.1018
- 三步长：相对误差 **1.59e-5 / 4.84e-5 / 1.16e-3**
- 最佳步长 **1.59e-5 < 1e-4** → 光滑点 **PASS**
- `stiff_guard` 离散切换未与光滑点混报

## 真实 Newton 失败

`solve_cluster_node` / `cluster_newton` 在病理孔口 `K=1e-20`、不一致 C± 下返回 `ok=False`，`rinf≈1.03e-4`（不是 `raise` 注入）。  
`ProductionMOC` 合同：`stat[1]<0.5` → `NewtonFailedError`，训练步阻断。  
fixups 证实该路径 **PASS**。  
把同一 K 塞进**完整井时间步进**时，该井的真实 C± 仍可能收敛——完整井自由滚动上的失败未找到，记 **INSUFFICIENT**（不是把注入异常当过门）。

## 未做 / 超时

| 项 | 标记 |
|---|---|
| Tensor ZVB 记忆递推 | NOT_RUN |
| 长窗 nx=64 ZVB 参考 | TIMEOUT（已中止） |
| 用 mini_loop 过门 | 未使用 |
| 段传播学习训练 | 未启动 |

## 段传播最小接口（仅当门全过才提交）

本轮**不提交**可学习段传播接口。阻塞见 `round3_audit.md` 第 4 问。
