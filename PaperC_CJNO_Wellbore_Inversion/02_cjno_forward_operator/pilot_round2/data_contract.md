# Round-2 数据与评价合同

## 接口分离

```
predict(physical_inputs, initial_state, query_grid) -> predictions
loss(predictions, targets) -> scalars
```

`predict` **禁止**读取未来 `p_head`、`node_H`、`node_pc`、`node_sq`。
初始状态只允许：`t=0` 已观测井口/节点迹，或由物理稳态算出的 \((H,Q,q,p_c)\)。

## 必须显式输入的变化量

井：\(L,D,a,\rho,\nu,\varepsilon,TVD,Q_0,t_s,t_c\)。  
簇/孔：\(x_j\)，逐孔 \(C_d,A_{perf},\kappa,I_f,R_f,C_f,G_l\)，簇 \(p_{res}\)。  
由此计算 \(K_p=\rho/(2C_d^2 A^2)\)，**不**用簇级 `eq` 替代异质逐孔正演。

固定假设（写入 config，不得假装已输入）：Brunone 不用；生产记忆为 ZVB 递归核，闭环参考模型在节点上做时间递推、段内无空间记忆场；趾端 `dead_end`；关阀为 cosine ramp，\(t_s=1\,\mathrm{s}\)（与生成器一致）。

## 时间与组批

- 查询坐标：物理 \(t\) 与 \(\tau=(t-t_s)/(4L/a)\)。
- 共同网格：所有案例 \(t_i=t_s+i\Delta t_q\)，\(i=0..T-1\)，\(T,\Delta t_q\) 预注册。
- 重采样：先抗混叠低通，再抽样。padding 只加在网格**之后**且用 mask，不改有效点的物理时间。
- 节点标签只在 NPZ 保存时刻监督；插值不是新真值。

## 扰动与静默

各节点用**自己的** \(t=0\)（或稳态）\(H_j,Q_j^\pm,p_{c,j}\) 构造扰动。  
\(H,Q,\sum q\) 分别按各自尺度无量纲化，恢复 SI 后评价。  
静默：\(\|u_{pert}\|_2/\|u\|_2 < 10^{-6}\)，只由目标计算。

## 嵌套子集

`subset_seed=20260907` 固定 train 排列；`n∈{1,8,64,128,256,512,all}` 取前 n 例。  
`model_seed` 分开。小样本过拟合的 norm 只用该子集；正式学习曲线用预注册的 train 前 256 例 norm（`norm_scope` 写入检查点）。

## 首波

在关阀结束后第一个 \(4L/a\) 窗内，用与真值同一极性的局部峰匹配。  
无峰 / 错峰 / 多峰不可判定 → 失败标记，不计入 \(\Delta\phi_1\) 均值。

## 检查点

必含：model、optimizer、完整 cfg、norm、case_ids、manifest SHA256、code SHA256、Python/NumPy/Torch/CUDA RNG、epoch、global_step。  
`data_version = {manifest_sha, case_ids_sha, query_protocol}`，**不是**代码 digest。  
同一检查点分别评估 train/val；best（按登记规则）与 last 分列。

## 本轮禁止

打开旧 test；`--confirm-30k`；覆盖 stage-1 / round1 NPZ 与 goldens；正式 FIM/反演。
