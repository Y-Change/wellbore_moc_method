# 数据加载与缓存说明

本轮**不写全场插值缓存**。训练直接只读 Stage-1 NPZ。

## 源 NPZ（pilot_2000/cases）

| 字段 | 含义 | 采样 |
|---|---|---|
| t, p_head, Q_head | 井口压力 [Pa]、流量 [m³/s] | MOC 原生 Δt |
| node_t, node_H, node_Qm, node_Qp, node_sq, node_pc | 节点水头、Q−、Q+、分流、孔均 pc | node_decim≈16 |
| node_z | 记忆状态，形状 (nrec, N×2×M)，左右迹 × 核通道 | 同节点 |
| params_json / meta_json | 几何参数；grid/seg_Cr 在 meta | — |

没有 `p_obs`、没有 `snap_H/Q`。

## 适配器窗口（dataset.py）

- 停泵后 `window_periods` 个 \(4L/a\)（默认 4）。
- 井口再 `wellhead_decim`（默认 4）→ 训练 Nyquist ≈ 40–50 Hz，**不能验收 70 Hz**。
- 节点最多 `node_max_len` 点均匀抽取；上采样不增加带宽。
- 簇维 pad 到 N_MAX=12，mask 标记有效簇。
- 归一化只在 train 上计算。

## 监督范围

第一轮目标 = 井口波形 + 已存节点迹/记忆。禁止把节点稀疏迹插成 MOC 全场真值。
