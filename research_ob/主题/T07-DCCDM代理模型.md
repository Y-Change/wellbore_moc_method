---
type: research-topic
topic_id: T07
status: active
paper_status: experimenting
tags:
  - research-topic
  - dccdm
  - neural-operator
  - diffusion
  - direct-inverse
---

# T07 神经直接反演与 DCCDM

## 核心科学问题

井口水击压力能否在物理分辨率约束下稳定反演多裂缝位置；物理特征融合和概率生成相对于确定性直接检测能否提供可验证的额外价值？

## 当前结论

1. **扩散主线暂停**：DCCDM overfit16 中 epsilon 损失下降约 99.55%，但逆采样约 1120 峰/样本、micro-F1≈0.007，raw MAE 远差于全零基线（[[EXP-20260729-001-DCCDM-overfit16稀疏生成失败]]）。
2. **确定性直接反演可行**：P0-A PhaseNet 风格单通道 `H_wh`→事件热图，已通过 memorization，并完成 subset512 泛化 pilot（[[EXP-20260730-001-P0A原始波形直接裂缝定位]]）。
3. **Held-out 量级**：subset512→val128，阈值 0.7；physical F1≈0.880（P≈0.953，R≈0.818），中位深度误差≈2.0 m，精确计数约 70%。瓶颈偏漏检而非假峰洪泛。

## 当前工作与证据

| 阶段 | 结果 | 状态 |
|---|---|---|
| 数据审计 | 6000 有效；主研究池 4136 / 挑战池 1864；split 2896/620/620 | PASS |
| Oracle val128 | physical/grid F1=1；深度误差 ~1–3 m | PASS |
| Overfit1/16 | 事件全检出、0 FP | event-primary-v2 PASS |
| Subset512 | physical F1≈0.880；count MAE≈0.47；depth med/P95≈2.0/6.6 m | 已完成（gate 仅 completed） |
| Locked test / 多 seed / 分层 | — | 未做 |
| 倒谱/解频散/LISTA 增益 | — | 未做 |
| 扩散修复（v-pred / zero-SNR 等） | — | 暂停，待确定性基线稳定 |

代码：`neural_operator/direct_inverse/`、`neural_operator/`  
数据：`output/direct_inverse/manifests/`、`output/lhs_dataset/data/`

## 待验证

- [ ] 为 subset512/full 注册硬性 event F1、count、depth gate（不仅 `completed`）。
- [ ] 裂缝数与间距分层指标；50–75 m 带。
- [ ] 全量 train → locked test；三模型种子。
- [ ] 固定倒谱、解频散、LISTA 相对 raw P0-A 的单因素增益。
- [ ] 仅在确定性基线通过后，再测扩散修复。

## 论文（原 P06，叙事已转向）

**定位**：以确定性直接反演为可审计基线，再论证物理特征融合与概率生成是否带来额外价值（DCCDM 本身暂为失败对照/待修复支线）。

| 候选论点 | 实验 | 指标 | 状态 |
|---|---|---|---|
| 原始波形可直接定位可分辨裂缝 | P0-A overfit + subset512 | event F1 / depth | 部分支持 |
| Held-out / locked test 稳定 | 待 full | F1 / count / depth | 待验证 |
| 物理融合优于 raw | 待做 | ΔF1 | 待验证 |
| 修复后扩散优于确定性 | 待做 | sample quality | 暂停 |
