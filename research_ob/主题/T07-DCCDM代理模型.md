---
type: research-topic
topic_id: T07
status: frozen
paper_status: paper-c-frozen
tags:
  - research-topic
  - dccdm
  - neural-operator
  - direct-inverse
---

# T07 神经直接反演与 DCCDM

> **frozen（2026-08-12）**。结案：[[PaperC_MOC主路径结案]]。扩散暂停；不加深 PhaseNet / LISTA / 纯 FNO 反演。

## 核心问题

井口压力能否稳定反演多缝位置；学习方法相对全波形似然有无额外价值？

## 结论与数据

| 结论 | 关键数 | 证据 |
|---|---|---|
| 近距 held-out 最好 | PhaseNet N=8192 raw F1≈0.746 | EXP-001 系列 |
| 训练技巧无效 | 宽 FWHM / count / phys3 均不抬升；LISTA F1≈0.02 | EXP-005/006/008/009 |
| 瓶颈非表示能力 | Oracle F1≈0.999、overfit 过、held-out 回落 | EXP-004 对照 |
| 合成基准：估计器低效 | CRB~cm，MLE eff≈1；热图损失 ≠ 全波形似然 | [[EXP-20260806-002]]/003 |
| 迁移掉点 | PhaseNet F1 0.746→0.232（steady 配对） | [[EXP-20260807-002]] |
| FNO 波形 PASS | 中位 rel L2≈0.029（`fno_close2000`） | [[EXP-20260808-001]] |
| FNO 反演 FAIL | vs MOC：single≈2.7 m，dual≈72 m | [[EXP-20260808-002]] |
| FNO+MOC 混合 | 单缝 PASS；双缝需大吸引域 | [[EXP-20260809-001]] |

处方：**MOC 全波形为主；FNO/检测器仅粗初值或对照。**

文献（泄漏函数标签、CFL 滤波算子、不完全物理降权）：[[00_索引与可迁移方法总表]]

## 已知限制（不修）

旧评估容差曾有 80 m 下限；输入/输出分辨率未解耦；20–50 m 间距空白；多为无噪声仿真。细节见历史实验，非本主题待办。

## 论文论点

| 论点 | 状态 |
|---|---|
| 加密网格抬高 Oracle | 已支持 |
| 技巧修复 held-out | 不支持 |
| LISTA 强对照 | 不支持 |
| brunone→steady 掉点 | 已支持 |
| 波形≠反演；MOC 主 | 已支持 |
| Paper C | [[paper_C_identifiability_efficiency]] frozen |
