---
type: freeze-note
paper_id: Paper-C
status: frozen
date: 2026-08-12
topics: [T01, T05, T07]
canonical: true
tags:
  - freeze
  - paper-c
  - moc-primary
---

# Paper C / MOC 主路径 — 结案固化（2026-08-12）

> **停工声明**：自本日起不再在可辨识性诊断、FNO 反演冲关、MOC 管线扩展（含 N3 失配压测）上继续投入。
> 下文为该线**唯一结案摘要**；细节以实验笔记与 `output/` 为准，不在多处重复维护。

## 锁定结论

1. **无信息论近距墙**：CRB ~cm（例：σ_x≈0.025 m @ SNR=40 dB, fc=20 Hz）。
2. **全波形 MOC-MLE 效率可达界**：RMSE/CRB ≈ 0.96–1.39（n≤5 仍约 0.86–1.18）。
3. **启发式远离 CRB**：倒谱 SNR=∞ 仍 ~12 m；近距 P0 不稳定。
4. **实用墙 = 模型误差**：brunone↔steady ~30 m；−1% 波速非线性 |bias|≈120 m（一阶 TOF 28–36 m 低估）。
5. **配对失配剪刀差**：倒谱在 steady 上升，P0/PhaseNet 下降（PhaseNet F1 0.746→0.232）。
6. **缝数 ≠ 位置 CRB**：嵌套 BIC exact-count=1.0（oracle）；PhaseNet 盲计数 ~30.5%。
7. **工程处方**：MOC 全波形 MLE 为主；倒谱/FNO 仅粗初值或对照；不宣称纯 FNO 反演过关。

## 方法原型（已冻结，不再扩展）

入口：`analysis/identifiability/run_moc_pipeline.py`  
输出：`output/analysis/identifiability/moc_pipeline/`

| 能力 | 结果 | 实验 |
|---|---|---|
| 倒谱初值 → MOC 精修（single） | init~12 m → 0.012 m | [[EXP-20260809-002-MOC主路径管线烟测]] |
| 簇约束倒谱（dual_10m） | init~21 m → 0.028 m | [[EXP-20260809-003-MOC管线双缝倒谱初值]] |
| 部署 GN（`x_grid` 重算 J） | hat≈oracle | [[EXP-20260809-004-MOC管线部署GN]] |
| FNO 波形门槛 | rel L2 0.029 PASS | [[EXP-20260808-001-FNO主域重训]] |
| 纯 FNO-MLE | FAIL（single~2.7 m, dual~72 m） | [[EXP-20260808-002-FNO反演门槛未过关]] |
| FNO 初值 + MOC | 单缝 PASS；双缝需大吸引域 | [[EXP-20260809-001-FNO初值MOC精修混合门槛]] |

## 明确不做（已取消）

- N3：波速 −1% 全量压测 MOC 管线
- 纯 FNO / 可微代理 CRB 级反演冲关
- 加深 PhaseNet / LISTA /「智能拼接」
- 本线继续扩写英文全文、补图注以外的新实验

## 文稿与入口

| 文档 | 状态 |
|---|---|
| [[SPEJ_三篇论文框架]] | vault 转发；正文在 `docs/PaperA_…` / `PaperB_…` / `PaperC_…`（C 不新做实验） |
| [[paper_C_identifiability_efficiency]] | **frozen**（internal 初稿保留，不再作为活跃工程驱动） |
| [[PaperC_identifiability_efficiency_outline]] | superseded（提纲存档） |
| [[技术路线与核心问题]] | 诊断+原型段 **frozen**；其余主题见 Paper A/B |
| [[下一阶段执行计划]] | **superseded**（由本结案页取代） |

## 主题页对应

- [[T01-倒谱分辨率极限]] — Paper C 主张与 CRB/MLE 证据已结
- [[T05-稀疏反卷积]] — P0/LISTA 相对 MLE 定位已结；不再加深
- [[T07-DCCDM代理模型]] — FNO 辅 / PhaseNet 迁移掉点已结；不再加深
