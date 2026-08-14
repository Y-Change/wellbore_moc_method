---
type: literature-note
citation_key: Bohorquez2020
title: Leak Detection and Topology Identification in Pipelines Using Fluid Transients and Artificial Neural Networks
authors: Bohorquez, J.; Alexander, B.; Simpson, A. R.; Lambert, M. F.
year: 2020
journal: Journal of Water Resources Planning and Management
doi: 10.1061/(ASCE)WR.1943-5452.0001187
status: read
topics: [T05, T07]
tags:
  - literature
  - ann
  - topology
---

# ANN 从压力时程辨识接头与泄漏（Bohorquez et al. 2020）

## 1. 基本信息

- 作者：Jessica Bohorquez, Bradley Alexander, Angus R. Simpson, Martin F. Lambert
- 年份：2020
- 期刊：J. Water Resour. Plan. Manage.
- 阅读状态：已读（2026-08-13，全文 PDF）

## 2. 一句话总结

> 用瞬变水头时程训练 ANN，分别预测接头位置/两侧管径，以及泄漏位置/尺寸；测试阶段不需要水力模型，计算量集中在训练。

## 3. 方法

- 数据全部由数值瞬变模型生成（文中为首次联合应用，无实验）。
- 输入：一条瞬变水头时程。输出：特征位置与尺寸（接头或泄漏，分开训练）。
- 强调测试时不需要系统先验或完整模型。

## 4. 主要结果（文内数值，仅数值域）

- 接头位置：95% 样本误差 < 2.32 m；两侧管径几乎完美。
- 泄漏位置：95% 样本误差 < 3.0 mm（管长尺度下极小）；尺寸平均绝对误差 0.03 mm。

这些数只在**与训练同分布的无噪声数值测试**上成立。

## 5. 可迁移到本项目

- 「端到端：波形 → 位置」与 P0A 相同；文献证明在匹配的 MOC 分布上可以极准。
- **不能**据此声称现场或 Brunone 失配下可用。作者自己把实验/复杂系统列为未来工作。
- 接头 vs 泄漏分开建模：对应你们「射孔/变径 vs 裂缝」若要 ANN，应分头或用 Waqar 的 leak function，而不是一个回归向量硬塞变长缝数。
- 训练要用与测试相同的摩阻；steady 训、Brunone 测会重复你们的剪刀差。

## 6. 局限性

- 仅数值；无噪声、无 UF、无多泄漏重叠的系统报告。
- 与 Waqar 2025 相比，输出维绑死特征个数，扩展性差。

## 7. 关联

- [[Waqar2025_泄漏函数与领域引导ML]]
