---
type: literature-note
citation_key: AziziNadian2025
title: A two-stage approach to detecting pipeline leaks using wavelet transform and inverse transient analysis
authors: Azizi Nadian, H.; Ranzi, R.; Rahmanshahi, M.; Vafaei Rad, N.; Shafai Bejestan, M.
year: 2025
journal: Urban Water Journal
doi: 10.1080/1573062X.2025.2594547
url: https://doi.org/10.1080/1573062x.2025.2594547
status: read
topics: [T05, T07]
tags:
  - literature
  - inverse-transient-analysis
  - two-stage
  - wavelet
---

# 两阶段泄漏检测：小波粗定位 + 逆瞬变反演（ITA）精修

## 1. 基本信息

- 作者：Hossein Azizi Nadian, Roberto Ranzi, Mostafa Rahmanshahi, Nasim Vafaei Rad, Mahmood Shafai Bejestan
- 年份：2025-12（在线）
- 期刊：Urban Water Journal
- 阅读状态：已读（2026-08-02，基于摘要与高亮）

## 2. 一句话总结

> 先用小波分析估计泄漏个数与大致位置（粗定位），再用逆瞬变分析（ITA）对实测与模型全波形做优化拟合，精修位置与尺寸——两阶段解耦「个数/粗位置」与「精参数」。

## 3. 研究问题

- 传统 ITA 同时反演位置与尺寸导致参数空间大、计算贵、易陷入局部最优，如何降维？

## 4. 方法

### 物理模型

- 1D 瞬变流 + 黏弹性管道（viscoelastic）

### 数学方法

- 阶段 1：小波分析 → 泄漏个数 + 近似位置（±1% 管长范围）
- 阶段 2：ITA（正演 + 优化）→ 在粗位置邻域内精修位置与尺寸

### 实验或数据

- 2 个数值测试 + 4 个实验测试

### 评价指标

- 位置误差、尺寸误差（相对百分比）

## 5. 主要结果

1. 小波阶段位置精度 ≈ 2%（候选范围 ±1% 管长）
2. ITA 精修：数值测试位置误差 <1.5%、尺寸误差 <2.2%；实验 <1.8% / <3.1%
3. 两阶段把 ITA 的搜索空间限制在粗定位邻域，显著降低计算成本与多峰优化风险

## 6. 关键公式

- 无新公式；沿用小波奇异点 + ITA 目标函数 $\min_{\theta} \| H_{\mathrm{meas}} - H_{\mathrm{mod}}(\theta) \|$

## 7. 关键图表

- 两阶段流程图；粗定位 vs 精修结果对比表

## 8. 局限性

- 针对泄漏（单个/少数），非多裂缝强干涉
- ITA 阶段仍需正演模型准确（核不确定问题未解决，与本项目 Re-dependent k 同源）
- 黏弹性管道与本项目井筒（弹性 + Brunone）机制不同

## 9. 我的评价

- 可信之处：有实验验证，误差量级可信
- 可质疑之处：粗定位的「个数」由小波人为判读，盲性不如本项目 S0 协议
- 与当前工作的差异：本项目 S2 盲检测矩阵相当于其阶段 1 的协议化版本；其阶段 2 相当于 T07 直接反演/ITA 对照

## 10. 与本项目的关系

- 支撑主题：T05（两阶段范式）、T07（精修思路）
- 支撑论文：Paper B / T05 方法对照
- 可复现实验：在本项目 `bench_stratified` 上复刻「小波粗定位 → MOC-ITA 精修」流水线，与盲检测 + LISTA 同数据对比
- 可借鉴方法：**两阶段「个数/粗位置 → 精参数」范式**与本项目路线（盲检测 S0-S2 → 精修）结构一致，可作论文方法论对照
- 潜在创新差异：本项目的「盲」检测协议与多缝先验缺失场景超出其验证范围

## 11. 可引用内容

- 支撑论述：两阶段解耦降低 ITA 成本与多峰风险
- 建议引用位置：T05 外部方法对照 / Paper B related work

## 12. 关联笔记

- [[瞬变波异常检测五类方法综述]]
- [[Che2021_瞬变波异常检测方法综述]]
