---
type: literature-note
citation_key: FerranteBrunone2003
title: Pipe-system diagnosis and leak detection by unsteady-state tests. 2. Wavelet analysis
authors: Ferrante, M.; Brunone, B.
year: 2003
journal: Journal of Hydraulic Engineering
doi: 10.1061/(ASCE)0733-9429(2003)129:7(495)
url: https://doi.org/10.1061/(ASCE)0733-9429(2003)129:7(495)
status: read
topics: [T01, T04]
tags:
  - literature
  - wavelet
  - reflection-detection
---

# 小波分析用于瞬变反射检测（Ferrante & Brunone 2003）

> 相关后续：Ferrante, Brunone & Meniconi 2007（J. Hydraul. Eng. 133(11)）、Ferrante et al. 2009（分支管网，小波 + Lagrangian 模型）。

## 1. 基本信息

- 作者：M. Ferrante, B. Brunone
- 年份：2003
- 期刊：Journal of Hydraulic Engineering（系列 II）
- 阅读状态：已读（2026-08-02，基于综述与摘要）

## 2. 一句话总结

> 用小波变换定位压力瞬变中的奇异点（反射波到时），作为倒谱之外更直接的时-频反射检测工具。

## 3. 研究问题

- 在噪声与多反射干扰下，如何自动、准确地识别反射波到时？

## 4. 方法

### 物理模型

- 1D 瞬变流 + 泄漏反射

### 数学方法

- 连续小波变换（CWT）在时间-尺度平面检测奇异点；噪声经小波去噪
- 反射到时 → $x = a\,t/2$ 定位

### 实验或数据

- 实验室管道泄漏测试

### 评价指标

- 定位误差 vs 传统到时读取

## 5. 主要结果

1. 小波显著提高反射到时识别精度（自动奇异点检测，抗噪）
2. 多尺度分解可将不同频带的反射成分分离，缓解波形展宽导致的到时模糊
3. 成为后续领域内反射法的事实标准预处理工具

## 6. 关键公式

- $W(a,b) = \int H(t)\, \psi^*_{a,b}(t)\, dt$（CWT 定义）；奇异点 = 尺度轴上模极大值线

## 7. 关键图表

- 含噪压力信号的小波谱图 + 奇异点链提取

## 8. 局限性

- 多裂缝强重叠时，相邻奇异点仍难分开（分辨率受小波母函数带宽限制）
- 到时特征对波形退化敏感（onset 模糊，与 [[T04-Brunone频散效应]] 的表观速度分离问题同源）

## 9. 我的评价

- 可信之处：领域经典、被广泛复现
- 可质疑之处：对「重叠 + 退化」双失效模式（本项目核心困难）改善有限
- 与当前工作的差异：本项目 T01 用倒谱（周期域），小波是时-频域补充；两者可组合（小波去噪 → 倒谱定位）

## 10. 与本项目的关系

- 支撑主题：T01（倒谱的对照工具）、T04（波形退化下的到时精度）
- 支撑论文：Paper B（T01/T04 related work）
- 可复现实验：在本项目 `bench_stratified` 分层含噪基准集上实现 CWT 奇异点检测，与倒谱盲检测同协议对比（对应 S2 协议）
- 可借鉴方法：小波去噪作为盲检测前处理；多尺度分解作为频带分离通道
- 潜在创新差异：多缝强干涉下 CWT 的 SRF 实测；与 T05 稀疏方法统一评分

## 11. 可引用内容

- 支撑论述：小波反射检测是倒谱之外的主流时频工具
- 建议引用位置：Paper B 方法对照（信号处理工具谱系）

## 12. 关联笔记

- [[瞬变波异常检测五类方法综述]]
- [[Che2021_瞬变波异常检测方法综述]]
