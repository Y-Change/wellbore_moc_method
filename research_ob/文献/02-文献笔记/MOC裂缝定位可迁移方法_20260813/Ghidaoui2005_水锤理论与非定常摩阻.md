---
type: literature-note
citation_key: Ghidaoui2005
title: A review of water hammer theory and practice
authors: Ghidaoui, M. S.; Zhao, M.; McInnis, D. A.; Axworthy, D. H.
year: 2005
journal: Applied Mechanics Reviews
doi: 10.1115/1.1828050
status: read
topics: [T04, T06]
tags:
  - literature
  - water-hammer
  - unsteady-friction
  - moc
---

# 水锤理论与实践综述（Ghidaoui et al. 2005）

## 1. 基本信息

- 作者：Mohamed S. Ghidaoui, Ming Zhao, Duncan A. McInnis, David H. Axworthy
- 年份：2005
- 期刊：Applied Mechanics Reviews 58(1): 49–76
- 阅读状态：已读（2026-08-13，全文 PDF）

## 2. 一句话总结

> 从控制体推导经典 1D 水锤方程，系统评述定常/非定常摩阻、MOC 离散误差，以及用瞬变反演标定波速与摩阻的可辨识性。

## 3. 研究问题

- 经典 1D 方程的假设何时成立？壁面剪切何时不可忽略？
- 瞬时加速度型 UF 与卷积型 UF 各自适用什么时间尺度？
- 反演 \(a\)、\(f\) 时 identifiability / uniqueness 如何保证？

## 4. 方法

### 物理模型

- 连续与动量的面积平均形式，Mach \(\ll 1\) 时回到经典

\[
\frac{g}{a^2}\frac{\partial H}{\partial t}+\frac{\partial V}{\partial x}=0,\quad
\frac{\partial V}{\partial t}+g\frac{\partial H}{\partial x}+\frac{\tau_w \pi D}{\rho A}=0.
\]

- 壁面剪切重要性由无量纲 \(G \sim z L M f/(2D)+z T_d/(L/a)\)：模拟时间远超第一周期、管很长、\(f\) 大或 \(D\) 小时，剪切必须保留。

### 非定常摩阻（对本项目直接相关）

- Daily 型：\(\tau_w=\tau_{ws}+k\rho D/4\cdot\partial V/\partial t\)。**常 \(k\) 与实验符合差**（文中 Fig. 4）。
- Daily 数据：加速 \(k=0.01\)，减速 \(k=0.62\)；Shuy 甚至给出符号相反的结论。因此 **\(k\) 不是物性常数**。
- Brunone 修正（含对流项）因简单、能再现衰减而被广泛应用。
- Zielke 卷积（层流精确）及 Vardy–Brown 向湍流推广；时间尺度：慢瞬变只需瞬时加速度，快水击则历史项冻结。

### 数值

- 固定网格 MOC 是主流；Courant 不整时空间/时间插值等价于数值耗散与频散（EHDE 分析）。
- 这对「用 MOC 数据训练/反演」是提醒：插值会污染高频，进而污染倒谱与 MLE。

### 反演

- 伴随方程迭代标定 \(a\)、\(f\)。
- 小波：反射到时 \(\times a/2\) 定位泄漏。
- 明确提出 identifiability 与 uniqueness：Merit 函数须有可辨的全局最小。

## 5. 主要结果（与定位相关）

1. 经典方程在单向、轴对称、低 Mach、锚定管道下成立。
2. 常 \(k\) Brunone 不能同时拟合加速与减速。
3. 螺旋涡出现后，1D 模型与实验误差可在约 6 个周期后达 100%——长窗全波形反演会把湍流结构误差当成裂缝。

## 6. 可迁移到本项目

| 做法 | 说明 |
|---|---|
| 定常 vs Brunone 数据**分库、分 replica** | 不要用 steady 正演去拟合 Brunone 观测 |
| 不把 \(k\) 当扫描「物理常数」去外推现场 | 与 T04「固定 k 矩阵 ≠ Re-dependent k」一致 |
| 长窗 ITA 前先检查 Courant/插值 | 避免数值频散冒充 Brunone 频散 |
| 反演时联合或先标定 \(a\) | 文献已把波速当作第一标定对象 |

## 7. 局限性

- 2005 年，无 MFP/超分辨；泄漏反演只到小波与伴随。
- 井筒多裂缝、滤失柔度未涉及。

## 8. 关联

- [[00_索引与可迁移方法总表]]
- [[T04-Brunone频散效应]]
