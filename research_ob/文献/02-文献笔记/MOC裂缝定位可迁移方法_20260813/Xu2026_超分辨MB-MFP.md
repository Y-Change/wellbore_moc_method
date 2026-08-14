---
type: literature-note
citation_key: Xu2026
title: Enhanced multi-leak detection in pressurized pipelines using super-resolution matched-field processing
authors: Xu, C.; Waqar, M.; Louati, M.; Ghidaoui, M. S.
year: 2026
journal: Water Research
doi: 10.1016/j.watres.2025.124855
status: read
topics: [T01, T05]
tags:
  - literature
  - mfp
  - super-resolution
  - order-selection
---

# 超分辨修正宽带 MFP 检测簇状多泄漏（Xu et al. 2026）

## 1. 基本信息

- 作者：Can Xu, Muhammad Waqar, Moez Louati, Mohamed S. Ghidaoui
- 年份：2026（Water Research 289: 124855）
- 阅读状态：已读（2026-08-13，全文 PDF）。博士论文展开见 [[PhD三篇_方法摘要]]。

## 2. 一句话总结

> 把单泄漏高分辨 MB-MFP 扩到多泄漏：在白/色/脉冲噪声和间距 \(<0.5\lambda_{\min}\) 下分开簇状泄漏，并用 log-likelihood 估计未知泄漏个数。

## 3. 研究问题

- 「靠近」= 间距小于衍射极限 \(\lambda_{\min}/2\)。
- 多泄漏信号叠加；\(K\neq P\) 时会出现分裂峰或合并峰。
- 旁瓣与主瓣混淆。

## 4. 方法

### 物理 / 数据

泄漏引起的水头是各泄漏贡献的线性组合（同 Wang 2018）：

\[
\Delta h(\omega,x_M)=\sum_{p=1}^{P}s_{L_p}^* G(\omega,x_M,x_{L_p}^*)+n.
\]

HDPE 用 K-V 的 \(a(\omega)\)。数值例：弹性管 \(L=200\) m，\(a_0=1000\) m/s，\(f_{th}=1.25\) Hz，用到 \(20f_{th}\) ⇒ \(\lambda_{\min}=40\) m。测点 \(x_M=[180,200]\) m。激励：下游阀脉冲。

### MB-MFP

单频缺陷泛函是归一化内积 \(|B(\omega_j,x_L)|=|\langle\Delta h_j,G_j\rangle|/(\|\Delta h_j\|\|G_j\|)\)。宽带修正把多频组合并加旁瓣抑制（MB-MFP1 旁瓣抵消，MB-MFP2 峰提取）。多泄漏时在 \(K\) 维位置空间搜索。

### 未知个数

对 \(K=1,2,3,4\) 做 MB-MFP，画 log-likelihood：接近真 \(P\) 时陡升，\(K\ge P\) 后变平。\(K>P\) 时多余峰幅度很小或叠在真位置上。

## 5. 主要结果

- 无噪声：间距 50 m（\(1.25\lambda_{\min}\)）和 8 m（\(0.2\lambda_{\min}\)）都能分开。
- 实验室 HDPE \(L=144\) m，\(a_0=365\) m/s；双漏 (67.83, 98.86) m，P-2 估计 (68, 99) m，尺寸接近真值；P-1 合成一个错位置；P-3 多一个近零尺寸假峰。
- 作者列出现场不确定：波速、摩阻、稳态流量、结构知识不足。

## 6. 可迁移到本项目

- **缝数选择**：与你们 BIC/exact-count 实验同类。可在 MLE 残差上画 \(\ell(K)\)，看拐平，不要只看倒谱峰个数。
- **旁瓣**：倒谱旁瓣会被当成后缝。可借鉴「峰提取 / 旁瓣抵消」——只在主瓣轨迹（沿频率收敛到真位置）上积分。
- 超分辨依赖**正确的多缝线性模型 + 全频点**，不是把倒谱窗加长。
- 井口单点、无第二测点：文中 \(N=2\) 传感器的阵列增益你们没有，分辨率会差一截。

## 7. 局限性

- 实验室泄漏是孔口，不是裂缝柔度。
- 未来工作才处理波速/摩阻不确定——而那正是你们的主矛盾。

## 8. 关联

- [[Wang2018_线性化MLE超分辨]]
- [[PhD三篇_方法摘要]]
