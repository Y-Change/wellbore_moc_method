---
type: literature-note
citation_key: Wang2020MFPUI
title: Pipeline leak localization using matched-field processing incorporating prior information of modeling error
authors: Wang, X.; Waqar, M.; Yan, H.-C.; Louati, M.; Ghidaoui, M. S.; Lee, P. J.; Meniconi, S.; Brunone, B.; Karney, B.
year: 2020
journal: Mechanical Systems and Signal Processing
doi: 10.1016/j.ymssp.2020.106849
status: read
topics: [T04, T05, T06]
tags:
  - literature
  - mfp
  - model-error
  - uncertainty
---

# 带建模误差先验的 MFP（Wang et al. 2020）

## 1. 基本信息

- 作者：Wang, Waqar, Yan, Louati, Ghidaoui, Lee, Meniconi, Brunone, Karney
- 年份：2020
- 期刊：MSSP 143: 106849
- 阅读状态：已读（2026-08-13，全文 PDF）

## 2. 一句话总结

> 把失配分成随机噪声 \(e\) 与系统建模误差 \(u\)；若 \(u\) 在有/无泄漏时不变，则用无泄漏试验估计 \(\hat u\)，从数据里减掉后再做标准 MFP，不增加计算量。

## 3. 研究问题

标准 MFP 假设失配是高斯随机。现场更大的是 epistemic error：弯头、未知堵塞、波速/杨氏模量、边界（如 PRV）。忽略 \(u\) 会扭曲 \(|B|^2\) 的峰。

## 4. 方法（可迁移的数据处理）

数据模型：

\[
h_1=h^{NL}+u+e_1,\quad h_2=h(x_L,s_L)+u+e_2.
\]

关键假设：**\(u\) 与泄漏无关、试验间不变。**

无泄漏测量给出 \(\hat u=h_1-h^{NL}(\hat q_1)\)。检测用差分

\[
\Delta h_{12}=(h_2-h^{NL}(\hat q_2))-(h_1-h^{NL}(\hat q_1))=s_L G(x_L)+e_{12}.
\]

然后标准 MFP：\(\hat x_L=\arg\max |\Delta h_{12}^H G|^2/(G^H G)\)。改变的是**输入数据**，不是优化器。

频域：用 \(\omega/\omega_{th}\in[0,14]\) 的 FRF。HDPE 用 Kelvin–Voigt 的 \(a(\omega)\)。

文中写明：**反演模型忽略 unsteady friction**，因为对该 HDPE 管，只含粘弹性、不含 UF 已能再现衰减与频散。

## 5. 主要结果（实验，不编造）

| 实验 | 设定 | 普通 MFP | MFP-UI |
|---|---|---|---|
| I 实验室单漏 \(x_L=28.06\) m | 较干净 | 29.28 m（误差 1.22 m） | 28.08 m（误差 0.02 m） |
| II 佩鲁贾 | 单漏；双漏间隔约 27 m | 双漏时普通 MFP 找不到两个 | MFP-UI 改善；单漏模型在间距 \(\gtrsim\lambda/2\) 时可出双峰 |
| III Beacon Hill 准现场 | PRV 动态边界，试验日期间隔约 3 周 | 弹性模型误差大，尺寸严重高估（\(10.14\times10^{-4}\) vs 真值 \(\sim0.7\times10^{-4}\) m²） | 误差 1.57 m；尺寸 \(1.41\times10^{-4}\)；加 VE 后普通 MFP 误差 8.79 m，MFP-UI 1.21 m |

## 6. 可迁移到本项目

这是对你们 **steady replica vs Brunone 数据 ~30 m 失配墙** 最直接的文献对应。

| 场景 | 建议 |
|---|---|
| 仿真（有无缝孪生） | 同摩阻、同网格做 \(h_{\mathrm{intact}}\)，再 \(\Delta h=h_{\mathrm{frac}}-h_{\mathrm{intact}}\)，然后倒谱/MLE。这消除的是网格、边界、摩阻共同的 \(u\)，留下裂缝散射。 |
| 仿真（摩阻形式失配） | 若观测是 Brunone、replica 是 steady，\(u\) **依赖裂缝**（UF 与波形历史有关），本文假设不成立，差分消不掉。必须 twin 也用 Brunone。 |
| 现场 | 需要「可视为无新裂缝」的基线停泵；压裂井通常没有。不能直接 MFP-UI。 |

波速/摩擦标定量进 replica，而不是事后在倒谱里乘一个因子。文中加 VE 后两种 MFP 都变好，说明 **物理核比处理器更重要**。

## 7. 局限性

- \(u\) 与泄漏独立：大裂缝改变平均流和 Re，从而改变 UF，假设破坏。
- 忽略 UF 只对 VE 主导的塑料管成立；钢井筒相反。

## 8. 关联

- [[00_索引与可迁移方法总表]]
- [[EXP-20260807-002-失配可迁移性剪刀差]]
