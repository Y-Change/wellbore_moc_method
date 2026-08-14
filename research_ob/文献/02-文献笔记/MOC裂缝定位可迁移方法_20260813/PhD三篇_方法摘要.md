---
type: literature-note
citation_key: PhD_three_summaries
title: Three HKUST theses on TR and MFP
authors: Waqar 2022; Grigoropoulos 2022; Xu 2025
status: read
topics: [T01, T04, T06]
tags:
  - literature
  - thesis
  - method-summary
---

# 三篇博士论文：方法摘要（不代替期刊文）

待读目录中三篇 PhD 体量分别为约 274 / 392 / 190 页。下面只提取与 MOC 数据处理、摩阻、定位有关的章节结构；算法细节以对应期刊笔记为准。未做逐章翻译。

---

## 1. Waqar 2022 — 瞬变波 + 时间反演的管道状态评估

对应期刊：[[Waqar2023_时间反演截断匹配]]。

目录中可直接用的块：

- Ch.4：有阻尼系统的时间可逆范围（Scenario II: damped system；Range of time-reversibility）。
- Ch.5：缺陷检测算法；多泄漏；**IRF 的替代**；单管数值；管网；稳定性；实验。

对 Brunone：先读「damped system / time-reversibility range」——TR 不是无条件成立，有可逆时间窗口。短窗策略与此一致。

IRF 替代：不一定要显式反卷积得到 \(I(t)\)，匹配滤波用 \(\Delta H\) 与格林函数卷积即可（与倒谱「先反卷积再找峰」是另一条路）。

---

## 2. Grigoropoulos 2022 — 高频压力波时间反演

对应期刊方向：[[Nasraoui2025_高频TR-MUSIC成像]]（阵列成像是后续；本文是高频导波物理）。

与水击定位关系弱，但有两条机制提醒：

- 充液管存在多模态与频散曲线（实验 vs 解析 \(k_x(\omega)\)）；群速度随频率变。你们 T04 的「严格频散」在 kHz 导波里是真实模态频散，在 Brunone 1D 里仍是待证假设，不要把两套语言混用。
- 有平均流时互易破坏：\(G(A|B)\neq G(B|A)\)（40 kHz 实验，静止重合、有流则相位差）。停泵后流速低，但反演若假设互易，应声明停泵后准静止。

激励：高斯调制正弦；带宽取功率谱降到峰值 0.01 处。可借鉴为「报告激励有效带宽」的操作定义。

---

## 3. Xu 2025 — 压力管道参数反演的 MFP 进展

对应期刊：[[Xu2026_超分辨MB-MFP]]；材料反演章节未单独成待读 PDF。

框架（Ch.2–3）：

1. 时域 1D 方程 → 频域 → 传递矩阵（场矩阵 + 泄漏点矩阵）。
2. 多泄漏 = 完好场矩阵 + 各泄漏散射矩阵之和（线性化）。
3. 单频 MFP → 修正宽带 MB-MFP → 旁瓣抵消（MB-MFP1）→ 峰提取（MB-MFP2）。

旁瓣机制（Ch.4）：在「位置–频率」平面上，旁瓣轨迹随频率收敛到真泄漏位置；宽带积分应沿这些轨迹，而不是单频切片上最高的假峰。

Ch.5：未知泄漏数 + 超分辨。Ch.6：用同一套 MFP 反演 HDPE 的 K-V 系数（材料，不是泄漏），并比较六年老化蠕变。

噪声分类（术语表）：白、蓝（高频更强）、脉冲。做 MLE 前要知道噪声谱，必要时预白化。

**Mismatch** 被定义为 replica 场的任何误差（参数、物理简化、数值）。**Degradation** 指主瓣丢失或错位。你们的形式失配墙是 degradation，不是随机噪声。

---

## 阅读优先级（针对本项目）

1. Waqar 期刊文 + 论文 Ch.4 阻尼可逆窗口  
2. Xu 期刊文 + 论文 Ch.4 旁瓣轨迹  
3. Grigoropoulos：仅当讨论高频/导波或互易假设时
