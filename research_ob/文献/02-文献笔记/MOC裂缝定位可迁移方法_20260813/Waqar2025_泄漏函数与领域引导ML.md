---
type: literature-note
citation_key: Waqar2025
title: Pipeline leak detection using hydraulic transients and domain-guided machine learning
authors: Waqar, M.; Memon, A. M.; Louati, M.; Ghidaoui, M. S.; Alhems, L. M.; Meniconi, S.; Brunone, B.; Capponi, C.
year: 2025
journal: Mechanical Systems and Signal Processing
doi: 10.1016/j.ymssp.2024.111967
status: read
topics: [T05, T07]
tags:
  - literature
  - leak-function
  - domain-guided-ml
---

# 领域引导 ML：泄漏函数作固定维标签（Waqar et al. 2025）

## 1. 基本信息

- 作者：Waqar, Memon, Louati, Ghidaoui, Alhems, Meniconi, Brunone, Capponi
- 年份：2025
- 期刊：MSSP 224: 111967
- 阅读状态：已读（2026-08-13）。本地另有一份误名为「MFP6」的 PDF，正文即本文。

## 2. 一句话总结

> 不把泄漏个数写进网络输出维，而把沿管长的 **leak function**（高斯瓣，峰=位置，幅=尺寸，宽~\(\lambda_{\min}\)）当回归目标；用标定后的瞬变孪生模型生成数据，再按阀关闭时间、稳态流量、泄漏个数做迭代精炼。

## 3. 方法

### 标签设计（最值得抄）

每个泄漏是密度瓣，宽度由激励最短波长决定（与 MFP/TR 主瓣同物理）。输出长度固定（文中 101 点），与 \(n_L\) 无关。管网则各管 leak function 拼接。

位置采样：\(\lambda\) 到 \(L-\lambda\)，避免贴边界。优先多采小泄漏（更难检）。

### 训练数据

- 孪生模型须先用**无泄漏试验标定**。
- 数值：6000 样本 × 1200 维压力输入；70/20/10 划分。
- 实验室：无泄漏 Case 标定后，2000 单漏 + 2000 双漏；泄漏面积 16.8–72 mm² 均匀随机。
- 逐步适配：阀关闭时间 \(T_c\)（控制瓣宽）→ 停泵前流量 \(Q_0\)（控制 Joukowsky 幅值，反射系数与之成正比）→ 更多泄漏个数。

### 网络

前馈 ANN，隐层 400-400-300-300-200；指标 MSE/RMSE/MAE/\(R^2\)/PCC。作者强调也可用别的回归器。

输入可拼多传感器；文中管网例仍只用一个测点。

## 4. 可迁移到本项目

你们 P0A/PhaseNet/LISTA 的「沿深度的事件图」与 leak function 同构。

| 文献做法 | 建议 |
|---|---|
| 瓣宽 = \(\lambda_{\min}=a/f_{\max}\) | 标签高斯 σ 随激励/窗带宽变，不要固定像素宽 |
| 输出维不绑缝数 | 与盲计数一致；峰值检测后再 BIC |
| 训练分布含 \(T_c,Q_0\) | 停泵斜坡、排量必须进 LHS，否则尺寸头会漂 |
| 无泄漏标定孪生 | 现场难；仿真里应对 steady 与 Brunone **分别**训练，或把摩阻类型当输入条件 |
| 从小 SNR/小泄漏开始采 | 与你们 DR 扫描一致 |

不要用 steady 数据训的网络去测 Brunone 波形——即你们已做的失配剪刀差。

## 5. 局限性

- 实验室是粘弹性管；领域精炼针对 \(T_c\) 与 \(Q_0\)，不是 UF 核。
- 纯数值到现场仍有域移；作者用逐步 refinement 而不是一次训完。

## 6. 关联

- [[Bohorquez2020_ANN拓扑与泄漏]]
- [[T07-DCCDM代理模型]]
