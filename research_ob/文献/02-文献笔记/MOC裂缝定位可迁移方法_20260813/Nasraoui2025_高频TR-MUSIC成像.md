---
type: literature-note
citation_key: Nasraoui2025
title: High resolution imaging of pressurised water supply lines
authors: Nasraoui, S.; Louati, M.; Ghidaoui, M. S.
year: 2025
journal: Mechanical Systems and Signal Processing
doi: 10.1016/j.ymssp.2025.112455
status: read
topics: [T06]
tags:
  - literature
  - time-reversal
  - music
  - high-frequency
---

# 高频 TR-MUSIC 管道成像（Nasraoui et al. 2025）

## 1. 基本信息

- 作者：Saber Nasraoui, Moez Louati, Mohamed S. Ghidaoui
- 年份：2025
- 期刊：MSSP 228: 112455
- 阅读状态：已读（2026-08-13）

## 2. 一句话总结

> 从单入口放入收发阵列，采集 10–100 kHz 的 MIMO 响应，用单频 TR-MUSIC 对堵塞/壁厚缺陷成像（不只定位）。

## 3. 方法要点

- 采样率经验：\(F_s \approx 10\times f_{\max}\)。
- 时域 \(M\times M\times N\) 矩阵 → FFT 到单频，因该频段噪声低。
- SVD 分信号/噪声子空间；阈值 10%，扰动几个百分点像不变。
- 成像前 **减去基线（完好系统）响应** 以突出散射。
- 测量时长截断（实验室 \(T=0.012\) s）以避免罐反射。
- 单次快照噪声敏感；约 20 次实现平均后像稳定。
- Hanning 窗宽约为记录长度的 1/20。
- 激励：高斯调制正弦，中心 55–70 kHz，带宽 \(0.9f_0\)–\(1.1f_0\)。

## 4. 为何对本项目优先级低

- 频段高 3–4 个数量级；井口停泵水击是 Hz–百 Hz 平面波，不是管内导波阵列。
- 需要沿轴的换能器阵列，不是单点井口。
- 可借鉴的只有原则：基线相减、截断掉已知边界反射、多实现降噪。这些在 Waqar 2023 / Wang 2020 里已经以水击频段写过。

## 5. 关联

- [[Waqar2023_时间反演截断匹配]]
- [[PhD三篇_方法摘要]]（Grigoropoulos 高频 TR）
