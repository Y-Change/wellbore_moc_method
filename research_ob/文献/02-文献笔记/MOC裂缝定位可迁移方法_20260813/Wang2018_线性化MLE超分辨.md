---
type: literature-note
citation_key: Wang2018MLE
title: Identification of multiple leaks in pipeline: Linearized model, maximum likelihood, and super-resolution localization
authors: Wang, X.; Ghidaoui, M. S.
year: 2018
journal: Mechanical Systems and Signal Processing
doi: 10.1016/j.ymssp.2018.01.042
status: read
topics: [T01, T05]
tags:
  - literature
  - mle
  - super-resolution
  - crb
---

# 多泄漏线性化 + MLE 超分辨（Wang & Ghidaoui 2018 MSSP）

## 1. 基本信息

- 作者：Xun Wang, Mohamed S. Ghidaoui
- 年份：2018
- 期刊：MSSP 107: 529–548
- 阅读状态：已读（2026-08-13，全文 PDF）

## 2. 一句话总结

> 多泄漏非线性耦合是 \(O(s_L^2)\)，可忽略后，位置与尺寸分离估计；MLE 在间距 \(<\lambda_{\min}/2\) 时仍能分辨，且在高斯白噪声下等价于最大 SNR 的 MFP。

## 3. 研究问题

- 多缺陷解析解含交叉项，优化维数高。能否线性化并分开估位置与尺寸？
- 衍射极限 \(\lambda_{\min}/2\) 是否是信息论界限？

## 4. 方法（可迁移的数据处理）

### 正演

频域传递矩阵。测点靠近下游。正演用完整传递矩阵，反演用线性化模型（故意模型误差，用于检验稳健性）。

### 线性化数据模型

\[
h(\omega_j,x_m)=h^{NL}(\omega_j,x_m)+\sum_{n=1}^{N}s_{L_n}G(\omega_j,x_{L_n},x_m)+n_{jm}
\]

\(n_{jm}\) 独立复高斯。非白噪声：**先预白化**再套白噪声 MLE。

上游流量 \(q(x_U)\)：在紧邻上游加一测点，假设该短段无泄漏，用传递矩阵反推（KDP 类）。

### 估计步骤（算法可抄）

1. 对候选位置向量 \(x_L\)，尺寸的 LS 闭式：\(\hat s=(G^HG)^{-1}G^H\Delta h\)。
2. 位置：\(\hat x_L=\arg\max_{x_L}\ \Delta h^H G(G^HG)^{-1}G^H\Delta h\)。
3. 代回求 \(\hat s\)。

单泄漏 MFP 是 \(N=1\) 的特例。

### 评价指标

位置 RMSE（30 次噪声实现）、CRLB、与只用谐振峰的对照。

## 5. 主要结果

1. 泄漏越小、线性化越好；三泄漏比双泄漏线性化误差大。
2. 例：间距 60 m \(=0.23\lambda_{\min}\) 时，MFP(1) 不能分开，MLE 仍准确定位与定量。
3. 近距时位置 CRLB 仍约 1 m 量级（文中 \(\lambda_{\min}=258\) m，间距约 20–\(\lambda/2\)），说明**近距可分是问题固有的，不是倒谱/匹配滤波的分辨率墙**。
4. 用全部频率（步长 \(0.02\omega_{th}\) 到 \(30\omega_{th}\)）明显优于只用奇数谐振峰。
5. SNR = −3 dB、间距 \(0.08\lambda_{\min}\) 时，两个估计可能塌成「中间一个大泄漏」——分辨率失败但总尺寸仍有意义。
6. 文称 MFP 在 SNR 低至约 −3 dB、波速不确定时仍可用（指引自其 JHE MFP 文）。

## 6. 可迁移到本项目

与 Paper C 一致：位置 CRB 不随间距恶化；倒谱受变换限制，全波形 MLE 可达界。

落地建议：

- 倒谱只提供 \(x_L\) 初值；精修用上述分离 MLE（你们 `analysis/identifiability/mle.py` 已是这条路）。
- **用全部频率点**，不要只喂谐振峰或倒谱峰。
- Brunone：把 \(G\) 换成含 UF 的频域核，或只在短窗（反射为主）上用弹性 \(G\)。
- 裂缝「尺寸」对应 \(s_L=C_d A_L\)；你们还有 Cf，线性化是否仍 \(O(\theta^2)\) 要单独数值检验，不能默认。
- 井口单点 ⇒ 文中 \(M\) 个传感器变成 \(M=1\)，CRB 会变差，但算法结构不变。

## 7. 局限性

- 定常摩擦进入传播函数 \(\mu\)；UF 未进反演核。
- 小泄漏假设；压裂裂缝不一定小。
- 需要已知或可估的 \(q(x_U)\)；井筒停泵后上游流量边界与实验室阀不同。

## 8. 关联

- [[00_索引与可迁移方法总表]]
- [[Xu2026_超分辨MB-MFP]]
- [[T01-倒谱分辨率极限]]
