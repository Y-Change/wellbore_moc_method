# Brunone 非定常摩阻下的水击波形退化：倒谱裂缝拾取为何变糊

**English title:** Brunone Unsteady Friction and Waveform Degradation of Wellhead Water-Hammer Signals: Why Cepstral Fracture Picks Blur

**目标期刊：** SPE Journal · **Paper B** · 姊妹篇：[Paper A](./PaperA_Topology-Organized_Cepstral_Response_倒谱拓扑组织与首缝非局部性.md) · [Paper C](./PaperC_Identifiability_and_Error_Budgets_可辨识性与误差预算.md)

**关键词（≤5）：** water hammer; unsteady friction; Brunone; waveform degradation; cepstrum

**一句话问题：** Brunone 项如何改变井口波形的频带、到时和波包，从而使倒谱拾取出现系统偏差——这是耗散导致的回声估计器失效，还是信息论上的不可分辨？

证据入口：`research_ob/主题/T04-Brunone频散效应.md` · EXP-20260730-020 · EXP-20260730-024（已完成）· EXP-20260801-001 · EXP-20260730-010

主张只取已审计观察。**EXP-024 已于 2026-08-14 完成**；现行合成正文为 [Paper BC](./PaperBC_Waveform_Degradation_and_Identifiability_波形退化与可辨识性.md)。本稿保留为 B 段细目，不再平行扩写。不把倒谱旁瓣当墙、不把 \(k = 0.2\) 当现场、不把频率选择性衰减写成 \(c_p(\omega)\)。

本稿**取代** `docs/paper2_brunone_dispersion.md`（严格频散 / Rayleigh 最小缝距表 / \(k\)–压裂液对照 / 38 m 墙）。那些主张不得进入本稿。

本稿为**中文工作框架**。英文摘要附于文末，供日后 SPEJ 投稿使用。

---

## SPE Journal 共同约束

- 篇幅 ≤ 10,000 词；图 ≤ 20。多余材料进 Supplementary Materials。
- 结构：题目 → 摘要（<500 词、无文献）→ 关键词 → 引言 → 理论/方法 → 数据 → 结果 → 编号结论 → 符号表 → 参考文献（author–year）。
- SI 单位。必须独立可读。共同正演：一维 MOC、Courant = 1、趾端定压、集总裂缝节点、井口水头。
- **禁止**：把 \(\Delta d_{\mathrm{DR}}\approx 38.36\,\mathrm{m}\) 当分辨率墙；把规定 \(k\) 写成滑溜水 / 线性胶 / 冻胶；把频率选择性衰减写成已测 \(c_p(\omega)\)。
- 投稿顺序：C → B（先完成 EXP-024）→ A。
- 系列工程句：全波形反演须用对摩阻形式；倒谱只作初值；波速必须标定。

本稿建议主文 8,000–9,000 词；主图 **6–8**。

---

## 核心创新点

1. 固定 \(k \times\) 间距 **30** 工况四裂缝矩阵：随 \(k\) 增大，反射峰衰减、展宽、后移并融合。锚点 \(D = 20\,\mathrm{m}\)、\(k = 0.2\)：峰漂移约 **277 ms**、EST 约 **101 ms**（投稿数字以 EXP-024 审计表为准）。
2. STFT：高频相对低频减弱（频率选择性耗散）。三种表观速度 \(a_{f0}\)、\(a_{\mathrm{onset}}\)、\(a_{\mathrm{peak}}\) 降幅不同，单一标量波速不再描述退化波形。
3. Re 相关 Brunone、盲倒谱：单缝深度偏差约 **10–20 m**（容差扫描）；稳态对照约 **2 m**。LHS 实际 \(k\) 为弱档（中位约 **0.014**，8 s 探针）。不能用 \(k = 0.2\) 图外推现场。
4. 姊妹篇 C（只引用、不重做）：位置 CRB 仅膨胀约 **1.1–1.3 倍**；摩阻**形式**失配约 **30 m**，大于 \(k\) 错 10% 的约 **0.7–1.1 m**。

**不要**作为创新点：严格频散；Rayleigh 最小可分辨缝距表；\(k\) 与压裂液类型对照表；38 m 墙。

---

## 摘要（中文）

非定常 Brunone 摩阻常被用来解释倒谱无法分辨近距裂缝。本文把波形退化与信息损失分开。在规定 \(k = 0\)–\(0.2\)、四条等缝、间距 5–100 m 的 30 工况矩阵上，观察到频率选择性耗散、峰后移和包络展宽；\(k = 0.2\)、间距 20 m 时当前指标给出峰漂移约 277 ms、能量弥散时间约 101 ms。由基频、onset 和峰值提取的表观速度降幅不同，单一标量波速不再稳定。这些观察支持尖锐波前的频率选择性耗散，但不构成相速度 \(c_p(\omega)\) 的测量。在 Re 相关 \(k\) 的井筒算例上，盲倒谱单缝深度系统偏差约 10–20 m，准稳态对照在 2 m 容差下即成功。LHS 语料中实现 \(k\) 为弱档（8 s 探针中位约 0.014），远低于 \(k = 0.2\) 示意。姊妹篇可辨识性结果表明 Brunone 下位置 CRB 仅膨胀约 1.1–1.3 倍，而用稳态副本拟合 Brunone 数据会造成约 30 m 深度偏差。因此倒谱“模糊”是耗散波前下的估计器畸变，不是衍射型分辨率墙，且在全波形反演中次于摩阻形式失配。历史上 135 m 容差掩盖了 10–20 m 偏差，不得再引用为 Brunone 定位成功。

---

## 章节结构

### 1 引言

- 现场黏稠压裂液 → 非定常摩阻；文献常把“峰糊了”写成不可分辨。
- 必须分开的三件事：时域峰融合；倒谱拾取偏差；信息论 CRB（引用姊妹篇 C，不重做扫描）。
- 废弃口径（一小节说完）：历史 \(\Delta d_{\mathrm{DR}} \approx 38\,\mathrm{m}\) 是特定倒谱变换的旁瓣宽，随动态范围剧烈变化，**不是**本文基线。证据：EXP-20260730-025、EXP-20260801-001、EXP-20260806-002。

### 2 理论

- Brunone 项：\(J_u \propto k\bigl(\partial V/\partial t + a\,\mathrm{sign}(V)\,|\partial V/\partial x|\bigr)\)。陡波前（高频丰富）耗散更强，等效频率选择性衰减。
- 操作定义：峰漂移；EST（累积能量 E10–E90 的时间跨度，**不是**半高宽）；共同参考 STFT 频带能量；三种表观速度 \(a_{f0}\)、\(a_{\mathrm{onset}}\)、\(a_{\mathrm{peak}}\)。
- 历史上 onset 与 peak 的 26%/74% 分解是操作定义，**不是**“传播频散 / 波形频散”的唯一物理分解。
- 两套 \(k\) 不得混用：外加常数 \(k\)（机制隔离）vs `brunone_k(Re)`（LHS）。

### 3 数据

- 矩阵 A：\(k \in \{0, 0.01, 0.02, 0.05, 0.1, 0.2\} \times D \in \{5, 10, 20, 50, 100\}\,\mathrm{m}\) = **30** 个四裂缝工况。路径 `output/analysis/brunone_spacing_effect/`。审计 EXP-020。
- 矩阵 B：LHS Re 相关；盲协议 EXP-20260801-001；\(k\) 探针 EXP-20260730-010。**必须写进数据节：** 探针仅覆盖 tf = 8 s 与单一参数组合，不能当作 50 s 全程等效常数 \(k\)。
- 实现 \(k\) 分布（探针）：\(|V|\) 加权平均 0.0079；中位 0.0138；P95 0.0345。

### 4 结果：固定 \(k\) 的波形退化

- 图 1 波形演化（`fig1_waveform_evolution.png`）。
- 图 2 峰漂移与 EST 随 \(k\)（`fig2_trend.png`）；**数字以 EXP-20260730-024 为准**。
- 图 3 共同窗 STFT 与频带能量表（**EXP-024 重算**）。旧“60–150 Hz 衰减超过 40 dB”**不得沿用**，除非复算一致。
- 图 4 三种表观速度随 \(k\)。图注写明不能当作 \(c_p(\omega)\)。
- 图 5 双峰对比度热力图（**EXP-024 统一公式、合成双峰单元测试之后**）。旧 `fig4_rayleigh_heatmap.png` 仅可进 SI 并标注 provisional，主文不称 Rayleigh 判据。

### 5 结果：Re 相关 \(k\) 下的倒谱偏差

- 5 个 Brunone 单缝：容差 10 m 时 F1 = 0，20 m 时 F1 = 1 → 系统偏差落在 10–20 m。
- 稳态对照：2 m 容差即为 1。
- 全库盲倒谱 F1@10 m = 0.238 是**检测器表现**，与上述系统偏差分开写（全库还含假峰与多缝混叠）。
- 实现 \(k\) 分布表，强调弱 \(k\) 档 vs \(k = 0.2\) 示意。

### 6 讨论

- 机制等级：已观察耗散与峰形畸变 → 解释为回声估计器被带着走 → **未证明**严格频散。
- 工程：\(k = 0.2\) 是上界示意；现场-like \(k\) 更接近弱档；用稳态核套 Brunone 数据的偏差（约 30 m，引 C）大于倒谱 10–20 m 偏差。
- 模态阻尼率 \(\alpha\) 与 \(\sum k_{\mathrm{leak}}\) 秩相关 0.87–0.92，与缝数仅 0.44–0.51，**不是**缝深/缝数定位器（EXP-20260802-001 P3；一小节即可）。

### 7 结论

见下一节。

---

## 编号结论（草案）

1. 规定 Brunone \(k\) 使裂缝相关井口峰衰减、展宽、后移并逐渐融合；\(k = 0.2\)、间距 20 m 时经审计的峰漂移与能量弥散时间是本文定量锚点。
2. 高频能量相对低频优先减弱。由基频、onset 和峰值提取的表观速度发生分离。这是频率选择性耗散，不是已报告的 \(c_p(\omega)\)。
3. 在雷诺数相关 Brunone 摩阻下，盲倒谱单缝深度系统偏差为 10–20 m；准稳态对照在 2 m 容差下即成功。历史上 135 m 匹配容差掩盖了该偏差。
4. LHS 语料中的实现 \(k\) 属于弱 Brunone 档，不得用 \(k = 0.2\) 示意代表，也不得在没有流变标定的情况下映射到命名压裂液类型。
5. 结合姊妹篇 Cramér–Rao 结果，倒谱模糊不是信息墙。摩阻形式失配是比 \(k\) 的中等误差更大的深度偏差源。

---

## 现有数据 vs 需补实验

**已有：**

- 30 工况时间序列与 `metrics_summary.csv`；EXP-020 现象审计。
- 盲协议 10–20 m 偏差；\(k\) 探针；CRB / 形式失配数字从 Paper C 引用。

**投稿前必须补（EXP-024）：**

- 共同 STFT 参考（窗、重叠、dB 参考写入清单）。
- 峰漂移 / EST 公式与分析窗敏感性。
- 合成双峰单元测试后的分离指标与新热力图。
- 机器可读频带能量 CSV（`output/analysis/brunone_spacing_effect/audit_v2/`）。

**本篇明确不做：** 复传递函数 / 展开相位的严格频散实验（另文）；把 \(k\) 映射到流变；重开 38 m 分辨率矩阵（EXP-20260730-021 已取消）。

---

## 从 `docs/paper2_brunone_dispersion.md` 作废、禁止写入本稿的主张

- 三种表观速度差异“直接证明频散”；
- 峰漂移 26%/74% 作为唯一物理分解；
- 未审计的 60–150 Hz “>40 dB” 作为主文数字；
- Rayleigh 可分辨性矩阵作为最小缝距表；
- \(k\) 区间对应滑溜水 / 线性胶 / 交联冻胶；
- \(k = 0.1\) 时“等效深度偏差约 100 m”与“\(D \le 20\,\mathrm{m}\) 完全不可分辨”作为现场结论；
- \(\Delta d_{\mathrm{DR}} = 38.36\,\mathrm{m}\) 作为分辨率极限。

---

## English abstract (for later SPEJ submission)

Unsteady Brunone friction is often invoked to explain why shut-in water-hammer cepstra fail to resolve closely spaced fractures. This paper separates waveform degradation from information loss. A 30-run matrix with prescribed Brunone coefficient \(k\) from 0 to 0.2 and four equal fractures at spacings of 5–100 m shows frequency-selective dissipation, peak delay, and envelope spreading. At \(k = 0.2\) and 20 m spacing the current metrics give a peak shift of approximately 277 ms and an energy-spread time of approximately 101 ms. Apparent speeds extracted from the water-hammer fundamental, waveform onset, and peak disagree, so a single scalar wavespeed is not a stable descriptor after degradation. These observations support dissipative, frequency-selective attenuation of sharp fronts; they do not constitute a measurement of frequency-dependent phase speed. On wellbore cases with Reynolds-number-dependent \(k\), blind cepstrum exhibits a 10–20 m systematic depth bias, whereas the same protocol on quasi-steady-friction controls is already successful at a 2 m tolerance. Realized \(k\) in the Latin-hypercube corpus is weak (median approximately 0.014 over an 8 s probe), far below the \(k = 0.2\) illustration. Companion identifiability results show that location Cramér–Rao bounds inflate only by a factor of about 1.1–1.3 under Brunone friction, whereas fitting a steady-friction replica to Brunone data biases depth by about 30 m. Cepstral “blur” is therefore estimator distortion under dissipative fronts, not a diffraction-type resolution wall, and is secondary to friction-form misspecification in a full-waveform inversion setting. Historical matching that used a 135 m tolerance concealed the 10–20 m bias and must not be cited as successful Brunone localization.
