# 单通道水击多裂缝定位的可辨识性与误差预算：信息、估计器与模型失配

**English title:** Identifiability and Error Budgets for Multi-Fracture Localization from Single-Gauge Water-Hammer Records

**目标期刊：** SPE Journal · **Paper C** · 姊妹篇：[Paper A](./PaperA_Topology-Organized_Cepstral_Response_倒谱拓扑组织与首缝非局部性.md) · [Paper B](./PaperB_Brunone_Waveform_Degradation_非定常摩阻波形退化与倒谱模糊.md)

**关键词（≤5）：** water hammer; Cramér-Rao bound; maximum likelihood; model misspecification; hydraulic fracture diagnostics

**一句话问题：** 近距多缝定位失败，是井口记录没有信息、是倒谱一类估计器低效，还是正演模型（摩阻形式 / 波速）把估计拉偏？

证据入口：`research_ob/主题/T01-倒谱分辨率极限.md` · `research_ob/主题/drafts/paper_C_identifiability_efficiency.md`（2026-08-12 冻结）· `research_ob/主题/drafts/PaperC_MOC主路径结案.md` · T05 / T07 冻结为支撑负结果。

**本篇不新做科学实验。** 现行与波形退化合成的投稿正文为 [Paper BC](./PaperBC_Waveform_Degradation_and_Identifiability_波形退化与可辨识性.md)（2026-08-14 初稿）。本稿保留可辨识性细目。投稿只需：英文全文、SPE 出图、去掉内部实验编号、数据/代码可用性、补充材料。FNO 不进摘要，不作为第三条主线。

本文不重做 420 工况拓扑矩阵，不宣称波形频散机制。姊妹篇 A 的“倒谱是网络观测量”和 B 的“倒谱被耗散带着走”，只作为估计器离界的物理解释。

本稿为**中文工作框架**。英文摘要附于文末，供日后 SPEJ 投稿使用。

---

## SPE Journal 共同约束

- 篇幅 ≤ 10,000 词；图 ≤ 20。FNO / 高阶细节进 Supplementary Materials。
- 结构：题目 → 摘要（<500 词、无文献）→ 关键词 → 引言 → 理论/方法 → 数据 → 结果 → 编号结论 → 符号表 → 参考文献（author–year）。
- SI 单位。必须独立可读。共同正演：一维 MOC、Courant = 1、井口观测。
- **禁止**：把 \(\Delta d_{\mathrm{DR}}\approx 38.36\,\mathrm{m}\) 当信息论墙。不宣称现场已达厘米级。
- 投稿顺序：**先投 C**（证据已冻）→ B → A。
- 系列工程句：全波形反演须用对摩阻形式；倒谱只作初值；波速必须标定。

主文压进 10,000 词上限；主图 **8–10**。

---

## 核心创新点（已冻，可直接写）

1. 估计器无关 CRB：\(f_c = 20\,\mathrm{Hz}\)、SNR = 40 dB 时单缝位置下界约 **0.025 m**；5–40 m 间距上 CRB 不恶化 → **无信息论近距墙**。
2. 全波形 MOC-MLE 效率比 RMSE/CRB ≈ **0.96–1.39**（\(n \le 5\) 仍约 **0.86–1.18**）；倒谱在 SNR = \(\infty\) 仍有约 **12 m** 系统误差。
3. 实用墙是模型误差：brunone ↔ steady 约 **30 m**；波速 −1% 的非线性 MLE 偏差中位约 **120 m**（一阶走时 28–36 m 仅量级）。
4. 配对剪刀差：倒谱在稳态数据上 F1 0.232 → 0.642；P0 / PhaseNet 下降。评价必须含失配集。
5. 处方：MOC 全波形 MLE 为主估计器；倒谱 / 学习器仅作初值或对照。缝数是模型选择，不是位置 CRB。

---

## 摘要（中文）

单通道井口水击的近距多缝定位在倒谱、稀疏反卷积、字典剥离和学习式检测上普遍失败。本文将三种互斥假说——信息受限、偏差受限、估计器低效——用 Fisher 信息 / CRB 与全波形 MLE 分开。对齐配置下（井长 5000 m，波速 1450 m/s，Brunone，记录 50 s，截止 20 Hz），SNR = 40 dB 时单缝深度 CRB 约 0.025 m，且在 5–40 m 间距上保持厘米级。全波形 MLE 贴到界（RMSE/CRB = 0.96–1.39；\(n\le 5\) 时仍为 0.86–1.18）。倒谱在无限信噪比下仍有约 12 m 系统误差。实用墙是模型误差：稳态与 Brunone 形式互换约 30 m；波速 −1% 的非线性偏差中位约 120 m（一阶 28–36 m 仅量级）。配对迁移呈剪刀差：倒谱在稳态数据上变好，剥离字典与 PhaseNet 变差。缝数判定是模型选择，不受位置 CRB 约束。处方是以 MOC 全波形 MLE 为主估计器，倒谱与学习器仅作初值或对照，并在失配集上评价。本文不宣称现场已达厘米级。

---

## 章节结构

### 1 引言

- 现象：方法族差异极大，近距失败位置一致。
- 三种互斥假说：信息 / 偏差 / 估计器。
- 倒谱旁瓣宽是低效对照，不是信息论极限。
- 不宣称现场已达厘米级。

### 2 正演模型与观测算子

- MOC；Courant = 1；\(y_k = s_k(\theta) + n_k\)；零相位 4 阶 Butterworth 限带，默认 \(f_c = 20\,\mathrm{Hz}\)；加性白噪声。
- 参数 \(\theta\)：位置、\(\log_{10} C_f\)、\(\log_{10} k_{\mathrm{leak}}\)、波速、Brunone \(k\)。
- A 档：只估位置（其余已知）。B 档：位置 + 物性 + 波速。

### 3 数据

- 对齐配置：\(L = 5000\,\mathrm{m}\)，\(a = 1450\,\mathrm{m/s}\)，\(t_f = 50\,\mathrm{s}\)。
- 效率场景：`single_4000`、`dual_10m`、`dual_40m`。
- 配对 LHS：几何一致的 brunone ↔ steady，剪刀差用 300 个工况。
- 不把 close2000 近距基准当作可迁移证明（该基准波速误差为零、摩阻与训练一致）。

### 4 结果：信息下界

- CRB 随间距、信噪比、带宽变化；近距不恶化。
- 似然面在真值邻域单峰。
- 长记录 + 趾端锚下，波速作为讨厌参数对绝对深度的惩罚降至约 ×1.00–1.01。
- 图：CRB vs 间距；失配剖面。证据 EXP-20260806-002、EXP-20260806-003。

### 5 结果：估计器效率

- 表：MLE / 倒谱 / P0 vs CRB（冻结表：单缝、SNR = 40 dB 时 MLE RMSE 0.029 m vs CRB 0.025 m，效率比 1.14；倒谱 12.0 m）。
- \(n = 3/4/5\) 等间距：效率比 0.86–1.18（EXP-20260807-001）。
- 图：RMSE vs SNR；效率比。

### 6 结果：失配误差预算

- 形式失配：brunone → steady RMSE **29.6 m**；steady → brunone **31.9 m**；形式配对为 0。
- 波速 −1%：非线性中位偏差约 **120 m**；一阶 28–36 m 仅量级（EXP-20260807-003）。
- 剪刀差（EXP-20260807-002）：倒谱 F1 0.232 → 0.642；P0 0.396 → 0.089；PhaseNet 0.746 → 0.232。
- 一小节：嵌套 BIC 精确计数 = 1.0（oracle 构型）vs PhaseNet 盲计数约 30.5%。缝数 ≠ 位置 CRB。

### 7 讨论

- 学习方法若沿用，须对齐全波形似然，并在失配集上验收。
- FNO：主域相对 \(L^2 \approx 0.029\) 可用；纯 FNO-MLE 不过关（单缝约 2.7 m，双缝约 72 m）；混合精修有条件。主文学一段 + SI 图，**不作为第三条主线**。

### 8 结论

见下一节。

**局限（必须保留，放在结论前或结论中）：**

- 位置 CRB / MLE 在给定模型阶数下定义；嵌套 BIC 不是盲搜索。
- 主结果：\(f_c = 20\,\mathrm{Hz}\)、加性白噪声；半真实 / 现场噪声未测。
- Courant = 1 改波速会重划网格；一阶走时公式仅量级。
- 效率实验为等间距簇、\(n \le 5\)、缝区约 3500–4800 m。
- 不宣称可微代理能够定位；主估计器为 MOC 全波形。

---

## 编号结论（草案）

1. 近距失败不是信息论墙：截止 20 Hz、SNR = 40 dB 时位置 CRB 为厘米级，且在 5–40 m 间距上不恶化。
2. 全波形 MOC-MLE 可达界（RMSE/CRB ≈ 0.96–1.39；\(n \le 5\) 时仍约 0.86–1.18）。倒谱在无限信噪比下仍有约 12 m 误差；近距字典剥离不稳定。
3. 实用墙是模型失配：摩阻形式失配约 30 m；波速 −1% 的非线性中位绝对偏差约 120 m。
4. 缝数判定独立于位置 CRB。嵌套 BIC 在 oracle 嵌套模型下可恢复缝数；盲计数是另一更难的问题。
5. 配对摩阻形式剪刀差成立：倒谱在稳态数据上变好，剥离与 PhaseNet 变差。摩阻匹配的基准会高估可迁移性。
6. 主估计器是 MOC 全波形 MLE，可联合讨厌参数。倒谱与学习检测器只作初值或对照。神经算子波形代理可以准，不等于反演准。
7. 上述结论针对给定带宽与白噪声的合成记录，不宣称现场已达厘米级。

---

## 现有数据 vs 需补实验

**已有（全部完成，本篇不新做科学实验）：**

- EXP-20260806-002 Fisher / CRB
- EXP-20260806-003 估计器效率
- EXP-20260807-001 高阶效率与未知缝数
- EXP-20260807-002 配对失配剪刀差
- EXP-20260807-003 波速失配偏差
- FNO 门槛：EXP-20260808-001 / 002、EXP-20260809-001
- MOC 管线烟测：EXP-20260809-002 / 003 / 004

数字以 `research_ob/主题/drafts/PaperC_MOC主路径结案.md` 为准。

**明确不补：** N3 波速全量压测；纯 FNO 反演冲关；盲坐标下降阶数搜索；半真实噪声（写入局限）。

**写作层必做（非新科学实验）：**

- 效率表与 JSON 核对后再进 Word 模板；
- 图按 SPE 300–600 ppi 重出，图题自包含；
- SI 单位与符号表；
- 摘要与正文去掉内部实验编号；
- 数据 / 代码可用性与补充材料（高阶效率、BIC、FNO 分层、复现命令）。

---

## English abstract (for later SPEJ submission)

Near-spacing multi-fracture localization from a single wellhead water-hammer gauge routinely fails across cepstrum, sparse deconvolution, dictionary peeling, and learning-based detectors. We separate three mutually exclusive hypotheses—information limit, bias limit, and estimator inefficiency—using the Fisher information / Cramér–Rao bound (CRB) and a full-waveform maximum-likelihood estimator (MLE) on a method-of-characteristics wellbore model. Under an aligned setting (well length 5000 m, wavespeed 1450 m/s, Brunone friction, 50 s records, 20 Hz sensor cutoff), the single-fracture depth CRB at 40 dB SNR is approximately 0.025 m, and the bound remains centimetre-scale across 5–40 m spacings. Full-waveform MLE attains the bound, with RMSE/CRB in 0.96–1.39 (still 0.86–1.18 for up to five equally spaced fractures). Cepstrum retains approximately 12 m systematic error even at infinite SNR. The practical wall is model error: fitting a steady-friction replica to Brunone data, or the reverse, yields about 30 m depth bias; a −1% wavespeed mismatch produces a nonlinear MLE median absolute bias of about 120 m (first-order travel-time predictions of 28–36 m are order-of-magnitude only). A paired transfer test confirms a scissors pattern: cepstrum improves when the data are steady-friction, whereas a peeling dictionary and a PhaseNet detector trained on Brunone data degrade. Fracture-count selection is a model-selection problem and is not constrained by the position CRB. The operational prescription is MOC full-waveform MLE as the primary estimator, optionally with nuisance parameters, with cepstrum and learned detectors used only as coarse initializers or baselines and evaluated on mismatch sets. These synthetic bounds are not a claim of centimetre-scale field accuracy.
