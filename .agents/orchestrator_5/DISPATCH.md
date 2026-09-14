# Dispatch Record

## 2026-09-13T14:28:38Z

You are the Project Orchestrator for the research project:
'面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究'.

Your Working Metadata Directory:
e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5
Project Working Directory:
e:\water_hammer_research\wellbore_moc_method\PaperC_CJNO_Wellbore_Inversion
Repository Root:
e:\water_hammer_research\wellbore_moc_method
Authoritative User Request:
e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md (see the section starting at ## 2026-09-13T14:28:38Z).

Integrity Mode: development

Core Requirements:
R1. 波动方程可微逆散射与声学层剥离算子 (Differentiable Layer-Stripping Layer):
- 基于管网一维瞬变流声学传递矩阵与层剥离（Layer-Stripping）逆散射原理，构建端到端可微声学解混算子；
- 严格沿井筒由跟部到趾端（x_1 → x_Nc）逐级补偿上游裂缝的透射衰减与多径串扰；
- 中间层显式输出物理反射率序列 Γ_j 与支路导纳 Y_b,j，消除多簇混叠均摊效应，神经网络仅负责补偿非线性摩阻与射孔耗散。

R2. 可解释神经算子与双轨物理映射网络 (Explainable Inversion Operator):
- 构建融合逆散射声学先验的深度神经算子网络（TG-DIS-DeepONet），输入实测瞬变水头波形与工况条件；
- 输出：
  1. 裂缝位置与起裂存在性检出（分类与亚米级定位）；
  2. 各簇进液分配能力 α_j（满足严格单纯形守恒 sum(α_j) = 1.0）；
  3. 裂缝水力储集顺应性 C_f,j 及声学导纳谱；
- 具备明确的物理白盒可解释性（反射系数物理对应、注意力热力图严格对应声学传播时延偏置）。

R3. 学术规范对标评测与两阶鲁棒性审计 (Benchmark & Noise Robustness Audit):
- 在 1,000 个案例物理数据集上建立全流程基线对标（1D-ResNet, 1D-FNO, DeepONet, TG-DeepONet），完成系统的消融实验；
- 引入白噪声/有色噪声（SNR 30dB, 20dB, 10dB）及声速扰动（±1%）实施论文级泛化压力测试；
- 产出符合中科院2区期刊标准的实验数据、误差分层分布、层剥离脉冲重构图、注意力归因热力图与出版级矢量图版。

R4. 完整学术级研究报告交付 (Final Research Report Deliverable):
- 编写并输出高密度、出版级的研究技术报告 `PaperC_CJNO_Wellbore_Inversion/phase3_inverse_scattering_report.md`；
- 内嵌高分辨率图表与详尽学术论述，全面系统地说明本轮取得的技术成效、相较前期预研的大幅跃升，并客观剖析当前单通道物理观测下的边界局限与客观不足。

Acceptance Criteria:
- [ ] 进液份额 α 决定系数 R² 突破 0.75（密集多簇工况）
- [ ] 进液份额 α 平均绝对误差 MAE < 0.08（全集），单簇/稀疏工况 MAE < 0.03
- [ ] 1D 空间等效定位 Wasserstein 距离 W1 < 5.0 m
- [ ] 多裂缝位置检出与起裂分类准确率 F1-score > 0.88（容差 ±10m）
- [ ] 物理单纯形守恒偏差严格满足 max|sum(α) - 1.0| < 10⁻⁶
- [ ] 在 20dB 强噪声扰动下，核心指标衰减幅度 < 15%
- [ ] 完成包含专业图表、深度学术论述与客观不足剖析的完整研究技术报告交付

Lifecycle & Coordination:
- Continuously maintain and update your `progress.md` and `BRIEFING.md` in `e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5\`.
- Coordinate and dispatch specialist subagents as needed.
- Report milestone completions and final completion back to Sentinel.
