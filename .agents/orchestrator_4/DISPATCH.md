# Dispatch Record

## 2026-09-13T14:27:32Z

You are the Project Orchestrator for the research project:
'面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究'.

Your Working Metadata Directory:
e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_4
Project Working Directory:
e:\water_hammer_research\wellbore_moc_method\PaperC_CJNO_Wellbore_Inversion
Repository Root:
e:\water_hammer_research\wellbore_moc_method
Authoritative User Request:
e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md (see the section starting at ## 2026-09-13T14:26:29Z).

Integrity Mode: development

Core Requirements:
R1. 波动方程逆散射与声学传递矩阵解耦层 (Inverse Scattering Layer):
- 基于管网一维瞬变流声学传递矩阵与层剥离（Layer-Stripping）逆散射原理，构建端到端可微声学解混算子；
- 严格沿井筒由跟部到趾端（x_1 → x_Nc）逐级补偿上游裂缝的透射衰减与多径串扰；
- 中间层显式输出物理反射率序列 Γ_j 与支路导纳 Y_b,j，神经网络仅负责补偿非线性摩阻与射孔耗散。

R2. 可解释神经算子与双轨物理映射网络 (Explainable Inversion Operator):
- 构建融合逆散射声学先验的深度神经算子网络（TG-DIS-DeepONet），输入实测瞬变水头波形与工况条件；
- 输出：
  1. 裂缝位置与起裂存在性检出（分类与定位）；
  2. 各簇进液分配能力 α_j（满足严格单纯形守恒 sum(α_j) = 1.0）；
  3. 裂缝水力储集顺应性 C_f,j 及声学导纳谱；
- 具备明确的物理白盒可解释性（反射系数物理对应、注意力热力图严格对应声学传播时延偏置）。

R3. 学术规范对标评测与鲁棒性审计系统 (Benchmark & Noise Robustness Audit):
- 在 1,000 个案例物理数据集上建立全流程基线对标（1D-ResNet, 1D-FNO, DeepONet, TG-DeepONet），完成系统的消融实验；
- 引入白噪声/有色噪声（SNR 30dB, 20dB, 10dB）及声速扰动（±1%）实施论文级泛化压力测试；
- 产出符合中科院2区期刊标准的实验数据、误差分层分布、层剥离脉重构图、注意力归因热力图与出版级矢量图版。

R4. 专业学术研究总结技术报告交付 (Research Technical Report):
- 生成本轮完整的技术研究 Markdown 报告（`PaperC_CJNO_Wellbore_Inversion/inverse_scattering_inversion_report.md`）；
- 报告包含：
  1. 波动方程传递矩阵与层剥离逆散射理论推导及数学公式体系；
  2. 逆散射神经算子网络架构与前向后向梯度流设计；
  3. 专业学术图表（真值散点图、层剥离脉冲序列波形图、注意力物理热力图、多指标对标柱状图）；
  4. 客观论述实验成效（指标突破、物理守恒性、多簇解混增益）；
  5. 深入剖析客观不足与理论边界（单通道不适定性残差、超密簇极限、噪声衰减规律及未来可微 MOC 闭环指引）。

Acceptance Criteria:
- 进液份额 α 决定系数 R² 突破 0.75（密集多簇工况）
- 进液份额 α 平均绝对误差 MAE < 0.08（全集），单簇/稀疏工况 MAE < 0.03
- 1D 空间等效定位 Wasserstein 距离 W1 < 5.0 m
- 多裂缝位置检出与起裂分类准确率 F1-score > 0.88（容差 ±10m）
- 物理单纯形守恒偏差严格满足 max|sum(α) - 1.0| < 10⁻⁶
- 在 20dB 强噪声扰动下，核心指标衰减幅度 < 15%
- 产出高分辨率出版级图版（PNG 与矢量 SVG）及详尽技术报告 `inverse_scattering_inversion_report.md`

Lifecycle & Coordination:
- Continuously maintain and update your `progress.md` and `BRIEFING.md` in `e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_4\`.
- Coordinate and dispatch specialist subagents as needed.
- Report milestone completions and final completion back to Sentinel.
