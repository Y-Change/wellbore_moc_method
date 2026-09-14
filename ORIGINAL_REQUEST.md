# Original User Request

## Initial Request — 2026-09-09T06:44:41Z

系统化开展新版 MOC 求解器下裂缝物理参数对井筒水锤瞬态波形与声学倒谱特征的敏感性消融实验，定量揭示各参数对波形阶跃、能量衰减、频域损耗及倒谱定位幅值的映射规律，并生成学术级研究报告与标准数据集。

Working directory: e:/water_hammer_research/wellbore_moc_method/output/fracture_parameter_sensitivity
Integrity mode: development

## Requirements

### R1. 单变量消融与正交矩阵数值仿真
针对裂缝位置 $x_f$、柔度系数 $C_H$、滤失系数 $k_\text{leak}$、射孔阻抗 $R_p$、分流权重 $w_i$ 及裂缝间距 $\Delta x$ 建立基准取值区间与敏感性扫描网格。基于 `moc_simulate/wellbore_moc.py` 运行批量 MOC 仿真，保证数值解满足质量守恒与物理可行性。

### R2. 双摩阻形态（Steady vs Brunone）对照分析
对每组关键物理参数扫描，同步在 Darcy-Weisbach 纯稳态摩阻与 Brunone 非定常摩阻下运行仿真，分离并定量对比管壁瞬态剪切耗散与裂缝局部动力学响应的交互耦合效应。

### R3. 时频-倒谱多维度特征提取与指标矩阵构建
从仿真时程中自动提取首波 Joukowsky 降落、波前最大梯度 $(\partial H/\partial t)_\text{max}$、多时间窗 RMS 水头衰减率、FFT 高频（$f>1.5\,\mathrm{Hz}$）能量占比、1D 实倒谱峰值深度与幅值，以及 2D 滑窗倒谱空间分辨率，汇总为结构化 CSV/JSON 敏感性定量表。

### R4. 学术级综合研究报告与图版交付
绘制包含参数阶跃响应图、包络衰减对比图、频谱对数耗散图及 1D/2D 倒谱特写的高质量图集，输出具备发表水准的 Markdown 研究报告（README.md），系统阐明各参数的物理主导机制与深层神经网络反演的特征敏感边界。

## Acceptance Criteria

### 数值收敛与数据规范
- [ ] 全部仿真工况 100% 收敛通过（PASS），无任何 NaN/Inf 异常。
- [ ] 导出标准 schema `moc_lhs_v2.1` 兼容的 NPZ 数据集与高精度 CSV 时程文件。

### 敏感性量化与完整性
- [ ] 输出包含全部被测参数维度的结构化量化指标表 (`sensitivity_metrics.csv` 及 JSON)。
- [ ] 明确给出各裂缝参数对倒谱幅值、峰位偏差及波形衰减率的灵敏度排序或影响等级。

### 交付物与出版级图表
- [ ] 产出至少 4 组对应核心参数消融机制的对比图版（分辨率 $\ge 200\,\mathrm{dpi}$，含清楚物理标注）。
- [ ] 产出完整的机理分析报告 `README.md`，逻辑闭环，论据完备。

## Follow-up — 2026-09-11T11:37:49Z

# Teamwork Project Prompt — Final

实施水力压裂井筒水锤 MOC_V2 报告拆分重构，并基于生产级 MOC_V2 正演内核完成现场级 Base Case（4500m，3裂缝，10m间距）与 7 大专题敏感性仿真、Rainbow 2D 倒谱图版生成及独立学术技术报告撰写。

Working directory: e:\water_hammer_research\wellbore_moc_method
Integrity mode: development

## Requirements

### R1. 理论主报告精炼重构
- 修改 `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md`：
  - 删除原第六章与第七章；
  - 剥离原第三、四、五章（将其内容与图版迁移归档，为独立实验报告让路）；
  - 保留并完善第 1 章（V1 缺陷剖析与原始错误公式、Figure 0 缺陷图）与第 2 章（完整 MOC 控制方程、摩阻、顺应性解耦、射孔节流、斜坡边界、自洽初场、牛顿迭代收敛性证明与 2.6.4 五大典型裂缝类型划分）；
  - 同步更新 `docs/moc_v2_technical_report/build_report.py`，确保理论主报告构建通过且全量断言通过。

### R2. 现场级多裂缝参数敏感性正演流水线
- 编写一键式仿真与绘图脚本 `docs/moc_v2_technical_report/run_sensitivity_study.py`，调用 `moc_simulate.v2` 生产级内核完成以下全部正演计算：
  - **Base Case 基准工况**：水平井全长 $L=5000\,\mathrm{m}$，3 簇裂缝位于 $[4500, 4510, 4520]\,\mathrm{m}$（起点 $4500\,\mathrm{m}$，间距 $10\,\mathrm{m}$，趾端死水区 $480\,\mathrm{m}$），基准物性 $C_f=0.01\,\mathrm{m^2}, k_{leak}=1.0\times 10^{-4}\,\mathrm{m^{2.5}/s}, K_p=5.43\times 10^5\,\mathrm{s^2/m^5}, t_c=1.0\,\mathrm{s}$；
  - **专题 1（裂缝数量）**：$N_c \in [1, 2, 3, 4, 5, 6, 7, 8]$，起点 $4500\,\mathrm{m}$，间距 $10\,\mathrm{m}$；
  - **专题 2（裂缝间距）**：3 簇裂缝，起点 $4500\,\mathrm{m}$，间距 $d \in [5, 10, 15, 20, 25, 30, 50, 80]\,\mathrm{m}$；
  - **专题 3（顺应性储量）**：$C_f \in [0.002, 0.005, 0.010, 0.020, 0.030]\,\mathrm{m^2}$；
  - **专题 4（拟达西滤失）**：$k_{leak} \in [0.2, 0.6, 1.0, 3.0, 10.0]\times 10^{-4}\,\mathrm{m^{2.5}/s}$（包含 Type V 断层强滤失）；
  - **专题 5（限流射孔流阻）**：$K_p \in [1.5, 3.5, 5.43, 10.0, 25.0]\times 10^5\,\mathrm{s^2/m^5}$（对应孔数 $16, 8, 6, 4, 2$）；
  - **专题 6（关泵斜坡历时）**：$t_c \in [0.0, 0.5, 1.0, 1.5, 2.0]\,\mathrm{s}$；
  - **专题 7（多簇进液能力综合组合）**：对比 5 组典型工况：
    * 【中，中，中】：基准均匀进液；
    * 【高，中，中】：跟部首簇突进型；
    * 【高，中，高】：两头优势马鞍型；
    * 【中，中，高】：趾端逆向优势型；
    * 【死，中，高】：首簇砂堵死簇型。

### R3. Nature 级 7 大专题独立图版生成 (Rainbow 2D 倒谱)
- 在 `docs/moc_v2_technical_report/sensitivity_figures/` 目录下生成 Figure 1 至 Figure 7，每张图版均导出 300 DPI PNG 与矢量 SVG 格式；
- 每个专题对应一张独立复合图版，包含：
  - Panel a: 100s 全时程与关泵初期水头波形演化对比；
  - Panel b: 1维实倒谱曲线族（以垂直标线明确标注裂缝真实位置）；
  - Panel c/d: 典型工况的 **Rainbow 色阶 2D 连续倒谱云图（Cepstrogram）**（标注裂缝深度线，**不添加文本检出率统计判据**）；
- 统一字体（无衬线）、字号、标签标注与排版规范，完全符合 Nature-skill 科研绘图标准。

### R4. 独立的现场级正演响应与参数敏感性学术分析报告
- 撰写完整的学术级 Markdown 报告 `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md`：
  - 包含完整的引言背景、Base Case 参数定义、7 大专题物理机制深度剖析、Figure 1~7 高清嵌入与双语详细图注、多工况特征对比总结表；
  - 针对声波衰减、透射扼流、微间距混响、大反弹储能释放、断层强滤失退水与多簇进液非均匀性进行力学与声学机理深度阐述；
  - 报告排版精美，公式符号全部配齐定义与 SI 单位。

## Acceptance Criteria

### 仿真计算与代码执行
- [ ] `docs/moc_v2_technical_report/run_sensitivity_study.py` 能够独立一键执行完毕，无异常退出。
- [ ] 7 组专题实验所有用例（共 38+ 仿真）成功求解，数据完整且无 NaN/Inf。

### 图版交付完整性
- [ ] `docs/moc_v2_technical_report/sensitivity_figures/` 下成功生成 Figure 1 至 Figure 7 全部 14 份文件（7 个 PNG + 7 个 SVG）。
- [ ] 2D 倒谱图色阶严格采用 Rainbow（`cmap='rainbow'`），标注真实裂缝位置，且无多余的自动检出率文本覆盖。
- [ ] PNG 图像分辨率严格达到 300 DPI 以上，矢量 SVG 可无损缩放。

### 报告与文档交付
- [ ] 原理论报告 `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md` 成功剥离第 3~7 章，仅保留第 1~2 章，`build_report.py` 执行并通过。
- [ ] 新报告 `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md` 包含完整的 7 大专题章节、参数对比表、深入机理分析与 7 组图版嵌入。

### 质量与回归测试
- [ ] 执行 `pytest`：全库原有 99 项测试保持 100% 全部通过，无任何破坏回退。

## Follow-up — 2026-09-11T11:44:40Z

拆分原技术报告并新建《MOC_V2 现场工况正演与参数敏感性分析报告》（`MOC_V2_Simulation_Sensitivity_Report.md`）。以实际油田水平井现场尺度为基准（Base Case: 3 簇裂缝，起点 4500m，间距 10m），利用生产级 `moc_simulate.v2` 求解器执行 7 大专题敏感性仿真实验（裂缝数量 1~8、间距 5~80m、顺应性 Cf、滤失 kleak、射孔流阻 Kp、关泵斜坡 tc、非均匀进液能力组合），采用统一的 Nature 级图版规范绘制时域波形、1D 倒谱及 Rainbow 色阶 2D 连续倒谱图（标注真实裂缝深度，不作检出率文本判据），输出完整的专业技术分析报告。

Working directory: e:/water_hammer_research/wellbore_moc_method
Integrity mode: development

## Requirements

### R1. 原技术报告精简与理论聚焦 (Theory Report Streamlining)
- 修改 `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md`：
  - 删除原第六章与第七章；
  - 剥离原第三、四、五章（移交新报告）；
  - 保留并完善第 1 章（V1 缺陷剖析与数学方程）与第 2 章（V2 完备控制方程、摩阻、顺应性解耦、射孔节流、斜坡边界、自洽初场、牛顿收敛性与 2.6.4 五大类型划分）；
  - 同步调整 `docs/moc_v2_technical_report/build_report.py`，确保其断言校验全部通过。

### R2. 现场级 Base Case 与 7 大专题仿真流水线 (Simulation & Sensitivity Pipeline)
- 编写 `docs/moc_v2_technical_report/run_sensitivity_study.py`，基于 `moc_simulate.v2` 生产级内核执行全流程正演仿真：
  - **Base Case**：$L=5000\,\mathrm{m}, D=0.1397\,\mathrm{m}, a=1450\,\mathrm{m/s}, V_0=1.0\,\mathrm{m/s}, H_0=300\,\mathrm{m}, H_{ext}=100\,\mathrm{m}$；3 条裂缝位于 $[4500, 4510, 4520]\,\mathrm{m}$，参数为 $C_f=0.01\,\mathrm{m^2}, k_{leak}=1.0\times 10^{-4}\,\mathrm{m^{2.5}/s}, K_p=5.43\times 10^5\,\mathrm{s^2/m^5}, t_c=1.0\,\mathrm{s}$（余弦平滑关泵）；
  - **专题 1（裂缝数量）**：$N_c \in [1, 2, 3, 4, 5, 6, 7, 8]$，起点 $4500\,\mathrm{m}$，等间距 $10\,\mathrm{m}$；
  - **专题 2（裂缝间距）**：3 条裂缝，起点 $4500\,\mathrm{m}$，间距扫描 $[5, 10, 15, 20, 25, 30, 50, 80]\,\mathrm{m}$；
  - **专题 3（顺应性储量）**：基于 Base Case，扫描 $C_f \in [0.002, 0.005, 0.010, 0.020, 0.030]\,\mathrm{m^2}$；
  - **专题 4（拟达西滤失）**：基于 Base Case，扫描 $k_{leak} \in [0.2, 0.6, 1.0, 3.0, 10.0]\times 10^{-4}\,\mathrm{m^{2.5}/s}$（覆盖致密到 Type V 断层强滤失）；
  - **专题 5（限流射孔流阻）**：基于 Base Case，扫描 $K_p \in [1.5, 3.5, 5.43, 10.0, 25.0]\times 10^5\,\mathrm{s^2/m^5}$（对应孔数 $16, 8, 6, 4, 2$）；
  - **专题 6（关泵斜坡历时）**：基于 Base Case，扫描 $t_c \in [0.0, 0.5, 1.0, 1.5, 2.0]\,\mathrm{s}$；
  - **专题 7（进液能力组合）**：基于 Base Case，对比 5 组典型进液工况：
    * 【中，中，中】：基准均匀进液；
    * 【高，中，中】：跟部首簇突进型；
    * 【高，中，高】：两头优势马鞍型（中间受应力阴影强挤压）；
    * 【中，中，高】：趾端逆向优势型；
    * 【死，中，高】：首簇砂堵死簇型。

### R3. Nature 级统一图版绘制与 Rainbow 2D 倒谱 (Visualization & Rainbow Cepstrograms)
- 在 `docs/moc_v2_technical_report/sensitivity_figures/` 下输出 7 大专题独立图版（`fig1_fracture_count_sensitivity` 至 `fig7_intake_capacity_combinations`），每个图版均提供 300 DPI PNG 与矢量 SVG；
- 图版结构统一规范：
  - **Panel a**：时域水头波形时程与局部特征演化对比；
  - **Panel b**：1维实倒谱曲线族（以垂直虚线精准标定裂缝真实位置）；
  - **Panel c/d**：代表性工况的 **Rainbow 色阶（`cmap='rainbow'` / `jet` / `turbo`）2D 连续时空倒谱云图**，在距离轴上标注裂缝位置标线，**不添加文本检出率统计判据**。

### R4. 撰写《MOC_V2 现场工况正演与参数敏感性分析报告》 (Technical Report Writing)
- 新建文档 `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md`，深度融入上述 7 组仿真数据与图版；
- 全文按学术规范组织，涵盖引言、基准工况、7 个专题的深度物理力学机理剖析（声阻抗匹配、多簇混响干涉、波前导数锐化、断层强滤失泄压等），以及面向后续智能反演数据集构建的工程指导建议。

## Acceptance Criteria

### 仿真数据与图版完整性
- [ ] Base Case 严格满足 $L=5000\,\mathrm{m}$，裂缝在 $[4500, 4510, 4520]\,\mathrm{m}$，参数完全对齐现场实际。
- [ ] 7 大专题的所有算例全部成功执行，数据无 NaN/Inf，无虚假激波。
- [ ] `sensitivity_figures/` 下完整生成 7 组高分辨率图版（共 14 份文件：7 份 PNG + 7 份 SVG），2D 倒谱图色阶严格为 Rainbow。
- [ ] 倒谱图仅标定裂缝位置，不包含文本检出率判定。

### 报告与理论自洽性
- [ ] 原报告 `MOC_V2_Physics_Upgrade_Report.md` 已剥离第 3~7 章，`build_report.py` 运行通过且校验全 PASS。
- [ ] 新报告 `MOC_V2_Simulation_Sensitivity_Report.md` 内容详实、图文并茂、公式与参数释义规范。

### 全库代码健康度
- [ ] 全库自动化测试 `pytest` 保持 100% 全部通过（99 项以上）。

## 2026-09-13T14:26:29Z

面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究，以一维波动方程传递矩阵的逆散射层剥离滤波（Inverse Scattering Layer-Stripping）为声学解混基石，深度融合神经算子与声学时延偏置注意力，支撑硕士学位论文与中科院2区（JCR Q2）学术论文发表。持续推进直至产出客观实验结果，最终输出本轮完整学术研究报告（Markdown），附带专业图表、机理论述、显著成效与客观不足剖析。

Working directory: e:\water_hammer_research\wellbore_moc_method\PaperC_CJNO_Wellbore_Inversion
Integrity mode: development

## Requirements

### R1. 波动方程逆散射与声学传递矩阵解耦层 (Inverse Scattering Layer)
基于管网一维瞬变流声学传递矩阵与层剥离（Layer-Stripping）逆散射原理，构建端到端可微声学解混算子，严格沿井筒由跟部到趾端（x_1 → x_Nc）逐级补偿上游裂缝的透射衰减与多径串扰，中间层显式输出物理反射率序列 Γ_j 与支路导纳 Y_b,j，神经网络仅负责补偿非线性摩阻与射孔耗散。

### R2. 可解释神经算子与双轨物理映射网络 (Explainable Inversion Operator)
构建融合逆散射声学先验的深度神经算子网络（TG-DIS-DeepONet），输入实测瞬变水头波形与工况条件，输出：
1. 裂缝位置与起裂存在性检出（分类与定位）；
2. 各簇进液分配能力 α_j（满足严格单纯形守恒 sum(α_j) = 1.0）；
3. 裂缝水力储集顺应性 C_f,j 及声学导纳谱；
具备明确的物理白盒可解释性（反射系数物理对应、注意力热力图严格对应声学传播时延偏置）。

### R3. 学术规范对标评测与鲁棒性审计系统 (Benchmark & Noise Robustness Audit)
在 1,000 个案例物理数据集上建立全流程基线对标（1D-ResNet, 1D-FNO, DeepONet, TG-DeepONet），完成系统的消融实验；引入白噪声/有色噪声（SNR 30dB, 20dB, 10dB）及声速扰动（±1%）实施论文级泛化压力测试；产出符合中科院2区期刊标准的实验数据、误差分层分布、层剥离脉冲重构图、注意力归因热力图与出版级矢量图版。

### R4. 专业学术研究总结技术报告交付 (Research Technical Report)
生成本轮完整的技术研究 Markdown 报告（`PaperC_CJNO_Wellbore_Inversion/inverse_scattering_inversion_report.md`），报告包含：
1. 波动方程传递矩阵与层剥离逆散射理论推导及数学公式体系；
2. 逆散射神经算子网络架构与前向后向梯度流设计；
3. 专业学术图表（真值散点图、层剥离脉冲序列波形图、注意力物理热力图、多指标对标柱状图）；
4. 客观论述实验成效（指标突破、物理守恒性、多簇解混增益）；
5. 深入剖析客观不足与理论边界（单通道不适定性残差、超密簇极限、噪声衰减规律及未来可微 MOC 闭环指引）。

## Acceptance Criteria

### 裂缝识别与进液能力核心指标 (Inversion Performance)
- [ ] 进液份额 α 决定系数 R² 突破 0.75（密集多簇工况）
- [ ] 进液份额 α 平均绝对误差 MAE < 0.08（全集），单簇/稀疏工况 MAE < 0.03
- [ ] 1D 空间等效定位 Wasserstein 距离 W1 < 5.0 m
- [ ] 多裂缝位置检出与起裂分类准确率 F1-score > 0.88（容差 ±10m）
- [ ] 物理单纯形守恒偏差严格满足 max|sum(α) - 1.0| < 10⁻⁶
- [ ] 在 20dB 强噪声扰动下，核心指标衰减幅度 < 15%
- [ ] 产出高分辨率出版级图版（PNG 与矢量 SVG）及详尽技术报告 `inverse_scattering_inversion_report.md`

## 2026-09-13T14:28:38Z

面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究，以一维波动方程传递矩阵的逆散射层剥离滤波（Inverse Scattering Layer-Stripping）为声学解混基石，深度融合神经算子与声学时延偏置注意力，支撑硕士学位论文与中科院2区（JCR Q2）学术论文发表。持续迭代推进直到取得实质成效的结果，最终产出包含专业图表、深度论述、量化成效与客观不足剖析的完整研究技术报告。

Working directory: e:\water_hammer_research\wellbore_moc_method\PaperC_CJNO_Wellbore_Inversion
Integrity mode: development

## Requirements

### R1. 波动方程可微逆散射与声学层剥离算子 (Differentiable Layer-Stripping Layer)
基于管网一维瞬变流声学传递矩阵与层剥离（Layer-Stripping）逆散射原理，构建端到端可微声学解混算子，严格沿井筒由跟部到趾端（x_1 → x_Nc）逐级补偿上游裂缝的透射衰减与多径串扰，中间层显式输出物理反射率序列 Γ_j 与支路导纳 Y_b,j，消除多簇混叠均摊效应，神经网络仅负责补偿非线性摩阻与射孔耗散。

### R2. 可解释神经算子与双轨物理映射网络 (Explainable Inversion Operator)
构建融合逆散射声学先验的深度神经算子网络（TG-DIS-DeepONet），输入实测瞬变水头波形与工况条件，输出：
1. 裂缝位置与起裂存在性检出（分类与亚米级定位）；
2. 各簇进液分配能力 α_j（满足严格单纯形守恒 sum(α_j) = 1.0）；
3. 裂缝水力储集顺应性 C_f,j 及声学导纳谱；
具备明确的物理白盒可解释性（反射系数物理对应、注意力热力图严格对应声学传播时延偏置）。

### R3. 学术规范对标评测与两阶鲁棒性审计 (Benchmark & Noise Robustness Audit)
在 1,000 个案例物理数据集上建立全流程基线对标（1D-ResNet, 1D-FNO, DeepONet, TG-DeepONet），完成系统的消融实验；引入白噪声/有色噪声（SNR 30dB, 20dB, 10dB）及声速扰动（±1%）实施论文级泛化压力测试；产出符合中科院2区期刊标准的实验数据、误差分层分布、层剥离脉冲重构图、注意力归因热力图与出版级矢量图版。

### R4. 完整学术级研究报告交付 (Final Research Report Deliverable)
编写并输出高密度、出版级的研究技术报告 `PaperC_CJNO_Wellbore_Inversion/phase3_inverse_scattering_report.md`，内嵌高分辨率图表与详尽学术论述，全面系统地说明本轮取得的技术成效、相较前期预研的大幅跃升，并客观剖析当前单通道物理观测下的边界局限与客观不足。

## Acceptance Criteria

### 裂缝识别与进液能力核心指标 (Inversion Performance)
- [ ] 进液份额 α 决定系数 R² 突破 0.75（密集多簇工况）
- [ ] 进液份额 α 平均绝对误差 MAE < 0.08（全集），单簇/稀疏工况 MAE < 0.03
- [ ] 1D 空间等效定位 Wasserstein 距离 W1 < 5.0 m
- [ ] 多裂缝位置检出与起裂分类准确率 F1-score > 0.88（容差 ±10m）
- [ ] 物理单纯形守恒偏差严格满足 max|sum(α) - 1.0| < 10⁻⁶
- [ ] 在 20dB 强噪声扰动下，核心指标衰减幅度 < 15%
- [ ] 完成包含专业图表、深度学术论述与客观不足剖析的完整研究技术报告交付
