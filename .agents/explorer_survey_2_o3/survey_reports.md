# MOC_V2 技术报告重构与敏感性仿真报告架构调研报告
## Technical Survey Report on MOC_V2 Theory Report Streamlining (R1) & Simulation Sensitivity Report Architecture (R4)

**调查员 (Investigator)**: Explorer 2 (Technical Report & Build Pipeline Investigator)  
**工作目录 (Working Directory)**: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_2_o3`  
**报告日期**: 2026-09-11  
**适用工单**: R1（理论主报告精简与构建校验）、R4（现场级正演响应与参数敏感性学术分析报告架构设计）

---

## 1. 调研背景与任务目标

根据现场压裂工程与高水平学术发表规范要求，原技术报告 `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md` 需要进行“理论主线与现场实验彻底解耦”的重大架构拆分重构：
1. **理论主报告精炼聚焦（R1）**：将 `MOC_V2_Physics_Upgrade_Report.md` 塑造为纯粹的流体力学、特征线法数值离散、裂缝断裂力学柔度解耦、限流射孔非线性节流扼流与牛顿迭代收敛性证明的理论基石报告。剥离基于原四裂缝（$x_f \in [4100, 4120, 4140, 4160]\,\mathrm{m}$）的数值实验章节（原第 3、4、5 章），删除过渡性的参数对照表与反演接口展望（原第 6、7 章），保留并完善第 1 章（V1 缺陷剖析与 Figure 0）与第 2 章（完备控制方程体系与 2.6.4 五大裂缝类型）。同步更新 `docs/moc_v2_technical_report/build_report.py` 自动化构建校验脚本，确保全量断言通过。
2. **现场级仿真与参数敏感性独立报告架构（R4）**：以实际油田长水平井现场尺度为基准（Base Case: 全长 $5000\,\mathrm{m}$，3 簇裂缝位于 $[4500, 4510, 4520]\,\mathrm{m}$，间距 $10\,\mathrm{m}$），围绕生产级 `moc_simulate.v2` 求解器执行的 7 大专题敏感性仿真（裂缝数量、间距、顺应性 $C_f$、滤失 $k_{leak}$、射孔流阻 $K_p$、关泵斜坡 $t_c$、非均匀进液组合），撰写具有国际顶级学术水准的独立 Markdown 报告 `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md`，深度融入 Nature 级 Figure 1~7 图版（含 Rainbow 色阶 2D 连续倒谱云图，标注真实深度，无冗余文本判定）。

---

## 2. 理论主报告 `MOC_V2_Physics_Upgrade_Report.md` 深度解构

### 2.1 现有文件现状与章节体量统计
- **文件路径**: `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md`
- **总行数**: 878 行
- **总字节数**: 117,494 字节（约 70,030 字符）
- **章节分布统计**:

| 章节编号与名称 | 起始-结束行号 | 字符数 (Chars) | UTF-8 字节数 (Bytes) | 处置决策 (R1/R4) |
| :--- | :---: | :---: | :---: | :--- |
| **标题、执行摘要与目录** | 1 - 57 | 2,750 | 4,820 | **保留并更新**（移除 Ch 3-7 目录，摘要聚焦理论） |
| **第一章：原始仿真器（MOC_V1）物理缺陷剖析与错误公式溯源** | 58 - 123 | 4,730 | 8,240 | **全量保留并完善**（含 1.1~1.3 与 Figure 0） |
| **第二章：MOC_V2 物理内核数学建模与数值求解体系** | 124 - 709 | 47,788 | 79,697 | **核心基石，全量保留**（含 2.1~2.8 全部子节与公式） |
| **第三章：时域波形全景与局部的四阶段物理演进对比（Figure 1 深度解析）** | 710 - 743 | 2,860 | 4,810 | **剥离迁移归档**（四裂缝时域波形实验） |
| **第四章：2D 倒谱演进与多簇裂缝声学照亮机制（Figure 2 深度解析）** | 744 - 781 | 3,320 | 5,610 | **剥离迁移归档**（四裂缝 2D 倒谱演进实验） |
| **第五章：射孔阻抗与关泵动力学参数敏感性阵列（Figure 3 深度解析）** | 782 - 830 | 4,210 | 7,150 | **剥离迁移归档**（四裂缝敏感性雷达实验） |
| **第六章：物理内核全要素参数对照矩阵** | 831 - 860 | 2,490 | 4,140 | **删除移除**（属于过渡性对比表） |
| **第七章：结论与后续智能反演接口展望** | 861 - 889 | 1,882 | 3,027 | **删除移除**（反演与实验展望统一交由新报告） |

### 2.2 第一章（MOC_V1 缺陷剖析）保留内容审查
第一章详细记录了原版仿真器的 6 大核心物理缺陷，必须完整保留：
1. **1.1 缺陷精炼提纲与六大核心病态**:
   - 趾端边界激波污染（$V(L, 0) = 0$ 强切触发 $+147.8\,\mathrm{m}$ 假激波）；
   - 定常摩阻坡降缺失（平直初场违背达西沿程平衡 $dH/dx = -J$）；
   - 流体分流与死水区缺失（违背质量守恒 $\sum q_{j,0} = Q_0$，死水区缺失）；
   - 并联声学短路完全屏蔽（$K_p = 0 \implies \Gamma \to -1, T \to 0$）；
   - 微尺度顺应性导致抽空塌陷（$C_f = 10^{-5}\,\mathrm{m^2} \implies H_{min} = -57.45\,\mathrm{m}$）；
   - 理想阶跃关泵激发高频混响伪峰（$\Delta t_c \to 0 \implies |dH/dt|_{max} \sim 1.48 \times 10^5\,\mathrm{m/s}$，激发 $36.25\,\mathrm{Hz}$ 腔体混响）。
2. **1.2 原始缺陷数学方程与边界公式溯源**:
   - 包含 6 个错误方程公式与物理机理展开式，形成对比基准。
3. **1.3 Figure 0 缺陷诊断图版与双语图注**:
   - 图像引用: `![Figure 0: MOC_V1 原始缺陷诊断图 (趾端假激波与空间速度场断裂)](figures/fig0_v1_baseline_defects.png)`；
   - 双语图注完全保留，对应文件 `docs/moc_v2_technical_report/figures/fig0_v1_baseline_defects.png` 与 `.svg` 保持现状。

### 2.3 第二章（MOC_V2 完备物理内核）保留内容审查
第二章是整个 MOC_V2 算法的数学理论心脏，共有 8 个二级子节与 17 个三/四级小节，必须 100% 完整保留并配齐符号释义：
1. **2.1 井筒瞬变流基本控制方程与特征线法（MOC）数值离散求解体系**:
   - 2.1.1 井筒瞬变流基本控制 PDE：
     $$\frac{\partial H}{\partial t} + \frac{a^2}{g A} \frac{\partial Q}{\partial x} = 0, \quad \frac{\partial Q}{\partial t} + g A \frac{\partial H}{\partial x} + g A J = 0$$
   - 2.1.2 特征线相容方程严格推导：特征方向 $\frac{dx}{dt} = \pm a$；
   - 2.1.3 正负特征线微分形式：$dH \pm B dQ \pm a J dt = 0$，特征阻抗 $B = \frac{a}{g A}$；
   - 2.1.4 时空网格离散与 Courant 稳定性条件：$Cr = \frac{a \Delta t}{\Delta x} = 1.0$；
   - 2.1.5 MOC 菱形时空特征网格拓扑图示（ASCII 拓扑网格）；
   - 2.1.6 正负李曼不变量（Riemann Invariants）：
     $$C_P = H_{i-1}^n + B Q_{i-1}^n - R Q_{i-1}^n |Q_{i-1}^n| - a \Delta t J_{u, i-1}^n$$
     $$C_M = H_{i+1}^n - B Q_{i+1}^n + R Q_{i+1}^n |Q_{i+1}^n| + a \Delta t J_{u, i+1}^n$$
     其中沿程达西摩阻因子 $R = \frac{f \Delta x}{2 g D A^2}$；
   - 2.1.7 普通内节点 $(i, n+1)$ 显式代数解：
     $$H_i^{n+1} = \frac{C_P + C_M}{2}, \quad Q_i^{n+1} = \frac{C_P - C_M}{2 B}, \quad C_P^{\text{vel}} = \frac{C_P}{B A}, \quad C_M^{\text{vel}} = \frac{C_M}{B A}$$
   - 2.1.8 Brunone 非定常摩阻项离散与平滑因子：
     $$J_u = \frac{k_B}{2 g} \left( \frac{\partial V}{\partial t} + a \cdot \mathrm{sign}(V) \left| \frac{\partial V}{\partial x} \right| \right), \quad V_{smooth} = 0.05\,\mathrm{m/s}$$
   - 2.1.9 参数物理意义释义表（首现参数高密度表）。
2. **2.2 摩阻定义**:
   - Darcy-Weisbach 稳态摩阻与 Brunone 非恒定壁面剪切力。
3. **2.3 地质参数定义与深度解耦论证**:
   - 2.3.1 顺应性双重机理解耦：岩石骨架弹性形变 $C_{p, rock}$ vs 流体声容 $C_{p, fluid}$；
   - 2.3.2 三大经典裂缝力学构型柔度解析解：
     $$C_{p, rock}^{Penny} = \frac{8 (1 - \nu^2)}{3 E} R^3 = \frac{4 (1 - \nu)}{3 G_{shear}} R^3$$
     $$C_{p, rock}^{PKN} = \frac{\pi (1 - \nu^2) h_f^2 L_f}{E}, \quad C_{p, rock}^{KGD} = \frac{\pi (1 - \nu^2) h_f L_f^2}{E}$$
   - 2.3.3 文献量级对齐（Luo et al. 2023, Valko 1995: $C_p \sim 10^{-6}\,\mathrm{m^3/Pa}, C_f \sim 0.01\,\mathrm{m^2}$）；
   - 2.3.4 Carter 滤失与拟达西滤失双轨制对齐（$k_{leak}, H_{ext}$）。
4. **2.4 限流射孔非线性节流扼流耦合模型与文献对齐**:
   - Berchenko (1998) 节流方程：$\Delta H_{perf} = K_p q_p |q_p|$，其中 $K_p = \frac{1}{2 g C_d^2 A_p^2}$。
5. **2.5 现场关泵斜坡动力学边界模型与文献对齐**:
   - 关泵历时 $t_c \in [0.5, 2.0]\,\mathrm{s}$，余弦平滑过渡函数。
6. **2.6 基态流场初始化与现场压力体系对齐**:
   - 2.6.1 基态动量守恒空间解析积分解（流速逐级分流，趾端死水区 $V=0$）；
   - 2.6.2 井口基准水头 $H_{wellhead, 0}$ 与超静水范式；
   - 2.6.3 现场工程绝对参数与超静水相对水头对照表（含 $40.0\sim 95.0\,\mathrm{MPa}$ 对齐 $H_{wellhead} \in [200, 600]\,\mathrm{m}$，地层压力 $30.0\sim 65.0\,\mathrm{MPa}$ 对齐 $H_{ext} \in [50, 200]\,\mathrm{m}$，净压力 $1.5\sim 4.5\,\mathrm{MPa}$，达西摩阻 $0.5, 3.0\,\mathrm{MPa}$）；
   - 2.6.4 水平井多簇压裂 5 大典型裂缝类型划分：
     * Type I: 优势发育主进液簇；
     * Type II: 均衡/正常发育簇；
     * Type III: 受抑/欠发育弱进液簇；
     * Type IV: 砂堵闭合/未起裂死簇；
     * Type V: 沟通天然断层/强微裂缝簇；
     * 幂律耦合公式与 `sample_preset_scenario`。
7. **2.7 非线性耦合方程唯一物理实根定理与牛顿迭代收敛性**:
   - 裂缝节点耦合方程：$H_{well, j}(q_p) = H_{moc0} - \frac{B}{2} q_p$；
   - 残差函数 $F(q_p) = q_p - \left[ \frac{C_f}{\Delta t} (H_{frac, j}(q_p) - H_{frac, j}^{old}) + q_{leak, j}(H_{frac, j}(q_p)) \right] = 0$；
   - 严格单调性证明 $F'(q_p) \ge 1.0 > 0$，保证根的唯一性与牛顿法 2~4 次机内双精度二次收敛。
8. **2.8 声学阻抗网络拓扑、反射/透射系数解析推导与破除短路机理**:
   - 声学特征阻抗 $Z_w = \rho a / A$；
   - 射孔阻抗 $Z_p = 2 \rho K_p q_{p,0}$；
   - 裂缝容抗 $Z_c = \Delta t / (\rho g C_f)$；
   - 并联总阻抗 $Z_{shunt} = Z_p + Z_c$；
   - 反射与透射系数：$\Gamma = \frac{-Z_w}{2 Z_{shunt} + Z_w}, \quad T = \frac{2 Z_{shunt}}{2 Z_{shunt} + Z_w}$；
   - 严格阐明 $Z_p = 0 \implies \Gamma \to -1, T \to 0$（短路屏蔽）与 $Z_p \sim 10^7\text{--}10^8 \implies T \sim 10\%\sim 50\%$（破除短路）的物理机制。

### 2.4 原第 3~7 章剥离与删除清单

| 原章节 | 内容与图表 | 原测试工况 | 剥离/迁移/删除去向 |
| :--- | :--- | :--- | :--- |
| **第三章** | 100s 波形宏观演化、前 12s 局部声学特写、关泵压力变化率 $|dH/dt|$、Figure 1 及图注 | Quad 四裂缝 $[4100, 4120, 4140, 4160]\,\mathrm{m}$ | **剥离出理论主报告**。其波形演化与波前导数分析方法被提炼并升级至新报告 Base Case 与专题 6 |
| **第四章** | 2D 倒谱时空云图演化、声学短路病态诊断、限流射孔照亮、一阶导数增强、Figure 2 及图注 | Quad 四裂缝 $[4100, 4120, 4140, 4160]\,\mathrm{m}$ | **剥离出理论主报告**。其 2D 倒谱照亮机理升级至新报告并采用统一 Rainbow 色阶云图呈现 |
| **第五章** | 射孔孔数 $N_p$ 相变、关泵斜坡历时 $t_c$ 线性时滞律 $\Delta t_{half} \approx 0.64 t_c$、物理完备度雷达、量化基准对比、Figure 3 及图注 | Quad 四裂缝 $[4100, 4120, 4140, 4160]\,\mathrm{m}$ | **剥离出理论主报告**。其参数敏感性机理直接映射至新报告专题 5、专题 6 与交叉综合分析中 |
| **第六章** | 6.1 MOC_V1 原版 vs MOC_V2 升级版全要素参数对照表 (Table 6.1) | 历史升级对照 | **彻底删除**。理论主报告已有 1.1 与 2.6.3 高密度对照，无需重复保留 |
| **第七章** | 7.1 本次物理内核升级的核心成就、7.2 对后续闭环智能反演算法的支撑展望 | 总结与反演展望 | **彻底删除**。理论主报告末尾改由精炼的小结与新报告指引替代；反演展望移至新报告第 11 章 |

---

## 3. 构建脚本 `build_report.py` 机制深度审计与调整方案

### 3.1 `build_report.py` 运行机制解析
通过代码审计，`docs/moc_v2_technical_report/build_report.py`（985 行）具有以下运行逻辑：
1. **代码内嵌全文本生成**: 变量 `REPORT_CONTENT` 包含了整个报告的完整 Markdown 字符串（行 12 至 889）；
2. **写回目标文件**:
   ```python
   with open(REPORT_PATH, "w", encoding="utf-8") as f:
       f.write(REPORT_CONTENT.strip() + "\n")
   ```
   **关键发现**：`build_report.py` 是报告的**单一事实来源（Single Source of Truth）**。如果仅仅修改 `MOC_V2_Physics_Upgrade_Report.md` 而不修改 `build_report.py` 中的 `REPORT_CONTENT`，一旦执行 `python build_report.py`，外部文件修改将被完全覆盖回退！
3. **自校验流水线（Integrity Assertions）**:
   - **长度断言**: `assert len(generated_content) > 50000, f"Report content too short: {len(generated_content)} bytes"`
     *注：该断言实际统计的是 Python `str` 字符数而非底层字节数*；
   - **LaTeX 语法定界符配对平衡性断言**:
     ```python
     c2 = generated_content.count("$$")
     c1 = generated_content.count("$") - c2 * 2
     assert c2 % 2 == 0, f"Unbalanced $$ in report: count={c2}"
     assert c1 % 2 == 0, f"Unbalanced single $ in report: count={c1}"
     ```
   - **关键词全要素存在性断言**:
     `required_keywords` 包含 62 个结构标题、关键公式、文献引用与核心参数（行 908 至 976）。

### 3.2 移除第 3~7 章后的断言失效分析
通过在只包含第 1~2 章文本（前 709 行）的环境下运行沙箱测试，审计结论如下：
1. **`required_keywords` 关键词检查**:
   - **惊人发现**：`build_report.py` 中定义的全部 62 个关键词，**全部 100% 分布在第 2 章内**（涉及 2.1、2.3、2.6、2.6.4、2.7），**第 3~7 章中没有任何关键词属于 `required_keywords`**！
   - **结论**：直接剔除第 3~7 章，`required_keywords` 仍然 100% 存在，**不会发生关键词缺失报警**！
2. **LaTeX 公式配对检查**:
   - 第 1~2 章内的 `$$` 定界符出现 212 次（106 对，偶数）；
   - 第 1~2 章内的单 `$` 定界符（排除 `$$` 后）出现 1454 次（727 对，偶数）；
   - **结论**：第 1~2 章自身的数学公式完全闭合，**不会触发 LaTeX 平衡报警**！
3. **报告长度检查 `len(generated_content) > 50000`**:
   - 原包含 7 章的完整报告字符数为 70,030 字符（117,494 字节）；
   - 剥离第 3~7 章后，第 1~2 章原始字符数为 55,268 字符（92,757 字节）；
   - 若将目录中的 3~7 章条目清除，字符数约为 54,960 字符；
   - 虽然 54,960 > 50,000 可以通过断言，但容错余量仅剩约 9.9%（~4,960 字符）。若精简过程中字符数跌破 50,000，该断言将立即抛出 `AssertionError: Report content too short`！
   - 此外，断言提示信息为 `bytes`，但实际检测的是 `len(str)`（字符数），在双字节中文字符环境下，9.2 万字节仅对应 5.5 万字符。

### 3.3 `build_report.py` 精确修改清单

为保证 `build_report.py` 既能正确生成精炼后的主报告，又能在全量断言下 100% PASS，Worker M1 必须实施以下精确修改：
1. **修改 `REPORT_CONTENT`**:
   - **标题与副标题调整**:
     ```markdown
     # MOC_V2 水力压裂井筒水击波物理内核升级技术报告
     ## Physical Kernel Upgrade, Mathematical Modeling, and Hydrodynamic Principles in Wellbore Transient Flow (MOC_V2)
     ```
   - **执行摘要（Executive Summary）更新**:
     保留四大物理升级闭环总结，明确指出本报告聚焦于第一性原理数学方程与求解器内核，全尺寸现场仿真与 7 大专题敏感性分析详见姊妹篇报告 `MOC_V2_Simulation_Sensitivity_Report.md`。
   - **目录（TOC）更新**:
     彻底删除原第 3、4、5、6、7 章的目录项，仅保留第一章（1.1~1.3）与第二章（2.1~2.8）。
   - **正文内容更新**:
     保留第一章全部内容（1.1、1.2、1.3 及 Figure 0）；保留第二章全部内容（2.1~2.8）；在 2.8 节后增加规范的理论总结小结与新实验报告导引段落；彻底截断删除原第 3~7 章的正文与旧图版引用。
2. **修改断言校验代码**:
   - 将长度断言调整为既安全又严格的阈值：
     ```python
     # Theory report must remain a comprehensive mathematical reference (>45,000 characters)
     assert len(generated_content) >= 45000, f"Theory report content too short: {len(generated_content)} chars"
     assert len(generated_content.encode("utf-8")) > 75000, f"Theory report byte size too small: {len(generated_content.encode('utf-8'))} bytes"
     ```
   - 保留 LaTeX `$$` 与 `$` 平衡校验；
   - 保留并完善 `required_keywords` 列表，确保涵盖第二章所有关键公式与理论概念；
   - 增加对 Figure 0 图像文件存在性的自动化校验：
     ```python
     fig0_path = os.path.join(SCRIPT_DIR, "figures", "fig0_v1_baseline_defects.png")
     assert os.path.isfile(fig0_path), f"Figure 0 image not found at {fig0_path}"
     ```

---

## 4. 独立敏感性分析报告 `MOC_V2_Simulation_Sensitivity_Report.md` 架构蓝图 (R4)

### 4.1 报告基本定位与学术标准
- **报告文件名**: `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md`
- **对应任务**: R4（现场级多裂缝参数敏感性学术分析报告）
- **定位**: 面向计算流体力学、石油工程与声学地球物理顶级期刊（如 *SPE Journal*, *Journal of Fluid Mechanics*, *Geophysical Journal International*）的学术级长篇研究报告。
- **排版与表述规范**:
  - 全文采用严谨的学术 Markdown 格式；
  - 所有物理量首次出现必须给出明确符号定义与标准 SI 单位（如阻抗 $\mathrm{Pa\cdot s/m^3}$、顺应性 $\mathrm{m^2}$、节流流阻 $\mathrm{s^2/m^5}$、滤失系数 $\mathrm{m^{2.5}/s}$）；
  - 嵌入全部 7 张独立复合图版（`sensitivity_figures/fig1_...png` 至 `fig7_...png`），并配齐中英双语详细图注（Figure Captions）；
  - 倒谱图严格使用 Rainbow 色阶，标注真实深度线，无任何文本检出率遮挡。

### 4.2 详细章节架构大纲设计

```markdown
# MOC_V2 现场工况多裂缝水锤正演响应与参数敏感性学术研究报告
## Transient Waveform Dynamics, Acoustic Illumination, and Parametric Sensitivity in Multi-Cluster Fractured Wellbores (MOC_V2 Field Study)

---

### 执行摘要 (Executive Summary)
- 现场级水平井水锤仿真背景与核心科学问题
- Base Case 基准工况设计要点（5000m 井长，3 簇，10m 间距）
- 7 大专题敏感性扫描的核心发现总括（相变窗口、时滞规律、混响抑制、强滤失泄压、马鞍型分流失衡）
- 对后续水击反演成像与智能解释的工程启示

---

### 第一章：引言与现场工况尺度定义 (Introduction & Field Framework)
- 1.1 现场多段多簇水力压裂井筒水锤监测工程背景
- 1.2 声阻抗突变、微间距混响与声波照亮的基本挑战
- 1.3 MOC_V2 物理求解器在现场尺度下的数值实现范式
- 1.4 本报告研究框架与 7 大专题研究路线

---

### 第二章：现场基准工况 (Base Case) 定义与水动力-声学基线
- 2.1 现场几何与流体物性基准参数表
  * 井筒几何：全长 $L = 5000\,\mathrm{m}$，内径 $D = 0.1397\,\mathrm{m}$（5.5 寸套管），管壁粗糙度 $\epsilon = 0.045\,\mathrm{mm}$；
  * 声学与流体：波速 $a = 1450\,\mathrm{m/s}$，密度 $\rho = 1000\,\mathrm{kg/m^3}$，动力粘度 $\mu = 0.001\,\mathrm{Pa\cdot s}$；
  * 初始基态：注入流速 $V_0 = 1.0\,\mathrm{m/s}$（排量 $Q_0 = 0.01533\,\mathrm{m^3/s} \approx 0.92\,\mathrm{m^3/min}$），井口基准水头 $H_0 = 300\,\mathrm{m}$，远场孔隙水头 $H_{ext} = 100\,\mathrm{m}$；
  * 裂缝布局：3 簇裂缝位于 $[4500, 4510, 4520]\,\mathrm{m}$（跟端第一簇位于 $4500\,\mathrm{m}$，簇间距 $d = 10\,\mathrm{m}$，末端至趾端封闭盲端死水区长达 $480\,\mathrm{m}$）；
  * 裂缝与射孔物性：基准顺应性 $C_f = 0.01\,\mathrm{m^2}$，滤失系数 $k_{leak} = 1.0 \times 10^{-4}\,\mathrm{m^{2.5}/s}$，射孔流阻 $K_p = 5.43 \times 10^5\,\mathrm{s^2/m^5}$（对应孔数 $N_p = 6$），关泵历时 $t_c = 1.0\,\mathrm{s}$（平滑余弦关泵）。
- 2.2 基态自洽场与死水区特征线相容性
- 2.3 Base Case 压力水头时程演化与 1D/2D 倒谱基准基线

---

### 第三章：专题 1——裂缝数量敏感性与多簇声能衰减规律 (Topic 1: Fracture Count)
- 3.1 实验工况设计：裂缝簇数 $N_c \in [1, 2, 3, 4, 5, 6, 7, 8]$（起点 4500m，间距 10m）
- 3.2 水动力学时程响应对比（宏观反弹与分流阶梯）
- 3.3 声波穿透耗散机制：多级串联并联阻抗网络的能量逐级分流
- 3.4 1D 倒谱识别深度与深层裂缝“声学阴影”衰减极限
- 3.5 Rainbow 2D 连续倒谱云图特征分析（簇间串扰与时空能量扩散）
- 3.6 嵌入 Figure 1（300 DPI PNG / SVG）与学术级双语图注

---

### 第四章：专题 2——密集裂缝簇间距敏感性与微间距声学混响 (Topic 2: Fracture Spacing)
- 4.1 实验工况设计：3 簇裂缝，间距 $d \in [5, 10, 15, 20, 25, 30, 50, 80]\,\mathrm{m}$（起点 4500m）
- 4.2 腔体共振物理机制：极密裂缝间的高频往复混响特征频率 $f_{rev} = a / (2d)$
- 4.3 空间分辨率退化与倒谱主峰混叠极限分析（$d < 10\,\mathrm{m}$ 时 Rayleigh 判据极限）
- 4.4 宽间距（$d \ge 50\,\mathrm{m}$）独立脉冲解耦与窄间距（$d \le 10\,\mathrm{m}$）干涉相长/相消
- 4.5 Rainbow 2D 倒谱云图中的空间干涉纹样解析
- 4.6 嵌入 Figure 2（300 DPI PNG / SVG）与学术级双语图注

---

### 第五章：专题 3——裂缝弹性顺应性与地质储能反哺效应 (Topic 3: Fracture Compliance)
- 5.1 实验工况设计：顺应性扫描 $C_f \in [0.002, 0.005, 0.010, 0.020, 0.030]\,\mathrm{m^2}$
- 5.2 宏观开端大反弹幅值 $\Delta H_{reb}$ 与 $C_f$ 的非线性标度律
- 5.3 缝内高压流体回吐速率与水头衰减时间常数 $\tau = R_{eq} C_f$
- 5.4 1D 倒谱峰值深度、半高宽与裂缝体积储能的定量映射关系
- 5.5 Rainbow 2D 倒谱云图中的能量滞留效应
- 5.6 嵌入 Figure 3（300 DPI PNG / SVG）与学术级双语图注

---

### 第六章：专题 4——拟达西滤失系数与地层压降退水动力学 (Topic 4: Leakoff Coefficient)
- 6.1 实验工况设计：$k_{leak} \in [0.2, 0.6, 1.0, 3.0, 10.0] \times 10^{-4}\,\mathrm{m^{2.5}/s}$
- 6.2 致密储层保压形态 vs Type V 断层强滤失退水过程对比
- 6.3 压力水头向地层孔隙水头 $H_{ext} = 100\,\mathrm{m}$ 跌落的非稳态泄压机理
- 6.4 滤失通量对声阻抗界面的虚部扰动与倒谱幅值阻尼衰减
- 6.5 Rainbow 2D 倒谱云图中的能量快速衰亡表征
- 6.6 嵌入 Figure 4（300 DPI PNG / SVG）与学术级双语图注

---

### 第七章：专题 5——限流射孔流阻与声学扼流照亮窗口 (Topic 5: Perforation Resistance)
- 7.1 实验工况设计：$K_p \in [1.5, 3.5, 5.43, 10.0, 25.0] \times 10^5\,\mathrm{s^2/m^5}$（孔数 $16, 8, 6, 4, 2$）
- 7.2 水动力学节流压降与能量反弹的双向制约
- 7.3 声学扼流圈物理机理：透射系数 $T$ 与反射系数 $\Gamma$ 的 Pareto 权衡
- 7.4 欠节流短路屏蔽（孔数 $\ge 8$）与过节流强端面反射（孔数 $\le 4$）的双重失效机理
- 7.5 限流最佳工作窗口（$N_p = 6$）现场设计判据
- 7.6 嵌入 Figure 5（300 DPI PNG / SVG）与学术级双语图注

---

### 第八章：专题 6——关泵斜坡动力学历时与高频滤波效应 (Topic 6: Pump Shutdown Ramp)
- 8.1 实验工况设计：关泵历时 $t_c \in [0.0, 0.5, 1.0, 1.5, 2.0]\,\mathrm{s}$（含瞬态阶跃）
- 8.2 波前最大时间梯度 $|dH/dt|_{max}$ 对数衰减规律与吉布斯振荡抑制
- 8.3 半幅反弹时滞严格线性规律：$\Delta t_{half} \approx 0.64 t_c$ 的现场校准意义
- 8.4 $\mathrm{sinc}^2(\pi f t_c)$ 频谱滚降低通滤波对密集裂缝混响伪峰的压制机理
- 8.5 Rainbow 2D 倒谱云图中的波前锐度与空间拖尾对比
- 8.6 嵌入 Figure 6（300 DPI PNG / SVG）与学术级双语图注

---

### 第九章：专题 7——多簇进液能力非均匀性与压裂地质力学组合 (Topic 7: Multi-Cluster Combinations)
- 9.1 现场 5 大典型进液组合工况设计：
  * Case 1【中，中，中】：基准均匀进液（$w = [1.0, 1.0, 1.0]$）；
  * Case 2【高，中，中】：跟部首簇突进优势型（$w = [1.8, 1.0, 1.0]$）；
  * Case 3【高，中，高】：两头优势马鞍型（中间受强应力阴影挤压，$w = [1.5, 0.5, 1.5]$）；
  * Case 4【中，中，高】：趾端逆向优势型（$w = [0.8, 1.0, 1.8]$）；
  * Case 5【死，中，高】：首簇砂堵死簇型（$w = [0.0, 1.2, 1.8]$）。
- 9.2 多簇非均匀分流下的稳态初始水头剖面与动量差分
- 9.3 5 类工况时域水头波形的特征分离与诊断标志
- 9.4 倒谱空间能量分布对多簇进液能力不均衡的映射指纹
- 9.5 Case 5 砂堵工况下“首簇声学盲区破除后深部剧烈反弹”物理实证
- 9.6 嵌入 Figure 7（300 DPI PNG / SVG）与学术级双语图注

---

### 第十章：全要素参数敏感性横向综合与灵敏度层级矩阵
- 10.1 7 大专题核心参数对水锤响应特征的综合敏感度排序（Sobol 指数分析/定性评级）
- 10.2 参数灵敏度矩阵表（水头跌落、大反弹幅值、衰减斜率、倒谱信噪比、检出分辨率）
- 10.3 参数之间的正交解耦特性论证（时滞解耦、幅值解耦、频域解耦）

---

### 第十一章：面向深度学习与智能反演数据集构建的工程指导
- 11.1 正向特征空间的非线性奇异区识别与避免（如 $N_p \ge 12$ 短路区）
- 11.2 倒谱空间高维特征向量提取与数据预处理（一阶导数增强与距离轴切片）
- 11.3 训练集拉丁超立方（LHS）采样的边界推荐取值表
- 11.4 物理约束神经网络（PINN）与水击反演网络损失函数设计建议

---

### 第十二章：结论 (Conclusions)
- 12.1 现场级 MOC_V2 仿真与敏感性研究核心结论
- 12.2 水锤波形态解释与压裂多簇评价实践建议

---

### 附录 A：全书符号物理释义与国际标准单位 (Nomenclature & SI Units)
- 完整的参数、变量、无量纲数对照表。

### 附录 B：数值求解收敛性与质量守恒自检矩阵
- 38+ 算例全量收敛性、最大残差与质量连续性检验汇总。
```

---

## 5. 7 大专题图版与可视化规范 (Figure 1~7 & Rainbow 2D 倒谱)

### 5.1 图版存储与文件命名
所有图版必须由后续可视化脚本（建议为 `run_sensitivity_study.py` 或配套绘图模块）输出至指定目录：
- 目标目录: `docs/moc_v2_technical_report/sensitivity_figures/`
- 文件命名列表（共 14 个文件：7 个 PNG + 7 个 SVG）：
  1. `fig1_fracture_count_sensitivity.png` / `.svg`
  2. `fig2_fracture_spacing_sensitivity.png` / `.svg`
  3. `fig3_fracture_compliance_sensitivity.png` / `.svg`
  4. `fig4_leakoff_sensitivity.png` / `.svg`
  5. `fig5_perforation_resistance_sensitivity.png` / `.svg`
  6. `fig6_ramp_closure_sensitivity.png` / `.svg`
  7. `fig7_intake_capacity_combinations.png` / `.svg`

### 5.2 统一图版复合结构（4-Panel Layout）
每张专题图版必须保持统一的学术出版标准（双栏宽度 180 mm，300 DPI）：
- **Panel a（时域宏观与波前局部）**:
  - 全时程 100s 压力水头演化，叠加关泵初期微观波列放大特写；
  - 清晰标注稳态基线 $H_0 = 300\,\mathrm{m}$ 与地层孔隙水头 $H_{ext} = 100\,\mathrm{m}$；
- **Panel b（1D 实倒谱曲线族）**:
  - 沿井深轴（$4400\sim 4600\,\mathrm{m}$）展开的 1D 实倒谱曲线簇；
  - **关键标注**：使用垂直红色虚线精确标出各簇裂缝的真实位置（如 $4500, 4510, 4520\,\mathrm{m}$）；
- **Panel c & d（2D 连续倒谱时空云图 Cepstrogram）**:
  - 横轴为时间（$t \in [0, 100]\,\mathrm{s}$），纵轴为重构井深（$z \in [4400, 4600]\,\mathrm{m}$）；
  - **严格色阶要求**：必须采用 **Rainbow 色阶**（`cmap='rainbow'`，或等效平滑高对比色阶），展现鲜明的彩虹色能量带；
  - **真实裂缝深度标线**：在云图纵轴上以细虚线（如白色或浅灰色虚线）标定真实裂缝深度；
  - **严格排除项**：**严禁在云图上覆盖“检出率 100%”、“Peak Found”等自动化检测判定文本**，保持学术图版的纯净与高保真物理美感。

---

## 6. 各工单 (R1~R4) 落地实施路线与接口契约

为了让后续实施阶段的 Worker 能够无缝开展工作，梳理各工单的具体接口与协作流程如下：

### 6.1 Worker M1 实施要点（R1 理论主报告精简与构建校验）
- **操作对象**:
  1. `docs/moc_v2_technical_report/build_report.py`
  2. `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md`
- **执行步骤**:
  1. 备份现有文件（防止误删）；
  2. 修改 `build_report.py` 中的 `REPORT_CONTENT`，删除原第 3~7 章，更新标题、执行摘要与目录，在 2.8 节后添加精炼的理论总结与新报告导引；
  3. 修改 `build_report.py` 底部的断言阈值（保持 LaTeX 平衡与 62 个理论关键词不变，长度断言适配 45,000+ 字符）；
  4. 运行 `python docs/moc_v2_technical_report/build_report.py`，确保其自动重写 `MOC_V2_Physics_Upgrade_Report.md` 并输出 `Validation PASSED`；
  5. 检查 `git diff`，确认未损坏第 1 章与第 2 章的核心公式。

### 6.2 Worker M2 实施要点（R2 现场级仿真流水线开发）
- **操作对象**: `docs/moc_v2_technical_report/run_sensitivity_study.py`
- **调用内核**: `moc_simulate.v2.core.solver.simulate_v2` 与配置体系 `moc_simulate.v2.configs`
- **执行步骤**:
  1. 建立 Base Case 标准工况字典；
  2. 循环执行 7 大专题的所有 38+ 组算例（$8 + 8 + 5 + 5 + 5 + 5 + 5 = 41$ 仿真）；
  3. 收集并验证所有时程数据（$H_{wh}(t), V_{wh}(t)$），计算 1D 与 2D 倒谱，确保 100% 收敛无 NaN/Inf；
  4. 导出时程数据与汇总指标为后续绘图与报告引用提供支持。

### 6.3 Worker M3 实施要点（R3 Nature 级图版生成）
- **操作对象**: 绘图代码嵌入于 `run_sensitivity_study.py` 或独立绘图脚本
- **输出目标**: `docs/moc_v2_technical_report/sensitivity_figures/fig1` 至 `fig7`（PNG + SVG，共 14 个文件）
- **核心标准**: Rainbow 色阶 2D 倒谱云图，真实裂缝标线，无文本判据，300 DPI。

### 6.4 Worker M4 实施要点（R4 学术分析报告撰写）
- **操作对象**: `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md`
- **执行步骤**:
  1. 按照本调研报告第 4 节的 12 章完整架构编写全文；
  2. 嵌入 `sensitivity_figures/` 下的 7 张图版，编写详实的中英双语图注；
  3. 深入剖析 7 大专题的物理力学机理，整合各专题参数指标对比表；
  4. 严格校验 Markdown 排版与 SI 单位一致性。

### 6.5 全库回归测试（R5 回归保障）
- 在任何修改完成后，必须执行 `pytest`，确保全库原有 99 项单元与集成测试 100% 通过，无任何回退。

---

*调研完成日期*: 2026-09-11  
*报告提交路径*: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_2_o3\survey_reports.md`
