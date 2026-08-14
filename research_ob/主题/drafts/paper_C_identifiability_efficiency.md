---
type: paper-draft
paper_id: Paper-C
status: frozen
topics:
  - T01
  - T05
  - T07
canonical: true
freeze_date: 2026-08-12
---

# 可辨识性、估计器效率与误差预算：多裂缝水击反演为何失败

**English working title:** Identifiability, Estimator Efficiency, and Error Budgets: Why Multi-Fracture Water-Hammer Inversion Fails

> **2026-08-12 frozen**：不再作为活跃工程驱动；结案摘要见 [[PaperC_MOC主路径结案]]。
> 主张链由 [[EXP-20260806-002-Fisher信息与CRB可辨识性分析]] 与 [[EXP-20260806-003-估计器效率验证]] 等支撑；Paper B 的 38.36 m「分辨率」叙事降级为某一特定倒谱变换的旁瓣宽度，不是信息论极限。
>
> **SPE Journal 框架正文：** [`docs/PaperC_Identifiability_and_Error_Budgets_可辨识性与误差预算.md`](../../../docs/PaperC_Identifiability_and_Error_Budgets_可辨识性与误差预算.md)。Vault 转发页 [[SPEJ_三篇论文框架]]。本篇不新做实验。

---

## 0. Abstract

### 0.1 中文

多裂缝水击诊断中，倒谱、稀疏反卷积、字典剥离与学习式反演在近距工况上普遍失败。本文用估计器无关的 Fisher 信息 / Cramér-Rao 界（CRB）与全波形极大似然估计器（MLE）区分三种互斥假说：信息受限、偏差受限、估计器低效。在对齐物理配置（L=5000 m，a=1450 m/s，brunone，tf=50 s，fc=20 Hz）下：SNR=40 dB 时单缝位置 CRB 约 0.025 m；全波形 MLE 效率比 RMSE/CRB = 0.96–1.39（n≤5 时仍约 0.86–1.18），贴到界；倒谱即使 SNR=∞ 仍有约 12 m 系统误差。真正的实用墙是模型误差：摩阻形式失配（brunone↔steady）约 30 m；−1% 波速的非线性 MLE 偏差中位约 120 m（一阶 TOF 预测 28–36 m 仅作量级）。配对失配剪刀差成立：倒谱在 steady 上变好，P0/PhaseNet 变差。结论：合成正确模型上失败主因是估计器低效；失配下由偏差主导。处方是以 **MOC 全波形 MLE 为主估计器**（可联合讨厌参数），倒谱/学习式检测与 FNO 仅作对照或粗初值辅助；并在失配集上评估。缝数判定是模型选择问题，CRB 不对其构成约束。

### 0.2 English

Near-well multi-fracture localization from single-channel water-hammer records routinely fails across cepstrum, sparse deconvolution, dictionary peeling, and learning-based detectors. We separate three mutually exclusive hypotheses—information limit, bias limit, and estimator inefficiency—using the Fisher information / Cramér–Rao bound (CRB) and a full-waveform maximum-likelihood estimator (MLE). Under an aligned MOC setting (L=5000 m, a=1450 m/s, Brunone friction, tf=50 s, fc=20 Hz), the single-fracture depth CRB at SNR=40 dB is ≈0.025 m, and the MLE efficiency ratio RMSE/CRB lies in 0.96–1.39 (still ≈0.86–1.18 for n≤5). Cepstrum retains ≈12 m systematic error even at SNR=∞. The practical wall is model error: friction-form mismatch (brunone↔steady) yields ≈30 m depth bias; a −1% wavespeed mismatch produces a nonlinear MLE median |bias|≈120 m (first-order TOF predictions of 28–36 m are order-of-magnitude only). A paired transfer scissors test holds: cepstrum improves on steady data while P0 and PhaseNet degrade. Thus synthetic failures under a correct model are estimator inefficiency; field-like failures are bias-dominated. The prescription is **MOC full-waveform MLE as the primary estimator** (optionally with nuisance parameters), with cepstrum/learned detectors and FNO used only as baselines or coarse initializers, evaluated on mismatch sets. Fracture-count selection is a model-selection problem and is not constrained by the position CRB.

---

## 1. Introduction

### 1.1 现象

项目已积累大量互相独立的方法（倒谱、CWT、L1/BSD 稀疏反卷积、字典剥离、MUSIC、LISTA、DCCDM、PhaseNet），它们在 5–20 m 近距工况上全部失败或仅达 F1≈0.7。方法族差异极大而失败位置一致，提示瓶颈可能不在算法侧。

### 1.2 三种互斥假说

1. **信息受限**：数据里没有分辨该间距所需的信息。
2. **偏差受限**：信息充足，但正演模型错了（波速、摩阻），估计被系统性拉偏。
3. **估计器低效 / 优化受限**：信息充足、模型正确，但所用估计器离有效界甚远，或似然面多峰。

Paper B 将倒谱谱支撑宽度 \(\Delta d_{\mathrm{DR}}=38.36\,\mathrm{m}\)（DR=80 dB）表述为操作分辨率基线。该量描述的是某一特定变换的旁瓣宽度，不是信息论极限；本文将它降级为估计器低效的对照。

### 1.3 两段式结论预告

- **合成正确模型**：瓶颈是估计器——全波形 MLE 可达 CRB。
- **模型失配 / 现场**：瓶颈是偏差——形式误差约 30 m，不能被单一标定参数完全吸收。

### 1.4 贡献边界

本文不宣称新的现场定位算法已达厘米级；宣称的是：在给定观测噪声与传感器带宽下，任何无偏估计器的位置方差下界，以及现有启发式方法相对该下界的效率差距。

**方法角色（项目锁定）**：CRB 级精修以 MOC 全波形为准；FNO 仅辅助（快速波形 / 粗初值），不宣称可微代理反演过关。

---

## 2. Forward model and observation operator

### 2.1 正演

井筒水击用特征线法（MOC）求解，Courant=1。裂缝位置吸附到网格 \(dx=a\,dt\approx 1.45\,\mathrm{m}\)。摩阻取 Brunone 非定常模型（对照实验用 steady）。趾端为定压储层边界，观测窗 \(t_f=50\,\mathrm{s}\) 覆盖多次井筒往返。

### 2.2 观测算子

\[
y_k=s_k(\theta)+n_k,\quad n_k\sim\mathcal N(0,\sigma^2)\ \text{i.i.d.}
\]

\(s(\theta)\) 为停泵后井口 \(H_{wh}\) 经零相位 4 阶 Butterworth 限带（截止 \(f_c\)）后的采样。\(\sigma\) 由 SNR 反推，定义与分层基准一致。默认 \(f_c=20\,\mathrm{Hz}\)（差分收敛域内：\(f_c\cdot dt\lesssim 0.021\)）。

### 2.3 参数向量

\[
\theta=(x_0,\ldots,x_{n-1},\log_{10}C_{f,i},\log_{10}k_{\mathrm{leak},i},a,\log_{10}k_{\mathrm{br}})
\]

A 档估计：仅 \(x\)（其余已知）。B 档：\(x+C_f+k_{\mathrm{leak}}+a\)。

---

## 3. Fisher information and Cramér-Rao bound

### 3.1 定义

\[
\mathrm{FIM}=\frac{J^\top J}{\sigma^2},\qquad
\mathrm{Cov}(\hat\theta)\succeq\mathrm{FIM}^{-1}.
\]

Jacobian 用中心差分；位置步长取整数格点，波速扰动取 \(|\delta a/a|=1/N\) 并扣除网格吸附漂移。

### 3.2 关键观察（EXP-002）

| 观察 | 数值 / 结论 | 证据等级 | 边界 |
|---|---|---|---|
| 近距 CRB 不恶化 | 5–40 m 间距上 \(\sigma_x\) 仍为厘米级 | 实测 | fc=20 Hz，SNR 扫描，tf=20 s 早期扫描 |
| SNR=40 dB、fc=20 Hz 单缝 | \(\sigma_x\approx 0.03\,\mathrm{m}\)（对齐配置下 0.025 m） | 实测 | 缝数已知 |
| 波速讨厌参数（短记录） | 绝对深度惩罚 ×1.2–1.8；间距惩罚 ≈×1.00–1.02 | 实测 | tf=20 s |
| 似然面真值邻域 | profile/grid 扫描显示单峰 | 实测 | 双缝、已知阶数 |
| 高带宽标度 | \(\sigma_x\propto f_c^{-0.358}\)（收敛域） | 实测 | dt=2e-4，fc≤100 Hz |

**主张 1（无信息墙）**：在本文观测模型下，5–40 m 间距不存在信息论分辨墙。证据：[[EXP-20260806-002]]、[[EXP-20260806-003]]；图见 `output/analysis/identifiability/`。

### 3.3 观测窗与趾端锚（EXP-003 细化）

在 tf=50 s、含趾端 2L/a 反射时，B 档 CRB 的绝对深度波速惩罚降至 ≈×1.00–1.01。短记录下「报间距不报深度」仍是正确处方；长记录下联合估 \(a\) 代价很小。

---

## 4. Estimator efficiency

### 4.1 实验设计

场景：`single_4000`、`dual_10m`、`dual_40m`（与 P0 字典物理配置对齐）。同一批含噪数据上对照：全波形 MLE（网格 + 一步 Gauss-Newton）、P0 字典剥离、倒谱。n_mc=30，SNR∈{∞,40,30,20} dB。

### 4.2 效率比

| 场景 | SNR | MLE RMSE / m | CRB / m | 效率比 | 倒谱 / m | P0 / m |
|---|---:|---:|---:|---:|---:|---:|
| single_4000 | ∞ | 0 | — | — | 12.0 | 0.12 |
| single_4000 | 40 | 0.029 | 0.025 | **1.14** | 12.0 | 0.12 |
| single_4000 | 30 | 0.086 | 0.080 | **1.07** | 294 | 0.13 |
| single_4000 | 20 | 0.354 | 0.255 | **1.39** | 473 | 0.20 |
| dual_10m | 40 | 0.039 | 0.031 | **1.26** | 309 | 353 |
| dual_10m | 30 | 0.093 | 0.097 | **0.96** | 290 | 7.0 |
| dual_40m | 40 | 0.034 | 0.032 | **1.04** | 12.8 | 24.3 |
| dual_40m | 20 | 0.326 | 0.323 | **1.01** | 404 | 21.0 |

**主张 2（界可达）**：全波形 MLE 效率比 0.96–1.39。证据等级：实测。路径：`output/analysis/identifiability/efficiency/main/`；图 `fig1_rmse_vs_snr.png`、`fig2_efficiency_ratio.png`、`fig4_gap_orders.png`。

**主张 3（倒谱系统偏差）**：SNR=∞ 时倒谱仍有约 12 m 误差，而正确模板的 P0 单缝仅 0.12 m。证据等级：实测。

### 4.3 缝数已知假设的作用域

CRB 与上述效率比均在**给定模型阶数 \(n\)** 下定义。它们回答的是：若已知有 \(n\) 条缝，位置能估多准。它们**不回答**：数据支持几个缝。

| 问题类型 | 统计框架 | 本文状态 |
|---|---|---|
| 位置 / 物性估计 | 参数估计 + CRB | EXP-002/003/001(20260807) 已闭环（n≤5） |
| 缝数判定 | 模型选择 / 检测 | 嵌套 BIC exact-count=1.0（EXP-20260807-001）；PhaseNet 盲计数 30.5% 为不同问题 |
| 阶数错误下的位置偏差 | 失配估计 | 待补 |

误读风险：把「无信息墙」说成「缝数也可无损恢复」。正文禁止该表述。PhaseNet 在 close2000 子集上 exact-count≈30.5%，已提示计数远难于定位。

### 4.4 n≥3 外推（已补 → EXP-20260807-001）

n=3/4/5（等间距 10/40 m）一步 GN 效率比 **0.86–1.18**，门控 ≤3 全过。坐标下降与穷举网格在 dual 上位置差 = 0。证据等级：**实测**。路径：`output/analysis/identifiability/highorder/main/`。

---

## 5. Model misspecification budget

### 5.1 形式误差（实测）

| 真模型 → 拟合 | RMSE / m | 一阶预测 bias | absorbed_frac | k_scale 吸收后 |
|---|---:|---:|---:|---:|
| brunone → steady | **29.6** | 0.34 | 0.010 | — |
| steady → brunone | **31.9** | −2.49 | 0.033 | 24.7（k̂=0.72） |
| brunone → brunone | 0 | 0 | 0 | 0 |

一阶线性化严重低估形式误差（残差几乎正交于 \(\partial s/\partial x\)）。`k_scale` 只能部分吸收。

**主张 4（实用墙是模型误差）**：摩阻形式失配 → ~30 m 偏差 ≫ CRB。证据等级：实测（单缝 SNR=∞）。路径：`output/analysis/identifiability/mismatch_mle/main/`。

### 5.2 参数型讨厌误差（非线性复核）

1% 波速误差的一阶预测深度偏差 28–36 m（EXP-002）。非线性全波形 MLE 复核（a_data=1435.5，a_model=1450）给出单缝中位 |bias|≈**120 m**（比值 3–4），门控 ratio<2 **未通过**（[[EXP-20260807-003-波速失配偏差复核]]）。Courant=1 改 a 触发网格重划分，一阶连续 TOF 公式仅作**量级估计**。这加强「实用墙是模型误差」：真实偏差可能比一阶更大。

### 5.3 差分迁移预测（已证实）

配对集 `lhs_dataset_2000` ↔ `lhs_dataset_2000_steady`（2000/2000 几何一致）。剪刀差成立（[[EXP-20260807-002-失配可迁移性剪刀差]]，n=300 + PhaseNet test=300）：

| 方法 | brunone F1 | steady F1 | ΔF1 |
|---|---:|---:|---:|
| 倒谱 | 0.232 | 0.642 | **+0.410** |
| P0 | 0.396 | 0.089 | **−0.307** |
| PhaseNet | 0.746 | 0.232 | **−0.514** |

阈值重校准（0.35→0.20）几乎不挽回 PhaseNet。控制变量是隐含正演模型与数据生成模型的距离。

---

## 6. Implications for learning-based inversion

### 6.1 损失设计

热图 / 峰匹配损失不对应观测似然；全波形 \(\|y-s_\theta\|^2\)（或可微代理）才与 CRB 同轴。PhaseNet 的 F1≈0.75 与 MLE 厘米级之间的鸿沟，首先是目标函数选择，其次才是网络容量。

### 6.2 评价协议

现有 close2000 基准波速误差为零、摩阻与训练一致，**高估可迁移性**。必须报告失配测试集（至少 steady 配对；最好含 ±1% 波速）。

### 6.3 代理模型门槛（分层；MOC 主 / FNO 辅）

| 门槛 | 结果 | 含义 |
|---|---|---|
| 波形 rel L2（主域重训后） | ≈0.029 PASS（[[EXP-20260808-001]]） | FNO 可作快速波形代理 |
| 纯 FNO-MLE vs MOC-MLE | single≈2.7 m（~105×CRB）；dual≈72 m **FAIL**（[[EXP-20260808-002]]） | **不可**宣称可微全波形定位 |
| FNO 初值 + MOC 精修 | 单缝 2.6→0.007 m PASS；双缝需 R≈40 格才拉回（[[EXP-20260809-001]]） | FNO 仅粗初值；精修归 MOC |

旧权重在 2000 域 rel L2≈0.32（[[EXP-20260807-004]]）已由同域重训取代。项目策略锁定：**位置估计以 MOC 全波形 MLE 为主**；FNO / 倒谱 / 学习式检测为辅助或对照。

---

## 7. Conclusions

1. 近距失败不是信息论墙：CRB 为厘米级（fc=20 Hz，SNR=40 dB）。
2. 全波形 **MOC-MLE** 可达界（效率比 0.96–1.39）；倒谱 / 近距 P0 离界 2–3 个数量级。
3. 实用墙是模型失配：形式误差 ~30 m；参数误差可部分被讨厌参数吸收，形式误差大多不可。
4. 缝数判定独立于位置 CRB；嵌套 BIC 在理想构型下可辨，盲计数（PhaseNet 30.5%）是另一问题。
5. 配对摩阻失配剪刀差成立：倒谱变好、P0/PhaseNet 变差——现有基准高估可迁移性。
6. **主路径是 MOC 全波形（可联合讨厌参数）**；FNO 波形可用但反演不过关，仅可作粗初值；学习方法若沿用须对齐全波形似然并在失配集验收；一阶波速偏差公式仅量级参考（非线性可大 3–4×）。

---

## 证据总表

| ID | 论点 | 支撑实验 | 图表 / 路径 | 证据等级 | 适用边界 |
|---|---|---|---|---|---|
| C1 | 无信息墙，σ_x≈0.03 m | EXP-002/003 | efficiency figures；CRB sweep | 实测 | n≤2，fc≤20 Hz（dt=1 ms），缝数已知 |
| C2 | MLE 效率比 0.96–1.39 | EXP-003 | fig1, fig2 | 实测 | 同上；一步 GN |
| C3 | 倒谱 @∞ 有 12 m 系统误差 | EXP-003 | fig4 | 实测 | 对齐配置倒谱管线 |
| C4 | P0 近距双缝不稳定 | EXP-003 负结果 | — | 实测 | dual_10m |
| C5 | 摩阻形式失配 ~30 m | EXP-003 | mismatch_mle | 实测 | 单缝，SNR=∞ |
| C6 | σ_x ∝ fc^−0.358 | EXP-003 | bandwidth/hifreq | 实测 | 收敛域 fc≤100 Hz |
| C7 | 长记录下深度与间距几乎同等可辨 | EXP-003 | fig3 | 实测 | tf=50 + 趾端锚 |
| C8 | 短记录中间距是波速不变量 | EXP-002 | wavespeed valley | 实测 | tf=20 s |
| C9 | 1% 波速 → 一阶 28–36 m；非线性 ~120 m | EXP-002；EXP-20260807-003 | wavespeed_mismatch | 实测（非线性）+ 推断（一阶） | Courant=1；6/8 探针 |
| C10 | n≥3 仍可达界 | EXP-20260807-001 | highorder/main | 实测 | n≤5，等间距簇 |
| C11 | 缝数 BIC 可选（嵌套） | EXP-20260807-001 | order_selection/main | 实测（嵌套oracle） | 非盲搜；对 n_eff 不敏感 |
| C12 | 配对失配剪刀差 | EXP-20260807-002 | transfer/main | 实测 | 300 classical + PN test |
| C13 | 似然面单峰支持坐标下降 | EXP-002；coord_validate | gate JSON | 实测 | dual 门控通过 |
| C14 | FNO：波形可用、反演不可、混合辅 | EXP-0801/0802/0901 | fno_* / hybrid | 实测 | 主域 close2000；精修归 MOC |

推断档条目：C9 的一阶部分、正文中对 PhaseNet 目标函数的机制解释。不超过 5 条。

---

## 附录 A：证据缺口清单（更新后）

| 缺口 | 阻断哪条主张 | 补证成本 | 本阶段？ |
|---|---|---|---|
| n=3/4/5 效率比 | C10 | 已完成 | **已做** |
| 缝数未知 / BIC | C11 | 已完成（嵌套oracle） | **已做**；盲搜仍缺 |
| 配对摩阻掉点剪刀差 | C12 | 已完成 | **已做** |
| ±1% 波速非线性 | C9 | 探针完成 | **已做**（门控未过→限定公式） |
| 盲坐标下降阶数选择 | C11 外推 | 中高 | 否（limitation） |
| FNO 分层门槛 | §6.3 | EXP-004/0801/0802/0901：波形 PASS、反演 FAIL、混合有条件 | **已做**；策略=MOC 主 |
| 半真实 / 现场噪声 | 外部效度 | 高 | **否**（limitation） |
| n>5 或连续缝网 | 极端拓扑 | 高 | **否** |

---

## 附录 B：Data / Code availability

| 图 / 表 | 复现命令 | 输出 |
|---|---|---|
| Fig1–4 效率 | `python -m analysis.identifiability.run_efficiency --tag main ...`；`plot_efficiency --tag main` | `output/analysis/identifiability/efficiency/main/figures/` |
| 失配表 | `python -m analysis.identifiability.run_mismatch_mle --tag main ...` | `.../mismatch_mle/main/` |
| 带宽标度 | `python -m analysis.identifiability.run_bandwidth_check --tag hifreq ...` | `.../bandwidth/hifreq/` |
| CRB 扫描 | `python -m analysis.identifiability.run_crb_sweep --tag main_dt1ms ...` | EXP-002 输出树 |
| 高阶效率 | `python -m analysis.identifiability.run_highorder_efficiency --tag main` | `.../highorder/main/figures/eff_vs_n.png` |
| 阶数 BIC | `python -m analysis.identifiability.run_order_selection --tag main` | `.../order_selection/main/figures/exact_count_vs_snr.png` |
| 迁移剪刀差 | `python -m analysis.identifiability.run_transfer_check --tag main --max-cases 300` | `.../transfer/main/transfer_summary.json` |
| 波速探针 | `python -m analysis.identifiability.run_wavespeed_mismatch --tag probe8` | `.../wavespeed_mismatch/probe8/` |
| 波速/k CLI | `python moc_simulate/run_lhs_batch_simulate.py --wavespeed 1435.5 --seed 42 ...` | 配对 LHS 集 |

---

## 附录 C：与 Paper B 的关系

Paper B（[`docs/PaperB_Brunone_Waveform_Degradation_非定常摩阻波形退化与倒谱模糊.md`](../../../docs/PaperB_Brunone_Waveform_Degradation_非定常摩阻波形退化与倒谱模糊.md)）写 Brunone 波形退化与倒谱拾取偏差，**不以** \(\Delta d_{\mathrm{DR}}=38.36\,\mathrm{m}\) 为操作分辨率基线。该量若出现，仅作为某一倒谱变换的旁瓣宽、估计器低效对照。信息论表述以本稿与 [[PaperC_MOC主路径结案]] 为准。

---

## 附录 D：Limitations（投稿前必须保留）

1. **缝数已知 vs 盲计数**：位置 CRB / MLE 效率在给定阶数下定义。嵌套 BIC exact-count=1.0 使用 oracle 构型，**不是**盲搜索；PhaseNet exact-count≈30.5% 才是盲计数现实基线。
2. **传感器与噪声**：主结果 fc=20 Hz、加性白噪声；半真实/现场噪声未测。
3. **波速一阶公式**：Courant=1 重网格下，−1% 波速的非线性偏差约 120 m，一阶 28–36 m 仅量级参考。
4. **几何范围**：效率实验为等间距簇、n≤5、缝区 3500–4800 m；非等间距 / 更大 n 未覆盖。
5. **F1@10 m**：容差小于形式失配偏差 ~30 m；失配叙事优先报告深度误差与配对 ΔF1。
6. **代理模型**：FNO 主域波形 PASS、纯反演 FAIL；混合精修依赖足够大的 MOC 吸引域。本稿**不宣称**可微代理定位；主估计器为 MOC 全波形（结案见 [[PaperC_MOC主路径结案]]）。

---

## 附录 E：图注清单（Figure inventory）

| 编号 | 文件 | 说明 |
|---|---|---|
| Fig.1 | `output/analysis/identifiability/efficiency/main/figures/fig1_rmse_vs_snr.png` | RMSE vs SNR：MLE / 倒谱 / P0 / CRB |
| Fig.2 | `.../fig2_efficiency_ratio.png` | 效率比 RMSE/CRB |
| Fig.3 | `.../fig3_depth_vs_spacing.png` | 绝对深度 vs 间距（B 档） |
| Fig.4 | `.../fig4_gap_orders.png` | 方法族数量级差距 |
| Fig.5 | `output/analysis/identifiability/figures/fig1_crb_vs_spacing.png` | CRB vs 间距（无信息墙） |
| Fig.6 | `.../fig3_wavespeed_penalty.png` | 波速讨厌参数惩罚 |
| Fig.7 | `.../fig5_misfit_profile.png` | 似然面单峰 |
| Fig.8 | `output/analysis/identifiability/highorder/main/figures/eff_vs_n.png` | n=3/4/5 效率比 |
| Fig.9 | `output/analysis/identifiability/order_selection/main/figures/exact_count_vs_snr.png` | 嵌套 BIC exact-count |
| Tab.1 | `transfer/main/transfer_summary.json`；`transfer/main/figures/fig_scissors_f1.png` | 配对剪刀差（倒谱/P0/PhaseNet） |
| Tab.2 | `mismatch_mle/main/` + `wavespeed_mismatch/probe8/` | 形式失配 ~30 m；波速非线性 ~120 m |
| Fig.10 | `fno_accuracy/close2000_retrain/` + `fno_mle_gate/main/` + `fno_moc_hybrid/` | FNO 分层：波形 PASS / 反演 FAIL / 混合有条件 PASS |
