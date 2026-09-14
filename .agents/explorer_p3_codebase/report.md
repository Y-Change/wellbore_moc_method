# PaperC 智能反演系统代码库深度剖析与 TG-DIS-DeepONet 架构设计集成方案

**项目名称**: 面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究  
**报告归属**: Explorer P3 Codebase (`.agents/explorer_p3_codebase/report.md`)  
**调查对象**: `PaperC_CJNO_Wellbore_Inversion/` 全库源码、消融实验、测试套件与历史基线  
**完成时间**: 2026-09-13  
**执行角色**: Codebase & Neural Operator Investigator  

---

## 摘要 (Executive Summary)

本报告针对 PaperC 项目第二阶段（TG-CJ-DeepONet）与第三阶段核心任务（波动方程可微逆散射层剥离算子与 TG-DIS-DeepONet 架构研发），开展了彻底的代码库源码审计、实验指标归因、物理瓶颈分析以及全新可微神经算子的系统级架构设计。

通过对全库 5 大模型（`1D-ResNet`、`1D-FNO`、`Vanilla DeepONet`、`CJ-Cep-DeepONet`、`TG-CJ-DeepONet`）、3 大门控模块、注意力机制、损失函数库、评估指标与测试集的逐行审查，我们明确了当前系统的核心优势与致命瓶颈：
1. **核心优势**：基于物理声学往返到时 $\tau_j = t_s + 2x_j/a$ 的相对窗提取（`RelativeWindowGating`）将波前强制对齐，使 $\alpha$ $R^2$ 从 Phase 1 的 $0.4148$ 提升至 Phase 2 的 **$0.5666$**，$\alpha$ MAE 降至 **$0.1336$**，$W_1$ 降至 **$8.39\,\mathrm{m}$**，且 Masked Softmax 严格实现了浮点级单纯形物理守恒（偏差 $< 1.2 \times 10^{-7}$）。
2. **深层瓶颈（物理死结）**：面对密集多簇（间距 $5\sim 40\,\mathrm{m}$，反射时差仅 $6.9\sim 55.2\,\mathrm{ms}$），波形在多簇间发生强烈透射衰减与多径串扰。现有网络依赖纯注意力机制拟合非线性波前叠加，导致网络陷入**“多簇均摊陷阱”**（Equalization Trap），密集多簇下 $\alpha$ $R^2$ 停滞在 $0.5666$（未达 $>0.75$ 门槛），$C_f$ MRE 停滞在 $46.1\%$，且当前代码库尚未集成裂缝起裂分类头与容差 $\pm 10\,\mathrm{m}$ 的 F1-score 评测逻辑。
3. **突破路径**：必须引入一维波动方程传递矩阵的**可微层剥离（Differentiable Inverse Scattering Layer-Stripping, DIS）算子**。沿井筒由跟端到趾端逐级递推，显式计算局部物理反射系数 $\Gamma_j \in (-1, 0)$、透射率 $T_j = 1 + \Gamma_j$ 与支路导纳 $Y_{b,j}$，前向完全剥离上游累积透射功率损耗 $\prod_{k < j} T_k^2$，反向通过链式法则实现声学因果梯度流。

---

# 第一部分：现有代码库全景审计与模型实现剖析

## 1.1 模块与文件物理清单

`PaperC_CJNO_Wellbore_Inversion/` 现有代码架构清晰、解耦完备，主要目录结构与职责如下：

| 模块路径 | 核心文件 / 类 | 物理职责与输入输出 |
| :--- | :--- | :--- |
| `src/models/resnet1d.py` | `ResNet1D`, `ResBlock1D` | 1D 时序残差卷积基线，输入 `(B, 2, 4096)`，输出离散 $\hat{\boldsymbol{\alpha}}, \hat{\boldsymbol{C}}_f$ 与高斯重构场 |
| `src/models/fno1d.py` | `FNO1D`, `SpectralConv1d` | 1D 傅里叶神经算子基线，32 阶频域截断复数谱卷积，提取全局频响阻抗 |
| `src/models/deeponet.py` | `VanillaDeepONet`, `BranchNet`, `TrunkNet` | 经典 DeepONet，Branch 编码波形与工况，Trunk 傅里叶编码连续坐标 $x/L$，点积查询 |
| `src/models/cj_cep_deeponet.py` | `CJCepDeepONet`, `VoronoiPooling1D` | Phase 1 原型，波形+倒谱+工况+簇集合多模态融合，连续 Trunk 解码 + Voronoi 守恒空间池化 |
| `src/models/tg_cj_deeponet.py` | `TGCJDeepONet`, `GlobalWaveEncoder` | Phase 2 旗舰主干，集成时间门控、声学偏置自注意力与双轨协同头 |
| `src/modules/time_gating.py` | `TimeGatingModule`, 3 种 Gating 类 | 计算理论到时 $\tau_j = t_s + 2x_j/a$，实现 `relative_window`、`gaussian_gating`、`ceps_patch` |
| `src/modules/acoustic_transformer.py` | `AcousticBiasedTransformer`, `AcousticBiasedMultiheadAttention` | 注入声学时延物理偏置 $-\gamma \|x_i - x_j\|/a$，约束 $\gamma \ge 0$，簇间解耦自注意力 |
| `src/modules/dual_track_heads.py` | `DualTrackHead`, `DiscretePreciseHead`, `ContinuousTrunkHead` | 轨 1（离散精准头 Masked Softmax）+ 轨 2（连续 Trunk 场 Voronoi 积分）+ 双轨协同 |
| `src/losses.py` | `CompositeInversionLoss` | 复合损失：Simplex KL、Log-Huber、解析/连续 1D Wasserstein-1、双轨一致性 $L_{\text{cons}}$ |
| `src/metrics.py` | `compute_inversion_metrics` | 评估指标计算：$\alpha$ MAE, $R^2$, $C_f$ MRE, $C_f$ $\log_{10}$ MAE, $W_1$ (m), Simplex 偏差, $N_c$ 分层统计 |
| `src/dataset.py` | `PilotInversionDataset` | 1,000 例 H5 数据集载入，4096 点波形降采样，一阶差分，1024 点倒谱插值，500 点连续场构造，NPZ 缓存 |
| `experiments/train_ablation.py` | `train_single_ablation` | 4 大消融组自动化训练流水线（AdamW, CosineAnnealingLR, 60 epochs） |
| `experiments/evaluate_ablation.py` | `evaluate_ablation` | 100 例未见测试集全流程多模型对比评估与出版级图版生成脚本 |
| `tests/test_tg_models.py` | 7 项全流程测试 | 覆盖到时计算、3 种门控可微梯度、声学偏置单调性、单纯形约束、双轨一致性损失等 |
| `tests/test_pilot_models.py` | 5 项测试 | 覆盖基础模型管线、单样本边界、Wasserstein 连续网格等 |

---

## 1.2 主干模型 TG-CJ-DeepONet 的细粒度技术剖析

`TGCJDeepONet`（`src/models/tg_cj_deeponet.py`）是当前反演系统最先进的主干模型，其设计遵循三级前向流水线：

```
井口水头时程 wave: (B, 2, 4096)               已知射孔簇深度 positions: (B, M), 工况 cond: (B, 3)
                │                                            │
                ▼                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. 物理到时对齐提取模块 (TimeGatingModule)                                            │
│    - 理论往返到时: tau_j = t_s + 2 * x_j / a                                           │
│    - 提取模式: relative_window (截取 [tau - 50ms, tau + 250ms] -> 128点 -> 1D ConvNet)│
│    - 输出: cluster_tokens in R^{B, M, 64}, tau in R^{B, M}                             │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. 声学时延偏置解耦 Transformer (AcousticBiasedTransformer)                            │
│    - 坐标高频傅里叶位置编码 (8阶) + 工况投影 MLP 注入                                  │
│    - 声学传播时差物理偏置: A_{ij} = Softmax( (q_i k_j^T)/sqrt(d) - gamma * |x_i-x_j|/a)│
│    - gamma = Softplus(theta_gamma) >= 0 (物理非负距离衰减惩罚)                         │
│    - 输出: decoupled features H in R^{B, M, 64}, attn_maps in R^{B, L, H, M, M}       │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. 双轨协同解码头 (DualTrackHead)                                                      │
│    ┌──────────────────────────────────────────┐ ┌────────────────────────────────────┐ │
│    │ 轨 1: 离散精准解码头 (DiscretePreciseHead) │ │ 轨 2: 连续场辅助头 (ContinuousTrunk) │ │
│    │ - 独立 MLP -> 未归一化 logits            │ │ - z_well 融合全局波形与解耦特征      │ │
│    │ - Masked Softmax 严格保障 sum(alpha) = 1 │ │ - 10阶傅里叶 Trunk 解码连续密度场    │ │
│    │ - 对数顺应性: log10(Cf / Cf0)            │ │ - Voronoi 守恒空间数值积分           │ │
│    └────────────────────┬─────────────────────┘ └─────────────────┬──────────────────┘ │
│                         │                                         │                    │
│                         └────────────────────┬────────────────────┘                    │
│                                              ▼                                         │
│                    双轨协同一致性约束: L_cons = sum_j |alpha_j - alpha_field_j|        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### (1) 物理到时计算与门控机制 (`src/modules/time_gating.py`)
- 声速解析：$a = \mathrm{cond}[1] \times 20.0 + 1450.0\,\mathrm{m/s}$。
- 理论到达时间：$\tau_j = t_s + \frac{2 x_j}{a}$，未激活簇通过掩码重置为 $t_s$。
- 相对窗机制（`RelativeWindowGating`）：
  利用 PyTorch `F.grid_sample`，对每簇在时间范围 $[\tau_j - 0.05\,\mathrm{s}, \tau_j + 0.25\,\mathrm{s}]$（总长 $300\,\mathrm{ms}$）提取局部波形并可微插值至 $N_w = 128$ 点。该变换使得**所有不同深度的簇在局部时间轴上首波起跳点完全对齐在 $t = 50\,\mathrm{ms}$ 处**，彻底抹平了井深引起的绝对时延差异。

### (2) 声学时延偏置自注意力 (`src/modules/acoustic_transformer.py`)
- 注意力矩阵公式：
  $$\mathbf{S}_{ij} = \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d_k}} - \gamma \frac{|x_i - x_j|}{a}$$
- $\gamma = \mathrm{softplus}(\theta_\gamma) \ge 0$：初始化参数使 $\gamma \approx 10.0\,\mathrm{s}^{-1}$。
- 物理意义：两簇空间跨度越远，声波双程传播时延差越大，注意力得分受到的惩罚越大，强制自注意力抑制长距离虚假相关，聚焦于局部干涉解耦。
- Key Mask 机制：对填充的未激活簇（$j > N_c$），其 Key 列填入 $-10^9$，Softmax 后注意力严格为 $0$。

### (3) 双轨解码与物理单纯形保障 (`src/modules/dual_track_heads.py`)
- **轨 1（离散精准头）**：
  $$\hat{\alpha}_j = \frac{\exp(z_j) \cdot \mathbb{I}(j \in \mathcal{C}_{\text{valid}})}{\sum_{k \in \mathcal{C}_{\text{valid}}} \exp(z_k)}$$
  在 PyTorch 中，通过 `logits.masked_fill(~mask, -1e9)` 随后经 `F.softmax`，最后执行显式重整化：
  `pred_alpha = pred_alpha / pred_alpha.sum(dim=-1, keepdim=True)`。
  这确保了在单精度浮点运算下，$\max |\sum \hat{\alpha}_j - 1.0| < 1.2 \times 10^{-7}$，远超 $< 10^{-6}$ 的物理要求。
- **顺应性对数映射**：
  直接预测常用对数标量 $\beta_j = \log_{10}(\hat{C}_{f,j} / C_{f,0})$，换算为 $\hat{C}_{f,j} = C_{f,0} \cdot 10^{\beta_j}$ 及自然对数 $\ln(\hat{C}_{f,j} / C_{f,0}) = \beta_j \ln 10$。
- **轨 2（连续场辅助头）**：
  Trunk 网络基于 10 阶傅里叶基点乘生成 $N_{\text{grid}} = 500$ 点的连续流体密度场 $\hat{m}_\alpha(x)$，通过内置的 `VoronoiPooling1D` 在各簇 Voronoi 积分区间内做 5 点 Gauss-Legendre 复合数值积分，输出辅助流量分配 $\hat{\alpha}_j^{\text{field}}$。

---

## 1.3 训练管线与损失函数结构 (`src/losses.py`, `experiments/train_ablation.py`)

### 复合损失函数配比
当前训练采用多目标自适应物理损失函数：
$$\mathcal{L}_{\text{total}} = \lambda_\alpha \mathcal{L}_\alpha + \lambda_c \mathcal{L}_c + \lambda_w \mathcal{L}_w + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}$$
各损失项具体定义如下：
1. **$\mathcal{L}_\alpha$ (流量分配损失, $\lambda_\alpha = 1.0$)**：
   $$\mathcal{L}_\alpha = \mathcal{L}_{KL} + 0.5 \times \mathcal{L}_{L1} = -\sum_{j \in \mathcal{C}} \alpha_j \ln(\hat{\alpha}_j + \epsilon) + 0.5 \frac{1}{N_c} \sum_{j \in \mathcal{C}} |\hat{\alpha}_j - \alpha_j|$$
2. **$\mathcal{L}_c$ (顺应性对数 Huber 损失, $\lambda_c = 1.0$)**：
   在对数空间 $\ln(C_f / C_{f,0})$ 执行 Smooth L1 回归（阈值 $\delta = 0.2$），有效抑制裂缝高压气塞或超小尺度顺应性的离群梯度。
3. **$\mathcal{L}_w$ (1D 空间 Wasserstein 测度损失, $\lambda_w = 0.01$)**：
   $$W_1(\hat{m}, m) = \int_0^L |\hat{F}_m(x) - F_m(x)| dx$$
   以米为物理单位。未归一化前 $W_1 \approx 8\sim 10\,\mathrm{m}$，乘以 $\lambda_w = 0.01$ 后量级平衡在 $0.08\sim 0.10$。
4. **$\mathcal{L}_{\text{cons}}$ (双轨物理协同一致性损失, $\lambda_{\text{cons}} = 0.5$)**：
   $$\mathcal{L}_{\text{cons}} = \frac{1}{N_c} \sum_{j=1}^{N_c} |\hat{\alpha}_j - \hat{\alpha}_j^{\text{field}}|$$
   强迫离散解码与连续场 Voronoi 守恒积分闭环一致。

### 训练优化超参数
- 优化器：`AdamW(lr=1e-3, weight_decay=1e-4)`
- 学习率调度：`CosineAnnealingLR(T_max=60, eta_min=1e-5)`
- 梯度截断：`clip_grad_norm_(max_norm=5.0)`
- 验证频率：每 2 轮评估一次，基于综合评分 `alpha_mae + cf_log10_mae` 挑选最佳模型权重保存在 `output/weights/{exp_name}_best.pt`。

---

## 1.4 评估指标体系剖析 (`src/metrics.py`)

系统内置指标与当前验收要求对标如下：

| 指标名称 | 计算公式与物理含义 | 阶段 2 验收门槛 | 当前 TG-Rel-Bias 表现 | Phase 3 目标门槛 |
| :--- | :--- | :---: | :---: | :---: |
| **$\alpha$ MAE** | $\frac{1}{\sum N_c} \sum \|\hat{\alpha}_{ij} - \alpha_{ij}\|$ | $< 0.150$ | **0.1336** | $< 0.08$ (全集), $< 0.03$ (单簇) |
| **$\alpha$ $R^2$** | $1 - \frac{\sum (\alpha - \hat{\alpha})^2}{\sum (\alpha - \bar{\alpha})^2}$ (决定系数) | $> 0.500$ | **0.5666** | $> 0.75$ (密集多簇) |
| **$C_f$ MRE** | $\mathrm{Median}(\|\hat{C}_f - C_f\| / C_f) \times 100\%$ | $< 50.0\%$ | **46.1%** | $< 25.0\%$ |
| **$C_f$ $\log_{10}$ MAE** | $\mathrm{Mean}(\|\log_{10}(\hat{C}_f / C_f)\|)$ | $< 0.350$ | **0.3315** | $< 0.200$ |
| **$W_1$ (m)** | 解析离散 CDF 差积分 $\sum \|\Delta F_k\| \Delta x_k$ | $< 10.0\,\mathrm{m}$ | **8.39 m** | $< 5.0\,\mathrm{m}$ |
| **Simplex Max Dev** | $\max \|\sum \hat{\alpha}_j - 1.0\|$ | $< 10^{-6}$ | **$1.19 \times 10^{-7}$** | $< 10^{-6}$ (严格保持) |
| **F1-score ($\pm 10\mathrm{m}$)** | 容差 $10\mathrm{m}$ 裂缝起裂定位检出 F1 | — | **未实现 / 缺失** | $> 0.88$ |
| **20dB 噪声鲁棒性** | SNR 20dB 下指标衰减幅度 | — | **未测试 / 缺失** | 指标衰减 $< 15\%$ |

---

# 第二部分：Phase 2 消融评测数据深入反思与瓶颈归因

## 2.1 全集 100 例测试集宏观对比

以下数据直接提取自生产环境 `PaperC_CJNO_Wellbore_Inversion/output/ablation_metrics_summary.json`：

| 模型代号 | 架构与机制 | $\alpha$ MAE ↓ | $\alpha$ $R^2$ ↑ | $C_f$ MRE ↓ | $C_f \log_{10}$ MAE ↓ | $W_1$ (m) ↓ | Simplex 最大偏差 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `resnet` | 1D-ResNet (Phase 1 基准) | 0.1449 | 0.5072 | 46.83% | 0.3527 | 9.55 m | $1.19 \times 10^{-7}$ |
| `fno` | 1D-FNO (Phase 1 基准) | 0.1449 | 0.5072 | 45.73% | 0.3516 | 9.55 m | $1.19 \times 10^{-7}$ |
| `cj_cep_deeponet` | CJ-Cep-DeepONet (Phase 1) | 0.1548 | 0.4148 | 46.43% | 0.3517 | 9.77 m | $1.19 \times 10^{-7}$ |
| `tg_best_nobias` | TG-Rel-NoBias (无声学偏置) | 0.1449 | 0.5364 | 47.96% | 0.3521 | 9.20 m | $1.19 \times 10^{-7}$ |
| `tg_gaussian_bias` | TG-Gauss-Bias (高斯门控) | 0.1447 | 0.5113 | 45.86% | 0.3521 | 9.55 m | $1.19 \times 10^{-7}$ |
| `tg_ceps_patch_bias` | TG-Patch-Bias (倒谱切片) | 0.1449 | 0.5092 | 46.06% | 0.3523 | 9.54 m | $1.19 \times 10^{-7}$ |
| `tg_relative_bias` | **TG-Rel-Bias (Phase 2 旗舰)** | **0.1336** | **0.5666** | 46.09% | **0.3315** | **8.39 m** | **$1.19 \times 10^{-7}$** |

## 2.2 变簇数 ($N_c \in [1..6]$) 分层数据透视

分析 Phase 2 旗舰模型 `tg_relative_bias` 在不同簇数下的表现（样本量分布：17/19/17/12/19/16）：
- **$N_c = 1$ (单簇)**：$\alpha$ MAE 严格为 **0.0000**，$W_1 = 0.00\,\mathrm{m}$，$C_f$ MRE 为 $17.28\%$。单簇工况完全可辨识。
- **$N_c = 2$**：$\alpha$ MAE = 0.2259，$W_1 = 5.22\,\mathrm{m}$，$C_f$ MRE = $37.09\%$。两簇强干涉开始显现。
- **$N_c = 3$**：$\alpha$ MAE = 0.1633，$W_1 = 6.04\,\mathrm{m}$，$C_f$ MRE = $42.08\%$。
- **$N_c = 4$**：$\alpha$ MAE = 0.1560，$W_1 = 14.20\,\mathrm{m}$，$C_f$ MRE = $56.73\%$。
- **$N_c = 5$**：$\alpha$ MAE = 0.1269，$W_1 = 13.92\,\mathrm{m}$，$C_f$ MRE = $56.62\%$。
- **$N_c = 6$**：$\alpha$ MAE = 0.1002，$W_1 = 12.68\,\mathrm{m}$，$C_f$ MRE = $42.26\%$。

## 2.3 Phase 2 核心瓶颈与物理机理归因 (Root Cause Analysis)

为什么在引入了物理到时对齐和声学时延注意力后，$\alpha$ $R^2$ 依然卡在 $0.5666$，未达到 Acceptance Criteria 的 $0.75$？

1. **上游多簇的透射扼流与混叠串扰（Transmission Choking & Crosstalk）**：
   关泵水击反射波自井底向井口回传时，第 $j$ 簇的反射波必须穿透前序所有裂缝 $1, 2, \dots, j-1$。
   若上游裂缝已经起裂并具有较大支路导纳 $Y_{b,k}$，声波透射系数 $T_k = \frac{2}{2 + Z_0 Y_{b,k}} < 1.0$。
   当声波穿过前 $j-1$ 个裂缝时，其总透射衰减因子为 $\prod_{k=1}^{j-1} T_k^2$。
   **现有 TG-CJ-DeepONet 仅通过时域窗口截取，未在物理上补偿上游透射损耗**。井口观测到的第 $j$ 簇回波幅值被严重压低，网络极易将其误判为“进液量极小”，产生严重的系统性跟部假阳性与趾端欠估计。
2. **多簇均摊陷阱（Equalization Trap）**：
   在簇间距仅 $5\sim 20\,\mathrm{m}$ 时，往返时延差仅 $6.9\sim 27.6\,\mathrm{ms}$，时域重叠严重。由于缺乏物理正向散射方程的逆解耦约束，神经注意力机制在处理欠定相干叠加时倾向于输出收缩解（$\hat{\alpha}_j \approx 1/N_c$），从而规避极端的方差惩罚。这是 $R^2$ 停留在 $0.56$ 的数学根源。
3. **缺少显式物理白盒中间层**：
   当前网络直接将卷积/门控表征输入 Transformer 和解码头，没有显式计算物理反射率 $\Gamma_j$ 与支路导纳 $Y_{b,j}$。这使得黑盒网络在面对非线性摩阻和射孔压降时缺乏物理锚点。
4. **功能缺失**：
   - 缺少裂缝起裂检出分类器（Existence / Breakdown Classifier）；
   - 缺少容差 $\pm 10\,\mathrm{m}$ 的 F1-Score 动态匹配评估算法；
   - 缺少 SNR 30dB/20dB/10dB 强噪声与声速扰动审计实验管线。

---

# 第三部分：可微逆散射层剥离算子 (DIS Layer) 理论推导与数学规范

## 3.1 一维水击声学传递矩阵与支路导纳

在井筒水力瞬变流中，水头特征阻抗为 $Z_0 = \frac{a}{g A}$。
将射孔簇等效为主管道上的并联水力支路，支路流体进入动态阻抗为 $Z_{b,j}(s)$，导纳为 $Y_{b,j}(s) = 1 / Z_{b,j}(s)$。

在节点 $x_j$ 处，水头连续、流量守恒：
$$H_j^-(t) = H_j^+(t) = H_j(t)$$
$$Q_{in,j}(t) - Q_{out,j}(t) = q_{b,j}(t) = Y_{b,j} * H_j(t)$$

根据波动方程分离入射波 $H^+$ 与反射波 $H^-$：
$$H = H^+ + H^-, \quad Q = \frac{1}{Z_0}(H^+ - H^-)$$
联立解得射孔簇局部声学反射系数 $\Gamma_j$ 与透射系数 $T_j$：
$$\Gamma_j = - \frac{Z_0 Y_{b,j}}{2 + Z_0 Y_{b,j}} \in (-1, 0]$$
$$T_j = 1 + \Gamma_j = \frac{2}{2 + Z_0 Y_{b,j}} \in (0, 1]$$

反向映射公式（由反射率反解物理支路导纳）：
$$Y_{b,j} = - \frac{2 \Gamma_j}{Z_0 (1 + \Gamma_j)}$$

在稳态下，各簇进液份额 $\alpha_j$ 正比于其导纳实部（有效进液能力）：
$$\alpha_j = \frac{\mathrm{Re}(Y_{b,j})}{\sum_{k=1}^{N_c} \mathrm{Re}(Y_{b,k})}$$

## 3.2 Schur/Bruckstein 递归层剥离算法与数学公式体系

考虑排列在水平段的 $N_c$ 簇裂缝，深度坐标满足：
$$0 < x_1 < x_2 < \dots < x_{N_c} \le L$$
井口关泵产生向趾端传播的初始水击波前 $P_0^+$（Joukowsky 阶跃 $\Delta H_J = \frac{a \Delta V}{g}$）。

由于因果律，声波到达各簇的双程往返时延严格单调递增：
$$\tau_1 < \tau_2 < \dots < \tau_{N_c}, \quad \tau_j = t_s + \frac{2 x_j}{a}$$

### 递归层剥离推导：
1. **第 1 簇（跟端首簇 $x_1$）**：
   入射波幅值为未受任何上游裂缝干扰的初始脉冲 $A_1^{inc} = P_0^+$。
   在井口时刻 $\tau_1$ 到达的首波反射回波仅来自第 1 簇的单次反射：
   $$R_1 = \Gamma_1 P_0^+ \implies \Gamma_1 = \frac{R_1}{P_0^+}$$
   透射穿过第 1 簇继续向下游传播的波前为：
   $$P_1^{trans} = T_1 P_0^+ = (1 + \Gamma_1) P_0^+$$
2. **第 2 簇（$x_2$）**：
   到达第 2 簇的下行波幅值已被第 1 簇削弱为 $P_1^{trans} = (1+\Gamma_1) P_0^+$。
   在第 2 簇处产生的反射波为 $\Gamma_2 P_1^{trans}$。
   该反射波返回井口时，又必须逆向穿过第 1 簇，再次乘以透射系数 $T_1$：
   $$R_2 = T_1 \cdot (\Gamma_2 P_1^{trans}) = T_1^2 \Gamma_2 P_0^+ = (1 + \Gamma_1)^2 \Gamma_2 P_0^+$$
   因此，消除第 1 簇扼流效应后的真实物理反射率为：
   $$\Gamma_2 = \frac{R_2}{(1 + \Gamma_1)^2 P_0^+}$$
3. **一般形式（第 $j$ 簇递归推导）**：
   对于第 $j$ 簇，其井口观测到的初至反射脉冲 $R_j$ 经历了前序所有 $1 \sim j-1$ 簇的往返双重透射衰减：
   $$R_j = \left( \prod_{k=1}^{j-1} T_k^2 \right) \Gamma_j P_0^+ = \left( \prod_{k=1}^{j-1} (1 + \Gamma_k)^2 \right) \Gamma_j P_0^+$$
   **逆散射层剥离恢复公式**：
   $$\Gamma_j = \frac{R_j}{\left( \prod_{k=1}^{j-1} (1 + \Gamma_k)^2 \right) P_0^+}$$
   透射系数更新：
   $$T_j = 1 + \Gamma_j$$
   累积透射功率因子：
   $$\mathcal{T}_{1:j} = \prod_{k=1}^j (1 + \Gamma_k)^2$$

## 3.3 端到端可微实现与前向后向梯度流

为将上述经典层剥离公式无缝嵌入 PyTorch 神经算子，避免除以零和数值不稳定性，我们设计**可微软约束可微层剥离层（Differentiable Layer-Stripping Layer, DIS Layer）**：

### 前向图计算 (Forward Pass)
设神经网络到时门控输出的未归一化回波特征强度为 $u_j \in \mathbb{R}$（由局部卷积网络提取）：
1. 采用 Sigmoid 软激活严格限制初始单次反射率：
   $$\tilde{\Gamma}_j = - \mathrm{sigmoid}(w_r u_j + b_r) \cdot \gamma_{\max}, \quad \gamma_{\max} = 0.95$$
   保证 $1 + \tilde{\Gamma}_j \ge 0.05 > 0$，从数学上根除奇异值。
2. 循环递推解耦：
   - 初始化累积透射功率：$\mathcal{T}_0 = 1.0$
   - 对 $j = 1, \dots, N_c$：
     $$\Gamma_j = \frac{\tilde{\Gamma}_j}{\sqrt{\mathcal{T}_{j-1}} + \epsilon_{\text{stab}}}$$
     截断保护：$\Gamma_j \leftarrow - \mathrm{clamp}(-\Gamma_j, \min=10^{-4}, \max=0.95)$
     $$T_j = 1.0 + \Gamma_j$$
     $$\mathcal{T}_j = \mathcal{T}_{j-1} \cdot T_j^2$$
     物理支路导纳：
     $$Y_{b,j} = - \frac{2 \Gamma_j}{Z_0 (1.0 + \Gamma_j)}$$

### 后向梯度流 (Backward Gradient Flow)
由于整个递归过程只包含标量乘法、加法与平方，PyTorch Autograd 计算图完整保持可微。
根据多元微分链式法则：
$$\frac{\partial \mathcal{L}}{\partial \Gamma_k} = \frac{\partial \mathcal{L}}{\partial \alpha_k}\frac{\partial \alpha_k}{\partial \Gamma_k} + \sum_{j=k+1}^{N_c} \frac{\partial \mathcal{L}}{\partial \alpha_j}\frac{\partial \alpha_j}{\partial \mathcal{T}_{j-1}}\frac{\partial \mathcal{T}_{j-1}}{\partial T_k}\frac{\partial T_k}{\partial \Gamma_k}$$
**物理内涵**：当网络优化下游趾端裂缝 $j$ 的预测时，梯度不仅直接回传给 $j$，还会通过 $\frac{\partial \mathcal{T}_{j-1}}{\partial T_k}$ 反向约束上游跟端裂缝 $k$ 的透射率。若上游裂缝预测反射率过高，会导致下游累积透射率过低，触发惩罚梯度自动纠偏。这在反向传播层面建立了真正的多簇声学耦合因果链！

---

# 第四部分：TG-DIS-DeepONet 全新架构设计与集成方案

## 4.1 总体架构设计图

```
====================================================================================================
                        TG-DIS-DeepONet 端到端可解释物理反演网络架构
====================================================================================================

      井口水头时程 wave: (B, 2, 4096)                 已知簇坐标与声速: positions, cond
                      │                                              │
                      ▼                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. 物理到时对齐提取模块 (Time-Gating): 理论往返到时 tau_j = t_s + 2*x_j/a            │
  │    利用可微一维重采样 (grid_sample) 截取每簇局部回波窗口 [tau-50ms, tau+250ms] -> 128点 │
  └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                      │ raw_tokens v_j: (B, M, d_in)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. 可微逆散射层剥离解耦层 (Differentiable Layer-Stripping Layer, DIS Layer) [★核心创新] │
  │    - 严格沿井筒由跟端到趾端 (j = 1 -> Nc) 逐级递归                                    │
  │    - 显式计算上游累积透射功率衰减: T_{1:j-1} = prod_{k=1}^{j-1} (1 + Gamma_k)^2      │
  │    - 动态补偿解耦出物理真值反射率: Gamma_j in (-1, 0)                                 │
  │    - 显式导出物理支路导纳标量: Y_{b,j} = -2*Gamma_j / [Z_0 * (1 + Gamma_j)]           │
  │    - 物理特征拼接增强: h_j = [v_j || Gamma_j || log(Y_{b,j}) || T_{1:j-1}]            │
  └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                      │ physically stripped tokens: (B, M, d_model)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. 非对称声学时延因果偏置解耦 Transformer (Asymmetric Acoustic-Biased Transformer)      │
  │    A_{ij} = Softmax( q_i k_j^T / sqrt(d) - gamma_sym |x_i-x_j|/a - gamma_asym ReLU(x_j-x_i)/a )│
  │    融入跟-趾方向波传播声学因果非对称偏置, 彻底消除跨空间远距假相关                    │
  └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                      │ decoupled features H: (B, M, d_model)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 4. 多任务可解释解耦解码头 (Multi-Task Decoupling Heads):                                │
  │    ┌───────────────────────────────────┐    ┌────────────────────────────────────────┐ │
  │    │ 头 A: 裂缝起裂存在性分类头        │    │ 头 B: 亚米级位置细化微调头             │ │
  │    │ p_exist_j = Sigmoid(MLP(H_j))     │    │ delta_x_j = clamp(MLP(H_j), -10, 10) m │ │
  │    └─────────────────┬─────────────────┘    └───────────────────┬────────────────────┘ │
  │    ┌─────────────────┴─────────────────┐    ┌───────────────────┴────────────────────┐ │
  │    │ 头 C: 严格物理单纯形流量分配头    │    │ 头 D: 连续导纳谱与水力顺应性头         │ │
  │    │ alpha_j = MaskedSoftmax(z_j, p)   │    │ log10(Cf_j / Cf0) = MLP(H_j)           │ │
  │    │ 满足 max|sum(alpha) - 1.0| < 1e-6 │    │ 连续流体密度场 m_alpha(x) & c_H(x)     │ │
  │    └───────────────────────────────────┘    └────────────────────────────────────────┘ │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

## 4.2 DIS 层详细实现设计 (`src/modules/layer_stripping.py`)

建议新建模块文件 `PaperC_CJNO_Wellbore_Inversion/src/modules/layer_stripping.py`，核心代码逻辑规范如下：

```python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableLayerStripping(nn.Module):
    """
    波动方程可微逆散射层剥离算子 (DIS Layer):
    基于一维水力瞬变流传递矩阵，沿井筒跟端至趾端 (x_1 -> x_Nc) 递归解耦上游裂缝透射衰减。
    显式输出物理反射率 Gamma_j 与支路导纳 Y_{b,j}。
    """
    def __init__(self, in_dim: int = 64, d_model: int = 64, g: float = 9.80665, A_pipe: float = 0.015328):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.g = g
        self.A_pipe = A_pipe # D=0.1397m -> A = pi * (D/2)^2 = 0.015328 m^2

        # 初始反射率估计器 (由到时窗口特征抽取反射强度标量)
        self.refl_mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 剥离后物理特征升维融合投影层
        # 输入: [原始特征 v_j (in_dim), Gamma_j (1), log_Yb (1), T_cum (1)] -> 总维数 in_dim + 3
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim + 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        cluster_tokens: torch.Tensor, # (B, M, in_dim)
        positions: torch.Tensor,      # (B, M) [m]
        wavespeed: torch.Tensor,      # (B, 1) [m/s]
        mask: torch.Tensor,           # (B, M) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            stripped_tokens: (B, M, d_model) 解耦增强表征
            gamma: (B, M) 物理反射率序列 Gamma_j in (-1, 0)
            yb: (B, M) 物理支路导纳序列 Y_{b,j}
            t_cum: (B, M) 累积透射功率损耗因子
        """
        B, M, _ = cluster_tokens.shape
        device = cluster_tokens.device

        # 水头特征阻抗 Z_0 = a / (g * A)
        a = wavespeed.view(B, 1) # (B, 1)
        Z_0 = a / (self.g * self.A_pipe) # (B, 1) ~ 1450 / (9.8 * 0.0153) ~ 9647.0 s/m^2

        raw_refl_logits = self.refl_mlp(cluster_tokens).squeeze(-1) # (B, M)

        gamma_list = []
        yb_list = []
        t_cum_list = []

        # 累积透射功率初始化为 1.0
        T_cum_prev = torch.ones(B, device=device) # (B,)

        for j in range(M):
            # 基础反射强度预测 in (0, 0.95)
            base_refl = torch.sigmoid(raw_refl_logits[:, j]) * 0.95 # (B,)
            
            # 逆散射层剥离：补偿上游所有裂缝的双程累积透射损耗
            gamma_j_val = - (base_refl / (torch.sqrt(T_cum_prev) + 1e-4))
            gamma_j = torch.clamp(gamma_j_val, min=-0.95, max=-1e-4) # 物理反射率严格为负

            # 单层透射率 T_j = 1 + Gamma_j in (0.05, 1.0)
            T_j = 1.0 + gamma_j
            
            # 物理支路导纳 Y_{b,j} = - 2 * Gamma_j / [Z_0 * (1 + Gamma_j)]
            Y_b_j = - (2.0 * gamma_j) / (Z_0.squeeze(-1) * T_j + 1e-6)

            # 更新下一级上游累积透射功率 (若当前簇为假阳性/未激活则不衰减)
            m_j = mask[:, j].float()
            effective_T_sq = (T_j ** 2) * m_j + 1.0 * (1.0 - m_j)
            T_cum_next = T_cum_prev * effective_T_sq

            gamma_list.append(gamma_j)
            yb_list.append(Y_b_j)
            t_cum_list.append(T_cum_prev)

            T_cum_prev = T_cum_next

        gamma = torch.stack(gamma_list, dim=1) # (B, M)
        yb = torch.stack(yb_list, dim=1)       # (B, M)
        t_cum = torch.stack(t_cum_list, dim=1) # (B, M)

        # 掩码清零
        gamma = torch.where(mask, gamma, torch.zeros_like(gamma))
        yb = torch.where(mask, yb, torch.zeros_like(yb))

        # 特征拼接与升维投影
        log_yb = torch.log(torch.clamp(yb, min=1e-7))
        log_yb = torch.where(mask, log_yb, torch.zeros_like(log_yb))
        
        phys_feat = torch.stack([gamma, log_yb, t_cum], dim=-1) # (B, M, 3)
        combined = torch.cat([cluster_tokens, phys_feat], dim=-1) # (B, M, in_dim + 3)
        stripped_tokens = self.out_proj(combined)
        stripped_tokens = torch.where(mask.unsqueeze(-1), stripped_tokens, torch.zeros_like(stripped_tokens))

        return stripped_tokens, gamma, yb, t_cum
```

## 4.3 方向因果声学时延偏置注意力升级

在当前对称偏置 $-\gamma \frac{|x_i - x_j|}{a}$ 的基础上，升级为**因果非对称偏置矩阵**：
水锤波由井口关泵注入，能量主流沿跟端（Heel）向趾端（Toe）传播；上游裂缝对下游裂缝的声学扰动具有前向强因果性，而下游对上游的直接影响受到时间因果延迟制约。
因此，注意力偏置矩阵设计为：
$$\mathbf{B}_{ij} = - \gamma_{\text{sym}} \frac{|x_i - x_j|}{a} - \gamma_{\text{dir}} \frac{\max(0, x_j - x_i)}{a}$$
其中：
- $\gamma_{\text{sym}} \ge 0$ 控制几何邻域相互干涉衰减；
- $\gamma_{\text{dir}} \ge 0$ 显式对“反因果”（下游 Key 强控上游 Query）施加额外单向声学时延阻尼。

## 4.4 裂缝存在性分类头与 $\pm 10\mathrm{m}$ 容差 F1-score 算法

### (1) 分类与亚米级位置细化头设计
为彻底满足 Acceptance Criteria 对起裂检出 F1-score 的要求，在解码端新增独立分类头与定位偏移头：
```python
# 裂缝起裂存在性分类头 (二分类概率)
self.cls_head = nn.Sequential(
    nn.Linear(d_model, 32),
    nn.GELU(),
    nn.Linear(32, 1),
    nn.Sigmoid(),
)

# 亚米级空间定位残差头
self.loc_head = nn.Sequential(
    nn.Linear(d_model, 32),
    nn.GELU(),
    nn.Linear(32, 1),
    nn.Tanh(), # 输出 [-1, 1] 映射到 +/- 10.0 m 范围
)
```

### (2) 容差 $\pm 10\mathrm{m}$ F1-score 评估算法实现规范
在 `src/metrics.py` 中新增 `compute_detection_f1_score` 函数：
```python
def compute_detection_f1_score(
    pred_probs: np.ndarray,      # (N, M) 存在性预测概率 in [0, 1]
    pred_positions: np.ndarray,  # (N, M) 预测裂缝深度 (m)
    true_mask: np.ndarray,       # (N, M) 真实起裂布尔掩码
    true_positions: np.ndarray,  # (N, M) 真实裂缝深度 (m)
    prob_threshold: float = 0.5,
    tolerance_m: float = 10.0,
) -> dict:
    """
    计算容差 +/- 10m 下的裂缝检出 Precision, Recall 与 F1-score。
    基于二分图贪心距离匹配。
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    N, M = pred_probs.shape
    for i in range(N):
        # 提取当前样本预测为起裂的簇位置
        pred_act = pred_positions[i][pred_probs[i] >= prob_threshold]
        # 提取当前样本真实起裂的簇位置
        true_act = true_positions[i][true_mask[i]]

        if len(pred_act) == 0 and len(true_act) == 0:
            continue
        elif len(pred_act) == 0:
            total_fn += len(true_act)
            continue
        elif len(true_act) == 0:
            total_fp += len(pred_act)
            continue

        # 距离矩阵匹配 (Greedy Matching)
        matched_true = set()
        matched_pred = set()
        for p_idx, p_x in enumerate(pred_act):
            best_dist = float("inf")
            best_t_idx = -1
            for t_idx, t_x in enumerate(true_act):
                if t_idx in matched_true:
                    continue
                dist = abs(p_x - t_x)
                if dist <= tolerance_m and dist < best_dist:
                    best_dist = dist
                    best_t_idx = t_idx
            if best_t_idx != -1:
                matched_true.add(best_t_idx)
                matched_pred.add(p_idx)

        tp = len(matched_pred)
        fp = len(pred_act) - tp
        fn = len(true_act) - len(matched_true)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }
```

---

# 第五部分：鲁棒性审计实验体系设计 (R3 规范)

为满足 Acceptance Criteria 中“在 20dB 强噪声扰动下，核心指标衰减幅度 < 15%”以及中科院 2 区学术论文的标准，必须在数据管线中嵌入规范的鲁棒性压力测试注入器：

## 5.1 双阶扰动生成机制
1. **白噪声与有色噪声注入 (SNR 30dB, 20dB, 10dB)**：
   对归一化水头波形 $H(t)$，依目标信噪比计算噪声功率 $\sigma_n^2 = \frac{\sigma_s^2}{10^{\mathrm{SNR}/10}}$：
   - 高斯白噪声：$n_w(t) \sim \mathcal{N}(0, \sigma_n^2)$；
   - 现场实测有色粉红噪声（$1/f$ 谱衰减）：对白噪声经一阶低通滤波并缩放功率，模拟离心泵低频振颤与传感器漂移。
2. **声速扰动 ($\pm 1.0\%$)**：
   在理论往返到时计算 $\tau_j = t_s + 2x_j / \tilde{a}$ 中，人为注入声速测量误差 $\tilde{a} = a \times (1 \pm 0.01)$（即 $\pm 14.5\,\mathrm{m/s}$），测试到时对齐对现场声速不准时的容错性。

## 5.2 鲁棒性退化率审计判据
定义核心指标 $\mathcal{M} \in \{\alpha\text{-MAE}, R^2, W_1\}$ 在噪声下的退化率：
$$\Delta_{\text{degrade}} = \frac{|\mathcal{M}_{\text{noise}} - \mathcal{M}_{\text{clean}}|}{\mathcal{M}_{\text{clean}}} \times 100\%$$
验收门槛：在 SNR = 20dB 强噪声扰动下，$\Delta_{\text{degrade}} \le 15.0\%$。

---

# 第六部分：测试套件与代码集成落地路线图

## 6.1 现有测试验证基准与运行现状
我们已通过命令行实测验证了现有测试套件的健康状态：
1. `pytest -q tests/test_pilot_models.py`：**5 项测试全量通过 (5 passed in 3.22s)**；
2. `pytest -q tests/test_tg_models.py`：**7 项测试全量通过 (7 passed in 10.78s)**；
3. 全库原有测试 100% 绿色可用，测试架构以 PyTorch 张量形态校验、单纯形偏差断言、可微梯度反传检查为主。

## 6.2 实施集成三步走计划 (Implementation Roadmap)

针对下游 Worker 和 Reviewer 团队，建议按以下清晰工序开展代码实现与实验：

### 阶段 1：核心模块构建与单元测试 (M1 - DIS Operator)
1. **新建文件**：`PaperC_CJNO_Wellbore_Inversion/src/modules/layer_stripping.py`，实现 `DifferentiableLayerStripping` 类；
2. **新建测试**：`tests/test_dis_layer.py`：
   - 验证单层与多层前向计算正确性（反射率严格在 $(-1, 0)$，累积透射功率单调非增）；
   - 验证对输入特征与可学习参数的全程梯度反传；
   - 验证单簇边界与 5m 极密簇数值稳定性。

### 阶段 2：模型集成与训练流水线 (M2 - TG-DIS-DeepONet)
1. **新建模型**：`PaperC_CJNO_Wellbore_Inversion/src/models/tg_dis_deeponet.py`，构建 `TGDISDeepONet`：
   - 集成 `TimeGatingModule` + `DifferentiableLayerStripping` + `AcousticBiasedTransformer` + `MultiTaskHeads`；
   - 包含起裂存在性分类头 `p_exist`、单纯形流量分配头 `alpha`、顺应性头 `log_cf`、位置偏移头 `delta_x`；
2. **升级指标与损失**：
   - 在 `src/losses.py` 中增加起裂分类 BCE 损失项 $\lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}$；
   - 在 `src/metrics.py` 中增加 `compute_detection_f1_score`；
3. **编写训练脚本**：`experiments/train_dis.py`，完成 TG-DIS-DeepONet 的 60 轮基线训练。

### 阶段 3：全基线对标、鲁棒性审计与图版报告交付 (M3/M4 - Benchmark & Report)
1. **横向对标**：在 100 例测试集上对标 5 大模型（1D-ResNet, 1D-FNO, DeepONet, TG-DeepONet, TG-DIS-DeepONet）；
2. **两阶鲁棒性测试**：注入 SNR 30dB, 20dB, 10dB 噪声及 $\pm 1\%$ 声速扰动，记录指标退化率；
3. **输出图版**：绘制 Nature 规范的多模型对比柱状图、层剥离脉冲解耦波形图与注意力热力图；
4. **撰写报告**：输出完整的出版级学术技术总结报告 `phase3_inverse_scattering_report.md`。

---

# 结论 (Conclusion)

本调查报告为 PaperC 第三阶段研发确立了坚实的工程与物理理论基石：
1. 现有代码库架构规范、测试完备，Phase 2 实现了波前到时对齐，但受制于上游透射扼流与多径混叠，导致 $\alpha$ $R^2$ 停滞在 $0.5666$；
2. 引入基于波动方程传递矩阵的**可微逆散射层剥离算子（DIS Layer）**是击破“多簇均摊陷阱”的最优物理求解路径；
3. 本报告提出的 TG-DIS-DeepONet 架构、端到端梯度流推导、非对称声学因果注意力矩阵以及 $\pm 10\mathrm{m}$ 容差 F1 评估算法，完全覆盖了 Authoritative User Request 的各项验收指标，可立即移交下游任务开展代码实施与实验评测。
