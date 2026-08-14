---
type: research-topic
topic_id: T04
status: active
paper_status: paper-bc-drafted
tags:
  - research-topic
  - brunone
  - waveform-degradation
  - dispersion-hypothesis
---

# T04 Brunone 波形退化与频散假设

> 文件名保留“Brunone频散效应”以避免既有链接断裂；当前证据主线已调整为波形退化和频率选择性衰减，严格频散仍是待验证假设。

## 核心科学问题

Brunone 非定常摩阻如何改变水击波的频带能量、到时特征、波包宽度和多裂缝可分辨能力？这些变化中哪些属于耗散，哪些能证明严格频散？

## 适用范围

现有主矩阵使用固定 k 的 monkey-patch 参数研究，包含 6 个 k × 5 个间距 = 30 个四裂缝工况。它用于隔离参数效应，不等价于现场 Re-dependent Brunone 模型或流变标定。

## 已观察结果

- 高频成分相对低频显著减弱，支持频率选择性耗散。
- 随 k 增大，反射峰衰减、展宽、后移并逐步融合。
- D=20 m、k=0.2 时，EXP-024 复算给出峰漂移 **277.0 ms**、EST **101.0 ms**（四窗重复）。
- 共同参考 STFT：同一锚点 60–150 Hz 相对 k=0 为 **−44.1 dB**，0–20 Hz 为 −12.9 dB。
- `a_f0`、`a_onset`、`a_peak` 降幅 11.0% / 1.3% / 4.7%。
- 时域 \(C_v\) 随 k 增大、D 减小而下降（合成双峰单元测试通过）；**不**作为最小缝距表。
- **Re-dependent k 下的单缝深度偏差落在 10–20 m 之间**（[[EXP-20260801-001-盲检测协议落地与历史结果复算]]
  的容差扫描：5 个 brunone 单缝 case 在 10 m 容差下 F1 全为 0，在 20 m 容差下全为 1；
  steady 对照在 2 m 容差下即为 1）。该偏差此前被 135 m 匹配容差完全掩盖。

## 固定 k 扫描与 Re-dependent k 必须分开

`analysis/brunone_spacing_effect/` 的 k×D 矩阵把 k 作为外部常数扫描；LHS 数据集则由 `brunone_k(Re)` 随局部瞬时 Re 变化。两者不能直接互换查表。

[[EXP-20260730-010-LHS数据集实际生效k实测]] 对 `lhs_dataset_2000` 的阶段性探针记录：

| 统计量 | 实际 k |
|---|---:|
| `|V|` 加权平均 | 0.0079 |
| 中位数 | 0.0138 |
| P75 | 0.0259 |
| P95（层流分支封顶） | 0.0345 |

该探针仅覆盖 tf=8 s 和单一参数组合，不能直接替代 50 s 全程或定义“等效常数 k”。

## 解释与机制假设

- **已支持的解释**：Brunone 项对尖锐波前和高频成分产生更强耗散，从而改变峰形和到时估计。
- **条件性解释**：传播历史和波形畸变共同造成三种表观速度分离。
- **未证明的假设**：存在可由非线性相位、`c_p(ω)` 或群延迟刻画的严格频散。
- **视角互换（2026-08-02，文献）**：管道领域 TWD 阻尼法（[[Wang2002_瞬变阻尼法泄漏定位]]）
  把缺陷信息放在「阻尼率」里——Brunone 阻尼因此既是需补偿/标定的噪声源（T06 路线），
  也可作为**多周期模态阻尼率特征**（信息源，未探索路线）。见
  [[瞬变波异常检测五类方法综述]]。
- **阻尼率实测（2026-08-03，[[EXP-20260802-001-方法迁移试验混叠分离]]）**：
  lhs_dataset_2000 全量 2000 case 各谐波模态阻尼率 α 与 Σkleak 秩相关 0.87–0.92、
  与 n_frac 0.44–0.51、与 ΣCf 0.17–0.32；无裂缝 intact 对照 α1=0.035。
  结论：阻尼率携带裂缝信息但由滤失主导，宜作**滤失诊断通道**而非独立缝数估计器。

## 证据入口

### 正式实验

- [[EXP-20260730-010-LHS数据集实际生效k实测]] — completed，局部探针
- [[EXP-20260730-020-T04既有波形退化输出证据审计]] — completed
- [[EXP-20260801-001-盲检测协议落地与历史结果复算]] — completed；给出 Re-dependent k 下
  单缝深度偏差 10–20 m 的盲测量，并否定「Brunone 下倒谱匹配成功」的历史表述
- [[EXP-20260806-001-LHS稳态摩阻对照数据集2000]] — completed；与 `lhs_dataset_2000` 同规格的
  steady 平行语料（非逐 case 几何配对）
- [[EXP-20260730-024-T04波形退化指标审计]] — **completed**（2026-08-14）
- [[EXP-20260730-025-旧分辨率与机制强结论证据不足]] — negative result

### 代码与输出

- `analysis/brunone_spacing_effect/run_simulations.py`
- `analysis/brunone_spacing_effect/plot_results.py`
- `analysis/brunone_spacing_effect/plot_tf_dispersion.py`
- Brunone LHS：`output/lhs_dataset_2000/`；steady 对照：`output/lhs_dataset_2000_steady/`
- `analysis/brunone_spacing_effect/probe_realised_k.py`
- `output/analysis/brunone_spacing_effect/metrics_summary.csv`
- `output/analysis/brunone_spacing_effect/fig1_waveform_evolution.png`
- `output/analysis/brunone_spacing_effect/fig2_trend.png`
- `output/analysis/brunone_spacing_effect/fig5_tf_dispersion.png`
- EXP-024：`analysis/brunone_spacing_effect/audit_waveform_degradation.py` → `output/analysis/brunone_spacing_effect/audit_v2/`

### 概念

- [[Brunone非定常摩阻仿真建模]]
- [[双峰分离判据与EST指标]]
- [[瞬变波异常检测五类方法综述]]（TWD 阻尼法视角）
- [[Wang2002_瞬变阻尼法泄漏定位]]

## 已知反例、失败与边界

- STFT 主要证明频率依赖衰减，不直接证明不同频率以不同相速度传播。
- 三种表观速度来自不同特征和传播历程，其差异不是严格频散的直接证据。
- 旧 Rayleigh 热力图已由 EXP-024 的 \(C_v\) 取代；\(C_v\) 只报融合趋势，仍不作最小缝距表。
- 26%/74% 是 onset/peak 操作定义下的表观时延分解，不是唯一物理分解。
- k 与滑溜水、线性胶、交联冻胶的映射尚未由本项目流变数据验证。
- FWHM 在强波形退化时出现假性收窄；EST 在当前矩阵更一致，但尚不能泛化为普适最优指标。
- **Brunone 波形退化不是深度反演失败的主因**（[[EXP-20260806-002]] / [[EXP-20260806-003]]）：
  联合位置 CRB 上 brunone 相对 steady 只膨胀 1.12–1.32 倍，远不足以解释 10–20 m 倒谱偏差。
  Brunone 系数错 10% 的一阶偏差为 0.68–1.05 m；摩阻**形式**互换的非线性 MLE 为 29.6 / 31.9 m
  （一阶线性化 0.6–10 m 低估，不得代替）。倒谱 10–20 m 与形式失配同量级、小于形式失配。

## 待验证 / 下一步

- [ ] 标定 k：`MocConfig.brunone_k_scale` 已可作为待估/失配参数（EXP-20260806-002 引入），
      需把它接入反演管线并给出 k 的可辨识性（当前只作为讨厌参数评估过）。
- [x] 以共同 STFT 参考重新导出可审计频带能量表（[[EXP-20260730-024]]）。
- [x] 统一双峰分离公式、阈值和合成单元测试（\(C_v\)；不作 Rayleigh 墙）。
- [x] 审计 onset 阈值、分析窗、EST 和峰漂移敏感性（锚点四窗重复 277/101 ms）。
- [ ] 复测 tf=50 s 的 k 分布并定义可审计的等效常数 k。
- [ ] 若保留严格频散主张，新增复传递函数、展开相位、多传播距离和群延迟实验。
- [ ] 建立 k 与实际压裂液流变性质的可追踪映射。

## 论文：Paper B

**SPEJ 框架：** 现行正文 [`docs/PaperBC_…`](../../docs/PaperBC_Waveform_Degradation_and_Identifiability_波形退化与可辨识性.md) · vault 转发 [[SPEJ_三篇论文框架]]  
**B 段细目**：[[paper_B_resolvability_waveform_degradation]]

**定位**：量化 Brunone 引起的频率选择性衰减、表观传播变化和波形退化。倒谱旁瓣宽不是本文分辨率基线；严格频散与 Rayleigh 最小缝距表不作为 claim。`docs/paper2_brunone_dispersion.md` 主张已作废。现行正文：[`docs/PaperBC_…`](../../docs/PaperBC_Waveform_Degradation_and_Identifiability_波形退化与可辨识性.md)。

| 论点 | 正式实验 | 数据/图表 | 证据等级 | 限制 | 状态 |
|---|---|---|---|---|---|
| Brunone 导致峰漂移和 EST 展宽 | EXP-024 | audit_v2/fig2，277/101 ms | 强 | 固定 k、D=20 锚点 | 已支持 |
| Brunone 导致频率选择性衰减 | EXP-024 | STFT −44.1 dB vs −12.9 dB | 强 | 规定 k，不是 \(c_p(\omega)\) | 已支持 |
| 时域双峰分离随 k 退化 | EXP-024 | \(C_v\) 热力图 + 单元测试 | 中等 | 不作最小缝距表 | 趋势已支持 |
| 三种表观速度反映不同特征敏感性 | EXP-024 | 11.0/1.3/4.7% | 中等 | 非窄带相速度 | 已支持 |
| Brunone 产生严格频散 | 无 | 无相位/群延迟证据 | 不足 | 条件性未来实验 | 不作为 claim |

**关联主题**：[[T01-倒谱分辨率极限]]、[[T06-频散补偿后向传播]]

**文献（UF 模型、短窗无阻尼 replica、建模误差差分）**：[[00_索引与可迁移方法总表]]
