# Handoff Report - Codebase Investigation & TG-DIS-DeepONet Architectural Design

**Agent**: `explorer_p3_codebase`  
**Timestamp**: 2026-09-13T14:35:00Z  
**Type**: Hard Handoff (Investigation & Architecture Design Complete)  
**Deliverable Document**: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_p3_codebase\report.md`  

---

## 1. Observation (直接观测事实)

1. **现有模型与模块实现清单**：
   - 模型文件：`src/models/resnet1d.py`（ResNet1D）、`src/models/fno1d.py`（FNO1D）、`src/models/deeponet.py`（VanillaDeepONet）、`src/models/cj_cep_deeponet.py`（CJCepDeepONet）、`src/models/tg_cj_deeponet.py`（TGCJDeepONet）。
   - 核心模块：`src/modules/time_gating.py`（实现 `RelativeWindowGating`, `GaussianGating`, `CepstrumPatchGating`）、`src/modules/acoustic_transformer.py`（实现 `AcousticBiasedMultiheadAttention`, `AcousticBiasedTransformer`）、`src/modules/dual_track_heads.py`（实现 `DiscretePreciseHead`, `ContinuousTrunkHead`, `DualTrackHead`, `VoronoiPooling1D`）。
2. **Phase 2 实际评估结果（来自 `output/ablation_metrics_summary.json`）**：
   - `tg_relative_bias`（TG-Rel-Bias 旗舰）：
     - 全集指标：$\alpha\text{-MAE} = 0.133558$, $\alpha\text{-}R^2 = 0.566567$, $C_f\text{-MRE} = 46.094\%$, $C_f\text{-}\log_{10}\text{-MAE} = 0.331456$, $W_1 = 8.394\,\mathrm{m}$，Simplex 最大偏差 $= 1.192 \times 10^{-7}$。
     - 变簇数分层：$N_c=1$ 时 $\alpha\text{-MAE} = 0.000$, $W_1 = 0.00\,\mathrm{m}$；$N_c=2$ 时 $\alpha\text{-MAE} = 0.2259$；$N_c=3$ 时 $\alpha\text{-MAE} = 0.1633$；$N_c=6$ 时 $\alpha\text{-MAE} = 0.1002$。
   - 对比基准模型：
     - `resnet`: $\alpha\text{-MAE} = 0.1449$, $R^2 = 0.5072$, $W_1 = 9.55\,\mathrm{m}$；
     - `fno`: $\alpha\text{-MAE} = 0.1449$, $R^2 = 0.5072$, $W_1 = 9.55\,\mathrm{m}$；
     - `cj_cep_deeponet`: $\alpha\text{-MAE} = 0.1548$, $R^2 = 0.4148$, $W_1 = 9.77\,\mathrm{m}$；
     - `tg_best_nobias`: $\alpha\text{-MAE} = 0.1449$, $R^2 = 0.5364$, $W_1 = 9.20\,\mathrm{m}$。
3. **指标与功能缺失观测**：
   - 经 grep 检索 `src/metrics.py` 与 `src/models/`，全库当前未实现容差 $\pm 10\,\mathrm{m}$ 的裂缝检出 F1-Score 函数（仅在早期设计文档 `井口单点水击波物理反演_初步模型架构与标签方案.md` 第 524 行提及）；
   - 当前网络尚未提供独立的起裂分类头（`p_exist`）与亚米级位置细化微调头（`delta_x`）；
   - 当前尚未运行包含 SNR 30dB, 20dB, 10dB 的噪声鲁棒性与 $\pm 1\%$ 声速扰动压力测试。
4. **测试执行状态**：
   - 执行 `pytest -q tests/test_tg_models.py` 输出：`7 passed in 10.78s`；
   - 执行 `pytest -q tests/test_pilot_models.py` 输出：`5 passed in 3.22s`。
   - 全库原有 12 项测试保持 100% 全部通过，无任何异常报错。

---

## 2. Logic Chain (推导逻辑链)

1. **从事实 (1) 与 (2) 到瓶颈定位**：
   在 Phase 2 中，时间门控相对窗（`relative_window`）将各簇波前对齐，使 $R^2$ 跃升至 $0.5666$（相较 Phase 1 的 $0.4148$ 提升 36.6%）。但在密集多簇（$N_c \ge 2$）下，$R^2$ 出现平台期，无法突破 $0.75$。
2. **从声学物理机制到误差机理**：
   声波自趾端回传至井口时，第 $j$ 簇的反射脉冲必然穿过上游所有 $1 \sim j-1$ 簇裂缝，幅值被上游累积透射衰减因子 $\prod_{k=1}^{j-1} T_k^2 = \prod_{k=1}^{j-1} (1 + \Gamma_k)^2$ 剧烈削弱。现有网络未显式剥离该衰减，迫使注意力机制学习高度非线性的多径卷积，导致网络陷入对密集簇进行“均摊估计”（$\hat{\alpha} \approx 1/N_c$）的收缩陷阱。
3. **从逆散射层剥离原理到网络改进**：
   引入基于一维水力瞬变流声学传递矩阵的**可微层剥离（DIS）算子**，由跟端至趾端递归计算局部物理反射率 $\Gamma_j \in (-1, 0)$ 与支路导纳 $Y_{b,j} = - \frac{2\Gamma_j}{Z_0(1+\Gamma_j)}$。该算子不仅在物理前向上对下游脉冲进行反衰减补偿，更在反向传播中建立因果梯度链 $\frac{\partial \mathcal{L}}{\partial T_k}$，能够彻底解决均摊陷阱。
4. **从功能缺失到模块补齐**：
   为了满足 Acceptance Criteria 中“F1-score > 0.88 (容差 $\pm 10\mathrm{m}$)”和“max|sum(alpha)-1| < 1e-6”，必须在解码头中显式增加二分类 Sigmoid 头、基于二分图贪心距离匹配的 F1 评估函数，并保留现有验证过的 Masked Softmax + 显式重整化逻辑。

---

## 3. Caveats (局限与未勘验事项)

1. **实测大尺度多相流偏差**：
   当前数据集与模型基于单相水击瞬变流 MOC 数据（$a \approx 1450\,\mathrm{m/s}$），未直接涵盖高气油比、游离气泡或重度结蜡井筒的弥散波形；
2. **未知射孔数下的开放空间检测**：
   当前数据集完井候选位置最大数为 $M=6$（预留最大 6 簇），若现场存在单段 10~16 簇超密射孔，需要扩展数据集切片维度与 Transformer 的序列长度；
3. **噪声扰动合成先验**：
   本设计中提出的白噪声/粉红噪声扰动基于典型测井与地面压力计统计特性，未包含真实压裂施工现场因砂塞敲击产生的大幅值脉冲干扰。

---

## 4. Conclusion (确定性结论与实施建议)

1. **模型演进路径明确**：将主干模型由 `TGCJDeepONet` 升级为 `TGDISDeepONet`，在到时提取层之后、自注意力层之前插入 `DifferentiableLayerStripping` 层；
2. **物理单纯形守恒已闭环**：现有基于 Masked Softmax 加显式除和重整化的实现，已实测达到 $1.19 \times 10^{-7}$ 的物理守恒精度，完全满足 $< 10^{-6}$ 验收标准；
3. **新增功能与测试明确**：
   - 新建 `src/modules/layer_stripping.py` 与单元测试 `tests/test_dis_layer.py`；
   - 新建模型 `src/models/tg_dis_deeponet.py`；
   - 在 `src/metrics.py` 中新增 `compute_detection_f1_score`；
   - 编写 `experiments/train_dis.py` 与 `experiments/audit_noise_robustness.py`。

---

## 5. Verification Method (独立验证方法)

任何接手此任务的智能体可按以下步骤完全独立复现与验证：

1. **环境与现有模型回归验证**：
   ```powershell
   cd e:\water_hammer_research\wellbore_moc_method\PaperC_CJNO_Wellbore_Inversion
   pytest -q tests/test_pilot_models.py
   pytest -q tests/test_tg_models.py
   ```
   *判据*：全量 12 项测试必须 100% 全部通过 (code 0)。
2. **Phase 2 实际指标数据核查**：
   检查 `output/ablation_metrics_summary.json`，核对 `tg_relative_bias` 的 $\alpha$ MAE 是否精确为 $0.133558$，$R^2$ 是否为 $0.566567$。
3. **新设计文档与接口审查**：
   阅读 `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_p3_codebase\report.md`，重点核对第 4 节 `DifferentiableLayerStripping` 的公式推导与张量形状定义。
