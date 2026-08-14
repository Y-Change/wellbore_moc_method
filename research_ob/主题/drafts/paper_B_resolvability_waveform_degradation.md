---
type: paper-draft
paper_id: Paper-B
status: evidence-restructuring
topics:
  - T04
  - T01
canonical: true
note: 主张以 docs/PaperBC 2026-08-14 初稿为准；EXP-024 已完成。
---

# Brunone 非定常摩阻下的波形退化

**English working title:** Brunone-Induced Waveform Degradation in Multi-Fracture Water-Hammer Diagnostics

> 本稿写 **T04 波形退化**（频率选择性衰减、峰漂移、表观波速分离）。  
> **不要**把倒谱谱支撑宽度当分辨率墙或本文中心量。近距可分性已由 CRB/MLE 回答，见 [[T01-倒谱分辨率极限]]、[[PaperC_MOC主路径结案]]。
>
> **SPE Journal 框架正文：** 现行合成稿 [`docs/PaperBC_Waveform_Degradation_and_Identifiability_波形退化与可辨识性.md`](../../../docs/PaperBC_Waveform_Degradation_and_Identifiability_波形退化与可辨识性.md)（2026-08-14 初稿）。本页只保留 B 段细目。Vault 转发页 [[SPEJ_三篇论文框架]]。`docs/paper2_brunone_dispersion.md` 的频散/Rayleigh/流体对照主张已作废。

## 1. 中心问题与贡献边界

Brunone 项如何改变井口波形的频带能量、到时和波包宽度？其中哪些是耗散，哪些还不能叫严格频散？

不在本文主张：倒谱有一条与间距无关的信息论分辨极限。倒谱只是低效估计器。

## 2. 倒谱旁瓣宽（废弃口径，一小节说完）

历史上用 \(B_{\mathrm{coh}}\) 推出过约 38 m 的 \(\Delta d_{\mathrm{DR}}\)（steady、DR=80 dB）。那是该变换的旁瓣宽，随 DR 剧烈变化，且盲协议下 5 m 双缝可分。本文不再以它为基线。证据：[[EXP-20260730-025]]、[[EXP-20260801-001]]、[[EXP-20260806-002]]。

## 3. Brunone k×D 波形退化矩阵

现有矩阵包括：

| 维度 | 取值 |
|---|---|
| Brunone k | 0、0.01、0.02、0.05、0.1、0.2 |
| 间距 D | 5、10、20、50、100 m |
| 工况数 | 30 |

数据入口：`output/analysis/brunone_spacing_effect/audit_v2/metrics_v2.csv`（EXP-024）。旧 `metrics_summary.csv` 仅作历史对照。

证据审计：[[EXP-20260730-020]]（现象）· [[EXP-20260730-024]]（公式与数字）。

## 4. 结果 I：峰漂移和能量弥散

### 4.1 已观察结果

随 k 增大，反射峰衰减、展宽、向后漂移并逐步融合。D=20 m、k=0.2 时，EXP-024 给出峰漂移 **277.0 ms**、EST **101.0 ms**（四窗重复）。

候选图：`output/analysis/brunone_spacing_effect/audit_v2/fig1_waveform_evolution.png`、`fig2_peakshift_est.png`。

### 4.2 指标解释边界

- EST 是 E10–E90 的能量时间跨度；D=20 m、k=0.2 时 FWHM 从 k=0.1 的 38 ms 收到 31 ms（假性收窄），EST 从 51 ms 增到 101 ms。
- 约 26%/74% 的分解是 onset 与 peak 操作定义下的表观时延分解，不是“传播频散/波形频散”的唯一物理分解。
- 锚点四窗敏感性已测：峰漂移与 EST 不变。

## 5. 结果 II：频率选择性衰减

共同参考 STFT（Hann，nperseg=128，重叠 120，dB 参考 = 同间距 k=0）下，D=20 m、k=0.2：60–150 Hz **−44.1 dB**，20–60 Hz −25.4 dB，0–20 Hz −12.9 dB。rFFT 对照 60–150 Hz 为 −44.9 dB。旧“>40 dB”在该参考下成立，必须带参考清单。

图：`audit_v2/fig3_stft_common.png`、`fig4_band_energy.png`。证据 [[EXP-20260730-024]]。

## 6. 结果 III：双峰分离退化

合成等幅高斯上 \(C_v\) 随分离度单调；30 工况热力图上小间距、大 k 时 \(C_v\to 0\)（D=20 m：k=0.05 为 0.14，k≥0.1 为 0）。**不**写成最小可分辨缝距。旧 `fig4_rayleigh_heatmap.png` 不进主文。图：`audit_v2/fig6_cv_heatmap.png`、`fig7_cv_unit_test.png`。

## 7. 表观速度与严格频散的边界

`a_f0`、`a_onset` 和 `a_peak` 随 k 的变化幅度不同，说明不同特征提取方法在波形退化后不再由单一标量稳定描述。但三者分别受全井多周期、阈值触发和峰形变化影响，不能直接等价为不同频带的相速度。

现阶段结论：

- 已观察：高频选择性衰减、峰漂移、EST 展宽、表观速度指标分离；
- 解释：传播历史和波形畸变共同影响到时估计；
- 未证明：非线性相位、频率相关相速度或群延迟意义上的严格频散。

若未来恢复 strict dispersion claim，需另做复传递函数、展开相位、多传播距离和群延迟实验。

## 8. 讨论框架

倒谱旁瓣宽、盲 F1、时域双峰分离是三件不同的事，不要合成「一条分辨率曲线」。本文只主张波形退化观察；定位极限用 CRB/MLE。

工程外推：当前 k 是固定 monkey-patch，不是 Re-dependent 现场模型；不能把固定-k 表写成某类压裂液的普适最小间距。

## 9. 推荐图表

1. Brunone 波形演化和 EST/峰漂移  
2. 共同参考 STFT 与频带能量表  
3. 统一公式后的双峰分离热力图  
4. 三种表观速度指标与适用边界  

不要做「谱支撑 vs 盲检测」对照图当主图。

## 10. Claim–evidence 状态

| 论点 | 证据 | 状态 |
|---|---|---|
| 历史倒谱匹配存在真值泄漏 | EXP-20260801-001 | 已确认并修复 |
| Brunone 单缝深度偏差约 10–20 m | EXP-20260801-001 | 已支持（容差扫描） |
| Brunone 导致峰漂移和 EST 展宽 | EXP-024 | 已支持（277/101 ms，四窗重复） |
| Brunone 导致频率选择性衰减 | EXP-024 | 已支持（−44.1 vs −12.9 dB） |
| 双峰分离随 k 退化 | EXP-024 | 趋势已支持；不作最小缝距表 |
| Brunone 导致严格频散 | 无 | 不作为当前 claim |
| 倒谱谱支撑是信息论墙 | EXP-002/003/025 | **否证，不要写** |

## 11. 投稿前门槛

- [x] STFT 数字可由机器表复算（`audit_v2/band_energy.csv`）；
- [x] 双峰分离公式、代码、阈值和热力图一致（\(C_v\)，不作 Rayleigh 墙）；
- 全文不把频率选择性衰减写成严格频散；
- 全文不以倒谱旁瓣宽当分辨率。
- 现行完整初稿在 Paper BC，不在本页平行维护两套正文。

