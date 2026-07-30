---
type: experiment
experiment_id: EXP-20260729-001
status: failed
topic: T07
tags:
  - experiment
  - negative-result
  - dccdm
  - diffusion
---

# EXP-20260729-001 DCCDM overfit16 稀疏生成失败

## 基本信息

- 实验编号：EXP-20260729-001
- 日期：2026-07-29
- 状态：failed
- 研究主题：[[T07-DCCDM代理模型]]
- 论文项目：无独立论文主页

## 1. 实验目的

检验条件扩散 DCCDM 在固定 16 个训练样本上是否能够同时降低 epsilon 噪声预测损失，并从纯噪声逆采样得到稀疏、位置正确的 Cf 加权裂缝图。

## 2. 研究假设

- H1：若训练链路和扩散目标适配当前任务，16 样本应能通过 sample-quality memorization gate。
- H2：epsilon 诊断损失下降应伴随 raw map 优于全零基线、低假峰率和较高事件 F1。

## 3. 实验配置

| 参数 | 数值 | 说明 |
|---|---:|---|
| 样本数 | 16 | 固定训练/诊断样本 |
| 序列长度 | 4096 | 30 s 时间轴 |
| Epoch | 200 | 无观测噪声 |
| Diffusion steps | 100 | 线性 beta 1e-4 到 0.02 |
| Base dim | 16 | 条件 U-Net |
| 条件 | 2 通道 | 标准化正 quefrency 倒谱 + 坐标 |
| 随机种子 | 42 | 可复现采样 |

## 4. 代码与命令

- 代码入口：`neural_operator/train_diffusion.py`
- 配置文件：run 内 `manifest.json`
- 运行命令：见 `output/dccdm/runs/20260729-234908-964216-overfit16-seed42-overfit16-v1/manifest.json`
- Git commit / 工作树状态：run manifest 记录为 dirty working tree
- 随机种子：42

## 5. 数据与输出

- 输入数据：`output/lhs_dataset/data/`
- 输出目录：`output/dccdm/runs/20260729-234908-964216-overfit16-seed42-overfit16-v1/`
- 核心结果文件：`gate_result.json`、`evaluations/metrics_summary.json`、`evaluations/final_predictions.npz`
- 关键图片：`plots/training_loss.png`、`evaluations/figures/`

## 6. 结果

### 定量结果

| 指标 | 结果 | 基线 | 变化 |
|---|---:|---:|---:|
| epsilon 诊断损失 | 约 0.00681 | 1.5284 | 下降约 99.55% |
| raw map MAE | 10.811 | 全零 0.245 | 约差 44 倍 |
| micro precision | 0.00363 | — | 大量假峰 |
| micro recall | 1.000 | — | 由密集假峰造成 |
| micro F1 | 0.00723 | 全零 0 | 不可用 |
| 平均检测峰数 | 约 1119.9/样本 | 真实 1–6 | 基底塌缩 |
| FP | 17,854 | — | 严重 |

### 定性现象

- raw 预测在全时间域保持正基底，约 8–16，而非背景 0 加少数稀疏峰。
- 阈值 1 的后处理无法消除基底，因为所有输出均超过阈值。
- 低 epsilon loss 与实际逆采样质量完全脱钩。

## 7. 分析与解释

- 是否支持假设：不支持 H1/H2。
- 观察：noise-prediction 目标已优化，但 sample-quality gate 失败。
- 解释：100-step 线性 schedule 的终点 `alpha_bar_T≈0.3636`，仍保留约 60% 信号，而推理从纯 Gaussian 开始，存在训练—采样端点不匹配；同时稀疏背景目标未被 epsilon MSE 显式保护。
- 混杂因素：Gaussian 目标原 `sigma=0.15 s` 过宽；扩散来自图像任务的修正尚未在本项目验证。

## 8. 异常与失败

- 这是有效负结果，不应通过提高后处理阈值或继续全量训练掩盖。
- gate 正确阻止了 6000 样本全量扩散训练。

## 9. 结论

> epsilon 损失下降 99.55% 不能证明 DCCDM 能生成稀疏裂缝图；当前扩散主线暂停，先建立确定性直接反演基线。

## 10. 下一步

- [x] 暂停全量扩散训练。
- [x] 设计 PhaseNet 风格 P0-A 原始波形直接检测基线。
- [ ] 仅在确定性模型通过 sample-quality gate 后，再考虑 zero-terminal-SNR、v-prediction 和 Min-SNR 修复扩散。

## 11. 可用于论文的证据

- 可支撑论点：扩散去噪训练损失不是极稀疏物理反演的充分模型选择指标。
- 可使用图表：失败样本 raw/postprocessed 图与 loss 曲线。
- 尚缺证据：修复后扩散与直接监督基线的公平比较。
