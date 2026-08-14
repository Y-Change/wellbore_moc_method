---
type: paper-draft
paper_id: Paper-A
status: first-draft
topics:
  - T02
  - T03
canonical: true
---

# 级联裂缝网络中的水击波拓扑衰减与非局部首缝响应

**English working title:** Topology-Controlled Water-Hammer Attenuation and Nonlocal Primary-Fracture Response in Cascaded Fracture Networks

> 本稿是 T02/T03 的 vault 骨架。**现行正文**在 [`docs/PaperA_Topology-Organized_Cepstral_Response_倒谱拓扑组织与首缝非局部性.md`](../../../docs/PaperA_Topology-Organized_Cepstral_Response_倒谱拓扑组织与首缝非局部性.md)（2026-08-14 初稿：方法节取自旧稳态稿，数字取自 EXP-20260814-001）。
>
> 现阶段主张“拓扑组织的衰减”和“非局部首缝响应”；相长/相消路径、反常扩散和非德拜弛豫仍是待验证解释。Vault 转发页 [[SPEJ_三篇论文框架]]。

## 1. 中心问题与贡献边界

本文回答两个相互关联的问题：

1. 后续裂缝的相对倒谱响应主要随物理距离还是裂缝节点序号变化？
2. 下游裂缝网络是否会改变首缝的表观倒谱响应，使孤立单缝近似失效？

当前可安全主张：

- 在准稳态达西摩阻的 420 个主工况中，裂缝序号轴比物理距离轴呈现更紧凑的衰减曲线组织。360 个 \(n\ge 3\) 工况：Pow(idx) 中位 \(R^2=0.987\)，Pow(dx) 0.908，\(\Delta R^2\) 中位 0.080，358/360 拓扑更优（[[EXP-20260814-001-T02工况级拓扑统计重算]]）。
- 首缝 `P_2D` 随首缝深度、裂缝总数和间距普遍非单调（70/70、60/60、41/42 条序列），说明首缝观测量不是局部裂缝属性的简单单值函数。

当前不可作为已证实贡献：

- 单节点固定透射损失已被定量推导；
- 展宽指数 `β<1` 已证明反常扩散或非德拜弛豫；
- 已经唯一识别相长/相消反射路径。

## 2. 模型与数据口径

采用一维 MOC 瞬态水击模型，并以准稳态 Darcy 摩阻作为传播基线。裂缝节点使用集总柔度与滤失边界。

主矩阵口径：

| 维度 | 取值 | 数量 |
|---|---|---:|
| 首缝深度 `X1` | 2000、2500、3000、3500、4000、4500 m | 6 |
| 间距 `S` | 10、20、…、100 m | 10 |
| 总裂缝数 `n` | 2、3、…、8 | 7 |
| steady 工况 | `6×10×7` | **420** |

证据审计：[[EXP-20260730-019-T02-T03既有输出证据审计]]。

数据入口：

- `output/analysis/decay_regression/03_extracted_peaks_csv/decay_table.csv`
- `output/analysis/decay_regression/04_collapse_and_scaling_pidx/`
- `output/analysis/decay_regression/06_first_frac_energy/`

## 3. 结果 I：拓扑序号组织的衰减

### 3.1 已观察结果

以物理距离 `Δx` 为横轴时，不同间距工况形成发散曲线族；以裂缝序号 `frac_idx` 为横轴时，曲线明显收紧。现有 steady 签名图可作为主图候选：

`output/analysis/decay_regression/04_collapse_and_scaling_pidx/steady_collapse_vs_divergence_x1_4000.png`

### 3.2 当前解释

这一现象与“裂缝节点的反射、透射和分流对级联响应具有重要作用”的解释一致。在当前等参数、等间距基线内，节点数量比单纯传播距离具有更高描述力。

### 3.3 尚缺证据

- 逐 case 的 Pow(idx)/Pow(dx) 统计表、残差、`ΔR²` 与置信区间 — **已由 EXP-20260814-001 完成**；
- 非等间距和非等参数裂缝下的稳定性；
- 统一 n=5 的 Cf/Kleak **全网格**（现仅 42 格子集；k 不普适）；
- 拓扑指数与单节点等效系数的理论关系。

对应 planned 实验：[[EXP-20260730-022-T02拓扑统计与Cf-Kleak统一n5复核]]。

## 4. 结果 II：首缝非局部响应

### 4.1 已观察结果

现有 steady 数据显示三类非单调变化：

- 固定 `n=2, S=50 m` 时，`P_2D` 随 `X1` 先降后升；
- 固定 `X1=2000 m, S=100 m` 时，增加下游裂缝数可使首缝响应回升，`n=8` 可高于 `n=2`；
- 固定 `X1=2000 m, n=8` 时，`S≈30–40 m` 附近出现低谷，`S=90–100 m` 明显回升。

候选三面板主图：

- `output/analysis/decay_regression/06_first_frac_energy/steady_1_vs_x1_2d.png`
- `output/analysis/decay_regression/06_first_frac_energy/steady_2_vs_ntotal_2d.png`
- `output/analysis/decay_regression/06_first_frac_energy/steady_3_vs_spacing_2d.png`

### 4.2 当前解释

非单调性表明首缝表观峰受到下游网络反馈，单调屏蔽和孤立单缝模型不足。多次反射、能量分流、相位叠加以及倒谱窗耦合均是可能机制。

### 4.3 尚缺因果验证

现有输出没有复谱相位、路径截断或节点消融，因此正文应使用“与全局相干耦合一致”，而不是“已经证明相长/相消干涉”。

对应 planned 实验：[[EXP-20260730-023-T03首缝路径相位与参数消融]]。

## 5. 讨论框架

### 5.1 相对后缝响应与首缝绝对响应必须分开

- T02 使用 `α_i=P_i/P_1`，描述后缝相对响应；
- T03 使用首缝绝对 `P_1`，关注下游网络对首缝观测值的反馈。

归一化会固定首缝为 1，因此两者不能用同一种“能量衰减”语言混写。建议统一称“normalized cepstral response”和“apparent primary-fracture cepstral peak”。

### 5.2 物理解释等级

| 层级 | 当前状态 |
|---|---|
| 距离轴发散、序号轴收紧 | 已观察 |
| 首缝三参数非单调 | 已观察 |
| 节点散射是重要控制因素 | 与数据一致的解释 |
| 特定路径相长/相消 | 待消融验证 |
| 反常扩散/非德拜弛豫 | 假设，不作为主贡献 |

### 5.3 对反演的含义

当前结果挑战的是独立单缝、单次散射或局部可分离响应假设，而不是线性波动方程本身的线性叠加原理。反演模型需要表达节点级联和全局网络条件依赖。

## 6. 推荐图表

1. 模型与级联路径示意图；
2. `Δx` 发散 vs `frac_idx` 坍缩签名图；
3. Pow(idx)/Pow(dx) 统计比较与置信区间；
4. Cf/Kleak 统一 n=5 稳健性图；
5. 首缝 `X1/n/S` 三面板；
6. 路径开启、细间距和参数消融图。

展宽指数图降为补充材料，除非后续建立明确理论映射。

## 7. Claim–evidence 状态

| 论点 | 证据 | 状态 |
|---|---|---|
| steady 主矩阵为 420 工况 | `decay_table.csv`；EXP-019 | 已审计 |
| 拓扑轴比距离轴更紧凑 | collapse 图；EXP-019 | 现象已支持 |
| Pow(idx) 优于 Pow(dx) | EXP-20260814-001，360 工况 | 已支持 |
| 首缝响应受下游网络影响 | 三组非单调图；70/70、60/60、41/42 | 已支持 |
| 节点参数变化下标度稳定 | EXP-022；n=5 子集 42 格 | 排序稳健、k 不普适；全网格未补 |
| 相位干涉解释极小值 | EXP-023 | planned |

## 8. 投稿前门槛

- [x] 420 case 统计表、CI 与残差（[[EXP-20260814-001-T02工况级拓扑统计重算]]）
- [ ] 完成统一 n=5 Cf/Kleak 全网格复核（现仅 42 格）
- [ ] 至少完成 n=1、逐缝开启和一个细间距极小值复现
- 若相位未闭环，标题和结论继续使用“nonlocal response/network coupling”（现行 Paper A 初稿已如此）
