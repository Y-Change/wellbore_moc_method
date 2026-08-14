---
type: literature-note
citation_key: PINN_FPINO_group
title: PINN and filtering physics-informed neural operator for hydraulic transients
authors: Waqar et al. 2022; Ye et al. 2022; Li et al. 2025
year: 2022-2025
status: read
topics: [T04, T07]
tags:
  - literature
  - pinn
  - fno
  - surrogate
---

# PINN / FPINO：用代理模型处理未建模阻尼（三篇合记）

三篇都在待读根目录，主题都是「MOC 贵或边界不全时，用物理+数据重构/代理」，合记以免重复。定位算法本身不在这三篇里。

---

## A. Waqar et al. 2022（IAHR）PINN 学水击波场

- 训练数据来自 MOC。普通 NN 只喂边界学不会水击；PINN 把 1D 波动方程 + BC 放进损失。
- 优化器：L-BFGS 与 MOC 符合好；Adam 收敛慢。
- 对信号时长/带宽敏感：更长或更宽频需要更密网络。
- 额外测点位置会影响约束效果。
- 当时尚未把摩擦与 FSI 放进 PINN（文末列为延伸）。

**迁移**：PINN 当 MOC 代理时，高频/长时程要加宽网络；不要默认 Adam。摩擦未进物理项 ⇒ 不能当 Brunone 代理。

---

## B. Ye, Do, Zeng, Lambert 2022（Water Research 221: 118828）

**问题**：现场 BC/IC 不全，传统 MOC 跑不起来；UF 与 VE 难标定。

**结构**：PINN 输出 \((h,q)\)；损失 = 数据项 + PDE 项（标准连续+动量，**不含**粘弹性/UF）。配点用自动微分。

**传感器**：靠近边界的测点更有用。

**关键数值（阻尼失配）**：观测由粘弹性 MOC 生成，PINN 物理仍是无阻尼方程。

| \(w_f\)（PDE 权重） | 相对 L2 |
|---|---:|
| 0（纯 ANN） | 最大 |
| 0.1 | 1.70% |
| 0.01 | 0.54% |

结论：物理模型不完整时，**减弱 PDE 惩罚、让含阻尼的数据主导**，才能跟上真衰减。

实验与管网数值也做了，要点相同：未知边界下仍可重构；测点靠近边界更好。

**迁移到 Brunone**：若 FNO/PINN 的物理项是 steady 方程、数据是 Brunone，应减小物理损失权重，而不是加大。这与「FNO 主域精度未过关」的可能原因一致（假设，不是本项目新实验）。

---

## C. Li et al. 2025 FPINO（Filtering PINO）

**问题**：DeepONet/PINO 把全时段 BC/观测塞进分支网，但 \(t_s\) 处的状态只依赖 CFL 锥内的信息；其余是物理上的无关输入。在线反演还面临未建模阻尼（文中点名 UF、VE）。

**正演数据**：MOC。控制方程含 Darcy–Weisbach；VE 用 Kelvin–Voigt 蠕变。

**信息滤波**：按 MOC 网格，\(t_{b,i}=t_s-d_i/a\)；权重在锥内近 1、锥外近 0。为未建模阻尼放宽走时，\(\beta=1.2\)。

**混合损失**：

\[
L=w_f L_{\mathrm{PDE}}+w_{\mathrm{sim}}L_{\mathrm{simulated}}+w_{\mathrm{real}}L_{\mathrm{real}}
\]

\(L_{\mathrm{real}}\) 用来吸收数值模型缺的耗散。实验室铜管：用 P1、P5 作 BC 跑 MOC 得模拟集，与实测的小差异归因于未建模阻尼、参数误差与测量噪声；降采样到 500 Hz。

**迁移**：

- 神经算子输入不要堆整条 50 s 时程；按特征线只喂因果窗口。
- \(\beta>1\) 专门为 UF/频散走时误差而设——Brunone 数据上可把 \(\beta\) 当作超参扫。
- 三项损失分离：错误摩阻模型时降低 \(w_f\)，保留 \(L_{\mathrm{real}}\) 或 \(L_{\mathrm{sim}}\)（若 sim 已是 Brunone MOC）。

---

## 合用建议

1. 代理模型：物理项与数据摩阻一致，或按 Ye 的办法降 \(w_f\)。
2. 算子输入：按 FPINO 做 CFL 滤波。
3. 这三篇解决的是**重构/正演加速**，裂缝定位仍应接到 MLE/MFP/倒谱，而不是指望 PINN 直接输出缝深。
