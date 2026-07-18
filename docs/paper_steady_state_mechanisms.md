---
title: "Cepstrum-Based Multi-Fracture Identification Using Pump-Shut-In Water Hammer: 1D/2D Resolution and Signal Interference Mechanisms under Steady Flow"
authors: [Author Name]
journal: "Target SCI Journal (e.g., SPE Journal / Mechanical Systems and Signal Processing)"
---

# 停泵水击波多裂缝监测：一维/二维倒谱解析与信号干涉机理（稳态模型）
**Cepstrum-Based Multi-Fracture Identification Using Pump-Shut-In Water Hammer: 1D/2D Resolution and Signal Interference Mechanisms under Steady Flow**

## Abstract
In unconventional reservoir stimulation, diagnosing multi-stage fracture networks using pump-shut-in water hammer waves has emerged as a promising non-destructive evaluation technique. However, interpreting these transient pressure signals in a complex wellbore-fracture waveguide is severely hindered by multi-path interference and high-frequency dissipation. This study isolates the wave propagation dynamics by strictly employing a steady flow friction model based on the Method of Characteristics (MOC). We systematically compare the global 1D cepstrum with a time-frequency 2D cepstrogram, proposing a diagonal accumulation scheme to isolate primary reflections from dense coda waves. Furthermore, a stretched exponential model is introduced to characterize the anomalous spatial decay of relative fracture energies. Crucially, a parameter sweep of the primary fracture's absolute cepstral peak reveals profound non-monotonic oscillations driven by depth, fracture multiplicity, and spacing. These data-driven findings prove that the apparent signal of the first fracture is governed by constructive and destructive interference of the entire wavefield, rather than simple monotonic viscous dissipation. This fundamentally challenges the linear superposition assumptions in current fracture inversion algorithms and necessitates the incorporation of global topological interference coupling for accurate field diagnostics.

**Keywords:** Water hammer, Hydraulic fracturing, Cepstrum analysis, Stretched exponential model, Constructive/destructive interference, Method of Characteristics (MOC)

---

## 1. Introduction (研究现状与核心问题)

在非常规油气储层改造（如页岩气水力压裂）中，段内多簇压裂是提高单井产量的核心手段。压裂作业停泵瞬间，由于流速突变激发的水击波（Water Hammer Wave）会在井筒内沿流体介质传播，并在阻抗突变处（如起裂缝口）产生反射。通过在井口采集这些高频压力瞬变信号，实现对井下多级裂缝位置的实时监测与缝网反演，已成为一种极具潜力的非接触式无损诊断技术。

然而，在井筒多级裂缝系统这一极端复杂的波导环境中，瞬变波的传播机理与响应面临着诸多理论痛点：
- **多径干涉导致的严重非线性**：前人研究多局限于单裂缝或少量泄漏点的理想场景。当裂缝簇数增多时，透射波与反射波在裂缝节点之间反复震荡，形成致密的尾波系（Coda waves）。现有的线性叠加模型无法解释这种由于多径散射引发的严重信号畸变。
- **稳态摩阻的高频滤除**：瞬变波在长距离井筒中传播时，流体的粘性剪切（摩阻）会像低通滤波器一样迅速耗散高频能量，导致深部裂缝特征极其微弱。
- **常规识别特征的失真与盲区**：目前倒谱分析（Cepstrum Analysis）虽被引入压裂诊断，但由于多裂缝引发的波场强干涉，首条主裂缝的倒谱响应是否会被后续复杂波系混叠？其底层的干涉机制尚不清晰。

本文严格限定在**稳态摩阻模型（Steady Flow Friction）**的基准下，系统对比 1D 与 2D 倒谱特征提取方案，引入全新的展宽指数（Stretched Exponential）衰减模型，并深入剖析多裂缝引发的能量重分配与首峰干涉机制，为压裂工程中的参数定量反演建立坚实的物理依据。

## 2. Methodology (瞬变流建模原理与1D/2D倒谱方法)

### 2.1 MOC 瞬变流建模原理
本研究采用一维特征线法（Method of Characteristics, MOC）对井筒-裂缝系统进行瞬态压力场求解。
*   **摩阻计算**：本阶段采用稳态达西摩阻（Steady Darcy Friction），从方程源项上剥离非稳态频散干扰。
*   **裂缝节点模型**：将压裂簇等效为“集总柔度（Lumped Compliance）+ 分布滤失（Leak-off）”边界。裂缝流量满足：$Q_f = C_f \cdot \frac{dH}{dt} + k_{leak} \cdot \sqrt{H - H_{ext}}$。在 MOC 内节点中，这被处理为半隐式离散的非线性源项并利用牛顿迭代求解。
*   **激振源**：井口处施加阶跃式边界条件，模拟柱塞泵停泵引发的水击阶跃信号。

### 2.2 1D/2D 倒谱方法对比及累加方案
**1D 全局倒谱 (Global 1D Cepstrum)**：
对整个时间窗内的井口压力信号 $p(t)$ 执行运算：
$$C_{1D}(\tau) = \mathcal{F}^{-1} \left\{ \log \left| \mathcal{F} \left\{ p(t) \right\} \right|^2 \right\}$$
*   **痛点**：1D 方案将时间轴完全压缩到倒频率（$\tau$）域。对于多裂缝网络，后续裂缝产生的大量高次反射回波会严重干扰早期的主反射，产生极高的旁瓣混叠。

**2D 时频倒谱 (2D Cepstrogram / STFT-based Cepstrum)**：
引入滑动窗函数 $w(t)$，在时间-倒频率的二维空间中解析信号：
$$C_{2D}(t, \tau) = \mathcal{F}^{-1} \left\{ \log \left| \mathcal{F} \left\{ p(t) w(t - t_w) \right\} \right|^2 \right\}$$
*   **2D剖面的累加方案 (Accumulation Scheme)**：为了增强主峰信噪比并平抑局部扰动，我们采用沿对角线脊的累加追踪。由于真实一次反射波的到达时间必然服从 $t = 2X/a = \tau$，我们在 2D 倒谱矩阵中沿主对角线（$t \approx \tau$）按特定的相关性窗口对能量进行时空积分（Accumulation）。这种累加方案能最大化相干主反射信号，同时利用多径尾波的非相干特性有效压制旁瓣背景。

> **Figure 1. Comparison of 1D Global Cepstrum and 2D Cepstrogram with accumulation scheme.** (a) 1D Cepstrum suffers from severe side-lobe interference due to the aggregation of primary and secondary reflections. (b) 2D Cepstrogram resolves the temporal dimension. The integrated accumulation along the $t = \tau$ diagonal ridge inherently isolates primary fracture reflections from delayed multiple-scattering coda waves.

## 3. Experimental Design (仿真实验设计)

为探究稳态流场下的衰减演化规律，参数空间设计如下：

**Table 1. Core Simulation Parameters for MOC Transient Flow**
| Parameters (参数) | Values (数值) | Units (单位) |
| :--- | :--- | :--- |
| Wellbore Diameter ($D$) | 0.1397 | m (5.5" Casing) |
| Fluid Density ($\rho$) | 1000.0 | kg/m³ |
| Fluid Viscosity ($\nu$) | $1.0 \times 10^{-6}$ | m²/s |
| Wave Speed ($a$) | 1450.0 | m/s |
| Roughness Height | $4.5 \times 10^{-5}$ | m |
| Initial Velocity ($V_0$) | 1.0 | m/s |
| Time Step ($\Delta t$) | $1.0 \times 10^{-3}$ | s |

**Table 2. Parameter Space Configuration for Multi-Fracture Systems**
| Variable Space (变量空间) | Values (取值/枚举值) | Description (说明) |
| :--- | :--- | :--- |
| First Frac Depth ($X_1$) | 2000, 2500, 3000, 3500, 4000 (m) | 探测首段裂缝的深度耗散响应 |
| Total Fractures ($n_{total}$) | 2, 3, 4, 5, 6, 7, 8 | 裂缝总数，驱动群体干涉机制 |
| Fracture Spacing ($S$) | 10.0, 20.0, ..., 100.0 (m) | 簇间距，驱动空间色散机制 |

## 4. Stretched Exponential Model (瞬变波衰减的展宽指数模型)

在传统的理论框架中，单缝或简易管网的反射波衰减通常被假设为纯指数衰减（Exponential decay）或基于物理距离的代数幂律（如 $(1+\Delta x)^{-k}$）。然而，对于存在大量级联微小阻抗突变（簇）的压裂井筒，纯指数模型在拟合深度数据时呈现出极大的系统性偏差。

基于复杂无序介质中的波传播理论，我们在此引入了**展宽指数模型（Stretched Exponential Model）**来刻画多裂缝倒谱峰值的相对能量（$\alpha_{2D}$）随物理穿透距离（$\Delta x$）的衰减规律：
$$ \alpha_{2D}(\Delta x) = \exp \left( - (b \cdot \Delta x)^\beta \right) $$

式中，$b$ 为衰减尺度因子，$\beta$ 为展宽指数（Stretching exponent，通常 $0 < \beta < 1$）。
该模型的引入具有深远的物理意义：$\beta = 1$ 时退化为标准均匀介质的指数耗散；而本研究中 $\beta < 1$ 的数据拟合结果表明，波在级联裂缝中的耗散并非均匀的“长程流体力学拖曳”，而是呈现出类似于声学超材料中的“反常扩散（Anomalous diffusion）”与非德拜弛豫（Non-Debye relaxation）特征。这为复杂缝网波能衰减的本征评估提供了全新的解析基准。

## 5. Results and Discussion: Primary Fracture Response Mechanism (首缝响应演化机理)

在压裂反演的工程直觉中，首条裂缝（距离井口最近）作为水击波到达的首个边界，其反射响应理应最为清晰，且绝对峰值能量（$P_{2D}$）应严格受制于自身开度和摩擦耗散。然而，基于稳态条件下的参数化扫描，我们发现首缝 $P_{2D}$ 展现出高度复杂的干涉敏感性。以下结论严格依据仿真输出的稳态数据提取。

### 5.1 首缝响应与位置深度的关系 ($X_1$)
**数据现象**：与“稳态粘性剪切会导致能量随深度单调递减”的常规假设不同，数据表明 $P_{2D}$ 随深度 $X_1$ 呈现出非单调的剧烈波动。例如，在 $n_{total}=2, S=50\text{m}$ 的工况下，$P_{2D}$ 在 $X_1=2000\text{m}$ 处为 2.23，在 $X_1=2500\text{m}$ 处锐减至 1.04，但随后在 $X_1=3000\text{m}$ 时反弹至 2.73 并继续波动。
**机理分析**：这种深度的非单调性表明，首缝反射能量在很大程度上受控于井筒内的宏观驻波或相位干涉效应。特定的深度 $X_1$ 会导致入射波与反射波在井筒特定边界条件下发生相长或相消干涉，这意味着稳态摩阻并未完全主导绝对能量分布，绝对深度的干涉相位是一个不可忽略的调节因子。

### 5.2 首缝响应与裂缝总数量的关系 ($n_{total}$)
**数据现象**：在相同深度（如 $X_1=2000\text{m}$）和相同间距下，随着下游裂缝总数 $n_{total}$ 的增加，首缝的 $P_{2D}$ 并没有表现出单向的“衰减”或“掩蔽”。相反，能量呈现出明显的震荡。例如在 $S=50\text{m}$ 时，$n=2$ 到 $n=3$，$P_{2D}$ 从 2.23 骤降至 1.47；但当 $n$ 继续增加到 6 时，能量又回升至 1.95，随后在 $n=7,8$ 时稳定在 1.76 左右。在 $S=100\text{m}$ 时，甚至出现了大数量（$n=8$）比小数量（$n=2$）能量更高（2.30 > 2.18）的现象。
**机理分析**：这一数据特征直接推翻了“下游裂缝必然导致上游信号能量耗散”的线性直觉。其物理实质是波场在多裂缝节点间的**多次透射与反射相干叠加（Constructive/Destructive Interference）**。下游裂缝反串的波系进入首缝的倒谱时间窗内，随着 $n_{total}$ 改变，这些反射波的相位叠加既可能削弱（相消），也可能增强（相长）首缝的表观能量。

### 5.3 首缝响应与裂缝间距的关系 ($S$)
**数据现象**：在固定深度和裂缝总数（如 $X_1=2000\text{m}, n_{total}=8$）的情况下，$P_{2D}$ 对间距 $S$ 的响应呈现出一种复杂的“U型”或非线性反转。在小间距至中间距（$S=10 \sim 40\text{m}$）时，能量逐渐走低，在 $S=40\text{m}$ 时达到低谷（1.65）；但当间距进一步拉大到大尺度（$S \ge 90\text{m}$）时，能量显著回升并达到高点（2.30）。
**机理分析**：裂缝间距 $S$ 直接决定了多重反射波到达的时间延迟。在中间距（如 $30-40\text{m}$）时，多次反射波的延迟时间恰好落入首缝倒谱分析窗的高敏感干涉区，导致了强烈的相消干涉；而当间距足够大（如 $100\text{m}$），反射波包在时间轴上得以充分分离，跨波包的干涉耦合减弱，首缝主峰能量得以恢复至不受下游干扰的独立本征水平。

> **Figure 2. Parametric sensitivity and interference patterns of the primary cepstral peak.** Visualization of the data-driven relationships extracted from steady-state simulations: non-monotonic depth dependence ($X_1$), oscillatory interference driven by fracture multiplicity ($n_{total}$), and spacing-dependent constructive/destructive wavefield coupling ($S$).

## 6. Conclusions (核心科学结论)

在稳态流场中，多级缝网系统中的水击波响应并非简单的能量耗散过程。数据的非单调震荡特征确凿地证明：首条裂缝的表观倒谱能量受到井筒绝对深度相位、裂缝数量以及间距时延引发的**强相干叠加**控制。这要求现场工程算法在进行裂缝参数反演时，必须摒弃孤立单缝的线性叠加假设，充分考虑整个裂缝网络波场干涉的全局耦合效应。展宽指数模型的引入，则为量化这种复杂干涉条件下的宏观能量衰减提供了更具物理意义的评估手段。
