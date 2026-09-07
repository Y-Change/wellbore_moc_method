# -*- coding: utf-8 -*-
import os

target = r"e:\water_hammer_research\wellbore_moc_method\PaperB_考虑brunone的倒谱识别\文章撰写\PaperB_SPEJ_中文稿.md"

content = """# 井筒倒谱测深中特征时钟分叉的非定常摩阻模型依赖性

**摘要**：基于水击压力波的同态倒谱分析是水平井多段压裂裂缝几何定位的重要方法。现有诊断理论通常基于理想无耗散声波传播假设，将实测倒谱峰位的滞后简单归因于流体-岩石系统的等效声速折减。本文基于考虑瞬时对流加速度的一维非定常摩阻模型（Brunone IAB）与经典权重函数卷积模型（Vardy–Brown WFB），建立了多裂缝井筒水力瞬变特征线法（MOC）高精度前向求解体系，系统追踪了波前起跳时钟（$t_{\\mathrm{onset}}$）、波形峰值时钟（$t_{\\mathrm{peak}}$）、能量中位时刻时钟（$t_{E50}$）以及倒谱主峰时钟（$\\tau_{\\mathrm{cep}}$）在管壁非定常耗散下的演化规律。研究表明：在管壁剪切弛豫作用下，水击回声波包经历严重的能量弥散与频选耗散，导致特征时钟发生分叉——波前起跳点在常态弱摩阻下（$k \\le 0.02$）保持稳定（走时偏差 $\\le 1.0\\,\\mathrm{ms}$），而波峰、能量中位时刻与倒谱时钟呈现显著右偏；在 Brunone IAB 弱摩阻（$k=0.01$）阶跃工况下，倒谱时钟产生 $+13.0\\,\\mathrm{ms}$ 滞后，导致相对稳态 Darcy 基准出现约 $+9.43\\,\\mathrm{m}$ 的深部假象。然而，该测深偏移表现出强烈的模型依赖性：纯径向剪切扩散的 WFB 卷积模型预测的倒谱偏移为 $0.00\\,\\mathrm{m}$，未复现 IAB 模型的偏深现象；在局部瞬时雷诺数动态闭合模型 $k(\\mathrm{Re})$ 下，实倒谱正峰估计误差为 $+11.80\\,\\mathrm{m}$，冻结模倒谱极大值提取误差达 $+18.73\\,\\mathrm{m}$，分别对应确定算法协议下的响应。有限关井历时扫描表明，倒谱相对偏移随关井时间延长从阶跃的 $+9.43\\,\\mathrm{m}$ 单调收敛至 $+5.08\\,\\mathrm{m}$，揭示实测时延为激振源谱与井筒耗散传播的耦合总响应。相对阻尼灵敏度响应面进一步表明，常态弱摩阻下井筒阻尼增量仅 $+7.0\\%$，远低于大滤失响应，两者阻尼效应相当仅在参数极端上界成立。研究证实所谓“等效声速折减”本质上是特征时钟分化与波包弥散的观测效应，为压裂水击波形解释提供了严格的物理依据。

**关键词**：水击波；同态倒谱；特征时钟分叉；非定常摩阻；水力压裂；模型形式不确定性

---

## 1. 引言

在非常规油气藏水力压裂施工中，实时获取井下射孔簇的开启状态、水力裂缝位置及缝口柔量对于评价段内均匀压裂效果至关重要（Qiu et al., 2022; Dong et al., 2024）。在各类非侵入式监测技术中，水击瞬变压力分析法利用泵关断瞬态激发的高频水击压力波在井筒与水力裂缝间的反射回声信号，通过井口高频压力记录反演井下裂缝参数，因其无需起下井下仪器、成本低廉且具备施工全过程覆盖能力而受到广泛关注（Sun et al., 2025）。

为从强噪声与多重反射干扰的井口水击响应中准确提取微弱的回声走时，同态解卷积（Homomorphic Deconvolution）与实倒谱（Real Cepstrum）分析技术被引入压裂诊断领域（Childers et al., 1977）。倒谱分析通过对压力信号的对数幅值谱进行逆傅里叶变换，将频域中的周期性回声纹理映射为倒频域（Quefrency）中孤立的 Delta 峰值，从而实现回声时延 $\\tau$ 的高分辨率估计。随后利用平面声波理论测深公式：
$$x = \\frac{a\\,\\tau}{2} \\tag{1}$$
即可直接计算反射断面的空间几何深度。式中 $a$ 为压力波在流体-井筒系统中的传播声速。

然而，在压裂现场实际应用中，倒谱反演得到的裂缝深度往往显著大于套管射孔施工记录的真实物理深度，文献中常报道出现 $5\\sim 20\\,\\mathrm{m}$ 的系统性偏深现象。针对这一系统性偏差，目前现场工程界普遍采用“等效声速折减”进行经验性校准——即假设含砂流体或井筒微结构导致声速从清水理论值（约 $1450\\,\\mathrm{m/s}$）下降至 $1380\\sim 1400\\,\\mathrm{m/s}$，以此强制将测深结果向几何射孔位置压缩。

尽管这种经验声速折减能够在单点几何标定上获得形式上的吻合，但其物理合理性长期存疑：
1. **科学问题 1（RQ1：特征时钟分叉机制）：** 在强耗散井筒瞬变流中，水击波包的波前起跳点（Wavefront Onset）、波形几何峰值（Peak）、能量中位时刻（$E_{50}$）与倒谱极大值（Cepstral Peak）是否保持同步传播？若发生分化，倒谱峰位的右移究竟是介质本构波速发生了降低，还是波形弥散引发的特征时钟分化？
2. **科学问题 2（RQ2：模型形式与边界参数依赖性）：** 倒谱时延滞后是否普遍存在于所有瞬态摩阻机制中？基于瞬时加速度的经验闭合模型（如 Brunone IAB）与基于层剪切扩散的理论卷积模型（如 Vardy–Brown WFB）对测深偏差的预测是否存在根本性分歧？有限关井历时（$T_c$）与阻尼参数如何调控该偏差？
3. **科学问题 3（RQ3：频选耗散与算法实现响应）：** 井筒非定常摩阻引发的高频选择性衰减如何映射至倒频域？动态雷诺数依赖下的绝对测深误差在不同极值拾取协议下表现出怎样的数值特征？
4. **科学问题 4（RQ4：摩阻与地层滤失的相对阻尼边界）：** 井筒壁面非定常剪切耗散与压裂缝口高导流滤失的衰减效应是否存在灵敏度边界？能否在阻尼分析中加以区分？

为此，本文建立了一维水力瞬变 MOC 高精度正演模拟体系，系统开展了四特征时钟追踪、模型形式对比、有限关井扫描、多簇缝网弥散及阻尼灵敏度响应面分析，以期阐明井筒倒谱测深偏差的物理本质。

---

## 2. 水击波非定常摩阻正演模型与时钟定义

### 2.1 控制方程与非定常壁面剪切闭合

在微可压缩液体与线弹性井壁圆柱形管道中，考虑非定常壁面剪切应力的一维连续性方程与动量守恒方程表述为（Wylie and Streeter, 1993; Ghidaoui et al., 2005）：
$$\\frac{\\partial H}{\\partial t} + \\frac{a^2}{g} \\frac{\\partial V}{\\partial z} = 0 \\tag{2}$$
$$\\frac{\\partial V}{\\partial t} + g \\frac{\\partial H}{\\partial z} + \\frac{4\\tau_w}{\\rho D} = 0 \\tag{3}$$
式中 $H(z, t)$ 为测压管水头（$\\mathrm{m}$）；$V(z, t)$ 为断面平均流速（$\\mathrm{m/s}$）；$a$ 为压力波在流体-管柱系统中的传播声速（基准值取 $1450\\,\\mathrm{m/s}$）；$g = 9.81\\,\\mathrm{m/s^2}$ 为重力加速度；$D = 0.1397\\,\\mathrm{m}$ 为 5.5 英寸套管内径；$\\rho = 1000\\,\\mathrm{kg/m^3}$ 为流体密度；$\\tau_w$ 为壁面总剪切应力。

总剪切应力分解为稳态拟 Darcy 项 $\\tau_{ws}$ 与非定常剪切项 $\\tau_{wu}$ 之和：
$$\\tau_w = \\tau_{ws} + \\tau_{wu} = \\frac{1}{8}\\rho f V|V| + \\tau_{wu} \\tag{4}$$
式中 $f$ 为达西稳态摩阻系数，在紊流下通过 Zigrand–Swamee（Swamee–Jain 族）显式公式根据雷诺数与相对粗糙度计算确定。

针对非定常项 $\\tau_{wu}$，本文对比两类具有代表性的一维闭合模型：

#### 1. 基于瞬时对流加速度的 IAB 模型（Brunone et al., 1991, 2000; Vitkovsky et al., 2000）
一维动量方程中局部与对流加速度的经验加权闭合：
$$\\tau_{wu} = \\frac{\\rho D k}{4} \\left( \\frac{\\partial V}{\\partial t} + a \\cdot \\operatorname{sgn}(V) \\left| \\frac{\\partial V}{\\partial z} \\right| \\right) \\tag{5}$$
式中 $k$ 为无量纲 Brunone 摩阻系数。在特征线法（MOC）正向网格离散中，特征线方向的非定常摩阻项 $J_u$ 离散为：
$$J_{u,L} = \\frac{k \\Delta t}{g} \\left[ \\left( \\frac{\\partial V}{\\partial t} \\right)_{i-1} + a \\cdot \\operatorname{sgn}(V_{i-1}) \\left| \\frac{\\partial V}{\\partial z} \\right|_{i-1} \\right] \\tag{6}$$
$$J_{u,R} = \\frac{k \\Delta t}{g} \\left[ \\left( \\frac{\\partial V}{\\partial t} \\right)_{i+1} + a \\cdot \\operatorname{sgn}(V_{i+1}) \\left| \\frac{\\partial V}{\\partial z} \\right|_{i+1} \\right] \\tag{7}$$
本文基准算例取常数弱摩阻 $k = 0.01$，并扫描 $k \\in \\{0, 0.01, 0.02, 0.05\\}$。

#### 2. 基于加权函数卷积的 WFB 模型（Vardy and Brown, 2003, 2004）
基于流体微元径向动量扩散积分推导的严格一维卷积形式：
$$\\tau_{wu}(t) = \\frac{2\\rho \\nu}{R} \\int_0^t \\frac{\\partial V}{\\partial t}(t - t') W(t') \\,\\mathrm{d}t' \\tag{8}$$
式中 $W(t')$ 为 Vardy–Brown 紊流剪切权重函数，在代码中通过 10 阶指数衰减基函数快速求和逼近：
$$W(\\tau) \\approx \\sum_{j=1}^{10} m_j \\exp(-n_j \\tau) \\tag{9}$$
式中 $\\tau = \\nu t / R^2$ 为无量纲扩散时间。该模型严格反映边界层纯剪切扩散机制，不包含轴向对流项。

### 2.2 裂缝与井底水力边界条件

在压裂井筒中，水力裂缝被建模为井筒断面处的侧向集总水力分支（Qiu et al., 2022; Sun et al., 2025）：
$$V_L(x_f, t) - V_R(x_f, t) = \\frac{Q_f(t)}{A} = \\frac{C_H}{A} \\frac{\\mathrm{d}H_f}{\\mathrm{d}t} + \\frac{k_{\\mathrm{leak}}}{A} \\sqrt{\\max(0, H_f - H_{\\mathrm{ext}})} \\tag{10}$$
式中 $V_L$ 与 $V_R$ 分别为裂缝节点上、下游断面的流速；$Q_f(t)$ 为注入单条裂缝的总瞬态流量；$A = \\pi D^2 / 4$ 为井筒截面积；$C_H$ 为水头定义下的裂缝集总水力柔量（基准值取 $C_H = 1.0 \\times 10^{-5}\\,\\mathrm{m^2}$，对应压力柔量 $C_{\\mathrm{frac}} = C_H / (\\rho g) \\approx 1.019 \\times 10^{-9}\\,\\mathrm{m^3/Pa}$）；$k_{\\mathrm{leak}} = 1.0 \\times 10^{-4}\\,\\mathrm{m^2/s/\\sqrt{m}}$ 为非线性平方根滤失系数；$H_{\\mathrm{ext}} = 100.0\\,\\mathrm{m}$ 为远场恒定孔隙水头。

井底（$z = L = 5000\\,\\mathrm{m}$）设定为恒定压力水头边界（$H(L, t) = H_0 = 300\\,\\mathrm{m}$），模拟水平段盲板或深部地层稳定压力连通。

### 2.3 井口关井激励与有限关井时间源函数

井口（$z = 0$）受控于高压压裂泵关断水力瞬变。激振控制量为断面速度边界条件：
$$V(0, t) = \\begin{cases} 
V_0, & t < t_s \\\\
V_0 \\left(1 - \\dfrac{t - t_s}{T_c}\\right), & t_s \\le t \\le t_s + T_c \\\\
0, & t > t_s + T_c
\\end{cases} \\tag{11}$$
式中 $T_c$ 为阀门关断历时。数值实现中，所谓的“阶跃关井”（$T_c = 1\\,\\mathrm{ms}$）在代码中严格为跨越单个时间离散步长 $\\Delta t = 1.0\\,\\mathrm{ms}$ 的快速线性关断。

在有限关井时间扫描中（$T_c \\in [50, 1000]\\,\\mathrm{ms}$），关井终值流速始终为 0，改变的是入射压力脉冲的上升沿斜率与频带宽度。关井时间展宽引起倒谱相对深度偏移的收窄（如 $T_c=1\\,\\mathrm{ms}$ 时为 $+9.43\\,\\mathrm{m}$，$T_c=200\\,\\mathrm{ms}$ 时为 $+5.80\\,\\mathrm{m}$，$T_c=1000\\,\\mathrm{ms}$ 时为 $+5.08\\,\\mathrm{m}$），本质上是**入射源波包频谱与井筒-裂缝信道耗散传播的耦合总响应**，本文不声称已将源项与传播项完全解耦。

### 2.4 特征线法（MOC）数值离散与基准工况

沿特征线 $\\mathrm{d}z/\\mathrm{d}t = \\pm a$ 将控制方程转化为常微分相容方程，并满足 Courant 准则：
$$\\mathrm{Cr} = \\frac{a\\,\\Delta t}{\\Delta z} = 1.0 \\tag{12}$$
井筒全长 $L = 5000\\,\\mathrm{m}$，声速 $a = 1450\\,\\mathrm{m/s}$，空间步长 $\\Delta z = 1.45\\,\\mathrm{m}$，时间步长 $\\Delta t = 1.0\\,\\mathrm{ms}$（网格节点数 $N_z = 3449$）。初始稳定流动工况冻结为：$V_0 = 1.0\\,\\mathrm{m/s}$，井口测压水头 $H_0 = 300\\,\\mathrm{m}$，首条裂缝深度 $X_1 = 4100\\,\\mathrm{m}$。

### 2.5 时间步长敏感性与空间网格映射说明

为检验特征线法（MOC）时间离散步长对特征时钟提取与测深偏差的影响，在单缝工况（$n=1, X_1 = 4100\\,\\mathrm{m}$）下开展了三档时间步长敏感性对比（$\\Delta t \\in \\{0.5, 1.0, 2.0\\}\\,\\mathrm{ms}$，满足 $\\mathrm{Cr} = 1.0$）。

在 MOC 网格构建中，空间网格数按 $N = \\operatorname{round}(L / (a\\Delta t))$ 计算，实际空间步长为 $\\Delta z = L / N$，裂缝物理位置 $X_1 = 4100\\,\\mathrm{m}$ 映射至最近网格节点 $i_f = \\operatorname{round}(X_1 / \\Delta z)$，未采用连续插值处理。各档网格的实际映射位置与几何误差见下表：

| 时间步长 $\\Delta t$ | 空间网格数 $N$ | 空间步长 $\\Delta z$ | 裂缝就近节点 $i_f$ | 映射物理位置 | 几何截断误差 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.5\\,\\mathrm{ms}$** | 6897 | $0.724953\\,\\mathrm{m}$ | 5656 | **$4100.33\\,\\mathrm{m}$** | **$+0.33\\,\\mathrm{m}$** |
| **$1.0\\,\\mathrm{ms}$** | 3448 | $1.450116\\,\\mathrm{m}$ | 2827 | **$4099.19\\,\\mathrm{m}$** | **$-0.81\\,\\mathrm{m}$** |
| **$2.0\\,\\mathrm{ms}$** | 1724 | $2.900232\\,\\mathrm{m}$ | 1414 | **$4100.93\\,\\mathrm{m}$** | **$+0.93\\,\\mathrm{m}$** |

在三档时间步长下，各算例均采用单步线性关断（即 $T_c = \\Delta t$ 的 `velocity_step` 协议），对比稳态 Darcy（$k=0$）与 Brunone IAB（$k=0.01$）的四特征时钟提取结果见表 2。

**表 2: 三档时间步长下四特征时钟响应与相对漂移敏感性对比表 ($n=1, X_1 = 4100\\,\\mathrm{m}$)**

| 时间步长 $\\Delta t$ | 摩阻状态 | $t_{\\mathrm{onset}}$ (s) | $t_{\\mathrm{peak}}$ (s) | $t_{E50}$ (s) | $\\tau_{\\mathrm{cep}}$ (s) | 相对 $\\delta x_{\\mathrm{onset}}$ | 相对 $\\delta x_{\\mathrm{peak}}$ | 相对 $\\delta x_{E50}$ | 相对 $\\delta x_{\\mathrm{cep}}$ | 相对 $\\Delta\\tau_{\\mathrm{cep}}$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.5\\,\\mathrm{ms}$** | 稳态 Darcy ($k=0$) | 6.6555 | 6.6555 | 6.6555 | 5.6560 | $0.00\\,\\mathrm{m}$ | $0.00\\,\\mathrm{m}$ | $0.00\\,\\mathrm{m}$ | **$0.00\\,\\mathrm{m}$** | $0.0\\,\\mathrm{ms}$ |
| | Brunone IAB ($k=0.01$) | 6.6565 | 6.6695 | 6.6688 | 5.6695 | $+0.73\\,\\mathrm{m}$ | $+10.15\\,\\mathrm{m}$ | $+9.65\\,\\mathrm{m}$ | **$+9.79\\,\\mathrm{m}$** | $+13.5\\,\\mathrm{ms}$ |
| **$1.0\\,\\mathrm{ms}$** | 稳态 Darcy ($k=0$) | 6.6530 | 6.6530 | 6.6530 | 5.6540 | $0.00\\,\\mathrm{m}$ | $0.00\\,\\mathrm{m}$ | $0.00\\,\\mathrm{m}$ | **$0.00\\,\\mathrm{m}$** | $0.0\\,\\mathrm{ms}$ |
| | Brunone IAB ($k=0.01$) | 6.6530 | 6.6670 | 6.6659 | 5.6670 | $0.00\\,\\mathrm{m}$ | $+10.15\\,\\mathrm{m}$ | $+9.38\\,\\mathrm{m}$ | **$+9.43\\,\\mathrm{m}$** | $+13.0\\,\\mathrm{ms}$ |
| **$2.0\\,\\mathrm{ms}$** | 稳态 Darcy ($k=0$) | 6.6540 | 6.6540 | 6.6539 | 5.6560 | $0.00\\,\\mathrm{m}$ | $0.00\\,\\mathrm{m}$ | $0.00\\,\\mathrm{m}$ | **$0.00\\,\\mathrm{m}$** | $0.0\\,\\mathrm{ms}$ |
| | Brunone IAB ($k=0.01$) | 6.6540 | 6.6680 | 6.6663 | 5.6680 | $0.00\\,\\mathrm{m}$ | $+10.15\\,\\mathrm{m}$ | $+8.94\\,\\mathrm{m}$ | **$+8.70\\,\\mathrm{m}$** | $+12.0\\,\\mathrm{ms}$ |

数值结果表明：
1. **相对漂移同号同量级：** 在 $\\Delta t \\in \\{0.5, 1.0, 2.0\\}\\,\\mathrm{ms}$ 三档步长下，Brunone IAB 相对稳态 Darcy 的倒谱相对漂移 $\\delta x_{\\mathrm{cep}}$ 分别为 $+9.79\\,\\mathrm{m}$、$+9.43\\,\\mathrm{m}$ 与 $+8.70\\,\\mathrm{m}$（对应相对时延 $+13.5\\,\\mathrm{ms}$、$+13.0\\,\\mathrm{ms}$ 与 $+12.0\\,\\mathrm{ms}$），保持严格同号（正向偏深）且处于同一物理量级；相对波峰漂移 $\\delta x_{\\mathrm{peak}}$ 在三档网格下均为 $+10.15\\,\\mathrm{m}$，能量中位时刻漂移 $\\delta x_{E50}$ 保持在 $+8.94\\sim +9.65\\,\\mathrm{m}$ 区间；波前起跳相对漂移在 $\\Delta t=0.5\\,\\mathrm{ms}$ 时为 $+0.73\\,\\mathrm{m}$（$+1.0\\,\\mathrm{ms}$），在 $1.0\\,\\mathrm{ms}$ 与 $2.0\\,\\mathrm{ms}$ 时为 $0.0\\,\\mathrm{ms}$，仍满足时延 $\\le 1.0\\,\\mathrm{ms}$ 的特征；
2. **源项与网格耦合说明：** 各档测试采用单步快速关断（$T_c = \\Delta t$），网格步长变化同时耦合了激振源脉冲上升沿的微弱差异；各档间 $\\delta x_{\\mathrm{cep}}$ 的微小差异（$0.5\\,\\mathrm{ms}$ 与 $1.0\\,\\mathrm{ms}$ 相差 $0.36\\,\\mathrm{m}$，$1.0\\,\\mathrm{ms}$ 与 $2.0\\,\\mathrm{ms}$ 相差 $0.73\\,\\mathrm{m}$）与倒谱采样时间分辨率对应的测深量子（$a\\Delta t / 2 \\approx 0.725\\,\\mathrm{m}$）处于同一量级。三档对比旨在检验不同时间步长下时钟分叉与相对漂移的数值稳定性与方向一致性，不作极限收敛阶的严格声明；本文主文中的基准量化指标 $+9.43\\,\\mathrm{m}$（$\\Delta\\tau = +13.0\\,\\mathrm{ms}$）严格绑定于基准离散协议 $\\Delta t = 1.0\\,\\mathrm{ms}$。

### 2.6 信号处理管线与四种特征时钟定义

为杜绝后处理口径歧义，信号处理流程严格冻结如下：

#### 1. 分析通道与同态解卷积加法分离原理
全篇计算均采用井口水头变化率 $\\mathrm{d}H/\\mathrm{d}t$ 通道，自动消除低频静水压漂移。

在频域中，井口实测水击信号 $Y(f)$ 可表示为激振源项 $S(f)$ 与井筒-裂缝系统传递函数 $H(f)$ 的乘积：
$$Y(f) = S(f) H(f) \\tag{13}$$
对幅值谱取自然对数，频域乘积关系转化为对数谱的线性叠加：
$$\\ln |Y(f)| = \\ln |S(f)| + \\ln |H(f)| \\tag{14}$$
进而通过逆傅里叶变换将对数谱映射至倒频域（Quefrency），定义实倒谱 $C_y(\\tau)$：
$$C_y(\\tau) = \\mathcal{F}^{-1}\\left\\{ \\ln |Y(f)| \\right\\} = C_s(\\tau) + C_h(\\tau) \\tag{15}$$
实倒谱具备零相位与双边偶对称特性（$C_y(-\\tau) = C_y(\\tau)$）。在时域卷积转化为倒频域加法叠加的框架下，若源项与信道回声在倒频域占据不同支撑区间，则为两者的分离提供了理论可能。

#### 2. 实倒谱计算与峰值拾取流程
在关井后 $t \\in [1.0, 50.0]\\,\\mathrm{s}$（$N = 49{,}000$ 点）截取信号并施加全 Hann 窗 $w_{\\mathrm{Hann}}(t)$：
$$C(\\tau) = \\mathcal{F}^{-1} \\left\\{ \\ln \\left| \\mathcal{F} \\left\\{ \\left( \\frac{\\mathrm{d}H_{\\mathrm{wh}}(t)}{\\mathrm{d}t} - \\overline{\\frac{\\mathrm{d}H_{\\mathrm{wh}}}{\\mathrm{d}t}} \\right) w_{\\mathrm{Hann}}(t) \\right\\} \\right| \\right\\} \\tag{16}$$
实倒谱计算完成后，在理论几何走时搜索窗口 $\\mathcal{W}_{\\mathrm{search}}$ 内检索 $|C(\\tau)|$ 的局部极大值，确定回声倒谱时钟位置 $\\tau_{\\mathrm{cep}}$（首条裂缝 $X_1 = 4100\\,\\mathrm{m}$ 对应 $\\tau_{\\mathrm{geom}} = 2X_1/a = 5.6552\\,\\mathrm{s}$）：
$$\\tau_{\\mathrm{cep}} = \\arg\\max_{\\tau \\in \\mathcal{W}_{\\mathrm{search}}} |C(\\tau)| \\tag{17}$$
式中搜索窗口 $\\mathcal{W}_{\\mathrm{search}}$ 在基准阶跃关井下为几何走时邻域 $[\\tau_{\\mathrm{geom}} \\pm 80\\,\\mathrm{ms}]$；在有限关井 $T_c$ 扫描中，为容纳展宽波包自适应设定为 $[\\tau_{\\mathrm{geom}} - 80\\,\\mathrm{ms}, \\tau_{\\mathrm{geom}} + 80\\,\\mathrm{ms} + \\min(0.5, T_c)]$。

#### 3. 四种特征时钟定义
1. **波前起跳时钟 ($t_{\\mathrm{onset}}$)：** 窗口内信号绝对值首次达到局部反射波峰值 $1\\%$ 的时刻：
   $$t_{\\mathrm{onset}} = \\min \\left\\{ t \\;\\middle|\\; \\left|\\frac{\\mathrm{d}H}{\\mathrm{d}t}(t)\\right| \\ge 0.01 \\max \\left|\\frac{\\mathrm{d}H}{\\mathrm{d}t}\\right| \\right\\} \\tag{18}$$
2. **波形峰值时钟 ($t_{\\mathrm{peak}}$)：** 反射波包梯度局部极大值时刻：
   $$t_{\\mathrm{peak}} = \\arg\\max_{t} \\left|\\frac{\\mathrm{d}H}{\\mathrm{d}t}(t)\\right| \\tag{19}$$
3. **能量中位时刻时钟 ($t_{E50}$)：** 波包累计能量积分达到 $50\\%$ 的中位时刻：
   $$t_{E50} \\implies \\int_{t_{\\mathrm{onset}}}^{t_{E50}} \\left(\\frac{\\mathrm{d}H}{\\mathrm{d}t}\\right)^2 \\,\\mathrm{d}t = \\frac{1}{2} \\int_{t_{\\mathrm{onset}}}^{t_{\\mathrm{end}}} \\left(\\frac{\\mathrm{d}H}{\\mathrm{d}t}\\right)^2 \\,\\mathrm{d}t \\tag{20}$$
4. **倒谱时钟 ($\\tau_{\\mathrm{cep}}$)：** 式 (17) 提取的倒频域极大值峰位。

#### 4. 测深与误差公式
- 绝对测深：$x_{\\mathrm{est}} = a(t_{\\mathrm{clk}} - t_s)/2$ 或 $x_{\\mathrm{est}} = a\\tau_{\\mathrm{cep}}/2$（$t_s = 1.0\\,\\mathrm{s}$）；
- 绝对误差：$\\Delta x = x_{\\mathrm{est}} - X_1$；
- 相对稳态漂移：$\\delta x = \\Delta x(k) - \\Delta x(k=0) = a\\Delta t_{\\mathrm{clk}}/2$。

### 2.7 动态雷诺数依赖的 Brunone 摩阻闭合模型

除常数弱摩阻假定（$k=0.01$）外，经典水力瞬变理论中 Brunone 系数通常被表征为随局部瞬时流动雷诺数 $\\mathrm{Re}(z, t) = \\rho |V(z, t)| D / \\mu$ 动态演化的闭合函数（Vardy and Brown, 2003; Bergant et al., 2008）：
$$k(\\mathrm{Re}) = \\frac{\\sqrt{C^*}}{2} \\tag{21}$$
式中 $C^*$ 为 Vardy 剪切衰减系数，在 `wellbore_moc.py` 中分段闭合实现如下：
- 当 $\\mathrm{Re} < 1.0$ 时，$k = 0$；
- 层流区（$1.0 \\le \\mathrm{Re} < 2000$）：$C^* = 4.76 \\times 10^{-3}$（对应 $k \\approx 0.0345$）；
- 紊流区（$\\mathrm{Re} \\ge 2000$）：
  $$C^* = \\frac{7.41}{\\mathrm{Re}^{\\log_{10}(14.3 / \\mathrm{Re}^{0.05})}} \\tag{22}$$

在本文数值工况中，仅对初始稳态雷诺数 $\\mathrm{Re}_0 \\approx 1.397 \\times 10^5$ 下的初始摩阻系数 $k_0 \\approx 0.0384$ 进行了基准核算，非定常流动中 $k(z, t)$ 沿井筒全场时空演化分布的极值与中位数特征有待后续探针日志进一步归档。随着关井后流速振荡衰减，$k(\\mathrm{Re})$ 沿井筒空间和时间动态变化，呈现出强烈的非线性耗散特征。该动态闭合模型与固定常数 $k=0.01$ 在物理机制与参数维度上不属于同一闭合形式。

特别需要说明的是：在动态雷诺数归档数据（`PaperB_考虑brunone的倒谱识别/02_时域波形退化_峰漂移与展宽/data/onset_correction_verdict.csv`）中，由于单缝工况（$n=1, X_1=4100\\,\\mathrm{m}$）不受缝间距参数影响，标号为 D5 至 D100 的各算例中的单缝波形在数值上完全相同（属于**同一条单缝动态波形**）。对该条相同波形采用不同倒谱实现协议进行处理，产生了两组明确的绝对测深误差：
1. **一维实倒谱正峰估计：** 采用单侧实倒谱 $C(\\tau)$ 极大值检索，得到 $\\tau_{\\mathrm{cep}} = 5.6715\\,\\mathrm{s}$，对应绝对测深误差 $\\Delta x_{\\mathrm{cep}} = \\mathbf{+11.80\\,\\mathrm{m}}$；
2. **冻结模倒谱极值估计：** 采用模倒谱 $|C(\\tau)|$ 极大值检索时，强非线性剪切引起的负向副瓣极值被模算子捕获，提取得到 $\\tau_{\\mathrm{cep}} = 5.6810\\,\\mathrm{s}$，对应绝对测深误差 $\\Delta x_{\\mathrm{cep}} = \\mathbf{+18.73\\,\\mathrm{m}}$。

这两组数值分别严格对应上述**两种确定的倒谱算法实现**，反映了特定非线性剪切波形在不同极大值拾取协议下的数值响应，不可混为随机的 $10\\sim 20\\,\\mathrm{m}$ 统计误差区间。

为系统梳理本文各项数值仿真算例与主文科学结论的对应关系，表 1 给出了全篇前向计算工况的分类、参数配置及其在正文论据链中的明确角色。

**表 1: 前向数值仿真算例分类与主文证据角色表**

| 算例类别 | 仿真参数与物理配置 | 涉及主要图表 | 在主文中的证据角色与科学论点 |
| :--- | :--- | :--- | :--- |
| **主基准结果**<br>(Primary Baseline) | 单缝（$n=1, X_1=4100\\,\\mathrm{m}$），阶跃关井（$T_c=1\\,\\mathrm{ms}$），常数摩阻 $k \\in \\{0, 0.01, 0.02, 0.05\\}$ | 图 2, 图 3<br>表 3 | **时钟分叉与表观波速分化：** 证实 $k=0.01$ 下波峰与倒谱均滞后 $+13.0\\,\\mathrm{ms}$（$\\delta x_{\\mathrm{cep}} = +9.43\\,\\mathrm{m}$），而起跳时钟在 $k \\le 0.02$ 下保持在 $0\\sim 1\\,\\mathrm{ms}$ 内稳定；证明表观波速折减源于特征时钟分化而非介质声速退化。 |
| **模型形式对比**<br>(Model Form Comparison) | 单缝（$n=1$），阶跃关井，对比稳态 Darcy、Brunone IAB（$k=0.01$）与 Vardy–Brown WFB 卷积模型 | 图 6(c)<br>表 4 | **经验闭合模型的形式不确定性：** 证实纯径向剪切扩散（WFB）下 $\\delta x_{\\mathrm{cep}} = 0.00\\,\\mathrm{m}$，未复现 IAB 模型的 $+9.43\\,\\mathrm{m}$ 偏深，揭示测深偏移对一维经验加速度闭合形式的强依赖性。 |
| **压力测试：关井历时**<br>(Stress: Valve Ramp) | 单缝（$n=1, k=0.01$），线性关井扫描 $T_c \\in [1, 1000]\\,\\mathrm{ms}$ | 图 6(b)<br>表 4 | **源谱与传播耦合总响应：** 倒谱偏深随关井时间展宽从 $+9.43\\,\\mathrm{m}$（$T_c=1\\,\\mathrm{ms}$）单调收敛至 $+5.08\\,\\mathrm{m}$（$T_c=1000\\,\\mathrm{ms}$），表明有限关井时间改变了偏深幅度，该响应为激振源谱与井筒耗散的耦合响应。 |
| **压力测试：多簇缝网**<br>(Stress: Multi-cluster) | 4 簇射孔段（Matrix A，$n=4, D \\in [5, 100]\\,\\mathrm{m}$），$k=0.01$ | 图 4, 图 5A, 图 5(b,c)<br>表 6 | **时频耗散与波包弥散演化：** 揭示高频（$60\\sim 150\\,\\mathrm{Hz}$）衰减 $-17\\,\\mathrm{dB}$ 及 2D 倒谱脊线随分析窗展宽粘连机制；波峰对比度 $C_v$ 随间距单调退化，作为波包弥散度量，不声称物理硬阈值。 |
| **压力测试：阻尼灵敏度**<br>(Stress: Damping Sensitivity) | 单缝（$n=1$），$k \in [0, 0.05] \times k_{\\mathrm{leak}} \in [10^{-5}, 5\\times 10^{-4}]\\,\\mathrm{m^2/s/\\sqrt{m}}$，$H_{\\mathrm{ext}}=100\\,\\mathrm{m}$ | 图 6(a) | **相对阻尼灵敏度边界：** 考察井筒摩阻与地层滤失对系统总衰减的相对贡献，证实两者的阻尼增量相当仅在参数网格极端上界成立（$+55\\%$ vs $+49\\%$）；常态弱摩阻下井筒阻尼仅贡献 $+7.0\\%$，远低于大滤失响应，呈现清晰的相对灵敏度差异。 |
| **压力测试：时间步长**<br>(Stress: Time-Step Sensitivity) | 单缝（$n=1, X_1=4100\\,\\mathrm{m}$），Darcy（$k=0$）与 IAB（$k=0.01$），三档步长 $\\Delta t \\in \\{0.5, 1.0, 2.0\\}\\,\\mathrm{ms}$（$T_c=\\Delta t$） | 表 2 | **时间步长敏感性与数值稳定性：** 证实三档时步下倒谱相对漂移分别为 $+9.79\\,\\mathrm{m}$、$+9.43\\,\\mathrm{m}$ 与 $+8.70\\,\\mathrm{m}$，保持严格同号与同量级；说明时钟分叉在不同时间步长下的稳定性，主文 $+9.43\\,\\mathrm{m}$ 仍绑定于 $\\Delta t=1.0\\,\\mathrm{ms}$ 基准。 |
| **动态雷诺数闭合**<br>(Dynamic $k(\\mathrm{Re})$) | 单缝（$n=1$），Vardy 分段动态 $k(\\mathrm{Re})$，单缝波形在 D5–D100 算例中严格相同 | 图 5(a)<br>表 4 | **估计器实现响应差异：** 在同一条动态波形上，实倒谱正峰估计误差为 $+11.80\\,\\mathrm{m}$，冻结模倒谱因捕获负副瓣极值偏移至 $+18.73\\,\\mathrm{m}$，严格对应两种算法实现协议，非随机统计区间。 |
| **未入主文探索算例**<br>(Omitted / Excluded) | Matrix B（非均质双缝不均柔量扫描）、Matrix C（重质稠油粘度扫描）、超强摩阻补充外推（$k=0.1, 0.2$） | 仅在图 2(b) 虚线做微弱外推示意 | **主文证据剔除：** 避免引入非对流主导或缺乏井下实测约束的未验证自由度，严格收敛于水基低粘流体与可重复核验的基准证据链。 |

*(注：主文核心机理链以单缝（$n=1$）常数 $k$ 为基准；Matrix A 缝网（$n=4, D \\in [5, 100]\\,\\mathrm{m}$）用于多簇波包弥散、EST 展宽、频带能量吸收及视波速退化评估，图表与正文中已严格标明 $n=4$ 适用范围，不混同于单缝机理)*

---

## 3. 数值仿真结果与机理分析

### 3.1 时域波形耗散衰减与脉冲弥散演化

图 2 给出了单缝基准工况（$n=1, X_1=4100\\,\\mathrm{m}$）在阶跃关井下，井口水头梯度 $\\mathrm{d}H_{\\mathrm{wh}}/\\mathrm{d}t$ 的时域波形演化特征。

**图 2** 井筒非定常摩阻下时域水击压力波形的耗散演化。
(a) 首个回声波包时域波形（$t \\in [6.5, 7.5]\\,\\mathrm{s}$）：稳态 Darcy 流（$k=0$）呈现对称陡峭尖峰，随着非定常摩阻系数 $k$ 增大，波峰幅值骤降、峰位明显后移；
(b) 能量弥散时间（EST，包含 $90\\%$ 能量的时间窗）随 $k$ 从 $1.0\\,\\mathrm{ms}$ 单调展宽至 $27.0\\,\\mathrm{ms}$；
(c) 井口压力波长周期时域衰减记录（$t \\in [0, 35]\\,\\mathrm{s}$），展示非定常摩阻对高频震荡分量的快速吸收。

![图 2](figures/Fig2_waveform_evolution.png)

对比分析表明：
- 稳态 Darcy 模型（$k=0$）下，反射波包保持严格对称且无时移；
- 引入 Brunone 非定常项后（$k=0.01$），边界层剪切弛豫使波形发生显著不对称畸变：波形上升沿变缓，下降沿拖尾拉长；
- 边界层剪切弛豫削平了高频陡峭梯度，使波峰 $t_{\\mathrm{peak}}$ 滞后 **$\\Delta t_{\\mathrm{peak}} = +13.0\\,\\mathrm{ms}$**；
- 反射波包显著展宽：包含 $90\\%$ 能量的能量弥散时间（$\\mathrm{EST}$）从 $k=0$ 时的 $1.0\\,\\mathrm{ms}$ 单调展宽至 $k=0.01$ 时的 $8.0\\,\\mathrm{ms}$、$k=0.02$ 时的 $12.0\\,\\mathrm{ms}$ 及 $k=0.05$ 时的 $27.0\\,\\mathrm{ms}$（图 2(b)）。
- 在 $35\\,\\mathrm{s}$ 井口全周期衰减记录中（图 2(c)），高频微小波动在数个周期内迅速衰减，仅保留低频井筒驻波主模态（理论基频 $f_0 = a/4L = 0.0725\\,\\mathrm{Hz}$；FFT 离散谱线为 $3/49 \\approx 0.0612\\,\\mathrm{Hz}$）。

---

### 3.2 特征时钟分叉与表观波速分化特征

为定量考察非定常摩阻对不同到达时间估计器的影响，系统提取波前起跳（$t_{\\mathrm{onset}}$）、波形峰值（$t_{\\mathrm{peak}}$）、能量中位时刻（$t_{E50}$）与倒谱主峰（$\\tau_{\\mathrm{cep}}$）四种特征时钟，并对比其相对稳态基准（$k=0$）的漂移响应。

表 3 与图 3 给出了单缝工况下的四时钟响应与测深误差矩阵。

**表 3: 单缝工况下四种特征时钟响应与测深误差矩阵 ($n=1, X_1 = 4100\\,\\mathrm{m}$)**

| 摩阻状态 | $t_{\\mathrm{onset}}$ (s) | $t_{\\mathrm{peak}}$ (s) | $t_{E50}$ (s) | $\\tau_{\\mathrm{cep}}$ (s) | $\\Delta t_{\\mathrm{onset}}$ | $\\Delta t_{\\mathrm{peak}}$ | $\\Delta t_{E50}$ | $\\Delta\\tau_{\\mathrm{cep}}$ | $\\Delta x_{\\mathrm{onset}}$ (m) | $\\Delta x_{\\mathrm{peak}}$ (m) | $\\Delta x_{\\mathrm{cep}}$ (m) | 相对稳态 $\\delta x_{\\mathrm{cep}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **稳态 Darcy ($k=0$)** | 6.653 | 6.654 | 6.653 | 5.654 | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | **$-1.58$** | $-0.85$ | $-0.85$ | $0.00\\,\\mathrm{m}$ |
| **弱摩阻 ($k=0.01$)** | 6.653 | 6.667 | 6.666 | 5.667 | **0.0 ms** | **+13.0 ms** | **+13.0 ms** | **+13.0 ms** | **$-1.58$** | $+8.58$ | $+8.58$ | **$+9.43\\,\\mathrm{m}$** |
| **中等摩阻 ($k=0.02$)** | 6.654 | 6.680 | 6.679 | 5.677 | **+1.0 ms** | **+26.0 ms** | **+25.7 ms** | **+23.0 ms** | **$-0.85$** | $+18.00$ | $+15.83$ | **$+16.68\\,\\mathrm{m}$** |
| **强摩阻 ($k=0.05$)** | 6.659 | 6.718 | 6.714 | 5.704 | **+6.0 ms** | **+64.0 ms** | **+60.6 ms** | **+50.0 ms** | **$+2.78$** | $+45.55$ | $+35.40$ | **$+36.25\\,\\mathrm{m}$** |

**图 3** 特征时钟分叉与表观波速分化特征。
(a) 单缝工况（$n=1$）下四种特征时钟的时延量 $\\Delta t$ 随 $k$ 的演化关系，展示 $k \\le 0.02$ 时 $t_{\\mathrm{onset}}$ 保持在 $0\\sim 1\\,\\mathrm{ms}$（$k=0.05$ 时为 $+6.0\\,\\mathrm{ms}$），而 $t_{\\mathrm{peak}}, t_{E50}, \\tau_{\\mathrm{cep}}$ 呈现不同程度的右移；
(b) 绝对测深误差 $\\Delta x = x_{\\mathrm{est}} - X_1$，展示起跳时钟在本文工况下保持在 $[-1.58, +2.78]\\,\\mathrm{m}$ 区间；
(c) 表观波速分化（Matrix A 缝网，$n=4, D=20\\,\\mathrm{m}$）：波前起跳波速 $a_{\\mathrm{onset}} \\approx 1450.6\\,\\mathrm{m/s}$ 基本不退化，而表观波峰波速退化至 $a_{\\mathrm{peak}} \\approx 1433.3\\,\\mathrm{m/s}$，驻波模态波速降至 $a_{f0} \\approx 1397.0\\,\\mathrm{m/s}$。

![图 3](figures/Fig3_clock_splitting.png)

分析表明：
1. **起跳时钟稳定性：** 在 $k \\le 0.02$ 范围内，起跳时钟时延 $\\Delta t_{\\mathrm{onset}}$ 保持在 $0.0\\sim 1.0\\,\\mathrm{ms}$，对应绝对测深误差为 $[-1.58, -0.85]\\,\\mathrm{m}$；在较大摩阻 $k=0.05$ 下，起跳时钟微移 $+6.0\\,\\mathrm{ms}$（对应误差 $+2.78\\,\\mathrm{m}$）；
2. **波峰与倒谱时钟的响应特征：** 在常态弱摩阻 $k=0.01$ 下，波峰时钟与倒谱主峰时延一致，均为 $\\Delta t_{\\mathrm{peak}} = \\Delta\\tau_{\\mathrm{cep}} = +13.0\\,\\mathrm{ms}$，折算相对稳态偏移为：
   $$\\delta x_{\\mathrm{cep}} = \\frac{1450\\,\\mathrm{m/s} \\times 0.0130\\,\\mathrm{s}}{2} \\approx \\mathbf{+9.43\\,\\mathrm{m}}$$
   而在中等至强摩阻下（$k=0.02, 0.05$），两者呈现显著但非严格同步的滞后（如 $k=0.05$ 时波峰滞后 $+64.0\\,\\mathrm{ms}$，而倒谱峰滞后 $+50.0\\,\\mathrm{ms}$）；
3. **能量中位时刻与波形展宽：** 能量中位时刻 $t_{E50}$ 随 $k$ 的增大单调后移（$k=0.01$ 时为 $+13.0\\,\\mathrm{ms}$；$k=0.05$ 时为 $+60.6\\,\\mathrm{ms}$），反映了波包能量向后拖尾的形态演化；
4. **表观波速分化假象：** 图 3(c) 表明，由于各特征时钟在耗散过程中的滞后幅度不同，采用不同时钟反算得到的等效波速呈现分化：起跳波速 $a_{\\mathrm{onset}} \\approx 1450.6\\,\\mathrm{m/s}$ 贴近介质固有声速，而波峰表观波速降至 $a_{\\mathrm{peak}} = 1433.3\\,\\mathrm{m/s}$，驻波模态波速降至 $a_{f0} = 4Lf_0 = 1397.0\\,\\mathrm{m/s}$。若将基于峰值或模态周期的表观滞后直接归因于介质本构声速的物理折减，将掩盖流动边界层剪切弛豫的动态影响。

四特征时钟分叉是时域波形与倒频域峰位表现出的直接数值观测现象，而频选耗散是其背后的候选物理机制，将在下一节进一步讨论。

---

### 3.3 频域选择性耗散与同态衰减核

在经典倒谱理论中，信号被视作无耗散离散回声序列的对数谱线性叠加：
$$\\ln |Y(f)| \\approx \\ln |S(f)| + \\sum_{m=1}^M \\alpha_m \\cos(2\\pi f \\tau_m) \\tag{23}$$
逆傅里叶变换将周期波纹 $\\cos(2\\pi f \\tau_m)$ 变换为倒频域 Delta 尖峰。

**图 4** 时频选择性耗散特征与同态对数谱衰减核。
(a) 短时傅里叶变换（STFT）时频谱对比（$t \\in [1, 30]\\,\\mathrm{s}$）：Darcy 稳态流（上图）高频谐波贯穿全时长，非定常摩阻 $k=0.02$（下图）在 $10\\,\\mathrm{s}$ 内 $50\\,\\mathrm{Hz}$ 以上高频被强烈吸收，两者共用 $[-60, +20]\\,\\mathrm{dB}$ 色标；
(b) 频段积分衰减对比（相对于 Darcy 基准，Matrix A 缝网，$n=4, D=20\\,\\mathrm{m}$），显示低频（$0\\sim 20\\,\\mathrm{Hz}$）衰减极小（$-1.6\\,\\mathrm{dB}$），而高频（$60\\sim 150\\,\\mathrm{Hz}$）衰减剧烈（$-17.0\\,\\mathrm{dB}$）；
(c) 同态对数谱理论衰减核 $\\Delta\\ln|Y(f)| = -\\alpha(f)$ 示意图（示意性说明，非拟合曲线与非时移证明）。

![图 4](figures/Fig4_stft_homomorphic.png)

但在物理非定常流中，该假设被破坏（图 4）：
- STFT 时频谱（图 4(a)）清晰显示，Darcy 流中高达 $150\\,\\mathrm{Hz}$ 的高频谐波能量持续存在，而在非定常摩阻下，$50\\,\\mathrm{Hz}$ 以上高频能量在前 $10\\,\\mathrm{s}$ 内被耗散吸收入底；
- 频段量化积分（图 4(b)）证实：$k=0.01$ 时低频（$0\\sim 20\\,\\mathrm{Hz}$）仅衰减 **$-1.6\\,\\mathrm{dB}$**，而高频（$60\\sim 150\\,\\mathrm{Hz}$）剧烈衰减 **$-17.0\\,\\mathrm{dB}$**；$k=0.05$ 时高频衰减达到 **$-40.5\\,\\mathrm{dB}$**；
- 这一频选耗散效应使对数幅值谱呈现高频下倾趋势（图 4(c)）。需要明确的是：实倒谱 $C_y(\\tau)$ 在倒频域中满足加法分离且具备双边偶对称性，对数谱的高频下倾仅反映了频选滤波的幅值衰减特征，在数学上不构成单向因果时移卷积核的严格推导证明。倒谱峰的实际走时偏移依赖于具体流体阻力模型及其时域波形演化。

为规范本文涉及的各项测深数值与物理机理归属，表 4 将四项核心测深偏差严格绑定至各自的前向仿真与估计器协议，杜绝跨工况混称；表 5 则给出了计算声速假定对测深结果的响应方向对照。

**表 4: 关键倒谱测深偏差与工况估计器协议绑定表**

| 工况 / 阻力模型 | 关井协议 | 测深估计器 | 相对/绝对偏差 | 物理与数值含义 |
| :--- | :--- | :--- | :--- | :--- |
| **Brunone IAB ($k=0.01$)** | 阶跃关井 ($T_c=1\\,\\mathrm{ms}$) | 实倒谱 $|C(\\tau)|$ 极大值 | 相对稳态 $\\delta x_{\\mathrm{cep}} = \\mathbf{+9.43\\,\\mathrm{m}}$ | 弱非定常对流项引起的基准走时偏移 ($\\Delta\\tau = +13.0\\,\\mathrm{ms}$) |
| **Brunone IAB ($k=0.01$)** | 有限关井 ($T_c=0.2\\sim 1.0\\,\\mathrm{s}$ 斜坡) | 实倒谱 $|C(\\tau)|$ 极大值 | 相对稳态 $\\delta x_{\\mathrm{cep}} = \\mathbf{+5.80} \\sim \\mathbf{+5.08\\,\\mathrm{m}}$ | 阀门关断展宽使相对偏移收窄 ($\\Delta\\tau = +8.0 \\sim +7.0\\,\\mathrm{ms}$) |
| **动态雷诺数 $k(\\mathrm{Re})$** | 阶跃关井 ($T_c=1\\,\\mathrm{ms}$) | 一维实倒谱 $C(\\tau)$ 极大值 | 绝对误差 $\\Delta x_{\\mathrm{cep}} = \\mathbf{+11.80\\,\\mathrm{m}}$ | 强剪切非线性衰减下实倒谱正峰估计值 ($\\tau_{\\mathrm{cep}}=5.6715\\,\\mathrm{s}$) |
| **动态雷诺数 $k(\\mathrm{Re})$** | 阶跃关井 ($T_c=1\\,\\mathrm{ms}$) | 冻结模倒谱 $|C(\\tau)|$ 极大值 | 绝对误差 $\\Delta x_{\\mathrm{cep}} = \\mathbf{+18.73\\,\\mathrm{m}}$ | 负向副瓣极值被模算子捕获形成的包络偏移 ($\\tau_{\\mathrm{cep}}=5.6810\\,\\mathrm{s}$) |

*(注：所有工况均采用井口 $\\mathrm{d}H/\\mathrm{d}t$ 分析通道与全 Hann 窗截取（$t \\in [1, 50]\\,\\mathrm{s}$）；倒谱峰值检索窗口基准为 $[\\tau_{\\mathrm{geom}} \\pm 80\\,\\mathrm{ms}]$，在有限关井 $T_c$ 扫描中按代码实现自适应扩展至 $[\\tau_{\\mathrm{geom}} - 80\\,\\mathrm{ms}, \\tau_{\\mathrm{geom}} + 80\\,\\mathrm{ms} + \\min(0.5, T_c)]$ 以覆盖宽脉冲；上述四项偏差严格对应各自正演工况与估计器定义，正文分析中禁止跨工况混称)*

**表 5: 计算声速假设与测深估值响应方向对照表**

| 参数状态 / 物理假定 | 测深公式 $x_{\\mathrm{est}} = a\\,\\tau_{\\mathrm{cep}}/2$ 响应 | 物理机理与工程评价 |
| :--- | :--- | :--- |
| **人为下调计算声速** (如 $1450 \\to 1390\\,\\mathrm{m/s}$) | 在固定实测时延 $\\tau_{\\mathrm{cep}}$ 下，计算深度 $x_{\\mathrm{est}}$ **单调减小** | 虽能强制将偏深的单点估值压缩以贴近已知射孔深度，但会扭曲簇间距与深部结构 |
| **真实介质声速保持** ($a_{\\mathrm{onset}} \\approx 1450.6\\,\\mathrm{m/s}$) | 物理波前起跳点传播速度未降低，时延增加由波形耗散滞后引起 | 波速未发生介质本构衰减，时钟分叉为波包演化响应，不宜用声速折减代偿 |

*(注：若正演中流体真实物理声速 $a$ 发生降低，例如携砂或含气导致等效声速折减，则物理声波传播走时增大；这属于正演物理介质性质变化，与反演算法中人为下调假设 $a$ 的响应方向完全不同)*

### 3.4 动态同态脊线演化与系统性测深偏差分层

当水击波的高频分量被非定常摩阻强烈吸收后（图 4），在二维滑窗倒谱平面（Cepstrogram）上直接表现为同态反射脊线随分析窗向后推进而展宽模糊；在紧密缝簇条件下，相邻脊线在晚期分析窗内严重粘连融合。二维倒谱脊线拓扑演化直观揭示了耗散波包在倒频域的弥散退化过程，与后文定量表征的能量弥散时间（$\\mathrm{EST}$）、脉冲半高宽（$\\mathrm{FWHM}$）及多簇峰谷对比度（$C_v$）同方向、同机制。需要特别说明的是：**滑窗倒谱仅显示同态脊线随窗口演化趋势；测深数字仍严格以全窗一维模倒谱 $|C(\\tau)|$ 为准。**

**图 5A** 井筒非定常摩阻下二维滑窗倒谱（Cepstrogram）动态同态脊线演化。
(a) 单缝稳态流（$n=1, k=0$）：同态脊线清晰细窄，严格锁定于 $x \\approx 4099\\,\\mathrm{m}$ 理论位置；
(b) 单缝 $k=0.01$：主脊仍近几何深度，右侧拖尾增强、晚窗展宽；相对偏深 $+9.43\\,\\mathrm{m}$ 以全窗 1D 倒谱为准（图 3、图 5(a)）；
(c) 4 簇压裂段宽间距（$n=4, D=20\\,\\mathrm{m}, k=0.01$）：多条同态脊线清晰分离，保持良好空间独立性；
(d) 4 簇压裂段窄间距（$n=4, D=5\\,\\mathrm{m}, k=0.01$）：相邻裂缝脊线在晚期窗口内粘连合并为单条宽带（注：滑动 Hann 窗 $T_{\\mathrm{win}}=30\\,\\mathrm{s}$，步长 $\\mathrm{hop}=0.25\\,\\mathrm{s}$，通道 $\\mathrm{d}H/\\mathrm{d}t$，四格共用色标，红色/橙色虚线标明裂缝真值坐标 $X_i$）。

![图 5A](figures/Fig5A_cepstrogram_bridge.png)

**表 6: 二维倒谱现象与一维定量指标物理映射对照表**

| 二维倒谱现象 | 定量指标 | 对应主图 |
| :--- | :--- | :--- |
| 主脊近几何深度、能量向右拖尾 | 相对稳态偏深 $\\delta x_{\\mathrm{cep}} = +9.43\\,\\mathrm{m}$ | 图 3、图 5(a) |
| 单缝脊线随分析窗向后变宽 | $\\mathrm{EST}$ 从 $1.0\\,\\mathrm{ms} \\to 8.0\\,\\mathrm{ms}$；$\\mathrm{FWHM}$ 展宽 | 图 2(b)、图 5(c) |
| 晚期分析窗脊线对比度下降模糊 | $\\mathrm{STFT}$ 高频分量在前 $5\\sim 10\\,\\mathrm{s}$ 内耗散消失 | 图 4(a) |
| $D=5\\,\\mathrm{m}$ 脊线粘连，$D=20\\,\\mathrm{m}$ 仍可分离 | 峰谷对比度 $C_v = 0.0$ / $C_v = 1.0$（$n=4, k=0.01$） | 图 5(b) |

为消除概念混淆，图 5(a) 将系统性偏差清晰划分为相对稳态漂移与绝对真值误差两组：

**图 5** 倒谱系统性测深偏差分层与多簇缝网空间可分辨性退化。
(a) 测深偏差分层：相对稳态漂移量 $\\delta x_{\\mathrm{cep}}$（阶跃关井 $+9.43\\,\\mathrm{m}$，斜坡关井 $T_c=0.2\\,\\mathrm{s}$ 时 $+5.80\\,\\mathrm{m}$）与动态雷诺数 $k(Re)$ 下绝对测深误差 $\\Delta x_{\\mathrm{cep}}$（一维实倒谱 $+11.80\\,\\mathrm{m}$，模倒谱 $+18.73\\,\\mathrm{m}$），以虚线物理隔断；
(b) 4 簇射孔缝网（Matrix A，$n=4$）时域波峰对比度 $C_v$ 随簇间距 $D \\in [5, 100]\\,\\mathrm{m}$ 的演化规律，展示 $k=0.01$ 时 $D=5\\,\\mathrm{m}$ 发生融合（$C_v=0$），$D \\ge 20\\,\\mathrm{m}$ 可分；
(c) 反射波包退化双轴对比（$D=20\\,\\mathrm{m}$）：归一化峰值梯度 $\\mathrm{d}H/\\mathrm{d}t|_{\\max}/A_0$（紫色左轴）骤降至 $0.05$，归一化脉冲半高宽 $\\mathrm{FWHM}/\\mathrm{FWHM}_0$（珊瑚色右轴）展宽超过 8 倍。

![图 5](figures/Fig5_failure_modes.png)

#### 1. 相对稳态漂移（$\\delta x_{\\mathrm{cep}}$）与有限关井时间响应
以理想无耗散 Darcy 稳态基准（$k=0$）为参照，单缝在阶跃关井下产生的倒谱峰位偏移为：
$$\\delta x_{\\mathrm{cep}} = \\frac{1450\\,\\mathrm{m/s} \\times (+13.0\\,\\mathrm{ms})}{2} \\approx \\mathbf{+9.43\\,\\mathrm{m}}$$
当阀门关井时间延长为真实有限时间斜坡时，激振源频谱高频截断使壁面剪切耗散率相对下降，相对稳态偏深单调收窄：$T_c = 0.05\\,\\mathrm{s}$ 时为 $+7.25\\,\\mathrm{m}$，$T_c = 0.2\\,\\mathrm{s}$ 时为 $\\mathbf{+5.80\\,\\mathrm{m}}$，$T_c = 1.0\\,\\mathrm{s}$ 时为 $\\mathbf{+5.08\\,\\mathrm{m}}$。这表明有限关井时间改变了相对偏深的幅度，在 $T_c \\ge 200\\,\\mathrm{ms}$ 时收敛至 $+5.80\\sim +5.08\\,\\mathrm{m}$，未随关井时间进一步延长而消失。

#### 2. 动态雷诺数依赖下的绝对测深误差（$\\Delta x_{\\mathrm{cep}}$）
在局部瞬时雷诺数动态闭合模型 $k(\\mathrm{Re})$ 下，由于初始雷诺数高达 $1.397 \\times 10^5$，早期紊流剪切导致极强的高频衰减。此时提取的一维实倒谱正峰位置为 $\\tau_{\\mathrm{cep}} = 5.6715\\,\\mathrm{s}$，对应绝对测深误差为：
$$\\Delta x_{\\mathrm{cep}} = \\frac{1450\\,\\mathrm{m/s} \\times (5.6715 - 1.0)\\,\\mathrm{s}}{2} - 4100\\,\\mathrm{m} = \\mathbf{+11.80\\,\\mathrm{m}}$$
若采用冻结模倒谱 $|C(\\tau)|$ 极大值拾取算法，强非线性耗散引发的负向副瓣极值被模算子转为正向峰值，拾取到的峰位进一步右移至 $\\tau_{\\mathrm{cep}} = 5.6810\\,\\mathrm{s}$，计算绝对误差达 $\\mathbf{+18.73\\,\\mathrm{m}}$。

图 5(a) 通过物理隔断明确指出：$+9.43\\,\\mathrm{m}$ 与 $+5.80\\,\\mathrm{m}$ 属于常数弱摩阻下的相对稳态漂移 $\\delta x_{\\mathrm{cep}}$；而 $+11.80\\,\\mathrm{m}$ 与 $+18.73\\,\\mathrm{m}$ 则属于动态强剪切 $k(\\mathrm{Re})$ 下不同倒谱算法实现产生的绝对误差 $\\Delta x_{\\mathrm{cep}}$。

#### 3. 缝网空间可分辨性退化（$C_v$）
针对 4 簇射孔缝网（Matrix A，$n=4$），定义相邻反射波峰间的时间谷值对比度：
$$C_v = \\frac{A_{\\mathrm{peak}} - A_{\\mathrm{valley}}}{A_{\\mathrm{peak}}} \\tag{24}$$
图 5(b) 给出了对比度随簇间距 $D$ 的演化规律：
- 当 $k=0$ 时，即便在 $D = 5\\,\\mathrm{m}$ 极窄间距下，回声波峰依然陡峭分立，$C_v = 1.0$；
- 引入常态弱非定常摩阻（$k=0.01$）后，由于脉冲半高宽展宽超过 8 倍（图 5(c)），在 $D = 5\\,\\mathrm{m}$ 时相邻反射波峰完全重叠，$C_v = 0.00$；当簇间距增大至 $D \\ge 20\\,\\mathrm{m}$ 时，波谷对比度回升至 $C_v \\ge 0.95$。

---

### 3.5 井筒摩阻与地层滤失的相对阻尼灵敏度响应面

为考察井筒非定常摩阻与裂缝滤失在能量衰减维度的相互影响，本文计算了全系统一阶衰减阻尼比 $\\zeta$（基于井口波形的连续小波变换 CWT 脊线提取）以及三大模型压力测试对比（图 6）。

**图 6** 井筒摩阻与地层滤失的相对阻尼灵敏度响应面及压力测试。
(a) 阻尼比 $\\zeta(k, k_{\\mathrm{leak}})$ 响应曲面（单缝 $n=1, H_{\\mathrm{ext}}=100\\,\\mathrm{m}$），展示常态弱摩阻下井筒阻尼仅贡献 $+7.0\\%$，与大滤失响应存在明显灵敏度差异；
(b) 关井历时扫描：相对偏深 $\\delta x_{\\mathrm{cep}}$ 从阶跃的 $+9.43\\,\\mathrm{m}$ 单调收敛至 $+5.08\\,\\mathrm{m}$；
(c) 阻力模型对比：Darcy（$\\delta x = 0.00\\,\\mathrm{m}$）、Brunone IAB（$\\delta x = +9.43\\,\\mathrm{m}$）与 Vardy–Brown WFB（$\\delta x = 0.00\\,\\mathrm{m}$）。

![图 6](figures/Fig6_damping_surface.png)

响应面分析表明：
1. **相对阻尼灵敏度差异：** 在常态弱非定常摩阻（$k=0.01$）下，井筒摩阻引起的阻尼比增量仅为 $+7.0\\%$，远低于缝口中高滤失引起的阻尼增长；井筒摩阻增量与滤失阻尼增量相当的现象仅在参数极端上界（$k=0.05, k_{\\mathrm{leak}}=5\\times 10^{-4}\\,\\mathrm{m^2/s/\\sqrt{m}}$，分别贡献 $+55\\%$ 与 $+49\\%$）成立；
2. **有限关井时间的单调收敛：** 图 6(b) 证实，随着关井时间 $T_c$ 延长，入射脉冲高频成分削弱，倒谱相对偏移从 $+9.43\\,\\mathrm{m}$ 单调下降并收敛至 $+5.08\\,\\mathrm{m}$；
3. **模型形式不确定性（Model Form Uncertainty）：** 图 6(c) 的对比揭示出重大机理差异：在相同的离散网格与物性参数下，严格求解一维扩散卷积方程的 Vardy–Brown WFB 模型预测的倒谱测深偏差为 $\\delta x_{\\mathrm{cep}} = \\mathbf{0.00\\,\\mathrm{m}}$，其波形峰值与倒谱峰位均未出现 IAB 模型的系统性右移。这一结果表明：**水击波倒谱测深的理论偏深现象在很大程度上取决于一维非定常摩阻经验闭合形式的选择**。

---

## 4. 讨论与工程建议

### 4.1 特征时钟分叉对等效波速假定的澄清
长期以来，压裂诊断工程界将倒谱反演得到的偏深估值解释为“携砂含气或多孔介质引起的流体等效声速折减”（通常下调 $3\\%\\sim 8\\%$）。

本文的四特征时钟追踪与机理分析证实：
1. **介质声速未发生本构折减：** 波前起跳时钟 $t_{\\mathrm{onset}}$ 在常态弱摩阻下（$k \\le 0.02$）保持稳定（时延 $\\le 1.0\\,\\mathrm{ms}$，对应波速 $a_{\\mathrm{onset}} \\approx 1450.6\\,\\mathrm{m/s}$），证明流体-管柱系统的固有物理声速在传播过程中并未衰减；
2. **偏深本质为波形耗散滞后：** 倒谱峰的右移源于管壁剪切弛豫对高频波成分的选择性吸收，导致波峰几何中心与倒频域能量极大值发生后移；
3. **经验声速折减的潜在危害：** 人为下调计算声速虽可在单点数值上将首条裂缝拉回射孔位置，但该修正属于全局线性缩放，无法校正非定常耗散带来的波包展宽与簇间分辨率退化，在多簇压裂段将导致簇间距与深部裂缝几何反演的严重畸变。

### 4.2 倒谱测深在压裂诊断中的修正准则与建议
针对非定常摩阻引发的测深偏差，提出以下数值与解释建议：
1. **关注起跳时钟与波峰时钟的分化：** 在快关井且信噪比良好（中高 SNR）的数值工况下，波前起跳时钟（$t_{\\mathrm{onset}}$）受非定常耗散扰动显著小于波峰与倒谱时钟（$k \\le 0.02$ 下时延 $\\le 1.0\\,\\mathrm{ms}$），但起跳检测对噪声与微小阈值敏感，在现场应用中宜作为走时参考而非唯一绝对校准基准；
2. **避免盲目套用固定距离修正：** 本文揭示的 $+5\\sim +9\\,\\mathrm{m}$（IAB 模型）或 $+11.8\\sim +18.7\\,\\mathrm{m}$（动态雷诺数模型）测深偏差高度依赖于关井时间 $T_c$、非定常闭合形式及倒谱极大值提取协议，不能作为普适现场常数直接从测深结果中简单扣减；
3. **建立全波包非稳态正演匹配框架：** 建议从单一峰位时延反演转向包含非定常摩阻的 MOC 全波形匹配或频域联合反演，消除波形弥散对几何位置诊断的干扰。

### 4.3 压裂缝网诊断的实际可分辨能力评价
在多簇密集压裂段（Matrix A，$n=4$），非定常摩阻引起的波包半高宽展宽（超过 8 倍）对段内多裂缝的几何识别提出了挑战：
1. **窄间距下的波包融合：** 在常态弱摩阻下（$k=0.01$），当簇间距 $D = 5\\,\\mathrm{m}$ 时，相邻反射波峰在时域完全重叠淹没（$C_v = 0.00$），二维倒谱脊线融合成单条宽带；
2. **波包弥散度量特征：** 当簇间距增大至 $D \\ge 20\\,\\mathrm{m}$ 时，时域对比度回升至 $C_v \\ge 0.95$；$C_v$ 随簇间距退化反映的是波包弥散程度，不应将其外推为普遍物理分辨率硬阈值。

---

## 5. 结论

本文针对水平井压裂水击瞬变压力倒谱诊断中的系统性测深偏差，建立了考虑非定常摩阻的一维水力瞬变特征线求解与多时钟追踪体系，取得如下主要结论：

1. **揭示了特征时钟分叉现象：** 在非定常壁面剪切作用下，水击波波前起跳时钟在常态弱摩阻下（$k \\le 0.02$）保持稳定（时延 $\\le 1.0\\,\\mathrm{ms}$），而波峰、能量中位时刻与倒谱时钟呈现显著右偏；在 Brunone IAB 弱摩阻（$k=0.01$）阶跃工况下，倒谱时钟产生 $+13.0\\,\\mathrm{ms}$ 滞后，导致相对稳态偏深约 $+9.43\\,\\mathrm{m}$，证实所谓“等效声速降低”本质上是特征时钟分化与波包弥散的观测效应；
2. **明确了摩阻模型的形式不确定性：** 经典 Vardy–Brown WFB 权重函数卷积模型预测的倒谱测深偏差为 $0.00\\,\\mathrm{m}$，未复现 IAB 模型的偏深现象，表明一维井筒倒谱测深偏差在很大程度上受制于经验加速度闭合形式的选择，具有显著的模型依赖性；
3. **阐明了关井历时的耦合调控规律：** 有限关井历时（$T_c = 0.05\\sim 1.0\\,\\mathrm{s}$）改变了偏深幅度，使倒谱相对偏深从阶跃关井的 $+9.43\\,\\mathrm{m}$ 单调收敛至 $+5.08\\,\\mathrm{m}$，表明实测时延漂移为激振源谱与井筒耗散传播的耦合响应；
4. **厘清了摩阻与滤失的相对灵敏度差异：** 井筒壁面摩阻与地层滤失对系统总衰减的相对贡献取决于参数区间；在常态弱摩阻下，井筒非定常项引起的阻尼增量（$+7.0\\%$）远小于有效滤失响应，仅在参数网格极端大值上界两者阻尼增量才具有相当量级（$+55\\%$ vs $+49\\%$），表现出明显的相对灵敏度差异，但弱摩阻下的较小占比并不构成两者的独立解耦证明。

---

## 符号表 (Nomenclature)

- $a$ = 压力波在流体-井筒系统中的传播声速，$\\mathrm{m/s}$
- $A$ = 井筒横截面积，$\\mathrm{m^2}$
- $C(\\tau)$ = 信号实倒谱序列，无量纲
- $C_f, C_H$ = 裂缝集总水力柔量（水头形式），$\\mathrm{m^2}$
- $C_{\\mathrm{frac}}$ = 裂缝压力形式水力柔量，$\\mathrm{m^3/Pa}$
- $C_v$ = 多簇相邻反射波峰谷对比度，无量纲
- $D$ = 套管内径或多簇射孔缝间距，$\\mathrm{m}$
- $f$ = 达西-韦斯巴赫稳态摩阻系数，无量纲
- $g$ = 重力加速度，取 $9.81\\,\\mathrm{m/s^2}$
- $H$ = 测压管水头，$\\mathrm{m}$
- $H_{\\mathrm{ext}}$ = 远场孔隙压力水头，$\\mathrm{m}$
- $k$ = Brunone 无量纲非定常摩阻系数
- $k_{\\mathrm{leak}}$ = 裂缝非线性平方根滤失系数，$\\mathrm{m^2/s/\\sqrt{m}}$
- $L$ = 井筒总长度，$\\mathrm{m}$
- $Q_f$ = 裂缝侧向注入瞬态流量，$\\mathrm{m^3/s}$
- $\\mathrm{Re}$ = 瞬时流动雷诺数，无量纲
- $t$ = 时间变量，$\\mathrm{s}$
- $t_s$ = 关井起始时刻，$\\mathrm{s}$
- $T_c$ = 阀门关断历时，$\\mathrm{s}$
- $t_{\\mathrm{onset}}$ = 波前起跳特征时钟时刻，$\\mathrm{s}$
- $t_{\\mathrm{peak}}$ = 反射波峰特征时钟时刻，$\\mathrm{s}$
- $t_{E50}$ = 累计能量 $50\\%$ 中位时刻时钟，$\\mathrm{s}$
- $V$ = 井筒断面平均流速，$\\mathrm{m/s}$
- $X_1$ = 首条水力裂缝物理深度，$\\mathrm{m}$
- $x_{\\mathrm{est}}$ = 水击波反演估计深度，$\\mathrm{m}$
- $\\Delta t$ = MOC 时间离散步长，$\\mathrm{s}$
- $\\Delta z$ = MOC 空间网格步长，$\\mathrm{m}$
- $\\Delta x$ = 测深估计值相对于物理真值的绝对误差，$\\mathrm{m}$
- $\\delta x_{\\mathrm{cep}}$ = 相对稳态 Darcy 基准（$k=0$）的倒谱测深漂移量，$\\mathrm{m}$
- $\\tau$ = 倒频域 Quefrency 时延变量，$\\mathrm{s}$
- $\\tau_{\\mathrm{cep}}$ = 倒谱主峰对应的时延时钟，$\\mathrm{s}$
- $\\tau_w$ = 管壁总剪切应力，$\\mathrm{Pa}$
- $\\rho$ = 流体密度，$\\mathrm{kg/m^3}$
- $\\nu$ = 流体运动粘度，$\\mathrm{m^2/s}$
- $\\zeta$ = 基于连续小波变换（CWT）提取的系统一阶衰减阻尼比，无量纲

---

## 参考文献 (References)

1. Bergant, A., Simpson, A. R., and Vitkovsky, J. (2008). Developments in Unsteady Pipe Flow Friction Modelling. *Journal of Hydraulic Research*, 46(sup1), 144–157. DOI: [10.1080/00221686.2008.9521953](https://doi.org/10.1080/00221686.2008.9521953)
2. Brunone, B., Golia, U. M., and Greco, M. (1991). Some Remarks on the Momentum Equations for Fast Transients. In *Proceedings of the 6th International Conference on Pressure Surges*, BHR Group, Cranfield, UK, 201–209.
3. Brunone, B. (2000). Velocity Profiles and Unsteady Pipe Friction in Transient Flow. *Journal of Water Resources Planning and Management*, 126(4), 236–244. DOI: [10.1061/(ASCE)0733-9496(2000)126:4(236)](https://doi.org/10.1061/(ASCE)0733-9496(2000)126:4(236))
4. Childers, D. G., Skinner, D. P., and Kemerait, R. C. (1977). The Cepstrum: A Guide to Processing. *Proceedings of the IEEE*, 65(10), 1428–1443. DOI: [10.1109/PROC.1977.10747](https://doi.org/10.1109/PROC.1977.10747)
5. Dong, X., Zhu, H., Wang, X. et al. (2024). Multi-Fracture Parameter Inversion Method for Horizontal Wells Based on Water Hammer Pressure Wave. *Energy*, 290, 130180. DOI: [10.1016/j.energy.2023.130180](https://doi.org/10.1016/j.energy.2023.130180)
6. Gabry, M., Triki, A., and Trabelsi, M. (2025). Experimental Investigation of Unsteady Friction and Dynamic Leakage Effects in Viscoelastic Pipes. *Ocean Engineering*, 318, 120150. DOI: [10.1016/j.oceaneng.2024.120150](https://doi.org/10.1016/j.oceaneng.2024.120150)
7. Ghidaoui, M. S., Zhao, M., McInnis, D. A. et al. (2005). A Review of Water Hammer Theory and Practice. *Applied Mechanics Reviews*, 58(1), 49–76. DOI: [10.1115/1.1828050](https://doi.org/10.1115/1.1828050)
8. Qiu, Y., Hu, X., Zhou, F. et al. (2022). Hydraulic Fracture Diagnosis Using High-Frequency Water Hammer Pressure Waves in Fracturing Operations. *Journal of Petroleum Science and Engineering*, 215, 110425. DOI: [10.1016/j.petrol.2022.110425](https://doi.org/10.1016/j.petrol.2022.110425)
9. Sun, B., Zhang, L., Wang, Z. et al. (2025). Comprehensive Modeling of Pressure Transient Propagation in Multistage Fractured Wellbores. *SPE Journal*, 30(01), 112–128. DOI: [10.2118/218000-PA](https://doi.org/10.2118/218000-PA)
10. Vardy, A. E., and Brown, J. M. B. (2003). Transient Turbulent Friction in Smooth Approach Pipes. *Journal of Sound and Vibration*, 259(5), 1011–1036. DOI: [10.1006/jsvi.2002.5160](https://doi.org/10.1006/jsvi.2002.5160)
11. Vardy, A. E., and Brown, J. M. B. (2004). Transient Turbulent Friction in Fully Rough Pipes. *Journal of Sound and Vibration*, 270(1-2), 233–257. DOI: [10.1016/S0022-460X(03)00492-5](https://doi.org/10.1016/S0022-460X(03)00492-5)
12. Vitkovsky, J. V., Lambert, M. F., and Simpson, A. R. (2000). Advances in Unsteady Friction Modelling in Transient Pipe Flow. In *Proceedings of the 8th International Conference on Pressure Surges*, The Hague, The Netherlands, 471–482.
13. Wylie, E. B., and Streeter, V. L. (1993). *Fluid Transients in Systems*. Prentice Hall, Englewood Cliffs, NJ.
"""

with open(target, "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")

print(f"File successfully written to {target} with size {len(content)} characters.")
