# 井筒倒谱测深中特征时钟分叉的非定常摩阻模型依赖性

**摘要**：基于水击压力波的同态倒谱分析是水平井多段压裂裂缝几何定位的重要方法。现有诊断理论通常基于理想无耗散声波传播假设，将实测倒谱峰位的滞后简单归因于流体-岩石系统的等效声速折减。本文实现了基于 Brunone 局部加速度项的离散闭合（IAB），并以十项指数近似 Vardy–Brown 权函数的递推形式作为 WFB 对照，建立了一维特征线法（MOC）数值正演与多时钟后处理框架，系统追踪波前起跳时钟（$t_{\mathrm{onset}}$）、波形峰值时钟（$t_{\mathrm{peak}}$）、能量中位时刻时钟（$t_{E50}$）以及模倒谱峰时钟（$\tau_{\mathrm{cep}}=\arg\max |C|$）在管壁非定常耗散下的演化。研究表明：在本文 IAB 离散闭合下，特征时钟发生分叉——波前起跳点在常数弱摩阻（$k \le 0.02$）下保持稳定（走时偏差 $\le 1.0\,\mathrm{ms}$），而波峰、能量中位时刻与模倒谱时钟显著右偏；IAB $k=0.01$ 阶跃工况下，模倒谱时钟相对稳态 Darcy 基准滞后 $+13.0\,\mathrm{ms}$（约 $+9.43\,\mathrm{m}$）。该偏移具有模型形式不确定性：在本文 $1\,\mathrm{ms}$ 时间离散与指定 $|C|$ 拾峰协议下，WFB 相对于 Darcy 的倒谱偏移低于约一个时间步（$|\delta x_{\mathrm{cep}}|\lesssim 0.73\,\mathrm{m}$），未复现 IAB 同量级偏深。有限关井在所考察的四个 $T_c$ 取值上，相对偏移约介于 $+5.08\,\mathrm{m}$ 与 $+9.43\,\mathrm{m}$，为源谱与传播的耦合总响应。弱摩阻下井筒阻尼增量约 $+7.0\%$，与滤失相当仅在参数网格极端上界成立。所谓“等效声速折减”在本文 IAB 算例中应解释为特征时钟分化，而不能外推为全部非定常闭合的普遍规律。

**关键词**：水击波；同态倒谱；特征时钟分叉；非定常摩阻；水力压裂；模型形式不确定性

---

## 1. 引言

在非常规油气藏水力压裂施工中，实时获取井下射孔簇的开启状态、水力裂缝位置及缝口柔量对于评价段内均匀压裂效果至关重要（Qiu et al., 2022; Dong et al., 2024）。在各类非侵入式监测技术中，水击瞬变压力分析法利用泵关断瞬态激发的高频水击压力波在井筒与水力裂缝间的反射回声信号，通过井口高频压力记录反演井下裂缝参数，因其无需起下井下仪器、成本低廉且具备施工全过程覆盖能力而受到广泛关注（Sun et al., 2025）。

为从强噪声与多重反射干扰的井口水击响应中准确提取微弱的回声走时，同态解卷积（Homomorphic Deconvolution）与实倒谱（Real Cepstrum）分析技术被引入压裂诊断领域（Childers et al., 1977）。倒谱分析通过对压力信号的对数幅值谱进行逆傅里叶变换，将频域中的周期性回声纹理映射为倒频域（Quefrency）中孤立的 Delta 峰值，从而实现回声时延 $\tau$ 的高分辨率估计。随后利用平面声波理论测深公式：
$$x = \frac{a\,\tau}{2} \tag{1}$$
即可直接计算反射断面的空间几何深度。式中 $a$ 为压力波在流体-井筒系统中的传播声速。

然而，在压裂现场实际应用中，倒谱反演得到的裂缝深度往往大于套管射孔施工记录的真实物理深度；文献中偶有报道数米至二十米量级的偏深，**该区间属于既有文献的经验观察，不是本文的普适数值结论，也不得当作现场统一修正量**。针对这一偏差，现场常采用“等效声速折减”——假设含砂流体或井筒微结构使声速从清水理论值（约 $1450\,\mathrm{m/s}$）降至 $1380\sim 1400\,\mathrm{m/s}$，从而把测深结果压回射孔位置。

尽管这种经验折减能在单点几何标定上取得形式上的吻合，其物理合理性仍待检验。本文只提出两个研究问题：

1. **RQ1（特征时钟分叉）：** 在指定井筒几何与关井边界下，非定常摩阻是否使波前起跳（$t_{\mathrm{onset}}$）、波峰（$t_{\mathrm{peak}}$）、能量中位时刻（$t_{E50}$）与模倒谱峰（$\tau_{\mathrm{cep}}=\arg\max |C|$）产生不同响应？若发生分叉，倒谱右移应解释为介质本构波速降低，还是波形弥散引起的特征时钟分化？
2. **RQ2（模型形式与边界响应）：** 该时钟偏移对摩阻闭合形式（Brunone IAB 与 Vardy–Brown WFB）、关井源函数以及滤失等边界参数是否保持同一方向与量级？

本文贡献限于三点：（i）建立四时钟比较框架；（ii）量化 Brunone IAB 条件下的时钟分叉与指定倒谱估计器偏深；（iii）通过 WFB 对照揭示该偏深的模型形式不确定性。频选耗散、动态 $k(\mathrm{Re})$ 下两种倒谱实现、以及摩阻–滤失相对灵敏度，均作为 RQ2 的参数敏感性分析，不另立研究问题。

**图 1** 井筒水击倒谱测深的特征时钟框架（示意图，非拟合波形、非卷积证明）。
(a) 井口关井激发、裂缝反射与壁面非定常剪切示意（几何基准 $X_1=4100\,\mathrm{m}$）；
(b) 四种特征时钟定义示意：起跳 $t_{\mathrm{onset}}$、波峰 $t_{\mathrm{peak}}$、能量中位时刻 $t_{E50}$ 与模倒谱 $\tau_{\mathrm{cep}}^{|C|}$；IAB 弱摩阻下后三类时钟相对起跳右偏，定量见 §3.3；
(c) Childers 梳状谱假设在频选耗散下的崩塌示意。

![图 1](figures/Fig1_conceptual_framework.png)

---

## 2. 水击波非定常摩阻正演模型与时钟定义

### 2.1 控制方程与非定常壁面剪切闭合

在微可压缩液体与线弹性井壁圆柱形管道中，考虑非定常壁面剪切应力的一维连续性方程与动量守恒方程表述为（Wylie and Streeter, 1993; Ghidaoui et al., 2005）：
$$\frac{\partial H}{\partial t} + \frac{a^2}{g} \frac{\partial V}{\partial z} = 0 \tag{2}$$
$$\frac{\partial V}{\partial t} + g \frac{\partial H}{\partial z} + \frac{4\tau_w}{\rho D} = 0 \tag{3}$$
式中 $H(z, t)$ 为测压管水头（$\mathrm{m}$）；$V(z, t)$ 为断面平均流速（$\mathrm{m/s}$）；$a$ 为压力波在流体-管柱系统中的传播声速（基准值取 $1450\,\mathrm{m/s}$）；$g = 9.81\,\mathrm{m/s^2}$ 为重力加速度；$D = 0.1397\,\mathrm{m}$ 为 5.5 英寸套管内径；$\rho = 1000\,\mathrm{kg/m^3}$ 为流体密度；$\tau_w$ 为壁面总剪切应力。

总剪切应力分解为稳态拟 Darcy 项 $\tau_{ws}$ 与非定常剪切项 $\tau_{wu}$ 之和：
$$\tau_w = \tau_{ws} + \tau_{wu} = \frac{1}{8}\rho f V|V| + \tau_{wu} \tag{4}$$
式中 $f$ 为达西稳态摩阻系数，在紊流下通过 Zigrand–Swamee（Swamee–Jain 族）显式公式根据雷诺数与相对粗糙度计算确定。

针对非定常项 $\tau_{wu}$，本文对比两类一维闭合，并明确：**下列数值结果代表本文所定义的离散闭合，而非所有标准 IAB/WFB 实现的普适响应。**

#### 1. 基于瞬时对流加速度的 IAB 离散闭合（Brunone et al., 1991, 2000; Vitkovsky et al., 2000）
连续形式上，一维动量方程中局部与对流加速度的经验加权闭合写为：
$$\tau_{wu} = \frac{\rho D k}{4} \left( \frac{\partial V}{\partial t} + a \cdot \operatorname{sgn}(V) \left| \frac{\partial V}{\partial z} \right| \right) \tag{5}$$
式中 $k$ 为无量纲 Brunone 摩阻系数。特征线方向的非定常增量在代码中取
$$J_u = \frac{k}{2}\,\Delta t\left[ \frac{\partial V}{\partial t} + a\,\tanh\!\left(\frac{V}{V_s}\right)\left|\frac{\partial V}{\partial z}\right| \right],\qquad V_s=0.05\,\mathrm{m/s} \tag{6}$$
即以 $\tanh(V/V_s)$ 平滑替代 $\operatorname{sgn}(V)$，消除 $V\approx 0$ 处的符号跳变。对裂缝节点邻域，因断面速度存在跃变、跨缝 $\partial V/\partial z$ 不可靠，将对应特征脚点的 $J_u$ 强制置零（式 7）：
$$J_u\big|_{\text{fracture-adjacent feet}} = 0 \tag{7}$$
因此本文 IAB 是带速度符号平滑与裂缝邻域屏蔽的局部加速度型离散闭合，不是未经修改的 $\operatorname{sgn}(V)$ 公式的逐字实现。基准算例取常数弱摩阻 $k = 0.01$，并扫描 $k \in \{0, 0.01, 0.02, 0.05\}$。

#### 2. 十项指数近似的 WFB 权函数递推（Vardy and Brown, 2003, 2004）
连续形式上，径向动量扩散给出卷积目标
$$\tau_{wu}(t) = \frac{2\rho \nu}{R} \int_0^t \frac{\partial V}{\partial t}(t - t') W(t') \,\mathrm{d}t' \tag{8}$$
式中 $W(t')$ 为 Vardy–Brown 紊流剪切权重函数。数值实现并不采用解析卷积，而是对目标 $W(\tau)\approx C^*/\sqrt{\tau}$ 作十项指数拟合，并以递推状态更新非定常增量：
$$W(\tau) \approx \sum_{j=1}^{10} m_j \exp(-n_j \tau) \tag{9}$$
式中 $\tau = \nu t / R^2$ 为无量纲扩散时间。WFB 在裂缝邻域同样将非定常增量置零。该对照用于检验倒谱偏移是否随闭合形式改变；**未**按总衰减率或首波幅值重新匹配 IAB 与 WFB，故二者不是衰减对齐后的公平比较，也不代表一切 WFB 实现在全部频率上只有幅值吸收、绝无相位效应。

### 2.2 裂缝与井底水力边界条件

在压裂井筒中，水力裂缝被建模为井筒断面处的侧向集总水力分支（Qiu et al., 2022; Sun et al., 2025）：
$$V_L(x_f, t) - V_R(x_f, t) = \frac{Q_f(t)}{A} = \frac{C_H}{A} \frac{\mathrm{d}H_f}{\mathrm{d}t} + \frac{k_{\mathrm{leak}}}{A} \sqrt{\max(0, H_f - H_{\mathrm{ext}})} \tag{10}$$
式中 $V_L$ 与 $V_R$ 分别为裂缝节点上、下游断面的流速；$Q_f(t)$ 为注入单条裂缝的总瞬态流量；$A = \pi D^2 / 4$ 为井筒截面积；$C_H$ 为水头定义下的裂缝集总水力柔量（基准值取 $C_H = 1.0 \times 10^{-5}\,\mathrm{m^2}$，对应压力柔量 $C_{\mathrm{frac}} = C_H / (\rho g) \approx 1.019 \times 10^{-9}\,\mathrm{m^3/Pa}$）。滤失项采用**平方根压差型准稳态经验边界** $Q_{\mathrm{leak}}=k_{\mathrm{leak}}\sqrt{\Delta H}$，**不是**经典 Carter $1/\sqrt{t}$ 时间滤失模型。由 $[Q]=\mathrm{m^3/s}$ 与 $[\sqrt{H}]=\mathrm{m^{1/2}}$，系数单位取
$$[k_{\mathrm{leak}}]=\mathrm{m^{5/2}/s}$$
基准值 $k_{\mathrm{leak}} = 1.0 \times 10^{-4}\,\mathrm{m^{5/2}/s}$；$H_{\mathrm{ext}} = 100.0\,\mathrm{m}$ 为远场恒定孔隙水头。

含滤失裂缝的严格稳态初场需同时满足沿程流量递减、各缝局部水头与滤失流量、以及井口—井底质量守恒，须迭代求解 $V(z)$。当前正演采用简化初场：沿井近似均匀 $V\approx V_0$，趾端恒定水头，**未**闭合上述质量守恒迭代。因此 §3.10 的 $k\times k_{\mathrm{leak}}$ 结果应读作该简化初场下的**相对敏感性测试**，而不是严格稳态初始化后的定量基准。

井底（$z = L = 5000\,\mathrm{m}$）设定为恒定压力水头边界（$H(L, t) = H_0 = 300\,\mathrm{m}$），模拟水平段盲板或深部地层稳定压力连通。

### 2.3 井口关井激励与有限关井时间源函数

井口（$z = 0$）受控于高压压裂泵关断水力瞬变。激振控制量为断面速度边界条件：
$$V(0, t) = \begin{cases} 
V_0, & t < t_s \\
V_0 \left(1 - \dfrac{t - t_s}{T_c}\right), & t_s \le t \le t_s + T_c \\
0, & t > t_s + T_c
\end{cases} \tag{11}$$
式中 $T_c$ 为阀门关断历时。数值实现中，所谓的“阶跃关井”（$T_c = 1\,\mathrm{ms}$）在代码中严格为跨越单个时间离散步长 $\Delta t = 1.0\,\mathrm{ms}$ 的快速线性关断。

在有限关井时间扫描中（$T_c \in [50, 1000]\,\mathrm{ms}$），关井终值流速始终为 0，改变的是入射压力脉冲的上升沿斜率与频带宽度。关井历时改变入射波包频谱，并与井筒传播耗散共同决定倒谱时延，定量结果见 §3.5。

### 2.4 特征线法（MOC）数值离散与基准工况

沿特征线 $\mathrm{d}z/\mathrm{d}t = \pm a$ 将控制方程转化为常微分相容方程，并满足 Courant 准则：
$$\mathrm{Cr} = \frac{a\,\Delta t}{\Delta z} = 1.0 \tag{12}$$
井筒全长 $L = 5000\,\mathrm{m}$，声速 $a = 1450\,\mathrm{m/s}$，空间步长 $\Delta z = 1.45\,\mathrm{m}$，时间步长 $\Delta t = 1.0\,\mathrm{ms}$（网格节点数 $N = 3448$，与 §3.1 基准档一致）。初始稳定流动工况冻结为：$V_0 = 1.0\,\mathrm{m/s}$，井口测压水头 $H_0 = 300\,\mathrm{m}$，首条裂缝深度 $X_1 = 4100\,\mathrm{m}$。

### 2.5 网格映射与时间步设计

空间网格数按 $N = \operatorname{round}(L / (a\Delta t))$ 计算，实际空间步长为 $\Delta z = L / N$，裂缝物理位置 $X_1 = 4100\,\mathrm{m}$ 映射至最近网格节点 $i_f = \operatorname{round}(X_1 / \Delta z)$，未采用连续插值。敏感性计算另取 $\Delta t \in \{0.5, 1.0, 2.0\}\,\mathrm{ms}$（$\mathrm{Cr}=1$），各档几何映射如下；四时钟数值结果见 §3.1 表 3。

| 时间步长 $\Delta t$ | 空间网格数 $N$ | 空间步长 $\Delta z$ | 裂缝就近节点 $i_f$ | 映射物理位置 | 几何截断误差 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.5\,\mathrm{ms}$** | 6897 | $0.724953\,\mathrm{m}$ | 5656 | **$4100.33\,\mathrm{m}$** | **$+0.33\,\mathrm{m}$** |
| **$1.0\,\mathrm{ms}$** | 3448 | $1.450116\,\mathrm{m}$ | 2827 | **$4099.19\,\mathrm{m}$** | **$-0.81\,\mathrm{m}$** |
| **$2.0\,\mathrm{ms}$** | 1724 | $2.900232\,\mathrm{m}$ | 1414 | **$4100.93\,\mathrm{m}$** | **$+0.93\,\mathrm{m}$** |

各档均采用单步线性关断（$T_c = \Delta t$ 的 `velocity_step` 协议）。倒谱测深量子约为 $a\Delta t/2$（基准档 $0.725\,\mathrm{m}$）。本文不将 MOC 表述为“高精度求解体系”；网格对比只检验同号同量级，不作收敛阶声明。

### 2.6 信号处理管线与四种特征时钟定义

为杜绝后处理口径歧义，信号处理流程严格冻结如下：

#### 1. 分析通道与同态解卷积加法分离原理
全篇计算均采用井口水头变化率 $\mathrm{d}H/\mathrm{d}t$ 通道，自动消除低频静水压漂移。

在频域中，井口实测水击信号 $Y(f)$ 可表示为激振源项 $S(f)$ 与井筒-裂缝系统传递函数 $H(f)$ 的乘积：
$$Y(f) = S(f) H(f) \tag{13}$$
对幅值谱取自然对数，频域乘积关系转化为对数谱的线性叠加：
$$\ln |Y(f)| = \ln |S(f)| + \ln |H(f)| \tag{14}$$
进而通过逆傅里叶变换将对数谱映射至倒频域（Quefrency），定义实倒谱 $C_y(\tau)$：
$$C_y(\tau) = \mathcal{F}^{-1}\left\{ \ln |Y(f)| \right\} = C_s(\tau) + C_h(\tau) \tag{15}$$
实倒谱具备零相位与双边偶对称特性（$C_y(-\tau) = C_y(\tau)$）。在时域卷积转化为倒频域加法叠加的框架下，若源项与信道回声在倒频域占据不同支撑区间，则为两者的分离提供了理论可能。

#### 2. 实倒谱计算与峰值拾取流程
在关井后 $t \in [1.0, 50.0]\,\mathrm{s}$（$N = 49{,}000$ 点）截取信号并施加全 Hann 窗 $w_{\mathrm{Hann}}(t)$：
$$C(\tau) = \mathcal{F}^{-1} \left\{ \ln \left| \mathcal{F} \left\{ \left( \frac{\mathrm{d}H_{\mathrm{wh}}(t)}{\mathrm{d}t} - \overline{\frac{\mathrm{d}H_{\mathrm{wh}}}{\mathrm{d}t}} \right) w_{\mathrm{Hann}}(t) \right\} \right| \right\} \tag{16}$$
实倒谱计算完成后，在理论几何走时搜索窗口 $\mathcal{W}_{\mathrm{search}}$ 内按协议拾取峰位。全文区分两种倒谱估计器，不再统称“实倒谱主峰”：

- **模倒谱峰**（magnitude real cepstrum）：$\tau_{\mathrm{cep}}^{|C|}=\arg\max_{\tau\in\mathcal{W}_{\mathrm{search}}}|C(\tau)|$（式 17；表 4、图 4、有限关井及动态 $k(\mathrm{Re})$ 的 $|C|$ 行均用此协议）；
- **符号实倒谱峰**（signed real cepstrum）：$\tau_{\mathrm{cep}}^{C}=\arg\max_{\tau\in\mathcal{W}_{\mathrm{search}}}C(\tau)$（仅用于动态 $k(\mathrm{Re})$ 的对照行）。

$$\tau_{\mathrm{cep}}^{|C|} = \arg\max_{\tau \in \mathcal{W}_{\mathrm{search}}} |C(\tau)| \tag{17}$$
式中搜索窗口 $\mathcal{W}_{\mathrm{search}}$ 在基准阶跃关井下为几何走时邻域 $[\tau_{\mathrm{geom}} \pm 80\,\mathrm{ms}]$（表 4 代码实现为 $[5.575, 5.735]\,\mathrm{s}$）；在有限关井 $T_c$ 扫描中，为容纳展宽波包自适应设定为 $[\tau_{\mathrm{geom}} - 80\,\mathrm{ms}, \tau_{\mathrm{geom}} + 80\,\mathrm{ms} + \min(0.5, T_c)]$。

#### 3. 四种特征时钟定义（与提取代码一致）
首波窗口在表 4 / 图 3 中取 $t\in[6.5, 7.5]\,\mathrm{s}$；图 4 的 Darcy/IAB/WFB 对照取 $t\in[t_{\mathrm{geom}}\pm 0.5]\,\mathrm{s}$（$t_{\mathrm{geom}}=t_s+2X_1/a$）。窗口内令 $(\mathrm{d}H/\mathrm{d}t)_{+}=\max(0,\mathrm{d}H/\mathrm{d}t)$，阈值参照**该窗口内 $\mathrm{d}H/\mathrm{d}t$ 的正部最大值**，**不取绝对值**。

1. **波前起跳时钟 ($t_{\mathrm{onset}}$)：** 窗口内 $\mathrm{d}H/\mathrm{d}t$ 首次达到窗口正部峰值 $1\%$ 的时刻：
   $$t_{\mathrm{onset}} = \min \left\{ t \;\middle|\; \frac{\mathrm{d}H}{\mathrm{d}t}(t) \ge 0.01 \max_{t\in\mathcal{W}_{\mathrm{pkt}}} \frac{\mathrm{d}H}{\mathrm{d}t} \right\} \tag{18}$$
2. **波形峰值时钟 ($t_{\mathrm{peak}}$)：** 窗口内满足高度门限 $0.10\max(\mathrm{d}H/\mathrm{d}t)$、最小间距 $2\,\mathrm{ms}$ 的**第一个**局部峰；若无峰则退化为窗口 $\arg\max(\mathrm{d}H/\mathrm{d}t)$：
   $$t_{\mathrm{peak}} = t_{\mathrm{first\ peak}}\left(\frac{\mathrm{d}H}{\mathrm{d}t}\right) \tag{19}$$
3. **能量中位时刻时钟 ($t_{E50}$)：** 仅累积 $\mathrm{d}H/\mathrm{d}t$ 的**正部**平方，达到窗口总正部能量 $50\%$ 的时刻：
   $$t_{E50} \implies \int_{t_{\mathrm{win,0}}}^{t_{E50}} \left(\frac{\mathrm{d}H}{\mathrm{d}t}\right)_{+}^{2} \,\mathrm{d}t = \frac{1}{2} \int_{\mathcal{W}_{\mathrm{pkt}}} \left(\frac{\mathrm{d}H}{\mathrm{d}t}\right)_{+}^{2} \,\mathrm{d}t \tag{20}$$
4. **倒谱时钟 ($\tau_{\mathrm{cep}}$)：** 若无另行声明，均指模倒谱峰 $\tau_{\mathrm{cep}}^{|C|}$（式 17）。示意见图 1，定量对比见 §3.3。

**表 1: 四时钟估计器协议（与提取代码一致）**

| 指标 | 输入 | 窗口 | 预处理 | 峰值对象 | 阈值 / 搜索规则 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $t_{\mathrm{onset}}$ | $\mathrm{d}H/\mathrm{d}t$ | 表 4：$[6.5,7.5]\,\mathrm{s}$；图 4：$[t_{\mathrm{geom}}\pm0.5]\,\mathrm{s}$ | **不取绝对值** | 首次超过阈值 | $1\%$ of $\max(\mathrm{d}H/\mathrm{d}t)$ in window |
| $t_{\mathrm{peak}}$ | $\mathrm{d}H/\mathrm{d}t$ | 同上 | 原信号（正峰） | 窗口内**首个**局部峰 | height $=10\%$ of $\max(\mathrm{d}H/\mathrm{d}t)$，distance $=2\,\mathrm{ms}$ |
| $t_{E50}$ | $\mathrm{d}H/\mathrm{d}t$ | 同上 | 正部 $(\mathrm{d}H/\mathrm{d}t)_{+}$ | 累积能量 $50\%$ | $p(t)=(\mathrm{d}H/\mathrm{d}t)_{+}^{2}$ |
| 模倒谱 $\tau_{\mathrm{cep}}^{|C|}$ | $\mathrm{d}H/\mathrm{d}t$ | $t\in[1,50]\,\mathrm{s}$ | 去均值、Hann | $\arg\max |C(\tau)|$ | $\tau_{\mathrm{geom}}\pm 80\,\mathrm{ms}$（有限关井见正文） |
| 符号实倒谱 $\tau_{\mathrm{cep}}^{C}$ | 同上 | 同上 | 同上 | $\arg\max C(\tau)$ | **仅**动态 $k(\mathrm{Re})$ 对照行 |

#### 4. 测深与误差公式
- 绝对测深：$x_{\mathrm{est}} = a(t_{\mathrm{clk}} - t_s)/2$ 或 $x_{\mathrm{est}} = a\tau_{\mathrm{cep}}/2$（$t_s = 1.0\,\mathrm{s}$）；
- 绝对误差：$\Delta x = x_{\mathrm{est}} - X_1$；
- 相对稳态漂移：$\delta x = \Delta x(k) - \Delta x(k=0) = a\Delta t_{\mathrm{clk}}/2$。

### 2.7 动态雷诺数依赖的 Brunone 摩阻闭合模型

除常数弱摩阻假定（$k=0.01$）外，经典水力瞬变理论中 Brunone 系数通常被表征为随局部瞬时流动雷诺数 $\mathrm{Re}(z, t) = |V(z, t)| D / \nu$ 动态演化的闭合函数（Vardy and Brown, 2003; Bergant et al., 2008）：
$$k(\mathrm{Re}) = \frac{\sqrt{C^*}}{2} \tag{21}$$
式中 $C^*$ 为 Vardy 剪切衰减系数，在 `wellbore_moc.py` 中分段闭合实现如下：
- 当 $\mathrm{Re} < 1.0$ 时，$k = 0$；
- 层流区（$1.0 \le \mathrm{Re} < 2000$）：$C^* = 4.76 \times 10^{-3}$（对应 $k \approx 0.0345$）；
- 紊流区（$\mathrm{Re} \ge 2000$）：
  $$C^* = \frac{7.41}{\mathrm{Re}^{\log_{10}(14.3 / \mathrm{Re}^{0.05})}} \tag{22}$$

在本文数值工况中，运动粘度取 $\nu=1.0\times 10^{-6}\,\mathrm{m^2/s}$（`WELL_CONFIG['fluid_viscosity']`，运动粘度而非动力粘度）。初始稳态雷诺数 $\mathrm{Re}_0 = V_0 D/\nu \approx 1.397 \times 10^5$，对应初始 $k_0 \approx 0.0384$。**每个内节点、每个时间步** $n\ge 2$ 用该节点特征脚点的瞬时 $|V|$ 更新 $k(\mathrm{Re})$，不是仅在初始状态冻结。$\mathrm{Re}<1$ 时硬切断 $k=0$，对 $\mathrm{Re}\to 0$ **无**额外平滑；层流/紊流在 $\mathrm{Re}=2000$ 处分段，存在约 $0.0345\to 0.0315$ 的间断。关井后流速振荡衰减，$k(\mathrm{Re})$ 随局部瞬时雷诺数变化。该动态闭合与固定常数 $k=0.01$ 不属于同一参数化。单缝（$n=1, X_1=4100\,\mathrm{m}$）对该波形分别采用符号实倒谱 $C(\tau)$ 正峰与模倒谱 $|C(\tau)|$ 极大值两种拾取协议，定量对比见表 5 与 §3.6；生效 $k$ 的快照统计见表 7。

为覆盖 RQ1 与 RQ2，表 2 汇总本文 7 组数值试验设计。全篇计算统一基于水基清水压裂液（$\rho = 1000\,\mathrm{kg/m^3}$，$\nu = 1.0\times 10^{-6}\,\mathrm{m^2/s}$）与 5.5 英寸套管几何。Set 1–4 针对单缝基准下的时钟分化、模型形式与源项/估计器协议；Set 5 考察密集射孔簇的波包干涉；Set 6 与 Set 7 分别计算摩阻–滤失相对灵敏度与时间离散敏感性。本文聚焦牛顿水基压裂液；非牛顿流变在非定常管流中的本构演化不在一维声波框架内讨论。图 2(b) 中 $k=0.1,0.2$ 虚线仅作趋势外推。

**表 2: 井筒水击与倒谱识别数值模拟工况设计表**

| 工况组别 | 研究目的 | 摩阻闭合 | 裂缝配置 | 关井协议 | 扫描/对比参数 | 对应图表 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Set 1** 单缝基准与时钟分叉 | 考察管壁非定常剪切对四特征时钟的差异化影响 | Brunone IAB 与 Darcy 稳态基准 | 单缝 $n=1$，$X_1=4100\,\mathrm{m}$，$C_H=10^{-5}\,\mathrm{m^2}$ | 阶跃关断 $T_c=1.0\,\mathrm{ms}$ | $k\in\{0,0.01,0.02,0.05\}$ | 图 2, 图 3；表 4 |
| **Set 2** 摩阻模型形式对比 | 比较不同离散闭合对倒谱时钟的影响 | Darcy / 本文 IAB 离散闭合 / 十项指数 WFB | 单缝 $n=1$，$X_1=4100\,\mathrm{m}$ | 阶跃关断 $T_c=1.0\,\mathrm{ms}$ | 对流加速度项 vs 径向扩散近似 | 图 4；表 5 |
| **Set 3** 有限关井历时 | 考察激振源脉冲频宽与井筒传播耗散的联合作用 | Brunone IAB（$k=0.01$） | 单缝 $n=1$，$X_1=4100\,\mathrm{m}$ | 线性斜坡 $T_c\in[1,1000]\,\mathrm{ms}$ | $T_c\in\{1,50,200,1000\}\,\mathrm{ms}$ | 图 5；表 5 |
| **Set 4** 动态雷诺数闭合 | 比较同一动态波形上两种倒谱拾取协议 | 动态 $k(\mathrm{Re})$（Vardy 分段） | 单缝 $n=1$，$X_1=4100\,\mathrm{m}$ | 阶跃关断 $T_c=1.0\,\mathrm{ms}$ | $\arg\max C$ vs $\arg\max|C|$ | 表 5, 表 7 |
| **Set 5** 多簇压裂段干涉 | 考察密集多裂缝回声在非定常耗散下的波包重叠 | Brunone IAB（$k=0.01$ 及 $k$ 扫描） | 4 簇，$n=4$，$X_1=4100\,\mathrm{m}$ | 阶跃关断 $T_c=1.0\,\mathrm{ms}$ | $D\in\{5,10,20,50,100\}\,\mathrm{m}$ | 图 6, 图 7, 图 8；表 8 |
| **Set 6** 摩阻与滤失灵敏度 | 比较管壁耗散与缝口滤失对系统总衰减的相对贡献（简化初场） | Brunone IAB，$k\in[0,0.05]$ | 单缝 $n=1$；$k_{\mathrm{leak}}\in[10^{-5},5\times 10^{-4}]\,\mathrm{m^{5/2}/s}$ | 阶跃关断 $T_c=1.0\,\mathrm{ms}$ | $k\times k_{\mathrm{leak}}$ 全因子网格，$H_{\mathrm{ext}}=100\,\mathrm{m}$ | 图 9 |
| **Set 7** 时间步长敏感性 | 检验 MOC 时间步长与就近节点映射对四时钟提取的影响（$T_c=\Delta t$，不作收敛阶声明） | Darcy / IAB（$k=0.01$） | 单缝 $n=1$，$X_1=4100\,\mathrm{m}$ | 单步关断 $T_c=\Delta t$ | $\Delta t\in\{0.5,1.0,2.0\}\,\mathrm{ms}$（$\mathrm{Cr}=1$） | 表 3 |

---

## 3. 数值仿真结果与机理分析

### 3.1 数值验证与离散敏感性

为检验时间离散对四时钟提取的影响，在单缝（$n=1, X_1=4100\,\mathrm{m}$）下对比 $\Delta t\in\{0.5,1.0,2.0\}\,\mathrm{ms}$（$\mathrm{Cr}=1$，单步关断 $T_c=\Delta t$）。表 3 给出 Darcy（$k=0$）与本文 IAB 离散闭合（$k=0.01$）的相对漂移。

**表 3: 三档时间步长下四特征时钟响应与相对漂移敏感性（$n=1, X_1 = 4100\,\mathrm{m}$）**

| 时间步长 $\Delta t$ | 摩阻状态 | $t_{\mathrm{onset}}$ (s) | $t_{\mathrm{peak}}$ (s) | $t_{E50}$ (s) | $\tau_{\mathrm{cep}}^{|C|}$ (s) | 相对 $\delta x_{\mathrm{onset}}$ | 相对 $\delta x_{\mathrm{peak}}$ | 相对 $\delta x_{E50}$ | 相对 $\delta x_{\mathrm{cep}}$ | 相对 $\Delta\tau_{\mathrm{cep}}$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.5\,\mathrm{ms}$** | 稳态 Darcy ($k=0$) | 6.6555 | 6.6555 | 6.6555 | 5.6560 | $0.00\,\mathrm{m}$ | $0.00\,\mathrm{m}$ | $0.00\,\mathrm{m}$ | **$0.00\,\mathrm{m}$** | $0.0\,\mathrm{ms}$ |
| | IAB 离散闭合 ($k=0.01$) | 6.6565 | 6.6695 | 6.6688 | 5.6695 | $+0.73\,\mathrm{m}$ | $+10.15\,\mathrm{m}$ | $+9.65\,\mathrm{m}$ | **$+9.79\,\mathrm{m}$** | $+13.5\,\mathrm{ms}$ |
| **$1.0\,\mathrm{ms}$** | 稳态 Darcy ($k=0$) | 6.6530 | 6.6530 | 6.6530 | 5.6540 | $0.00\,\mathrm{m}$ | $0.00\,\mathrm{m}$ | $0.00\,\mathrm{m}$ | **$0.00\,\mathrm{m}$** | $0.0\,\mathrm{ms}$ |
| | IAB 离散闭合 ($k=0.01$) | 6.6530 | 6.6670 | 6.6659 | 5.6670 | $0.00\,\mathrm{m}$ | $+10.15\,\mathrm{m}$ | $+9.38\,\mathrm{m}$ | **$+9.43\,\mathrm{m}$** | $+13.0\,\mathrm{ms}$ |
| **$2.0\,\mathrm{ms}$** | 稳态 Darcy ($k=0$) | 6.6540 | 6.6540 | 6.6539 | 5.6560 | $0.00\,\mathrm{m}$ | $0.00\,\mathrm{m}$ | $0.00\,\mathrm{m}$ | **$0.00\,\mathrm{m}$** | $0.0\,\mathrm{ms}$ |
| | IAB 离散闭合 ($k=0.01$) | 6.6540 | 6.6680 | 6.6663 | 5.6680 | $0.00\,\mathrm{m}$ | $+10.15\,\mathrm{m}$ | $+8.94\,\mathrm{m}$ | **$+8.70\,\mathrm{m}$** | $+12.0\,\mathrm{ms}$ |

在三档步长下，IAB 相对 Darcy 的 $\delta x_{\mathrm{cep}}$ 分别为 $+9.79$、$+9.43$ 与 $+8.70\,\mathrm{m}$，同号且同量级；$\delta x_{\mathrm{peak}}$ 均为 $+10.15\,\mathrm{m}$。各档 $T_c=\Delta t$，网格变化同时耦合源脉冲上升沿。档间差异与测深量子（$a\Delta t/2$）同量级。三档对比不作极限收敛阶声明；正文基准 $+9.43\,\mathrm{m}$ 绑定于 $\Delta t=1.0\,\mathrm{ms}$ 与表 1 的模倒谱协议。

### 3.2 时域波形耗散衰减与脉冲弥散演化

图 2 给出了阶跃关井下井口水头梯度 $\mathrm{d}H_{\mathrm{wh}}/\mathrm{d}t$ 的时域演化：面板 (a)、(c) 为单缝基准（$n=1, X_1=4100\,\mathrm{m}$）；面板 (b) 的能量弥散时间取自 Matrix A（$n=4, D=20\,\mathrm{m}$）。

**图 2** 井筒非定常摩阻下时域水击压力波形的耗散演化。
(a) 单缝（$n=1, X_1=4100\,\mathrm{m}$）首个回声波包（相对几何到时）；稳态 Darcy（$k=0$）为陡峭尖峰，随常数 $k$ 增大峰幅下降、峰位后移；
(b) 能量弥散时间 $\mathrm{EST}$（正部能量累积从 $10\%$ 到 $90\%$ 的时间跨度，列 `est_s`，**不是** FWHM）随 $k$ 展宽（Matrix A，$n=4, D=20\,\mathrm{m}$；$k=0.1,0.2$ 虚线为趋势外推，不纳入定量讨论）；
(c) 单缝井口长周期衰减（$t \in [0, 35]\,\mathrm{s}$）。机制波形 (a)(c) 为 $n=1$；(b) 为四缝操作性指标，不与 (a)(c) 当作同一工况。

![图 2](figures/Fig2_waveform_evolution.png)

对比分析表明：
- 稳态 Darcy 模型（$k=0$）下，反射波包保持严格对称且无时移；
- 引入 Brunone 非定常项后（$k=0.01$），边界层剪切弛豫使波形发生显著不对称畸变：波形上升沿变缓，下降沿拖尾拉长；
- 边界层剪切弛豫削平了高频陡峭梯度，使波峰 $t_{\mathrm{peak}}$ 滞后 **$\Delta t_{\mathrm{peak}} = +13.0\,\mathrm{ms}$**；
- 反射波包显著展宽：能量弥散时间 $\mathrm{EST}=t(E{=}0.90)-t(E{=}0.10)$（正部 $(\mathrm{d}H/\mathrm{d}t)_{+}^{2}$ 累积，`metrics_v2.csv` 列 `est_s`，Matrix A，$n=4$，$D=20\,\mathrm{m}$）从 $k=0$ 时的 $1.0\,\mathrm{ms}$ 展宽至 $k=0.01$ 的 $8.0\,\mathrm{ms}$、$k=0.02$ 的 $12.0\,\mathrm{ms}$、$k=0.05$ 的 $27.0\,\mathrm{ms}$（图 2(b)）。同表 `fwhm_s` 分别为 $3.0$、$10.0$、$14.0$、$25.0\,\mathrm{ms}$，与 EST 不是同一指标。
- 在 $35\,\mathrm{s}$ 井口记录中（图 2(c)），高频分量迅速衰减。**理论**四分之一波长基频 $f_{\mathrm{th}}=a/(4L)=0.0725\,\mathrm{Hz}$，由构造对应 $4Lf_{\mathrm{th}}=1450\,\mathrm{m/s}$。对关井后约 $49\,\mathrm{s}$ 记录，$\Delta f=1/49\approx 0.0204\,\mathrm{Hz}$，栅格谱线 $3/49\approx 0.0612\,\mathrm{Hz}$ 只是离散网格，**不是**计算 $a_{f0}$ 所用频率（见 §3.3）。

---

### 3.3 特征时钟分叉与表观波速分化特征

为定量考察非定常摩阻对不同到达时间估计器的影响，按表 1 协议提取 $t_{\mathrm{onset}}$、$t_{\mathrm{peak}}$、$t_{E50}$ 与模倒谱峰 $\tau_{\mathrm{cep}}^{|C|}$，并对比其相对稳态基准（$k=0$）的漂移。

表 4 与图 3 给出单缝工况下的四时钟响应与测深误差矩阵（倒谱列为 $\arg\max|C|$）。

**表 4: 单缝工况下四特征时钟响应与测深误差矩阵 ($n=1, X_1 = 4100\,\mathrm{m}$；倒谱为模倒谱峰)**

| 摩阻状态 $k$ | 起跳时钟 $t_{\mathrm{onset}}$<br>到时 [时延 $\Delta t$] | 波峰时钟 $t_{\mathrm{peak}}$<br>到时 [时延 $\Delta t$] | 能量中位 $t_{E50}$<br>到时 [时延 $\Delta t$] | 倒谱时钟 $\tau_{\mathrm{cep}}$<br>到时 [时延 $\Delta\tau$] | 测深绝对误差 $\Delta x$ (m)<br>(起跳 / 波峰 / 倒谱) | 相对稳态漂移<br>$\delta x_{\mathrm{cep}}$ (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **稳态 Darcy ($k=0$)** | $6.653\,\mathrm{s}$ ($0.0\,\mathrm{ms}$) | $6.654\,\mathrm{s}$ ($0.0\,\mathrm{ms}$) | $6.653\,\mathrm{s}$ ($0.0\,\mathrm{ms}$) | $5.654\,\mathrm{s}$ ($0.0\,\mathrm{ms}$) | $-1.58$ / $-0.85$ / $-0.85$ | $0.00$ |
| **弱摩阻 ($k=0.01$)** | $6.653\,\mathrm{s}$ ($0.0\,\mathrm{ms}$) | $6.667\,\mathrm{s}$ (**$+13.0\,\mathrm{ms}$**) | $6.666\,\mathrm{s}$ (**$+13.0\,\mathrm{ms}$**) | $5.667\,\mathrm{s}$ (**$+13.0\,\mathrm{ms}$**) | $-1.58$ / $+8.58$ / $+8.58$ | **$+9.43$** |
| **中等摩阻 ($k=0.02$)** | $6.654\,\mathrm{s}$ ($+1.0\,\mathrm{ms}$) | $6.680\,\mathrm{s}$ (**$+26.0\,\mathrm{ms}$**) | $6.679\,\mathrm{s}$ (**$+25.7\,\mathrm{ms}$**) | $5.677\,\mathrm{s}$ (**$+23.0\,\mathrm{ms}$**) | $-0.85$ / $+18.00$ / $+15.83$ | **$+16.68$** |
| **强摩阻 ($k=0.05$)** | $6.659\,\mathrm{s}$ ($+6.0\,\mathrm{ms}$) | $6.718\,\mathrm{s}$ (**$+64.0\,\mathrm{ms}$**) | $6.714\,\mathrm{s}$ (**$+60.6\,\mathrm{ms}$**) | $5.704\,\mathrm{s}$ (**$+50.0\,\mathrm{ms}$**) | $+2.78$ / $+45.55$ / $+35.40$ | **$+36.25$** |

*(注：括号内为相对稳态基准的时延增量 $\Delta t$；$\Delta x = a(t-t_s)/2 - X_1$；$\delta x_{\mathrm{cep}} = a\Delta\tau_{\mathrm{cep}}/2$)*

**图 3** 特征时钟分叉与表观波速分化特征。
(a) 单缝（$n=1$）四时钟时延 $\Delta t$ 随 $k$ 演化：$k \le 0.02$ 时 $t_{\mathrm{onset}}$ 保持在 $0\sim 1\,\mathrm{ms}$（$k=0.05$ 时 $+6.0\,\mathrm{ms}$），$t_{\mathrm{peak}}, t_{E50}, \tau_{\mathrm{cep}}^{|C|}$ 不同程度右移；
(b) 绝对测深误差 $\Delta x = x_{\mathrm{est}} - X_1$，起跳时钟在 $[-1.58, +2.78]\,\mathrm{m}$；
(c) 表观波速（Matrix A，$n=4, D=20\,\mathrm{m}$，`metrics_v2.csv`）：$k=0$ 时 $a_{\mathrm{onset}}=1450.6\,\mathrm{m/s}$、$a_{f0}=1409.4\,\mathrm{m/s}$；$k=0.05$ 时 $a_{\mathrm{peak}}=1433.3\,\mathrm{m/s}$、插值模态 $a_{f0}=1397.0\,\mathrm{m/s}$。$a_{f0}=4L f_{\mathrm{peak}}$，其中 $f_{\mathrm{peak}}$ 为理论基频邻域 $[0.5,1.5]f_{\mathrm{th}}$ 内抛物线插值谱峰，对应 $f_{\mathrm{peak}}=a_{f0}/(4L)=0.06985\,\mathrm{Hz}$（$k=0.05$），**不是** $f_{\mathrm{th}}=0.0725\,\mathrm{Hz}$，也**不是** FFT 栅格 $3/49\approx 0.0612\,\mathrm{Hz}$。面板 (c) 为四缝，不与 (a)(b) 单缝混读。

![图 3](figures/Fig3_clock_splitting.png)

分析表明：
1. **起跳时钟稳定性：** 在 $k \le 0.02$ 范围内，起跳时钟时延 $\Delta t_{\mathrm{onset}}$ 保持在 $0.0\sim 1.0\,\mathrm{ms}$，对应绝对测深误差为 $[-1.58, -0.85]\,\mathrm{m}$；在较大摩阻 $k=0.05$ 下，起跳时钟微移 $+6.0\,\mathrm{ms}$（对应误差 $+2.78\,\mathrm{m}$）；
2. **波峰与模倒谱时钟的响应特征：** 在常态弱摩阻 $k=0.01$ 下，波峰时钟与模倒谱峰时延一致，均为 $\Delta t_{\mathrm{peak}} = \Delta\tau_{\mathrm{cep}}^{|C|} = +13.0\,\mathrm{ms}$，折算相对稳态偏移为：
   $$\delta x_{\mathrm{cep}} = \frac{1450\,\mathrm{m/s} \times 0.0130\,\mathrm{s}}{2} \approx \mathbf{+9.43\,\mathrm{m}}$$
   而在中等至强摩阻下（$k=0.02, 0.05$），两者呈现显著但非严格同步的滞后（如 $k=0.05$ 时波峰滞后 $+64.0\,\mathrm{ms}$，而倒谱峰滞后 $+50.0\,\mathrm{ms}$）；
3. **能量中位时刻与波形展宽：** 能量中位时刻 $t_{E50}$ 随 $k$ 的增大单调后移（$k=0.01$ 时为 $+13.0\,\mathrm{ms}$；$k=0.05$ 时为 $+60.6\,\mathrm{ms}$），反映了波包能量向后拖尾的形态演化；
4. **表观波速分化假象：** 图 3(c) 表明，采用不同时钟反算的等效波速随 $k$ 分化。$a_{\mathrm{onset}}$ 在 $k=0$ 时为 $1450.6\,\mathrm{m/s}$，贴近正演给定的 $a=1450\,\mathrm{m/s}$；$k=0.05$ 时插值模态 $a_{f0}=4Lf_{\mathrm{peak}}=1397.0\,\mathrm{m/s}$。若把峰值或模态周期的表观滞后直接写成介质本构声速折减，将掩盖剪切弛豫引起的时钟分化。该 $a_{f0}$ **不是** $4L\times 0.0725\,\mathrm{Hz}$。

四特征时钟分叉是时域波形与倒频域峰位的直接数值观测。IAB 下 $+9.43\,\mathrm{m}$ 模倒谱右偏尚不能外推为全部非定常阻力闭合的普遍结论；下节对比不同离散闭合以澄清模型依赖性。

---

### 3.4 摩阻闭合的模型形式不确定性（Darcy / IAB / WFB）

为检验倒谱测深偏差是否随一维非定常摩阻闭合形式而改变，在相同空间网格（$\Delta z = 1.45\,\mathrm{m}, \Delta t = 1.0\,\mathrm{ms}$）、相同流体物性（$a = 1450\,\mathrm{m/s}, \nu = 1.0\times 10^{-6}\,\mathrm{m^2/s}$）与相同阶跃关井条件（$n=1, X_1 = 4100\,\mathrm{m}, T_c = 1\,\mathrm{ms}$）下，对比稳态 Darcy、本文 IAB 离散闭合（$k=0.01$）与十项指数 WFB 的四时钟响应（图 4）。三者均含裂缝邻域非定常项置零。

**图 4** 三种摩阻闭合下的倒谱相对漂移（单缝 $n=1$，阶跃关井 $T_c=1\,\mathrm{ms}$；IAB 取 $k=0.01$；模倒谱 $\arg\max|C|$）。在 $1\,\mathrm{ms}$ 离散下 WFB 相对 Darcy 未检出超过约一个时间步的偏移（$|\delta x_{\mathrm{cep}}|\lesssim 0.73\,\mathrm{m}$），IAB 为 $+9.43\,\mathrm{m}$。对比未按总衰减率重新匹配，不作为现场误差的上下确界。

![图 4](figures/Fig4_model_form_comparison.png)

1. **WFB 的时钟表现：** 十项指数 WFB 在阶跃关井下 $\tau_{\mathrm{cep}}^{|C|} = 5.6540\,\mathrm{s}$，与 Darcy 落在同一倒频谱线，相对漂移的记录值为 $0.00\,\mathrm{m}$（$\Delta\tau=0$）。在 $\Delta t=1\,\mathrm{ms}$ 下测深量子为 $a\Delta t/2=0.725\,\mathrm{m}$，故应读作 **$|\delta x_{\mathrm{cep}}|\lesssim 0.73\,\mathrm{m}$**，而不是物理真值恰好为零。该结果未复现 IAB 的 $+9.43\,\mathrm{m}$。起跳 $\delta x_{\mathrm{onset}} = 0.00\,\mathrm{m}$，波峰 $\delta x_{\mathrm{peak}} = +0.73\,\mathrm{m}$（$+1.0\,\mathrm{ms}$，1 个 $\Delta t$），能量中位时刻 $\delta x_{E50} = +0.01\,\mathrm{m}$。
2. **机制解释的边界：** IAB 离散闭合含对流加速度项（式 5–7）。高频波前通过时，大的 $|\partial V/\partial z|$ 可削平峰前沿并造成峰位推迟。WFB 以径向扩散权函数的指数和近似描述壁面剪切。**在本文参数、十项拟合、关井边界与裂缝屏蔽协议下**，WFB 的倒谱峰移明显小于 IAB。这一结果与“该时间尺度内 WFB 主要表现为幅值衰减”的解释相一致，但**不能**单凭一次对照证明 WFB 在所有频率都没有相位效应，或径向扩散必然不造成倒谱峰移。进一步验证需要匹配总衰减率、改变 $\nu$、$T_c$ 与权函数截断。两类实现**未**对全局总衰减率作人为对齐。
3. **模型形式而非现场区间：** Darcy、WFB 与 IAB 是不同离散闭合在同态倒谱上的响应，不能当作现场测深误差的确定性上下确界，也不能外推为“凡非定常摩阻必导致倒谱偏深”。

---

### 3.5 有限关井时间对源谱和倒谱漂移的影响

泵阀关断具有有限动作历时。在单缝（$n=1, X_1 = 4100\,\mathrm{m}, k=0.01$）下扫描 $T_c \in \{1, 50, 200, 1000\}\,\mathrm{ms}$（图 5 与表 5）：

**图 5** 有限关井历时对相对漂移的影响（单缝 $n=1$，本文 IAB 离散闭合 $k=0.01$，模倒谱）。在所考察的四个 $T_c$ 上，$\delta x_{\mathrm{cep}}$ 从 $+9.43\,\mathrm{m}$ 单调下降至 $+5.08\,\mathrm{m}$；波峰时钟同期变化。不写成连续参数意义上的收敛规律。

![图 5](figures/Fig5_tc_sweep.png)

1. **四档响应：** $T_c=1\,\mathrm{ms}$ 时 $\delta x_{\mathrm{cep}}=\mathbf{+9.43\,\mathrm{m}}$（$+13.0\,\mathrm{ms}$）；$50\,\mathrm{ms}$ 时 $+7.25\,\mathrm{m}$；$200\,\mathrm{ms}$ 时 $+5.80\,\mathrm{m}$；$1000\,\mathrm{ms}$ 时 $+5.08\,\mathrm{m}$。
2. **源谱与传播的耦合：** 随 $T_c$ 延长，源谱高频被截断，相对漂移在四个取样点上单调下降。倒谱时延是入射波包频谱与井筒耗散传播的耦合响应。
3. **外推范围：** $T_c\ge 200\,\mathrm{ms}$ 时相对偏深不再降至零。该结果受限于本文井筒几何与弱摩阻 $k=0.01$，不宜直接外推为通用的现场固定常数扣减。

---

### 3.6 动态 $k(\mathrm{Re})$ 与倒谱拾峰协议敏感性

在同一条动态 $k(\mathrm{Re})$ 单缝波形上，符号实倒谱峰 $\arg\max C$ 给出 $\tau_{\mathrm{cep}}^{C}=5.6715\,\mathrm{s}$，$\Delta x_{\mathrm{cep}}=\mathbf{+11.80\,\mathrm{m}}$；模倒谱峰 $\arg\max|C|$ 因捕获负副瓣给出 $\tau_{\mathrm{cep}}^{|C|}=5.6810\,\mathrm{s}$，$\Delta x_{\mathrm{cep}}=\mathbf{+18.73\,\mathrm{m}}$。二者是**同一波形、两种拾取协议**，不是跨工况的 $10\sim 20\,\mathrm{m}$ 统计区间。$k$ 在每个内节点、每个时间步 $n\ge 2$ 由瞬时 $\mathrm{Re}$ 更新。关井后 $0.5\,\mathrm{s}$ 间隔空间快照（表 7）给出：$\mathrm{Re}$ 中位数 $2.43\times 10^{4}$，$\mathrm{Re}$ 的 $P95=1.20\times 10^{5}$；$k$ 中位数 $0.012$，$k$ 的 $P95=k_{\max}=0.0345$（层流分支上界）；$\mathrm{Re}<2000$ 的时空占比约 $11.7\%$。表 5 绑定偏差；表 6 给出固定观测 $\tau$ 时改假设声速 $a$ 的测深方向。

**表 5: 关键倒谱测深偏差与工况估计器协议绑定表**

| 工况 / 阻力模型 | 关井协议 | 测深估计器 | 相对/绝对偏差 | 物理与数值含义 |
| :--- | :--- | :--- | :--- | :--- |
| **本文 IAB 离散闭合 ($k=0.01$)** | 阶跃关井 ($T_c=1\,\mathrm{ms}$) | 模倒谱 $\arg\max|C|$ | 相对稳态 $\delta x_{\mathrm{cep}} = \mathbf{+9.43\,\mathrm{m}}$ | 对流加速度型闭合下的基准走时偏移 ($\Delta\tau = +13.0\,\mathrm{ms}$) |
| **本文 IAB 离散闭合 ($k=0.01$)** | 有限关井 ($T_c=0.2\sim 1.0\,\mathrm{s}$ 斜坡) | 模倒谱 $\arg\max|C|$ | 相对稳态 $\delta x_{\mathrm{cep}} = \mathbf{+5.80} \sim \mathbf{+5.08\,\mathrm{m}}$ | 四个 $T_c$ 取样点上相对偏移收窄 ($\Delta\tau = +8.0 \sim +7.0\,\mathrm{ms}$) |
| **动态雷诺数 $k(\mathrm{Re})$** | 阶跃关井 ($T_c=1\,\mathrm{ms}$) | 符号实倒谱 $\arg\max C$ | 绝对误差 $\Delta x_{\mathrm{cep}} = \mathbf{+11.80\,\mathrm{m}}$ | 同一动态波形上的正峰协议 ($\tau_{\mathrm{cep}}^{C}=5.6715\,\mathrm{s}$) |
| **动态雷诺数 $k(\mathrm{Re})$** | 阶跃关井 ($T_c=1\,\mathrm{ms}$) | 模倒谱 $\arg\max|C|$ | 绝对误差 $\Delta x_{\mathrm{cep}} = \mathbf{+18.73\,\mathrm{m}}$ | 同一波形上负副瓣被模算子捕获 ($\tau_{\mathrm{cep}}^{|C|}=5.6810\,\mathrm{s}$) |
| **十项指数 WFB** | 阶跃关井 ($T_c=1\,\mathrm{ms}$) | 模倒谱 $\arg\max|C|$ | 相对 Darcy $|\delta x_{\mathrm{cep}}|\lesssim 0.73\,\mathrm{m}$ | 与 Darcy 落在同一 $1\,\mathrm{ms}$ 谱线，非物理零真值 |

*(注：分析通道为井口 $\mathrm{d}H/\mathrm{d}t$，Hann 窗 $t \in [1, 50]\,\mathrm{s}$；检索窗基准 $[\tau_{\mathrm{geom}} \pm 80\,\mathrm{ms}]$，有限关井下扩展为 $[\tau_{\mathrm{geom}} - 80\,\mathrm{ms}, \tau_{\mathrm{geom}} + 80\,\mathrm{ms} + \min(0.5, T_c)]$。各行分别对应各自正演工况与估计器定义。)*

**表 6: 计算声速假设与测深估值响应方向对照表**

| 参数状态 / 物理假定 | 测深公式 $x_{\mathrm{est}} = a\,\tau_{\mathrm{cep}}/2$ 响应 | 物理机理与工程评价 |
| :--- | :--- | :--- |
| **人为下调计算声速** (如 $1450 \to 1390\,\mathrm{m/s}$) | 在固定实测时延 $\tau_{\mathrm{cep}}$ 下，计算深度 $x_{\mathrm{est}}$ **单调减小** | 虽能强制将偏深的单点估值压缩以贴近已知射孔深度，但会扭曲簇间距与深部结构 |
| **正演固定 $a=1450\,\mathrm{m/s}$ 时 $a_{\mathrm{onset}} \approx 1450.6\,\mathrm{m/s}$** | 起跳时钟对 IAB 弱摩阻不敏感，时延增加主要由波包型时钟体现 | 这是固定声速正演中的估计器差异，**不是**现场固有声速不变的实验证明；不宜用反演侧声速折减代偿 |

*(注：若正演中流体真实物理声速 $a$ 发生降低，例如携砂或含气导致等效声速折减，则物理声波传播走时增大；这属于正演物理介质性质变化，与反演算法中人为下调假设 $a$ 的响应方向完全不同。)*

**表 7: 动态 $k(\mathrm{Re})$ 单缝算例的生效系数快照统计（关井后 $0.5\,\mathrm{s}$ 间隔空间场）**

| 工况 | $\nu$ ($\mathrm{m^2/s}$) | $\mathrm{Re}$ 中位数 | $\mathrm{Re}$ $P95$ | $k$ 中位数 | $k$ $P95$ | $k_{\max}$ | $\mathrm{Re}<2000$ 占比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $n=1$，$X_1=4100\,\mathrm{m}$，动态 $k(\mathrm{Re})$ | $1.0\times 10^{-6}$ | $2.43\times 10^{4}$ | $1.20\times 10^{5}$ | $0.0120$ | $0.0345$ | $0.0345$ | $11.7\%$ |

更新规则：每个内节点、每个 MOC 步 $n\ge 2$；表中分位数来自快照而非逐步全场归档。$\mathrm{Re}<1$ 时 $k=0$；$\mathrm{Re}=2000$ 处分段间断。

---

### 3.7 频域选择性耗散与倒谱响应

在经典倒谱理论中，信号被视作无耗散离散回声序列的对数谱线性叠加：
$$\ln |Y(f)| \approx \ln |S(f)| + \sum_{m=1}^M \alpha_m \cos(2\pi f \tau_m) \tag{23}$$
逆傅里叶变换将周期波纹 $\cos(2\pi f \tau_m)$ 变换为倒频域 Delta 尖峰。

**图 6** 时频选择性耗散与倒谱响应。
(a) 单缝（$n=1$）STFT（$t \in [1, 30]\,\mathrm{s}$）：Darcy（上）高频谐波贯穿全时长，$k=0.02$（下）在 $10\,\mathrm{s}$ 内 $50\,\mathrm{Hz}$ 以上被强烈吸收，共用 $[-60, +20]\,\mathrm{dB}$ 色标；
(b) 频段积分衰减（相对 Darcy，Matrix A，$n=4, D=20\,\mathrm{m}$），低频（$0\sim 20\,\mathrm{Hz}$）$-1.6\,\mathrm{dB}$，高频（$60\sim 150\,\mathrm{Hz}$）$-17.0\,\mathrm{dB}$；与 (a) 不是同一裂缝配置；
(c) 概念性的对数谱衰减示意，**不是**由 MOC 时程反演得到的传递函数，也不用于证明倒谱峰必然发生单向时移（schematic）。

![图 6](figures/Fig4_stft_homomorphic.png)

在物理非定常流中，该假设被破坏（图 6）：
- STFT（图 6(a)）显示 Darcy 流中高达 $150\,\mathrm{Hz}$ 的高频谐波持续存在，而在非定常摩阻下 $50\,\mathrm{Hz}$ 以上能量在前 $10\,\mathrm{s}$ 内被吸收；
- 频段积分（图 6(b)）：$k=0.01$ 时低频仅衰减 **$-1.6\,\mathrm{dB}$**，高频 **$-17.0\,\mathrm{dB}$**；$k=0.05$ 时高频达 **$-40.5\,\mathrm{dB}$**；
- 对数幅值谱呈现高频下倾（图 6(c)）。实倒谱 $C_y(\tau)$ 满足倒频域加法分离与双边偶对称；对数谱下倾反映频选幅值衰减，不构成单向因果时移卷积核的推导证明。倒谱峰走时偏移依赖于具体阻力模型及其时域波形。

---

### 3.8 二维滑窗倒谱脊线演化

§3.2 至 §3.7 给出了单裂缝回声在井筒非定常流中的时钟分化与时频衰减。实际水平井压裂段常含多条密集射孔簇（Matrix A，$n=4$）。此时单波包展宽会在时域造成相邻回声重叠，并在倒频域引起同态脊线粘连。本节考察密集缝网下的二维滑窗倒谱。

**图 7** 井筒非定常摩阻下二维滑窗倒谱（Cepstrogram）同态脊线演化。
(a) 单缝稳态流（$n=1, k=0$）：脊线细窄，锁定于 $x \approx 4099\,\mathrm{m}$；
(b) 单缝 $k=0.01$：主脊仍近几何深度，右侧拖尾增强、晚窗展宽；相对偏深 $+9.43\,\mathrm{m}$ 以全窗 1D 倒谱为准（图 3、图 5）；
(c) 4 簇宽间距（$n=4, D=20\,\mathrm{m}, k=0.01$）：多条脊线分立；
(d) 4 簇窄间距（$n=4, D=5\,\mathrm{m}, k=0.01$）：晚期窗口内相邻脊线粘连为单条宽带（滑动 Hann 窗 $T_{\mathrm{win}}=30\,\mathrm{s}$，步长 $\mathrm{hop}=0.25\,\mathrm{s}$，通道 $\mathrm{d}H/\mathrm{d}t$，四格共用色标；虚线为裂缝真值 $X_i$）。

![图 7](figures/Fig5A_cepstrogram_bridge.png)

**表 8: 二维倒谱现象与一维定量指标对照**

| 二维倒谱现象 | 定量指标 | 对应主图 |
| :--- | :--- | :--- |
| 主脊近几何深度、能量向右拖尾 | 相对稳态偏深 $\delta x_{\mathrm{cep}} = +9.43\,\mathrm{m}$ | 图 3、图 5 |
| 单缝脊线随分析窗向后变宽 | $\mathrm{EST}$ 从 $1.0\,\mathrm{ms} \to 8.0\,\mathrm{ms}$；$\mathrm{FWHM}$ 展宽 | 图 2(b)、图 8(b) |
| 晚期分析窗脊线对比度下降 | $\mathrm{STFT}$ 高频分量在前 $5\sim 10\,\mathrm{s}$ 内耗散 | 图 6(a) |
| $D=5\,\mathrm{m}$ 脊线粘连与 $D=20\,\mathrm{m}$ 分立 | 峰谷对比度 $C_v = 0.00$ / $C_v = 1.00$（$n=4, k=0.01$） | 图 8(a) |

滑窗倒谱展示同态脊线随分析窗推进的趋势；测深数值仍以全窗一维模倒谱 $|C(\tau)|$ 为准。窄间距下相邻脊线在晚期窗内粘连融合（图 7 与表 8）。

---

### 3.9 多簇时域峰谷对比度与波包展宽

为定量表征多簇波包形态退化，定义相邻反射波峰间的谷值对比度
$$C_v = \frac{A_{\mathrm{peak}} - A_{\mathrm{valley}}}{A_{\mathrm{peak}}} \tag{24}$$

**图 8** 多簇缝网时域峰谷对比度与波包演化（Matrix A，$n=4$）。
(a) $C_v$ 随簇间距 $D\in[5,100]\,\mathrm{m}$：$k=0.01$ 时 $D=5\,\mathrm{m}$ 重叠（$C_v=0.00$），$D=20\,\mathrm{m}$ 时 $C_v=1.00$，作为波包弥散度量，不设物理硬阈值；
(b) $D=20\,\mathrm{m}$ 观测点，横轴为 $k\in[0,0.05]$：归一化峰值幅值下降，归一化 $\mathrm{FWHM}$ 展宽超过 8 倍。

![图 8](figures/Fig8_multicluster_cv.png)

1. **稳态基准（$k=0$）：** 即便 $D=5\,\mathrm{m}$，回声波峰仍陡峭分立，$C_v=1.00$。
2. **非定常耗散（$k=0.01$）：** 归一化半高宽展宽超过 8 倍（图 8(b)）。$D=5\,\mathrm{m}$ 时相邻波峰融合成单峰，$C_v=0.00$；$D=20\,\mathrm{m}$ 时 $C_v=1.00$（Matrix A 离散采样点）。
3. **度量含义：** $C_v$ 刻画多脉冲重叠程度，不构成井下多缝识别的物理硬阈值。

---

### 3.10 井筒摩阻与地层滤失的相对阻尼灵敏度

为比较井筒非定常摩阻与裂缝滤失对能量衰减的相对贡献，由井口波形连续小波变换（CWT）脊线提取一阶阻尼比 $\zeta$（图 9）。

**图 9** 阻尼比 $\zeta(k, k_{\mathrm{leak}})$ 响应面（单缝 $n=1$，$H_{\mathrm{ext}}=100\,\mathrm{m}$；平方根压差型准稳态滤失，非 Carter）。弱摩阻下井筒阻尼增量约 $+7.0\%$；与大滤失增量相当仅出现在参数网格上界。含滤失初场为均匀 $V\approx V_0$ 的简化处理，本图为相对敏感性测试。

![图 9](figures/Fig9_damping_surface.png)

在常态弱非定常摩阻（$k=0.01$）下，井筒摩阻引起的阻尼比增量约为 $+7.0\%$，低于缝口中高滤失引起的阻尼增长；两者增量相当仅在参数上界（$k=0.05$，$k_{\mathrm{leak}}=5\times 10^{-4}\,\mathrm{m^{5/2}/s}$，分别约 $+55\%$ 与 $+49\%$）成立。弱摩阻下的较小占比并不构成两者的独立解耦证明。因含滤失初场未迭代至严格稳态，上述百分比只用于相对排序。

---

## 4. 讨论与工程建议

### 4.1 特征时钟分叉对等效波速假定的澄清
长期以来，压裂诊断工程界将倒谱反演得到的偏深估值解释为“携砂含气或多孔介质引起的流体等效声速折减”（通常下调 $3\%\sim 8\%$）。

本文数值结果支持以下解释：
1. **onset 对 IAB 弱摩阻不敏感：** 在本文固定 $a=1450\,\mathrm{m/s}$ 的正演中，波前起跳时钟 $t_{\mathrm{onset}}$ 在常态弱摩阻下（$k \le 0.02$）保持稳定（时延 $\le 1.0\,\mathrm{ms}$，对应 $a_{\mathrm{onset}} \approx 1450.6\,\mathrm{m/s}$）。onset 估计器对 IAB 阻尼变化的敏感性显著低于 peak、$E_{50}$ 和模倒谱时钟；这**不等同于**对现场流体—管柱系统固有声速不变的实验性证明。
2. **偏深与耗散的模型依赖关系：** 在本文 IAB 离散闭合下，模倒谱峰右移与频选耗散并存；在十项指数 WFB 下，尽管同样存在壁面剪切引起的幅值耗散，相对 Darcy 的倒谱偏移仍低于约一个时间步（$|\delta x_{\mathrm{cep}}|\lesssim 0.73\,\mathrm{m}$）。这表明频选耗散本身并不构成倒谱系统性偏深的充分条件，偏深幅度取决于具体阻力闭合及其离散实现。
3. **经验声速折减的潜在危害：** 人为下调计算声速虽可在单点数值上将首条裂缝拉回射孔位置，但该修正属于全局线性缩放，无法校正非定常耗散带来的波包展宽与时域重叠演化，在多簇压裂段将导致簇间距与深部裂缝几何反演的严重畸变。

### 4.2 倒谱测深在压裂诊断中的修正准则与建议
针对非定常摩阻引发的测深偏差，提出以下数值与解释建议：
1. **关注起跳时钟与波峰时钟的分化：** 在快关井且信噪比良好（中高 SNR）的数值工况下，波前起跳时钟（$t_{\mathrm{onset}}$）受非定常耗散扰动显著小于波峰与倒谱时钟（$k \le 0.02$ 下时延 $\le 1.0\,\mathrm{ms}$），但起跳检测对噪声与微小阈值敏感，在现场应用中宜作为走时参考而非唯一绝对校准基准；
2. **避免盲目套用固定距离修正：** 本文揭示的 $+9.43\,\mathrm{m}$（阶跃关井）与 $T_c=0.2\,\mathrm{s}$ 的 $+5.80\,\mathrm{m}$ 与 $T_c=1.0\,\mathrm{s}$ 的 $+5.08\,\mathrm{m}$（斜坡关井）相对漂移，以及 $+11.80\,\mathrm{m}$ 与 $+18.73\,\mathrm{m}$ 绝对估计误差，分别严格对应各自前向仿真与倒谱算法实现协议，不能作为普适现场常数直接从测深结果中简单扣减；
3. **建立全波包非稳态正演匹配框架：** 建议从单一峰位时延反演转向包含非定常摩阻的 MOC 全波形匹配或频域联合反演，消除波形弥散对几何位置诊断的干扰。

### 4.3 压裂多簇缝网的时域峰谷对比度演化
在多簇密集压裂段（Matrix A，$n=4$），非定常摩阻引起的波包半高宽展宽（超过 8 倍）导致反射波包在时域发生重叠与形态退化：
1. **窄间距下的波包重叠：** 在常态弱摩阻下（$k=0.01$），当簇间距 $D = 5\,\mathrm{m}$ 时，相邻反射波峰在时域重叠融合成单峰（$C_v = 0.00$），二维倒谱脊线融合成单条宽带；
2. **波包弥散度量特征：** 当簇间距为 $D = 20\,\mathrm{m}$ 时，峰谷对比度为 $C_v = 1.00$；$C_v$ 随簇间距退化反映的是波包弥散程度，仅作为本算例矩阵离散采样点的特征数值，不应将其外推为普遍物理硬阈值。

---

## 5. 结论

本文针对水平井压裂水击瞬变压力倒谱诊断中的测深偏差，建立了一维 MOC 数值正演与多时钟后处理框架，取得如下主要结论：

1. **揭示了本文 IAB 离散闭合下的特征时钟分叉：** 在常数弱摩阻（$k \le 0.02$）下，波前起跳时钟保持稳定（时延 $\le 1.0\,\mathrm{ms}$），而波峰、能量中位时刻与模倒谱时钟显著右偏；$k=0.01$ 阶跃工况下模倒谱相对稳态偏深约 $+9.43\,\mathrm{m}$。在本文 IAB 算例中，所谓“等效声速降低”应解释为特征时钟分化与波包弥散，而不能写成全部非定常摩阻的普遍规律；
2. **明确了摩阻模型的形式不确定性：** 在本文 $1\,\mathrm{ms}$ 离散与指定 $|C|$ 拾峰协议下，十项指数 WFB 相对 Darcy 的倒谱偏移低于约一个时间步（$|\delta x_{\mathrm{cep}}|\lesssim 0.73\,\mathrm{m}$），未复现 IAB 同量级偏深，表明倒谱测深偏差具有显著的闭合形式依赖性；
3. **阐明了关井历时的耦合响应：** 在所考察的四个 $T_c$ 取值上，倒谱相对偏深从阶跃关井的 $+9.43\,\mathrm{m}$ 单调下降至 $+5.08\,\mathrm{m}$，表明实测时延漂移为激振源谱与井筒耗散传播的耦合响应，而不是连续参数意义上的收敛规律；
4. **厘清了摩阻与滤失的相对灵敏度差异：** 井筒壁面摩阻与地层滤失对所定义阻尼指标的相对贡献取决于参数区间；在常态弱摩阻下，井筒非定常项引起的阻尼增量（$+7.0\%$）小于有效滤失响应，仅在参数网格极端上界两者才具有相当量级（$+55\%$ vs $+49\%$）。该比较基于平方根压差型准稳态滤失与简化初场，弱摩阻下的较小占比并不构成独立解耦证明。

后续工作包括：含滤失严格稳态初场与质量守恒残差；无裂缝光滑管 Darcy 基准与单波衰减；十项权函数拟合误差；匹配总衰减后再对比 IAB 与 WFB；固定 $k$ 的 $n=2$ 双缝复核；以及实验室或现场压力记录约束。本文不展开全波形可辨识性或 Fisher 信息分析。

---

## 符号表 (Nomenclature)

- $a$ = 压力波在流体-井筒系统中的传播声速，$\mathrm{m/s}$
- $A$ = 井筒横截面积，$\mathrm{m^2}$
- $C(\tau)$ = 信号实倒谱序列，无量纲
- $C_f, C_H$ = 裂缝集总水力柔量（水头形式），$\mathrm{m^2}$
- $C_{\mathrm{frac}}$ = 裂缝压力形式水力柔量，$\mathrm{m^3/Pa}$
- $C_v$ = 多簇相邻反射波峰谷对比度，无量纲
- $D$ = 套管内径或多簇射孔缝间距，$\mathrm{m}$
- $f$ = 达西-韦斯巴赫稳态摩阻系数，无量纲
- $g$ = 重力加速度，取 $9.81\,\mathrm{m/s^2}$
- $H$ = 测压管水头，$\mathrm{m}$
- $H_{\mathrm{ext}}$ = 远场孔隙压力水头，$\mathrm{m}$
- $k$ = Brunone 无量纲非定常摩阻系数
- $k_{\mathrm{leak}}$ = 平方根压差型准稳态滤失系数，$\mathrm{m^{5/2}/s}$
- $L$ = 井筒总长度，$\mathrm{m}$
- $Q_f$ = 裂缝侧向注入瞬态流量，$\mathrm{m^3/s}$
- $\mathrm{Re}$ = 瞬时流动雷诺数，无量纲
- $t$ = 时间变量，$\mathrm{s}$
- $t_s$ = 关井起始时刻，$\mathrm{s}$
- $T_c$ = 阀门关断历时，$\mathrm{s}$
- $t_{\mathrm{onset}}$ = 波前起跳特征时钟时刻，$\mathrm{s}$
- $t_{\mathrm{peak}}$ = 反射波峰特征时钟时刻，$\mathrm{s}$
- $t_{E50}$ = 累计能量 $50\%$ 中位时刻时钟，$\mathrm{s}$
- $V$ = 井筒断面平均流速，$\mathrm{m/s}$
- $X_1$ = 首条水力裂缝物理深度，$\mathrm{m}$
- $x_{\mathrm{est}}$ = 水击波反演估计深度，$\mathrm{m}$
- $\Delta t$ = MOC 时间离散步长，$\mathrm{s}$
- $\Delta z$ = MOC 空间网格步长，$\mathrm{m}$
- $\Delta x$ = 测深估计值相对于物理真值的绝对误差，$\mathrm{m}$
- $\delta x_{\mathrm{cep}}$ = 相对稳态 Darcy 基准（$k=0$）的倒谱测深漂移量，$\mathrm{m}$
- $\tau$ = 倒频域 Quefrency 时延变量，$\mathrm{s}$
- $\tau_{\mathrm{cep}}^{|C|}$ = 模倒谱峰，$\arg\max |C(\tau)|$，$\mathrm{s}$
- $\tau_{\mathrm{cep}}^{C}$ = 符号实倒谱峰，$\arg\max C(\tau)$，$\mathrm{s}$
- $\tau_w$ = 管壁总剪切应力，$\mathrm{Pa}$
- $\rho$ = 流体密度，$\mathrm{kg/m^3}$
- $\nu$ = 流体运动粘度，$\mathrm{m^2/s}$
- $\zeta$ = 基于连续小波变换（CWT）提取的系统一阶衰减阻尼比，无量纲

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
